"""Tracked particles/grains on the existing MORQEX material-reference lattice.

Phase labels, particle identity and grain identity have distinct meanings.
Grain orientation does not supply missing elastic, diffusion or fatigue data.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from morqex.geometry import MicrostructureData


@dataclass(frozen=True)
class PolycrystalData:
    structure: MicrostructureData
    particle_ids: np.ndarray
    grain_ids: np.ndarray
    grain_rotations: np.ndarray
    grain_particles: np.ndarray
    metadata: dict

    def __post_init__(self):
        particles=np.array(self.particle_ids,copy=True)
        grains=np.array(self.grain_ids,copy=True)
        rotations=np.array(self.grain_rotations,dtype=float,copy=True)
        parents=np.array(self.grain_particles,copy=True)
        shape=self.structure.labels.shape
        active=np.isin(self.structure.labels,[p for p,k in self.structure.phase_kinds.items() if k=='active'])
        if particles.shape!=shape or grains.shape!=shape or particles.dtype.kind not in 'iu' or grains.dtype.kind not in 'iu':raise ValueError('Integer particle/grain fields on the full ZYX material lattice required')
        if not active.any() or np.any(particles<0) or np.any(grains<0):raise ValueError('Nonempty active material and nonnegative identity fields required')
        if not np.array_equal(particles>0,active) or not np.array_equal(grains>0,active):raise ValueError('Positive particle and grain IDs exactly on active material required')
        count=int(grains.max(initial=0))
        if not np.array_equal(np.unique(grains[active]),np.arange(1,count+1)):raise ValueError('Contiguous positive grain IDs required')
        if rotations.shape!=(count,3,3) or parents.shape!=(count,) or parents.dtype.kind not in 'iu' or not np.isfinite(rotations).all():raise ValueError('One rotation and parent particle per grain required')
        if not np.allclose(rotations@rotations.swapaxes(1,2),np.eye(3),rtol=0,atol=1e-12) or not np.allclose(np.linalg.det(rotations),1.,rtol=0,atol=1e-12):raise ValueError('Grain orientations must be proper rotations')
        for grain,parent in enumerate(parents,1):
            if not np.all(particles[grains==grain]==parent):raise ValueError('A grain cannot cross particle identity')
            if ndimage.label(grains==grain)[1]!=1:raise ValueError('Each grain must be face-connected')
        for name,array in [('particle_ids',particles),('grain_ids',grains),('grain_rotations',rotations),('grain_particles',parents)]:
            array.flags.writeable=False;object.__setattr__(self,name,array)
        object.__setattr__(self,'metadata',dict(self.metadata))

    @property
    def fingerprint(self):
        h=hashlib.sha256(self.structure.fingerprint.encode())
        for value in (self.particle_ids,self.grain_ids,self.grain_rotations,self.grain_particles):
            h.update(str(value.shape).encode());h.update(value.dtype.str.encode());h.update(value.tobytes())
        return h.hexdigest()

    def rotated_tensor_table(self, crystal_tensor):
        tensor=np.asarray(crystal_tensor,float)
        if tensor.shape!=(3,3) or not np.isfinite(tensor).all() or not np.allclose(tensor,tensor.T,rtol=0,atol=1e-14*max(np.max(abs(tensor)),np.finfo(float).tiny)):
            raise ValueError('Finite symmetric crystal-frame tensor required')
        return self.grain_rotations@tensor@self.grain_rotations.swapaxes(1,2)

    def grain_boundary_faces(self):
        """Full-lattice face IDs and true reference areas, no invented GB width."""
        ids=np.arange(self.grain_ids.size).reshape(self.grain_ids.shape)
        cells=[];axes=[];area=[]
        for axis in range(3):
            a=[slice(None)]*3;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None);a=tuple(a);b=tuple(b)
            valid=(self.grain_ids[a]>0)&(self.grain_ids[b]>0)&(self.grain_ids[a]!=self.grain_ids[b])&(self.particle_ids[a]==self.particle_ids[b])
            cells.extend(np.column_stack([ids[a][valid],ids[b][valid]]));axes.extend([2-axis]*int(valid.sum()))
            area.extend([np.prod(self.structure.spacing_zyx_m)/self.structure.spacing_zyx_m[axis]]*int(valid.sum()))
        return {'voxel_pairs':np.asarray(cells,dtype=np.int64).reshape(-1,2),
                'normal_axis_xyz':np.asarray(axes,dtype=np.int8),'reference_area_m2':np.asarray(area)}

    def save_hdf5(self,path):
        import h5py
        from pathlib import Path
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.pending')
        with h5py.File(temp,'w') as f:
            f.attrs.update(schema='morqex-polycrystal-1',fingerprint=self.fingerprint,
                phase_kinds=json.dumps(self.structure.phase_kinds),geometry_metadata=json.dumps(self.structure.provenance),
                microstructure_metadata=json.dumps(self.metadata),spacing_zyx_m=self.structure.spacing_zyx_m,origin_zyx_m=self.structure.origin_zyx_m)
            for name,a in [('phase_labels',self.structure.labels),('particle_ids',self.particle_ids),('grain_ids',self.grain_ids),
                           ('orientation_table',self.grain_rotations),('grain_particles',self.grain_particles)]:
                f.create_dataset(name,data=a,compression='gzip')
            for name,a in self.grain_boundary_faces().items():f.create_dataset('grain_boundaries/'+name,data=a,compression='gzip')
        temp.replace(path)
        return path

    @classmethod
    def load_hdf5(cls,path):
        import h5py
        with h5py.File(path,'r') as f:
            if f.attrs['schema']!='morqex-polycrystal-1':raise ValueError('Unknown polycrystal schema')
            structure=MicrostructureData(f['phase_labels'][:],tuple(f.attrs['spacing_zyx_m']),tuple(f.attrs['origin_zyx_m']),
                {int(k):v for k,v in json.loads(f.attrs['phase_kinds']).items()},json.loads(f.attrs['geometry_metadata']))
            result=cls(structure,f['particle_ids'][:],f['grain_ids'][:],f['orientation_table'][:],f['grain_particles'][:],json.loads(f.attrs['microstructure_metadata']))
            if result.fingerprint!=f.attrs['fingerprint']:raise ValueError('Polycrystal checksum mismatch')
            for name,expected in result.grain_boundary_faces().items():
                if not np.array_equal(f['grain_boundaries/'+name][:],expected):raise ValueError('Stored grain boundaries disagree with the material lattice')
        return result


def partition_grains(structure,particle_ids,*,grains_per_particle=8,seed=20261126,rotations=None):
    """Seeded Euclidean Voronoi clipped to each actual 3D particle.

    Disconnected clipped fragments become distinct grains with the same parent
    seed orientation; actual count and minimum size are reported. This is a
    numerical morphology, not a fitted sintering or texture model.
    """
    particles=np.asarray(particle_ids)
    if particles.shape!=structure.labels.shape or particles.dtype.kind not in 'iu':raise ValueError('Integer full-lattice particle IDs required')
    active=np.isin(structure.labels,[p for p,k in structure.phase_kinds.items() if k=='active'])
    if not active.any() or np.any(particles<0) or not np.array_equal(particles>0,active):raise ValueError('Positive particle IDs exactly on nonempty active material required')
    if type(grains_per_particle) is not int or grains_per_particle<1:raise ValueError('Positive integer grain count required')
    rng=np.random.default_rng(seed);labels=np.zeros(particles.shape,dtype=np.int32);table=[];parents=[];seeds=[]
    grain=0
    for particle in np.unique(particles[particles>0]):
        coordinates=np.argwhere(particles==particle)
        if len(coordinates)<grains_per_particle:raise ValueError('More grain seeds than material voxels')
        xyz=(coordinates+.5)*np.asarray(structure.spacing_zyx_m)
        selected=[int(rng.integers(len(xyz)))];dist=np.full(len(xyz),np.inf)
        for _ in range(1,grains_per_particle):
            dist=np.minimum(dist,np.sum((xyz-xyz[selected[-1]])**2,axis=1));selected.append(int(np.argmax(dist)))
        owner=cKDTree(xyz[selected]).query(xyz)[1]
        for local in range(grains_per_particle):
            mask=np.zeros(particles.shape,bool);mask[tuple(coordinates[owner==local].T)]=True
            components,count=ndimage.label(mask)
            rotation=Rotation.random(random_state=rng).as_matrix()
            for component in range(1,count+1):
                grain+=1;labels[components==component]=grain;table.append(rotation);parents.append(int(particle));seeds.append(coordinates[selected[local]].tolist())
    if rotations is not None:
        supplied=np.asarray(rotations,float)
        if supplied.shape!=(grain,3,3):raise ValueError('Explicit orientations must match the actual connected grain count')
        table=supplied
    sizes=np.bincount(labels.ravel())[1:]
    return PolycrystalData(structure,particles,labels,np.asarray(table),np.asarray(parents,dtype=np.int32),
        {'seed':seed,'requested_grains_per_particle':grains_per_particle,'actual_grains':grain,'grain_seed_zyx':seeds,
         'minimum_grain_voxels':int(sizes.min()),'grain_volumes_m3':(sizes*np.prod(structure.spacing_zyx_m)).tolist(),
         'orientation_distribution':'Haar-uniform SO(3)' if rotations is None else 'user-supplied validated SO(3)',
         'scope':'Initial numerical polycrystal; no fatigue evolution or material calibration implied'})
