"""Conforming labelled PLCs and the source-integrated TetGen quality mesher.

No faces are removed on failure. Geometry reduction is always explicit.
"""
from dataclasses import dataclass, field
import json, time
from pathlib import Path
import numpy as np
from scipy import ndimage
from morqex.microstructure.mesh.voxel import FEMesh


@dataclass
class SurfacePLC:
    points: np.ndarray                 # XYZ metres
    facets: object                     # triangles or planar polygons
    seeds: dict                        # original region ID -> (k,3) XYZ metres
    markers: np.ndarray | None = None
    holes: object = field(default_factory=list)
    region_volumes: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def viewer(self,**kwargs):
        from morqex.visualization.adaptive_mesh import AdaptiveMeshView
        return AdaptiveMeshView(self,**kwargs)

    def show(self,**kwargs):return self.viewer(**kwargs).display()

    def __post_init__(self):
        self.points=np.asarray(self.points,dtype=float)
        if self.points.ndim!=2 or self.points.shape[1]!=3 or len(self.points)<4 or not np.isfinite(self.points).all():
            raise ValueError('Finite (N,3) XYZ points required')
        raw=self.facets if isinstance(self.facets,np.ndarray) and self.facets.ndim==2 else list(self.facets)
        if not len(raw):raise ValueError('PLC facets are empty')
        invalid=(raw.dtype.kind not in 'iu' or raw.shape[1]<3 or raw.min()<0 or raw.max()>=len(self.points)) if isinstance(raw,np.ndarray) else any(
               np.asarray(f).dtype.kind not in 'iu' or np.asarray(f).size<3 or
               np.asarray(f).min()<0 or np.asarray(f).max()>=len(self.points) for f in raw)
        if invalid:
            raise ValueError('Valid integer facet indices required')
        self.facets=np.asarray(raw,dtype=np.int32) if isinstance(raw,np.ndarray) else [np.asarray(f,dtype=np.int32) for f in raw]
        if isinstance(self.facets,np.ndarray):
            if self.facets.shape[1]<3:raise ValueError('At least3 vertices per facet')
            for start in range(0,len(raw),50000):
                f=self.facets[start:start+50000]
                if np.any(np.diff(np.sort(f,axis=1),axis=1)==0):raise ValueError('Repeated facet vertex')
                p=self.points[f]-self.points[f[:,:1]];scale=np.linalg.norm(p,axis=2).max(axis=1)
                cross=np.cross(p[:,1:-1],p[:,2:]);norm=np.linalg.norm(cross,axis=2)
                normal=cross[np.arange(len(p)),norm.argmax(axis=1)]
                n=np.linalg.norm(normal,axis=1)
                if np.any(n<=scale**2*1e-12):raise ValueError('Degenerate PLC facet')
                if np.any(np.abs(np.einsum('nki,ni->nk',p,normal)).max(axis=1)>1e-9*scale*n):raise ValueError('Nonplanar PLC facet')
        else:
            for f in self.facets:
                if f.ndim!=1 or len(f)<3 or len(set(f))!=len(f):raise ValueError('Invalid PLC polygon indices')
                p=self.points[f]-self.points[f[0]];scale=np.max(np.linalg.norm(p,axis=1))
                cross=np.cross(p[1:-1],p[2:]);normal=cross[np.argmax(np.linalg.norm(cross,axis=1))]
                if not scale or np.linalg.norm(normal)<=scale**2*1e-12:raise ValueError('Degenerate PLC facet')
                if np.max(np.abs(p@normal))>1e-9*scale*np.linalg.norm(normal):raise ValueError('Nonplanar PLC facet')
        self.markers=np.ones(len(raw),np.int32) if self.markers is None else np.asarray(self.markers)
        if self.markers.shape!=(len(raw),) or self.markers.dtype.kind not in 'iu' or np.any(self.markers<=0):
            raise ValueError('One positive integer marker per facet required')
        self.seeds={int(k):np.asarray(v,float).reshape(-1,3) for k,v in self.seeds.items()}
        if not self.seeds or any(k<0 or not len(v) or not np.isfinite(v).all() for k,v in self.seeds.items()):
            raise ValueError('At least one finite XYZ region seed per included region required')
        self.holes=np.asarray(self.holes,float).reshape(-1,3)
        if not np.isfinite(self.holes).all():raise ValueError('Finite hole seeds required')


def _component_seeds(mask,spacing,origin,*,bounded=False):
    labels,count=ndimage.label(mask)
    if not count:return np.empty((0,3))
    distance=ndimage.distance_transform_edt(np.pad(mask,1),sampling=spacing)[1:-1,1:-1,1:-1]
    ids=np.arange(1,count+1)
    if bounded:
        outer=np.unique(np.concatenate([labels[0].ravel(),labels[-1].ravel(),labels[:,0].ravel(),
            labels[:,-1].ravel(),labels[:,:,0].ravel(),labels[:,:,-1].ravel()]))
        ids=ids[~np.isin(ids,outer)]
    if not len(ids):return np.empty((0,3))
    indices=np.asarray(ndimage.maximum_position(distance,labels,ids)).reshape(-1,3)
    return ((indices+.5)*spacing+origin)[:,::-1]


def voxel_plc(model,*,regions=None,maximum_facets=500000):
    """Exact, shared multiphase unit-quad interfaces; no independent phase smoothing.

    Preserves every selected voxel and its volume. Nonmanifold edge/point
    contacts can be rejected by TetGen and must be resolved by the caller.
    """
    a=model.regions
    chosen=tuple(map(int,np.unique(a) if regions is None else regions))
    if not chosen or len(set(chosen))!=len(chosen) or not set(chosen)<=set(map(int,np.unique(a))):
        raise ValueError('Select existing distinct region labels')
    if a.max()>np.iinfo(np.int64).max:raise ValueError('Region labels must fit int64')
    grid=np.array(a.shape)+1;quads=[];pairs=[];count=0
    offsets=[np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0]]),
             np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]]),
             np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]])]
    # Two label slabs per axis: a shared interface is emitted exactly once.
    for axis in range(3):
        pad=[(0,0)]*3;pad[axis]=(1,1);p=np.pad(a.astype(np.int64),pad,constant_values=-1)
        lo=[slice(None)]*3;hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        left,right=p[tuple(lo)],p[tuple(hi)]
        mask=(left!=right)&(np.isin(left,chosen)|np.isin(right,chosen))
        indices=np.argwhere(mask);count+=len(indices)
        if count>maximum_facets:raise ValueError(f'PLC needs more than {maximum_facets} facets; choose an explicit geometry/budget')
        corners=indices[:,None,:]+offsets[axis][None,:,:]
        face_ids=np.ravel_multi_index(corners.reshape(-1,3).T,tuple(grid)).reshape(-1,4)
        reverse=left[mask]>right[mask];face_ids[reverse]=face_ids[reverse][:,::-1]
        quads.append(face_ids)
        pairs.append(np.sort(np.column_stack((left[mask],right[mask])),axis=1))
    ids,inverse=np.unique(np.concatenate(quads),return_inverse=True)
    points=(np.column_stack(np.unravel_index(ids,tuple(grid)))*model.spacing+model.origin)[:,::-1]
    pair_values,markers=np.unique(np.concatenate(pairs),axis=0,return_inverse=True)
    spacing=np.asarray(model.spacing);origin=np.asarray(model.origin)
    seeds={k:_component_seeds(a==k,spacing,origin) for k in chosen}
    holes=_component_seeds(~np.isin(a,chosen),spacing,origin,bounded=True)
    volumes={k:float(np.count_nonzero(a==k)*np.prod(spacing)) for k in chosen}
    return SurfacePLC(points,inverse.reshape(-1,4),seeds,markers+1,holes,volumes,
        {'surface_method':'voxel_plc','region_names':model.region_names,
         'interface_pairs':{str(i+1):list(map(int,p)) for i,p in enumerate(pair_values)},
         'geometry':'exact voxel boundaries; -1 denotes exterior; ZYX arrays / XYZ coordinates',
         'oriented_material_pairs':True,'box_planes_preserved':set(chosen)==set(map(int,np.unique(a)))})


def marching_cubes_plc(model,*,region,step_size=1,maximum_facets=500000):
    """Single-region closed contour at voxel-centred level0.5; geometry is approximate.

    step_size>1 explicitly coarsens geometry, never the original volume data.
    Separate per-phase contours are not a conforming multiphase interface.
    """
    from skimage.measure import marching_cubes
    if region not in np.unique(model.regions):raise ValueError('Region not present')
    if type(step_size) is not int or step_size<1:raise ValueError('Positive integer step_size required')
    mask=model.regions==region
    # Pad by a multiple of step_size so the original domain's two ends are sampled.
    pad=step_size
    p,faces,_,_=marching_cubes(np.pad(mask,pad).astype(np.float32),.5,step_size=step_size,allow_degenerate=False)
    if len(faces)>maximum_facets:raise ValueError('Marching-cubes facet budget exceeded')
    points=((p-pad+.5)*model.spacing+model.origin)[:,::-1]
    edges=np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    if np.any(np.unique(edges,axis=0,return_counts=True)[1]!=2):raise ValueError('Contour is not a closed manifold')
    # Use the deepest point of each component. Coarsening can erase a component;
    # seeds and output-label checks then reject the resulting mesh.
    seeds={int(region):_component_seeds(mask,np.asarray(model.spacing),np.asarray(model.origin))}
    holes=_component_seeds(~mask,np.asarray(model.spacing),np.asarray(model.origin),bounded=True)
    return SurfacePLC(points,faces,seeds,holes=holes,metadata={
        'surface_method':'marching_cubes','step_size':step_size,'region_names':model.region_names,
        'interface_pairs':{'1':[-1,int(region)]},
        'voxel_volume_m3':float(mask.sum()*np.prod(model.spacing)),
        'geometry':'approximate level0.5 contour, not exact voxel boundaries'})


@dataclass(frozen=True)
class TetGenOptions:
    radius_edge_ratio: float = 2.0
    minimum_dihedral: float = 0.0
    maximum_volume: float | None = None  # SI m3
    region_maximum_volumes: dict = field(default_factory=dict)
    maximum_steiner: int = 300000
    maximum_cells: int = 2000000
    preserve_surface: bool = False
    consistency_check: bool = True
    require_quality: bool = False
    coplanarity_tolerance: float = 1e-12
    merge_coplanar: bool = True
    facet_separation_angle: float = 179.9
    isolated: bool = False
    native_timeout_seconds: float = 180.
    region_labelling: str = 'seeds'

    @classmethod
    def for_adaptive(cls,**overrides):
        """Merge only practically coplanar facets, retaining curved geometry."""
        return cls(**({'facet_separation_angle':179.99999,'isolated':True}|overrides))

    def __post_init__(self):
        if not np.isfinite(self.radius_edge_ratio) or self.radius_edge_ratio<1.:
            raise ValueError('radius_edge_ratio must be finite and >=1')
        if not np.isfinite(self.minimum_dihedral) or not 0<=self.minimum_dihedral<=20:
            raise ValueError('minimum_dihedral must be0..20 degrees')
        for v in [self.maximum_volume,*self.region_maximum_volumes.values()]:
            if v is not None and (not np.isfinite(v) or v<=0):raise ValueError('Positive finite maximum volume in m3 required')
        if type(self.maximum_steiner) is not int or self.maximum_steiner<0:raise ValueError('Nonnegative Steiner budget required')
        if type(self.maximum_cells) is not int or self.maximum_cells<1:raise ValueError('Positive cell budget required')
        if not np.isfinite(self.coplanarity_tolerance) or not 0<self.coplanarity_tolerance<=1e-6:raise ValueError('coplanarity_tolerance must be positive and<=1e-6')
        if not np.isfinite(self.facet_separation_angle) or not 0<self.facet_separation_angle<=180:raise ValueError('facet_separation_angle must be in(0,180] degrees')
        if not np.isfinite(self.native_timeout_seconds) or self.native_timeout_seconds<=0:raise ValueError('Positive finite native time budget required')
        if self.region_labelling not in ('seeds','interfaces'):raise ValueError('region_labelling must be seeds or interfaces')


@dataclass
class TetGenMesh(FEMesh):
    facets: object = None
    facet_markers: object = None
    report: dict = field(default_factory=dict)

    def viewer(self,**kwargs):
        from morqex.visualization.notebook import MeshView
        return MeshView(self,**kwargs)

    def show(self,**kwargs):return self.viewer(**kwargs).display()

    def adaptive_viewer(self,**kwargs):
        from morqex.visualization.adaptive_mesh import AdaptiveMeshView
        return AdaptiveMeshView(self,**kwargs)

    def save_hdf5(self,path):
        from .voxel import external_path
        import h5py
        path=external_path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(path,'x') as f:
            for name,value in [('points_m',self.points),('tetrahedra',self.cells),('phase',self.phases),
                               ('facets',self.facets),('facet_markers',self.facet_markers)]:
                f.create_dataset('mesh/'+name,data=value,compression='gzip',compression_opts=1,shuffle=True)
            f.attrs.update(complete=True,mesher=self.report.get('mesher','TetGen1.6.0'),report=json.dumps(self.report),
                node_count=len(self.points),tetrahedra=len(self.cells),selected_regions=np.unique(self.phases),
                coordinate_units='m',coordinate_order='XYZ')
        from .meshing import MeshArtifact
        from xml.sax.saxutils import escape
        xdmf=path.with_suffix('.xdmf');fn=escape(path.name)
        cell_number='UInt' if self.cells.dtype.kind=='u' else 'Int'
        phase_number='UInt' if self.phases.dtype.kind=='u' else 'Int'
        xdmf.write_text(f'''<Xdmf Version="3.0"><Domain><Grid Name="TetGen">
<Topology TopologyType="Tetrahedron" NumberOfElements="{len(self.cells)}"><DataItem Dimensions="{len(self.cells)} 4" NumberType="{cell_number}" Precision="{self.cells.dtype.itemsize}" Format="HDF">{fn}:/mesh/tetrahedra</DataItem></Topology>
<Geometry GeometryType="XYZ"><DataItem Dimensions="{len(self.points)} 3" Format="HDF" NumberType="Float" Precision="{self.points.dtype.itemsize}">{fn}:/mesh/points_m</DataItem></Geometry>
<Attribute Name="region" Center="Cell"><DataItem Dimensions="{len(self.cells)}" NumberType="{phase_number}" Precision="{self.phases.dtype.itemsize}" Format="HDF">{fn}:/mesh/phase</DataItem></Attribute></Grid></Domain></Xdmf>''')
        return MeshArtifact(path,xdmf,len(self.points),len(self.cells),tuple(map(int,np.unique(self.phases))))


def tetra_quality(points,cells):
    """Measured metrics, including sliver-sensitive minimum dihedral, in bounded chunks."""
    if not np.all(np.isfinite(points)):raise ArithmeticError('Nonfinite Tet4 coordinates')
    values=[];ratios=[];angles=[]
    for start in range(0,len(cells),50000):
        p=points[cells[start:start+50000]];edge=p[:,1:]-p[:,:1]
        volume=np.linalg.det(edge)/6
        if np.any(~np.isfinite(volume)) or np.any(volume<=0):raise ArithmeticError('Nonpositive or nonfinite Tet4 volume')
        lengths=np.linalg.norm(p[:,:,None]-p[:,None,:],axis=3)
        shortest=lengths[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]].min(axis=1)
        centre=np.linalg.solve(2*edge,np.sum(edge**2,axis=2)[...,None])[...,0]
        ratio=np.linalg.norm(centre,axis=1)/shortest
        normals=[]
        for a,b,c in [[1,2,3],[0,3,2],[0,1,3],[0,2,1]]:
            n=np.cross(p[:,b]-p[:,a],p[:,c]-p[:,a]);normals.append(n/np.linalg.norm(n,axis=1)[:,None])
        n=np.stack(normals,axis=1);i,j=np.triu_indices(4,1)
        angle=np.degrees(np.arccos(np.clip(-np.einsum('nki,nki->nk',n[:,i],n[:,j]),-1,1))).min(axis=1)
        if not np.all(np.isfinite(ratio)) or not np.all(np.isfinite(angle)):raise ArithmeticError('Nonfinite Tet4 shape metrics')
        values.append(volume);ratios.append(ratio);angles.append(angle)
    v=np.concatenate(values);r=np.concatenate(ratios);d=np.concatenate(angles)
    return {'minimum_volume_m3':float(v.min()),'total_volume_m3':float(v.sum()),
            'radius_edge_max':float(r.max()),'radius_edge_p95':float(np.quantile(r,.95)),
            'minimum_dihedral_degrees':float(d.min()),'dihedral_p05_degrees':float(np.quantile(d,.05))},v


def _verify_seeds(points,cells,attributes,seeds):
    """Check actual containment; TetGen can ignore seeds and auto-label a region.

    Nearby-centroid search is only an accelerator. A complete bounding-box
    candidate search is the fallback, never a nearest-cell material guess.
    Inputs use normalized coordinates to avoid SI cancellation.
    """
    from scipy.spatial import cKDTree
    vertices=points[cells];centres=vertices.mean(axis=1)
    tree=cKDTree(centres);near=tree.query(seeds[:,:3],k=min(32,len(cells)))[1]
    near=np.asarray(near).reshape(len(seeds),-1)
    bounds=None
    for seed,candidates in zip(seeds,near):
        def contains(indices):
            p=vertices[indices];basis=p[:,1:]-p[:,:1]
            bary=np.linalg.solve(basis.transpose(0,2,1),(seed[:3]-p[:,0])[...,None])[...,0]
            inside=np.all(bary>=-1e-9,axis=1)&(bary.sum(axis=1)<=1+1e-9)
            return np.any(inside&(attributes[indices]==seed[3]))
        if contains(candidates):continue
        if bounds is None:bounds=(vertices.min(axis=1),vertices.max(axis=1))
        low,high=bounds
        candidates=np.flatnonzero(np.all((low-1e-10<=seed[:3])&(high+1e-10>=seed[:3]),axis=1))
        if not len(candidates) or not contains(candidates):
            raise ArithmeticError(f'A region seed is not contained in its requested material: encoded region{int(seed[3])}, normalized XYZ{seed[:3].tolist()}; inspect holes, contours and region labels')


def tetgen_mesh(geometry,*,regions=None,surface_method='voxel_plc',step_size=1,
                maximum_facets=500000,options=None,workspace=None):
    from .tetgen_backend import execute,identity
    from .voxel import VoxelModel
    options=TetGenOptions() if options is None else options
    if not isinstance(options,TetGenOptions):raise TypeError('TetGenOptions required')
    start=time.perf_counter()
    if isinstance(geometry,SurfacePLC):
        if regions is not None:raise ValueError('PLC regions are defined by its seeds')
        plc=geometry
    elif isinstance(geometry,VoxelModel):
        if surface_method=='voxel_plc':plc=voxel_plc(geometry,regions=regions,maximum_facets=maximum_facets)
        elif surface_method=='marching_cubes':
            if regions is None or len(regions)!=1:raise ValueError('Marching cubes requires exactly one explicit region')
            plc=marching_cubes_plc(geometry,region=regions[0],step_size=step_size,maximum_facets=maximum_facets)
        else:raise ValueError('surface_method must be voxel_plc or marching_cubes')
    else:raise TypeError('VoxelModel or SurfacePLC required')
    if len(plc.facets)>maximum_facets:raise ValueError('PLC facet budget exceeded')
    if options.isolated:
        from .tetgen_worker import isolated_mesh
        return isolated_mesh(plc,options,workspace,maximum_facets)
    if options.region_labelling=='interfaces':
        from .interface_labelling import interface_tetgen
        return interface_tetgen(plc,options,workspace)
    centre=plc.points.min(axis=0);scale=float(np.ptp(plc.points,axis=0).max())
    if scale<=0:raise ValueError('Degenerate geometry')
    labels=sorted(plc.seeds);decode=np.array(labels,dtype=np.int64)
    if set(options.region_maximum_volumes)-set(labels):raise ValueError('Unknown region maximum-volume keys')
    seeds=[]
    for index,k in enumerate(labels,1):
        maximum=options.region_maximum_volumes.get(k,options.maximum_volume)
        for p in plc.seeds[k]:seeds.append([*((p-centre)/scale),index,-1 if maximum is None else maximum/scale**3])
    if any(np.any((np.asarray(plc.seeds[k])<plc.points.min(axis=0))|
                  (np.asarray(plc.seeds[k])>plc.points.max(axis=0))) for k in labels):
        raise ValueError('A region seed is outside the PLC bounding box')
    switches=f'p{options.facet_separation_angle:.8f}q{options.radius_edge_ratio:.8g}/{options.minimum_dihedral:.8g}AzfQS{options.maximum_steiner}T{options.coplanarity_tolerance:.8g}'
    if options.maximum_volume is not None or options.region_maximum_volumes:switches+='a'
    if options.preserve_surface:switches+='Y'
    if options.consistency_check:switches+='C'
    if not options.merge_coplanar:switches+='M'
    seed_array=np.array(seeds);refinement_attempts=0
    # TetGen's quality refinement may leave small residual volume violations.
    # Re-run with tighter *input* sizes, keeping the caller's acceptance limit.
    # This adds accuracy rather than relaxing the validation threshold.
    for refinement_attempts in range(3):
        p,t,a,faces,markers=execute((plc.points-centre)/scale,plc.facets,plc.markers,seed_array,
            (plc.holes-centre)/scale,switches,workspace=workspace,maximum_cells=options.maximum_cells)
        retry=False
        volume=np.abs(np.linalg.det(p[t[:,1:]]-p[t[:,:1]]))/6*scale**3
        for index,k in enumerate(labels,1):
            bound=options.region_maximum_volumes.get(k,options.maximum_volume)
            if bound is None:continue
            selected=(a[:,0]==index)
            if not selected.any():continue  # Output seed checks below give a precise error.
            ratio=float(volume[selected].max()/bound)
            if ratio>1+1e-8:
                seed_array[seed_array[:,3]==index,4]*=.8/ratio;retry=True
        if not retry:break
    if retry:raise ArithmeticError('Requested maximum Tet4 volume not reached after three refinements; increase budgets or revise the PLC')
    p=p*scale+centre
    if a.shape[1]!=1:raise ArithmeticError('TetGen did not return exactly one region attribute')
    ids=np.rint(a[:,0]).astype(int)
    if np.any(abs(ids-a[:,0])>1e-8) or set(ids)!=set(range(1,len(labels)+1)):
        raise ArithmeticError('Missing/unseeded region in TetGen output; inspect PLC and seeds')
    _verify_seeds((p-centre)/scale,t,ids,seed_array)
    phases=decode[ids-1]
    # TetGen's internal orientation differs from our positive determinant convention.
    signed=np.linalg.det(p[t[:,1:]]-p[t[:,:1]])
    t[signed<0]=t[signed<0][:,[1,0,2,3]]
    quality,volumes=tetra_quality(p,t)
    errors={}
    for k,expected in plc.region_volumes.items():
        error=abs(float(volumes[phases==int(k)].sum())/expected-1);errors[str(k)]=error
        if error>1e-8:raise ArithmeticError(f'Exact PLC region {k} volume error {error}')
    # f outputs all faces; retain explicitly marked PLC faces including interfaces.
    boundary=markers>0
    report={'mesher':'source-integrated TetGen1.6.0','source_identity':identity(),'switches':switches,
        'input_points':len(plc.points),'input_facets':len(plc.facets),'nodes':len(p),'tetrahedra':len(t),
        'region_volume_relative_errors':errors,'quality':quality,'geometry':plc.metadata,
        'region_volumes_m3':{str(int(k)):float(volumes[phases==k].sum()) for k in np.unique(phases)},
        'elapsed_seconds':time.perf_counter()-start,
        'refinement_attempts':refinement_attempts+1,
        'quality_limits_satisfied':bool(quality['radius_edge_max']<=options.radius_edge_ratio*(1+1e-6) and
            quality['minimum_dihedral_degrees']+1e-6>=options.minimum_dihedral),
        'mapping':'TetGen region attributes; no unique parent voxel for boundary-fitted tetrahedra'}
    if options.require_quality and not report['quality_limits_satisfied']:
        raise ArithmeticError('Measured TetGen quality misses requested limits; inspect PLC small angles and increase refinement budget')
    if 'voxel_volume_m3' in plc.metadata:
        report['voxel_volume_relative_difference']=quality['total_volume_m3']/plc.metadata['voxel_volume_m3']-1
    return TetGenMesh(p,t,phases,facets=faces[boundary],facet_markers=markers[boundary],report=report)
