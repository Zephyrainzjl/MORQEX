"""Explicit near-spherical particle packs with resolvable CBD/pore gaps.

Continuous source particles and voxel labels are both retained. This is new
geometry generation, not interpolation of a small voxel image. MicroStructPy's
actual Sphere primitive performs the voxel-center classification.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree,distance
from scipy.sparse.csgraph import minimum_spanning_tree
from ..geometry import MicrostructureData
from .statistics import Distribution,connectivity


@dataclass(frozen=True)
class SpherePackSpec:
    shape:tuple=(32,32,32)
    particle_count:int=56
    radius_voxels:float=3.3
    radius_cv:float=.04
    minimum_gap_voxels:float=1.05
    cbd_fraction:float=.14
    spacing_m:tuple=(.8e-6,)*3
    seed:int=104
    radius_distribution:Distribution|None=None
    maximum_relaxation_steps:int=800
    def __post_init__(self):
        if len(self.shape)!=3 or min(self.shape)<8 or any(int(n)!=n for n in self.shape) or self.particle_count<1 or self.radius_voxels<2.5 or not 0<=self.radius_cv<=.3 or self.minimum_gap_voxels<1 or not 0<self.cbd_fraction<1:raise ValueError('Resolved near-spherical grid, positive population and CBD fraction required')
        if len(self.spacing_m)!=3 or not np.isfinite(self.spacing_m).all() or min(self.spacing_m)<=0 or len(set(self.spacing_m))!=1:raise ValueError('This sphere adapter requires positive isotropic SI spacing')


def sphere_pack(spec):
    from ..vendor.microstructpy.geometry import Sphere
    shape=np.array(spec.shape,int);rng=np.random.default_rng(spec.seed)
    counts=np.maximum(1,np.floor(shape/(2*spec.radius_voxels+spec.minimum_gap_voxels)).astype(int))
    if np.prod(counts)<spec.particle_count:raise ValueError('Particle population does not fit the resolved-gap placement budget; use a larger grid, smaller particles or fewer particles')
    sites=np.stack(np.meshgrid(*[(np.arange(n)+.5)*size/n for n,size in zip(counts,shape)],indexing='ij'),axis=-1).reshape(-1,3)
    selected=rng.choice(len(sites),size=spec.particle_count,replace=False);centres=sites[selected]+rng.uniform(-.12,.12,(spec.particle_count,3))
    if spec.radius_distribution is None:
        radii=spec.radius_voxels*np.exp(rng.normal(-.5*spec.radius_cv**2,spec.radius_cv,spec.particle_count))
    else:radii=np.asarray(spec.radius_distribution.sample(rng,spec.particle_count),float)
    if radii.min()<2.5 or np.any(2*radii+spec.minimum_gap_voxels>=shape.min()):raise ValueError('Draw contains under-resolved or non-fitting particle radii; change the declared distribution')
    worst=np.inf
    for iteration in range(spec.maximum_relaxation_steps):
        worst=0.
        for i,j in sorted(cKDTree(centres).query_pairs(2*radii.max()+spec.minimum_gap_voxels)):
            v=centres[j]-centres[i];length=np.linalg.norm(v);overlap=radii[i]+radii[j]+spec.minimum_gap_voxels-length
            if overlap>0:
                if length<1e-12:raise ValueError('Coincident pack seeds')
                correction=.51*overlap*v/length;centres[i]-=correction;centres[j]+=correction;worst=max(worst,overlap)
        centres=np.maximum(centres,radii[:,None]+.1);centres=np.minimum(centres,shape-radii[:,None]-.1)
        if worst<1e-7:break
    if worst>=1e-7:raise ValueError('Nonoverlap relaxation exceeded its budget; no silent shrinking of the requested size distribution')
    # Verify the final constrained positions, including the last boundary clamp.
    distances=distance.squareform(distance.pdist(centres));np.fill_diagonal(distances,np.inf)
    minimum=float(np.min(distances-radii[:,None]-radii[None,:])) if len(radii)>1 else np.inf
    if minimum<spec.minimum_gap_voxels-1e-6:raise ValueError('Final pack violates the minimum gap')
    coordinates=np.indices(tuple(shape),dtype=float).reshape(3,-1).T+.5;owners=np.zeros(int(np.prod(shape)),np.int32)
    for i,(centre,radius) in enumerate(zip(centres,radii)):
        inside=Sphere(radius=float(radius),center=centre[::-1]).within(coordinates[:,::-1])
        if np.any(owners[inside]):raise ValueError('Overlapping source particles after voxelization')
        owners[inside]=i+1
    owners=owners.reshape(tuple(shape));active=owners>0
    components,n=ndimage.label(active)
    if n!=spec.particle_count:raise ValueError('Digital particle count differs from source count; refine the grid/gap')
    # Conductive CBD links follow a minimal particle network, then an interface
    # coating consumes the remaining declared CBD volume. Pores stay true pores.
    tree=minimum_spanning_tree(distance.squareform(distance.pdist(centres))).tocoo()
    segments=[(centres[i],centres[j]) for i,j in zip(tree.row,tree.col)]
    for side in [0,1]:
        i=int(np.argmin(centres[:,2]-radii) if side==0 else np.argmax(centres[:,2]+radii));end=centres[i].copy();end[2]=side*shape[2]
        segments.append((centres[i],end))
    skeleton=np.zeros(len(coordinates),bool)
    for a,b in segments:
        v=b-a;t=np.clip(((coordinates-a)@v)/max(v@v,1e-30),0,1)
        skeleton|=np.sum((coordinates-a-t[:,None]*v)**2,axis=1)<=.85**2
    mandatory=ndimage.binary_dilation(skeleton.reshape(tuple(shape)),iterations=1)&~active
    number=int(round(spec.cbd_fraction*np.prod(shape)))
    if mandatory.sum()>number or number>=np.count_nonzero(~active):raise ValueError('CBD fraction cannot contain the declared connectivity network and a pore phase')
    cbd=mandatory.copy();available=np.flatnonzero(~(active|cbd));needed=number-int(cbd.sum())
    score=ndimage.distance_transform_edt(~active).ravel()[available]+rng.uniform(0,.15,len(available))
    if needed:cbd.ravel()[available[np.argpartition(score,needed-1)[:needed]]]=True
    labels=np.zeros(tuple(shape),np.uint8);labels[active]=1;labels[cbd]=2
    particles=[{'id':i+1,'centre_zyx_voxels':c.tolist(),'radius_voxels':float(r),'radius_m':float(r*spec.spacing_m[0]),'active_voxels':int(np.count_nonzero(owners==i+1))} for i,(c,r) in enumerate(zip(centres,radii))]
    provenance={'provider':'MORQEX resolved-gap spherical pack using pinned MicroStructPy Sphere',
        'recipe':asdict(spec),'particles':particles,'particle_count':len(particles),'minimum_surface_gap_voxels':minimum,
        'relaxation_steps':iteration+1,'CBD':'minimal particle bridges plus surface-biased coating; declared numerical morphology, not calibrated fabrication',
        'phase_fractions':{str(i):float(np.mean(labels==i)) for i in range(3)},
        'scope':'new finite voxel geometry with explicit source spheres; sphere count and phase connectivity checked; finite voxel particle shape still needs resolution convergence'}
    data=MicrostructureData(labels,spec.spacing_m,(0.,)*3,{0:'pore',1:'active',2:'cbd'},provenance)
    if not all(connectivity(data).values()):raise ValueError('Generated CBD/pore networks failed physical through-thickness connectivity')
    return data,owners
