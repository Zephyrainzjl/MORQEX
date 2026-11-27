from dataclasses import dataclass
from pathlib import Path
import importlib,importlib.util,json,shutil
import numpy as np
from morqex.microstructure import Volume


def external_path(path):
    target=Path(path).expanduser().resolve()
    container=Path(__file__).resolve().parents[2]
    # A checkout is ROOT/src/morqex; a wheel lives directly in site-packages
    # or a pip --target directory. Do not mistake the installation's parent
    # for a repository and reject legitimate sibling output directories.
    source=(container.parent if container.name=='src' and
            (container.parent/'pyproject.toml').is_file() else container)
    if target.is_relative_to(source):raise ValueError('Generated files must remain outside MORQEX source')
    return target


@dataclass
class VoxelModel:
    regions:np.ndarray
    spacing:tuple=(1e-6,)*3
    origin:tuple=(0.,)*3
    region_names:dict|None=None
    metadata:dict|None=None

    def __post_init__(self):
        a=np.asarray(self.regions)
        if a.ndim!=3 or not a.size or a.dtype.kind not in 'iu' or np.any(a<0):
            raise ValueError('Nonnegative integer region labels in a nonempty ZYX array required')
        if int(a.max())>np.iinfo(np.int32).max:
            raise ValueError('Region IDs must fit nonnegative signed32-bit mesh labels; remap larger IDs explicitly')
        Volume(a,self.spacing,self.origin)
        names={int(k):str(v) for k,v in (self.region_names or {int(i):f'region_{i}' for i in np.unique(a)}).items()}
        if set(np.unique(a))!=set(names):raise ValueError('Name every present region exactly once')
        self.regions=a;self.region_names=names;self.metadata=dict(self.metadata or {})

    def volume(self):return Volume(self.regions,self.spacing,self.origin,self.metadata)

    def save(self,path):
        import h5py
        path=external_path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(path,'w') as f:
            f.create_dataset('phase_labels',data=self.regions,compression='gzip',shuffle=True)
            f.attrs.update(spacing_zyx_m=self.spacing,origin_zyx_m=self.origin,
                region_names=json.dumps(self.region_names),metadata=json.dumps(self.metadata))
        return path

    @classmethod
    def load(cls,path):
        import h5py
        with h5py.File(path) as f:
            return cls(f['phase_labels'][:],tuple(f.attrs['spacing_zyx_m']),tuple(f.attrs['origin_zyx_m']),
                json.loads(f.attrs['region_names']),json.loads(f.attrs['metadata']))

    def tetrahedralize(self,**kwargs):
        from .meshing import tetrahedralize
        return tetrahedralize(self,**kwargs)

    def tetgen(self,**kwargs):
        from .tetgen import tetgen_mesh
        return tetgen_mesh(self,**kwargs)

    def adaptive_surface(self,**kwargs):
        from .adaptive import adaptive_surface
        return adaptive_surface(self,**kwargs)

    def adaptive_mesh(self,**kwargs):
        from .adaptive import adaptive_mesh
        return adaptive_mesh(self,**kwargs)

    def mesh_to(self,path,**kwargs):
        from .meshing import stream_tetrahedralize
        return stream_tetrahedralize(self,path,**kwargs)

    def show(self,**kwargs):
        """Display a live notebook view; return its update/save handle."""
        from morqex.visualization.notebook import show
        return show(self,**kwargs)

    def viewer(self,**kwargs):
        """Return a rich-display object without explicitly displaying it."""
        from morqex.visualization.notebook import VoxelView
        return VoxelView(self,**kwargs)


def from_regions(regions,*,spacing=(1e-6,)*3,origin=(0.,)*3,region_names=None,metadata=None):
    return VoxelModel(regions,tuple(spacing),tuple(origin),region_names,metadata)


@dataclass(frozen=True)
class Shape:
    kind:str
    region:int
    centre:tuple
    axes:tuple
    rotation:tuple|None=None
    irregularity:float=0.

    def __post_init__(self):
        if self.kind not in ('sphere','ellipsoid','box','cylinder','capsule','platelet'):
            raise ValueError('Unknown shape primitive')
        if type(self.region) is not int or self.region<0 or len(self.centre)!=3 or len(self.axes)!=3:
            raise ValueError('An integer region and XYZ? Use documented ZYX centres/axes')
        if not np.isfinite([*self.centre,*self.axes,self.irregularity]).all() or min(self.axes)<=0 or not 0<=self.irregularity<.35:
            raise ValueError('Finite centre, positive axes and bounded irregularity required')


def from_shapes(shapes,*,shape=(256,)*3,spacing=(1e-6,)*3,origin=(0.,)*3,background=0,region_names=None,
                progress=None,progress_every=10):
    """Voxelize SI ZYX primitives; later shapes overwrite earlier shapes.

    Rotation is a proper 3x3 matrix acting on local ZYX vectors. Cylinder and
    capsule long axes are local axis2; axes=(radius,radius,half_length).
    """
    if len(shape)!=3 or any(type(n) is not int or n<1 for n in shape):raise ValueError('Positive integer ZYX shape required')
    if np.prod(shape)>256**3:raise ValueError('Explicitly bounded API: at most256^3 cells')
    shapes=tuple(shapes)
    if progress is not None and not callable(progress):raise ValueError('Progress must be a callable')
    if type(progress_every) is not int or progress_every<1:raise ValueError('Positive progress interval required')
    if any(not isinstance(item,Shape) for item in shapes):raise ValueError('Shape instances required')
    if type(background) is not int or background<0:raise ValueError('Nonnegative background region required')
    largest=max([background,*[item.region for item in shapes]])
    dtype=np.result_type(np.min_scalar_type(largest),np.uint8)
    volume=Volume(np.full(shape,background,dtype),spacing,origin)
    last_progress=0
    for step,item in enumerate(shapes,1):
        if not isinstance(item,Shape):raise ValueError('Shape instances required')
        R=np.eye(3) if item.rotation is None else np.asarray(item.rotation,float)
        if R.shape!=(3,3) or not np.allclose(R.T@R,np.eye(3),atol=1e-10) or np.linalg.det(R)<.999999:
            raise ValueError('Proper orthogonal ZYX rotation required')
        centre=np.asarray(item.centre);axes=np.asarray(item.axes)
        envelope=axes.copy()
        if item.kind=='capsule':envelope[2]+=axes[0]
        if item.kind=='sphere':envelope[:]=axes[0]
        reach=(abs(R)@envelope)*1.35
        low=np.maximum(0,np.floor((centre-reach-volume.origin)/volume.spacing).astype(int))
        high=np.minimum(shape,np.ceil((centre+reach-volume.origin)/volume.spacing).astype(int))
        if np.any(high<=low):continue
        indices=np.indices(tuple(high-low),dtype=np.float32).reshape(3,-1).T+low
        local=((indices+.5)*volume.spacing+volume.origin-centre)@R
        if item.irregularity:
            q=local/axes
            warp=1+item.irregularity*np.sin(3*q[:,0]+q[:,1])*np.cos(4*q[:,2]-q[:,0])
            local/=warp[:,None]
        if item.kind in ('sphere','ellipsoid','platelet'):
            from morqex.vendor.microstructpy.geometry import Sphere,Ellipsoid
            primitive=Sphere(radius=float(axes[0])) if item.kind=='sphere' else Ellipsoid(axes=axes[::-1])
            inside=primitive.within(local[:,::-1])
        elif item.kind=='box':inside=np.all(abs(local)<=axes,axis=1)
        elif item.kind=='cylinder':inside=(np.sum(local[:,:2]**2,axis=1)<=axes[0]**2)&(abs(local[:,2])<=axes[2])
        else:
            axial=np.maximum(abs(local[:,2])-axes[2],0)
            inside=np.sum(local[:,:2]**2,axis=1)+axial**2<=axes[0]**2
        chosen=indices[inside].astype(int);volume.data[tuple(chosen.T)]=item.region
        if progress is not None and (step%progress_every==0 or step==len(shapes)):
            progress(VoxelModel(volume.data,tuple(spacing),tuple(origin),metadata={
                'completed_shapes':step,'total_shapes':len(shapes),'status':'partial construction snapshot'}))
            last_progress=step
    if progress is not None and shapes and last_progress!=len(shapes):
        progress(VoxelModel(volume.data,tuple(spacing),tuple(origin),metadata={
            'completed_shapes':len(shapes),'total_shapes':len(shapes),'status':'partial construction snapshot'}))
    return VoxelModel(volume.data,tuple(spacing),tuple(origin),region_names,
        {'generator':'MORQEX unified physical shape rasterizer','overlap':'last shape wins','coordinate_order':'ZYX; SI metres'})


BACKENDS={'shapes':'morqex.vendor.microstructpy.geometry','seeds':'morqex.vendor.microstructpy.seeding',
    'tessellation':'morqex.vendor.microstructpy.meshing','statistics':'morqex.generation.statistics',
    'gem':'morqex.microstructure','reconstruction':'morqex.microstructure.generate.reconstruction',
    'statistical_adapters':'morqex.generation.upstream','finite_elements':'morqex.microstructure.mesh'}


def framework(name):
    """MORQEX namespace portal to each integrated backend's complete public API.

    All original keyword arguments pass through unchanged. This is an adapter,
    not a claim to have reimplemented or verified every upstream method.
    """
    if name not in BACKENDS:raise ValueError('Available framework APIs: '+', '.join(BACKENDS))
    return importlib.import_module(BACKENDS[name])


def capabilities():
    from .tetgen_backend import library_path
    try:tetgen={'compiled':True,'library':str(library_path())}
    except RuntimeError:tetgen={'compiled':False,'enable':'MORQEX_BUILD_TETGEN=1 pip install . or build_tetgen(workspace)'}
    from .acvd_backend import load_acvd
    try:acvd={'compiled':True,'library':str(load_acvd().__file__)}
    except RuntimeError:acvd={'compiled':False,'enable':'MORQEX_BUILD_ACVD=1 pip install . or build_acvd(workspace)'}
    from .mmg_backend import library_path as mmg_library
    try:mmg={'compiled':True,'library':str(mmg_library())}
    except RuntimeError:mmg={'compiled':False,'enable':'MORQEX_BUILD_MMG=1 pip install . or build_mmg(workspace)'}
    return {'frameworks':BACKENDS,'tetgen':tetgen,'weighted_acvd':acvd,'mmg':mmg,'voxel_shape_limit':256**3,'coordinates':'ZYX arrays and SI spacing; XYZ FEM',
        'native_reconstruction':{'matlab':shutil.which('matlab'),'Rscript':shutil.which('Rscript'),
            'status':'Integrated MATLAB/R source; native execution requires original licensed runtimes and packages'},
        'optional':{name:importlib.util.find_spec(name) is not None for name in ['pyvoro','meshpy','gmsh','meshio','h5py','vtkmodules','triangle','tifffile']}}
