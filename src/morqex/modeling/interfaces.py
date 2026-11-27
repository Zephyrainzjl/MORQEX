"""Small explicit extension registry and TOML front end for geometry recipes."""
from pathlib import Path
import tomllib
import numpy as np
from .voxel import Shape,VoxelModel,from_regions,from_shapes
from .presets import electrode

_generators={'shapes':from_shapes,'regions':from_regions,'electrode':lambda **kwargs:electrode(**kwargs)[0]}
_meshers={}


def register_generator(name,callback):
    if not isinstance(name,str) or not name.isidentifier() or name in _generators or not callable(callback):
        raise ValueError('A new identifier and callable are required')
    _generators[name]=callback


def generate(method,**parameters):
    if method not in _generators:raise ValueError('Unknown registered geometry generator')
    result=_generators[method](**parameters)
    if not isinstance(result,VoxelModel):raise TypeError('Custom generators must return VoxelModel')
    return result


def register_mesher(name,callback):
    if not isinstance(name,str) or not name.isidentifier() or name in {'tetrahedral','streamed_tetrahedral'} or name in _meshers or not callable(callback):
        raise ValueError('A new mesher identifier and callable are required')
    _meshers[name]=callback


def mesh(model,*,method='streamed_tetrahedral',**parameters):
    from .meshing import tetrahedralize,stream_tetrahedralize,MeshArtifact
    from morqex.microstructure.mesh.voxel import FEMesh
    if not isinstance(model,VoxelModel):raise TypeError('VoxelModel input required')
    callbacks={'tetrahedral':tetrahedralize,'streamed_tetrahedral':stream_tetrahedralize,**_meshers}
    if method not in callbacks:raise ValueError('Unknown registered mesher')
    result=callbacks[method](model,**parameters)
    if not isinstance(result,(MeshArtifact,FEMesh)):raise TypeError('Mesh callbacks must return MeshArtifact or FEMesh')
    return result


def from_toml(path):
    path=Path(path).resolve();raw=tomllib.loads(path.read_text())
    if set(raw)-{'geometry','shapes'}:raise ValueError('Only [geometry] and [[shapes]] supported')
    cfg=dict(raw['geometry']);method=cfg.pop('method')
    if method=='shapes':cfg['shapes']=[Shape(**s) for s in raw.get('shapes',[])]
    if method=='regions':
        filename=(path.parent/cfg.pop('array')).resolve()
        cfg['regions']=np.load(filename,allow_pickle=False)
    if 'region_names' in cfg:cfg['region_names']={int(k):v for k,v in cfg['region_names'].items()}
    return generate(method,**cfg)
