"""Small explicit extension registry and TOML front end for geometry recipes."""
from pathlib import Path
import tomllib
import numpy as np
from .voxel import Shape,VoxelModel,from_regions,from_shapes
from .presets import electrode
from .resolved_presets import resolved_electrode

_generators={'shapes':from_shapes,'regions':from_regions,'electrode':lambda **kwargs:electrode(**kwargs)[0],'resolved_electrode':resolved_electrode}
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
    if not isinstance(name,str) or not name.isidentifier() or name in {'tetrahedral','streamed_tetrahedral','tetgen','gemsim_adaptive'} or name in _meshers or not callable(callback):
        raise ValueError('A new mesher identifier and callable are required')
    _meshers[name]=callback


def mesh(model,*,method='streamed_tetrahedral',**parameters):
    from .meshing import tetrahedralize,stream_tetrahedralize,MeshArtifact
    from morqex.microstructure.mesh.voxel import FEMesh
    from .tetgen import SurfacePLC,tetgen_mesh
    from .adaptive import adaptive_mesh
    if not isinstance(model,VoxelModel) and not (method=='tetgen' and isinstance(model,SurfacePLC)):
        raise TypeError('VoxelModel required; TetGen also accepts SurfacePLC')
    callbacks={'tetrahedral':tetrahedralize,'streamed_tetrahedral':stream_tetrahedralize,'tetgen':tetgen_mesh,'gemsim_adaptive':adaptive_mesh,**_meshers}
    if method not in callbacks:raise ValueError('Unknown registered mesher')
    result=callbacks[method](model,**parameters)
    if not isinstance(result,(MeshArtifact,FEMesh)):raise TypeError('Mesh callbacks must return MeshArtifact or FEMesh')
    return result


def from_toml(path):
    path=Path(path).resolve();raw=tomllib.loads(path.read_text())
    if set(raw)-{'geometry','shapes','meshing'}:raise ValueError('Only [geometry], [[shapes]] and [meshing] supported')
    cfg=dict(raw['geometry']);method=cfg.pop('method')
    if method=='shapes':cfg['shapes']=[Shape(**s) for s in raw.get('shapes',[])]
    if method=='regions':
        filename=(path.parent/cfg.pop('array')).resolve()
        cfg['regions']=np.load(filename,allow_pickle=False)
    if 'region_names' in cfg:cfg['region_names']={int(k):v for k,v in cfg['region_names'].items()}
    return generate(method,**cfg)


def mesh_from_toml(model,path,*,overrides=None):
    """Apply a [meshing] recipe to an existing geometry; output paths are explicit."""
    from .tetgen import TetGenOptions
    from .adaptive import AdaptiveMeshOptions
    path=Path(path).resolve();raw=tomllib.loads(path.read_text())
    cfg=dict(raw['meshing']);method=cfg.pop('method','streamed_tetrahedral')
    if 'options' in cfg:
        if method not in ('tetgen','gemsim_adaptive'):raise ValueError('Typed options require tetgen or gemsim_adaptive method')
        options=dict(cfg.pop('options'))
        if 'region_maximum_volumes' in options:
            options['region_maximum_volumes']={int(k):v for k,v in options['region_maximum_volumes'].items()}
        cfg['options']=(TetGenOptions if method=='tetgen' else AdaptiveMeshOptions)(**options)
    if 'tetgen_options' in cfg:
        if method!='gemsim_adaptive':raise ValueError('tetgen_options is only for gemsim_adaptive method')
        values=cfg['tetgen_options']
        if 'region_maximum_volumes' in values:values['region_maximum_volumes']={int(k):v for k,v in values['region_maximum_volumes'].items()}
        cfg['tetgen_options']=TetGenOptions(**values)
    if 'path' in cfg:cfg['path']=(path.parent/cfg['path']).resolve()
    if overrides:cfg.update(overrides)
    return mesh(model,method=method,**cfg)
