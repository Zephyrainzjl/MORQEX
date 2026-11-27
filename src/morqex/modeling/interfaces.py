"""Small explicit extension registry and TOML front end for geometry recipes."""
from pathlib import Path
import tomllib
import numpy as np
from .voxel import Shape,VoxelModel,from_regions,from_shapes
from .presets import electrode
from .resolved_presets import resolved_electrode
from .dense import dense_electrode,coated_electrode,porous_layer,uniform_layer,stack_layers
from .imports import tiff_voxels

_generators={'shapes':from_shapes,'regions':from_regions,'electrode':lambda **kwargs:electrode(**kwargs)[0],
    'resolved_electrode':resolved_electrode,'dense_electrode':dense_electrode,'coated_electrode':coated_electrode,
    'porous_layer':porous_layer,'uniform_layer':uniform_layer,'tiff':tiff_voxels,'stack_layers':stack_layers}
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
    if not isinstance(name,str) or not name.isidentifier() or name in {'tetrahedral','streamed_tetrahedral','tetgen','gemsim_adaptive','controlled_tetgen','partitioned_tetgen'} or name in _meshers or not callable(callback):
        raise ValueError('A new mesher identifier and callable are required')
    _meshers[name]=callback


def mesh(model,*,method='streamed_tetrahedral',**parameters):
    from .meshing import tetrahedralize,stream_tetrahedralize,MeshArtifact
    from morqex.microstructure.mesh.voxel import FEMesh
    from .tetgen import SurfacePLC,tetgen_mesh
    from .adaptive import adaptive_mesh
    from .resolution import controlled_tetgen
    from .partition_workflow import partitioned_tetgen
    if not isinstance(model,VoxelModel) and not (method=='tetgen' and isinstance(model,SurfacePLC)):
        raise TypeError('VoxelModel required; TetGen also accepts SurfacePLC')
    callbacks={'tetrahedral':tetrahedralize,'streamed_tetrahedral':stream_tetrahedralize,'tetgen':tetgen_mesh,'gemsim_adaptive':adaptive_mesh,'controlled_tetgen':controlled_tetgen,'partitioned_tetgen':partitioned_tetgen,**_meshers}
    if method not in callbacks:raise ValueError('Unknown registered mesher')
    result=callbacks[method](model,**parameters)
    if not isinstance(result,(MeshArtifact,FEMesh)):raise TypeError('Mesh callbacks must return MeshArtifact or FEMesh')
    return result


def from_toml(path):
    path=Path(path).resolve();raw=tomllib.loads(path.read_text())
    if set(raw)-{'geometry','shapes','meshing','layers','exports','remeshing','native_build'}:raise ValueError('Unknown geometry recipe section')
    cfg=dict(raw['geometry']);method=cfg.pop('method')
    if method=='shapes':cfg['shapes']=[Shape(**s) for s in raw.get('shapes',[])]
    if method=='regions':
        filename=(path.parent/cfg.pop('array')).resolve()
        cfg['regions']=np.load(filename,allow_pickle=False)
    if method=='tiff':cfg['path']=(path.parent/cfg['path']).resolve()
    if method=='stack_layers':
        layers=[]
        for layer in raw.get('layers',[]):
            item=dict(layer['geometry']);name=layer['name'];kind=item.pop('method')
            if kind=='regions':item['regions']=np.load((path.parent/item.pop('array')).resolve(),allow_pickle=False)
            if kind=='tiff':item['path']=(path.parent/item['path']).resolve()
            if 'shape' in item:item['shape']=tuple(item['shape'])
            layers.append((name,generate(kind,**item)))
        cfg['layers']=layers
    if 'shape' in cfg:cfg['shape']=tuple(cfg['shape'])
    if 'region_names' in cfg:cfg['region_names']={int(k):v for k,v in cfg['region_names'].items()}
    return generate(method,**cfg)


def mesh_from_toml(model,path,*,overrides=None):
    """Apply a [meshing] recipe to an existing geometry; output paths are explicit."""
    from .tetgen import TetGenOptions
    from .adaptive import AdaptiveMeshOptions
    path=Path(path).resolve();raw=tomllib.loads(path.read_text())
    cfg=dict(raw['meshing']);method=cfg.pop('method','streamed_tetrahedral')
    if 'resolution' in cfg:
        from .resolution import MeshResolution
        if method!='controlled_tetgen':raise ValueError('resolution requires controlled_tetgen')
        cfg['resolution']=MeshResolution(**cfg['resolution'])
    if 'options' in cfg:
        if method not in ('tetgen','gemsim_adaptive'):raise ValueError('Typed options require tetgen or gemsim_adaptive method')
        options=dict(cfg.pop('options'))
        if 'region_maximum_volumes' in options:
            options['region_maximum_volumes']={int(k):v for k,v in options['region_maximum_volumes'].items()}
        cfg['options']=(TetGenOptions if method=='tetgen' else AdaptiveMeshOptions)(**options)
    if 'tetgen_options' in cfg:
        if method not in ('gemsim_adaptive','controlled_tetgen','partitioned_tetgen'):raise ValueError('tetgen_options requires an adaptive, partitioned or controlled TetGen method')
        values=cfg['tetgen_options']
        if 'region_maximum_volumes' in values:values['region_maximum_volumes']={int(k):v for k,v in values['region_maximum_volumes'].items()}
        cfg['tetgen_options']=TetGenOptions.for_adaptive(**values)
    if 'adaptive_options' in cfg:
        if method!='controlled_tetgen':raise ValueError('adaptive_options requires controlled_tetgen')
        cfg['adaptive_options']=AdaptiveMeshOptions(**cfg['adaptive_options'])
    if 'path' in cfg:cfg['path']=(path.parent/cfg['path']).resolve()
    if method=='partitioned_tetgen' and 'directory' in cfg:cfg['directory']=(path.parent/cfg['directory']).resolve()
    if overrides:cfg.update(overrides)
    return mesh(model,method=method,**cfg)


def export_from_toml(mesh,model,path,*,directory=None):
    """Apply [exports]; coordinates remain SI internally, output units explicit."""
    from .exchange import export_bundle
    path=Path(path).resolve();cfg=dict(tomllib.loads(path.read_text()).get('exports',{}))
    target=directory or cfg.pop('directory',None)
    if target is None:raise ValueError('Explicit external export directory required')
    cfg.pop('directory',None)
    return export_bundle(mesh,(path.parent/target).resolve(),region_names=model.region_names,
        selections=model.metadata.get('selection_groups'),**cfg)


def remesh_from_toml(mesh,path,*,workspace=None,output=None):
    """Apply explicit [remeshing] to an existing Tet4 mesh, not voxel labels."""
    from .remeshing import RemeshOptions,remesh,remesh_artifact
    from .meshing import MeshArtifact
    raw=tomllib.loads(Path(path).read_text())
    if 'remeshing' not in raw:raise ValueError('TOML requires an explicit [remeshing] section')
    options=RemeshOptions(**raw['remeshing'])
    if isinstance(mesh,MeshArtifact):
        if output is None:raise ValueError('Artifact remeshing requires an explicit external output path')
        return remesh_artifact(mesh,output,options=options,workspace=workspace)
    if output is not None:raise ValueError('Use returned TetGenMesh.save_hdf5 for an in-memory input')
    return remesh(mesh,options=options,workspace=workspace)
