"""GEM: lazy public imports keep mesh/AI/simulation dependencies independent."""
from importlib import import_module
from .workspace import Workspace
from .volume import Volume
from .pipeline import Microstructure

__version__ = '0.1.0'
_legacy = {'DataVol': 'core.voxel_fields', 'ObjVol': 'core.voxel_fields',
           'PhaseVol': 'core.voxel_fields', 'Mesh': 'mesh.elements', 'TetMesh': 'mesh.elements',
           'TriMesh': 'mesh.elements', 'HexMesh': 'mesh.elements', 'Voxelizer': 'core.rasterization'}

def __getattr__(name):
    if name in _legacy:
        return getattr(import_module('morqex.microstructure.' + _legacy[name]), name)
    if name in ('generate', 'mesh', 'simulate', 'analysis', 'view', 'image', 'design'):
        return import_module('morqex.microstructure.' + name)
    raise AttributeError(name)
