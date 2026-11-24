from .voxel import FEMesh, voxel_to_tetra, voxel_to_hex, surface

def __getattr__(name):
    from importlib import import_module
    if name.startswith('__'):
        raise AttributeError(name)
    module = import_module('morqex.microstructure.mesh.elements')
    return getattr(module, name)
