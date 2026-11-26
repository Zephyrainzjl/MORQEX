"""Pinned MicroStructPy; optional meshing backends load on demand."""
from importlib import import_module
__version__ = '1.6.1'
def __getattr__(name):
    if name in {"cli","geometry","meshing","seeding","verification"}:
        module=import_module("."+name,__name__);globals()[name]=module;return module
    raise AttributeError(name)
