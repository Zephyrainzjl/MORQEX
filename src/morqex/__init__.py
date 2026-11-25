"""MORQEX: microstructure, reconstruction and energy-consistent multiscale research."""
from importlib import import_module

__version__ = "0.0.9"
__all__ = ["MicrostructureData", "Workspace"]


def __getattr__(name):
    # Resource launchers import workspace before child limits are installed.
    # Keep this root import free of NumPy, BLAS, Torch and GPU initialization.
    if name == "Workspace":
        return import_module(".workspace", __name__).Workspace
    if name == "MicrostructureData":
        return import_module(".geometry", __name__).MicrostructureData
    raise AttributeError(name)

