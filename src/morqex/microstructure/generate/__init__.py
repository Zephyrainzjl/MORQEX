"""v0: original MGENERATE GAN; v1: conditional diffusion, orthogonal score fusion."""
from importlib import import_module
from .config import GANConfig, train_gan, sample_gan
from .reconstruction import Reconstructor

def __getattr__(name):
    if name in ('VoxelDiffusionConfig','fit_volume','draw_volume'):
        return getattr(import_module('morqex.microstructure.generate.spatial'),name)
    if name in ('DiffusionConfig', 'train', 'sample'):
        return getattr(import_module('morqex.microstructure.generate.v1'), name)
    raise AttributeError(name)
