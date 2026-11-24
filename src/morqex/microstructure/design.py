"""Parameterized stochastic level-set baseline for ablations and morphology design."""
import numpy as np
from scipy.ndimage import gaussian_filter
from .volume import Volume

def random_field(shape=(64,64,64),fractions=(.3,.7),length_scale=(3.,3.,3.),seed=0,spacing=(1.,1.,1.)):
    f = np.asarray(fractions)
    if np.any(f < 0) or abs(f.sum()-1)>1e-8: raise ValueError('Invalid fractions')
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.standard_normal(shape),sigma=length_scale,mode='wrap')
    order = np.argsort(field.ravel())
    counts = np.floor(f*field.size).astype(int)
    counts[np.argsort(-(f*field.size-counts))[:field.size-counts.sum()]] += 1
    labels = np.repeat(np.arange(len(f)),counts)
    out = np.empty(field.size,np.uint16);out[order] = labels
    return Volume(out.reshape(shape),spacing=spacing,metadata={'method':'Gaussian random field','seed':seed})
