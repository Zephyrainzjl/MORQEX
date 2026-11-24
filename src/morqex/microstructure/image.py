"""Explicit CT processing; phase identities are never inferred from RGB channels."""
from pathlib import Path
import re
import numpy as np
from .volume import Volume

def load_stack(directory,spacing=(1.,1.,1.)):
    import tifffile
    files = sorted([p for p in Path(directory).iterdir() if p.suffix.lower() in ('.tif','.tiff')],
        key=lambda p:[int(s) if s.isdigit() else s for s in re.split(r'(\d+)',p.name)])
    if not files: raise ValueError('No TIFF slices')
    slices = [tifffile.imread(p) for p in files]
    if any(s.ndim != 2 or s.shape != slices[0].shape for s in slices):
        raise ValueError('Slices must be scalar 2D images with identical shape')
    return Volume(np.stack(slices),spacing=spacing,metadata={'source':str(Path(directory).resolve())})

def segment(array,thresholds=None,phases=2):
    from skimage.filters import threshold_multiotsu
    a = np.asarray(array)
    if not np.isfinite(a).all(): raise ValueError('Nonfinite image')
    thresholds = threshold_multiotsu(a,classes=phases) if thresholds is None else np.asarray(thresholds)
    return np.digitize(a,thresholds).astype(np.uint16)

def clean(volume,minimum_size=16):
    from scipy.ndimage import label,convolve
    a = volume.data.copy()
    # Compiled phase counts replace a Python callback at every voxel. Ties use
    # the lowest sorted label, exactly as the original modal operation did.
    labels=np.unique(a)
    majority=np.full(a.shape,labels[0],dtype=a.dtype)
    best=np.zeros(a.shape,np.uint8);kernel=np.ones((3,3,3),np.uint8)
    for phase in labels:
        local=convolve((a==phase).astype(np.uint8),kernel,mode='reflect')
        better=local>best;majority[better]=phase;best[better]=local[better]
    for phase in np.unique(a):
        ids,n = label(a == phase)
        counts = np.bincount(ids.ravel())
        bad = (counts[ids] < minimum_size) & (ids != 0)
        a[bad] = majority[bad]
    return Volume(a,volume.spacing,volume.origin,{**volume.metadata,'cleanup_min_size':minimum_size,
        'cleanup_changed_voxels':int(np.count_nonzero(a!=volume.data))})

def save_slices(volume,directory,axis='z',prefix='slice'):
    import tifffile
    if axis not in ('z','y','x'):raise ValueError('axis=z/y/x')
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    a=volume.data
    i={'z':0,'y':1,'x':2}[axis]
    for index in range(a.shape[i]):
        tifffile.imwrite(d/f'{prefix}_{index:05d}.tif',np.take(a,index,axis=i),photometric='minisblack')
    return d
