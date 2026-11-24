"""Explicit material priors for constrained generation (not learned model accuracy)."""
import numpy as np
from scipy.ndimage import distance_transform_edt,gaussian_filter
from morqex.microstructure.volume import Volume

def surface_coating(volume,substrate,coating,matrix,coating_fraction,seed=0,roughness=.15):
    """Project a binder phase onto the exterior of an active-material skeleton.

    Applicable only when binder is physically an interfacial coating. Preserves
    substrate voxels and allocates the requested binder count in the matrix.
    It does not guarantee percolation or correct transport, which need evaluation.
    """
    if len({substrate,coating,matrix})!=3 or not 0<=coating_fraction<=1:
        raise ValueError('Three distinct phase IDs and a valid fraction are required')
    a=volume.data.copy()
    if not np.isin(a,[substrate,coating,matrix]).all():raise ValueError('Unexpected phases')
    solid=a==substrate
    if not solid.any():raise ValueError('No substrate')
    count=round(coating_fraction*a.size)
    available=np.flatnonzero(~solid)
    if count>len(available):raise ValueError('Coating cannot replace protected substrate')
    distance=distance_transform_edt(~solid,sampling=volume.spacing)
    rng=np.random.default_rng(seed)
    texture=gaussian_filter(rng.standard_normal(a.shape),sigma=1.,mode='wrap')
    distance=distance+roughness*min(volume.spacing)*texture
    a[~solid]=matrix
    if count:
        selected=available[np.argpartition(distance.ravel()[available],count-1)[:count]]
        a.ravel()[selected]=coating
    return Volume(a,volume.spacing,volume.origin,{**volume.metadata,
        'material_prior':{'type':'surface_coating','substrate':int(substrate),
            'coating':int(coating),'matrix':int(matrix),'coating_fraction':coating_fraction,
            'roughness':roughness,'seed':seed}})
