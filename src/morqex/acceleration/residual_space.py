"""Streamed affine minimum-residual recombination of compatible iterates.

This is established Krylov/residual-smoothing algebra, not a novelty claim or
an error bound. Acceptance must use a fresh physical residual, not this Gram
matrix prediction. Snapshot arrays and all temporary files belong outside src.
"""
from pathlib import Path
import numpy as np

def minimum_residual_weights(residuals,*,chunk=131072):
    arrays=[np.asarray(r).reshape(-1) for r in residuals]
    if not arrays or len(arrays)>16 or any(a.size!=arrays[0].size for a in arrays):raise ValueError('1..16 equally sized residual arrays required')
    gram=np.zeros((len(arrays),)*2)
    for begin in range(0,arrays[0].size,chunk):
        block=np.stack([r[begin:begin+chunk] for r in arrays]);gram+=block@block.T
        for original in residuals:
            if hasattr(original,'_mmap'):
                import mmap
                original._mmap.madvise(mmap.MADV_DONTNEED)
    if not np.isfinite(gram).all():raise ValueError('Finite snapshot residuals required')
    scale=float(np.max(abs(gram)))
    if not scale:return np.eye(len(arrays))[0],0.
    G=gram/scale;system=np.block([[G,np.ones((len(arrays),1))],[np.ones((1,len(arrays))),np.zeros((1,1))]])
    rhs=np.r_[np.zeros(len(arrays)),1.];weights=np.linalg.lstsq(system,rhs,rcond=1e-12)[0][:-1]
    best=int(np.argmin(np.diag(gram)))
    if abs(weights.sum()-1)>1e-10 or np.abs(weights).sum()>100:weights=np.eye(len(arrays))[best]
    prediction=float(weights@gram@weights)
    if prediction>gram[best,best]*(1+1e-10):weights=np.eye(len(arrays))[best];prediction=float(gram[best,best])
    return weights,max(0.,prediction)**.5

def archive_iterate(checkpoint,destination):
    """Hard-link immutable u plus its physical fingerprint, without a data copy."""
    import json
    checkpoint=Path(checkpoint)
    if not (checkpoint/'state.json').exists():return None
    state=json.loads((checkpoint/'state.json').read_text());target=Path(destination)/state['generation'];target.mkdir(parents=True,exist_ok=True)
    if not (target/'u.npy').exists():(target/'u.npy').hardlink_to(checkpoint/'u.npy')
    (target/'state.json').write_text(json.dumps(state,indent=2));return target
