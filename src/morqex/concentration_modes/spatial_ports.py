"""Compact spatial ports from every voxel of a supplied CT volume.

One global modal inventory is shared across the macro bins. No independent
copy of that inventory is made at each DFN position. The resulting interface
still needs a global-state DFN submodel; this constructor does not claim that
the legacy per-X particle model has automatically acquired those equations.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .positive import ComponentBasis,full_features
from ..electrochemistry.conservative_ports import nodal_ports

def build_spatial_ports(directory,*,bins=5,chunk=32768):
    started=time.perf_counter();directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text());record=json.loads((directory/'concentration.json').read_text())
    for filename,digest in [('geometry.npz',meta['sha256']),('concentration.npz',record['sha256'])]:
        if hashlib.sha256((directory/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Geometry/concentration checksum mismatch')
    if not isinstance(bins,int) or not 1<=bins<=128:raise ValueError('1..128 explicit spatial bins required')
    basis=ComponentBasis.from_metadata(record['basis']);m=basis.dimension
    if m>512:raise MemoryError('Dense online port prototype capped at 512 states; use a sparse/block port backend for larger independent inventories')
    chunk=min(chunk,max(1,2**21//m))
    with np.load(directory/'geometry.npz') as a:components=a['components'];features=full_features(a);faces=a['faces']
    N=components.size;nx=components.shape[2]
    if bins>nx:raise ValueError('Macro bins cannot be thinner than one voxel in this voxel-aligned adapter')
    labels=np.arange(nx)*bins//nx;macro_volumes=np.bincount(labels,minlength=bins)/nx
    W=np.zeros((m,bins));A=np.zeros_like(W);flat=components.ravel();feature=features.reshape(3,-1);h=meta['voxel_size_m']
    for begin in range(0,N,chunk):
        ids=np.arange(begin,min(N,begin+chunk));ids=ids[flat[ids]>0]
        if not len(ids):continue
        P=basis.evaluate(flat[ids],feature[:,ids]);macro=labels[ids%nx]
        for p in np.unique(macro):
            selected=macro==p;W[:,p]+=P[selected].sum(axis=0)/N;A[:,p]+=P[selected].T@faces.ravel()[ids[selected]]/(N*h)
    with np.load(directory/'concentration.npz') as data:
        np.testing.assert_allclose(W.sum(axis=1),data['weights'],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(A.sum(axis=1),data['reaction_area'],rtol=1e-12,atol=1e-10)
    ports=nodal_ports(W.sum(axis=1),A,macro_volumes,provenance='complete CT positive POU; voxel-aligned spatial reaction quadrature')
    np.savez_compressed(directory/f'spatial_ports_{bins}.npz',volume_by_mode_macro=W,area_by_mode_macro=A,macro_volumes=macro_volumes,macro_bin_by_x_voxel=labels,
        site_modes=ports.site_modes,macro_interpolation=ports.macro_interpolation,site_area_weights=ports.area_weights)
    report={'bins':bins,'states':m,'reaction_ports':len(ports.area_weights),'geometry_sha256':meta['sha256'],'all_voxels':N,
        'macro_volumes':macro_volumes.tolist(),'active_fraction_by_bin':(W.sum(axis=0)/macro_volumes).tolist(),'seconds':time.perf_counter()-started,
        'mass_discrepancy':float(abs(W.sum()-meta['phase_fractions']['1'])),
        'scope':'one global modal inventory, full provided volume partitioned into voxel-aligned bins; FV cell-centre surface quadrature; interface matrices only, global DFN state integration is separate',
        'coordinates':'ZYX; X is the source declared through-plane axis; use the returned nonuniform bin volumes, not assumed equal DFN quadrature weights',
        'sha256':hashlib.sha256((directory/f'spatial_ports_{bins}.npz').read_bytes()).hexdigest()}
    (directory/f'spatial_ports_{bins}.json').write_text(json.dumps(report,indent=2));return report
