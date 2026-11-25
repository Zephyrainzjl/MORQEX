import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from morqex.acceleration.transport import PeriodicTransport

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--kind',choices=['electrolyte','electrode'],required=True);p.add_argument('--cbd-ionic',type=float,default=.2);a=p.parse_args()
    w=Path(os.environ['MORQEX_WORKSPACE']);directory=w/'operators/full_volume'/a.dataset;manifest=json.loads((directory/'manifest.json').read_text())
    if hashlib.sha256((directory/'geometry.npz').read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('CT geometry checksum mismatch')
    with np.load(directory/'geometry.npz') as data:phase=data['phase']
    coefficient=[1.,0.,a.cbd_ionic] if a.kind=='electrolyte' else [0.,1.,1.]
    if not 0<=a.cbd_ionic<=1:raise ValueError('Explicit CBD pore-scale ionic efficiency must be in [0,1]')
    model=PeriodicTransport(phase,coefficient,spacing=(manifest['voxel_size_m'],)*3);_,metrics=model.solve(rtol=1e-7,maxiter=10000)
    report={'dataset':a.dataset,'kind':a.kind,'phase_coefficients':coefficient,'metrics':metrics,'geometry_sha256':manifest['sha256'],
        'scope':'full provided CT volume; periodic homogenization; CBD coefficient is an explicitly uncalibrated study parameter'}
    (directory/(a.kind+'_transport.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
