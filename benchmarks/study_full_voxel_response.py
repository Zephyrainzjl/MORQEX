"""One restartable full supplied-volume CUDA elasticity response, no cropping."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import tifffile
from morqex.acceleration.voxel import NativeStencil
from morqex.mechanics import isotropic_stiffness

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='nmc1cal');p.add_argument('--iterations',type=int,default=100);p.add_argument('--rtol',type=float,default=1e-6);p.add_argument('--weak-pore',type=float,default=0.);p.add_argument('--preconditioner',choices=['fft','jacobi','coarse','phase-coarse','geometric'],default='coarse');a=p.parse_args()
    ws=Path(os.environ['MORQEX_WORKSPACE']);record=json.loads((ws/'data/ct/full_electrodes.json').read_text())[a.dataset]
    phase=np.ascontiguousarray(tifffile.imread(record['path']).transpose(record['axes_to_zyx']))
    if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Raw CT checksum mismatch')
    C=np.stack([isotropic_stiffness(a.weak_pore*375e9,.2) if a.weak_pore else np.zeros((6,6)),isotropic_stiffness(375e9,.2),isotropic_stiffness(1e9,.3)])
    beta=np.zeros((3,6));beta[1,:3]=1.25e-6/3;alpha=np.zeros_like(beta);alpha[1:,:3]=1e-5
    model=NativeStencil(phase,C,beta,alpha,spacing=(record['voxel_size_m'],)*3,backend='cuda',preconditioner=a.preconditioner);cp=model.xp
    concentration=cp.where(model.phase==1,5000.,0.);out=ws/'results/full_voxel'/a.dataset/f'weak_{a.weak_pore:.0e}';out.mkdir(parents=True,exist_ok=True)
    from morqex.acceleration.residual_space import archive_iterate
    archive_iterate(out/'checkpoint',out/'snapshots')
    u,metrics=model.solve(concentration,rtol=a.rtol,maxiter=a.iterations,checkpoint=out/'checkpoint')
    archive_iterate(out/'checkpoint',out/'snapshots')
    report={'dataset':a.dataset,'source':record,'shape':list(phase.shape),'voxels':phase.size,'degrees_of_freedom':3*phase.size,
        'elasticity_Pa':C.tolist(),'chemical_strain_per_mol_m3':beta.tolist(),'thermal_strain_per_K':alpha.tolist(),
        'input_concentration_deviation_mol_m3':5000.,'input_macro_strain':[0.]*6,'input_temperature_deviation_K':0.,
        'weak_pore_ratio':a.weak_pore,'metrics':metrics,'scope':'full provided volume, periodic compatible FD, true pores when ratio=0, fixed isotropic bulk elasticity; not a calibrated lifetime result'}
    if metrics['converged']:
        from morqex.visualization.voxel_fields import export_reference_fields
        import gc
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
        report['fields']=export_reference_fields(model,u,concentration,out/'fields')
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
