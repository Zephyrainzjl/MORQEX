"""Fresh FP64 residual minimization over compatible full-volume iterates."""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import time
import uuid
import numpy as np
import tifffile
from morqex.acceleration.voxel import NativeStencil,SOURCE
from morqex.acceleration.residual_space import minimum_residual_weights,archive_iterate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='nmc1cal');parser.add_argument('--rtol',type=float,default=1e-6);args=parser.parse_args()
    started=time.perf_counter();ws=Path(os.environ['MORQEX_WORKSPACE']);directory=ws/'results/full_voxel'/args.dataset/'weak_0e+00';report=json.loads((directory/'result.json').read_text());record=report['source']
    if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('CT checksum mismatch')
    phase=np.ascontiguousarray(tifffile.imread(record['path']).transpose(record['axes_to_zyx']));model=NativeStencil(phase,report['elasticity_Pa'],report['chemical_strain_per_mol_m3'],report['thermal_strain_per_K'],spacing=(record['voxel_size_m'],)*3,backend='cuda');cp=model.xp
    macro=report['input_macro_strain'];T=report['input_temperature_deviation_K'];c=cp.where(model.phase==1,report['input_concentration_deviation_mol_m3'],0.)
    identity=hashlib.sha256(model.phase_host.tobytes()+model.C_host.tobytes()+model.beta_host.tobytes()+model.alpha_host.tobytes()+np.asarray([*macro,T,*model.spacing]).tobytes()+SOURCE.read_bytes());identity.update(cp.asnumpy(c).tobytes());fingerprint=identity.hexdigest()
    archive_iterate(directory/'checkpoint',directory/'snapshots');paths=sorted((directory/'snapshots').glob('*'))
    if (directory/'checkpoint_phase_537').exists():paths.insert(0,directory/'checkpoint_phase_537')
    paths=paths[-16:];u=cp.empty((3,*model.shape));r=cp.empty_like(u);model.load(c,macro,T,out=r);norm0=model.budget.dot(r,r)**.5;residuals=[];norms=[]
    for path in paths:
        state=json.loads((path/'state.json').read_text())
        if state['fingerprint']!=fingerprint:raise ValueError('Snapshot physical model/load mismatch')
        stored=np.load(path/'u.npy',mmap_mode='r')
        if stored.shape!=u.shape or stored.dtype!=np.float64:raise ValueError('Snapshot dimensions/precision mismatch')
        for begin in range(0,u.size,131072):
            u.ravel()[begin:begin+131072]=cp.asarray(stored.ravel()[begin:begin+131072]);stored._mmap.madvise(mmap.MADV_DONTNEED)
        with model.budget.batch():model.residual(u,c,macro,T,out=r);norms.append(model.budget.dot(r,r)**.5/norm0)
        target=np.lib.format.open_memmap(path/'smoothing_residual.npy',mode='w+',dtype='float64',shape=u.shape)
        for begin in range(0,u.size,131072):
            target.ravel()[begin:begin+131072]=cp.asnumpy(r.ravel()[begin:begin+131072]);target.flush();target._mmap.madvise(mmap.MADV_DONTNEED)
        residuals.append(target)
    weights,prediction=minimum_residual_weights(residuals);iterates=[np.load(path/'u.npy',mmap_mode='r') for path in paths]
    for begin in range(0,u.size,131072):
        piece=sum((w*a.ravel()[begin:begin+131072] for w,a in zip(weights,iterates)),np.zeros(min(131072,u.size-begin)))
        u.ravel()[begin:begin+131072]=cp.asarray(piece)
        for a in iterates:a._mmap.madvise(mmap.MADV_DONTNEED)
    with model.budget.batch():model.residual(u,c,macro,T,out=r);actual=model.budget.dot(r,r)**.5/norm0
    result={'weights':weights.tolist(),'snapshots':[str(p) for p in paths],'snapshot_true_relative_residuals':norms,'predicted_relative_residual':prediction/norm0,'actual_relative_residual':actual,'accepted':actual<=min(norms)*(1+1e-8),'wall_seconds':time.perf_counter()-started,'scope':'established affine residual smoothing; physical FP64 residual independently recomputed, no change to geometry, pore stiffness, tolerance or load'}
    if result['accepted']:
        checkpoint=directory/'checkpoint';target=np.lib.format.open_memmap(checkpoint/'smoothed.tmp.npy',mode='w+',dtype='float64',shape=u.shape)
        for begin in range(0,u.size,131072):
            target.ravel()[begin:begin+131072]=cp.asnumpy(u.ravel()[begin:begin+131072]);target.flush();target._mmap.madvise(mmap.MADV_DONTNEED)
        del target;(checkpoint/'smoothed.tmp.npy').replace(checkpoint/'u.npy');state=json.loads((checkpoint/'state.json').read_text())
        old=report['metrics'];metrics={**old,'algorithm':'FP64 affine minimum residual in stored Krylov iterate space','initial_relative_residual':min(norms),'relative_residual':actual,'converged':actual<=args.rtol,'residual_history':[min(norms),actual],'iterations':0,'wall_seconds':result['wall_seconds'],'smoothing':result,'gpu':model.budget.metadata()}
        state.update(generation=uuid.uuid4().hex,metrics=metrics);(checkpoint/'state.json').write_text(json.dumps(state,indent=2))
        for name in ['cg.json','r.npy','p.npy']:(checkpoint/name).unlink(missing_ok=True)
        archive_iterate(checkpoint,directory/'snapshots');report['metrics']=metrics
        if metrics['converged']:
            del r,residuals,iterates;cp.get_default_memory_pool().free_all_blocks()
            from morqex.visualization.voxel_fields import export_reference_fields
            report['fields']=export_reference_fields(model,u,c,directory/'fields',macro=macro,temperature=T)
        (directory/'result.json').write_text(json.dumps(report,indent=2))
    (directory/'smoothing_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
