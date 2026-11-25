"""Memory-bounded mixed precision with acceptance by the true FP64 residual.

Only correction Krylov vectors and FFT preconditioning use FP32. The accumulated
solution, source evaluation, final residual and material contractions are FP64.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

def solve_refined(model,c,macro,T,*,rtol,maxiter,checkpoint):
    cp=model.xp;start=time.perf_counter();shape=(3,*model.shape)
    norm=lambda a:max(0.,model.budget.dot(a,a))**.5
    x=cp.zeros(shape);r64=model.load(c,macro,T);norm0=norm(r64);history=[]
    identity=hashlib.sha256(model.phase_host.tobytes()+model.C_host.tobytes()+model.beta_host.tobytes()+model.alpha_host.tobytes()+np.asarray([*macro,T,*model.spacing]).tobytes()+Path(__file__).with_name('voxel.cu').read_bytes())
    if checkpoint is not None:
        checkpoint=Path(checkpoint);checkpoint.mkdir(parents=True,exist_ok=True);host=cp.asnumpy(c);identity.update(host.tobytes());del host
        if (checkpoint/'state.json').exists():
            previous=json.loads((checkpoint/'state.json').read_text())
            if previous['fingerprint']!=identity.hexdigest():raise ValueError('Checkpoint geometry/material/load/source mismatch')
            data=np.load(checkpoint/'u.npy',mmap_mode='r');x[:]=cp.asarray(data);del data
    model.residual(x,c,macro,T,out=r64);relative=norm(r64)/max(norm0,1e-300)
    inv=cp.zeros((model.shape[0],model.shape[1],model.shape[2]//2+1),dtype='float32')
    for a,n in enumerate(model.shape):
        k=cp.arange(inv.shape[a]);view=[1,1,1];view[a]=len(k);inv+=(4*cp.sin(np.pi*k/n)**2*float(min(model.spacing)/model.spacing[a])**2).reshape(view)
    inv[0,0,0]=1.;cp.reciprocal(inv,out=inv);inv[0,0,0]=0.
    delta=cp.empty(shape,dtype='float32');r=cp.empty_like(delta);p=cp.empty_like(delta);z=cp.empty_like(delta);Ap=cp.empty_like(delta)
    iterations=0;outer=0
    while relative>rtol and iterations<maxiter:
        delta.fill(0);r[:]=r64;rho_old=None;inner0=norm(r);outer+=1
        for _ in range(min(150,maxiter-iterations)):
            with model.budget.batch():
                for k in range(3):
                    transform=cp.fft.rfftn(r[k]);transform*=inv;z[k]=cp.fft.irfftn(transform,s=model.shape,axes=(0,1,2))
                rho=model.budget.dot(r,z)
                if rho<=0 or not np.isfinite(rho):raise RuntimeError('Invalid mixed-precision correction residual')
                if rho_old is None:p[:]=z
                else:model._call('direction',[p,z,float(rho/rho_old),np.int64(p.size)],count=p.size)
                model.apply(p,Ap);den=model.budget.dot(p,Ap)
                if den<=0:raise RuntimeError('Invalid correction curvature; no silent weak pore')
                model._call('update',[delta,r,p,Ap,float(rho/den),np.int64(p.size)],count=p.size)
                rho_old=rho;inner=norm(r)/max(inner0,1e-300);iterations+=1
            if inner<.03 or time.perf_counter()-start>115:break
        with model.budget.batch():
            cp.add(x,delta,out=x);model.residual(x,c,macro,T,out=r64);relative=norm(r64)/max(norm0,1e-300)
        history.append(relative);print({'outer':outer,'iterations':iterations,'true_FP64_relative_residual':relative},flush=True)
        if time.perf_counter()-start>115:break
    converged=relative<=rtol
    metrics={'iterations':iterations,'outer_iterations':outer,'relative_residual':relative,'residual_history':history,'wall_seconds':time.perf_counter()-start,
        'converged':converged,'backend':'cuda','algorithm':'FP64 iterative refinement with FP32 correction CG/FFT','gpu':model.budget.metadata()}
    if checkpoint is not None:
        stored=np.lib.format.open_memmap(checkpoint/'u.npy',mode='w+',dtype='float64',shape=shape)
        for k in range(3):
            for begin in range(0,model.n,262144):stored[k].ravel()[begin:begin+262144]=cp.asnumpy(x[k].ravel()[begin:begin+262144])
        stored.flush();del stored;(checkpoint/'state.json').write_text(json.dumps({'fingerprint':identity.hexdigest(),'metrics':metrics},indent=2))
    del delta,r,p,z,Ap,inv,r64;cp.get_default_memory_pool().free_all_blocks()
    if not converged and checkpoint is None:raise RuntimeError(f'FP64 residual target not achieved: {relative}')
    return x,metrics
