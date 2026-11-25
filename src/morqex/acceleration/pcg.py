"""FP64 PCG with FP32 auxiliary-grid preconditioning, five-vector memory bound."""
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
from .poisson import FastPoisson

def solve_pcg(model,c,macro,T,*,rtol,maxiter,checkpoint):
    cp=model.xp;start=time.perf_counter();shape=(3,*model.shape);norm=lambda a:max(0.,model.budget.dot(a,a))**.5
    def restore_array(path,out):
        data=np.load(path,mmap_mode='r')
        if data.shape!=out.shape or data.dtype!=np.float64:raise ValueError('Invalid Krylov checkpoint array')
        for k in range(3):
            for begin in range(0,model.n,262144):out[k].ravel()[begin:begin+262144]=cp.asarray(data[k].ravel()[begin:begin+262144])
        del data
    x=cp.zeros(shape);r=model.load(c,macro,T);norm0=norm(r);history=[]
    identity=hashlib.sha256(model.phase_host.tobytes()+model.C_host.tobytes()+model.beta_host.tobytes()+model.alpha_host.tobytes()+np.asarray([*macro,T,*model.spacing]).tobytes()+Path(__file__).with_name('voxel.cu').read_bytes())
    if hasattr(model,'boundary_fingerprint'):identity.update(model.boundary_fingerprint.encode())
    previous={}
    if checkpoint is not None:
        checkpoint=Path(checkpoint);checkpoint.mkdir(parents=True,exist_ok=True);host=cp.asnumpy(c);identity.update(host.tobytes());del host
        if (checkpoint/'state.json').exists():
            previous=json.loads((checkpoint/'state.json').read_text())
            if previous['fingerprint']!=identity.hexdigest():raise ValueError('Checkpoint geometry/material/load/source mismatch')
            restore_array(checkpoint/'u.npy',x)
    model.residual(x,c,macro,T,out=r);relative=norm(r)/max(norm0,1e-300)
    from .diagonal import Diagonal
    from .coarse_elasticity import CoarseElasticity
    from .phase_coarse import PhaseCoarse
    from .geometric import GeometricMultigrid
    kind=model.preconditioner_kind
    p=cp.empty_like(x);Ap=cp.empty_like(x);z=cp.empty(shape,dtype='float32');preconditioner=Diagonal(model) if kind=='jacobi' else CoarseElasticity(model) if kind=='coarse' else PhaseCoarse(model) if kind=='phase-coarse' else GeometricMultigrid(model) if kind=='geometric' else FastPoisson(model);rho_old=None;previous_iterations=0
    algorithm=hashlib.sha256(kind.encode()+Path(__file__).read_bytes()+b''.join(Path(__file__).with_name(name).read_bytes() for name in ['diagonal.py','poisson.py','poisson.cu','coarse_elasticity.py','coarse_transport.cu','phase_coarse.py','phase_coarse.cu','geometric.py'])).hexdigest()
    if checkpoint is not None and (checkpoint/'cg.json').exists():
        saved=json.loads((checkpoint/'cg.json').read_text())
        if saved.get('algorithm')==algorithm and saved.get('generation')==previous.get('generation') and saved['rho_old'] is not None:
            # Validate the recurrence residual against a fresh physical
            # matvec before retaining conjugacy across bounded jobs.
            restore_array(checkpoint/'r.npy',Ap)
            Ap-=r;drift=norm(Ap)/max(norm(r),1e-300)
            if drift<1e-6 and saved['rho_old'] is not None:
                restore_array(checkpoint/'r.npy',r)
                restore_array(checkpoint/'p.npy',p)
                rho_old=float(saved['rho_old']);previous_iterations=int(saved['iterations_total'])
    cp.get_default_memory_pool().free_all_blocks()
    initial_relative=relative
    for iteration in range(maxiter):
        if relative<=rtol:break
        with model.budget.batch():
            preconditioner.apply(r,z);rho=model.budget.dot(r,z)
            if rho<=0 or not np.isfinite(rho):raise RuntimeError('Auxiliary-space preconditioner lost positive residual pairing')
            if rho_old is None:p[:]=z
            else:preconditioner.direction(p,z,rho/rho_old)
            model.apply(p,Ap);curvature=model.budget.dot(p,Ap)
            if curvature<=0:raise RuntimeError('Nonpositive physical CG curvature')
            model._call('update',[x,r,p,Ap,float(rho/curvature),np.int64(x.size)],count=x.size);rho_old=rho
            relative=norm(r)/max(norm0,1e-300)
            if (previous_iterations+iteration+1)%500==0 or relative<rtol:
                model.residual(x,c,macro,T,out=Ap);true_norm=norm(Ap)
                Ap-=r;drift=norm(Ap)/max(true_norm,1e-300)
                if drift>.05 or relative<rtol:
                    r+=Ap;relative=true_norm/max(norm0,1e-300);rho_old=None
            history.append(relative)
        if iteration%25==0:print({'iteration':iteration+1,'FP64_relative_residual':relative},flush=True)
        if time.perf_counter()-start>110:break
    model.residual(x,c,macro,T,out=Ap);relative=norm(Ap)/max(norm0,1e-300)
    metrics={'iterations':len(history),'initial_relative_residual':initial_relative,'relative_residual':relative,'residual_history':history,'wall_seconds':time.perf_counter()-start,
        'converged':relative<=rtol,'backend':'cuda','algorithm':f'FP64 PCG, FP32 {kind} preconditioner',
        'auxiliary_grid':list(preconditioner.fast),'iterations_total':previous_iterations+len(history),'gpu':model.budget.metadata()}
    if checkpoint is not None:
        generation=uuid.uuid4().hex
        for name,array in [('u',x)]+([] if metrics['converged'] else [('r',r),('p',p)]):
            stored=np.lib.format.open_memmap(checkpoint/(name+'.tmp.npy'),mode='w+',dtype='float64',shape=shape)
            for k in range(3):
                for begin in range(0,model.n,262144):stored[k].ravel()[begin:begin+262144]=cp.asnumpy(array[k].ravel()[begin:begin+262144])
            stored.flush();del stored;(checkpoint/(name+'.tmp.npy')).replace(checkpoint/(name+'.npy'))
        (checkpoint/'state.json').write_text(json.dumps({'fingerprint':identity.hexdigest(),'generation':generation,'metrics':metrics},indent=2))
        (checkpoint/'cg.json').write_text(json.dumps({'algorithm':algorithm,'generation':generation,'rho_old':None if metrics['converged'] else rho_old,'iterations_total':metrics['iterations_total']}))
        if metrics['converged']:
            for name in ['r.npy','p.npy']:(checkpoint/name).unlink(missing_ok=True)
    del r,p,Ap,z,preconditioner;cp.get_default_memory_pool().free_all_blocks()
    if not metrics['converged'] and checkpoint is None:raise RuntimeError(f'FP64 residual target not achieved: {relative}')
    return x,metrics
