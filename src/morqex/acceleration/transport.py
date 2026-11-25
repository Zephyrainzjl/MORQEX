"""GPU-parallel full-voxel periodic transport cell problems, harmonic faces.

This periodic homogenization is distinct from the legacy insulated/Dirichlet
cell problem. Its boundary condition is recorded rather than silently compared.
"""
from pathlib import Path
import time
import numpy as np
from scipy.fft import next_fast_len
from .poisson import FastPoisson

class ScalarPoisson(FastPoisson):
    def __init__(self,model,scaling):
        cp=model.xp;self.cp=cp;self.model=model;self.shape=model.shape;self.fast=tuple(next_fast_len(n,real=True) for n in self.shape);self.scaling=scaling
        names=['prolong32','prolong64','restrict32'];self.module=cp.RawModule(code=Path(__file__).with_name('poisson.cu').read_text(),options=('--std=c++14',),name_expressions=names)
        self.functions={name:self.module.get_function(name) for name in names};self.inverse=cp.zeros((self.fast[0],self.fast[1],self.fast[2]//2+1),dtype='float32')
        for a,n in enumerate(self.fast):
            k=cp.arange(self.inverse.shape[a]);v=4*cp.sin(np.pi*k/n)**2*(n/model.shape[a])**2*(min(model.spacing)/model.spacing[a])**2
            shape=[1,1,1];shape[a]=len(k);self.inverse+=v.reshape(shape)
        self.inverse[0,0,0]=1.;cp.reciprocal(self.inverse,out=self.inverse)
    def apply(self,r,out):
        cp=self.cp;scaled=cp.empty(self.shape,dtype='float32');cp.multiply(r,self.scaling,out=scaled,casting='unsafe')
        channel=self.transfer(scaled,self.fast);del scaled
        transformed=cp.fft.rfftn(channel);transformed*=self.inverse;del channel
        value=cp.fft.irfftn(transformed,s=self.fast,axes=(0,1,2));del transformed
        value=self.transfer(value,self.shape,transpose=True);value*=self.scaling;out[:]=value;return out

class PeriodicTransport:
    def __init__(self,phase,conductivity,spacing=(1.,1.,1.),*,budget=None):
        from .budget import GpuBudget
        self.budget=GpuBudget() if budget is None else budget
        import cupy as cp
        self.xp=cp;self.budget.stream.use();phase=np.asarray(phase);conductivity=np.asarray(conductivity,dtype=float)
        if phase.ndim!=3 or min(phase.shape)<2 or phase.dtype.kind not in 'ui' or conductivity.ndim!=1 or len(conductivity)>256 or phase.max()>=len(conductivity):raise ValueError('3D integer phases and one coefficient per phase required')
        if not np.isfinite(conductivity).all() or conductivity.min()<0 or len(spacing)!=3 or not np.isfinite(spacing).all() or min(spacing)<=0:raise ValueError('Nonnegative finite conductivities and positive spacing required')
        self.shape=phase.shape;self.n=phase.size;self.spacing=tuple(spacing);self.phase_host=phase;self.coefficient_host=conductivity;self.phase=cp.asarray(phase,dtype='uint8');self.c=cp.asarray(conductivity)
        self.ratio=cp.asarray(min(spacing)/np.asarray(spacing)[::-1]);self.module=cp.RawModule(code=f'#define NX {self.shape[2]}\n#define NY {self.shape[1]}\n#define NZ {self.shape[0]}\n'+Path(__file__).with_suffix('.cu').read_text(),options=('--std=c++14',),name_expressions=['matrix','forcing','advance','conjugate','flux'])
        self.functions={k:self.module.get_function(k) for k in ['matrix','forcing','advance','conjugate','flux']}
    def call(self,name,args):self.functions[name](((self.n+127)//128,),(128,),args)
    def apply(self,u,out):self.call('matrix',(u,out,self.phase,self.c,self.ratio));return out
    def solve(self,axis_xyz=0,*,rtol=1e-8,maxiter=2000,preconditioner='coarse'):
        if axis_xyz not in (0,1,2) or not 0<rtol<1 or not 1<=maxiter<=10000:raise ValueError('XYZ direction, bounded iterations and positive tolerance required')
        cp=self.xp;start=time.perf_counter();x=cp.zeros(self.shape);r=cp.empty_like(x);scaling=cp.empty(self.shape,dtype='float32')
        self.call('forcing',(r,scaling,self.phase,self.c,self.ratio,np.int32(axis_xyz)));norm=lambda a:max(0.,self.budget.dot(a,a))**.5;norm0=norm(r)
        if preconditioner not in {'jacobi','fft','coarse'}:raise ValueError('Unknown transport preconditioner')
        from .coarse_transport import CoarseTransport
        p=cp.empty_like(x);Ap=cp.empty_like(x);z=cp.empty(self.shape,dtype='float32');pre=ScalarPoisson(self,scaling) if preconditioner=='fft' else CoarseTransport(self,scaling) if preconditioner=='coarse' else None;old=None;relative=1. if norm0 else 0.;history=[]
        jacobi=cp.ElementwiseKernel('float64 r,float32 d','float32 z','z=r*d*d','morqex_scalar_jacobi')
        for iteration in range(maxiter):
            if relative<=rtol:break
            with self.budget.batch():
                if pre is None:jacobi(r,scaling,z)
                else:pre.apply(r,z)
                rho=self.budget.dot(r,z)
                if rho<=0:raise RuntimeError('Transport preconditioner lost positive pairing')
                self.call('conjugate',(p,z,np.float64(0. if old is None else rho/old),np.int64(self.n)))
                self.apply(p,Ap);curvature=self.budget.dot(p,Ap)
                if curvature<=0:raise RuntimeError('Transport cell lost CG curvature')
                self.call('advance',(x,r,p,Ap,np.float64(rho/curvature),np.int64(self.n)));old=rho;relative=norm(r)/norm0;history.append(relative)
            if iteration%100==0:print({'transport_iteration':iteration+1,'relative_residual':relative},flush=True)
            if time.perf_counter()-start>110:break
        self.call('forcing',(r,scaling,self.phase,self.c,self.ratio,np.int32(axis_xyz)));self.apply(x,Ap);r-=Ap;relative=norm(r)/max(norm0,1e-300)
        metrics={'iterations':len(history),'relative_residual':relative,'converged':relative<=rtol,'seconds':time.perf_counter()-start,'boundary':'periodic corrector, harmonic phase-face conductivity',
            'backend':f'CUDA FP64 matvec/CG, FP32 {preconditioner} preconditioner','gpu':self.budget.metadata()}
        del r,p,Ap,z,pre,scaling;cp.get_default_memory_pool().free_all_blocks()
        if not metrics['converged']:raise RuntimeError(f'Transport residual target not achieved: {relative}')
        flux=cp.empty((3,*self.shape));self.call('flux',(x,flux,self.phase,self.c,self.ratio,np.int32(axis_xyz)));means=cp.asnumpy(flux.mean(axis=(1,2,3)))
        metrics['effective_column']=means.tolist();return x,metrics
