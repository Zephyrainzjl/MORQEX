"""Symmetric geometric V-cycle, periodic adjoint trilinear transfers.

Only the preconditioner coarsens/quantizes phase fractions and adds a positive
coarse regularization. Fine C, true voids, source and FP64 equilibrium do not
change. Coarse operators are rediscretizations, not exact Galerkin matrices.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from .poisson import FastPoisson
from .diagonal import Diagonal
from .coarse_elasticity import coarse_matrix

def coarsen(a):
    out=a
    counts=np.ones(a.shape,dtype='float32')
    for axis in range(3):out=np.add.reduceat(out,np.arange(0,out.shape[axis],2),axis=axis);counts=np.add.reduceat(counts,np.arange(0,counts.shape[axis],2),axis=axis)
    return out/counts

class GeometricMultigrid(FastPoisson):
    def __init__(self,model):
        from .voxel import NativeStencil
        if len(model.C_host)!=3:raise ValueError('This geometric preconditioner adapter uses three phase tensors; choose another backend for more materials')
        cp=model.xp;self.cp=cp;self.model=model;self.levels=[model];self.factors=[1.];self.diagonal=[Diagonal(model).scaling];self.b=[];self.x=[];self.work=[]
        names=['prolong32','prolong64','restrict32','direction_mixed'];self.module=cp.RawModule(code=Path(__file__).with_name('poisson.cu').read_text(),options=('--std=c++14',),name_expressions=names);self.functions={name:self.module.get_function(name) for name in names}
        fraction1=(model.phase_host==1).astype('float32');fraction2=(model.phase_host==2).astype('float32');Q=16
        pairs=[(a,b) for a in range(Q+1) for b in range(Q+1-a)];lookup=np.full((Q+1,Q+1),-1,dtype='int16')
        tensors=[];regularizer=1e-4*max(float(np.linalg.eigvalsh(model.C_host).max()),1.)*np.eye(6)
        for k,(a,b) in enumerate(pairs):
            lookup[a,b]=k;tensors.append(((Q-a-b)*model.C_host[0]+a*model.C_host[1]+b*model.C_host[2])/Q+regularizer)
        tensors=np.asarray(tensors)
        while max(self.levels[-1].shape)>4:
            fraction1=coarsen(fraction1);fraction2=coarsen(fraction2);qa=np.clip(np.rint(Q*fraction1),0,Q).astype(int);qb=np.minimum(Q-qa,np.clip(np.rint(Q*fraction2),0,Q).astype(int));phase=lookup[qa,qb].astype('uint8')
            spacing=tuple(np.array(model.spacing)*np.array(model.shape)/np.array(phase.shape));level=NativeStencil(phase,tensors,spacing=spacing,backend='cuda',budget=model.budget)
            factor=float(level.scale*(model.n/level.n)*(min(model.spacing)/min(spacing))**2)
            self.levels.append(level);self.factors.append(factor);self.diagonal.append((Diagonal(level).scaling/np.float32(factor)).astype('float32'))
        self.fast=self.levels[-1].shape
        for k,level in enumerate(self.levels):
            shape=(3,*level.shape);self.b.append(cp.empty(shape,dtype='float32') if k else None);self.x.append(cp.empty(shape,dtype='float32') if k else None);self.work.append(cp.empty(shape,dtype='float32'))
        last=self.levels[-1];K,_=coarse_matrix(last.phase_host,last.C_host,last.spacing,block=1);K*=self.factors[-1];self.free=np.ones(K.shape[0],bool);self.free[:3]=False
        self.factor=cho_factor(K[self.free][:,self.free].toarray());self.host=np.zeros(K.shape[0])
        self.update=cp.ElementwiseKernel('T rhs,float32 inverse,float32 applied,float64 omega','float32 value','value+=omega*inverse*(rhs-applied)','morqex_gmg_chebyshev')
        self.residual=cp.ElementwiseKernel('T rhs','float32 applied','applied=rhs-applied','morqex_gmg_residual')
        # Twelve local displacement entries bound the normalized stiffness
        # spectrum by 12 for PSD C. Fixed Chebyshev pre/post polynomials are
        # symmetric and positive as a preconditioner on the physical range.
        self.weights=1/(6.075+5.925*np.cos(np.pi*(np.arange(2)+.5)/2))
        cp.get_default_memory_pool().free_all_blocks()
    def applied(self,k,x):
        self.levels[k].apply(x,self.work[k]);self.work[k]*=self.factors[k]
    def cycle(self,k,rhs,out):
        cp=self.cp
        if k==len(self.levels)-1:
            vector=cp.asnumpy(rhs).reshape(3,-1).T.ravel();self.host.fill(0.);self.host[self.free]=cho_solve(self.factor,vector[self.free]);out[:]=cp.asarray(self.host.reshape(-1,3).T.reshape(out.shape));return
        out.fill(0.)
        for weight in self.weights:self.applied(k,out);self.update(rhs,self.diagonal[k],self.work[k],float(weight),out)
        self.applied(k,out);self.residual(rhs,self.work[k])
        for component in range(3):self.b[k+1][component]=self.transfer(self.work[k][component],self.levels[k+1].shape,transpose=True)
        self.cycle(k+1,self.b[k+1],self.x[k+1])
        for component in range(3):out[component]+=self.transfer(self.x[k+1][component],self.levels[k].shape)
        for weight in reversed(self.weights):self.applied(k,out);self.update(rhs,self.diagonal[k],self.work[k],float(weight),out)
    def apply(self,r,out):self.cycle(0,r,out);return out
