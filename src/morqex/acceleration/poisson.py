"""Auxiliary smooth-size periodic grid, exact adjoint transfers and FFT inverse.

M=P^T L_fast^dagger P is positive on the mean-free subspace. Padding alone
would change periodic adjacency; periodic interpolation and its transpose avoid
that shortcut. The physical elasticity operator is unchanged.
"""
from pathlib import Path
import numpy as np
from scipy.fft import next_fast_len

class FastPoisson:
    def __init__(self,model):
        self.model=model;self.cp=model.xp;cp=self.cp;self.shape=model.shape
        self.fast=tuple(next_fast_len(n,real=True) for n in self.shape)
        names=['prolong32','prolong64','restrict32','direction_mixed']
        self.module=cp.RawModule(code=Path(__file__).with_suffix('.cu').read_text(),options=('--std=c++14',),name_expressions=names)
        self.functions={name:self.module.get_function(name) for name in names}
        self.scaling=cp.zeros((3,*self.shape),dtype='float32')
        indices=np.array([[0,5,4],[5,1,3],[4,3,2]])
        ratios=np.asarray(model.spacing).min()/np.asarray(model.spacing)[::-1]
        for component in range(3):
            b=np.zeros(6);b[component]=-ratios[component]
            for a in range(3):
                if a!=component:b[indices[component,a]]=-ratios[a]/np.sqrt(2)
            values=np.einsum('i,pij,j->p',b,model.C_host,b).astype(np.float32)
            self.scaling[component]=cp.asarray(values)[model.phase]
            for a in range(3):
                idx=indices[component,a];values=(model.C_host[:,idx,idx]*ratios[a]**2*(1. if component==a else .5)).astype(np.float32)
                self.scaling[component]+=cp.asarray(values)[cp.roll(model.phase,1,axis=2-a)]
        cp.ElementwiseKernel('float32 d','float32 out','out=d>0?rsqrtf(d):0','morqex_diagonal_scale')(self.scaling,self.scaling)
        cp.get_default_memory_pool().free_all_blocks()
        self.inverse=cp.zeros((self.fast[0],self.fast[1],self.fast[2]//2+1),dtype='float32')
        for a,n in enumerate(self.fast):
            k=cp.arange(self.inverse.shape[a]);term=4*cp.sin(np.pi*k/n)**2*(n/model.shape[a])**2*(min(model.spacing)/model.spacing[a])**2
            view=[1,1,1];view[a]=len(k);self.inverse+=term.reshape(view)
        # Diagonal scaling changes the constant mode. Keeping the auxiliary
        # zero-frequency inverse positive prevents a spurious null direction;
        # this is a preconditioner term, never a physical pore stiffness.
        self.inverse[0,0,0]=1.;cp.reciprocal(self.inverse,out=self.inverse)
    def transfer(self,array,target,transpose=False):
        cp=self.cp
        for axis in range(3) if not transpose else reversed(range(3)):
            if array.shape[axis]==target[axis]:continue
            shape=list(array.shape);shape[axis]=target[axis];out=cp.empty(shape,dtype='float32');count=int(np.prod(shape))
            name='restrict32' if transpose else 'prolong64' if array.dtype==np.float64 else 'prolong32'
            args=(array,out,*map(np.int32,reversed(array.shape)),np.int32(target[axis]),np.int32(2-axis),np.int64(count))
            self.functions[name](((count+127)//128,),(128,),args);array=out
        return array.astype('float32',copy=False)
    def apply(self,r,out):
        cp=self.cp
        for k in range(3):
            scaled=cp.empty(self.shape,dtype='float32');cp.multiply(r[k],self.scaling[k],out=scaled,casting='unsafe')
            channel=self.transfer(scaled,self.fast);del scaled
            transformed=cp.fft.rfftn(channel);transformed*=self.inverse;del channel
            value=cp.fft.irfftn(transformed,s=self.fast,axes=(0,1,2));del transformed
            value=self.transfer(value,self.shape,transpose=True);value*=self.scaling[k];out[k]=value
        return out
    def direction(self,p,z,beta):
        self.functions['direction_mixed'](((p.size+127)//128,),(128,),(p,z,np.float64(beta),np.int64(p.size)))
