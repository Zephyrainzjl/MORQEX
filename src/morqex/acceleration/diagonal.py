"""Exact assembled diagonal of the compatible stencil, without assembling K."""
import numpy as np

class Diagonal:
    def __init__(self,model):
        cp=model.xp;self.model=model;self.scaling=cp.zeros((3,*model.shape),dtype='float32');self.fast=()
        index=np.array([[0,5,4],[5,1,3],[4,3,2]]);ratios=np.asarray(model.spacing).min()/np.asarray(model.spacing)[::-1]
        for component in range(3):
            b=np.zeros(6);b[component]=-ratios[component]
            for axis in range(3):
                if axis!=component:b[index[component,axis]]=-ratios[axis]/np.sqrt(2)
            value=np.einsum('i,pij,j->p',b,model.C_host,b).astype(np.float32);self.scaling[component]=cp.asarray(value)[model.phase]
            for axis in range(3):
                k=index[component,axis];value=(model.C_host[:,k,k]*ratios[axis]**2*(1. if axis==component else .5)).astype(np.float32)
                self.scaling[component]+=cp.asarray(value)[cp.roll(model.phase,1,axis=2-axis)]
        cp.ElementwiseKernel('float32 d','float32 out','out=d>0?1.f/d:0','morqex_elastic_inverse_diagonal')(self.scaling,self.scaling)
        self.multiply=cp.ElementwiseKernel('float64 r,float32 d','float32 z','z=r*d','morqex_elastic_jacobi')
    def apply(self,r,out):self.multiply(r,self.scaling,out);return out
    def direction(self,p,z,beta):
        self.model.xp.ElementwiseKernel('float32 z,float64 beta','float64 p','p=(double)z+beta*p','morqex_elastic_jacobi_direction')(z,np.float64(beta),p)
