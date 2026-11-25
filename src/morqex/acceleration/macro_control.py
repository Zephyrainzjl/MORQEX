"""Exact six-component mean-stress control via a variational Schur complement."""
import numpy as np
import hashlib
from .voxel import NativeStencil

class StressControlledStencil(NativeStencil):
    def __init__(self,*args,stress_Pa=(0.,)*6,**kwargs):
        super().__init__(*args,**kwargs);fractions=np.bincount(self.phase_host.ravel(),minlength=len(self.C_host))/self.n
        average=np.einsum('p,pij->ij',fractions,self.C_host)*self.scale
        if np.linalg.eigvalsh(average).min()<=0:raise ValueError('Mean stress control needs a nondegenerate average elasticity')
        self.inverse_average=self.xp.asarray(np.linalg.inv(average));self.target=self.xp.asarray(stress_Pa,dtype='float64')
        self.boundary_fingerprint=hashlib.sha256(b'mean-stress'+np.asarray(stress_Pa,dtype=float).tobytes()).hexdigest()
        if self.target.shape!=(6,):raise ValueError('Six prescribed Kelvin mean stresses required')
    def macro_strain(self,u,c,T=0.):return self.inverse_average@(self.target-super().mean_stress(u,c,temperature=T))
    def apply(self,u,out=None):
        u=self._displacement(u)
        if out is None:out=self.xp.empty_like(u)
        self._check_output(out,u.shape,u.dtype)
        E=-(self.inverse_average@super().mean_stress(u))
        self._call('elastic_macro',[u,out,self.phase,self.C,E,*self.dims,self.ratios]);return out
    def load(self,concentration,macro=(0.,)*6,temperature=0.,out=None):
        if np.any(macro):raise ValueError('Use stress_Pa in the constructor for stress-controlled problems')
        E=self.inverse_average@(self.target-super().mean_stress(concentration=concentration,temperature=temperature))
        return super().load(concentration,E,temperature,out)
    def residual(self,u,concentration,macro=(0.,)*6,temperature=0.,out=None):
        E=self.macro_strain(u,concentration,temperature)
        return super().residual(u,concentration,E,temperature,out)
    def stress(self,u,concentration,macro=(0.,)*6,temperature=0.,out=None):
        return super().stress(u,concentration,self.macro_strain(u,concentration,temperature),temperature,out)
