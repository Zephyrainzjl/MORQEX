"""Analytic anisotropic periodic manufactured solution, independent of CG."""
import time
import numpy as np
from .voxel import NativeStencil

def manufactured_periodic(n):
    started=time.perf_counter();rng=np.random.default_rng(66);A=rng.normal(size=(6,6));C=A.T@A+np.eye(6)
    beta=np.array([.002,.001,-.001,.0004,-.0003,.0008]);T=np.zeros((6,3));T[0,0]=1.;T[5,1]=1/np.sqrt(2);T[4,2]=1/np.sqrt(2)
    coefficients=np.linalg.solve(T.T@C@T,T.T@C@beta);h=1/n;phase=np.ones((n,n,n),np.uint8)
    model=NativeStencil(phase,np.stack([np.zeros((6,6)),C]),np.stack([np.zeros(6),beta]),spacing=(h,)*3)
    c=np.broadcast_to(np.sin(2*np.pi*(np.arange(n)+.5)/n)[None,None,:],phase.shape).copy();u,metrics=model.solve(c,rtol=1e-11,maxiter=100)
    physical=model.displacement_m(u);physical-=physical.mean(axis=(1,2,3),keepdims=True)
    exact=-coefficients[:,None,None,None]*np.cos(2*np.pi*np.arange(n)/n)[None,None,None,:]/(2*np.pi)
    displacement_error=float(np.linalg.norm(physical-exact)/np.linalg.norm(np.broadcast_to(exact,physical.shape)))
    stress=model.stress(u,c);exact_stress=(C@(T@coefficients-beta))[:,None,None,None]*c
    stress_error=float(np.linalg.norm(stress-exact_stress)/np.linalg.norm(exact_stress));energy=float(-.5*np.mean(np.einsum('i,izyx->zyx',beta,stress)*c))
    exact_energy=float(.25*(T@coefficients-beta)@C@(T@coefficients-beta))
    return {'n':n,'h_m':h,'displacement_relative_L2':displacement_error,'stress_relative_L2':stress_error,'relative_energy_error':abs(energy-exact_energy)/exact_energy,
        'solver_relative_residual':metrics['relative_residual'],'wall_seconds':time.perf_counter()-started,
        'scope':'smooth homogeneous anisotropic periodic manufactured solution; staggered exact stress and O(h^2) displacement, not a CT/interface convergence result'}
