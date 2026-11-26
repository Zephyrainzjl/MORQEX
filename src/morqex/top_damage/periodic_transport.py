"""Periodic scalar voxel transport with a full symmetric effective tensor.

Three correctors share the graph and AMG preconditioner. Zero-conductivity
phases and floating components are treated exactly; no weak background phase.
Coordinates/gradient columns use XYZ while storage is ZYX. This is periodic
finite-volume numerical homogenization, not an intrinsic particle diffusivity.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg
from .contracts import _owned


@dataclass(frozen=True)
class PeriodicTransportResult:
    tensor: np.ndarray
    correctors: np.ndarray
    residuals: np.ndarray
    flux_energy_defect: float
    components: int
    iterations: tuple
    fingerprint: str
    scope: str = 'Periodic scalar local constitutive law; no anisotropic local tensor or crack aperture inferred.'

    def __post_init__(self):
        for name in ('tensor','correctors','residuals'):
            object.__setattr__(self,name,_owned(getattr(self,name)))


def periodic_transport(field,spacing_zyx_m,*,tolerance=1e-10,max_iterations=3000):
    k=np.asarray(field,float);h=np.asarray(spacing_zyx_m,float)
    if k.ndim!=3 or min(k.shape)<2 or not np.isfinite(k).all() or np.any(k<0):
        raise ValueError('Finite nonnegative scalar conductivity on a 3D lattice required')
    if h.shape!=(3,) or not np.isfinite(h).all() or np.any(h<=0):raise ValueError('Positive ZYX spacing required')
    if not np.isfinite(tolerance) or not 0<tolerance<1e-3 or type(max_iterations) is not int or max_iterations<1:
        raise ValueError('Valid transport tolerances required')
    digest=hashlib.sha256(b'morqex-periodic-transport-v1'+np.asarray(k.shape,np.int64).tobytes()+k.tobytes()+h.tobytes()).hexdigest()
    ids=np.flatnonzero(k>0);n=len(ids);index=np.full(k.shape,-1,np.int64);index.ravel()[ids]=np.arange(n)
    volume=float(np.prod(np.asarray(k.shape)*h));voxel_volume=float(np.prod(h))
    if not n:return PeriodicTransportResult(np.zeros((3,3)),np.zeros((*k.shape,3)),np.zeros(3),0.,0,(0,0,0),digest)
    ei=[];ej=[];eg=[];ed=[]
    for axis in range(3):
        neighbour=np.roll(index,-1,axis=axis);other=np.roll(k,-1,axis=axis)
        valid=(index>=0)&(neighbour>=0)
        kl=k[valid];kr=other[valid];small=np.minimum(kl,kr);large=np.maximum(kl,kr)
        ei.append(index[valid]);ej.append(neighbour[valid])
        eg.append(2*small/(1+small/large)*voxel_volume/h[axis]**2)
        delta=np.zeros((int(valid.sum()),3));delta[:,2-axis]=h[axis];ed.append(delta)
    i=np.concatenate(ei);j=np.concatenate(ej);g=np.concatenate(eg);delta=np.concatenate(ed)
    # For two voxels along an axis, the interior and periodic faces are
    # distinct physical edges with opposite incidence but the same +h shift.
    B=coo_matrix((np.r_[-np.ones(len(i)),np.ones(len(i))],
                   (np.r_[np.arange(len(i)),np.arange(len(i))],np.r_[i,j])),shape=(len(i),n)).tocsr()
    L=(B.T@diags(g)@B).tocsr();rhs=-np.asarray(B.T@(g[:,None]*delta))
    graph=coo_matrix((np.ones(len(i)),(i,j)),shape=(n,n)).tocsr()
    count,labels=connected_components(graph,directed=False)
    _,first=np.unique(labels,return_index=True);free=np.ones(n,bool);free[first]=False
    chi=np.zeros((n,3));iterations=[];residuals=[]
    A=L[free][:,free].tocsr()
    pre=None;scale=1.
    if A.shape[0]:
        import pyamg
        scale=float(A.diagonal().max());normalized=A/scale
        ml=pyamg.smoothed_aggregation_solver(normalized,symmetry='symmetric',max_coarse=48,
            presmoother=('gauss_seidel',{'sweep':'symmetric'}),postsmoother=('gauss_seidel',{'sweep':'symmetric'}))
        pre=ml.aspreconditioner()
    for axis in range(3):
        b=rhs[free,axis];counter=[0]
        def callback(_):counter[0]+=1
        if np.linalg.norm(b)>0:
            chi[free,axis],info=cg(normalized,b/scale,M=pre,rtol=tolerance,atol=0.,maxiter=max_iterations,callback=callback)
            if info:raise RuntimeError(f'Periodic corrector {axis} failed to converge: {info}')
        # Include gauge-node equations: every floating component must balance.
        r=L@chi[:,axis]-rhs[:,axis]
        physical_scale=max(np.linalg.norm(rhs[:,axis]),np.linalg.norm(B.T@(g*delta[:,axis])),np.finfo(float).tiny)
        residual=float(np.linalg.norm(r)/physical_scale) if np.linalg.norm(rhs[:,axis]) else float(np.linalg.norm(r))
        if residual>max(10*tolerance,1e-9):raise RuntimeError('Periodic conservation residual failed')
        residuals.append(residual);iterations.append(counter[0])
    jump=delta+B@chi
    tensor=jump.T@(g[:,None]*jump)/volume
    flux=delta.T@(g[:,None]*jump)/volume
    # Normalize by the Voigt constitutive scale, including disconnected zero-response cases.
    norm=max(float(np.linalg.norm(tensor)),float(np.mean(k)),np.finfo(float).tiny)
    defect=float(np.linalg.norm(flux-tensor)/norm)
    if defect>1e-7:raise RuntimeError('Periodic flux/energy identity failed')
    if np.min(np.linalg.eigvalsh(tensor)) < -1e-12*norm:raise RuntimeError('Effective tensor lost positive semidefiniteness')
    correctors=np.zeros((k.size,3));correctors[ids]=chi
    return PeriodicTransportResult(tensor,correctors.reshape((*k.shape,3)),np.asarray(residuals),defect,int(count),tuple(iterations),digest)
