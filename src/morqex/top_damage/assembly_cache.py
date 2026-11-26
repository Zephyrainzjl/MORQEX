"""Exact fixed-mesh P1 assembly reuse, independently checked against MFEM.

Cache geometry and the COO-to-CSR reduction map, not a frozen damaged operator.
Every changing reaction/diffusion coefficient is reassembled. This is standard
affine FE assembly reuse, not a new homogenization or reduced-order method.
"""
import numpy as np
from scipy.sparse import csr_matrix
from .backend import require_local_budget


class CachedP1ScalarBackend:
    def __init__(self, mesh, reference):
        require_local_budget()
        self.mesh=mesh;self.reference=reference
        self.manifest=reference.manifest | {'assembly':'exact-p1-geometry-cache'}
        n=len(mesh.points);e=len(mesh.cells)
        rows=np.broadcast_to(mesh.cells[:,:,None],(e,4,4)).reshape(-1)
        cols=np.broadcast_to(mesh.cells[:,None,:],(e,4,4)).reshape(-1)
        keys,self._inverse=np.unique(rows.astype(np.int64)*n+cols,return_inverse=True)
        self._indices=keys%n
        self._indptr=np.r_[0,np.cumsum(np.bincount(keys//n,minlength=n))]
        self._mass=mesh.volumes[:,None,None]*(np.ones((4,4))+np.eye(4))/20
        self.cache_bytes=sum(a.nbytes for a in (self._inverse,self._indices,self._indptr,self._mass))
        for array in (self._inverse,self._indices,self._indptr,self._mass):array.flags.writeable=False
        # Independent integration gate before activating acceleration.
        x=np.linspace(.5,1.5,e)
        tensor=x[:,None,None]*np.array([[2.,.3,.1],[.3,1.,.2],[.1,.2,3.]])
        actual,b=self.assemble(mesh,x,tensor,2*x)
        exact,rhs=reference.assemble(mesh,x,tensor,2*x)
        diff=actual-exact
        error=np.max(np.abs(diff.data),initial=0)/max(np.max(np.abs(exact.data)),np.finfo(float).tiny)
        rhs_error=np.max(np.abs(b-rhs))/max(np.max(np.abs(rhs)),np.finfo(float).tiny)
        if max(error,rhs_error)>1e-12:raise RuntimeError('Cached assembly failed independent MFEM gate')
        self.verification_error=float(max(error,rhs_error))

    def assemble(self, mesh, reaction, diffusion, source):
        require_local_budget()
        if mesh is not self.mesh:raise ValueError('Assembly cache belongs to a different mesh object')
        e=len(mesh.cells);n=len(mesh.points)
        a=np.asarray(np.broadcast_to(reaction,(e,)),float)
        D=np.asarray(np.broadcast_to(diffusion,(e,3,3)),float)
        f=np.asarray(np.broadcast_to(source,(e,)),float)
        if not all(np.isfinite(v).all() for v in (a,D,f)) or np.any(a<0):raise ValueError('Invalid finite-element coefficients')
        scale=max(np.max(np.abs(D)),np.finfo(float).tiny)
        if np.max(np.abs(D-D.swapaxes(1,2)))>1e-13*scale or np.min(np.linalg.eigvalsh(D)) < -1e-13*scale:raise ValueError('Symmetric positive semidefinite diffusion required')
        local=a[:,None,None]*self._mass+mesh.volumes[:,None,None]*np.einsum('eai,eij,ebj->eab',mesh.gradients,D,mesh.gradients,optimize=True)
        values=np.bincount(self._inverse,weights=local.ravel(),minlength=len(self._indices))
        A=csr_matrix((values,self._indices,self._indptr),shape=(n,n))
        b=np.bincount(mesh.cells.ravel(),weights=np.repeat(mesh.volumes*f/4,4),minlength=n)
        return A,b
