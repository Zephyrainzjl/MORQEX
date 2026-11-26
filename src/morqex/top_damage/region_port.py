"""Conservative finite-volume concentration ports without a fixed elastic operator.

Reuse MORQEX's voxel diffusion discretization; regions replace radial states.
The caller supplies the reactive-face mask after physical reachability analysis.
No damage threshold, wetting law or material deletion is inferred here.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from .contracts import _owned


@dataclass(frozen=True)
class RegionPort:
    voxel_ids: np.ndarray
    partition: np.ndarray
    weights: np.ndarray
    reaction_area: np.ndarray
    stiffness: object
    reference_concentration: float
    voxel_shape: tuple
    cell_weights: np.ndarray
    fingerprint: str

    @property
    def dimension(self): return len(self.weights)
    @property
    def active_fraction(self): return float(self.weights.sum())
    @property
    def area_density(self): return float(self.reaction_area.sum())

    @classmethod
    def from_diffusion(cls, diffusion, partition, reactive_faces, reference_concentration):
        """Piecewise-constant Galerkin aggregation on the SAME material cells.

        Unit local diffusivity supplies the geometric Laplacian (m^-2); the
        electrochemical model supplies positive mobility. Every region must be
        face-connected. Different diffusion components cannot share a state.
        """
        labels=np.asarray(partition);mask=np.asarray(reactive_faces)
        if not np.all(diffusion.diffusivity_XYZ==1.):
            raise ValueError('Unit geometric diffusivity required; mobility is applied by the electrochemical model')
        n=len(diffusion.weights)
        if labels.shape!=(n,) or labels.dtype.kind not in 'iu' or not np.array_equal(np.unique(labels),np.arange(labels.max()+1)):
            raise ValueError('Contiguous nonempty concentration regions required')
        if mask.shape!=diffusion.reactive.shape or mask.dtype.kind!='b' or np.any(mask & ~diffusion.reactive):
            raise ValueError('Explicit subset of physically declared reactive faces required')
        if not np.isfinite(reference_concentration) or reference_concentration<=0:raise ValueError('Positive reference concentration required')
        m=int(labels.max()+1);P=sparse.csr_matrix((np.ones(n),(np.arange(n),labels)),shape=(n,m))
        for region in range(m):
            ids=np.flatnonzero(labels==region)
            if len(np.unique(diffusion.components[ids]))!=1 or connected_components(diffusion.stiffness[ids][:,ids],directed=False)[0]!=1:
                raise ValueError('Region merges disconnected material inventories')
        w=np.asarray(P.T@diffusion.weights).ravel()
        cell_area=np.bincount(diffusion.face_cells,weights=diffusion.face_areas*mask,minlength=n)/diffusion.rve_volume
        A=np.asarray(P.T@cell_area).ravel();K=(P.T@diffusion.stiffness@P).tocsr()
        if A.sum()<=0:raise ValueError('At least one accessible reaction interface required')
        digest=hashlib.sha256(b'morqex-region-port-v1')
        for arr in (diffusion.voxel_ids,labels,w,A,K.indptr,K.indices,K.data,np.asarray(diffusion.volume.data.shape),np.array([reference_concentration])):
            digest.update(np.ascontiguousarray(arr).tobytes())
        K.data.flags.writeable=False;K.indices.flags.writeable=False;K.indptr.flags.writeable=False
        return cls(*[_owned(a) for a in (diffusion.voxel_ids,labels,w,A)],K,float(reference_concentration),
                   tuple(diffusion.volume.data.shape),_owned(diffusion.weights),digest.hexdigest())

    def lift(self,concentration):
        c=np.asarray(concentration,float)
        if c.shape[-1]!=self.dimension or not np.isfinite(c).all():raise ValueError('Finite region concentrations required')
        return c[...,self.partition]

    def restrict(self,values):
        a=np.asarray(values,float)
        if a.shape[-1]!=len(self.voxel_ids) or not np.isfinite(a).all():raise ValueError('Finite material-cell field required')
        P=sparse.csr_matrix((self.cell_weights,(self.partition,np.arange(len(self.partition)))),shape=(self.dimension,len(self.partition)))
        return np.asarray(a@P.T)/self.weights

    def project_mechanical_potential(self,values):
        """Lumped variational projection, not pointwise equality of chemical potentials."""
        return self.restrict(values)

    def field(self,concentration,*,inactive=np.nan):
        c=self.lift(concentration)
        a=np.full(c.shape[:-1]+(int(np.prod(self.voxel_shape)),),inactive,float)
        a[...,self.voxel_ids]=c
        return a.reshape(c.shape[:-1]+self.voxel_shape)
