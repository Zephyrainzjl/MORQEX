"""Independent FP64 Tet4 reference solver; fixed C factorization is reused."""
from dataclasses import dataclass
import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu
from .constraints import constraints
from .tensors import MANDEL_BASIS, validate_stiffness


class MechanicalSolveError(RuntimeError):
    pass


@dataclass
class ElasticState:
    displacement: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    elastic_energy: float
    potential_energy: float
    mean_stress: np.ndarray
    reaction: np.ndarray
    reduced_residual: float
    equilibrium_residual: float


class ElasticitySolver:
    def __init__(self, mesh, stiffness, *, boundary="kubc", pivot_tolerance=1e-13,factorize=True):
        started = time.perf_counter()
        self.mesh = mesh
        nc = len(mesh.cells)
        self.stiffness = np.broadcast_to(validate_stiffness(stiffness), (nc, 6, 6)).copy()
        self.B = np.einsum("sij,eaj->esai", MANDEL_BASIS, mesh.gradients).reshape(nc, 6, 12)
        row = np.broadcast_to((6*np.arange(nc)[:, None, None]+np.arange(6)[None, :, None]), self.B.shape)
        col = np.broadcast_to(mesh.dofs[:, None, :], self.B.shape)
        self.D = coo_matrix((self.B.ravel(), (row.ravel(), col.ravel())), shape=(6*nc, 3*len(mesh.points))).tocsr()
        self.D.eliminate_zeros()
        # Bound element COO temporaries independently of mesh size. The same
        # Tet4 bilinear form is summed; no stiffness/geometry is approximated.
        del row,col
        self.K=csr_matrix((3*len(mesh.points),)*2)
        for begin in range(0,nc,2048):
            end=min(nc,begin+2048)
            local=np.einsum('esi,est,etj,e->eij',self.B[begin:end],self.stiffness[begin:end],self.B[begin:end],mesh.volumes[begin:end],optimize=True)
            row=np.broadcast_to(mesh.dofs[begin:end,:,None],local.shape)
            col=np.broadcast_to(mesh.dofs[begin:end,None,:],local.shape)
            self.K+=coo_matrix((local.ravel(),(row.ravel(),col.ravel())),shape=self.K.shape).tocsr()
        del local,row,col
        self.constraint = constraints(mesh, boundary)
        self.P = self.constraint.projection
        self.Kr = (self.P.T @ self.K @ self.P).tocsc()
        self.scale = float(np.max(self.K.diagonal()))
        self.assembly_seconds = time.perf_counter()-started
        started = time.perf_counter()
        self.pivot_tolerance=pivot_tolerance
        if self.Kr.shape[0] and factorize:
            try:
                self.factor = splu(self.Kr/self.scale)
            except RuntimeError as exc:
                raise MechanicalSolveError("Singular constrained stiffness; inspect topology and rigid modes") from exc
            pivots = np.abs(self.factor.U.diagonal())
            if pivots.min() < pivot_tolerance * pivots.max():
                raise MechanicalSolveError("Near-singular stiffness or unresolved mechanism; do not use as reference")
        else:
            self.factor = None
        self.factorization_seconds = time.perf_counter()-started
        self.factorizations = int(self.factor is not None)
        self.linear_solves = 0
        centered = mesh.points-mesh.box.mean(axis=0)
        self.affine = np.einsum("kij,nj->nik", MANDEL_BASIS, centered).reshape(-1, 6)

    def solve_reduced(self, rhs):
        self.linear_solves += 1
        rhs = np.asarray(rhs, dtype=float)
        if not np.isfinite(rhs).all():
            raise ValueError("Nonfinite RHS")
        if not self.Kr.shape[0]:return np.zeros_like(rhs)
        if self.factor is None:
            self.factor=splu(self.Kr/self.scale);self.factorizations+=1
            pivots=np.abs(self.factor.U.diagonal())
            if pivots.min()<self.pivot_tolerance*pivots.max():raise MechanicalSolveError('Near-singular deferred stiffness')
        return self.factor.solve(rhs/self.scale)


    def eigenstrain_load(self, eigenstrain):
        eps = np.broadcast_to(np.asarray(eigenstrain, dtype=float), (len(self.mesh.cells), 6))
        if not np.isfinite(eps).all():
            raise ValueError("Finite eigenstrain required")
        stress = np.einsum("eij,ej->ei", self.stiffness, eps)
        return np.asarray(self.D.T @ (self.mesh.volumes[:, None]*stress).ravel())

    def solve(self, eigenstrain=0., *, macro_strain=None, prescribed_displacement=None, force=None, tolerance=1e-9):
        eps = np.broadcast_to(np.asarray(eigenstrain, dtype=float), (len(self.mesh.cells), 6))
        if macro_strain is not None and self.constraint.boundary == "neumann":
            raise ValueError("Neumann solver uses external force, not prescribed macro strain")
        lift = np.zeros(self.K.shape[0]) if macro_strain is None else self.affine @ np.asarray(macro_strain, dtype=float)
        if prescribed_displacement is not None:
            lift = lift + np.asarray(prescribed_displacement, dtype=float).reshape(-1)
        external = np.zeros(self.K.shape[0]) if force is None else np.asarray(force, dtype=float).reshape(-1)
        if external.shape != lift.shape or not np.isfinite(external).all() or not np.isfinite(lift).all():
            raise ValueError("Finite nodal force/displacement of the correct shape required")
        rhs = self.eigenstrain_load(eps)+external
        fluctuation = self.solve_reduced(self.P.T @ (rhs-self.K @ lift))
        u = lift+self.P @ fluctuation
        strain = np.asarray(self.D @ u).reshape(-1, 6)
        elastic = strain-eps
        stress = np.einsum("eij,ej->ei", self.stiffness, elastic)
        reaction = np.asarray(self.K @ u-rhs)
        denominator = max(np.linalg.norm(rhs), np.linalg.norm(self.K @ u), self.scale*np.linalg.norm(u)*np.finfo(float).eps, np.finfo(float).tiny)
        residual = np.linalg.norm(self.P.T @ reaction)/denominator
        physical = np.linalg.norm(self.constraint.equilibrium_projection.T @ reaction)/denominator
        if max(residual, physical) > tolerance:
            raise MechanicalSolveError(f"Equilibrium failed: reduced {residual:.3e}, including rigid gauges {physical:.3e}; check load balance")
        energy = .5*np.einsum("ei,ei,e->", elastic, stress, self.mesh.volumes)
        mean = np.einsum("e,ei->i", self.mesh.volumes, stress)/self.mesh.rve_volume
        return ElasticState(u.reshape(-1, 3), strain, stress, float(energy), float(energy-external @ u), mean,
                            reaction.reshape(-1, 3), float(residual), float(physical))


class DeferredElasticity:
    """Material-only holder until an intact mechanical solve is requested."""
    def __init__(self,mesh,stiffness,*,boundary='kubc'):
        self.mesh=mesh;self.stiffness=validate_stiffness(stiffness);self.boundary=boundary;self._solver=None
    def __getattr__(self,name):
        if self._solver is None:self._solver=ElasticitySolver(self.mesh,self.stiffness,boundary=self.boundary)
        return getattr(self._solver,name)
