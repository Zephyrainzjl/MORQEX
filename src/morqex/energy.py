"""Fixed-geometry, fixed-C energy/localization response and Galerkin compression.

Linear offline superposition and static condensation are established methods.
This module makes their input space and discrete error identities explicit.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import numpy as np


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class EnergyResponse:
    H: np.ndarray
    h: np.ndarray
    constant: float
    displacement_operator: np.ndarray
    displacement_offset: np.ndarray
    strain_operator: np.ndarray
    strain_offset: np.ndarray
    stress_operator: np.ndarray
    stress_offset: np.ndarray
    volumes: np.ndarray
    phases: np.ndarray
    rve_volume: float
    metadata: dict = field(default_factory=dict)

    @property
    def dimension(self): return len(self.h)

    def _q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape[-1:] != (self.dimension,) or not np.isfinite(q).all():
            raise ValueError("q must be finite [..., six Mandel strains plus concentration coefficients]")
        return q

    def energy_density(self, q):
        q = self._q(q)
        return .5*np.einsum("...i,ij,...j->...", q, self.H, q)+q @ self.h+self.constant

    def gradient(self, q): return self._q(q) @ self.H.T+self.h

    def macro_stress(self, q): return self.gradient(q)[..., :6]

    def generalized_chemical_force(self, q): return self.gradient(q)[..., 6:]

    def projected_chemical_potential(self, q, concentration_mass):
        mass = np.asarray(concentration_mass, dtype=float)
        if mass.shape != (self.dimension-6,)*2:
            raise ValueError("Concentration mass matrix has incorrect shape")
        force = self.generalized_chemical_force(q)
        if not force.shape[-1]: return force.copy()
        return np.linalg.solve(mass, force.reshape(-1, force.shape[-1]).T).T.reshape(force.shape)

    def macro_strain_at_stress(self, coefficients, target_stress):
        a = np.asarray(coefficients, dtype=float)
        target = np.asarray(target_stress, dtype=float)
        c = self.H[:6, :6]
        if np.linalg.eigvalsh(c).min() <= 1e-12*np.linalg.norm(c, 2):
            raise ValueError("Effective stiffness is singular/ill-conditioned; macro stress cannot determine a unique strain")
        return np.linalg.solve(c, (target-self.h[:6]-a @ self.H[:6, 6:].T).T).T

    def stress(self, q, selection=None):
        if np.isscalar(selection) and selection is not None: selection = [selection]
        op = self.stress_operator if selection is None else self.stress_operator[selection]
        offset = self.stress_offset if selection is None else self.stress_offset[selection]
        return np.einsum("nid,...d->...ni", op, self._q(q))+offset

    def strain(self, q, selection=None):
        if np.isscalar(selection) and selection is not None: selection = [selection]
        op = self.strain_operator if selection is None else self.strain_operator[selection]
        offset = self.strain_offset if selection is None else self.strain_offset[selection]
        return np.einsum("nid,...d->...ni", op, self._q(q))+offset

    def displacement(self, q, selection=None):
        if np.isscalar(selection) and selection is not None: selection = [selection]
        op = self.displacement_operator if selection is None else self.displacement_operator[selection]
        offset = self.displacement_offset if selection is None else self.displacement_offset[selection]
        return np.einsum("nid,...d->...ni", op, self._q(q))+offset

    def save(self, workspace, name):
        path = workspace.path("operators", name)
        path.mkdir(parents=True, exist_ok=True)
        fields = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "metadata"}
        destination = path / "response.npz"
        np.savez_compressed(destination, **fields)
        (path / "manifest.json").write_text(json.dumps({"schema": "morqex-linear-response-1", "metadata": self.metadata,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "units": {"W": "J/m^3", "stress": "Pa", "a": "mol/m^3"}}, indent=2))
        return path

    @classmethod
    def load(cls, path, *, expected_fingerprint=None):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["schema"] != "morqex-linear-response-1":
            raise ValueError("Unsupported operator schema")
        if hashlib.sha256((path / "response.npz").read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Operator checksum mismatch")
        if expected_fingerprint is not None and manifest["metadata"]["fingerprint"] != expected_fingerprint:
            raise ValueError("Operator does not match requested geometry/material/basis")
        with np.load(path / "response.npz", allow_pickle=False) as arrays:
            values = {k: arrays[k] for k in arrays.files}
        return cls(**values, metadata=manifest["metadata"])


class ResponseBuilder:
    def __init__(self, solver, eigenstrain_modes, *, reference_eigenstrain=0., reference_displacement=None):
        if solver.constraint.boundary not in {"kubc", "periodic"}:
            raise ValueError("Strain-controlled homogenization needs KUBC or periodic fluctuations")
        self.solver = solver
        nc = len(solver.mesh.cells)
        modes = np.array(eigenstrain_modes, dtype=float, copy=True)
        if modes.ndim != 3 or modes.shape[:2] != (nc, 6) or not np.isfinite(modes).all():
            raise ValueError("Chemical eigenstrain modes need shape (elements,6,m)")
        self.modes = modes
        self.inherent0 = np.broadcast_to(np.asarray(reference_eigenstrain, dtype=float), (nc, 6)).copy()
        self.lift0 = np.zeros(solver.K.shape[0]) if reference_displacement is None else np.asarray(reference_displacement, dtype=float).reshape(-1)
        self.affine = np.column_stack([solver.affine, np.zeros((solver.K.shape[0], modes.shape[2]))])
        self.eigen = np.concatenate([np.zeros((nc, 6, 6)), modes], axis=-1)
        f = np.asarray(solver.D @ self.affine).reshape(nc, 6, -1)-self.eigen
        weighted = np.einsum("eij,ejd,e->eid", solver.stiffness, f, solver.mesh.volumes)
        self.rhs = -np.asarray(solver.P.T @ (solver.D.T @ weighted.reshape(6*nc, -1)))
        self.offset_rhs = solver.P.T @ (solver.eigenstrain_load(self.inherent0)-solver.K @ self.lift0)

    def exact(self):
        joint = np.column_stack([self.rhs, self.offset_rhs])
        solution = self.solver.solve_reduced(joint)
        return self._response(solution[:, :-1], solution[:, -1], "exact-linear-response")

    def galerkin(self, basis):
        v = np.asarray(basis, dtype=float)
        if v.ndim != 2 or v.shape[0] != self.rhs.shape[0] or not np.isfinite(v).all():
            raise ValueError("Basis must live in the constrained fluctuation coordinates")
        if v.shape[1] == 0:
            reduced = np.zeros((len(v), self.rhs.shape[1]+1))
        else:
            if np.linalg.matrix_rank(v) != v.shape[1]:
                raise ValueError("Linearly dependent reduced basis")
            matrix = v.T @ (self.solver.Kr @ v)
            reduced = v @ np.linalg.solve(matrix, v.T @ np.column_stack([self.rhs, self.offset_rhs]))
        response = self._response(reduced[:, :-1], reduced[:, -1], "Galerkin-response")
        residual = np.column_stack([self.rhs, self.offset_rhs])-self.solver.Kr @ reduced
        # All inverse applications occur OFFLINE; no online original-grid traversal.
        gram = residual.T @ self.solver.solve_reduced(residual)/self.solver.mesh.rve_volume
        return response, ResidualCertificate(.5*(gram+gram.T))

    def displacement_basis(self, rank, *, energy_weighted=False):
        exact = self.solver.solve_reduced(self.rhs)
        if not 0 <= rank <= min(exact.shape):
            raise ValueError("Invalid displacement rank")
        if rank == 0: return np.empty((len(exact), 0))
        if energy_weighted:
            gram = exact.T @ (self.solver.Kr @ exact)
            values, vectors = np.linalg.eigh(gram)
            order = np.argsort(values)[::-1][:rank]
            v = exact @ vectors[:, order]
        else:
            v = np.linalg.svd(exact, full_matrices=False)[0][:, :rank]
        q, r = np.linalg.qr(v)
        if np.min(np.abs(np.diag(r))) < 1e-12*np.max(np.abs(np.diag(r))):
            raise ValueError("Requested rank exceeds the excited displacement span")
        return q[:, :rank]

    def _response(self, fluctuation, offset, label):
        s, mesh = self.solver, self.solver.mesh
        u = self.affine+s.P @ fluctuation
        u0 = self.lift0+s.P @ offset
        strain = np.asarray(s.D @ u).reshape(len(mesh.cells), 6, -1)
        strain0 = np.asarray(s.D @ u0).reshape(-1, 6)
        elastic, elastic0 = strain-self.eigen, strain0-self.inherent0
        stress = np.einsum("eij,ejd->eid", s.stiffness, elastic)
        stress0 = np.einsum("eij,ej->ei", s.stiffness, elastic0)
        hessian = np.einsum("eid,eif,e->df", elastic, stress, mesh.volumes)/mesh.rve_volume
        h = np.einsum("eid,ei,e->d", elastic, stress0, mesh.volumes)/mesh.rve_volume
        constant = .5*np.einsum("ei,ei,e->", elastic0, stress0, mesh.volumes)/mesh.rve_volume
        asym = np.linalg.norm(hessian-hessian.T)/max(np.linalg.norm(hessian), np.finfo(float).tiny)
        if asym > 1e-11: raise ValueError("Energy reciprocity lost")
        source = Path(__file__).resolve().parent
        implementation = hashlib.sha256(b"".join((source/name).read_bytes() for name in
            ("energy.py", "mechanics/reference.py", "mechanics/matrix_free.py", "mechanics/constraints.py", "mechanics/mesh.py", "mechanics/tensors.py"))).hexdigest()
        return EnergyResponse(.5*(hessian+hessian.T), h, float(constant), u.reshape(-1, 3, u.shape[1]), u0.reshape(-1, 3),
            strain, strain0, stress, stress0, mesh.volumes.copy(), mesh.phases.copy(), mesh.rve_volume,
            {"model": label, "boundary": s.constraint.boundary, "geometry": mesh.fingerprint,
             "fingerprint": _digest(mesh.points, mesh.cells, mesh.phases, mesh.box, s.stiffness, self.modes,
                                    self.inherent0, self.lift0, np.frombuffer((label+implementation).encode(), dtype=np.uint8), fluctuation),
             "implementation_sha256": implementation,
             "backend": type(s).__name__, "precision": "float64", "tensor_order": "xx yy zz sqrt2_yz sqrt2_xz sqrt2_xy",
             "assumptions": "fixed C, topology and linear chemical modes; no external nodal loads in this operator",
             "symmetry_error": float(asym), "rigid_modes_removed": s.constraint.rigid_modes_removed})


@dataclass
class ResidualCertificate:
    gram: np.ndarray  # Order [q,1], includes reference-state residual.

    def energy_norm_squared_density(self, q):
        q = np.asarray(q, dtype=float)
        z = np.concatenate([q, np.ones(q.shape[:-1]+(1,))], axis=-1)
        value = np.einsum("...i,ij,...j->...", z, self.gram, z)
        return np.maximum(value, 0.)

    def potential_error_density(self, q):
        return .5*self.energy_norm_squared_density(q)
