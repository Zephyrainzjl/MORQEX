"""Element matrix-free Tet4/CG alternative with a Jacobi preconditioner.

No global stiffness matrix or sparse factorization is assembled. Element blocks
and the strain-displacement map are retained. This is an experimental backend;
Jacobi is not expected to be competitive at high contrast or fine resolution.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg
from .constraints import constraints
from .reference import ElasticitySolver, MechanicalSolveError
from .tensors import MANDEL_BASIS, validate_stiffness


class MatrixFreeElasticitySolver(ElasticitySolver):
    def __init__(self, mesh, stiffness, *, boundary="kubc", relative_tolerance=1e-11, maximum_iterations=3000):
        started = time.perf_counter()
        if not 0 < relative_tolerance < 1 or maximum_iterations < 1:
            raise ValueError("Invalid iterative solver tolerance/budget")
        self.mesh = mesh
        nc, nd = len(mesh.cells), 3*len(mesh.points)
        self.stiffness = np.broadcast_to(validate_stiffness(stiffness), (nc, 6, 6)).copy()
        self.B = np.einsum("sij,eaj->esai", MANDEL_BASIS, mesh.gradients).reshape(nc, 6, 12)
        row = np.broadcast_to(6*np.arange(nc)[:, None, None]+np.arange(6)[None, :, None], self.B.shape)
        col = np.broadcast_to(mesh.dofs[:, None, :], self.B.shape)
        self.D = coo_matrix((self.B.ravel(), (row.ravel(), col.ravel())), shape=(6*nc, nd)).tocsr()
        self.D.eliminate_zeros()
        self.element_stiffness = np.einsum("esi,est,etj,e->eij", self.B, self.stiffness, self.B, mesh.volumes, optimize=True)
        self.constraint = constraints(mesh, boundary)
        self.P = self.constraint.projection
        nr = self.P.shape[1]
        def multiply(value):
            value = np.asarray(value)
            vector = value.ndim == 1
            if vector: value = value[:, None]
            local = np.einsum("eij,ejk->eik", self.element_stiffness, value[mesh.dofs], optimize=True)
            result = np.zeros((nd, value.shape[1]))
            np.add.at(result, mesh.dofs.ravel(), local.reshape(-1, value.shape[1]))
            return result[:, 0] if vector else result
        self.K = LinearOperator((nd, nd), matvec=multiply, matmat=multiply, dtype=np.float64)
        def constrained(value): return np.asarray(self.P.T @ multiply(self.P @ value))
        self.Kr = LinearOperator((nr, nr), matvec=constrained, matmat=constrained, dtype=np.float64)
        full_diagonal = np.zeros(nd)
        np.add.at(full_diagonal, mesh.dofs.ravel(), np.diagonal(self.element_stiffness, axis1=1, axis2=2).ravel())
        self.scale = float(full_diagonal.max())
        # Include same-element couplings identified by periodic DOF equivalence.
        mapping = np.full(nd, -1, dtype=int)
        rows, columns = self.P.nonzero(); mapping[rows] = columns
        element_map = mapping[mesh.dofs]
        diagonal = np.zeros(nr)
        for a in range(12):
            for b in range(12):
                selected = (element_map[:, a] >= 0) & (element_map[:, a] == element_map[:, b])
                np.add.at(diagonal, element_map[selected, a], self.element_stiffness[selected, a, b])
        if nr and (not np.isfinite(diagonal).all() or np.any(diagonal <= 0)):
            raise MechanicalSolveError("Nonpositive matrix-free diagonal; check constraints/topology")
        self.preconditioner = LinearOperator((nr, nr), matvec=lambda x: x/diagonal, dtype=np.float64)
        self.relative_tolerance, self.maximum_iterations = relative_tolerance, maximum_iterations
        self.assembly_seconds = time.perf_counter()-started
        self.factorization_seconds, self.factorizations, self.linear_solves = 0., 0, 0
        self.iterations = []
        self.affine = np.einsum("kij,nj->nik", MANDEL_BASIS, mesh.points-mesh.box.mean(axis=0)).reshape(-1, 6)

    def solve_reduced(self, rhs):
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.P.shape[1] or not np.isfinite(rhs).all():
            raise ValueError("Invalid constrained RHS")
        self.linear_solves += 1
        if not len(rhs): return np.zeros_like(rhs)
        vector = rhs.ndim == 1
        columns = rhs[:, None] if vector else rhs
        result = np.zeros_like(columns)
        for k, column in enumerate(columns.T):
            iterations = 0
            def count(_):
                nonlocal iterations
                iterations += 1
            solution, info = cg(self.Kr, column, M=self.preconditioner, rtol=self.relative_tolerance,
                atol=0., maxiter=self.maximum_iterations, callback=count)
            self.iterations.append(iterations)
            if info != 0:
                raise MechanicalSolveError(f"Matrix-free CG did not converge (info={info}, iterations={iterations}); no reference state returned")
            result[:, k] = solution
        return result[:, 0] if vector else result
