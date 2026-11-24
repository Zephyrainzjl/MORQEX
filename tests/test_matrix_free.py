import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator
from morqex.microstructure import Volume
from morqex.mechanics import TetMesh, ElasticitySolver, MatrixFreeElasticitySolver, MechanicalSolveError


@pytest.mark.parametrize("boundary", ["kubc", "periodic", "neumann"])
def test_element_operator_matches_qualified_sparse_reference(boundary):
    mesh = TetMesh.from_volume(Volume(np.ones((3,)*3, dtype=np.uint8), spacing=(1/3,)*3))
    rng = np.random.default_rng(81)
    a = rng.normal(size=(len(mesh.cells), 6, 6))
    stiffness = 1e8*(np.einsum("eji,ejk->eik", a, a)+np.eye(6))
    direct = ElasticitySolver(mesh, stiffness, boundary=boundary)
    iterative = MatrixFreeElasticitySolver(mesh, stiffness, boundary=boundary)
    x = rng.normal(size=(len(mesh.points)*3, 3))
    np.testing.assert_allclose(iterative.K @ x, direct.K @ x, atol=2e-6, rtol=1e-12)
    strain = None if boundary == "neumann" else rng.normal(scale=.001, size=6)
    eigenstrain = rng.normal(scale=.01, size=(len(mesh.cells), 6))
    exact = direct.solve(eigenstrain, macro_strain=strain)
    state = iterative.solve(eigenstrain, macro_strain=strain)
    np.testing.assert_allclose(state.displacement, exact.displacement, atol=1e-10)
    assert np.linalg.norm(state.stress-exact.stress)/np.linalg.norm(exact.stress) < 1e-9
    assert state.elastic_energy == pytest.approx(exact.elastic_energy, rel=1e-10)
    assert isinstance(iterative.K, LinearOperator) and isinstance(iterative.Kr, LinearOperator)
    assert iterative.factorizations == 0 and max(iterative.iterations) < 1000


def test_iteration_budget_failure_does_not_return_unconverged_state():
    mesh = TetMesh.from_volume(Volume(np.ones((3,)*3, dtype=np.uint8)))
    solver = MatrixFreeElasticitySolver(mesh, np.eye(6), maximum_iterations=1)
    with pytest.raises(MechanicalSolveError, match="did not converge"):
        solver.solve(np.random.default_rng(1).normal(size=(len(mesh.cells), 6)))
