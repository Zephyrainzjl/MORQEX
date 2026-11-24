import numpy as np
import pytest
from morqex.mechanics import (TetMesh, ElasticitySolver, MechanicalSolveError,
                             isotropic_stiffness, to_mandel, from_mandel, rotate_stiffness)
from morqex.mechanics.tensors import elastic_energy_and_mu
from morqex.microstructure import Volume


def cube(n=3, labels=None, spacing=None):
    a = np.ones((n, n, n), dtype=np.uint8) if labels is None else labels
    return TetMesh.from_volume(Volume(a, spacing=(1/n,)*3 if spacing is None else spacing))


def anisotropic(seed=7):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(6, 6))
    return 100*(a.T @ a+np.eye(6))


def test_mandel_energy_rotation_and_concentration_dependent_stiffness():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(3, 3)); eps = .01*(x+x.T)
    c = anisotropic()
    r, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(r) < 0: r[:, 0] *= -1
    ev = to_mandel(eps)
    np.testing.assert_allclose(from_mandel(ev), eps, atol=1e-14)
    np.testing.assert_allclose(ev @ ev, np.sum(eps**2), rtol=1e-14)
    rotated = to_mandel(r @ eps @ r.T)
    np.testing.assert_allclose(rotated @ rotate_stiffness(c, r) @ rotated, ev @ c @ ev, rtol=1e-13)
    beta = np.array([.02, .01, .03, .005, 0, 0])
    concentration = .4
    prime = .3*c
    _, _, mu = elastic_energy_and_mu(ev, concentration, c*(1+.3*concentration), beta, stiffness_derivative=prime)
    h = 1e-6
    def energy(t): return elastic_energy_and_mu(ev, t, c*(1+.3*t), beta)[0]
    assert mu == pytest.approx((energy(concentration+h)-energy(concentration-h))/(2*h), rel=1e-8)


@pytest.mark.parametrize("boundary", ["kubc", "periodic"])
def test_all_six_anisotropic_affine_patch_and_factor_reuse(boundary):
    mesh = cube()
    c = anisotropic()
    solver = ElasticitySolver(mesh, c, boundary=boundary)
    for eps in np.eye(6)*.001:
        s = solver.solve(macro_strain=eps)
        np.testing.assert_allclose(s.strain, np.broadcast_to(eps, s.strain.shape), atol=1e-14)
        np.testing.assert_allclose(s.mean_stress, c @ eps, atol=1e-11)
        assert s.elastic_energy == pytest.approx(.5*eps @ c @ eps, rel=1e-12)
        assert s.equilibrium_residual < 1e-11
    assert solver.factorizations == 1
    assert solver.linear_solves == 6


@pytest.mark.parametrize("boundary", ["kubc", "periodic", "neumann"])
def test_free_and_constrained_uniform_eigenstrain(boundary):
    mesh = cube()
    c = anisotropic()
    solver = ElasticitySolver(mesh, c, boundary=boundary)
    inherent = np.array([.01, .02, -.01, .005, .003, .002])
    if boundary == "neumann":
        state = solver.solve(inherent)
    else:
        state = solver.solve(inherent, macro_strain=inherent)
        constrained = solver.solve(inherent, macro_strain=np.zeros(6))
        np.testing.assert_allclose(constrained.mean_stress, -c @ inherent, atol=1e-10)
    np.testing.assert_allclose(state.stress, 0, atol=2e-10)
    assert state.elastic_energy < 1e-22


def test_pressure_virtual_work_reactions_and_unbalanced_load_rejection():
    mesh = cube()
    c = isotropic_stiffness(100, .25)
    solver = ElasticitySolver(mesh, c, boundary="neumann")
    force = mesh.pressure_load(.3)
    s = solver.solve(force=force)
    np.testing.assert_allclose(s.stress, np.broadcast_to([-.3, -.3, -.3, 0, 0, 0], s.stress.shape), atol=1e-11)
    assert force @ s.displacement.ravel() == pytest.approx(2*s.elastic_energy, rel=1e-11)
    np.testing.assert_allclose(s.reaction, 0, atol=1e-12)
    bad = force.copy(); bad[0] += .1
    with pytest.raises(MechanicalSolveError, match="load balance"):
        solver.solve(force=bad)


def laminate_stiffness(cs, fractions):
    # Analytic rank-one laminate: continuous normal traction and in-plane strain.
    a, b = np.array([0, 4, 5]), np.array([1, 2, 3])
    inv = [np.linalg.inv(c[np.ix_(a, a)]) for c in cs]
    m = sum(f*i for f, i in zip(fractions, inv))
    g = sum(f*i @ c[np.ix_(a, b)] for f, i, c in zip(fractions, inv, cs))
    s = sum(f*(c[np.ix_(b, b)]-c[np.ix_(b, a)] @ i @ c[np.ix_(a, b)]) for f, i, c in zip(fractions, inv, cs))
    l = np.linalg.inv(m)
    effective = np.zeros((6, 6))
    effective[np.ix_(a, a)] = l
    effective[np.ix_(a, b)] = l @ g
    effective[np.ix_(b, a)] = g.T @ l
    effective[np.ix_(b, b)] = s+g.T @ l @ g
    return effective


def test_anisotropic_periodic_laminate_and_hill_mandel():
    labels = np.ones((3, 3, 4), dtype=np.uint8); labels[:, :, 2:] = 2
    mesh = cube(labels=labels, spacing=(1/3, 1/3, 1/4))
    cs = [anisotropic(3), .3*anisotropic(5)]
    c = np.array([cs[int(p)-1] for p in mesh.phases])
    solver = ElasticitySolver(mesh, c, boundary="periodic")
    effective = laminate_stiffness(cs, [.5, .5])
    numerical = np.column_stack([solver.solve(macro_strain=e).mean_stress for e in np.eye(6)])
    np.testing.assert_allclose(numerical, effective, rtol=1e-11, atol=1e-10)
    e, delta = np.random.default_rng(2).normal(size=(2, 6))*.01
    s = solver.solve(macro_strain=e)
    v = solver.solve(macro_strain=delta)
    work = np.einsum("e,ei,ei->", mesh.volumes, s.stress, v.strain)/mesh.rve_volume
    assert work == pytest.approx(s.mean_stress @ delta, rel=1e-12)


@pytest.mark.parametrize("boundary", ["kubc", "periodic", "neumann"])
def test_true_void_and_isolated_solid_rigid_modes(boundary):
    labels = np.zeros((6, 6, 6), dtype=np.uint8)
    labels[1:3, 1:3, 1:3] = 1; labels[4:5, 4:5, 4:5] = 1
    v = Volume(labels, spacing=(1/6,)*3)
    mesh = TetMesh.from_volume(v, solid_phases=[1])
    assert mesh.n_components == 2
    assert mesh.volumes.sum() == pytest.approx(9/216)
    assert mesh.rve_volume == pytest.approx(1)
    solver = ElasticitySolver(mesh, anisotropic(), boundary=boundary)
    assert solver.constraint.rigid_modes_removed == 12
    s = solver.solve(np.array([.01, .02, .03, .001, 0, 0]))
    np.testing.assert_allclose(s.stress, 0, atol=1e-9)


def test_manufactured_quadratic_displacement_converges():
    errors = []
    c = isotropic_stiffness(100, .25)
    for n in (2, 4, 6):
        mesh = cube(n)
        solver = ElasticitySolver(mesh, c, boundary="kubc")
        exact = np.zeros_like(mesh.points); exact[:, 0] = .01*mesh.points[:, 0]**2
        s = solver.solve(prescribed_displacement=exact, force=mesh.body_load([-.02*c[0, 0], 0, 0]))
        exact_strain = np.zeros_like(s.strain); exact_strain[:, 0] = .02*mesh.centers[:, 0]
        errors.append(np.sqrt(np.einsum("e,ei,ei->", mesh.volumes, s.strain-exact_strain, s.strain-exact_strain)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[0]/errors[1] > 1.8


def test_invalid_material_and_nonperiodic_geometry_fail_explicitly():
    mesh = cube(2)
    with pytest.raises(ValueError, match="positive definite"):
        ElasticitySolver(mesh, np.zeros((6, 6)))
    labels = np.ones((3, 3, 3), dtype=np.uint8); labels[0, 0, 0] = 0
    mesh = TetMesh.from_volume(Volume(labels), solid_phases=[1])
    with pytest.raises(ValueError, match="matching solid nodes"):
        ElasticitySolver(mesh, isotropic_stiffness(1, .2), boundary="periodic")


def test_island_cut_by_periodic_boundary_retains_its_six_free_rigid_modes():
    labels = np.zeros((8,)*3, dtype=np.uint8)
    labels[3:5, 3:5, :2] = 1; labels[3:5, 3:5, 6:] = 1
    mesh = TetMesh.from_volume(Volume(labels, spacing=(1/8,)*3), solid_phases=[1])
    assert mesh.n_components == 2
    solver = ElasticitySolver(mesh, anisotropic(), boundary="periodic")
    assert solver.constraint.rigid_modes_removed == 6
    state = solver.solve(np.array([.01, .02, .03, .005, -.002, .001]))
    np.testing.assert_allclose(state.stress, 0, atol=1e-9)
