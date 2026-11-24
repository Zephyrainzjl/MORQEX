from pathlib import Path
import numpy as np
import pytest
from morqex.mechanics import TetMesh, ElasticitySolver, isotropic_stiffness
from morqex.microstructure import Volume
from morqex.energy import ResponseBuilder, EnergyResponse
from morqex.statistics import StressMoments, StressEnvelope, principal_tension, weighted_quantile, weighted_cvar
from morqex.workspace import Workspace


def problem(boundary="periodic"):
    labels = np.ones((3, 3, 4), dtype=np.uint8); labels[:, :, 2:] = 2
    mesh = TetMesh.from_volume(Volume(labels, spacing=(1/3, 1/3, 1/4)))
    cs = [isotropic_stiffness(1e9, .25), isotropic_stiffness(2e8, .3)]
    c = np.array([cs[int(p)-1] for p in mesh.phases])
    solver = ElasticitySolver(mesh, c, boundary=boundary)
    active = mesh.phases == 1
    phi = np.column_stack([active, active*np.sin(2*np.pi*mesh.centers[:, 1]), active*np.cos(2*np.pi*mesh.centers[:, 2])])
    beta = np.array([1e-6, 2e-6, .5e-6, 0., .2e-6, 0.])
    modes = beta[None, :, None]*phi[:, None, :]
    reference = active[:, None]*beta[None, :]*100.
    return solver, phi, beta, ResponseBuilder(solver, modes, reference_eigenstrain=reference), reference


@pytest.mark.parametrize("boundary", ["kubc", "periodic"])
def test_3d_response_fields_energy_and_shared_derivatives_against_direct_solves(boundary):
    solver, phi, beta, builder, reference = problem(boundary)
    response = builder.exact()
    rng = np.random.default_rng(20261124)
    for _ in range(3):
        q = rng.normal(size=response.dimension)*np.r_[np.full(6, .001), np.full(3, 1000.)]
        inherent = reference+np.einsum("eim,m->ei", builder.modes, q[6:])
        direct = solver.solve(inherent, macro_strain=q[:6])
        np.testing.assert_allclose(response.displacement(q), direct.displacement, atol=1e-13)
        np.testing.assert_allclose(response.stress(q), direct.stress, atol=1e-6, rtol=1e-11)
        np.testing.assert_allclose(response.strain(q), direct.strain, atol=1e-13)
        assert response.energy_density(q)*solver.mesh.rve_volume == pytest.approx(direct.elastic_energy, rel=1e-12)
        np.testing.assert_allclose(response.macro_stress(q), direct.mean_stress, rtol=1e-10, atol=1e-7)
        independent_g = -np.einsum("e,ei,eim->m", solver.mesh.volumes, direct.stress, builder.modes)/solver.mesh.rve_volume
        np.testing.assert_allclose(response.generalized_chemical_force(q), independent_g, atol=1e-11)
        step = np.r_[np.full(6, 1e-7), np.full(3, .01)]
        gradient_fd = np.array([(response.energy_density(q+np.eye(len(q))[j]*h)-response.energy_density(q-np.eye(len(q))[j]*h))/(2*h) for j, h in enumerate(step)])
        np.testing.assert_allclose(gradient_fd, response.gradient(q), rtol=1e-6, atol=1e-6)
        mass = phi.T @ (solver.mesh.volumes[:, None]*phi)/solver.mesh.rve_volume
        mu = response.projected_chemical_potential(q, mass)
        np.testing.assert_allclose(mass @ mu, independent_g, atol=1e-11)
        roi = np.arange(0, len(phi), 3)
        np.testing.assert_allclose(response.stress(q, roi), direct.stress[roi], atol=1e-6)
    assert solver.factorizations == 1
    scales = np.r_[np.full(6, .001), np.full(3, 1000.)]
    hscaled = scales[:, None]*response.H*scales[None, :]
    assert np.linalg.eigvalsh(hscaled).min() > -np.linalg.norm(hscaled)*1e-12


def test_galerkin_response_residual_identity_and_macro_energy_remain_consistent():
    solver, _, _, builder, _ = problem()
    exact = builder.exact()
    basis = builder.displacement_basis(3)
    reduced, certificate = builder.galerkin(basis)
    q = np.r_[np.array([.001, -.002, .003, 0., .0005, -.001]), [1200., -300., 500.]]
    error = (exact.displacement(q)-reduced.displacement(q)).ravel()
    truth = error @ (solver.K @ error)/solver.mesh.rve_volume
    assert certificate.energy_norm_squared_density(q) == pytest.approx(truth, rel=1e-10)
    assert reduced.energy_density(q)-exact.energy_density(q) == pytest.approx(.5*truth, rel=1e-10)
    averaged = np.einsum("e,ei->i", solver.mesh.volumes, reduced.stress(q))/solver.mesh.rve_volume
    np.testing.assert_allclose(averaged, reduced.macro_stress(q), atol=1e-6)


def test_free_macro_stress_and_precise_phase_moments():
    _, _, _, builder, _ = problem()
    response = builder.exact()
    a = np.array([2000., 100., -150.])
    e = response.macro_strain_at_stress(a, np.zeros(6))
    q = np.r_[e, a]
    np.testing.assert_allclose(response.macro_stress(q), 0, atol=1e-7)
    for selection in (None, response.phases == 1, response.phases == 2):
        moments = StressMoments.build(response, selection)
        mean, covariance = moments.evaluate(q)
        stress = response.stress(q, selection)
        weights = response.volumes if selection is None else response.volumes[selection]
        weights = weights/weights.sum()
        truth_mean = weights @ stress
        truth_cov = np.einsum("n,ni,nj->ij", weights, stress-truth_mean, stress-truth_mean)
        np.testing.assert_allclose(mean, truth_mean, atol=1e-7)
        np.testing.assert_allclose(covariance, truth_cov, rtol=1e-11, atol=.01)


def test_volume_weighted_quantiles_cvar_and_tail_bounds_for_held_out_loads():
    _, _, _, builder, _ = problem()
    response = builder.exact()
    scales = np.r_[np.full(6, .003), np.full(3, 3000.)]
    envelope = StressEnvelope.build(response, 12, scales)
    rng = np.random.default_rng(29)
    for _ in range(20):
        z = rng.normal(size=response.dimension); z *= rng.uniform(.1, 1)/np.linalg.norm(z)
        q = z*scales
        values = principal_tension(response.stress(q))
        for probability in (.90, .95, .99):
            risk = envelope.evaluate(q, probability=probability, threshold=1e6)
            exact = weighted_quantile(values, response.volumes, probability)
            exact_cvar = weighted_cvar(values, response.volumes, probability)
            assert risk["quantile_lower"]-1e-7 <= exact <= risk["quantile_upper"]+1e-7
            assert risk["cvar_lower"]-1e-7 <= exact_cvar <= risk["cvar_upper"]+1e-7
            fraction = response.volumes[values > 1e6].sum()/response.volumes.sum()
            assert risk["exceedance_lower"]-1e-12 <= fraction <= risk["exceedance_upper"]+1e-12
    with pytest.raises(ValueError, match="outside"):
        envelope.evaluate(scales*2)
    assert weighted_quantile([1, 2, 100], [.8, .19, .01], .95) == 2
    assert weighted_cvar([1, 2, 100], [.8, .19, .01], .99) == pytest.approx(100)


def test_operator_persistence_checksum_and_geometry_identity(tmp_path):
    _, _, _, builder, _ = problem()
    response = builder.exact()
    path = response.save(Workspace(tmp_path), "operator")
    loaded = EnergyResponse.load(path, expected_fingerprint=response.metadata["fingerprint"])
    q = np.zeros(response.dimension)
    np.testing.assert_array_equal(loaded.gradient(q), response.gradient(q))
    with pytest.raises(ValueError, match="does not match"):
        EnergyResponse.load(path, expected_fingerprint="different geometry")
    with (path / "response.npz").open("ab") as f: f.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        EnergyResponse.load(path)
