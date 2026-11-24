import numpy as np
import pytest
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion, conservative_box_projection, ConcentrationInfeasible
from morqex.concentration import ConcentrationBasis, ReducedDiffusion


def domain(n=4, particle_ids=None):
    return VoxelDiffusion(Volume(np.ones((n,)*3, dtype=np.uint8), spacing=(1/n,)*3),
                          active_phases=[1], diffusivity=(.1, .2, .3), particle_ids=particle_ids, exterior_reactive=True)


def test_closed_diffusion_inventory_positivity_and_boundary_flux_units():
    full = domain()
    c = np.random.default_rng(4).uniform(.1, .9, len(full.weights))
    original = full.volumes @ c
    for _ in range(5):
        state = full.step(c, .1); c = state.concentration
        assert state.inventory_mol == pytest.approx(original, abs=1e-14)
        assert state.inventory_error_mol == pytest.approx(0, abs=1e-14)
        assert c.min() >= .1 and c.max() <= .9
    flux = np.zeros_like(full.face_areas); flux[full.face_normals[:, 0] == -1] = .01
    state = full.step(c, .2, flux=flux)
    assert state.inventory_mol-original == pytest.approx(.2*.01, abs=1e-14)
    assert state.relative_residual < 1e-13


def test_particle_contact_does_not_create_unphysical_lithium_transport():
    ids = np.ones((4,)*3, dtype=np.uint8); ids[:, :, 2:] = 2
    full = domain(particle_ids=ids)
    assert full.n_components == 2
    c = np.where(ids.ravel() == 1, .1, .9)
    state = full.step(c, 10.)
    np.testing.assert_allclose(state.concentration, c, atol=1e-13)
    with pytest.raises(ValueError, match="every"):
        ConcentrationBasis.spectral(full, 1)


def test_diffusion_cosine_eigenmode_and_spatial_convergence():
    errors = []
    for n in (4, 8, 12):
        full = domain(n)
        c0 = 1+.2*np.cos(np.pi*full.centers[:, 0])
        c = c0.copy(); dt = .0002; steps = 100
        for _ in range(steps): c = full.step(c, dt).concentration
        discrete_eigenvalue = 4*.1*n*n*np.sin(np.pi/(2*n))**2
        discrete = 1+(c0-1)/(1+dt*discrete_eigenvalue)**steps
        np.testing.assert_allclose(c, discrete, atol=5e-13)
        exact = 1+(c0-1)*np.exp(-.1*np.pi**2*dt*steps)
        errors.append(np.sqrt(np.sum(full.weights*(c-exact)**2)))
    assert errors[0]/errors[1] > 3.5 and errors[1]/errors[2] > 2


def test_full_spectral_space_matches_independent_fv_steps():
    full = domain(3)
    basis = ConcentrationBasis.spectral(full, len(full.weights))
    reduced = ReducedDiffusion(full, basis)
    np.testing.assert_allclose(basis.mass, np.eye(len(full.weights)), atol=1e-12)
    c = np.random.default_rng(7).uniform(.2, .8, len(full.weights))
    a = basis.encode(c)
    flux = .001*np.ones(len(full.face_areas))
    for _ in range(3):
        c = full.step(c, .01, flux=flux).concentration
        a = reduced.step(a, .01, flux=flux)
        np.testing.assert_allclose(basis.decode(a), c, atol=1e-12)


@pytest.mark.parametrize("method", ["geometric", "spectral", "pod", "joint"])
def test_each_basis_preserves_each_particle_mass_with_truncation(method):
    ids = np.ones((4,)*3, dtype=np.uint8); ids[:, :, 2:] = 2
    full = domain(particle_ids=ids)
    snapshots = np.random.default_rng(31).uniform(.2, .8, (len(full.weights), 15))
    if method == "pod": basis = ConcentrationBasis.pod(full, snapshots, 5)
    elif method == "joint":
        basis = ConcentrationBasis.pod(full, snapshots, 5, target_action=lambda x: 10*x[:3])
    else: basis = getattr(ConcentrationBasis, method)(full, 5)
    decoded = basis.decode(basis.encode(snapshots))
    for k in range(2):
        mask = full.components == k
        np.testing.assert_allclose(full.weights[mask] @ decoded[mask], full.weights[mask] @ snapshots[mask], atol=1e-13)
    np.testing.assert_allclose(basis.mass, np.eye(5), atol=1e-12)


def test_mass_projection_fixes_clipped_shift_counterexample_without_losing_mass():
    c = np.r_[np.full(13, .01), np.full(14, .99)]
    w = np.ones(27)/27
    projected = conservative_box_projection(c, w, .95, lower=0., upper=1.)
    assert projected.min() >= 0 and projected.max() <= 1
    assert w @ projected == pytest.approx(.95, abs=1e-14)
    with pytest.raises(ConcentrationInfeasible): conservative_box_projection(c, w, 1.1)


def test_reduced_infeasibility_and_unsupported_diffusion_fail_explicitly():
    full = domain()
    basis = ConcentrationBasis.spectral(full, 4)
    reduced = ReducedDiffusion(full, basis)
    with pytest.raises(ConcentrationInfeasible):
        reduced.step(basis.encode(np.ones(len(full.weights))), .1, flux=100., maximum=1.)
    with pytest.raises(ValueError, match="full tensor"):
        VoxelDiffusion(full.volume, active_phases=[1], diffusivity=np.eye(3))
def test_reduced_space_without_particle_inventory_mode_is_rejected():
    volume = Volume(np.ones((2,)*3, dtype=np.uint8))
    full = VoxelDiffusion(volume, active_phases=[1], diffusivity=1.)
    bad = np.arange(8, dtype=float)-3.5
    basis = ConcentrationBasis(bad[:, None], full.weights, "deliberately-missing-mass", 1)
    with pytest.raises(ValueError, match="mass mode"):
        ReducedDiffusion(full, basis)
