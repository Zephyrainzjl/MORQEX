import numpy as np
import pytest
from scipy.linalg import eigvalsh
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.transport import ConcentrationInfeasible
from morqex.concentration import ConcentrationBasis, ReducedDiffusion, ControlledDiffusion
from morqex.certification import tree_poincare_lower_bound, DiffusionCertificate, StressErrorTransfer
from morqex.certification import CertifiedControlledDiffusion, ConcentrationAccuracyError


@pytest.mark.parametrize("separated", [False, True])
def test_tree_poincare_is_lower_than_independent_generalized_eigenvalues(separated):
    labels = np.ones((4,)*3, dtype=np.uint8)
    if separated: labels[:, :, 2] = 0
    diffusion = VoxelDiffusion(Volume(labels, spacing=(.2, .3, .4)), active_phases=[1], diffusivity=[.7, 1.3, .5])
    spectrum = eigvalsh(diffusion.stiffness.toarray(), diffusion.mass.toarray())
    positive = spectrum[spectrum > 1e-9]
    bound = tree_poincare_lower_bound(diffusion)
    assert 0 < bound <= positive.min()*(1+1e-12)


@pytest.mark.parametrize("kind", ["spectral", "resolvent"])
def test_certificate_bounds_full_time_trajectory_and_each_component_mass(kind):
    labels = np.ones((4,)*3, dtype=np.uint8); labels[:, :, 2] = 0
    diffusion = VoxelDiffusion(Volume(labels, spacing=(.2, .3, .4)), active_phases=[1], diffusivity=.2,
        reactive_phases=[0], exterior_reactive=True)
    patterns = np.column_stack([np.ones(len(diffusion.face_areas)), diffusion.face_normals[:, 0],
                                diffusion.face_centers[:, 1], diffusion.face_centers[:, 2]])
    basis = (ConcentrationBasis.spectral(diffusion, 7) if kind == "spectral" else
        ConcentrationBasis.boundary_resolvent(diffusion, 7, patterns))
    model = ReducedDiffusion(diffusion, basis)
    certificate = DiffusionCertificate(diffusion, basis, patterns)
    rng = np.random.default_rng(731)
    c = 100.+rng.uniform(-1., 1., len(diffusion.weights))
    a = basis.encode(c)
    bound = certificate.initialize(c, a)
    for n in range(30):
        dt = .01 if n < 15 else .023
        forcing = np.array([.1*np.cos(n), .04*np.sin(.7*n), .03*np.cos(.2*n), .02*np.sin(.3*n)])
        next_a = model.step(a, dt, flux=patterns @ forcing)
        c = diffusion.step(c, dt, flux=patterns @ forcing).concentration
        bound = certificate.step(bound, a, next_a, dt, forcing)
        error = c-basis.decode(next_a)
        actual = np.sqrt(diffusion.weights @ error**2)
        assert actual <= bound.total_bound+1e-10
        for component in range(diffusion.n_components):
            selected = diffusion.components == component
            assert abs(diffusion.weights[selected] @ error[selected]) < 1e-12
        a = next_a


def test_online_certificate_uses_no_full_field_reconstruction(monkeypatch):
    diffusion = VoxelDiffusion(Volume(np.ones((3,)*3, dtype=np.uint8)), active_phases=[1], diffusivity=1.)
    patterns = np.ones((len(diffusion.face_areas), 1))
    basis = ConcentrationBasis.spectral(diffusion, 4)
    certificate = DiffusionCertificate(diffusion, basis, patterns)
    a = basis.encode(np.full(27, 100.))
    state = certificate.initialize(np.full(27, 100.), a)
    monkeypatch.setattr(basis, "decode", lambda *_: (_ for _ in ()).throw(AssertionError("full field accessed")))
    state = certificate.step(state, a, a, .1, np.zeros(1))
    assert state.total_bound < 1e-9


def test_resolvent_preserves_component_masses_and_matches_included_transfer_moment():
    from scipy.sparse.linalg import spsolve
    diffusion = VoxelDiffusion(Volume(np.ones((3,)*3, dtype=np.uint8)), active_phases=[1], diffusivity=1.)
    patterns = (diffusion.face_normals[:, 0] > .5)[:, None].astype(float)
    shift = .7
    basis = ConcentrationBasis.boundary_resolvent(diffusion, 3, patterns, shifts=[shift], moments=2)
    reduced = ReducedDiffusion(diffusion, basis)
    source = np.asarray(diffusion.flux_operator @ patterns)[:, 0]
    full = spsolve(diffusion.stiffness+shift*diffusion.mass, source)
    solution = basis.decode(np.linalg.solve(reduced.stiffness+shift*reduced.mass, basis.values.T @ source))
    np.testing.assert_allclose(solution, full, rtol=1e-11, atol=1e-11)


def test_unreachable_response_dimension_is_not_filled_with_invented_directions():
    diffusion = VoxelDiffusion(Volume(np.ones((2,)*3, dtype=np.uint8)), active_phases=[1], diffusivity=1.)
    with pytest.raises(ValueError, match="Insufficient independent"):
        ConcentrationBasis.boundary_resolvent(diffusion, 3, np.zeros((len(diffusion.face_areas), 1)))


def test_channel_step_full_space_matches_fv_and_online_never_decodes(monkeypatch):
    diffusion = VoxelDiffusion(Volume(np.ones((3,)*3, dtype=np.uint8)), active_phases=[1], diffusivity=.1)
    patterns = np.ones((len(diffusion.face_areas), 1))
    basis = ConcentrationBasis.spectral(diffusion, 27)
    model = ControlledDiffusion(diffusion, basis, patterns, reference_concentration=100., maximum_concentration=200.)
    decode = basis.decode
    monkeypatch.setattr(basis, "decode", lambda *_: (_ for _ in ()).throw(AssertionError("online full reconstruction")))
    a, c = np.zeros(27), np.full(27, 100.)
    for _ in range(6):
        a = model.step(a, .1, [.01]); c = diffusion.step(c, .1, flux=patterns[:, 0]*.01).concentration
    np.testing.assert_allclose(100.+decode(a), c, atol=1e-11)


def test_concentration_bound_transfers_to_discrete_stress_and_principal_quantile():
    from morqex.mechanics import TetMesh, ElasticitySolver, isotropic_stiffness
    from morqex.energy import ResponseBuilder
    from morqex.coupling import bind_chemical_modes
    from morqex.statistics import principal_tension, weighted_quantile
    volume = Volume(np.ones((2,)*3, dtype=np.uint8), spacing=(.5,)*3)
    diffusion = VoxelDiffusion(volume, active_phases=[1], diffusivity=1.)
    basis = ConcentrationBasis(np.eye(8), diffusion.weights, "full-coordinate", 1)
    mesh = TetMesh.from_volume(volume)
    solver = ElasticitySolver(mesh, isotropic_stiffness(1e9, .25))
    response = ResponseBuilder(solver, bind_chemical_modes(solver, diffusion, basis, [1e-6]*3+[0.]*3)).exact()
    transfer = StressErrorTransfer.build(response, diffusion.weights)
    rng = np.random.default_rng(133)
    for _ in range(10):
        c, error = rng.normal(size=(2, 8))*1000.
        a = response.stress(np.r_[np.zeros(6), c]); b = response.stress(np.r_[np.zeros(6), c+error])
        norm = np.sqrt(diffusion.weights @ error**2)
        field_error = np.sqrt(np.einsum("e,ei,ei->", mesh.volumes/mesh.rve_volume, b-a, b-a))
        assert field_error <= transfer.stress_l2_bound(norm)
        q1 = weighted_quantile(principal_tension(a), mesh.volumes, .99)
        q2 = weighted_quantile(principal_tension(b), mesh.volumes, .99)
        assert abs(q1-q2) <= transfer.quantile_bound(norm)


def test_over_tolerance_step_is_rejected_without_mutating_previous_state():
    diffusion = VoxelDiffusion(Volume(np.ones((3,)*3, dtype=np.uint8)), active_phases=[1], diffusivity=.1)
    patterns = (diffusion.face_normals[:, 0] > .5)[:, None].astype(float)
    basis = ConcentrationBasis.spectral(diffusion, 2)
    model = CertifiedControlledDiffusion(diffusion, basis, patterns, reference_concentration=100., maximum_concentration=200.)
    state = model.initialize(np.zeros(27), error_tolerance=1e-6)
    old = state.coefficients.copy()
    with pytest.raises(ConcentrationAccuracyError) as error:
        model.step(state, .1, [.1], error_tolerance=1e-6)
    assert error.value.error_bound > error.value.tolerance
    np.testing.assert_array_equal(state.coefficients, old)
    assert state.time == 0.


def test_error_enclosure_rejects_negative_full_field_hidden_by_positive_mean_mode():
    diffusion = VoxelDiffusion(Volume(np.ones((3,)*3, dtype=np.uint8)), active_phases=[1], diffusivity=.01)
    patterns = (diffusion.face_normals[:, 0] < -.5)[:, None].astype(float)
    basis = ConcentrationBasis.spectral(diffusion, 1)
    mean_only = ControlledDiffusion(diffusion, basis, patterns, reference_concentration=1., maximum_concentration=2.)
    # The average-only state remains positive, although the resolved boundary
    # layer becomes negative. A mass-only feasibility test would miss this.
    approximate = mean_only.step(np.zeros(1), 2., [-1.])
    assert np.min(1.+basis.decode(approximate)) > 0
    with pytest.raises(ConcentrationInfeasible): diffusion.step(np.ones(27), 2., flux=-patterns[:, 0])
    certified = CertifiedControlledDiffusion(diffusion, basis, patterns, reference_concentration=1., maximum_concentration=2.)
    state = certified.initialize(np.zeros(27), error_tolerance=10.)
    with pytest.raises(ConcentrationInfeasible, match="error enclosure"):
        certified.step(state, 2., [-1.], error_tolerance=10.)
