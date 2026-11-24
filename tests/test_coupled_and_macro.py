import numpy as np
import pytest
import os
from pathlib import Path
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.mechanics import TetMesh, ElasticitySolver, isotropic_stiffness
from morqex.transport import VoxelDiffusion
from morqex.concentration import ConcentrationBasis, ReducedDiffusion
from morqex.energy import ResponseBuilder
from morqex.coupling import bind_chemical_modes, CoupledTransport, FARADAY
from morqex.macroscale import LayerStack
from morqex.adaptivity import AdaptiveResponse


def coupled(beta_scale=1e-5, full_space=False):
    labels = np.full((4,)*3, 2, dtype=np.uint8)
    labels[:, :, 0] = 0; labels[1:3, 1:3, 1:3] = 1
    volume = Volume(labels, spacing=(1e-6,)*3)
    full = VoxelDiffusion(volume, active_phases=[1], diffusivity=1e-14, reactive_phases=[0])
    if full_space:
        basis = ConcentrationBasis(np.eye(len(full.weights)), full.weights.copy(), "full-voxel", full.n_components)
    else: basis = ConcentrationBasis.spectral(full, 5)
    mesh = TetMesh.from_volume(volume, solid_phases=[1, 2])
    cs = {1: isotropic_stiffness(1e9, .25), 2: isotropic_stiffness(2e8, .3)}
    solver = ElasticitySolver(mesh, np.array([cs[int(p)] for p in mesh.phases]), boundary="kubc")
    modes = bind_chemical_modes(solver, full, basis, beta_scale*np.array([1., .6, 1.4, 0., 0., 0.]))
    builder = ResponseBuilder(solver, modes)
    response = builder.exact()
    model = CoupledTransport(response, ReducedDiffusion(full, basis), chemical_curvature=.1,
                             reference_concentration=20000., maximum_concentration=40000.,
                             bounds_policy="exact" if full_space else "sufficient")
    return model, builder, full, basis


def test_closed_bidirectional_coupling_dissipation_mass_and_reconstruction_cost():
    model, _, _, basis = coupled()
    state = model.initialize(np.array([100., 200., -100., 150., 50.]))
    initial_mass = state.inventory_mol
    # Default sufficient-bound stepping must not reconstruct the original field.
    saved = basis.decode
    basis.decode = lambda a: (_ for _ in ()).throw(AssertionError("online bulk reconstruction"))
    for _ in range(20):
        next_state, diagnostic = model.step(state, .5)
        assert next_state.free_energy_density <= state.free_energy_density+1e-8
        assert next_state.inventory_mol == pytest.approx(initial_mass, abs=1e-25)
        assert abs(diagnostic.energy_balance_density) < 1e-7
        assert abs(diagnostic.inventory_error_mol) < 1e-25
        assert diagnostic.diffusion_dissipation_density >= -1e-12
        state = next_state
    basis.decode = saved
    concentration = model.concentration(state)
    assert concentration.min() > 0 and concentration.max() < 40000


def test_zero_expansion_full_space_recovers_independent_fv_diffusion():
    model, _, full, basis = coupled(beta_scale=0., full_space=True)
    initial = np.random.default_rng(33).uniform(-500., 500., len(full.weights))
    state = model.initialize(initial)
    c = initial+20000.
    for _ in range(5):
        state, _ = model.step(state, .2)
        c = full.step(c, .2).concentration
    np.testing.assert_allclose(model.concentration(state), c, atol=1e-9)


def test_mechanical_feedback_changes_transport_and_reservoir_bv_balances_energy():
    model, _, _, _ = coupled()
    no_mechanics, _, _, _ = coupled(beta_scale=0.)
    initial = np.array([100., 200., -100., 150., 50.])
    a = model.initialize(initial); b = no_mechanics.initialize(initial)
    for _ in range(5):
        a, _ = model.step(a, 1.); b, _ = no_mechanics.step(b, 1.)
    assert np.linalg.norm(a.coefficients-b.coefficients) > 1e-3
    state = model.initialize(np.zeros(5))
    for _ in range(4):
        next_state, diagnostic = model.step(state, .2, exchange_flux=1e-4, reservoir_potential=500.)
        assert diagnostic.reaction_dissipation_density >= 0
        assert diagnostic.inward_current_A > 0
        assert next_state.inventory_mol-state.inventory_mol == pytest.approx(.2*diagnostic.inward_current_A/FARADAY, abs=1e-25)
        assert abs(diagnostic.energy_balance_density) < 1e-7
        state = next_state


def test_changed_prescribed_macro_strain_has_explicit_external_work():
    model, _, _, _ = coupled()
    state = model.initialize(np.zeros(5))
    changed, diagnostic = model.step(state, .2, macro_strain=np.array([.001, 0., 0., 0., 0., 0.]))
    assert diagnostic.mechanical_work_density > 0
    assert abs(diagnostic.energy_balance_density) < 1e-7
    assert changed.free_energy_density > state.free_energy_density


def test_layer_stack_series_solution_pressure_thickness_and_energy_derivative():
    volume = Volume(np.ones((2,)*3, dtype=np.uint8), spacing=(.5,)*3)
    mesh = TetMesh.from_volume(volume)
    responses, cs = [], []
    beta = np.array([1e-6, 1e-6, 1e-6, 0., 0., 0.])
    for young, nu in ((1e9, .25), (2e8, .3)):
        c = isotropic_stiffness(young, nu); cs.append(c)
        builder = ResponseBuilder(ElasticitySolver(mesh, c), np.broadcast_to(beta[None, :, None], (len(mesh.cells), 6, 1)))
        responses.append(builder.exact())
    lengths = np.array([1e-4, 2e-4]); coefficients = [np.array([1000.]), np.array([2000.])]
    stack = LayerStack(responses, lengths)
    free = sum(t*(c[0, :3].sum()/c[0, 0])*beta[0]*a[0] for t, c, a in zip(lengths, cs, coefficients))
    expected = -free/sum(t/c[0, 0] for t, c in zip(lengths, cs))
    constrained = stack.solve(coefficients, thickness_change=0.)
    assert constrained.axial_stress == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(constrained.stress[:, 0], expected, rtol=1e-12)
    assert abs(constrained.thickness_change) < 1e-20
    pressure = stack.solve(coefficients, pressure=1e5)
    assert pressure.axial_stress == -1e5
    h = 1e-11
    fd = (stack.solve(coefficients, thickness_change=h).energy_per_area-stack.solve(coefficients, thickness_change=-h).energy_per_area)/(2*h)
    assert fd == pytest.approx(expected, rel=1e-8)


def test_residual_driven_enrichment_reports_model_energy_correction():
    model, builder, _, _ = coupled()
    adaptive = AdaptiveResponse(builder, np.empty((builder.rhs.shape[0], 0)))
    q = np.r_[np.zeros(6), [100., 200., -100., 150., 50.]]
    exact = builder.exact()
    result = adaptive.evaluate(q, absolute_energy_tolerance=1e-8)
    assert result.enriched and result.rank == 1
    assert result.error_bound_density < 1e-8
    assert result.model_energy_correction_density <= 0
    assert result.response.energy_density(q) == pytest.approx(exact.energy_density(q), rel=1e-10)
    again = adaptive.evaluate(q, absolute_energy_tolerance=1e-8)
    assert not again.enriched


def test_checkpoint_restart_matches_uninterrupted_physical_trajectory(tmp_path):
    model, _, _, _ = coupled()
    state = model.initialize(np.array([100., 200., -100., 150., 50.]))
    for _ in range(3): state, _ = model.step(state, .5)
    workspace = Workspace(Path(os.environ["MORQEX_WORKSPACE"]))
    path = model.save_state(state, workspace, str(tmp_path.relative_to(workspace.root)/"restart.json"))
    restored = model.load_state(path)
    for _ in range(3):
        state, _ = model.step(state, .5); restored, _ = model.step(restored, .5)
    np.testing.assert_array_equal(restored.coefficients, state.coefficients)
    assert restored.time == state.time and restored.inventory_mol == state.inventory_mol
    other, _, _, _ = coupled(beta_scale=0.)
    with pytest.raises(ValueError, match="does not match"): other.load_state(path)
