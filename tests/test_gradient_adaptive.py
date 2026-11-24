import numpy as np
import pytest
from scipy.linalg import eigvalsh
from morqex.microstructure import Volume
from morqex.mechanics import TetMesh, ElasticitySolver, isotropic_stiffness
from morqex.transport import VoxelDiffusion
from morqex.concentration import ConcentrationBasis, ReducedDiffusion
from morqex.coupling import bind_chemical_modes, CoupledTransport
from morqex.energy import ResponseBuilder
from morqex.gradient_transport import QuadraticGradientSystem, PortGradientModel, EnergyAccuracyError
from morqex.adaptive_transport import AdaptiveGradientTransport, AdaptiveBudgetError


def problem(beta=1e-5, separated=False):
    labels = np.full((4,4,5),2,dtype=np.uint8); labels[:,:,0] = 0; labels[1:3,1:3,1:4] = 1
    volume = Volume(labels,spacing=(1e-6,)*3)
    particle_ids = None
    if separated:
        particle_ids = np.ones_like(labels); particle_ids[:,:,3:] = 2
    diffusion = VoxelDiffusion(volume,active_phases=[1],diffusivity=1e-14,reactive_phases=[0],particle_ids=particle_ids)
    mesh = TetMesh.from_volume(volume,solid_phases=[1,2])
    tensors = {1:isotropic_stiffness(1e9,.25),2:isotropic_stiffness(2e8,.3)}
    solver = ElasticitySolver(mesh,np.array([tensors[int(p)] for p in mesh.phases]))
    full_basis = ConcentrationBasis(np.eye(len(diffusion.weights)),diffusion.weights.copy(),"identity",diffusion.n_components)
    modes = bind_chemical_modes(solver,diffusion,full_basis,beta*np.array([1.,.6,1.4,0.,0.,0.]))
    response = ResponseBuilder(solver,modes).exact()
    patterns = np.column_stack([diffusion.reactive, diffusion.reactive*(diffusion.face_centers[:,1] < 2e-6)]).astype(float)
    system = QuadraticGradientSystem(response,diffusion,full_basis,patterns,chemical_curvature=.1,
        reference_concentration=20000.,maximum_concentration=40000.)
    return system,full_basis,patterns


def test_malformed_spectral_enclosure_is_rejected_before_step():
    from dataclasses import replace
    system,_,_ = problem()
    basis = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,3))
    model = PortGradientModel(system,basis,certificate_method="spectral",error_modes=2)
    state = model.initialize(np.zeros(system.n),error_tolerance=100.)
    for error in (replace(state.error,spectral_tail=-1.),
                  replace(state.error,spectral_proxy=np.full(2,np.nan)),
                  replace(state.error,spectral_uncertainty=10.)):
        with pytest.raises(ValueError): model.accept(replace(state,error=error),100.)


@pytest.mark.parametrize("beta",[0.,1e-5])
def test_full_gradient_system_matches_independent_existing_coupled_equations(beta):
    system,basis,patterns = problem(beta)
    previous_model = CoupledTransport(system.response,ReducedDiffusion(system.diffusion,basis),chemical_curvature=.1,
        reference_concentration=20000.,maximum_concentration=40000.,bounds_policy="exact")
    c = np.random.default_rng(19).normal(size=system.n)*100
    previous = previous_model.initialize(c)
    for k in range(5):
        u = np.array([1e-5,2e-5]); e = np.array([1e-4*k,0.,0.,0.,0.,0.])
        c = system.full_step(c,.5,u,macro_strain=e)
        previous,diag = previous_model.step(previous,.5,flux=patterns @ u,macro_strain=e)
        np.testing.assert_allclose(c,previous.coefficients,rtol=1e-10,atol=1e-8)
        assert system.energy(c,e) == pytest.approx(previous.free_energy_density,rel=1e-10,abs=1e-7)


def test_full_gradient_no_expansion_reduces_to_independent_fv():
    system,_,patterns = problem(0.)
    c = np.zeros(system.n); actual = np.full(system.n,20000.)
    for _ in range(5):
        c = system.full_step(c,.5,[1e-5,2e-5])
        actual = system.diffusion.step(actual,.5,flux=patterns @ [1e-5,2e-5]).concentration
    np.testing.assert_allclose(c,actual-20000.,atol=1e-9)


def test_port_projection_preserves_mass_energy_and_full_generator_error_bound():
    system,_,_ = problem(separated=True)
    basis = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,5))
    model = PortGradientModel(system,basis)
    c = np.zeros(system.n); state = model.initialize(c,error_tolerance=1e3)
    for k in range(20):
        u = np.array([1e-5,2e-5*np.sin(k)])
        e = np.array([1e-4*np.sin(.2*k),0.,0.,0.,0.,0.])
        c = system.full_step(c,.2,u,macro_strain=e)
        state,diag = model.step(state,.2,u,macro_strain=e,error_tolerance=1e3)
        actual_error = system.norm(c-basis.values @ state.coefficients)
        assert actual_error <= state.error.total_bound+1e-8
        assert abs(diag.energy_balance) < 1e-7
        assert diag.diffusion_dissipation >= -1e-9
        assert np.max(np.abs(diag.component_mass_error)) < 1e-25
        np.testing.assert_allclose(state.component_inventory,system.baseline_inventory+system.inventory_rows @ c,atol=1e-25)


def test_mass_riesz_directions_are_necessary_and_graph_gap_is_valid_for_coupling():
    system,_,_ = problem()
    constant = ConcentrationBasis.spectral(system.diffusion,1)
    with pytest.raises(ValueError,match="mass Riesz"):
        PortGradientModel(system,constant)
    symmetric = system.chol.T @ system.A @ system.chol
    spectrum = eigvalsh(symmetric)
    positive = spectrum[spectrum > 1e-8]
    assert system.gap_lower <= positive.min()*(1+1e-10)


def test_nested_transfer_keeps_nonzero_error_and_prevents_false_exactness():
    system,full_basis,_ = problem()
    basis = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,2))
    coarse = PortGradientModel(system,basis)
    initial = np.random.default_rng(71).normal(size=system.n)
    old = coarse.initialize(initial,error_tolerance=100.)
    assert old.error.total_bound > 1e-3
    from morqex.concentration import _orthogonal
    completed = _orthogonal(np.eye(system.n),system.weights,initial=basis.values,count=system.n)
    fine = PortGradientModel(system,ConcentrationBasis(completed,system.weights,"completed",basis.component_count))
    transferred,info = coarse.transfer_to(old,fine)
    assert transferred.error is old.error
    assert abs(info["energy_jump"]) < 1e-10 and max(abs(x) for x in info["mass_jump"]) < 1e-25
    np.testing.assert_array_equal(basis.values @ old.coefficients,completed @ transferred.coefficients)
    truth = system.full_step(initial,.5,[0.,0.])
    current,_ = fine.step(transferred,.5,[0.,0.],error_tolerance=100.)
    error = system.norm(truth-completed @ current.coefficients)
    assert 1e-3 < error <= current.error.total_bound+1e-8
    # Resetting the bound to zero at enrichment would falsely claim exactness.
    assert error > current.error.residual_norm*100


@pytest.mark.parametrize("certificate",["graph","spectral"])
def test_online_port_step_does_not_recover_bulk_or_query_full_system(monkeypatch,certificate):
    system,_,_ = problem()
    basis = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,3))
    model = PortGradientModel(system,basis,certificate_method=certificate,error_modes=2); state = model.initialize(np.zeros(system.n),error_tolerance=1.)
    def forbidden(*args,**kwargs): raise AssertionError("online full traversal")
    monkeypatch.setattr(system,"full_step",forbidden); monkeypatch.setattr(system,"energy",forbidden)
    monkeypatch.setattr(model.basis,"decode",forbidden)
    state,_ = model.step(state,.2,[1e-6,0.],error_tolerance=1.)
    assert state.time == .2


@pytest.mark.parametrize("strategy",["snapshot","resolvent-block","compressed-resolvent-block","state-preserving-restart"])
def test_adaptive_correction_meets_bound_without_exact_trajectory_or_energy_jump(strategy):
    system,_,_ = problem()
    basis = ConcentrationBasis.spectral(system.diffusion,2)
    adaptive = AdaptiveGradientTransport(system,basis,maximum_modes=system.n,enrichment_strategy=strategy,channel_scales=[1e-5,2e-5],restart_dimension=8)
    state = adaptive.initialize(np.zeros(system.n),error_tolerance=.01)
    truth = np.zeros(system.n); events = []
    for _ in range(8):
        truth = system.full_step(truth,.5,[1e-5,2e-5])
        state,diag = adaptive.step(state,.5,[1e-5,2e-5],error_tolerance=.01)
        error = system.norm(truth-adaptive.model.basis.values @ state.coefficients)
        assert error <= state.error.total_bound+1e-8 <= .01000001
        assert abs(diag.dynamics.energy_balance) < 1e-7
        events.extend(diag.events)
    assert events and adaptive.total_full_queries > 0
    assert max(abs(e["energy_jump"]) for e in events) < 1e-7
    if strategy != "state-preserving-restart": assert all(e["error_bound_before"] == e["error_bound_after"] for e in events)
    else: assert all(e["error_bound_after"] >= e["error_bound_before"]-1e-12 for e in events)
    if strategy != "snapshot": assert adaptive.total_resolvent_rhs > 0


def test_budget_exhaustion_rejects_without_changing_active_model_or_state():
    system,_,_ = problem()
    basis = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,2))
    adaptive = AdaptiveGradientTransport(system,basis,maximum_modes=basis.values.shape[1])
    state = adaptive.initialize(np.zeros(system.n),error_tolerance=1e-8)
    old = adaptive.model.fingerprint
    with pytest.raises(AdaptiveBudgetError):
        adaptive.step(state,.5,[1e-4,2e-4],error_tolerance=1e-8)
    assert adaptive.model.fingerprint == old and state.time == 0.


def test_macro_loading_enrichment_carries_work_and_error():
    system,_,_ = problem()
    adaptive = AdaptiveGradientTransport(system,ConcentrationBasis.spectral(system.diffusion,2),maximum_modes=system.n)
    state = adaptive.initialize(np.zeros(system.n),error_tolerance=.01)
    truth = np.zeros(system.n)
    for k in range(3):
        e = np.array([.001*(k+1),0.,0.,0.,0.,0.])
        truth = system.full_step(truth,.5,[0.,0.],macro_strain=e)
        state,diag = adaptive.step(state,.5,[0.,0.],macro_strain=e,error_tolerance=.01)
        assert system.norm(truth-adaptive.model.basis.values @ state.coefficients) <= state.error.total_bound+1e-8
        assert abs(diag.dynamics.energy_balance) < 1e-6
        assert diag.dynamics.mechanical_work > 0


def test_explicit_full_space_fallback_is_counted():
    system,_,_ = problem()
    basis = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,2))
    adaptive = AdaptiveGradientTransport(system,basis,maximum_modes=basis.values.shape[1],allow_full_fallback=True)
    state = adaptive.initialize(np.zeros(system.n),error_tolerance=.001)
    state,diag = adaptive.step(state,.5,[1e-4,2e-4],error_tolerance=.001)
    assert diag.used_full_fallback and diag.full_space_steps == 1 and adaptive.total_full_space_steps == 1
    assert adaptive.model.dimension == system.n
    assert diag.events[0]["error_bound_after"] == diag.events[0]["error_bound_before"]


@pytest.mark.parametrize("strategy,certificate",[("snapshot","graph"),("resolvent-block","graph"),("compressed-resolvent-block","graph"),("state-preserving-restart","graph"),("snapshot","spectral")])
def test_enriched_checkpoint_restart_preserves_trajectory_and_accumulated_bound(tmp_path,strategy,certificate):
    from morqex import Workspace
    system,_,_ = problem()
    adaptive = AdaptiveGradientTransport(system,ConcentrationBasis.spectral(system.diffusion,2),maximum_modes=system.n,enrichment_strategy=strategy,channel_scales=[1e-5,2e-5],restart_dimension=8,certificate_method=certificate,error_modes=2)
    state = adaptive.initialize(np.zeros(system.n),error_tolerance=.01)
    for _ in range(3): state,_ = adaptive.step(state,.5,[1e-5,2e-5],error_tolerance=.01)
    path = adaptive.save_checkpoint(state,Workspace(tmp_path),"restart")
    resumed,saved = AdaptiveGradientTransport.load_checkpoint(system,path)
    assert saved.error.total_bound == state.error.total_bound and saved.time == state.time
    for _ in range(3):
        state,_ = adaptive.step(state,.5,[1e-5,2e-5],error_tolerance=.01)
        saved,_ = resumed.step(saved,.5,[1e-5,2e-5],error_tolerance=.01)
    np.testing.assert_allclose(saved.coefficients,state.coefficients,atol=1e-11)
    assert saved.error.total_bound == pytest.approx(state.error.total_bound,rel=1e-10,abs=1e-12)
    other,_,_ = problem(0.)
    with pytest.raises(ValueError,match="does not match"):
        AdaptiveGradientTransport.load_checkpoint(other,path)


def test_mass_projection_matches_existing_two_field_coupling_on_same_basis():
    system,_,patterns = problem()
    basis = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,3))
    model = PortGradientModel(system,basis,projection="mass")
    # Independent pre-existing reduced equations use the identical chemical
    # eigenstrain space; no intentionally weak reassembly baseline is used.
    volume = system.diffusion.volume
    mesh = TetMesh.from_volume(volume,solid_phases=[1,2])
    tensors = {1:isotropic_stiffness(1e9,.25),2:isotropic_stiffness(2e8,.3)}
    solver = ElasticitySolver(mesh,np.array([tensors[int(p)] for p in mesh.phases]))
    response = ResponseBuilder(solver,bind_chemical_modes(solver,system.diffusion,basis,1e-5*np.array([1.,.6,1.4,0.,0.,0.]))).exact()
    previous_model = CoupledTransport(response,ReducedDiffusion(system.diffusion,basis),chemical_curvature=.1,
        reference_concentration=20000.,maximum_concentration=40000.)
    state = model.initialize(np.zeros(system.n),error_tolerance=1e3)
    previous = previous_model.initialize(np.zeros(model.dimension))
    for k in range(5):
        e = np.array([k*1e-4,0.,0.,0.,0.,0.]); u = [1e-5,2e-5]
        state,_ = model.step(state,.2,u,error_tolerance=1e3,macro_strain=e)
        previous,_ = previous_model.step(previous,.2,flux=patterns @ u,macro_strain=e)
        np.testing.assert_allclose(state.coefficients,previous.coefficients,atol=1e-9,rtol=1e-10)


@pytest.mark.parametrize("retained",[0,2,100])
def test_signed_error_dynamics_and_complement_bound_cover_full_trajectory(retained):
    system,_,_ = problem()
    basis = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,3))
    model = PortGradientModel(system,basis,certificate_method="spectral",error_modes=retained)
    c = np.random.default_rng(125).normal(size=system.n)*20
    state = model.initialize(c,error_tolerance=1e3)
    for k in range(40):
        u = [1e-5*np.cos(k*.3),2e-5*np.sin(k*.2)]
        e = np.array([.0003*np.sin(k*.1),0.,0.,0.,0.,0.])
        c = system.full_step(c,.2,u,macro_strain=e)
        state,_ = model.step(state,.2,u,macro_strain=e,error_tolerance=1e3)
        actual = system.norm(c-basis.values @ state.coefficients)
        assert actual <= state.error.total_bound+1e-8
        if retained == 100: assert state.error.total_bound-actual < 1e-5


def test_error_dynamics_accounts_for_nonzero_subspace_leakage():
    system,_,_ = problem()
    space = system.error_space(2)
    a,b = space.slow[:,0].copy(),space.tail[:,0].copy()
    theta = .1
    space.slow[:,0] = np.cos(theta)*a+np.sin(theta)*b
    space.tail[:,0] = -np.sin(theta)*a+np.cos(theta)*b
    space.configure_bounds(system)
    assert space.coupling > 1e-5
    basis = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,3))
    model = PortGradientModel(system,basis,certificate_method="spectral",error_modes=2)
    c = np.zeros(system.n);state = model.initialize(c,error_tolerance=1e3)
    for k in range(20):
        u = [1e-5,2e-5*np.sin(k)]
        c = system.full_step(c,.1,u)
        state,_ = model.step(state,.1,u,error_tolerance=1e3)
        assert system.norm(c-basis.values @ state.coefficients) <= state.error.total_bound+1e-8


@pytest.mark.parametrize("certificate",["graph","spectral"])
def test_non_nested_current_state_transfer_preserves_physics_and_certificate(certificate):
    from morqex.concentration import _orthogonal
    system,_,_ = problem()
    initial_basis = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,4))
    old = PortGradientModel(system,initial_basis,certificate_method=certificate,error_modes=2)
    initial = np.random.default_rng(193).normal(size=system.n)
    state = old.initialize(initial,error_tolerance=100.)
    seed = system.conservative_basis(ConcentrationBasis.spectral(system.diffusion,1))
    current = initial_basis.values @ state.coefficients
    values = _orthogonal(np.column_stack([current,np.arange(system.n)]),system.weights,initial=seed.values)
    new = PortGradientModel(system,ConcentrationBasis(values,system.weights,"non-nested",1),certificate_method=certificate,error_modes=2)
    transferred,info = old.transfer_preserving_state(state,new)
    assert transferred.error.total_bound > 1e-3
    np.testing.assert_allclose(new.basis.values @ transferred.coefficients,current,atol=1e-12)
    assert abs(info["energy_jump"]) < 1e-9 and max(abs(x) for x in info["mass_jump"]) < 1e-25
    assert system.norm(initial-new.basis.values @ transferred.coefficients) <= transferred.error.total_bound+1e-10
