import os
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from morqex.mechanics import TetMesh
from morqex.microstructure import Volume
from morqex.top_damage.backend import NativeConfig, MFEMScalarBackend
from morqex.top_damage.phase_field import AT2Damage, FatigueHistory
from morqex.top_damage.coupled_reference import SplitElasticity, staggered_step
from morqex.top_damage.assembly_cache import CachedP1ScalarBackend


@pytest.fixture(scope='module')
def backend():
    ws = os.environ.get('MORQEX_WORKSPACE')
    if not ws:
        pytest.skip('Native tests require the external resource-limited workspace')
    provider = os.environ.get('MORQEX_TEST_MFEM_PROVIDER', 'vendored')
    return MFEMScalarBackend(NativeConfig(provider=provider), ws)


def cube(n=3):
    return TetMesh.from_volume(Volume(np.ones((n,n,n), dtype=np.uint8), spacing=(1/n,)*3))


def independent_form(mesh, reaction, tensor, source):
    e = len(mesh.cells); n = len(mesh.points)
    a = np.broadcast_to(reaction, (e,)); D = np.broadcast_to(tensor, (e,3,3))
    f = np.broadcast_to(source, (e,))
    local = mesh.volumes[:,None,None]*(a[:,None,None]*(np.ones((4,4))+np.eye(4))/20
              +np.einsum('eai,eij,ebj->eab', mesh.gradients,D,mesh.gradients))
    rows = np.broadcast_to(mesh.cells[:,:,None], local.shape)
    cols = np.broadcast_to(mesh.cells[:,None,:], local.shape)
    A = coo_matrix((local.ravel(),(rows.ravel(),cols.ravel())),shape=(n,n)).tocsr()
    b = np.zeros(n)
    for i in range(4): np.add.at(b,mesh.cells[:,i],mesh.volumes*f/4)
    return A,b


def test_compiled_tensor_form_matches_independent_integrals(backend):
    mesh=cube(2);rng=np.random.default_rng(42);e=len(mesh.cells)
    Q=rng.normal(size=(e,3,3));D=Q.swapaxes(1,2)@Q+np.eye(3)
    a=rng.uniform(1,5,e);f=rng.normal(size=e)
    A,b=backend.assemble(mesh,a,D,f);exact,rhs=independent_form(mesh,a,D,f)
    np.testing.assert_allclose(A.toarray(),exact.toarray(),atol=2e-14,rtol=2e-14)
    np.testing.assert_allclose(b,rhs,atol=1e-15)
    np.testing.assert_allclose(A.toarray(),A.T.toarray(),atol=1e-14)
    cached=CachedP1ScalarBackend(mesh,backend)
    for scale in (.1,1.,3.):
        fast,fb=cached.assemble(mesh,a*scale,D*scale,f/scale)
        np.testing.assert_allclose(fast.toarray(),exact.toarray()*scale,atol=1e-13,rtol=2e-13)
        np.testing.assert_allclose(fb,rhs/scale,atol=1e-14)


def test_vertex_permutation_and_constant_diffusion_nullspace(backend):
    mesh=cube(2);rng=np.random.default_rng(9);order=rng.permutation(len(mesh.points));inv=np.argsort(order)
    p=TetMesh(mesh.points[order],inv[mesh.cells],mesh.phases,mesh.box)
    D=np.array([[2,.4,.2],[.4,1,.1],[.2,.1,3.]])
    A,b=backend.assemble(mesh,0,D,0);B,_=backend.assemble(p,0,D,0)
    np.testing.assert_allclose(B.toarray(),A.toarray()[np.ix_(order,order)],atol=2e-14)
    np.testing.assert_allclose(A@np.ones(A.shape[0]),0,atol=2e-14)
    interior=np.setdiff1d(np.arange(len(mesh.points)),mesh.outer_nodes())
    np.testing.assert_allclose((A@mesh.points[:,0])[interior],0,atol=2e-14)


def test_uniform_damage_analytic_and_unloading_irreversibility(backend):
    mesh=cube();model=AT2Damage(mesh,backend,2.,.2)
    psi=3.;s=model.solve(psi,np.zeros(len(mesh.points)))
    exact=2*(1-model.k)*psi/(2*(1-model.k)*psi+2/.2)
    np.testing.assert_allclose(s.damage,exact,atol=2e-13)
    assert s.projected_kkt<1e-12
    unload=model.solve(0.,s.damage)
    np.testing.assert_allclose(unload.damage,s.damage,atol=2e-13)
    assert s.objective_change_J<0


def test_variable_toughness_energy_derivative(backend):
    mesh=cube(2);rng=np.random.default_rng(15);e=len(mesh.cells)
    model=AT2Damage(mesh,backend,2+mesh.centers[:,0],.12)
    psi=rng.uniform(.1,3,e);factor=.2+.8*rng.random(e)
    d=rng.uniform(.15,.6,len(mesh.points));direction=rng.normal(size=len(d))
    A,b=model.assemble(psi,factor);h=1e-6
    fd=(sum(model.energies(d+h*direction,psi,factor))-sum(model.energies(d-h*direction,psi,factor)))/(2*h)
    assert fd==pytest.approx(direction@(A@d-b),rel=2e-8,abs=1e-9)


def test_precrack_profile_mesh_convergence(backend):
    errors=[];energies=[];ell=.2
    for n in (8,16,32):
        mesh=TetMesh.from_volume(Volume(np.ones((2,2,n),dtype=np.uint8),spacing=(.1,.1,1/n)))
        model=AT2Damage(mesh,backend,1.,ell)
        lower=np.zeros(len(mesh.points));lower[mesh.points[:,0]==0]=1
        s=model.solve(0.,lower)
        exact=np.cosh((1-mesh.points[:,0])/ell)/np.cosh(1/ell)
        errors.append(float(np.linalg.norm(s.damage-exact)/np.linalg.norm(exact)))
        energies.append(s.fracture_energy_J)
        assert s.projected_kkt<1e-9
    assert errors[2]<errors[1]<errors[0]
    assert errors[2]<.02
    exact_energy=.5*.2*.2*np.tanh(1/ell)
    assert abs(energies[2]/exact_energy-1)<.01


def test_fatigue_only_positive_variation_and_trial_rollback():
    accepted=FatigueHistory.zero(3)
    first=accepted.trial([1,2,3]);retry=accepted.trial([1,2,3])
    np.testing.assert_array_equal(first.accumulated_J_m3,retry.accumulated_J_m3)
    np.testing.assert_array_equal(accepted.accumulated_J_m3,0)
    unloaded=first.trial([0,1,2]);np.testing.assert_array_equal(unloaded.accumulated_J_m3,[1,2,3])
    reloaded=unloaded.trial([1,2,3]);np.testing.assert_array_equal(reloaded.accumulated_J_m3,[2,3,4])
    np.testing.assert_allclose(reloaded.toughness_factor(2),[1,(4/5)**2,(4/6)**2])
    with pytest.raises(ValueError): reloaded.accumulated_J_m3[0]=4


@pytest.mark.parametrize('bad',[0,-1,np.nan])
def test_invalid_fatigue_threshold(bad):
    with pytest.raises(ValueError): FatigueHistory.zero(2).toughness_factor(bad)


def test_negative_transport_and_invalid_damage_rejected(backend):
    mesh=cube(1)
    with pytest.raises(ValueError): backend.assemble(mesh,1,np.diag([1,1,-1]),0)
    with pytest.raises(ValueError): AT2Damage(mesh,backend,1.,.2).solve(-1.,np.zeros(len(mesh.points)))
    with pytest.raises(ValueError): AT2Damage(mesh,backend,1.,.2).solve(1.,np.full(len(mesh.points),1.1))


@pytest.mark.parametrize('kwargs',[{'jobs':2},{'device':'cuda'},{'precision':'float32'},{'slice_seconds':60}])
def test_build_profile_cannot_silently_exceed_qualification(kwargs):
    with pytest.raises(ValueError): NativeConfig(**kwargs)


def test_split_constitutive_gradient_tangent_and_compression(backend):
    mesh=cube(1);model=AT2Damage(mesh,backend,2,.2)
    mechanics=SplitElasticity(model,100.,.3)
    rng=np.random.default_rng(25)
    e=rng.normal(size=(len(mesh.cells),6))*.01;g=np.full(len(e),.4)
    direction=rng.normal(size=e.shape);h=1e-7
    stress,C,_,energy=mechanics.constitutive(e,g)
    sp,_,_,ep=mechanics.constitutive(e+h*direction,g)
    sm,_,_,em=mechanics.constitutive(e-h*direction,g)
    np.testing.assert_allclose((ep-em)/(2*h),np.sum(stress*direction,axis=1),rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose((sp-sm)/(2*h),np.einsum('eij,ej->ei',C,direction),rtol=1e-8,atol=1e-8)
    compression=np.broadcast_to([-.01,-.01,-.01,0,0,0],e.shape)
    damaged,_,plus,_=mechanics.constitutive(compression,g)
    intact,_,_,_=mechanics.constitutive(compression,np.ones(len(e)))
    np.testing.assert_allclose(damaged,intact,atol=1e-13)
    np.testing.assert_allclose(plus,0,atol=1e-30)


@pytest.mark.parametrize('boundary',['kubc','periodic','neumann'])
def test_split_mechanics_recovers_intact_solver_and_free_eigenstrain(backend,boundary):
    mesh=cube(2);model=AT2Damage(mesh,backend,2,.2)
    mechanics=SplitElasticity(model,100.,.25,boundary=boundary)
    d=np.zeros(len(mesh.points));eps=np.array([.01,.005,.002,.001,0,0])
    kw={} if boundary=='neumann' else {'macro_strain':np.zeros(6)}
    computed=mechanics.solve(d,eps,**kw);reference=mechanics.base.solve(eps,**kw)
    np.testing.assert_allclose(computed.mean_cell_stress_Pa,reference.stress,atol=1e-10)
    if boundary=='neumann':assert computed.elastic_energy_J<1e-20
    else:assert computed.elastic_energy_J==pytest.approx(reference.elastic_energy,rel=1e-10)


def test_staggered_uniform_solution_history_and_failed_trial_rollback(backend):
    mesh=cube(2);model=AT2Damage(mesh,backend,2,.2)
    mechanics=SplitElasticity(model,100.,.25)
    d=np.zeros(len(mesh.points));history=FatigueHistory.zero(len(mesh.cells))
    macro=np.array([.01,.01,.01,0,0,0])
    with pytest.raises(RuntimeError):
        staggered_step(mechanics,d,history,macro_strain=macro,max_iterations=1)
    np.testing.assert_array_equal(d,0);np.testing.assert_array_equal(history.accumulated_J_m3,0)
    result=staggered_step(mechanics,d,history,macro_strain=macro)
    psi=.5*mechanics.bulk[0]*.03**2
    exact=2*(1-model.k)*psi/(2*(1-model.k)*psi+2/.2)
    np.testing.assert_allclose(result.damage,exact,atol=1e-10)
    assert result.damage_kkt<1e-9
    assert np.max(np.diff(result.fixed_history_energy_trace_J))<=1e-12
    unload=staggered_step(mechanics,result.damage,result.fatigue,macro_strain=np.zeros(6))
    np.testing.assert_allclose(unload.damage,result.damage,atol=1e-9)
    np.testing.assert_allclose(unload.fatigue.accumulated_J_m3,result.fatigue.accumulated_J_m3,atol=1e-13)
