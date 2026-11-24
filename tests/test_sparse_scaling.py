import numpy as np
import pytest
from morqex.sparse_coupled import SparseChemoelasticSystem
from morqex.streamed_rom import StreamedEnergyROM,StressLocalization
from morqex.concentration import ConcentrationBasis
from test_gradient_adaptive import problem


def systems(beta=1e-5,separated=False):
    dense,identity,patterns=problem(beta,separated)
    sparse=SparseChemoelasticSystem(build_solver(dense),dense.diffusion,
        beta*np.array([1.,.6,1.4,0.,0.,0.]),patterns,chemical_curvature=.1,reference_concentration=20000.,maximum_concentration=40000.)
    return dense,sparse,identity,patterns


def build_solver(dense):
    from morqex.mechanics import TetMesh,ElasticitySolver,isotropic_stiffness
    mesh=TetMesh.from_volume(dense.diffusion.volume,solid_phases=[1,2])
    tensors={1:isotropic_stiffness(1e9,.25),2:isotropic_stiffness(2e8,.3)}
    return ElasticitySolver(mesh,np.array([tensors[int(p)] for p in mesh.phases]))


@pytest.mark.parametrize('beta',[0.,1e-5])
def test_sparse_hessian_actions_upper_bound_and_macro_derivatives(beta):
    dense,sparse,identity,_=systems(beta)
    rng=np.random.default_rng(84);c=rng.normal(size=sparse.n)*10;e=rng.normal(size=6)*1e-4
    z=sparse.scale_c*c
    np.testing.assert_allclose(sparse.scale_c*sparse.h_action(z),dense.H @ c,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(sparse.scale_c[:,None]*sparse.G,dense.G,rtol=1e-10,atol=1e-10)
    state=sparse.initialize(c,macro_strain=e)
    assert state.energy == pytest.approx(dense.energy(c,e),rel=1e-11,abs=1e-9)
    full=sparse.fields(c,e)
    assert state.energy == pytest.approx(full.elastic_energy/sparse.diffusion.rve_volume+.5*.1*np.dot(sparse.weights*c,c),rel=1e-10)
    for _ in range(5):
        v=rng.normal(size=sparse.n)
        assert v @ sparse.h_action(v) <= np.dot(sparse.local_upper*v,v)+1e-10
        assert v @ sparse.h_action(v) >= v @ v-1e-10


@pytest.mark.parametrize('method',['mixed-LU','Schur-GMRES'])
def test_sparse_coupled_steps_match_independent_dense_and_energy_ledger(method):
    dense,sparse,_,_=systems(separated=True)
    c=np.random.default_rng(8).normal(size=sparse.n)*10;state=sparse.initialize(c)
    for k in range(10):
        u=np.array([1e-5,2e-5*np.sin(k)]);e=np.array([1e-4*np.sin(k),0,0,0,0,0])
        c=dense.full_step(c,.5,u,macro_strain=e)
        state,d=sparse.step(state,.5,u,macro_strain=e,method=method,relative_tolerance=1e-12)
        np.testing.assert_allclose(state.deviation,c,rtol=1e-8,atol=1e-8)
        assert abs(d['energy_balance']) < 1e-7
        assert np.max(np.abs(d['component_mass_error'])) < 1e-24
    assert sparse.mixed_factorizations+sparse.krylov_factorizations == 1


@pytest.mark.parametrize('projection',['energy','mass'])
def test_streamed_rom_matches_existing_dense_rom_same_basis(projection):
    from morqex.gradient_transport import PortGradientModel
    dense,sparse,_,_=systems(separated=True)
    basis=ConcentrationBasis.spectral(sparse.diffusion,6)
    rom=StreamedEnergyROM(sparse,basis,projection=projection)
    old=PortGradientModel(dense,rom.basis,projection=projection)
    np.testing.assert_allclose(rom.H,old.H,atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(rom.mobility,old.mobility,atol=1e-10,rtol=1e-10)
    new=rom.initialize(np.zeros(sparse.n),sparse,error_tolerance=100.)
    previous=old.initialize(np.zeros(sparse.n),error_tolerance=100.)
    for k in range(5):
        u=[1e-5,2e-5];e=np.array([1e-4*k,0,0,0,0,0])
        new,d=rom.step(new,.5,u,macro_strain=e,error_tolerance=100.)
        previous,_=old.step(previous,.5,u,macro_strain=e,error_tolerance=100.)
        np.testing.assert_allclose(new.coefficients,previous.coefficients,atol=1e-8,rtol=1e-10)
        assert abs(d['energy_balance']) < 1e-7
        assert np.max(np.abs(d['component_mass_error'])) < 1e-24


def test_streamed_whole_error_bound_covers_initial_error_and_trajectory():
    _,system,_,_=systems()
    rom=StreamedEnergyROM(system,ConcentrationBasis.spectral(system.diffusion,3))
    c=np.random.default_rng(83).normal(size=system.n)*10
    state=rom.initialize(c,system,error_tolerance=100.)
    full=system.initialize(c)
    for k in range(15):
        u=[1e-5,2e-5*np.cos(k)]
        state,_=rom.step(state,.5,u,error_tolerance=100.)
        full,_=system.step(full,.5,u,method='mixed-LU')
        error=system.scale_c*(full.deviation-rom.basis.values @ state.coefficients)
        actual=np.sqrt(error @ system.h_action(error))
        assert actual <= state.error_bound+1e-9


def test_online_no_full_actions_and_optional_localization(monkeypatch):
    _,system,_,_=systems()
    rom=StreamedEnergyROM(system,ConcentrationBasis.spectral(system.diffusion,4))
    state=rom.initialize(np.zeros(system.n),system,error_tolerance=100.)
    selected=np.arange(len(system.solver.mesh.cells))[::3]
    local=StressLocalization(system,rom,selected)
    with pytest.raises(MemoryError): StressLocalization(system,rom,selected,maximum_bytes=1)
    def forbidden(*args,**kwargs): raise AssertionError('Full-grid work in online kernel')
    with monkeypatch.context() as m:
        m.setattr(system,'h_action',forbidden);m.setattr(system,'fields',forbidden);m.setattr(system,'solve_mechanical',forbidden)
        state,_=rom.step(state,.5,[1e-5,2e-5],error_tolerance=100.)
        batch=rom.batch_macro(np.tile(state.coefficients,(10,1)),np.tile(state.macro_strain,(10,1)))
    np.testing.assert_allclose(batch,np.tile(rom.macro_stress(state),(10,1)))
    truth=system.fields(rom.basis.values @ state.coefficients,state.macro_strain)
    stress=local.stress(state)
    np.testing.assert_allclose(stress,truth.stress[selected],atol=1e-6,rtol=1e-9)
    mean,variance,vm2=local.moments(state)
    w=local.volumes/local.volumes.sum()
    np.testing.assert_allclose(mean,w @ stress,atol=1e-8)
    np.testing.assert_allclose(variance,w @ (stress-mean)**2,atol=1e-2,rtol=1e-9)
    dev=stress.copy();dev[:,:3]-=stress[:,:3].mean(axis=1)[:,None]
    assert vm2 == pytest.approx(1.5*np.dot(w,np.sum(dev**2,axis=1)),rel=1e-10)


def test_streamed_certificate_rejects_accuracy_and_physical_failures():
    from morqex.gradient_transport import EnergyAccuracyError
    from morqex.transport import ConcentrationInfeasible
    from dataclasses import replace
    _,system,_,_=systems()
    rom=StreamedEnergyROM(system,ConcentrationBasis.spectral(system.diffusion,2))
    state=rom.initialize(np.zeros(system.n),system,error_tolerance=100.)
    with pytest.raises(EnergyAccuracyError): rom.step(state,.5,[1e-5,2e-5],error_tolerance=1e-15)
    with pytest.raises(ConcentrationInfeasible):rom.accept(replace(state,error_bound=1e6),1e7)
    with pytest.raises(ValueError):rom.accept(replace(state,fingerprint='wrong'),1.)
    with pytest.raises(ValueError):system.step(replace(system.initialize(np.zeros(system.n)),system_fingerprint='wrong'),.5,[0.,0.])


def test_detached_kernel_external_roundtrip_and_independent_restart(tmp_path):
    from morqex import Workspace
    from morqex.streamed_rom import OnlineEnergyKernel
    _,system,_,_=systems()
    rom=StreamedEnergyROM(system,ConcentrationBasis.spectral(system.diffusion,4))
    state=rom.initialize(np.zeros(system.n),system,error_tolerance=100.)
    kernel=rom.detach_online()
    assert not hasattr(kernel,'basis') and not hasattr(kernel,'system') and not hasattr(kernel,'offline_hessian_columns')
    state,_=kernel.step(state,.5,[1e-5,2e-5],error_tolerance=100.)
    path=kernel.save(Workspace(tmp_path),'compact',state)
    loaded,restored=OnlineEnergyKernel.load(path)
    a,_=kernel.step(state,.5,[1e-5,2e-5],error_tolerance=100.)
    b,_=loaded.step(restored,.5,[1e-5,2e-5],error_tolerance=100.)
    np.testing.assert_array_equal(a.coefficients,b.coefficients)
    assert a.error_bound == b.error_bound and loaded.online_array_bytes() == kernel.online_array_bytes()
    data=path/'kernel.npz';data.write_bytes(data.read_bytes()+b'x')
    with pytest.raises(ValueError): OnlineEnergyKernel.load(path)


def test_disconnected_mass_modes_are_required_and_macro_lifting_is_exact():
    _,system,_,_=systems(separated=True)
    basis=ConcentrationBasis.spectral(system.diffusion,2)
    with pytest.raises(ValueError):StreamedEnergyROM(system,basis,add_mass_riesz=False,add_macro_lifting=False)
    rom=StreamedEnergyROM(system,basis)
    # Effort of any declared macro strain lies in the H test space after lifting.
    vz=rom.offline_normalized_basis;hz=rom.offline_hessian_columns
    w=np.linalg.solve(rom.H,hz.T).T
    np.testing.assert_allclose(w @ (vz.T @ system.G),system.G,atol=1e-8,rtol=1e-9)
    bad=ConcentrationBasis(basis.values*2,basis.weights,'not-orthonormal',2)
    with pytest.raises(ValueError):StreamedEnergyROM(system,bad)


def test_detached_quadratic_statistics_need_no_micro_field():
    _,system,_,_=systems()
    rom=StreamedEnergyROM(system,ConcentrationBasis.spectral(system.diffusion,4))
    state=rom.initialize(np.zeros(system.n),system,error_tolerance=100.)
    state,_=rom.step(state,.5,[1e-5,2e-5],error_tolerance=100.)
    loc=StressLocalization(system,rom,np.arange(len(system.solver.mesh.cells)))
    stats=loc.detach_statistics()
    assert not hasattr(stats,'operator') and not hasattr(stats,'volumes')
    for a,b in zip(stats.moments(state),loc.moments(state)):np.testing.assert_allclose(a,b)


def test_coupled_resolvent_contains_declared_full_transfer_directions():
    dense,system,_,_=systems()
    basis,work=system.coupled_resolvent_basis(3,shifts=(.3,),moments=1)
    transfer=np.linalg.solve(.3*np.eye(system.n)+dense.operator,dense.b)
    reconstructed=basis.decode(basis.encode(transfer))
    np.testing.assert_allclose(reconstructed,transfer,rtol=1e-9,atol=1e-5)
    assert work['mixed_factorizations']==1 and work['right_hand_sides']==2


def test_periodic_full_anisotropy_and_shear_expansion_match_independent_fem():
    from morqex.microstructure import Volume
    from morqex.mechanics import TetMesh,ElasticitySolver
    from morqex.transport import VoxelDiffusion
    labels=np.ones((4,4,4),dtype=np.uint8);labels[1,1,1]=0
    volume=Volume(labels,spacing=(1e-6,)*3)
    diffusion=VoxelDiffusion(volume,active_phases=[1],diffusivity=[1e-14,2e-14,3e-14],reactive_phases=[0])
    mesh=TetMesh.from_volume(volume,solid_phases=[1])
    a=np.random.default_rng(136).normal(size=(6,6));stiffness=(a.T @ a+np.eye(6))*1e8
    solver=ElasticitySolver(mesh,stiffness,boundary='periodic')
    beta=np.array([1.,.4,1.6,.2,-.3,.1])*1e-6
    patterns=np.column_stack([diffusion.reactive,diffusion.reactive*(diffusion.face_normals[:,0] > .5)]).astype(float)
    system=SparseChemoelasticSystem(solver,diffusion,beta,patterns,chemical_curvature=.1,reference_concentration=20000.,maximum_concentration=40000.)
    c=np.random.default_rng(95).normal(size=system.n)*100;e=np.array([1.,2.,3.,-.4,.6,.9])*1e-4
    state=system.initialize(c,macro_strain=e);fem=system.fields(c,e)
    assert state.energy == pytest.approx(fem.elastic_energy/mesh.rve_volume+.05*np.dot(system.weights*c,c),rel=1e-10,abs=1e-8)
    rom=StreamedEnergyROM(system,ConcentrationBasis.spectral(diffusion,6))
    checked=rom.initialize(np.zeros(system.n),system,error_tolerance=100.)
    checked,_=rom.step(checked,.5,[1e-5,2e-5],macro_strain=e,error_tolerance=100.)
    direct=system.fields(rom.basis.values @ checked.coefficients,e)
    np.testing.assert_allclose(rom.macro_stress(checked),direct.mean_stress,rtol=1e-9,atol=1e-5)
