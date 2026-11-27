import os
import numpy as np
import pytest
from scipy.optimize import brentq
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.mechanics import TetMesh
from morqex.top_damage.backend import NativeConfig,MFEMScalarBackend
from morqex.top_damage.hyperelastic import HyperelasticBackend,Hyperelasticity
from morqex.top_damage.series_mechanics import MicrostructureLayer,HomogeneousLayer,solve_fixed_thickness,finite_series_step


@pytest.fixture(scope='module')
def backend():
    return HyperelasticBackend(NativeConfig(),Workspace(os.environ['MORQEX_WORKSPACE']))


def test_two_layer_stack_matches_independent_scalar_equilibrium(backend):
    E=np.array([100.,17.]);nu=.3;natural=np.array([1.02,1.005]);weights=np.array([.3,.7])
    layers=[HomogeneousLayer(backend,e,nu,chemical_F=np.diag([a,1,1])) for e,a in zip(E,natural)]
    def stress(lam,i):
        t=lam/natural[i];mu=E[i]/(2*(1+nu));K=E[i]/(3*(1-2*nu))
        return ((2*mu/3)*(t**(1/3)-t**(-5/3))+K*(t-1))/natural[i]
    root=brentq(lambda x:stress(x,0)-stress((1-weights[0]*x)/weights[1],1),.9,1.1,xtol=1e-14)
    expected=[root,(1-weights[0]*root)/weights[1]]
    state=solve_fixed_thickness(layers,weights)
    np.testing.assert_allclose(state.stretches,expected,rtol=0,atol=1e-11)
    assert state.common_nominal_stress_Pa==pytest.approx(stress(root,0),rel=1e-9)
    assert state.normalized_force_defect<1e-8 and state.mean_stretch_defect<1e-14
    assert abs(state.stretches[0]-state.stretches[1])>.001


def test_micro_condensed_tangent_and_boundary_work(backend):
    mesh=TetMesh.from_volume(Volume(np.ones((3,3,3),np.uint8),spacing=(1e-6,)*3))
    E=np.where(mesh.centers[:,0]<1.5e-6,93e9,1e9)
    m=Hyperelasticity(mesh,backend,E,.3,linear_solver='matrix-free-amg-minres')
    fc=np.broadcast_to(np.eye(3),(len(mesh.cells),3,3)).copy();fc[:,0,0]+= .02*(mesh.centers[:,1]/3e-6)
    layer=MicrostructureLayer(m,fc,1.)
    response=layer.evaluate(.998);eps=1e-6
    numerical=(layer.evaluate(.998+eps).nominal_stress_Pa-layer.evaluate(.998-eps).nominal_stress_Pa)/(2*eps)
    assert response.tangent_Pa==pytest.approx(numerical,rel=2e-6)
    assert response.state.equilibrium_residual<1e-10


def test_joint_fatigue_is_committed_once_after_macro_balance(backend):
    from morqex.top_damage.assembly_cache import CachedP1ScalarBackend
    from morqex.top_damage.phase_field import AT2Damage,FatigueHistory
    mesh=TetMesh.from_volume(Volume(np.ones((2,2,2),np.uint8),spacing=(.5e-6,)*3))
    scalar=CachedP1ScalarBackend(mesh,MFEMScalarBackend(NativeConfig(),os.environ['MORQEX_WORKSPACE']))
    models=[Hyperelasticity(mesh,backend,e,.3) for e in (93e9,5e9)]
    damage=[AT2Damage(mesh,scalar,10.,1e-6) for _ in models]
    initial=[np.zeros(len(mesh.points)) for _ in models]
    histories=[FatigueHistory.zero(len(mesh.cells)) for _ in models]
    fc=[np.diag([1.02,1.01,1.005]),np.diag([1.005,1.002,1.001])]
    result,stack,trace=finite_series_step(models,damage,initial,histories,fc,[.3e-6,.7e-6],tolerance=1e-7)
    assert stack.normalized_force_defect<1e-8
    assert np.max(np.diff(trace),initial=0)<=1e-8*trace[0]
    for model,accepted,old in zip(damage,result,histories):
        assert np.all(accepted.damage>=0) and accepted.damage.max()>0
        exact=old.trial(model.mean_degradation(accepted.damage)*accepted.mechanics.tensile_energy_J_m3)
        np.testing.assert_array_equal(accepted.fatigue.accumulated_J_m3,exact.accumulated_J_m3)
        np.testing.assert_array_equal(old.accumulated_J_m3,0)
        assert accepted.damage_kkt<1e-7
