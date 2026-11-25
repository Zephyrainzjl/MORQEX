import numpy as np
import pytest
from morqex.acceleration.voxel import NativeStencil
from morqex.mechanics import isotropic_stiffness

def test_native_buffers_reject_wrong_shapes_and_normalize_strided_input():
    phase=np.ones((4,4,4),np.uint8);C=np.stack([np.zeros((6,6)),isotropic_stiffness(100.,.2)]);model=NativeStencil(phase,C)
    source=np.arange(3*4**3,dtype=float).reshape(4,4,4,3).transpose(3,0,1,2)
    np.testing.assert_array_equal(model.apply(source),model.apply(source.copy()))
    with pytest.raises(ValueError,match='shape'):model.apply(np.zeros((3,4,4,3)))
    with pytest.raises(ValueError,match='grid'):model.load(np.zeros((3,3,3)))
    with pytest.raises(ValueError,match='FP64'):model.load(np.zeros(phase.shape),out=np.empty((3,*phase.shape),np.float32))

def test_compiled_voxel_energy_adjoint_and_positive_semidefinite():
    rng=np.random.default_rng(27);p=rng.integers(0,3,(4,4,4),dtype=np.uint8)
    C=np.stack([np.zeros((6,6)),isotropic_stiffness(100,.23),isotropic_stiffness(10,.3)])
    model=NativeStencil(p,C,spacing=(1.,2.,3.));u=rng.standard_normal((3,4,4,4));v=rng.standard_normal(u.shape)
    Au=model.apply(u);Av=model.apply(v)
    np.testing.assert_allclose(np.vdot(u,Av),np.vdot(v,Au),rtol=1e-12,atol=1e-12)
    assert np.vdot(u,Au)>0
    np.testing.assert_allclose(model.apply(np.ones_like(u)),0,atol=1e-14)

def test_periodic_homogeneous_macro_strain_and_free_expansion():
    p=np.ones((4,4,4),dtype=np.uint8);C=np.stack([np.zeros((6,6)),isotropic_stiffness(200,.25)])
    beta=np.zeros((2,6));beta[1,:3]=.01;model=NativeStencil(p,C,beta)
    c=np.ones(p.shape);E=(.01,.01,.01,0,0,0);u,metrics=model.solve(c,E)
    assert metrics['converged'];np.testing.assert_allclose(model.stress(u,c,E),0,atol=1e-12)
    E=(.1,0,0,0,0,0);u,_=model.solve(np.zeros_like(c),E)
    np.testing.assert_allclose(model.stress(u,np.zeros_like(c),E),np.broadcast_to((C[1]@E)[:,None,None,None],(6,*p.shape)),atol=1e-12)

def test_periodic_laminate_longitudinal_modulus():
    p=np.ones((4,4,8),dtype=np.uint8);p[:,:,:4]=2
    C=np.stack([np.zeros((6,6)),isotropic_stiffness(100,.25),isotropic_stiffness(10,.25)])
    model=NativeStencil(p,C);E=(.01,0,0,0,0,0);u,metrics=model.solve(np.zeros(p.shape),E,rtol=1e-10)
    expected=.01/(.5/C[1,0,0]+.5/C[2,0,0]);stress=model.stress(u,np.zeros(p.shape),E)
    np.testing.assert_allclose(stress[0],expected,rtol=1e-9,atol=1e-10)

def test_voxel_checkpoint_recomputes_true_residual_and_rejects_new_load(tmp_path):
    rng=np.random.default_rng(11);p=rng.integers(0,2,(4,4,4),dtype=np.uint8)
    C=np.stack([isotropic_stiffness(1,.25),isotropic_stiffness(100,.25)]);beta=np.zeros((2,6));beta[1,:3]=.001
    model=NativeStencil(p,C,beta);c=rng.random(p.shape)
    _,short=model.solve(c,maxiter=2,checkpoint=tmp_path/'state');assert not short['converged']
    x,full=model.solve(c,rtol=1e-9,checkpoint=tmp_path/'state');assert full['converged']
    rhs=model.load(c);np.testing.assert_allclose(np.linalg.norm(rhs-model.apply(x))/np.linalg.norm(rhs),full['relative_residual'],rtol=1e-5)
    with pytest.raises(ValueError,match='mismatch'):model.solve(c*2,checkpoint=tmp_path/'state')

def test_full_mean_stress_control_includes_lateral_free_expansion():
    from morqex.acceleration.macro_control import StressControlledStencil
    p=np.ones((4,4,4),dtype=np.uint8);C=np.stack([np.zeros((6,6)),isotropic_stiffness(200.,.25)])
    beta=np.zeros((2,6));beta[1,:3]=.01;model=StressControlledStencil(p,C,beta)
    c=np.ones(p.shape);u,metrics=model.solve(c,rtol=1e-10)
    np.testing.assert_allclose(model.macro_strain(u,c),[.01,.01,.01,0,0,0],atol=1e-12)
    np.testing.assert_allclose(model.stress(u,c),0,atol=1e-12)
    model=StressControlledStencil(p,C,beta,stress_Pa=(1.,0,0,0,0,0));u,_=model.solve(c,rtol=1e-10)
    np.testing.assert_allclose(model.stress(u,c).mean(axis=(1,2,3)),[1.,0,0,0,0,0],atol=1e-12)
    assert model.macro_strain(u,c)[1]<.01

def test_anisotropic_periodic_manufactured_displacement_converges_second_order():
    from morqex.acceleration.verification import manufactured_periodic
    rows=[manufactured_periodic(n) for n in [4,8,16]]
    assert max(r['stress_relative_L2'] for r in rows)<1e-8
    assert max(r['relative_energy_error'] for r in rows)<1e-8
    assert 3.8<rows[1]['displacement_relative_L2']/rows[2]['displacement_relative_L2']<4.2
