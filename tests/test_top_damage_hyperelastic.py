"""Finite deformation is checked against independent invariants and derivatives."""
import os
import numpy as np
import pytest
from morqex.top_damage.backend import NativeConfig
from morqex.top_damage.hyperelastic import HyperelasticBackend, Hyperelasticity, chemical_potential
from morqex.mechanics.mesh import TetMesh
from morqex.microstructure import Volume


@pytest.fixture(scope='module')
def backend():
    if not os.environ.get('MORQEX_WORKSPACE'):pytest.skip('Bounded external workspace required')
    return HyperelasticBackend(NativeConfig(),os.environ['MORQEX_WORKSPACE'])


def independent(F,Fc,mu,K,g):
    a=np.linalg.inv(Fc);fe=F@a;J=np.linalg.det(fe);b=J**(-2/3)
    invt=np.linalg.inv(fe).T;I1=np.sum(fe*fe)
    iso=mu/2*(b*I1-3);vp=K/2*max(J-1,0)**2;vm=K/2*min(J-1,0)**2
    pe=g*mu*b*(fe-I1/3*invt)+K*(g if J>=1 else 1)*(J-1)*J*invt
    return g*(iso+vp)+vm,(pe@a.T),iso+vp


def cube(n=2):
    return TetMesh.from_volume(Volume(np.ones((n,n,n),np.uint8),spacing=(1/n,)*3))


def test_mfem_energy_stress_and_consistent_tangent(backend):
    rng=np.random.default_rng(104);F=np.eye(3)+rng.normal(size=(7,3,3))*.09
    fc=np.broadcast_to(np.diag([1.03,1.01,.98]),F.shape)
    mu=np.linspace(2,8,7);K=np.linspace(10,30,7);g=np.linspace(.1,1,7)
    w,plus,p,C=backend.evaluate(F,fc,mu,K,g)
    for i in range(len(F)):
        ew,ep,ex=independent(F[i],fc[i],mu[i],K[i],g[i])
        np.testing.assert_allclose(w[i],ew,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(p[i],ep,rtol=1e-12,atol=1e-13)
        assert plus[i]==pytest.approx(ex,abs=1e-13)
    direction=rng.normal(size=F.shape);h=1e-6
    wp,_,pp,_=backend.evaluate(F+h*direction,fc,mu,K,g,tangent=False)
    wm,_,pm,_=backend.evaluate(F-h*direction,fc,mu,K,g,tangent=False)
    np.testing.assert_allclose((wp-wm)/(2*h),np.sum(p*direction,axis=(1,2)),rtol=2e-8,atol=2e-8)
    np.testing.assert_allclose((pp-pm)/(2*h),np.einsum('eij,ej->ei',C,direction.reshape(-1,9)).reshape(-1,3,3),rtol=2e-8,atol=2e-8)
    np.testing.assert_allclose(C,C.swapaxes(1,2),atol=2e-13)


def test_objectivity_anisotropic_chemical_free_expansion(backend):
    theta=.7;Q=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    fc=Q@np.diag([1.1,1.02,.97])@Q.T
    F=np.array([[[1.15,.1,0],[0,1.08,.04],[.03,0,.91]]])
    w,_,p,_=backend.evaluate(F,fc,3.,9.,.4)
    wr,_,pr,_=backend.evaluate(Q@F,fc,3.,9.,.4)
    np.testing.assert_allclose(w,wr,atol=1e-14)
    np.testing.assert_allclose(pr,Q@p,atol=1e-14)
    free,_,stress,_=backend.evaluate((Q@fc)[None],fc,3.,9.,.4)
    np.testing.assert_allclose(free,0,atol=2e-14)
    np.testing.assert_allclose(stress,0,atol=2e-14)


def test_mechanical_chemical_potential_and_reciprocal_derivative(backend):
    F=np.diag([1.08,1.12,.98])[None];beta=np.diag([.03,-.02,.05]);a=.4
    fc=np.eye(3)+a*beta;h=1e-5
    w,_,p,C=backend.evaluate(F,fc,5.,20.)
    analytic=chemical_potential(p,F,fc,beta)
    wp,_,pp,_=backend.evaluate(F,fc+h*beta,5.,20.)
    wm,_,pm,_=backend.evaluate(F,fc-h*beta,5.,20.)
    np.testing.assert_allclose((wp-wm)/(2*h),analytic,rtol=2e-8,atol=1e-9)
    direction=np.arange(9).reshape(1,3,3)/10
    _,_,pplus,_=backend.evaluate(F+h*direction,fc,5.,20.)
    _,_,pminus,_=backend.evaluate(F-h*direction,fc,5.,20.)
    dg=(chemical_potential(pplus,F+h*direction,fc,beta)-chemical_potential(pminus,F-h*direction,fc,beta))/(2*h)
    np.testing.assert_allclose(dg,np.sum((pp-pm)/(2*h)*direction,axis=(1,2)),rtol=1e-7,atol=1e-9)


def test_compression_not_weakened_and_inverted_elements_rejected(backend):
    F=np.eye(3)[None]*.9
    w,plus,p,_=backend.evaluate(F,np.eye(3),3.,10.,.01)
    wi,_,pi,_=backend.evaluate(F,np.eye(3),3.,10.,1.)
    np.testing.assert_allclose(w,wi,atol=1e-14);np.testing.assert_allclose(p,pi,atol=1e-14)
    np.testing.assert_allclose(plus,0,atol=1e-14)
    for bad in [np.diag([-1,1,1]),np.diag([0,1,1])]:
        with pytest.raises(ValueError):backend.evaluate(bad[None],np.eye(3),3.,10.)
    with pytest.raises(ValueError):backend.evaluate(F,np.diag([-1,1,1]),3.,10.)


@pytest.mark.parametrize('boundary',['kubc','periodic'])
def test_affine_patch_and_reference_power_identity(backend,boundary):
    mesh=cube(2);m=Hyperelasticity(mesh,backend,100.,.3,boundary=boundary)
    F=np.array([[1.12,.08,0],[0,1.03,0],[0,0,.94]])
    state=m.solve(macro_F=F)
    np.testing.assert_allclose(state.deformation_gradient,np.broadcast_to(F,(len(mesh.cells),3,3)),atol=2e-13)
    exact,stress,_=independent(F,np.eye(3),m.shear[0],m.bulk[0],1.)
    assert state.elastic_energy_J==pytest.approx(exact*mesh.volumes.sum(),rel=1e-12)
    variation=np.array([[.02,.03,0],[0,-.01,0],[0,0,.01]])
    du=mesh.points@variation.T
    assert np.sum(state.reaction_N*du)==pytest.approx(np.sum(stress*variation)*mesh.volumes.sum(),rel=1e-12)


def test_assembled_tangent_and_nonuniform_inclusion_equilibrium(backend):
    mesh=cube(3);m=Hyperelasticity(mesh,backend,50+100*(mesh.centers[:,0]>.5),.25)
    fc=np.broadcast_to(np.eye(3),(len(mesh.cells),3,3)).copy();fc[mesh.centers[:,0]>.5]*=1.035
    s=m.solve(fc,tolerance=2e-8)
    assert s.newton_iterations>0 and s.equilibrium_residual<2e-8 and s.minimum_jacobian>0
    rng=np.random.default_rng(2);direction=m.P@rng.normal(size=m.P.shape[1]);direction/=np.linalg.norm(direction)
    h=1e-6;u=s.displacement_m.ravel()
    e,f,A,*_=m.assemble(u,fc)
    ep,fp,*_=m.assemble(u+h*direction,fc,tangent=False)
    em,fm,*_=m.assemble(u-h*direction,fc,tangent=False)
    np.testing.assert_allclose((fp-fm)/(2*h),A@direction,rtol=2e-7,atol=1e-7)
    assert abs((ep-em)/(2*h)-f@direction)<1e-8


def test_free_chemical_expansion_is_zero_stress(backend):
    m=Hyperelasticity(cube(2),backend,100.,.3,boundary='neumann')
    s=m.solve(np.eye(3)*1.04,tolerance=1e-7)
    assert s.elastic_energy_J<1e-12
    assert np.max(np.abs(s.cauchy_stress_Pa))<1e-7


def test_matrix_free_action_and_amg_match_sparse(backend):
    mesh=cube(3);young=np.where(mesh.centers[:,0]<.5,100.,20.)
    sparse=Hyperelasticity(mesh,backend,young,.3)
    fast=Hyperelasticity(mesh,backend,young,.3,linear_solver='matrix-free-amg-minres')
    fc=np.broadcast_to(np.eye(3),(len(mesh.cells),3,3)).copy();fc[:,0,0]+= .03*mesh.centers[:,2]
    a=sparse.solve(fc,tolerance=1e-8);b=fast.solve(fc,tolerance=1e-8)
    np.testing.assert_allclose(a.displacement_m,b.displacement_m,atol=1e-9,rtol=1e-7)
    rng=np.random.default_rng(40);du=rng.normal(size=3*len(mesh.points))
    As=sparse.assemble(a.displacement_m,fc)[2];Af=fast.assemble(a.displacement_m,fc)[2]
    np.testing.assert_allclose(Af@du,As@du,rtol=1e-12,atol=1e-12)


def test_finite_fatigue_acceptance_and_unloading(backend):
    from morqex.top_damage.hyperelastic import finite_staggered_step
    from morqex.top_damage.phase_field import AT2Damage,FatigueHistory
    mesh=cube(2);m=Hyperelasticity(mesh,backend,100.,.25)
    pf=AT2Damage(mesh,backend.base,2.,.2);d=np.zeros(len(mesh.points));history=FatigueHistory.zero(len(mesh.cells))
    F=np.eye(3)*1.08
    with pytest.raises(RuntimeError):finite_staggered_step(m,pf,d,history,macro_F=F,max_iterations=1)
    np.testing.assert_array_equal(d,0);np.testing.assert_array_equal(history.accumulated_J_m3,0)
    s=finite_staggered_step(m,pf,d,history,macro_F=F)
    _,_,plus=independent(F,np.eye(3),m.shear[0],m.bulk[0],1.)
    exact=2*(1-pf.k)*plus/(2*(1-pf.k)*plus+2/.2)
    np.testing.assert_allclose(s.damage,exact,atol=1e-9)
    assert np.max(np.diff(s.fixed_history_energy_trace_J))<1e-12
    unload=finite_staggered_step(m,pf,s.damage,s.fatigue,macro_F=np.eye(3))
    np.testing.assert_allclose(unload.damage,s.damage,atol=1e-8)
    np.testing.assert_allclose(unload.fatigue.accumulated_J_m3,s.fatigue.accumulated_J_m3,atol=1e-10)
