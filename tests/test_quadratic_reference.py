import os,math
import numpy as np
import pytest
from morqex import Workspace
from morqex.mechanics import TetMesh
from morqex.microstructure import Volume
from morqex.top_damage.backend import NativeConfig,MFEMScalarBackend
from morqex.top_damage.refinement import RefinementBackend
from morqex.top_damage.hyperelastic import HyperelasticBackend
from morqex.top_damage.assembly_cache import CachedP1ScalarBackend
from morqex.top_damage.phase_field import AT2Damage,FatigueHistory
from morqex.top_damage.quadratic_reference import QuadraticGeometry,QuadraticHyperelasticity,QuadratureAT2,quadratic_staggered_step


@pytest.fixture(scope='module')
def setup():
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);cfg=NativeConfig()
    base=TetMesh.from_volume(Volume(np.ones((2,2,2),np.uint8),spacing=(.5e-6,)*3))
    refiner=RefinementBackend(cfg,ws);geo=QuadraticGeometry(base,refiner)
    hyper=HyperelasticBackend(cfg,ws)
    scalar=CachedP1ScalarBackend(base,MFEMScalarBackend(cfg,ws))
    return base,geo,hyper,refiner,scalar


def test_upstream_quadrature_moments_and_quadratic_gradient(setup):
    base,g,_,_,_=setup
    for a in range(6):
        for b in range(6-a):
            for c in range(6-a-b):
                actual=g.weights@(g.bary[:,1]**a*g.bary[:,2]**b*g.bary[:,3]**c)
                exact=6*math.factorial(a)*math.factorial(b)*math.factorial(c)/math.factorial(a+b+c+3)
                assert actual==pytest.approx(exact,rel=1e-13)
    x=g.points/1e-6;u=1e-6*np.column_stack([x[:,0]**2,x[:,0]*x[:,1],x[:,2]**2])
    gradient=np.einsum('eai,eqaj->eqij',u[g.cells],g.gradients_for_cells(0,len(base.cells))).reshape(-1,3,3)
    qx=np.einsum('qa,eai->eqi',g.bary,base.points[base.cells]).reshape(-1,3)/1e-6
    exact=np.zeros_like(gradient);exact[:,0,0]=2*qx[:,0];exact[:,1,0]=qx[:,1];exact[:,1,1]=qx[:,0];exact[:,2,2]=2*qx[:,2]
    np.testing.assert_allclose(gradient,exact,atol=2e-14)
    assert g.volumes.sum()==pytest.approx(base.volumes.sum(),rel=1e-13)
    assert g.lor.boundary()[1].sum()==pytest.approx(base.boundary()[1].sum(),rel=1e-13)


def test_quadratic_energy_force_tangent_and_heterogeneous_equilibrium(setup):
    base,g,hyper,refiner,_=setup
    E=np.where(base.centers[:,0]<.5e-6,93e9,3e9)
    model=QuadraticHyperelasticity(g,hyper,refiner,E,.3)
    x=g.points/1e-6;u=1e-8*np.column_stack([x[:,0]**2,x[:,0]*x[:,1],x[:,2]**2]).ravel()
    fc=np.diag([1.01,1.002,.998]);w,f,A,*_=model.assemble(u,fc)
    rng=np.random.default_rng(73);v=rng.normal(size=u.shape)*1e-8;eps=2e-5
    plus=model.assemble(u+eps*v,fc,tangent=False);minus=model.assemble(u-eps*v,fc,tangent=False)
    assert (plus[0]-minus[0])/(2*eps)==pytest.approx(f@v,rel=2e-6)
    np.testing.assert_allclose((plus[1]-minus[1])/(2*eps),A@v,rtol=2e-6,atol=np.linalg.norm(A@v)*1e-8)
    z=rng.normal(size=len(v));assert z@(A@v)==pytest.approx(v@(A@z),rel=1e-12)
    solved=model.solve(fc)
    assert solved.equilibrium_residual<1e-9
    free=model.P.T@solved.reaction_N.ravel()
    assert np.linalg.norm(free)<solved.equilibrium_force_scale_N*1e-9


def test_quadrature_damage_matches_exact_p1_for_constant_driving_and_energy_derivative(setup):
    base,g,_,_,scalar=setup
    q=QuadratureAT2(g,scalar,10.,.5e-6);linear=AT2Damage(base,scalar,10.,.5e-6)
    A,b=q.assemble(2e6);A0,b0=linear.assemble(2e6)
    assert np.max(np.abs((A-A0).data),initial=0)<1e-12*np.max(np.abs(A0.data))
    np.testing.assert_allclose(b,b0,rtol=1e-12)
    rng=np.random.default_rng(3);d=rng.uniform(.1,.3,len(base.points));v=rng.normal(size=len(d));psi=rng.uniform(1e6,3e6,g.quadrature_count)
    A,b=q.assemble(psi);eps=1e-6
    fd=(sum(q.energies(d+eps*v,psi))-sum(q.energies(d-eps*v,psi)))/(2*eps)
    assert fd==pytest.approx((A@d-b)@v,rel=1e-7)


def test_quadratic_coupled_fatigue_commits_once(setup):
    base,g,hyper,refiner,scalar=setup
    mechanics=QuadraticHyperelasticity(g,hyper,refiner,93e9,.3)
    damage=QuadratureAT2(g,scalar,10.,.5e-6);history=FatigueHistory.zero(g.quadrature_count)
    result=quadratic_staggered_step(mechanics,damage,np.zeros(len(base.points)),history,np.diag([1.01,1.002,1.005]),fatigue=True)
    assert result.damage.max()>0 and result.damage_kkt<1e-7
    expected=history.trial(damage.mean_degradation(result.damage)*result.mechanics.tensile_energy_J_m3)
    np.testing.assert_array_equal(result.fatigue.accumulated_J_m3,expected.accumulated_J_m3)
    np.testing.assert_array_equal(history.accumulated_J_m3,0.)
    assert np.diff(result.fixed_history_energy_trace_J).max(initial=0)<1e-8*result.fixed_history_energy_trace_J[0]


def test_nested_quadratic_transfer_and_variation_marking(setup):
    from morqex.top_damage.quadratic_reference import prolong_quadratic_field
    from morqex.top_damage.refinement_indicator import quadratic_indicator
    base,g,_,refiner,_=setup
    refined=refiner.refine(base,np.arange(0,len(base.cells),5));target=QuadraticGeometry(refined.mesh,refiner)
    def field(x):
        x=x/1e-6
        return np.column_stack([x[:,0]**2+2*x[:,1],x[:,1]*x[:,2],1+x[:,2]**2])
    transferred=prolong_quadratic_field(g,target,refined.parent_cells,field(g.points))
    np.testing.assert_allclose(transferred,field(target.points),atol=4e-14)
    psi=2e6;Gc=5.;ell=2e-6;k=1e-5
    value=2*(1-k)*psi/(2*(1-k)*psi+Gc/ell)
    P=np.broadcast_to(np.eye(3)*2e8,(g.quadrature_count,3,3)).copy()
    result=quadratic_indicator(g,P,np.full(len(base.points),value),np.full(g.quadrature_count,psi),90e9,Gc,ell,k)
    assert result['total_J'].sum()<1e-36
    oscillation=g.bary[:,0]-g.weights@g.bary[:,0]
    P[:g.nq,0,0]+=1e8*oscillation
    changed=quadratic_indicator(g,P,np.full(len(base.points),value),np.full(g.quadrature_count,psi),90e9,Gc,ell,k)
    assert changed['mechanical_variation_J'][0]>0
    assert np.count_nonzero(changed['mechanical_variation_J']>1e-35)==1
