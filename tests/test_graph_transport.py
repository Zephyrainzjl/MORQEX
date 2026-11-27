import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve
from morqex.top_damage.contracts import ConductingNetwork
from morqex.top_damage.graph_transport import KronOperator, voxel_network, certify_update, laplacian


def graph(g=(2., 3., 5.)):
    return ConductingNetwork(4,np.array([[0,1],[1,2],[2,3]]),np.array(g),np.array([3]),'electronic','analytical chain')


def test_series_resistance_sources_and_power():
    net=graph();op=KronOperator(net,np.array([0,3]));g=1/(1/2+1/3+1/5)
    np.testing.assert_allclose(op.matrix_S,g*np.array([[1,-1],[-1,1]]),atol=1e-14)
    f=np.array([.1,.3,-.2,.4]);result=op.recover([-.4,.8],f)
    L=laplacian(net);u=np.array([-.4,0.,0.,.8])
    u[1:3]=spsolve(L[1:3,1:3],f[1:3]-L[1:3][:,[0,3]]@u[[0,3]])
    np.testing.assert_allclose(result.potential_V,u,atol=1e-14)
    np.testing.assert_allclose(result.terminal_current_A,(L@u-f)[[0,3]],atol=1e-14)
    assert abs(result.power_defect_W)<1e-14
    assert result.relative_residual<1e-14


def test_islands_and_actual_edge_deletion():
    net=ConductingNetwork(6,np.array([[0,1],[1,2],[2,3],[4,5]]),np.array([1.,0.,1.,2.]),np.array([3]),'electronic','split chain')
    op=KronOperator(net,np.array([0,3]));out=op.recover([0.,1.])
    np.testing.assert_allclose(out.terminal_current_A,0.,atol=1e-14)
    np.testing.assert_allclose(out.potential_V,[0,0,1,1,0,0])
    with pytest.raises(ValueError,match='Floating'):
        op.recover([0.,1.],[0,0,0,0,1,-1])
    assert certify_update(graph(),graph([2,0,5])).rebuild


@pytest.mark.parametrize('axis',[0,1,2])
def test_voxel_homogeneous_and_layered_analytical(axis):
    shape=(4,5,6);h=(.7e-6,.9e-6,1.2e-6)
    uniform=voxel_network(np.full(shape,7.),h,axis=axis)
    solved=uniform.homogenize()
    np.testing.assert_allclose(solved['effective_S_m'],7.,rtol=1e-12)
    assert uniform.reservoir_accessible().all()
    k=np.full(shape,2.);s=[slice(None)]*3;s[axis]=slice(shape[axis]//2,None);k[tuple(s)]=10.
    result=voxel_network(k,h,axis=axis).homogenize()
    f=(shape[axis]//2)/shape[axis]
    np.testing.assert_allclose(result['effective_S_m'],1/(f/2+(1-f)/10),rtol=1e-12)


def test_no_fake_reachability_through_homogenization_boundary():
    k=np.zeros((3,3,5));k[0,0,:]=1.;k[2,2,:2]=1.
    v=voxel_network(k,(1.,)*3)
    assert v.reservoir_accessible()[0,0].all()
    assert not v.reservoir_accessible()[2,2].any()
    factors=np.ones((3,)+k.shape);factors[2,:,:,2]=0
    cut=voxel_network(k,(1.,)*3,face_factors=factors)
    assert cut.homogenize()['effective_S_m']<1e-25
    assert not cut.reservoir_accessible()[0,0,:3].any()


def test_schur_certificate_and_accumulated_drift():
    old=graph();new=graph([2.01,2.98,5.04]);certificate=certify_update(old,new,tolerance=.01)
    assert not certificate.rebuild
    S=KronOperator(old,np.array([0,3])).matrix_S;T=KronOperator(new,np.array([0,3])).matrix_S
    assert np.linalg.eigvalsh(T-certificate.response_lower_factor*S).min()>-1e-12
    assert np.linalg.eigvalsh(certificate.response_upper_factor*S-T).min()>-1e-12
    assert certify_update(old,graph([2.03,3.,5.]),tolerance=.01).rebuild
    with pytest.raises(ValueError):certify_update(old,new,tolerance=1.)


def test_full_fvm_independent_comparison():
    from morqex.top_damage.voxel_conductivity import effective_conductivity
    rng=np.random.default_rng(831);k=np.exp(rng.normal(size=(5,6,7)));k[1:3,2:4,1:3]=0
    for axis in (0,1,2):
        ref=effective_conductivity(k,(.7e-6,.8e-6,1.2e-6),axis=axis)
        actual=voxel_network(k,(.7e-6,.8e-6,1.2e-6),axis=axis).homogenize()
        np.testing.assert_allclose(actual['effective_S_m'],ref['effective'],rtol=1e-8)
        np.testing.assert_allclose(actual['potential'][k>0],ref['potential'][k>0],atol=1e-8)
def test_source_free_terminal_query_never_reenters_full_solver():
    from unittest.mock import patch
    from morqex.top_damage.graph_transport import voxel_network
    op=voxel_network(np.ones((3,3,3)),(1.,)*3).homogenize()['operator']
    class ForbiddenSolve:
        def solve(self,*args,**kwargs):
            raise AssertionError('A terminal-only query performed a full solve')
    with patch.object(op,'factor',ForbiddenSolve()):
        np.testing.assert_allclose(op.currents([0.,1.]),[-3.,3.],rtol=1e-12)
