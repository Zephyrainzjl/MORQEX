import numpy as np
import pytest
from morqex.top_damage.periodic_transport import periodic_transport


def test_homogeneous_tensor_and_two_cell_periodicity():
    for shape in [(2,2,2),(3,4,5)]:
        for k in [1e-15,1.,1e9]:
            r=periodic_transport(np.full(shape,k),(1e-6,2e-6,3e-6))
            np.testing.assert_allclose(r.tensor,k*np.eye(3),rtol=1e-12,atol=k*1e-14)
            np.testing.assert_allclose(r.correctors,0,atol=1e-20)
            assert r.flux_energy_defect<1e-12


def test_layered_tensor_series_parallel_exact():
    k=np.ones((4,4,8));k[:,:,4:]=5.
    r=periodic_transport(k,(1.,1.,1.))
    np.testing.assert_allclose(r.tensor,np.diag([5/3,3,3]),rtol=1e-9,atol=1e-10)
    assert max(r.residuals)<1e-8
    assert r.flux_energy_defect<1e-8


def test_true_zero_pores_and_floating_solid():
    empty=periodic_transport(np.zeros((3,3,3)),(1.,1.,1.))
    np.testing.assert_array_equal(empty.tensor,0)
    island=np.zeros((6,6,6));island[2:4,2:4,2:4]=2.
    r=periodic_transport(island,(1.,1.,1.))
    np.testing.assert_allclose(r.tensor,0,atol=1e-20)
    slab=np.zeros((4,4,6));slab[:,:,:3]=2.
    s=periodic_transport(slab,(1.,1.,1.))
    np.testing.assert_allclose(s.tensor,np.diag([0,1,1]),atol=1e-9)


def test_oblique_layers_resolve_offdiagonal_and_converge():
    errors=[]
    normal=np.array([1,1,0.])/np.sqrt(2)
    exact=3*np.eye(3)+(5/3-3)*np.outer(normal,normal)
    for n in [12,24,48]:
        z,y,x=np.indices((2,n,n));k=np.where((x+y)%n<n//2,1.,5.)
        r=periodic_transport(k,(1.,1/n,1/n))
        errors.append(np.linalg.norm(r.tensor-exact))
        assert r.tensor[0,1]<-.3
        np.testing.assert_allclose(r.tensor,r.tensor.T,atol=1e-12)
    assert errors[2]<errors[1]<errors[0]
    assert errors[-1]/np.linalg.norm(exact)<.04


def test_rigid_axis_permutation_and_constitutive_scaling():
    rng=np.random.default_rng(5);k=np.exp(rng.normal(size=(4,5,6)))
    a=periodic_transport(k,(.2,.3,.4));b=periodic_transport(k*7,(.2,.3,.4))
    np.testing.assert_allclose(b.tensor,a.tensor*7,rtol=1e-8,atol=1e-9)
    c=periodic_transport(k.transpose(2,1,0),(.4,.3,.2))
    np.testing.assert_allclose(c.tensor,a.tensor[::-1,::-1],rtol=1e-8,atol=1e-9)
    assert a.fingerprint!=b.fingerprint


def test_invalid_coefficients_fail():
    for value in [-1,np.nan,np.inf]:
        with pytest.raises(ValueError):periodic_transport(np.full((3,3,3),value),(1,1,1))
