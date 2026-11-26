import numpy as np
import pytest
from morqex.top_damage.crack_opening import reconstruct_opening


def cloud(n=17):
    a=np.linspace(-1,1,n)
    return np.stack(np.meshgrid(a,a,a,indexing='ij'),axis=-1).reshape(-1,3)


def fit(x,u,**kwargs):
    return reconstruct_opening(x,u,np.zeros((1,3)),[[1,0,0]],inner_distance_m=.3,
        outer_distance_m=.9,tangential_radius_m=.8,**kwargs)


def test_affine_deformation_does_not_create_a_crack():
    x=cloud();F=np.array([[1.4,.2,.1],[.1,.8,.2],[0,0,1.1]])
    u=x@(F-np.eye(3)).T+[.1,-.2,.3]
    r=fit(x,u).require_usable()
    np.testing.assert_allclose(r.aperture_m,0,atol=2e-15)
    np.testing.assert_allclose(r.jump_m,0,atol=2e-15)


def test_jump_and_finite_rotation_objectivity():
    x=cloud();F=np.array([[1.2,.1,0],[0,.9,.1],[0,0,1.1]])
    jump=np.array([.04,.02,-.01]);u=x@(F-np.eye(3)).T+(x[:,0]>0)[:,None]*jump
    r=fit(x,u).require_usable()
    np.testing.assert_allclose(r.jump_m[0],jump,atol=2e-15)
    normal=np.linalg.solve(F.T,[1,0,0]);normal/=np.linalg.norm(normal)
    np.testing.assert_allclose(r.signed_gap_m[0],normal@jump,atol=2e-15)
    a=1.9;Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
    rotated=(x+u)@Q.T+np.array([2.,3.,1.])-x
    rr=fit(x,rotated).require_usable()
    np.testing.assert_allclose(rr.jump_m,r.jump_m@Q.T,atol=3e-14)
    np.testing.assert_allclose(rr.current_normals,r.current_normals@Q.T,atol=3e-14)
    np.testing.assert_allclose(rr.aperture_m,r.aperture_m,atol=3e-14)
    np.testing.assert_allclose(rr.slip_m,r.slip_m@Q.T,atol=3e-14)


def test_negative_gap_retained_not_called_contact_solution():
    x=cloud();u=(x[:,0]>0)[:,None]*np.array([-.03,.02,0])
    r=fit(x,u).require_usable()
    assert r.signed_gap_m[0]==pytest.approx(-.03)
    assert r.aperture_m[0]==0
    np.testing.assert_allclose(r.slip_m[0],[0,.02,0],atol=1e-14)


def test_insufficient_samples_do_not_silently_mean_closed():
    x=cloud();u=np.zeros_like(x);d=np.where(x[:,0]<0,1.,0.)
    r=fit(x,u,sample_damage=d)
    assert not r.usable[0] and np.isnan(r.aperture_m[0])
    with pytest.raises(ValueError,match='Unresolved'):r.require_usable()
    y=x[x[:,1]==0];r=fit(y,np.zeros_like(y))
    assert not r.usable[0] and 'rank' in r.reasons[0]


def test_smooth_regularized_jump_and_support_sensitivity():
    x=cloud(25);true=.04
    errors=[]
    for width in [.15,.08,.04]:
        u=np.zeros_like(x);u[:,0]=true/2*np.tanh(x[:,0]/width)+.01*x[:,0]
        r=fit(x,u).require_usable();errors.append(abs(r.aperture_m[0]-true))
    assert errors[2]<errors[1]<errors[0]
    assert errors[-1]<true*1e-5


def test_unit_scaling_weights_and_fit_rejection():
    x=cloud();u=(x[:,0]>0)[:,None]*np.array([.04,0.,0.])
    a=fit(x,u,weights=np.linspace(1,2,len(x))).require_usable()
    b=reconstruct_opening(x*1e-6,u*1e-6,np.zeros((1,3)),[[1,0,0]],inner_distance_m=.3e-6,
        outer_distance_m=.9e-6,tangential_radius_m=.8e-6,weights=np.linspace(1,2,len(x))*1e-18).require_usable()
    np.testing.assert_allclose(b.jump_m/1e-6,a.jump_m,atol=1e-14)
    u[:,0]+=.1*x[:,0]**3
    bad=fit(x,u,maximum_fit_rms_m=1e-5)
    assert not bad.usable[0] and 'tolerance' in bad.reasons[0]
