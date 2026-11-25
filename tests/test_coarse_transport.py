import numpy as np
import pytest
from scipy import sparse
from morqex.acceleration.coarse_transport import coarse_matrix

def test_coarse_graph_is_exact_p0_galerkin_including_partial_periodic_blocks():
    shape=(7,8,9);rng=np.random.default_rng(17);phase=rng.integers(0,3,shape,dtype=np.uint8);coefficient=np.array([0.,1.,.2]);spacing=(2.,3.,4.);block=3
    coarse,cshape=coarse_matrix(phase,coefficient,spacing,block);N=phase.size;ids=np.arange(N).reshape(shape);grid=np.indices(shape);groups=np.ravel_multi_index(tuple(grid//block),cshape).ravel()
    P=sparse.coo_matrix((np.ones(N),(np.arange(N),groups)),shape=(N,int(np.prod(cshape)))).tocsr();rows=[];cols=[];values=[]
    here=coefficient[phase]
    for a in range(3):
        other=np.roll(here,-1,a);g=np.divide(2*here*other,here+other,out=np.zeros_like(here),where=here+other>0)*(min(spacing)/spacing[a])**2
        i=ids.ravel();j=np.roll(ids,-1,a).ravel();v=g.ravel();rows.extend([i,i,j,j]);cols.extend([i,j,i,j]);values.extend([v,-v,-v,v])
    K=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(N,N)).tocsr()
    np.testing.assert_allclose(coarse.toarray(),(P.T@K@P).toarray(),atol=1e-13)
    np.testing.assert_allclose(coarse@np.ones(coarse.shape[0]),0,atol=1e-13)
    assert np.linalg.eigvalsh(coarse.toarray()).min()>-1e-12

def test_elastic_coarse_is_exact_same_stencil_galerkin_with_kelvin_coupling():
    from morqex.acceleration.coarse_elasticity import coarse_matrix as elastic_coarse
    from morqex.acceleration.voxel import NativeStencil
    rng=np.random.default_rng(18);shape=(5,6,7);phase=rng.integers(0,3,shape,dtype=np.uint8);C=[]
    for k in range(3):
        A=rng.normal(size=(6,6));C.append(A.T@A if k else np.zeros((6,6)))
    model=NativeStencil(phase,C,spacing=(2.,3.,4.));K,coarse=elastic_coarse(phase,model.C_host,model.spacing,block=3)
    groups=np.ravel_multi_index(tuple(np.indices(shape)//3),coarse).ravel();n=phase.size;nc=int(np.prod(coarse));r=rng.normal(size=3*nc);s=rng.normal(size=3*nc)
    fine=lambda a:a.reshape(nc,3)[groups].T.reshape(3,*shape)
    assert r@(K@s)==pytest.approx(np.vdot(fine(r),model.apply(fine(s))),rel=1e-12,abs=1e-10)

def test_material_aware_coarse_matches_exact_fine_energy_projection():
    from morqex.acceleration.phase_coarse import assemble
    from morqex.acceleration.voxel import NativeStencil
    from morqex.mechanics import isotropic_stiffness
    rng=np.random.default_rng(21);shape=(5,6,7);phase=rng.integers(0,3,shape,dtype=np.uint8);C=np.stack([np.zeros((6,6)),isotropic_stiffness(100,.2),isotropic_stiffness(1,.3)])
    model=NativeStencil(phase,C);K,coarse=assemble(phase,model.C_host,model.spacing,3)
    group=np.ravel_multi_index(tuple(np.indices(shape)//3),coarse[:3])*3+phase;nc=int(np.prod(coarse));r=rng.normal(size=3*nc);s=rng.normal(size=3*nc)
    lift=lambda a:a.reshape(nc,3)[group.ravel()].T.reshape(3,*shape)
    assert r@(K@s)==pytest.approx(np.vdot(lift(r),model.apply(lift(s))),rel=1e-12,abs=1e-10)
