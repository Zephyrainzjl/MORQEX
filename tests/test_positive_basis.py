import json
import numpy as np
import pytest
from morqex.concentration_modes.positive import ComponentBasis
from morqex.concentration_modes.full_volume import project_volume

def test_componentwise_partition_positive_and_inventory_nullspaces():
    rng=np.random.default_rng(4);components=np.r_[np.ones(40,int),np.full(30,2),np.zeros(10,int)];features=rng.random((3,80));features[:,components==0]=0
    with pytest.raises(ValueError,match='independent'):ComponentBasis.allocate(components,features,target=1)
    basis=ComponentBasis.allocate(components,features,target=10);P=basis.evaluate(components,features)
    assert P.min()>=0
    np.testing.assert_allclose(P.sum(axis=1),components>0,atol=1e-14)
    for k in [1,2]:
        a=np.zeros(basis.dimension);a[basis.offsets[k-1]:basis.offsets[k]]=1
        np.testing.assert_allclose(P@a,components==k,atol=1e-14)

def test_chunked_full_projection_matches_explicit_face_galerkin(tmp_path):
    import hashlib
    shape=(4,5,6);rng=np.random.default_rng(12);components=np.ones(shape,np.int32);components[2:]=2;components[:,0,0]=0
    features=rng.random((3,*shape)).astype('float32');features[:,components==0]=0;faces=np.zeros(shape,np.uint8)
    for axis in range(3):
        for sign in [-1,1]:faces+=(components>0)&(np.roll(components,sign,axis)==0)
    # In a physical geometry distinct components never share an active face.
    components[2]=0;faces[:]=0
    for axis in range(3):
        for sign in [-1,1]:faces+=(components>0)&(np.roll(components,sign,axis)==0)
    np.savez_compressed(tmp_path/'geometry.npz',components=components,features=features,faces=faces)
    manifest={'sha256':hashlib.sha256((tmp_path/'geometry.npz').read_bytes()).hexdigest(),'voxel_size_m':.7,'phase_fractions':{'1':float(np.mean(components>0))}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest));report=project_volume(tmp_path,extra_nodes=6,chunk=17)
    basis=ComponentBasis.from_metadata(report['basis']);features[1]=faces/6;P=basis.evaluate(components,features);N=components.size;K=np.zeros((basis.dimension,)*2)
    q=P.reshape(*shape,basis.dimension)
    for axis in range(3):
        valid=(components>0)&(components==np.roll(components,-1,axis));delta=(np.roll(q,-1,axis)-q)[valid];K+=delta.T@delta/(N*.7**2)
    with np.load(tmp_path/'concentration.npz') as a:
        np.testing.assert_allclose(a['weights'],P.sum(axis=0)/N,atol=1e-14)
        np.testing.assert_allclose(a['reaction_area'],P.T@faces.ravel()/(N*.7),atol=1e-14)
        np.testing.assert_allclose(a['stiffness'],K,atol=1e-13)
    from morqex.concentration_modes.spatial_ports import build_spatial_ports
    projected=build_spatial_ports(tmp_path,bins=3,chunk=17)
    assert projected['mass_discrepancy']<1e-14
    with np.load(tmp_path/'spatial_ports_3.npz') as a:
        W=a['volume_by_mode_macro'];state=np.linspace(100,150,len(W));local_fraction=W.sum(axis=0)/a['macro_volumes'];local_mean=(W.T@state)/W.sum(axis=0)
        assert np.dot(a['macro_volumes']*local_fraction,local_mean)==pytest.approx(W.sum(axis=1)@state,rel=1e-14)

def test_cell_capacity_policy_reports_rate_and_changes_only_explicit_policy():
    import pybamm as pb
    from morqex.electrochemistry.cell_design import CellDesign
    p=pb.ParameterValues('Chen2020');windows=((.05,.9),(.1,.9))
    a,report=CellDesign(*windows).apply(p,active_fractions={'positive':.4})
    assert a['Nominal cell capacity [A.h]']==p['Nominal cell capacity [A.h]']
    assert report['physical_C_rate_at_0_1C_nameplate']!=pytest.approx(.1)
    b,balanced=CellDesign(*windows,policy='balance-negative-thickness',negative_to_positive_ratio=1.15).apply(p,active_fractions={'positive':.4})
    assert balanced['N_to_P_usable_capacity_ratio']==pytest.approx(1.15)
    assert balanced['physical_C_rate_at_0_1C_nameplate']==pytest.approx(.1)
    assert p['Positive electrode active material volume fraction']!=.4
