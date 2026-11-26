import numpy as np
import pytest
from morqex.geometry import MicrostructureData
from morqex.workspace import Workspace
from morqex.nodewise import BindingSpec,bind_microstructures
from morqex.generation import StatisticsCard,generate,characterize


def reference():
    z,y,x=np.indices((20,20,60));a=np.ones(x.shape,dtype=np.uint8)
    a[z<5]=0;a[(z>=5)&(z<8)]=2
    a[(y==10)&(x%7<3)&(z>=8)]=2
    return MicrostructureData(a,(1e-6,)*3,(0.,)*3,{0:'pore',1:'active',2:'cbd'},{'source':'synthetic layered regression'})


def test_binding_three_modes_and_native_coordinates(tmp_path):
    ws=Workspace(tmp_path);r=reference()
    shared=bind_microstructures(BindingSpec(shape=(16,)*3),3,'positive',ws,reference=r)
    assert len({b.operator_geometry.fingerprint for b in shared})==1
    slab=bind_microstructures(BindingSpec(mode='slab',shape=(16,)*3),3,'positive',ws,reference=r)
    assert len({b.operator_geometry.fingerprint for b in slab})==3
    assert sum(b.macro_weight for b in slab)==pytest.approx(1.)
    for b in slab:
        lo,hi=b.selection['slab_x_voxels'];crop=b.structure.provenance['crop_zyx'][2]
        assert lo<=crop[0]<crop[1]<=hi
        np.testing.assert_array_equal(b.structure.labels,r.labels[tuple(slice(*v) for v in b.structure.provenance['crop_zyx'])])
    with pytest.raises(ValueError,match='fit'):bind_microstructures(BindingSpec(mode='slab',shape=(16,)*3),4,'positive',ws,reference=r)
    card=StatisticsCard({0:.30,1:.55,2:.15},{p:(2e-6,)*3 for p in range(3)},r.phase_kinds,'synthetic statistics')
    statistical=bind_microstructures(BindingSpec(mode='statistical',shape=(16,)*3,statistics=card),3,'positive',ws,reference=r)
    assert len({b.operator_geometry.fingerprint for b in statistical})==1
    labels=statistical[0].structure.labels
    for p,f in card.fractions.items():assert abs(np.mean(labels==p)-f)<=1/labels.size
    np.testing.assert_array_equal(labels,generate(card,seed=104).labels)


def test_directional_S2_and_physical_length():
    r=reference();card,record=characterize(r)
    for p,f in card.fractions.items():
        for s2 in record['directional_S2'][p]:assert s2[0]==pytest.approx(f,abs=1e-6)
    h=MicrostructureData(r.labels,(2e-6,)*3,(0.,)*3,r.phase_kinds,r.provenance)
    doubled,_=characterize(h)
    for p in card.fractions:np.testing.assert_allclose(doubled.correlation_lengths_m[p],np.array(card.correlation_lengths_m[p])*2)


def test_upstream_namespace_geometry():
    from morqex.vendor.microstructpy.geometry import Sphere,Box
    s=Sphere(radius=2,center=(0,0,0))
    assert s.volume==pytest.approx(4/3*np.pi*8)
    np.testing.assert_array_equal(s.within([[0,0,0],[3,0,0]]),[True,False])
    assert Box(side_lengths=[2,3,4]).volume==pytest.approx(24.)


def test_sparse_cohesive_independent_energy_and_residual_bound(tmp_path):
    from morqex.gemsim import GEMsim
    from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
    from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode
    from morqex.topology import PorousCBD
    from morqex.cohesive import build_cohesive
    from morqex.sparse_cohesive import build_sparse_cohesive,sparse_cohesive_certificate
    p,_=microstructure_parameters();labels=np.ones((4,)*3,dtype=np.uint8);labels[0]=0;labels[1]=2
    data=MicrostructureData(labels,(1e-6,)*3,(0.,)*3,{0:'pore',1:'active',2:'cbd'},{'source':'test'})
    full=MicrostructureElectrode(data,GEMsim(Workspace(tmp_path)),p,'positive',topology=PorousCBD())
    partition=full.partition_labels(4);P=np.eye(partition.max()+1)[partition]
    exact=build_cohesive(full,P);reduced=build_sparse_cohesive(full,P,training_points=4,maximum_modes=100,retain_system=True)
    rng=np.random.default_rng(41)
    for d in [.03,.23,.59,.82]:
        scale=np.sqrt(np.maximum(exact.direct.diagonal(),1e-30));q=rng.normal(size=len(scale))/scale
        delta=.5*q@(reduced.hessian(d)-exact.hessian(d))@q
        bound=sparse_cohesive_certificate(reduced,q,d)['energy_error_upper_J_m3']
        assert delta>=-1e-10
        assert delta<=bound+1e-10
        assert delta<1e-5
        a=exact.recover(q,d)['stress'];b=reduced.recover(q,d)['stress']
        np.testing.assert_allclose(a,b,atol=1.,rtol=1e-3)
