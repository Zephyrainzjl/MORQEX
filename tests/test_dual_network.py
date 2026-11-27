import numpy as np
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.top_damage.region_port import RegionPort
from morqex.top_damage.dual_network import DualNetworkClosure


def test_reaction_requires_both_reservoirs_and_material_volume_is_retained():
    labels=np.ones((4,4,6),np.uint8);labels[0]=0;labels[-1]=2
    fv=VoxelDiffusion(Volume(labels,spacing=(1e-6,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0,2])
    port=RegionPort.from_diffusion(fv,(fv.coordinates_zyx[:,2]>=3).astype(int),fv.reactive,100.)
    closure=DualNetworkClosure(fv,port)
    electronic=np.where(labels==0,0.,10.);ionic=np.where(labels==1,0.,1.)
    intact=closure.evaluate(electronic,ionic)
    np.testing.assert_allclose(intact.reaction_fractions,1.)
    faces=np.ones((3,)+labels.shape);faces[2,:,:,2]=0
    split=closure.evaluate(electronic,ionic,electronic_face_factors=faces,solid_face_factors=faces)
    assert split.reaction_fractions[0]==0 and split.reaction_fractions[1]>0
    assert split.diffusion_edge_factors[0]==0
    assert split.diagnostics['material_volume_m3']==intact.diagnostics['material_volume_m3']
    assert not split.diagnostics['unresolved_split_regions']
    faces[2,:,:,0]=0
    unresolved=closure.evaluate(electronic,ionic,solid_face_factors=faces)
    assert unresolved.diagnostics['unresolved_split_regions']==[0]


def test_event_reuse_uses_last_rebuilt_state():
    labels=np.ones((3,3,4),np.uint8);labels[0]=0;labels[-1]=2
    fv=VoxelDiffusion(Volume(labels,spacing=(1e-6,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0,2])
    port=RegionPort.from_diffusion(fv,np.zeros(len(fv.voxel_ids),int),fv.reactive,100.)
    closure=DualNetworkClosure(fv,port,update_tolerance=.01)
    e=np.where(labels==0,0.,10.);i=np.where(labels==1,0.,1.)
    initial=closure.evaluate(e,i)
    small=closure.evaluate(.995*e,i)
    assert not small.diagnostics['electronic']['rebuilt']
    assert small.electronic_conductivity_S_m==initial.electronic_conductivity_S_m
    cumulative=closure.evaluate(.988*e,i)
    assert cumulative.diagnostics['electronic']['rebuilt']


def test_dimensionless_ionic_efficiency_is_independent_of_normalization():
    labels=np.ones((3,3,4),np.uint8);labels[0]=0;labels[-1]=2
    fv=VoxelDiffusion(Volume(labels,spacing=(1e-6,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0,2])
    port=RegionPort.from_diffusion(fv,np.zeros(len(fv.voxel_ids),int),fv.reactive,100.)
    e=np.where(labels==0,0.,10.);i=np.where(labels==1,0.,1.)
    results=[DualNetworkClosure(fv,port,ionic_reference_conductivity_S_m=k).evaluate(e,i) for k in (1.,7.3)]
    np.testing.assert_allclose(results[0].electrolyte_efficiency,results[1].electrolyte_efficiency,rtol=1e-12)
    np.testing.assert_allclose(results[0].ionic_potential,results[1].ionic_potential,atol=1e-12)
