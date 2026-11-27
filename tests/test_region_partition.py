import numpy as np
import pytest
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.top_damage.region_partition import connected_partition
from morqex.top_damage.region_port import RegionPort


def test_nested_connected_refinement_and_full_voxel_limit():
    labels=np.zeros((7,6,6),np.uint8);labels[1:3,1:5,1:5]=1;labels[4:6,1:5,1:5]=1
    fv=VoxelDiffusion(Volume(labels,spacing=(1e-6,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0])
    coarse=connected_partition(fv,6);fine=connected_partition(fv,12)
    for k in np.unique(fine):assert len(np.unique(coarse[fine==k]))==1
    for target in (6,12,len(fv.voxel_ids)):
        part=connected_partition(fv,target);port=RegionPort.from_diffusion(fv,part,fv.reactive,1.)
        assert port.dimension>=target
        assert port.weights.sum()==pytest.approx(fv.weights.sum())
        assert np.linalg.norm(port.stiffness@np.ones(port.dimension))<1e-13*np.linalg.norm(port.stiffness.data)
    np.testing.assert_array_equal(part,np.arange(len(part)))
    np.testing.assert_allclose(port.stiffness.toarray(),fv.stiffness.toarray())
    with pytest.raises(ValueError,match='disconnected'):connected_partition(fv,1)
