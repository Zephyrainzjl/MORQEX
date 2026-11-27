import numpy as np
import pytest
from morqex.mechanics import TetMesh
from morqex.microstructure import Volume


@pytest.mark.parametrize('second,components',[((1,1,1),2),((0,1,1),2),((0,0,1),1)])
def test_zero_area_contacts_do_not_weld_solids(second,components):
    labels=np.zeros((2,2,2),np.uint8);labels[0,0,0]=1;labels[second]=1
    volume=Volume(labels,spacing=(.2,.3,.4),origin=(2.,3.,4.))
    legacy=TetMesh.from_volume(volume,solid_phases=[1])
    split=TetMesh.from_volume(volume,solid_phases=[1],separate_point_contacts=True)
    assert legacy.n_components==1 and split.n_components==components
    np.testing.assert_array_equal(split.points[split.cells],legacy.points[legacy.cells])
    np.testing.assert_allclose(split.volumes,legacy.volumes,rtol=0,atol=0)
    np.testing.assert_array_equal(split.phases,legacy.phases)
    np.testing.assert_array_equal(split.voxel_ids,legacy.voxel_ids)


def test_fully_solid_block_has_identical_connectivity():
    v=Volume(np.ones((3,4,2),np.uint8),spacing=(.1,.2,.3))
    a=TetMesh.from_volume(v);b=TetMesh.from_volume(v,separate_point_contacts=True)
    np.testing.assert_array_equal(a.points,b.points);np.testing.assert_array_equal(a.cells,b.cells)
    assert b.n_components==1


def test_split_bodies_have_independent_fe_diffusion_and_mechanics():
    from morqex.mechanics.reference import ElasticitySolver
    from morqex.mechanics.tensors import isotropic_stiffness
    v=np.zeros((2,2,2),np.uint8);v[0,0,0]=v[0,1,1]=1
    mesh=TetMesh.from_volume(Volume(v),solid_phases=[1],separate_point_contacts=True)
    model=ElasticitySolver(mesh,isotropic_stiffness(100.,.3),boundary='neumann')
    first=np.flatnonzero(np.repeat(mesh.components==0,3));second=np.flatnonzero(np.repeat(mesh.components==1,3))
    assert model.K[first][:,second].nnz==0
    # Each floating body independently realizes its own stress-free expansion.
    eps=np.zeros((len(mesh.cells),6));eps[mesh.voxel_ids==0,:3]=.02;eps[mesh.voxel_ids!=0,:3]=-.01
    state=model.solve(eps,tolerance=1e-10)
    assert np.max(abs(state.stress))<1e-10
    assert model.constraint.rigid_modes_removed==12
    with pytest.raises(ValueError,match='explicit.*mapping'):
        from morqex.mechanics.constraints import constraints
        constraints(mesh,'periodic')
