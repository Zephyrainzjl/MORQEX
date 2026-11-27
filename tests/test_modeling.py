import numpy as np
import pytest
import h5py
from morqex.modeling import from_regions,Shape,from_shapes,stream_tetrahedralize,capabilities


def test_stream_matches_existing_global_mesh_with_anisotropic_units_and_voids(tmp_path):
    labels=np.zeros((7,6,5),np.uint8);labels[1:6,1:5,1:4]=3;labels[3,2:5,2:4]=8
    m=from_regions(labels,spacing=(.3e-6,.4e-6,.6e-6),origin=(2e-6,-1e-6,3e-6))
    reference=m.tetrahedralize(regions=[3,8])
    artifact=m.mesh_to(tmp_path/'mesh.h5',regions=[3,8],slab_depth=2)
    with h5py.File(artifact.hdf5) as f:
        np.testing.assert_array_equal(f['mesh/tetrahedra'][:],reference.cells)
        np.testing.assert_allclose(f['mesh/points_m'][:],reference.points,rtol=1e-14,atol=1e-20)
        np.testing.assert_array_equal(f['mesh/phase'][:],reference.phases)
        parent=np.repeat(np.flatnonzero(labels),6)
        np.testing.assert_array_equal(f['mesh/parent_voxel'][:],parent)
        assert bool(f.attrs['complete']) and f.attrs['volume_relative_error']<1e-12
    assert artifact.xdmf.exists()
    restored=type(m).load(artifact.hdf5)
    np.testing.assert_array_equal(restored.regions,labels)
    actual_view=artifact.viewer(first_cell=4,cell_count=30)
    assert 'Actual tetrahedral subset' in actual_view.document
    assert len(actual_view.document)>1000
    meshio=pytest.importorskip('meshio')
    independent=meshio.read(artifact.xdmf)
    np.testing.assert_allclose(independent.points,reference.points,rtol=1e-14,atol=1e-20)
    np.testing.assert_array_equal(independent.cells_dict['tetra'],reference.cells)
    with pytest.raises(FileExistsError):m.mesh_to(artifact.hdf5,regions=[3,8])


def test_array_labels_are_explicit_and_saved_without_semantic_guessing(tmp_path):
    a=np.full((4,5,6),17,np.uint16);a[:,2]=42
    m=from_regions(a,region_names={17:'cathode',42:'coating'})
    saved=m.save(tmp_path/'voxels.h5');loaded=type(m).load(saved)
    np.testing.assert_array_equal(loaded.regions,a)
    assert loaded.region_names==m.region_names
    with pytest.raises(ValueError):from_regions(a.astype(float))
    with pytest.raises(ValueError):from_regions(a,region_names={0:'pore'})
    with pytest.raises(ValueError):from_regions(np.full((2,2,2),-1))


def test_real_primitives_preserve_units_and_overwrite_order():
    m=from_shapes([Shape('box',1,(8e-6,8e-6,8e-6),(4e-6,)*3),
                   Shape('sphere',2,(8e-6,)*3,(2e-6,)*3)],shape=(16,)*3,spacing=(1e-6,)*3)
    assert m.regions[8,8,8]==2 and m.regions[5,5,5]==1 and m.regions[0,0,0]==0
    angle=.63;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    rod=from_shapes([Shape('capsule',1,(16e-6,)*3,(1.8e-6,1.8e-6,11e-6),tuple(map(tuple,R)),.2)],shape=(32,)*3)
    assert np.count_nonzero(rod.regions)>50
    indices=np.argwhere(rod.regions);assert np.ptp(indices[:,2])>np.ptp(indices[:,1])*3
    with pytest.raises(ValueError):from_shapes([Shape('box',1,(0,)*3,(1,)*3,tuple(map(tuple,2*np.eye(3))))])


def test_missing_backend_is_explicit():
    from morqex.modeling import framework
    assert framework('shapes').Sphere
    assert 'native_reconstruction' in capabilities()
    with pytest.raises(ValueError):framework('made-up')


def test_toml_and_registered_generator_share_the_region_contract(tmp_path):
    from morqex.modeling import from_toml,register_generator,generate
    source=np.ones((3,4,5),np.uint8);np.save(tmp_path/'labels.npy',source)
    config=tmp_path/'regions.toml';config.write_text("[geometry]\nmethod='regions'\narray='labels.npy'\nspacing=[1e-6,2e-6,3e-6]\n")
    np.testing.assert_array_equal(from_toml(config).regions,source)
    register_generator('test_custom_regions',lambda array:from_regions(array))
    np.testing.assert_array_equal(generate('test_custom_regions',array=source).regions,source)
    with pytest.raises(ValueError):register_generator('test_custom_regions',lambda:None)
    register_generator('test_invalid_output',lambda:np.zeros((2,2,2)))
    with pytest.raises(TypeError):generate('test_invalid_output')


def test_registered_meshers_must_return_the_shared_mesh_contract():
    from morqex.modeling import mesh,register_mesher
    model=from_regions(np.ones((2,3,4),np.uint8))
    register_mesher('test_custom_tetra',lambda model:model.tetrahedralize())
    assert len(mesh(model,method='test_custom_tetra').cells)==6*24
    with pytest.raises(ValueError):register_mesher('tetrahedral',lambda x:None)
    register_mesher('test_wrong_mesh',lambda model:np.zeros((2,3)))
    with pytest.raises(TypeError):mesh(model,method='test_wrong_mesh')


def test_progress_is_actual_partial_geometry_and_view_updates_same_model(tmp_path):
    from morqex.visualization.notebook import VoxelView
    blank=from_regions(np.zeros((16,)*3,np.uint8))
    view=VoxelView(blank,maximum_3d_size=8)
    snapshots=[]
    def progress(model):
        snapshots.append(int(np.count_nonzero(model.regions)))
        view.update(model)
    result=from_shapes([Shape('box',1,(5e-6,)*3,(2e-6,)*3),
                        Shape('box',2,(12e-6,)*3,(2e-6,)*3)],
                        shape=(16,)*3,progress=progress,progress_every=1)
    assert len(snapshots)==2 and 0<snapshots[0]<snapshots[1]
    np.testing.assert_array_equal(view.model.regions,result.regions)
    assert '<iframe' in view._repr_html_()
    saved=view.save_html(tmp_path/'view.html')
    assert 'DecompressionStream' in saved.read_text() and 'previewShape' in saved.read_text()
