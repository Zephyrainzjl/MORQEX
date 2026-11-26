from dataclasses import replace
import json
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from morqex.geometry import MicrostructureData
from morqex.top_damage.polycrystal import PolycrystalData, partition_grains
from morqex.top_damage.morphology import PolycrystalSpec, measure_geometry, write_polycrystal_vti, write_geometry_html


def small_polycrystal():
    labels=np.zeros((5,6,7),dtype=np.uint8);labels[1:4,1:5,1:6]=1
    structure=MicrostructureData(labels,(2e-6,3e-6,4e-6),(5e-6,6e-6,7e-6),{0:'pore',1:'active'},{})
    return partition_grains(structure,labels.astype(np.int32),grains_per_particle=4,seed=19)


def test_partition_preserves_material_and_proper_rotations():
    data=small_polycrystal()
    np.testing.assert_array_equal(data.grain_ids>0,data.structure.labels==1)
    assert len(data.grain_rotations)>=4
    np.testing.assert_allclose(np.linalg.det(data.grain_rotations),1,atol=1e-14)
    tensor=np.diag([1.,2.,4.])
    np.testing.assert_allclose(np.linalg.eigvalsh(data.rotated_tensor_table(tensor)),
                               np.tile([1.,2.,4.],(len(data.grain_rotations),1)),atol=1e-14)
    assert data.fingerprint==small_polycrystal().fingerprint
    assert not data.grain_ids.flags.writeable
    with pytest.raises(ValueError,match='proper rotations'):
        replace(data,grain_rotations=-data.grain_rotations)


def test_grain_faces_area_and_parent_identity():
    data=small_polycrystal();faces=data.grain_boundary_faces()
    a,b=faces['voxel_pairs'].T
    assert np.all(data.grain_ids.ravel()[a]!=data.grain_ids.ravel()[b])
    np.testing.assert_array_equal(data.particle_ids.ravel()[a],data.particle_ids.ravel()[b])
    delta=np.abs(np.array(np.unravel_index(a,data.grain_ids.shape))-np.array(np.unravel_index(b,data.grain_ids.shape)))
    np.testing.assert_array_equal(delta.sum(axis=0),1)
    spacing=np.asarray(data.structure.spacing_zyx_m)[::-1]
    np.testing.assert_allclose(faces['reference_area_m2'],np.prod(spacing)/spacing[faces['normal_axis_xyz']])


def test_hdf5_full_geometry_and_vti_axis_order(tmp_path):
    data=small_polycrystal();path=data.save_hdf5(tmp_path/'initial.h5')
    loaded=PolycrystalData.load_hdf5(path)
    assert loaded.fingerprint==data.fingerprint
    mesh=ET.parse(write_polycrystal_vti(data,tmp_path/'initial.vti')).getroot().find('ImageData')
    assert mesh.attrib['WholeExtent']=='0 7 0 6 0 5'
    np.testing.assert_allclose(np.fromstring(mesh.attrib['Spacing'],sep=' '),[4e-6,3e-6,2e-6])
    for field,expected in [('phase_labels',data.structure.labels),('grain_ids',data.grain_ids),('particle_ids',data.particle_ids)]:
        actual=np.fromstring(mesh.find(f"./Piece/CellData/DataArray[@Name='{field}']").text,sep=' ',dtype=int)
        np.testing.assert_array_equal(actual.reshape(data.structure.labels.shape),expected)
    import h5py
    with h5py.File(path,'r+') as f:f['particle_ids'][1,1,1]=9
    with pytest.raises(ValueError):PolycrystalData.load_hdf5(path)


@pytest.mark.parametrize('seed',[104,105,106])
def test_few_large_particles_multiple_seeds(seed,tmp_path):
    spec=PolycrystalSpec(seed=seed)
    data=spec.generate();metrics=measure_geometry(data)
    assert metrics['particle_count']==4
    assert metrics['grain_count']>=32
    assert metrics['minimum_grain_voxels']>0
    assert metrics['networks']['CBD']['spans_xyz'][0]
    assert sum(metrics['particle_voxel_counts'].values())==np.count_nonzero(data.structure.labels==1)
    assert all(n>700 for n in metrics['particle_voxel_counts'].values())
    assert max(abs(v) for v in metrics['sphere_volume_signed_relative_errors'].values())<.025
    assert abs(sum(metrics['phase_fractions'].values())-1)<1e-14
    html=write_geometry_html(data,metrics,tmp_path/'index.html').read_text()
    assert '__MORQEX_DATA__' not in html and 'initial_geometry_only' in html
    assert 'https://' not in html


def test_reject_invalid_geometry_identity_and_configuration():
    data=small_polycrystal()
    with pytest.raises(ValueError):partition_grains(data.structure,np.zeros_like(data.particle_ids))
    bad=data.particle_ids.copy();bad[0,0,0]=-1
    with pytest.raises(ValueError):replace(data,particle_ids=bad)
    for kwargs in ({'name':'../overwrite'},{'particle_count':True},{'grains_per_particle':0},
                   {'spacing_m':float('nan')},{'shape_zyx':[32.,32,32]}):
        with pytest.raises(ValueError):PolycrystalSpec(**kwargs)


def test_grain_boundary_corruption_is_rejected(tmp_path):
    import h5py
    path=small_polycrystal().save_hdf5(tmp_path/'initial.h5')
    with h5py.File(path,'r+') as f:
        f['grain_boundaries/reference_area_m2'][0]*=2
    with pytest.raises(ValueError,match='Stored grain boundaries'):PolycrystalData.load_hdf5(path)


def test_toml_entry_preserves_existing_distinct_geometry(tmp_path):
    from morqex.top_damage.morphology import create_polycrystal
    from pathlib import Path
    config=tmp_path/'case.toml'
    source=Path(__file__).resolve().parents[1]/'configs/top_damage_few_particles.toml'
    config.write_text(source.read_text())
    result=create_polycrystal(config,tmp_path/'work')
    folder=Path(result['directory']);manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['geometry_fingerprint']==PolycrystalData.load_hdf5(folder/'initial.h5').fingerprint
    before=(folder/'initial.h5').read_bytes()
    config.write_text(source.read_text().replace('seed = 104','seed = 105'))
    with pytest.raises(ValueError,match='different geometry/config'):
        create_polycrystal(config,tmp_path/'work')
    assert (folder/'initial.h5').read_bytes()==before
