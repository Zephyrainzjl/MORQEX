import hashlib
import pytest
from morqex.workspace import Workspace
from morqex.top_damage.network_geometry import load_cycle_geometry


def test_imported_geometry_is_identical_and_requires_content_identity(tmp_path):
    ws=Workspace(tmp_path)
    settings=dict(shape_zyx=[12,12,12],particle_count=8,radius_voxels=2.5,
                  radius_cv=0.,minimum_gap_voxels=1.,cbd_fraction=.35,grains_per_particle=2,seed=104)
    original,_,_=load_cycle_geometry(settings,ws)
    path=ws.path('data/imported.h5');original.save_hdf5(path)
    source={'source_hdf5':'data/imported.h5','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'solver_seed':104}
    restored,meta,seed=load_cycle_geometry(source,ws)
    assert restored.fingerprint==original.fingerprint==meta['fingerprint'] and seed==104
    with pytest.raises(ValueError,match='checksum'):
        load_cycle_geometry(source|{'sha256':'0'*64},ws)
    with pytest.raises(ValueError,match='external workspace'):
        load_cycle_geometry(source|{'source_hdf5':'../outside.h5'},ws)


def test_cycle_export_preserves_imported_reference_origin(tmp_path):
    import os,json,re
    import numpy as np,h5py
    from dataclasses import replace
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from morqex.top_damage.network_cycle import run_network_cycle
    from morqex.top_damage.network_export import export_network_history
    from morqex.microstructure import Volume
    from morqex.mechanics import TetMesh
    if not os.environ.get('MORQEX_WORKSPACE'):pytest.skip('Budgeted local runner required')
    ws=Workspace(os.environ['MORQEX_WORKSPACE'])
    data,_,_=load_cycle_geometry(dict(shape_zyx=[12]*3,particle_count=8,radius_voxels=2.5,radius_cv=0.,minimum_gap_voxels=1.,cbd_fraction=.35,grains_per_particle=2,seed=104),ws)
    offset=(3e-6,-2e-6,1e-6)
    data=replace(data,structure=replace(data.structure,origin_zyx_m=offset))
    tag=hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    path=ws.path('experiments/translated_'+tag+'.h5');data.save_hdf5(path)
    template=(Path(__file__).parents[1]/'configs/top_damage_network_cycle.toml').read_text().split('[microstructure]')[0]
    template=re.sub(r'output = ".*"','output = "translated_'+tag+'"',template)
    template=template.replace('[cycle]','[cycle]\nmechanics_enabled = false')
    template+='[microstructure]\nsource_hdf5 = '+json.dumps(str(path.relative_to(ws.root)))+'\nsha256 = '+json.dumps(hashlib.sha256(path.read_bytes()).hexdigest())+'\nsolver_seed = 104\n'
    config=tmp_path/'translated.toml';config.write_text(template)
    result=run_network_cycle(config,ws,steps=1)
    exported=export_network_history(result['directory'],ws,steps=[0])
    vti=ET.parse(Path(exported['directory'])/'node_000_step_000000.vti')
    np.testing.assert_array_equal(np.fromstring(vti.find('ImageData').attrib['Origin'],sep=' '),offset[::-1])
    expected=TetMesh.from_volume(Volume(data.structure.labels,spacing=data.structure.spacing_zyx_m,origin=offset),solid_phases=[1,2])
    with h5py.File(Path(result['directory'])/'mesh.h5') as f:np.testing.assert_array_equal(f['points_m'][:],expected.points)
