import json
import numpy as np
import pytest
from morqex.modeling import from_regions,partitioned_tetgen,TetGenOptions


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('surface_method',['voxel_plc','surface_nets'])
def test_generic_whole_voxel_mesh_anisotropic_offset_and_checkpoint(tmp_path,axis,surface_method):
    pytest.importorskip('triangle')
    if surface_method=='surface_nets':pytest.importorskip('vtkmodules.vtkFiltersCore')
    import h5py
    a=np.zeros((8,8,8),np.uint8);a[:,2:6,2:6]=7
    model=from_regions(a,spacing=(.7e-6,.8e-6,1e-6),origin=(3e-6,-2e-6,5e-6))
    options=TetGenOptions.for_adaptive(preserve_surface=True,merge_coplanar=False,maximum_cells=50000,maximum_steiner=10000,region_labelling='interfaces',coplanarity_tolerance=1e-15)
    target=tmp_path/f'axis{axis}'
    artifact=partitioned_tetgen(model,target,surface_method=surface_method,slab_depth=4,axis=axis,tetgen_options=options)
    with h5py.File(artifact.hdf5) as f:
        p=f['mesh/points_m'][:];r=json.loads(f.attrs['report'])
        np.testing.assert_allclose(p.min(0),np.array(model.origin)[::-1],rtol=1e-12,atol=1e-20)
        np.testing.assert_allclose(np.ptp(p,axis=0),np.array(model.spacing)[::-1]*8,rtol=1e-12)
        for k in (0,7):assert r['phase_volumes_m3'][str(k)]==pytest.approx(float((a==k).sum()*np.prod(model.spacing)),rel=1e-7)
        assert len(r['cap_checks'])==1
    before=artifact.hdf5.stat().st_mtime_ns
    resumed=partitioned_tetgen(model,target,surface_method=surface_method,slab_depth=4,axis=axis,tetgen_options=options)
    assert resumed.hdf5.stat().st_mtime_ns==before
    with pytest.raises(ValueError,match='different source/configuration'):
        partitioned_tetgen(model,target,surface_method=surface_method,slab_depth=3,axis=axis,tetgen_options=options)
    with pytest.raises(ValueError,match='complete labelled domain'):
        partitioned_tetgen(model,target,regions=[7])


def test_cli_publishes_checkpointed_mesh_at_explicit_output_filename(tmp_path):
    pytest.importorskip('triangle');meshio=pytest.importorskip('meshio')
    import os,subprocess,sys
    from pathlib import Path
    np.save(tmp_path/'labels.npy',np.ones((4,4,4),np.uint8))
    config=tmp_path/'recipe.toml';config.write_text('''[geometry]
method="regions"
array="labels.npy"
spacing=[1e-6,1e-6,1e-6]
[meshing]
method="partitioned_tetgen"
directory="checkpointed"
surface_method="voxel_plc"
slab_depth=2
[meshing.tetgen_options]
preserve_surface=true
merge_coplanar=false
region_labelling="interfaces"
maximum_cells=10000
''')
    env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+env.get('PYTHONPATH','')
    destination=tmp_path/'published/custom_name.h5'
    command=subprocess.run([sys.executable,'-m','morqex.modeling','generate',str(config),'--output',str(tmp_path/'source.h5'),
        '--mesh',str(destination),'--mesh-preview',str(tmp_path/'actual_interfaces.html')],env=env,check=True,capture_output=True,text=True)
    assert destination.is_file() and (tmp_path/'checkpointed/mesh.h5').is_file()
    back=meshio.read(destination.with_suffix('.xdmf'))
    assert len(back.points)>0 and len(back.cells_dict['tetra'])>0
    preview=(tmp_path/'actual_interfaces.html').read_text()
    assert 'sections_gzip' in preview and 'complete_surface_face_count' in preview
    status=json.loads(command.stdout.splitlines()[-1])
    assert status['tetrahedra']==len(back.cells_dict['tetra'])
    assert type(status['quality_limits_satisfied']) is bool


def test_content_addressed_checkpoints_keep_different_sources_separate(tmp_path):
    target=tmp_path/'cache'
    a=np.ones((4,4,4),np.uint8)
    options=TetGenOptions.for_adaptive(preserve_surface=True,merge_coplanar=False,maximum_cells=10000)
    first=partitioned_tetgen(from_regions(a),target,surface_method='voxel_plc',slab_depth=8,
        tetgen_options=options,content_addressed=True)
    stamp=first.hdf5.stat().st_mtime_ns
    a[:,:,2:]=7
    second=partitioned_tetgen(from_regions(a),target,surface_method='voxel_plc',slab_depth=8,
        tetgen_options=options,content_addressed=True)
    assert first.hdf5.parent!=second.hdf5.parent
    assert first.hdf5.parent.parent==second.hdf5.parent.parent==target
    assert first.hdf5.stat().st_mtime_ns==stamp
    assert first.selected_regions==(1,) and second.selected_regions==(1,7)
