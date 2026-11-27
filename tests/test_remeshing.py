"""Measured optional MMG size control, labels, topology and isolated failure."""
import json
import numpy as np
import pytest
from morqex.modeling import from_regions,TetGenMesh,material_interfaces,RemeshOptions,remesh,remesh_from_toml


def _example():
    a=np.zeros((8,8,8),np.uint8);a[:,:,4:]=7
    base=from_regions(a,spacing=(1e-6,)*3).tetrahedralize()
    f,m,pairs=material_interfaces(base.cells,base.phases)
    return TetGenMesh(base.points,base.cells,base.phases,facets=f,facet_markers=m,report={'geometry':{'interface_pairs':pairs}})


def _available():
    from morqex.modeling.mmg_backend import library_path
    try:return library_path()
    except RuntimeError:pytest.skip('Optional MMG backend not explicitly compiled')


def test_same_mesh_mmg_coarse_fine_retains_materials_topology_and_units(tmp_path):
    _available();source=_example();original=source.points.copy();original_cells=source.cells.copy()
    results=[]
    for edge in (3e-6,1.5e-6):
        config=tmp_path/(str(edge)+'.toml')
        config.write_text(f'[remeshing]\nuniform_edge_m={edge}\nhausdorff_m=1e-8\nmaximum_phase_volume_error=1e-8\n')
        result=remesh_from_toml(source,config)
        assert set(result.phases)=={0,7}
        assert result.report['native_process_isolated']
        assert result.report['geometry']['surface_method']=='MMG-adapted material interfaces'
        assert result.report['input_geometry']==source.report['geometry']
        assert result.report['surface_topology_before']==result.report['surface_topology_after']
        assert max(result.report['phase_volume_errors_relative_to_input_mesh'].values())<1e-8
        np.testing.assert_allclose(np.ptp(result.points,axis=0),8e-6,rtol=1e-10)
        results.append(result)
    assert len(results[0].cells)<len(source.cells)<len(results[1].cells)
    np.testing.assert_array_equal(source.points,original);np.testing.assert_array_equal(source.cells,original_cells)


@pytest.mark.parametrize('parameters',[{'uniform_edge_m':0.},{'minimum_edge_m':2.,'maximum_edge_m':1.},
    {'uniform_edge_m':1.,'optimize_only':True},{'gradation':1.},{'maximum_cells':False},{'maximum_phase_volume_error':0.}])
def test_remesh_option_contract(parameters):
    with pytest.raises(ValueError):RemeshOptions(**parameters)


def test_mmg_refuses_silent_budget_and_float_connectivity(tmp_path):
    source=_example()
    with pytest.raises(ValueError,match='Input exceeds'):remesh(source,options=RemeshOptions(maximum_cells=10))
    source.cells=source.cells.astype(float)
    with pytest.raises(ValueError,match='Integer connectivity'):remesh(source)
    with pytest.raises(ValueError,match='remeshing'):
        p=tmp_path/'empty.toml';p.write_text('');remesh_from_toml(source,p)


def test_mmg_vendored_core_matches_recorded_upstream_hashes():
    from pathlib import Path
    import hashlib
    from morqex.modeling.mmg_backend import VENDOR
    provenance=json.loads((VENDOR/'provenance.json').read_text())
    assert provenance['version']=='5.8.0' and provenance['core_modifications']==[]
    assert (VENDOR/'COPYING.LESSER').is_file()
    for item in provenance['files']:
        assert hashlib.sha256((VENDOR/item['path']).read_bytes()).hexdigest()==item['sha256']


def test_file_mmg_loads_input_only_in_worker_and_separates_output_budget(tmp_path):
    _available()
    from morqex.modeling import remesh_artifact
    import h5py,meshio
    original=_example();source=original.save_hdf5(tmp_path/'source.h5')
    target=tmp_path/'published/custom_coarse.h5'
    config=tmp_path/'remesh.toml';config.write_text('''[remeshing]
uniform_edge_m=3e-6
hausdorff_m=1e-8
maximum_cells=1000
maximum_input_cells=5000
maximum_phase_volume_error=1e-8
''')
    result=remesh_from_toml(source,config,output=target)
    assert result.tetrahedra<1000<source.tetrahedra
    with h5py.File(result.hdf5) as h:
        report=json.loads(h.attrs['report'])
        assert report['native_process_isolated'] and report['options']['isolated']
        assert 'only inside isolated worker' in report['input_loading']
    back=meshio.read(result.xdmf);assert len(back.cells_dict['tetra'])==result.tetrahedra
    with pytest.raises(ValueError,match='explicit external output'):remesh_from_toml(source,config)
    with pytest.raises(ValueError,match='Input exceeds'):
        remesh_artifact(source,tmp_path/'invalid.h5',options=RemeshOptions(maximum_cells=1000))
