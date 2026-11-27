import json
import numpy as np
import pytest
from morqex.modeling import from_regions,partitioned_tetgen,TetGenOptions


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('labelling',['seeds','interfaces'])
def test_aligned_exact_caps_retain_material_jumps_and_all_physical_interfaces(tmp_path,axis,labelling):
    """Independent cell-face enumeration checks the complete assembled PLC."""
    import h5py
    from morqex.modeling import material_interfaces
    a=np.zeros((8,8,8),np.uint8)
    a[:,2:6,2:6]=7
    other=[k for k in range(3) if k!=axis]
    selector=[slice(None)]*3;selector[axis]=slice(4,None);selector[other[0]]=slice(2,5)
    a[tuple(selector)]=11
    model=from_regions(a,spacing=(.7e-6,.8e-6,1e-6),origin=(3e-6,-2e-6,5e-6))
    target=tmp_path/f'aligned{axis}'
    options=TetGenOptions.for_adaptive(preserve_surface=True,merge_coplanar=False,region_labelling=labelling)
    artifact=partitioned_tetgen(model,target,surface_method='voxel_aligned',slab_depth=4,axis=axis,tetgen_options=options)
    with h5py.File(artifact.hdf5) as h:
        cells=h['mesh/tetrahedra'][:];phase=h['mesh/phase'][:]
        faces=h['mesh/facets'][:];markers=h['mesh/facet_markers'][:];r=json.loads(h.attrs['report'])
        derived,ids,pairs=material_interfaces(cells,phase)
        def canonical(f,m,p):
            labels=np.array([p[str(int(i))] for i in m]);rows=np.column_stack((np.sort(f,axis=1),np.sort(labels,axis=1)))
            return rows[np.lexsort(rows.T[::-1])]
        np.testing.assert_array_equal(canonical(faces,markers,r['geometry']['interface_pairs']),canonical(derived,ids,pairs))
        assert r['cap_checks'][0]['physical_material_jumps']>0
        assert r['cap_checks'][0]['identical_material_faces']>0
        assert r['source_voxel_cell_centroids_checked']==len(cells)
        assert max(r['source_voxel_phase_volume_errors'].values())<1e-8
    stamp=artifact.hdf5.stat().st_mtime_ns
    again=partitioned_tetgen(model,target,surface_method='voxel_aligned',slab_depth=4,axis=axis,tetgen_options=options)
    assert again.hdf5.stat().st_mtime_ns==stamp


def test_aligned_partition_checks_unknown_region_size_before_local_filter(tmp_path):
    options=TetGenOptions.for_adaptive(preserve_surface=True,region_maximum_volumes={9:1e-18})
    with pytest.raises(ValueError,match='Unknown region'):
        partitioned_tetgen(from_regions(np.zeros((4,4,4),np.uint8)),tmp_path/'bad',
            surface_method='voxel_aligned',tetgen_options=options)


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

@pytest.mark.parametrize('axis',[0,1,2])
def test_aligned_mesh_reproduces_layered_conduction_potential_current_and_energy(tmp_path,axis):
    """Physical transmission test, independent of the mesher's volume checks."""
    import h5py
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    a=np.zeros((8,8,8),np.uint8);selection=[slice(None)]*3;selection[axis]=slice(4,None)
    a[tuple(selection)]=7
    model=from_regions(a,spacing=(.7e-6,.8e-6,1e-6),origin=(3e-6,-2e-6,5e-6))
    artifact=partitioned_tetgen(model,tmp_path/'conduction',surface_method='voxel_aligned',slab_depth=4,axis=axis)
    with h5py.File(artifact.hdf5) as h:
        p=h['mesh/points_m'][:];t=h['mesh/tetrahedra'][:];phase=h['mesh/phase'][:]
    basis=p[t[:,1:]]-p[t[:,:1]];volume=np.linalg.det(basis)/6
    gradient=np.linalg.inv(basis).transpose(0,2,1)
    gradient=np.concatenate((-gradient.sum(1,keepdims=True),gradient),axis=1)
    conductivity=np.where(phase==0,2.,5.)
    local=np.einsum('nki,nli->nkl',gradient,gradient)*(conductivity*volume)[:,None,None]
    matrix=coo_matrix((local.ravel(),(np.repeat(t,4,axis=1).ravel(),np.tile(t,(1,4)).ravel())),shape=(len(p),len(p))).tocsr()
    z=p[:,2-axis]-model.origin[axis];length=8*model.spacing[axis];half=length/2
    ends=np.isclose(z,0.,rtol=0,atol=1e-15)|np.isclose(z,length,rtol=0,atol=1e-15)
    value=np.zeros(len(p));value[np.isclose(z,length,rtol=0,atol=1e-15)]=1.
    free=np.flatnonzero(~ends);fixed=np.flatnonzero(ends)
    value[free]=spsolve(matrix[free][:,free],-matrix[free][:,fixed]@value[fixed])
    resistance_per_area=half/2+half/5
    exact=np.where(z<=half,z/2,half/2+(z-half)/5)/resistance_per_area
    np.testing.assert_allclose(value,exact,rtol=0,atol=1e-9)
    reaction=matrix@value
    area=float(np.prod([8*model.spacing[k] for k in range(3) if k!=axis]))
    current=area/resistance_per_area
    assert reaction[np.isclose(z,length,rtol=0,atol=1e-15)].sum()==pytest.approx(current,rel=1e-9)
    assert reaction[np.isclose(z,0.,rtol=0,atol=1e-15)].sum()==pytest.approx(-current,rel=1e-9)
    assert .5*value@reaction==pytest.approx(.5*current,rel=1e-9)
    assert np.linalg.norm(reaction[free],np.inf)<current*1e-9


def test_aligned_pipeline_recovers_interrupted_part_and_assembly_without_recomputing_good_part(tmp_path):
    import h5py
    a=np.zeros((4,4,4),np.uint8);a[2:]=7
    model=from_regions(a);folder=tmp_path/'recover'
    first=partitioned_tetgen(model,folder,surface_method='voxel_aligned',slab_depth=2)
    stamp=(folder/'part_00.h5').stat().st_mtime_ns
    count=first.tetrahedra
    (folder/'validation.json').unlink();(folder/'part_01.json').unlink()
    with h5py.File(folder/'part_01.h5','r+') as h:h.attrs['complete']=False
    (folder/'mesh.pending').write_bytes(b'interrupted assembly sentinel')
    result=partitioned_tetgen(model,folder,surface_method='voxel_aligned',slab_depth=2)
    assert result.tetrahedra==count and (folder/'part_00.h5').stat().st_mtime_ns==stamp
    with h5py.File(result.hdf5) as h:assert h.attrs['complete']
    assert list((folder/'rejected').glob('*/part_01.h5'))
    assert list((folder/'rejected').glob('*/mesh.h5'))
    assert len(list(folder.glob('rejected_assembly_*.h5')))==1
