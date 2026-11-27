"""Independent physical-unit, named-domain and shared-interface checks."""
import json
import numpy as np
import pytest
from morqex.modeling import (from_regions,read_comsol,export_comsol,import_comsol,tiff_voxels,
    material_interfaces,TetGenMesh,export_bundle,dense_electrode,porous_layer,stack_layers)


@pytest.mark.parametrize('unit,factor',[('m',1.),('cm',1e-2),('mm',1e-3),('um',1e-6),('µm',1e-6),('nm',1e-9)])
def test_units_and_real_selection_objects(tmp_path,unit,factor):
    a=np.zeros((2,3,4),np.uint8);a[:,:,2:]=7
    model=from_regions(a,spacing=(.5e-6,)*3);mesh=model.tetrahedralize()
    path=export_comsol(mesh,tmp_path/'mesh.mphtxt',length_unit=unit,
        region_names={0:'electrolyte',7:'active'},selections={'electrode':{'regions':[0,7]}})
    data=read_comsol(path)
    np.testing.assert_allclose(data['points']*factor,mesh.points,rtol=1e-14,atol=1e-21)
    assert data['length_unit']==unit and data['selections']['electrode']['entities']==[1,2]
    assert len(data['tags'])==4 and data['selections']['region_0']['label']=='electrolyte'
    imported=import_comsol(path,length_unit=unit)
    np.testing.assert_allclose(imported.points,mesh.points,rtol=1e-14,atol=1e-21)
    assert imported.report['source_selections']==data['selections']
    assert set(imported.phases)=={1,2}


def test_tiff_import_preserves_original_gray_labels_and_axes(tmp_path):
    import tifffile
    a=np.zeros((5,6,7),np.uint8);a[1:4]=128;a[2,2:4]=255
    path=tmp_path/'source.tif';tifffile.imwrite(path,a,metadata={'axes':'ZYX'},photometric='minisblack')
    model=tiff_voxels(path,spacing=(.5e-6,)*3)
    np.testing.assert_array_equal(model.regions,a)
    assert model.region_names=={0:'gray_0',128:'gray_128',255:'gray_255'}
    assert model.metadata['segmentation_changed'] is False


def test_comsol_import_never_silently_reinterprets_declared_or_missing_units(tmp_path):
    model=from_regions(np.ones((2,3,4),np.uint8),spacing=(.5e-6,)*3)
    path=export_comsol(model.tetrahedralize(),tmp_path/'micrometres.mphtxt',length_unit='um')
    back=import_comsol(path)
    np.testing.assert_allclose(back.points,model.tetrahedralize().points,rtol=1e-14,atol=1e-21)
    assert back.report['geometry']['length_unit_source']=='MORQEX length_unit file comment'
    with pytest.raises(ValueError,match='conflicts'):import_comsol(path,length_unit='m')
    reference=tmp_path/'independent_reference.mphtxt'
    reference.write_text('\n'.join(s for s in path.read_text().splitlines() if not s.startswith('# MORQEX length_unit='))+'\n')
    with pytest.raises(ValueError,match='Explicit length_unit required'):import_comsol(reference)
    np.testing.assert_allclose(import_comsol(reference,length_unit='um').points,back.points)


@pytest.mark.parametrize('morphology',['granular','rods','platelets','agglomerates','bicontinuous'])
def test_native_dense_shapes_have_requested_fractions(morphology):
    model=dense_electrode(morphology,size=48,seed=172)
    assert model.regions.shape==(48,)*3 and set(np.unique(model.regions))=={0,1,2}
    assert abs((model.regions==1).mean()-.55)<=1/model.regions.size
    assert abs((model.regions==2).mean()-.11)<=1/model.regions.size
    assert model.metadata['no_tiling']


def test_stack_regions_and_layer_groups_are_distinct():
    h=(.5e-6,)*3
    anode=porous_layer((4,12,12),spacing=h,name='anode',seed=1)
    separator=porous_layer((3,12,12),spacing=h,name='separator',seed=2)
    cathode=from_regions(np.ones((8,12,12),np.uint8),spacing=h,region_names={1:'cathode_active'})
    model=stack_layers([('anode',anode),('separator',separator),('cathode',cathode)],axis=0)
    assert model.regions.shape==(15,12,12)
    assert model.metadata['layer_slices']=={'anode':[0,4],'separator':[4,7],'cathode':[7,15]}
    groups=model.metadata['layer_region_ids'];assert not set(groups['anode'])&set(groups['separator'])
    for layer,span in model.metadata['layer_slices'].items():
        assert set(np.unique(model.regions[slice(*span)]))==set(groups[layer])
    with pytest.raises(ValueError):stack_layers([('bad',from_regions(np.ones((2,10,12),np.uint8))),('cathode',cathode)])


def test_exchange_roundtrips_domains_and_actual_abaqus_surface_faces(tmp_path):
    pytest.importorskip('meshio')
    a=np.zeros((2,3,4),np.uint8);a[:,:,2:]=7
    base=from_regions(a,spacing=(.5e-6,)*3).tetrahedralize()
    f,m,pairs=material_interfaces(base.cells,base.phases)
    mesh=TetGenMesh(base.points,base.cells,base.phases,facets=f,facet_markers=m)
    report=export_bundle(mesh,tmp_path/'software',region_names={0:'electrolyte',7:'active'},length_unit='um')
    assert report['passed'] and len(report['checks'])==5
    lines=(tmp_path/'software/mesh.inp').read_text().splitlines();surfaces=[];active=False;surface_domain=None
    domains=np.searchsorted(np.unique(base.phases),base.phases)+1
    local={'S1':[0,1,2],'S2':[0,3,1],'S3':[1,3,2],'S4':[2,3,0]}
    for line in lines:
        if line.startswith('*'):
            active=line.startswith('*SURFACE')
            if active:surface_domain=int(line.rsplit('_domain_',1)[1])
            continue
        if active:
            cell,side=[v.strip() for v in line.split(',')]
            assert domains[int(cell)-1]==surface_domain
            surfaces.append(tuple(sorted(base.cells[int(cell)-1,local[side]])))
    assert set(surfaces)==set(map(tuple,np.sort(f,axis=1)))


def test_surface_nets_shared_geometry_and_bounded_volume_constraints():
    pytest.importorskip('vtkmodules.vtkFiltersCore')
    from morqex.modeling.surface_nets import surface_nets_plc
    from morqex.modeling.adaptive import _region_volumes
    a=np.zeros((24,)*3,np.uint8);a[3:15,4:19,5:18]=128;a[12:21,10:20,10:21]=255
    model=from_regions(a,spacing=(.5e-6,)*3)
    surface=surface_nets_plc(model,conserve_volumes=True)
    volumes=_region_volumes(surface.points,surface.facets,surface.markers,surface.metadata['interface_pairs'],surface.seeds)
    for k,v in volumes.items():assert abs(v/((a==k).sum()*np.prod(model.spacing))-1)<1e-6
    assert surface.metadata['volume_projection']['maximum_displacement_voxels']<=.35
    np.testing.assert_allclose(surface.points.min(0),0,atol=1e-20)
    np.testing.assert_allclose(surface.points.max(0),12e-6,atol=1e-20)


def test_partitioned_shared_interface_volume_mesh():
    pytest.importorskip('vtkmodules.vtkFiltersCore')
    from morqex.modeling import adaptive_mesh,AdaptiveMeshOptions
    from morqex.modeling.tetgen_backend import library_path
    try:library_path()
    except RuntimeError as e:pytest.skip(str(e))
    a=np.zeros((18,)*3,np.uint8);a[3:15,3:15,3:15]=128;a[9:16,9:16,9:16]=255
    model=from_regions(a)
    mesh=adaptive_mesh(model,options=AdaptiveMeshOptions(surface_method='surface_nets',conserve_phase_volumes=True,
        surface_partition_voxels=9,vertex_fraction=.4,levels=2,topology_retries=1))
    assert set(mesh.phases)=={0,128,255} and mesh.signed_volumes().min()>0
    np.testing.assert_allclose(mesh.signed_volumes().sum(),18**3*1e-18,rtol=1e-8)


def test_conforming_slab_assembly_preserves_materials(tmp_path):
    pytest.importorskip('triangle')
    pytest.importorskip('vtkmodules.vtkFiltersCore')
    from morqex.modeling import surface_nets_plc,TetGenOptions,tetgen_mesh
    from morqex.modeling.partitioned import cut_cap,slab_plc,assemble_slabs
    import h5py
    a=np.zeros((14,)*3,np.uint8);a[3:11,3:11,3:11]=7
    model=from_regions(a,spacing=(1.,)*3);source=surface_nets_plc(model)
    cap=cut_cap(source.points,source.facets,source.markers,source.metadata['interface_pairs'],6.5)
    paths=[]
    for i,(lo,hi) in enumerate(((0.,6.5),(6.5,14.))):
        slab=slab_plc(source,lo,hi,lower_cap=cap if i else None,upper_cap=None if i else cap)
        assert abs(sum(slab.region_volumes.values())/(14*14*(hi-lo))-1)<1e-10
        m=tetgen_mesh(slab,options=TetGenOptions.for_adaptive(preserve_surface=True,merge_coplanar=False))
        path=tmp_path/f'part{i}.h5';m.save_hdf5(path);paths.append(path)
    full=assemble_slabs(paths,tmp_path/'whole.h5')
    with h5py.File(full.hdf5) as f:
        report=json.loads(f.attrs['report']);assert len(report['cap_checks'])==1
        assert abs(sum(report['phase_volumes_m3'].values())/14**3-1)<1e-10
        assert set(np.unique(f['mesh/phase']))=={0,7}
        p=f['mesh/points_m'][:];faces=f['mesh/facets'][:]
        assert not np.any(np.isclose(p[faces,:,][...,2],6.5).all(1))
    with h5py.File(paths[0],'r+') as f:f['mesh/points_m'][0,0]=np.nan
    with pytest.raises(ArithmeticError,match='Nonfinite slab coordinates'):
        assemble_slabs(paths,tmp_path/'corrupt_must_not_publish.h5')
    assert not (tmp_path/'corrupt_must_not_publish.h5').exists()


def test_empty_assembly_cannot_publish_a_complete_empty_mesh(tmp_path):
    from morqex.modeling import assemble_slabs
    with pytest.raises(ValueError,match='At least one complete slab'):
        assemble_slabs([],tmp_path/'empty.h5')
    assert not (tmp_path/'empty.h5').exists()


def test_interface_labels_come_from_all_constrained_face_sides():
    pytest.importorskip('vtkmodules.vtkFiltersCore')
    from morqex.modeling import surface_nets_plc,tetgen_mesh,TetGenOptions
    a=np.zeros((14,)*3,np.uint8);a[3:11,3:11,3:11]=7;a[5:8,5:8,5:8]=128
    source=surface_nets_plc(from_regions(a))
    options=TetGenOptions.for_adaptive(preserve_surface=True,merge_coplanar=False,region_labelling='interfaces')
    result=tetgen_mesh(source,options=options)
    assert set(result.phases)=={0,7,128}
    assert result.report['native_geometric_compartments']>=3
    np.testing.assert_allclose(result.signed_volumes().sum(),14**3*1e-18,rtol=1e-9)


def test_one_source_controllable_interior_refinement_and_exact_voxel_boundary():
    from morqex.modeling import MeshResolution,controlled_tetgen,TetGenOptions
    source=from_regions(np.full((4,5,6),7,np.uint8),spacing=(1e-6,)*3)
    original=source.regions.copy();meshes=[]
    for count in (200,1600):
        result=controlled_tetgen(source,surface_method='voxel_plc',resolution=MeshResolution('balanced',target_cells=count),
            tetgen_options=TetGenOptions.for_adaptive(maximum_cells=100000))
        assert result.signed_volumes().min()>0
        assert result.signed_volumes().max()<=source.regions.size*1e-18/count*(1+1e-8)
        np.testing.assert_array_equal(source.regions,original)
        np.testing.assert_allclose(result.signed_volumes().sum(),120e-18,rtol=1e-10)
        meshes.append(result)
    assert len(meshes[1].cells)>len(meshes[0].cells)*2
    assert meshes[0].report['resolution_controls']['source_voxel_sha256']==meshes[1].report['resolution_controls']['source_voxel_sha256']


def test_resolution_size_requests_have_explicit_units_and_priority():
    from morqex.modeling import MeshResolution
    for name,n in [('coarse',4000),('balanced',16000),('fine',64000),('very_fine',256000)]:
        surface,native=MeshResolution(name).options(1e-12)
        assert native.maximum_volume==1e-12/n
    _,native=MeshResolution('fine',nominal_edge_m=2e-6).options(1.)
    np.testing.assert_allclose(native.maximum_volume,(2e-6)**3/(6*np.sqrt(2)),rtol=1e-15)
    with pytest.raises(ValueError):MeshResolution(maximum_volume_m3=1.,nominal_edge_m=1.)


def test_last_slab_may_contain_only_one_of_original_materials():
    pytest.importorskip('triangle');pytest.importorskip('vtkmodules.vtkFiltersCore')
    from morqex.modeling import surface_nets_plc,cut_cap,slab_plc
    a=np.zeros((16,)*3,np.uint8);a[3:10,3:10,3:10]=7
    source=surface_nets_plc(from_regions(a,spacing=(1.,)*3))
    cap=cut_cap(source.points,source.facets,source.markers,source.metadata['interface_pairs'],13.5)
    last=slab_plc(source,13.5,16.,lower_cap=cap)
    assert set(last.region_volumes)=={0}
    np.testing.assert_allclose(last.region_volumes[0],2.5*16**2,rtol=1e-12)


def test_exact_voxel_plc_material_orientation_and_divergence_volume():
    from morqex.modeling import voxel_plc
    from morqex.modeling.adaptive import _region_volumes
    a=np.zeros((6,7,8),np.uint8);a[1:5,1:6,1:7]=128;a[2:4,2:5,2:6]=255
    model=from_regions(a);plc=voxel_plc(model);q=plc.facets
    f=np.stack((q[:,[0,1,2]],q[:,[0,2,3]]),axis=1).reshape(-1,3)
    v=_region_volumes(plc.points,f,np.repeat(plc.markers,2),plc.metadata['interface_pairs'],plc.seeds)
    for k,volume in v.items():np.testing.assert_allclose(volume,(a==k).sum()*1e-18,rtol=1e-12)


def test_common_cap_includes_recovery_nodes_from_both_neighbors():
    triangle=pytest.importorskip('triangle')
    from morqex.modeling.partitioned import reconcile_cap
    p=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    segments=np.array([[0,1],[1,2],[2,3],[3,0]])
    original={'points':p,'triangles':np.array([[0,1,2],[0,2,3]]),'phases':np.zeros(2,np.int64),
        'segments':segments,'seeds':np.array([[.5,.5,0]]),'plane':.5,'other':[0,1],'axis':2}
    def cap(point):
        result=triangle.triangulate({'vertices':np.vstack((p,point)),'segments':segments},'pYQ')
        return {'points':result['vertices'],'triangles':result['triangles'],'phases':np.zeros(len(result['triangles']),np.int64)}
    left=cap([.25,.5]);right=cap([.75,.5]);common=reconcile_cap(original,left,right)
    assert common is not None and len(common['points'])==6
    assert reconcile_cap(common,common,common) is None
    assert set(common['phases'])=={0}


def test_imported_int64_cells_and_uint8_phases_have_matching_xdmf_types(tmp_path):
    meshio=pytest.importorskip('meshio')
    base=from_regions(np.full((2,3,4),128,np.uint8)).tetrahedralize()
    faces,markers,pairs=material_interfaces(base.cells,base.phases)
    full=TetGenMesh(base.points,base.cells.astype(np.int64),base.phases.astype(np.uint8),facets=faces,facet_markers=markers)
    artifact=full.save_hdf5(tmp_path/'imported.h5');back=meshio.read(artifact.xdmf)
    np.testing.assert_array_equal(back.cells_dict['tetra'],full.cells)
    np.testing.assert_array_equal(back.cell_data_dict['region']['tetra'],full.phases)


def test_planar_quality_refinement_keeps_all_physical_boundary_nodes():
    pytest.importorskip('triangle');pytest.importorskip('vtkmodules.vtkFiltersCore')
    from morqex.modeling import surface_nets_plc,cut_cap
    source=surface_nets_plc(from_regions(np.zeros((16,)*3,np.uint8),spacing=(1.,)*3))
    base=cut_cap(source.points,source.facets,source.markers,source.metadata['interface_pairs'],7.5)
    fine=cut_cap(source.points,source.facets,source.markers,source.metadata['interface_pairs'],7.5,minimum_angle=20.)
    assert fine['boundary_points_unchanged'] and fine['interior_quality_points']>0
    assert fine['area']==pytest.approx(base['area'],rel=1e-12)


def test_notebook_view_can_isolate_any_present_phase_including_zero():
    a=np.zeros((4,5,6),np.uint8);a[1:3]=7
    model=from_regions(a)
    view=model.viewer(show_region=0)
    assert '"show_region":0' in view.document and 'id="show"' in view.document
    with pytest.raises(ValueError):model.viewer(show_region=99)


@pytest.mark.parametrize('override,expected',[(None,'mm'),('um','um')])
def test_cli_uses_toml_resolution_and_export_units_with_explicit_override(tmp_path,override,expected):
    import os,subprocess,sys
    from pathlib import Path
    array=np.ones((2,3,4),np.uint8);np.save(tmp_path/'labels.npy',array)
    config=tmp_path/'config.toml';config.write_text('''[geometry]
method="regions"
array="labels.npy"
spacing=[0.1,0.1,0.1]
[meshing]
method="controlled_tetgen"
surface_method="voxel_plc"
[meshing.resolution]
precision="coarse"
target_cells=50
[meshing.tetgen_options]
maximum_cells=10000
isolated=true
[exports]
length_unit="mm"
formats=["comsol"]
''')
    env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+env.get('PYTHONPATH','')
    args=[sys.executable,'-m','morqex.modeling','generate',str(config),'--output',str(tmp_path/'vox.h5'),
        '--mesh',str(tmp_path/'mesh.h5'),'--export-dir',str(tmp_path/'software')]
    if override:args+=['--length-unit',override]
    subprocess.run(args,env=env,check=True,capture_output=True,text=True)
    back=read_comsol(tmp_path/'software/mesh.mphtxt');assert back['length_unit']==expected
    scale={'mm':1e-3,'um':1e-6}[expected]
    np.testing.assert_allclose(np.ptp(back['points'],axis=0)*scale,[.4,.3,.2],rtol=1e-12)


def test_requested_phase_fidelity_is_measured_for_raw_contours_too():
    from morqex.modeling import MeshResolution,controlled_tetgen,TetGenOptions
    source=from_regions(np.ones((4,5,6),np.uint8))
    native=TetGenOptions.for_adaptive(maximum_cells=10000)
    with pytest.raises(ArithmeticError,match='Requested source phase-volume fidelity'):
        controlled_tetgen(source,surface_method='marching_cubes',regions=[1],
            resolution=MeshResolution('coarse',target_cells=50,maximum_phase_volume_error=.001),tetgen_options=native)
    result=controlled_tetgen(source,surface_method='marching_cubes',regions=[1],
        resolution=MeshResolution('coarse',target_cells=50,maximum_phase_volume_error=.1),tetgen_options=native)
    error=result.report['resolution_controls']['measured_source_phase_volume_errors']['1']
    assert .001<error<.1
    with pytest.raises(ValueError,match='vertex_fraction requires'):
        controlled_tetgen(source,surface_method='voxel_plc',resolution=MeshResolution(vertex_fraction=.2))


def test_comsol_user_selections_cannot_silently_overwrite_domain_selections(tmp_path):
    source=from_regions(np.ones((2,3,4),np.uint8)).tetrahedralize();target=tmp_path/'collision.mphtxt'
    with pytest.raises(ValueError,match='collides'):
        export_comsol(source,target,selections={'region_1':{'regions':[1]}})
    assert not target.exists()
    export_comsol(source,target,include_region_selections=False,selections={'region_1':{'regions':[1]}})
    assert read_comsol(target)['selections']['region_1']['entities']==[1]


def test_large_face_ownership_hash_is_only_a_verified_candidate_index(monkeypatch):
    import morqex.modeling.exchange as exchange
    source=from_regions(np.ones((2,2,2),np.uint8)).tetrahedralize()
    facets,_,_=material_interfaces(source.cells,source.phases)
    expected,sides=exchange._facet_owners(source.cells,facets,len(source.points))
    actual,actual_sides=exchange._facet_owners(source.cells,facets,3000000)
    np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(actual_sides,sides)
    monkeypatch.setattr(exchange,'_face_hash',lambda f:np.zeros(len(f),np.uint64))
    # Multiple target collisions use exact tuple keys.
    actual,_=exchange._facet_owners(source.cells,facets,3000000)
    np.testing.assert_array_equal(actual,expected)
    # A query collision with the sole target is rejected by node equality.
    actual,_=exchange._facet_owners(source.cells,facets[:1],3000000)
    np.testing.assert_array_equal(actual,expected[:1])


def test_streaming_vtu_is_readable_by_meshio_and_native_vtk(tmp_path):
    meshio=pytest.importorskip('meshio');pytest.importorskip('vtkmodules.vtkIOXML')
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from morqex.modeling.exchange import export_bundle
    a=np.zeros((3,4,5),np.uint8);a[:,2:]=7
    base=from_regions(a,spacing=(1e-6,)*3).tetrahedralize()
    facets,markers,pairs=material_interfaces(base.cells,base.phases)
    full=TetGenMesh(base.points,base.cells,base.phases,facets=facets,facet_markers=markers)
    export_bundle(full,tmp_path/'vtu',formats=['vtu'],length_unit='um')
    file=tmp_path/'vtu/mesh.vtu';back=meshio.read(file)
    np.testing.assert_array_equal(back.cells_dict['tetra'],base.cells)
    np.testing.assert_array_equal(back.cells_dict['triangle'],facets)
    expected_domains=np.searchsorted([0,7],base.phases)+1
    np.testing.assert_array_equal(back.cell_data_dict['domain_id']['tetra'],expected_domains)
    np.testing.assert_array_equal(back.cell_data_dict['boundary_id']['triangle'],markers)
    reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(file));reader.Update();grid=reader.GetOutput()
    assert grid.GetNumberOfCells()==len(base.cells)+len(facets) and grid.GetNumberOfPoints()==len(base.points)
    np.testing.assert_array_equal(vtk_to_numpy(grid.GetPoints().GetData()),base.points/1e-6)
    np.testing.assert_array_equal(vtk_to_numpy(grid.GetCellData().GetArray('domain_id'))[:len(base.cells)],expected_domains)
    from morqex.modeling.vtu_stream import verify_with_vtk
    full.save_hdf5(tmp_path/'source.h5')
    assert verify_with_vtk(file,tmp_path/'source.h5',length_unit='um')['passed']


def test_streaming_medit_is_readable_by_independent_meshio(tmp_path):
    meshio=pytest.importorskip('meshio')
    a=np.zeros((3,4,5),np.uint8);a[:,:,3:]=128
    base=from_regions(a,spacing=(.5e-6,)*3).tetrahedralize()
    f,m,_=material_interfaces(base.cells,base.phases)
    full=TetGenMesh(base.points,base.cells,base.phases,facets=f,facet_markers=m)
    export_bundle(full,tmp_path/'medit',formats=['medit'],length_unit='mm')
    back=meshio.read(tmp_path/'medit/mesh.mesh')
    np.testing.assert_array_equal(back.cells_dict['tetra'],base.cells)
    np.testing.assert_array_equal(back.cells_dict['triangle'],f)
    np.testing.assert_allclose(back.points,base.points/1e-3,rtol=1e-13,atol=1e-20)
    np.testing.assert_array_equal(back.cell_data_dict['medit:ref']['tetra'],np.searchsorted([0,128],base.phases)+1)
    np.testing.assert_array_equal(back.cell_data_dict['medit:ref']['triangle'],m)


def test_quality_cannot_accept_nan_as_a_positive_volume_cell():
    from morqex.modeling import tetra_quality
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    cells=np.array([[0,1,2,3]],np.int32)
    assert tetra_quality(points,cells)[0]['minimum_volume_m3']>0
    points[1,0]=np.nan
    with pytest.raises(ArithmeticError,match='Nonfinite'):tetra_quality(points,cells)


def test_external_output_guard_distinguishes_checkout_and_installed_layout(tmp_path,monkeypatch):
    from morqex.modeling import voxel
    from morqex import workspace
    installation=tmp_path/'installed'
    monkeypatch.setattr(voxel,'__file__',str(installation/'morqex/modeling/voxel.py'))
    monkeypatch.setattr(workspace,'__file__',str(installation/'morqex/workspace.py'))
    assert voxel.external_path(tmp_path/'results/mesh.h5')==tmp_path/'results/mesh.h5'
    assert workspace.Workspace(tmp_path/'results').path('mesh.h5')==tmp_path/'results/mesh.h5'
    with pytest.raises(ValueError,match='outside MORQEX source'):
        voxel.external_path(installation/'morqex/mesh.h5')
    with pytest.raises(ValueError,match='outside MORQEX source'):
        workspace.Workspace(installation/'results')
    checkout=tmp_path/'checkout';checkout.mkdir();(checkout/'pyproject.toml').write_text('')
    monkeypatch.setattr(voxel,'__file__',str(checkout/'src/morqex/modeling/voxel.py'))
    monkeypatch.setattr(workspace,'__file__',str(checkout/'src/morqex/workspace.py'))
    with pytest.raises(ValueError,match='outside MORQEX source'):
        voxel.external_path(checkout/'results/mesh.h5')
    with pytest.raises(ValueError,match='outside MORQEX source'):
        workspace.Workspace(checkout/'results')
    assert voxel.external_path(tmp_path/'external.h5')==tmp_path/'external.h5'
