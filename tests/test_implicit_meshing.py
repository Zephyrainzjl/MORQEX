import os
import numpy as np
import pytest
from morqex.modeling import (from_regions,smooth_voxels,SmoothingOptions,ImplicitMeshOptions,
    implicit_mesh,voxel_fingerprint)


@pytest.fixture(scope='module')
def native_workspace():
    from morqex.modeling.cgal_backend import library_path
    workspace=os.environ.get('MORQEX_WORKSPACE')
    if workspace is None:pytest.skip('Use the external resource guard and explicitly compiled CGAL')
    try:library_path(workspace)
    except RuntimeError:pytest.skip('Optional CGAL backend not compiled for these sources')
    return workspace


def nested():
    z,y,x=np.indices((18,18,18));r2=(z-8.5)**2+(y-8.5)**2+(x-8.5)**2
    a=np.zeros((18,18,18),np.uint16);a[r2<6**2]=128;a[r2<4**2]=255
    return from_regions(a,spacing=(.8e-6,1e-6,1.2e-6),origin=(2e-6,-3e-6,5e-6))


def test_raw_input_fem_precision_does_not_change_default_smoothing(native_workspace):
    a=np.zeros((24,24,24),np.uint8);a[:,:,:12]=1
    model=from_regions(a,spacing=(1e-6,)*3);results=[]
    for precision in ('coarse','fine'):
        results.append(implicit_mesh(model,workspace=native_workspace,
            options=ImplicitMeshOptions(precision=precision,isolated=False)))
    coarse,fine=results
    assert coarse.report['geometry_score_sha256']==fine.report['geometry_score_sha256']
    assert coarse.report['preparation']['options']==fine.report['preparation']['options']
    assert coarse.report['preparation']['options']['minimum_feature_voxels']==2.
    assert len(fine.cells)>len(coarse.cells)


def test_small_smoothed_phase_calibration_retains_volume_in_actual_mesh(native_workspace):
    z,y,x=np.indices((32,32,32));a=np.zeros(z.shape,np.uint8)
    a[(z-15.5)**2+(y-15.5)**2+(x-15.5)**2<10**2]=128
    a[13:19,13:19,13:19]=255;a[4:7,15,15]=128
    model=from_regions(a,spacing=(.5e-6,)*3)
    prepared=smooth_voxels(model,options=SmoothingOptions(sigma_voxels=.7,continuous_volume_calibration=True))
    counts=[]
    for precision in ('coarse','fine'):
        mesh=implicit_mesh(prepared,workspace=native_workspace,options=ImplicitMeshOptions(precision=precision,isolated=False,maximum_quality_retries=2))
        assert mesh.report['geometry_limits_satisfied'] and mesh.report['quality_limits_satisfied']
        assert max(mesh.report['processed_voxel_phase_volume_errors'].values())<=.08
        counts.append(len(mesh.cells))
    assert counts[1]>counts[0]


@pytest.mark.parametrize('optimizer',['joint','joint_exude'])
def test_implicit_nested_materials_physical_box_and_quality(native_workspace,optimizer):
    m=nested();signature=voxel_fingerprint(m);p=smooth_voxels(m,options=SmoothingOptions(method='none'))
    result=implicit_mesh(p,workspace=native_workspace,options=ImplicitMeshOptions(precision='fine',optimization=optimizer,isolated=False,maximum_phase_volume_error=.15,optimization_seconds=8))
    assert voxel_fingerprint(m)==signature and p.validate() is p
    assert set(np.unique(result.phases))=={0,128,255}
    assert result.report['quality_limits_satisfied']
    assert result.report['quality']['radius_edge_max']<=2.000002
    assert result.report['quality']['minimum_dihedral_degrees']>=2.999999
    diagnostic=result.report['attempts'][-1]['native_events'][-1]['native_diagnostics']
    assert diagnostic['initial_radius_edge_max']<=result.report['attempts'][-1]['native_radius_edge_ratio']*(1+1e-6)
    assert diagnostic['final_radius_edge_max']==pytest.approx(result.report['quality']['radius_edge_max'],rel=1e-9)
    assert result.report['geometry']['full_grid_retained'] and not result.report['geometry']['resolution_reduction_applied']
    assert all(n>=1 for n in result.report['native_face_connected_components'].values())
    low=np.array(m.origin[::-1]);high=low+np.array(m.regions.shape[::-1])*m.spacing[::-1]
    assert np.all(result.points>=low-1e-10) and np.all(result.points<=high+1e-10)
    from morqex.modeling.exchange import _facet_owners
    owners,_=_facet_owners(result.cells,result.facets,len(result.points))
    for marker,pair in result.report['geometry']['interface_pairs'].items():
        selected=owners[result.facet_markers==int(marker)]
        actual=np.column_stack((result.phases[selected[:,0]],np.where(selected[:,1]<0,-1,result.phases[np.maximum(selected[:,1],0)])))
        assert np.all(np.sort(actual,axis=1)==pair)


def test_precision_changes_the_mesh_without_resmoothing_or_rebinning(native_workspace):
    p=smooth_voxels(nested(),options=SmoothingOptions(method='gaussian',sigma_voxels=.5))
    original=p.report['score_sha256'];counts=[]
    for precision in ('coarse','fine'):
        result=implicit_mesh(p,workspace=native_workspace,options=ImplicitMeshOptions(precision=precision,isolated=False,maximum_phase_volume_error=.2,optimization_seconds=8))
        counts.append(len(result.cells));assert result.report['geometry_score_sha256']==original
        assert result.report['geometry']['source_shape_zyx']==[18,18,18]
    assert counts[1]>counts[0]


def test_arbitrary_three_phase_junctions_and_isolated_worker(native_workspace):
    a=np.zeros((14,14,14),np.uint8);a[:,:,7:]=7;a[:,7:,7:]=21
    model=from_regions(a,spacing=(1.,)*3)
    result=implicit_mesh(model,workspace=native_workspace,smoothing=SmoothingOptions(method='none'),
        options=ImplicitMeshOptions(isolated=True,optimization_seconds=8,maximum_phase_volume_error=.1))
    assert result.report['quality_limits_satisfied'] and set(np.unique(result.phases))=={0,7,21}
    assert all(v==1 for v in result.report['native_face_connected_components'].values())
    assert result.report['attempts'][0]['native_events']


def test_independent_affine_conduction_patch_on_generated_mesh(native_workspace):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from morqex.modeling.exchange import _facet_owners
    m=from_regions(np.ones((12,13,14),np.uint8),spacing=(1.,)*3,origin=(3.,-2.,5.))
    result=implicit_mesh(m,workspace=native_workspace,smoothing=SmoothingOptions(method='none'),
        options=ImplicitMeshOptions(isolated=False,optimization_seconds=8))
    points=result.points;c=result.cells;x=points[c]
    matrix=np.concatenate((np.ones((*x.shape[:2],1)),x),axis=2)
    gradient=np.linalg.inv(matrix)[:,1:,:].transpose(0,2,1)
    volume=result.signed_volumes();local=volume[:,None,None]*np.einsum('nik,njk->nij',gradient,gradient)
    A=sparse.coo_matrix((local.ravel(),(np.repeat(c,4,axis=1).ravel(),np.tile(c,(1,4)).ravel())),shape=(len(points),)*2).tocsr()
    owners,_=_facet_owners(c,result.facets,len(points));boundary=np.unique(result.facets[owners[:,1]<0])
    free=np.setdiff1d(np.arange(len(points)),boundary);direction=np.array([.2,-.3,.7]);exact=points@direction
    u=exact.copy();u[free]=spsolve(A[free][:,free],-A[free][:,boundary]@u[boundary])
    np.testing.assert_allclose(u,exact,rtol=1e-10,atol=1e-10)
    assert float(u@A@u)/2==pytest.approx(float(volume.sum())*float(direction@direction)/2,rel=1e-10)
    assert np.linalg.norm((A@u)[free])<1e-9


def test_native_budget_rejection_preserves_the_original(native_workspace):
    m=nested();original=m.regions.copy()
    with pytest.raises(RuntimeError,match='vertices|maximum|budget'):
        implicit_mesh(m,workspace=native_workspace,options=ImplicitMeshOptions(isolated=False,maximum_nodes=10,maximum_cells=30),smoothing=SmoothingOptions(method='none'))
    np.testing.assert_array_equal(m.regions,original)


def test_toml_workflow_checkpoint_resume_export_and_hdf5_generator(native_workspace,tmp_path):
    import json
    from morqex.modeling import run_geometry
    m=nested();m.save(tmp_path/'input.h5')
    recipe=tmp_path/'geometry.toml'
    recipe.write_text('''[geometry]
method="hdf5"
path="input.h5"
[preparation]
method="none"
[meshing]
method="implicit_adaptive"
[meshing.options]
isolated=false
optimization_seconds=8.0
maximum_phase_volume_error=0.15
[exports]
formats=["comsol","vtu","gmsh","medit","abaqus"]
length_unit="um"
''')
    first=run_geometry(recipe,tmp_path/'run',workspace=native_workspace,preview=False,progress=lambda event:None)
    assert first.report['exchange']['passed'] and not first.report['mesh_cached']
    second=run_geometry(recipe,tmp_path/'run',workspace=native_workspace,preview=False,progress=lambda event:None)
    assert first.directory==second.directory and second.report['mesh_cached'] and second.report['preparation_cached']
    assert second.report['mesh_sha256']==first.report['mesh_sha256']
    assert second.mesh.nodes==first.mesh.nodes
    manifest=json.loads((second.directory/'mesh_checkpoint.json').read_text())
    assert manifest['mesh_report']['quality_limits_satisfied']
    m.metadata['material_revision']='independent_change'
    m.save(tmp_path/'input.h5')
    third=run_geometry(recipe,tmp_path/'run',workspace=native_workspace,preview=False,progress=lambda event:None)
    assert third.directory!=first.directory and not third.report['preparation_cached']
    from morqex.modeling import PreparedGeometry
    assert PreparedGeometry.load(third.prepared).model.metadata['material_revision']=='independent_change'


def test_volume_failure_refines_only_the_mesh_and_retries_with_identical_geometry(monkeypatch,tmp_path):
    # A controlled, valid two-material Tet4 problem isolates the adaptive
    # acceptance logic from the native mesher's own randomness.
    from morqex.modeling import implicit as module
    from morqex.modeling import cgal_backend
    a=np.zeros((2,2,2),np.uint8);a[:,:,1]=7;m=from_regions(a)
    prepared=smooth_voxels(m,options=SmoothingOptions(method='none'));signature=prepared.report['score_sha256']
    base=m.tetrahedralize();calls=[]
    def native(scores,sizing,spacing,config,limits,seeds,workspace,callback):
        points=base.points/1e-6;points=points.copy()
        if not calls:points[points[:,0]==1,0]=1.2
        calls.append(config.copy())
        return points,base.cells.astype(np.uint32),np.where(base.phases==0,1,2).astype(np.int32),np.ones(2,np.uint64)
    monkeypatch.setattr(cgal_backend,'library_path',lambda workspace=None:__file__)
    monkeypatch.setattr(module,'_native',native)
    result=implicit_mesh(prepared,workspace=tmp_path,options=ImplicitMeshOptions(isolated=False,optimization_seconds=0,maximum_quality_retries=1))
    assert len(calls)==2 and calls[1][2]==pytest.approx(calls[0][2]/2)
    assert calls[1][4]==calls[0][4]  # shape already passed; radius unchanged
    assert not result.report['attempts'][0]['geometry_passed']
    assert result.report['attempts'][1]['geometry_passed'] and result.report['quality_limits_satisfied']
    assert prepared.report['score_sha256']==signature and prepared.validate() is prepared
    np.testing.assert_array_equal(m.regions,a)
