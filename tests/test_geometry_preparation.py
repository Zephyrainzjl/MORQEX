import json
import numpy as np
import pytest
from morqex.modeling import (from_regions,SmoothingOptions,smooth_voxels,lower_resolution,
    PreparedGeometry,analyze_features,ProgressBar,TimingHistory,voxel_fingerprint)


def source():
    a=np.zeros((25,27,29),np.uint16);a[5:20,6:21,7:22]=128;a[8:17,9:18,10:19]=255
    a[3:6,12,14]=128;a[16,12,14]=128
    return from_regions(a,spacing=(.4e-6,.6e-6,.8e-6),origin=(3e-6,-4e-6,5e-6))


@pytest.mark.parametrize('method',['none','gaussian','majority','signed_distance','morphology'])
def test_fixed_grid_smoothing_preserves_source_box_and_labels_and_reports_real_changes(method):
    m=source();before=m.regions.copy();signature=voxel_fingerprint(m);events=[]
    p=smooth_voxels(m,options=SmoothingOptions(method=method,acceptance='report'),progress=events.append)
    np.testing.assert_array_equal(m.regions,before)
    assert voxel_fingerprint(m)==signature
    assert p.model.regions.shape==m.regions.shape and p.model.spacing==m.spacing and p.model.origin==m.origin
    assert set(np.unique(p.model.regions))=={0,128,255}
    assert p.scores.shape==(3,*m.regions.shape) and np.isfinite(p.scores).all()
    actual=np.count_nonzero(before!=p.model.regions)
    assert actual==p.report['fidelity']['changed_voxels'] and actual/before.size==p.report['fidelity']['changed_fraction']
    assert events[-1].finished and any(e.total==3 for e in events)
    assert p.validate() is p
    if method=='none':assert actual==0


def test_gaussian_removes_small_extreme_feature_at_identical_grid_size():
    a=np.zeros((25,25,25),np.uint8);a[5:20,5:20,5:20]=1;a[3:5,12,12]=1
    p=smooth_voxels(from_regions(a),options=SmoothingOptions(iterations=1,minimum_feature_voxels=4,acceptance='report'))
    assert p.model.regions.shape==a.shape and p.model.regions[3,12,12]==0
    assert p.model.regions[12,12,12]==1 and p.report['fidelity']['regions']['1']['iou']>.9


def test_strict_fidelity_and_topology_failures_are_not_silently_accepted():
    m=source()
    with pytest.raises(ArithmeticError,match='fidelity/topology'):
        smooth_voxels(m,options=SmoothingOptions(minimum_feature_voxels=4,maximum_changed_fraction=0))
    a=np.zeros((12,12,12),np.uint8);a[1,1,1]=7
    with pytest.raises(ArithmeticError):smooth_voxels(from_regions(a),options=SmoothingOptions(preserve_phase_volumes=False))


def test_complete_prepared_checkpoint_and_tamper_detection(tmp_path):
    p=smooth_voxels(source(),options=SmoothingOptions(method='none'))
    saved=p.save(tmp_path/'prepared.h5');q=PreparedGeometry.load(saved)
    np.testing.assert_array_equal(p.model.regions,q.model.regions);np.testing.assert_array_equal(p.scores,q.scores)
    assert p.report==q.report
    with pytest.raises(FileExistsError):p.save(saved)
    q.scores[0,0,0,0]+=.1
    with pytest.raises(ValueError,match='fields changed'):q.validate()
    p.model.regions[0,0,0]=128
    with pytest.raises(ValueError,match='labels'):p.validate()


@pytest.mark.parametrize('target',[(8,9,10),(13,14,15),(25,27,29)])
def test_resolution_reduction_conserves_physical_box_and_soft_phase_volumes(target):
    m=source();before=m.regions.copy();q,r=lower_resolution(m,shape=target)
    assert q.regions.shape==target
    np.testing.assert_allclose(np.array(m.regions.shape)*m.spacing,np.array(q.regions.shape)*q.spacing,rtol=1e-14)
    assert q.origin==m.origin and r['operation']=='lower_resolution'
    for k in np.unique(before):assert r['phase_score_fractions'][str(k)]==pytest.approx(float(np.mean(before==k)),rel=2e-7)
    np.testing.assert_array_equal(m.regions,before)


def test_reduction_phase_loss_rejected_and_different_from_smoothing():
    a=np.zeros((8,8,8),np.uint8);a[2,2,2]=255;m=from_regions(a)
    with pytest.raises(ArithmeticError,match='removed labels'):lower_resolution(m,factor=4)
    q,r=lower_resolution(m,factor=4,allow_phase_loss=True)
    assert q.regions.shape==(2,2,2) and q.spacing==(4e-6,)*3 and r['lost_regions']==[255]
    with pytest.raises(ValueError,match='upsample'):lower_resolution(m,shape=(9,8,8))


def test_feature_contacts_and_layer_thickness_are_physical_not_region_id_rules():
    a=np.zeros((9,10,11),np.uint16);a[3:6]=128
    m=from_regions(a,spacing=(.5e-6,.6e-6,.7e-6));r=analyze_features(m)
    assert r['regions']['128']['components_6']==1 and r['regions']['128']['spans_zyx']==[False,True,True]
    assert r['contact_areas_m2']['0:128']==pytest.approx(2*10*11*.6e-6*.7e-6)
    assert r['regions']['128']['thickness_proxy_m']['maximum']<=3*.5e-6
    assert r['source_fingerprint']==voxel_fingerprint(m)


def test_progress_json_and_local_eta_calibration(tmp_path):
    from io import StringIO
    stream=StringIO();smooth_voxels(source(),options=SmoothingOptions(method='none'),progress=ProgressBar(stream,json_lines=True,interval=0))
    events=[json.loads(s) for s in stream.getvalue().splitlines()]
    assert events[0]['completed'] is None and events[-1]['finished']
    history=TimingHistory(tmp_path/'timing.json');estimate,source_=history.estimate('smooth',100,10)
    assert estimate==10 and 'uncalibrated' in source_
    history.record('smooth',100,2);estimate,source_=history.estimate('smooth',200,10)
    assert estimate==4 and 'local' in source_


def test_phase_capacity_balance_protects_a_minor_region_without_changing_box():
    z,y,x=np.indices((29,31,33));a=np.zeros(z.shape,np.uint8)
    a[(z-14)**2+(y-15)**2+(x-16)**2<9**2]=128
    a[7:22,14:17,15:18]=255
    m=from_regions(a,spacing=(.4e-6,.6e-6,.8e-6))
    p=smooth_voxels(m,options=SmoothingOptions(sigma_voxels=.8,iterations=2,minimum_component_voxels=0,acceptance='report'))
    assert p.report['volume_balance']['tolerance_met']
    assert max(v['volume_relative_error'] for v in p.report['fidelity']['regions'].values())<=.01
    assert p.model.regions.shape==a.shape and p.model.spacing==m.spacing
    assert len(p.report['preparation_iterations'])==2


def test_physical_and_index_space_anisotropic_smoothing_match():
    m=source();sigma=(.35,.6,.9)
    a=smooth_voxels(m,options=SmoothingOptions(sigma_voxels=sigma,acceptance='report'))
    b=smooth_voxels(m,options=SmoothingOptions(sigma_m=tuple(np.array(sigma)*m.spacing),acceptance='report'))
    np.testing.assert_allclose(a.scores,b.scores,rtol=1e-6,atol=1e-7)
    np.testing.assert_array_equal(a.model.regions,b.model.regions)


def test_small_component_regularization_reports_removed_geometry():
    a=np.zeros((23,23,23),np.uint8);a[5:18,5:18,5:18]=7;a[2,2,2]=7
    p=smooth_voxels(from_regions(a),options=SmoothingOptions(minimum_feature_voxels=2,iterations=1))
    r=p.report['component_regularization']['regions']['7']
    assert r['selected_components']==1 and r['selected_voxels']==1
    assert p.model.regions[2,2,2]==0 and p.model.regions[11,11,11]==7
    assert a[2,2,2]==7


def test_failed_stage_progress_is_explicit_and_stops_html_refresh(tmp_path):
    from morqex.modeling import HTMLProgress
    from morqex.modeling.progress import progress_stage
    events=[];html=HTMLProgress(tmp_path/'progress.html')
    def callback(event):events.append(event);html(event)
    with pytest.raises(RuntimeError):
        with progress_stage('test failure',callback):raise RuntimeError('intentional')
    assert events[-1].finished and events[-1].status=='failed'
    text=(tmp_path/'progress.html').read_text()
    assert 'failed' in text and 'http-equiv="refresh"' not in text


def test_distance_reinitialization_preserves_all_final_labels_and_coordinates():
    model=source();o=dict(iterations=1,sigma_voxels=.5)
    filtered=smooth_voxels(model,options=SmoothingOptions(**o))
    distance=smooth_voxels(model,options=SmoothingOptions(**o,reconstruction='signed_distance'))
    np.testing.assert_array_equal(filtered.model.regions,distance.model.regions)
    assert distance.model.spacing==filtered.model.spacing and distance.model.origin==filtered.model.origin
    actual=np.array(distance.field_labels)[np.argmax(distance.scores,axis=0)]
    np.testing.assert_array_equal(actual,distance.model.regions)
    assert distance.report['score_sha256']!=filtered.report['score_sha256']
    assert distance.validate() is distance


def test_continuous_volume_calibration_changes_only_subvoxel_geometry():
    a=np.zeros((12,12,12),np.uint8);a[4:7,4:7,4:7]=7
    raw=from_regions(a);p=smooth_voxels(raw,options=SmoothingOptions(method='none',continuous_volume_calibration=True))
    np.testing.assert_array_equal(p.model.regions,a)
    r=p.report['continuous_volume_calibration']
    assert r['tolerance_met'] and r['nodal_labels_unchanged']
    assert max(r['nonnegative_nodal_gains'].values())>0
    assert max(r['independent_sampled_relative_errors'].values())<=.02
    np.testing.assert_array_equal(np.array(p.field_labels)[np.argmax(p.scores,axis=0)],a)


def test_named_domain_selections_invalidate_prepared_semantic_cache():
    model=source();model.metadata['selection_groups']={'electrode':[128,255]}
    p=smooth_voxels(model,options=SmoothingOptions(method='none'))
    model.metadata['selection_groups']={'electrode':[128],'interface_material':[255]}
    assert voxel_fingerprint(model)!=p.report['source_fingerprint']
    # Prepared metadata owns a separate top-level dictionary; changing a
    # prepared selection itself invalidates its semantic fingerprint too.
    p.model.metadata['selection_groups']={'different_electrode':[128,255]}
    with pytest.raises(ValueError,match='changed'):p.validate()
