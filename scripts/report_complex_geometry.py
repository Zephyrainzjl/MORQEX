"""Publish measured geometry qualification; keep unmet shape targets visible."""
import argparse,csv,html,json,os
from pathlib import Path
import xml.etree.ElementTree as ET

CASES=[('reference','Independent COMSOL / 独立参考','reference_comsol'),
       ('nmc','Original NMC TIFF / 原始NMC','nmc_tiff/exact_voxel_partitioned')]+[
       (name,name,'native/'+name+'/partitioned') for name in
       ('granular','rods','platelets','agglomerates','bicontinuous','cell_stack')]


def read(path):return json.loads(path.read_text()) if path.exists() else None


def latest_run(work,label):
    paths=sorted((work/'runs').glob('*_'+label+'/manifest.json'))
    if not paths:return {'stage':label,'status':'pending'}
    p=paths[-1];r=read(p)
    return {'stage':label,'status':'running' if 'exit_code' not in r else 'passed' if r.get('exit_code')==0 else 'failed',
        'manifest':str(p),'wall_seconds':r.get('wall_seconds'),
        'peak_rss_gib':r.get('peak_group_rss_bytes_sampled',0)/2**30,
        'source_snapshot':r.get('source_snapshot',{}),'resources':r['resources'],
        'stopped_because':r.get('stopped_because'),'source_changed_during_run':r.get('source_changed_during_run')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'))
    a=p.parse_args()
    if a.workspace is None:p.error('Specify an external --workspace')
    work=a.workspace.resolve();out=work/'nmc_complex_gallery'
    from morqex.modeling.voxel import external_path
    external_path(out)
    from complex_geometry_gallery import write_index
    write_index(out)
    models=[]
    for key,name,relative in CASES:
        path=out/relative;r=read(path/'validation.json')
        if r is None:models.append({'id':key,'name':name,'status':'pending'});continue
        report=r.get('report',r);q=report.get('quality',{})
        if not q and key=='reference':
            import h5py
            with h5py.File(path/'mesh.h5') as h:report=json.loads(h.attrs['report']);q=report['quality']
        errors=report.get('source_plc_phase_volume_errors',{})
        models.append({'id':key,'name':name,'nodes':r.get('nodes',report.get('nodes')),
            'tetrahedra':r.get('tetrahedra',report.get('tetrahedra')),
            'shape_target_met':report.get('quality_limits_satisfied'),
            'radius_edge_max':q.get('radius_edge_max'),'minimum_dihedral_degrees':q.get('minimum_dihedral_degrees'),
            'maximum_assembly_volume_error_vs_plc':max(errors.values()) if errors else None,
            'report':str(path/'validation.json'),'viewer':relative+'/mesh.html',
            'actual_roi_display':read(path/'roi_mesh.json'),
            'scope':'imported independent reference' if key=='reference' else 'whole-domain geometric conformity; shape checked separately'})
    exchange=read(out/'exchange_validation.json');browser=read(out/'browser_validation.json')
    dashboard=read(out/'dashboard_browser_validation.json')
    aligned_browser=read(out/'aligned_dashboard_browser_validation.json')
    scalar_verification=read(work/'geometry_transmission/validation.json')
    public_nmc=read(out/'nmc_tiff/current_api_aligned_optimized/public_api_validation.json') or read(out/'nmc_tiff/current_api_aligned/public_api_validation.json')
    alignment=read(out/'comparison_alignment.json')
    first=read(work/'benchmarks/tetgen_output/all_faces.json')
    second=read(work/'benchmarks/tetgen_output/constrained.json')
    output_benchmark=None
    if first and second:
        runs=[latest_run(work,label) for label in ('tetgen_output_all_faces_benchmark','tetgen_output_constrained_benchmark')]
        output_benchmark={'passed':first['input_sha256']==second['input_sha256'] and first['hashes']==second['hashes'],
            'input_cells':first['input_cells'],'output_cells':first['cells'],
            'all_faces':first['raw_returned_faces'],'constrained_faces':second['raw_returned_faces'],
            'native_and_validation_seconds':[first['native_and_validation_seconds'],second['native_and_validation_seconds']],
            'peak_rss_gib':[r.get('peak_rss_gib') for r in runs],
            'actual_switches':[first['switches'],second['switches']],
            'input_sha256':first['input_sha256'],'output_hashes':first['hashes'],'runs':runs,
            'scope':'same complete cross-section slab PLC, FP64, backend, tolerance and CPU; one measured execution per mode, not a statistical or whole-battery speedup claim'}
    stages=[latest_run(work,label) for label in ('geometry_release_regression','geometry_package_release_pure',
        'geometry_package_release_native','geometry_notebook_release','current_whole256_public_toml',
        'current_full_api_validation','current_full_nmc_toml','geometry_dashboard_browser','geometry_aligned_dashboard_browser',
        'geometry_scalar_transmission')]
    aligned=[]
    for key in ('nmc','granular','rods','platelets','agglomerates','bicontinuous','cell_stack'):
        folder=out/('nmc_tiff' if key=='nmc' else 'native/'+key)/'voxel_aligned'
        r=read(folder/'validation.json')
        if r is None:aligned.append({'id':key,'status':'pending'});continue
        bundles=[read(folder/'software'/fmt/'bundle.json') for fmt in ('comsol','vtu','gmsh','medit','abaqus')]
        aligned.append({'id':key,'nodes':r['nodes'],'tetrahedra':r['tetrahedra'],
            'quality':r['quality'],'shape_target_met':r['quality_limits_satisfied'],
            'maximum_source_voxel_phase_volume_error':max(r['source_voxel_phase_volume_errors'].values()),
            'source_voxel_cell_centroids_checked':r['source_voxel_cell_centroids_checked'],
            'native_vtk_readback':read(folder/'software/vtu/native_vtk_readback.json'),
            'actual_roi_display':read(folder/'roi_display.json'),
            'report':str(folder/'validation.json'),'format_checks':[c for b in bundles if b and b.get('passed') for c in b['checks']],
            'scope':'exact unchanged source voxels; complete centroid and phase-volume checks; shape acceptance separate'})
    aligned_runs=[latest_run(work,label) for label in ('aligned_whole256_granular','aligned_whole_nmc',
        'aligned_whole_rods','aligned_whole_platelets','aligned_whole_agglomerates','aligned_whole_bicontinuous','aligned_whole_cell_stack')]
    testfile=work/'reports/geometry_release_regression.xml';tests=None
    if testfile.exists():
        suite=ET.parse(testfile).getroot();suites=[suite] if suite.tag=='testsuite' else list(suite)
        tests={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        tests['passed']=tests['tests']-tests['failures']-tests['errors']-tests['skipped']
    candidates=[]
    for label in ('whole_mmg_granular_coarse','mmgs_whole_granular_surface','mmgs_whole_granular_nonmanifold',
        'mmgs_whole_relaxed_geometry','complete_reference_mmg_coarse_extended',
        'granular_slab_unfrozen_planar_subdivision','granular_slab_native_seed_merged_quality',
        'whole_granular_local_cavity_quality','whole_granular_global_delaunay_restore','adjacent_blocks_global_delaunay_restore',
        'full_reference_mmg_relaxed_hausdorff','whole256_weighted_surface_nets','whole256_weighted_native_constrained_faces',
        'monolithic_whole256_public_api','monolithic_whole_file_worker','compartment_graph_full_reference',
        'compartment_graph_whole256','compartment_graph_full_nmc','mmgs_whole_component_5m_budget',
        'aligned_quality_native_optimization','aligned_quality_native_optimization_corrected',
        'aligned_quality_native_optimization_safe_flags','aligned_quality_min5_default_optimizer',
        'aligned_boundary_interior_centres','aligned_cavity_part0_two_rings','aligned_checkerboard_quality','aligned_mmg_optimize_only',
        'aligned_tighter_native_q12','aligned_tighter_native_q1'):
        r=latest_run(work,label)
        if r['status']!='pending':candidates.append(r)
    unmet=[r['id'] for r in models if r.get('shape_target_met') is False]
    pending=[r['id'] for r in models if r.get('status')=='pending']
    aligned_unmet=[r['id'] for r in aligned if r.get('shape_target_met') is False]
    aligned_pending=[r['id'] for r in aligned if r.get('status')=='pending']
    result={'scope':'geometry/software qualification, not a battery-physics or publication certificate',
        'models':models,'exchange':exchange,'browser':browser,'dashboard_browser':dashboard,
        'aligned_dashboard_browser':aligned_browser,'comparison_alignment':alignment,'regression':tests,'stages':stages,
        'public_nmc_toml_publication':public_nmc,'scalar_transmission_verification':scalar_verification,
        'native_output_benchmark':output_benchmark,
        'unmet_shape_requests':unmet,'pending_models':pending,
        'voxel_aligned_models':aligned,'unmet_aligned_shape_requests':aligned_unmet,'pending_aligned_models':aligned_pending,
        'voxel_aligned_runs':aligned_runs,
        'all_shape_targets_passed':not (unmet or pending or aligned_unmet or aligned_pending),
        'candidate_runs_not_accepted_as_final_meshes':candidates,
        'units':{'canonical_mesh':'metres (SI)','reference_extent_m':'approximately1, as confirmed by owner',
            'tiff_spacing_m':5e-7,'tiff_spacing_status':'explicit modelling choice; not scan calibration',
            'tiff_phase_roles':'unassigned original gray0/128/255'},
        'limitations':['COMSOL/Abaqus application or solver execution not performed',
            'synthetic morphologies are not calibrated NMC reconstructions',
            'assembly volume error is relative to input PLC, not a pointwise surface-error bound',
            'complete HDF5 meshes are retained; HTML may sample actual faces for display',
            'mesh conformity and shape checks do not establish PDE convergence']}
    (out/'acceptance.json').write_text(json.dumps(result,indent=2))
    with (out/'mesh_catalogue.csv').open('w',newline='') as f:
        keys=['id','name','nodes','tetrahedra','shape_target_met','radius_edge_max','minimum_dihedral_degrees','maximum_assembly_volume_error_vs_plc']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(models)
    esc=lambda v:html.escape(str(v))
    fmt=lambda x: '—' if x is None else f'{x:,.0f}' if isinstance(x,int) else f'{x:.5g}'
    rows=[]
    for r in models:
        status=r.get('shape_target_met');label='PENDING / 未完成' if r.get('status')=='pending' else 'PASS / 通过' if status is True else 'UNMET / 未通过' if status is False else 'No request / 未指定'
        rows.append('<tr><td><a href="index.html#case='+r['id']+'">'+esc(r['name'])+'</a></td>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k in ('nodes','tetrahedra','radius_edge_max','minimum_dihedral_degrees'))+
            '<td class="'+('bad' if status is False or r.get('status')=='pending' else 'good')+'">'+label+'</td><td>'+fmt(r.get('maximum_assembly_volume_error_vs_plc'))+'</td></tr>')
    stage_rows=['<tr><td>'+esc(r['stage'])+'</td><td>'+esc(r['status'])+'</td><td>'+fmt(r.get('wall_seconds'))+'</td><td>'+fmt(r.get('peak_rss_gib'))+'</td></tr>' for r in stages+aligned_runs]
    ecount=len(exchange.get('checks',[])) if exchange else 0;bcount=len(browser.get('checks',[])) if browser else 0
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><title>MORQEX · measured geometry qualification</title>
<style>body{font:15px system-ui;color:#173b4b;background:#edf3f6;margin:24px}main{max-width:1500px;margin:auto}section{padding:22px;background:white;margin:18px 0;border-radius:14px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce7ec}a{color:#096d83}.bad{color:#af3228}.good{color:#146c4e}nav{display:flex;gap:18px;flex-wrap:wrap}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:900px){table{font-size:12px}td,th{padding:5px}body{margin:10px}}</style>
<main><h1>MORQEX · measured geometry qualification / 实测几何验收</h1>
<nav><a href="index.html">Interactive structures / 交互结构</a><a href="acceptance.json">Full evidence / 完整证据JSON</a><a href="mesh_catalogue.csv">Mesh catalogue / 网格目录CSV</a></nav>
<section><h2>Separate acceptance gates / 分项验收</h2><p>__COUNTS__</p>
<p class="bad">__SHAPE__</p><p>The complete TIFF is252×252×253. Five independently generated morphology families are256³; the layered example is256×256×340. The COMSOL reference is a separate geometry with ≈1m extent. TIFF spacing0.5µm is a caller choice, not measured calibration; its gray labels remain unassigned.</p>
<p>完整TIFF不是参考COMSOL的对应体素，不能将二者逐点比较。合成形貌不等于真实材料重构。几何共形、相体积、单元形状、格式兼容和物理离散收敛分别检查；本报告不宣称完成电池物理或期刊验收。</p></section>
<section><h2>Complete meshes / 完整体网格</h2><table><thead><tr><th>Structure</th><th>Nodes</th><th>Tet4</th><th>Max radius/edge</th><th>Min angle °</th><th>Requested shape</th><th>Assembly volume error vs PLC</th></tr></thead><tbody>__ROWS__</tbody></table>
<p>Volume errors are relative to the mesher's input surface; they do not certify pointwise CT fidelity. Exact voxel boundaries and approximated Surface Nets are separate APIs. All source and volume arrays stay complete; the gallery labels any display sampling.</p>
<p>体积误差以输入PLC为参考，不是逐点CT误差界。原始阶梯体素界面、光滑共享曲面和加权ACVD分别保留；界面共形不代表单元形状达标。</p></section>
<section><h2>Reproducible bounded runs / 有限资源复现</h2><table><thead><tr><th>Stage</th><th>Status</th><th>Wall s</th><th>Peak process-group GiB</th></tr></thead><tbody>__STAGES__</tbody></table>
<p>Manifests retain source snapshots, one-CPU limits, memory caps and actual peak use. Geometry qualification uses no GPU. 未执行COMSOL/Abaqus应用或物理求解器导入；完整格式回读与求解器验证分开。</p></section>
<section><h2>Rejected or incomplete candidates / 未作为最终结果的候选</h2><p>Process exit0 means execution completed; it does not mean shape, topology or geometry acceptance. Candidate geometry and logs stay external; none replaces a qualified source mesh solely because it ran.</p><pre>__CANDIDATES__</pre></section></main></html>'''
    tests_text='pending' if tests is None else f"{tests['passed']}/{tests['tests']} passed; {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped"
    dcount=len(dashboard.get('checks',[])) if dashboard else 0
    aligned_formats=sum(len(r.get('format_checks',[])) for r in aligned)
    aligned_vtk=sum(bool(r.get('native_vtk_readback',{}).get('passed')) for r in aligned if r.get('native_vtk_readback'))
    aligned_dash_count=len(aligned_browser.get('checks',[])) if aligned_browser else 0
    aligned_roi_count=len(aligned_browser.get('roi_actual_face_checks',[])) if aligned_browser else 0
    doc=doc.replace('__COUNTS__',f'{ecount} original-route + {aligned_formats} exact-route full-model format round trips · {aligned_vtk}/7 new native VTK readbacks · {bcount} original offline interface/section checks · {dcount} original + {aligned_dash_count} exact-route dashboard structures · {aligned_roi_count}/7 complete actual ROI views · regression: {tests_text}')
    notices=[]
    if unmet:notices.append('Original-route shape requests UNMET: '+', '.join(unmet))
    if aligned_unmet:notices.append('Aligned-route shape requests UNMET: '+', '.join(aligned_unmet))
    if pending:notices.append('Pending original models: '+', '.join(pending))
    if aligned_pending:notices.append('Pending aligned models: '+', '.join(aligned_pending))
    shape_text='; '.join(notices) if notices else 'All requested shape targets met.'
    doc=doc.replace('__SHAPE__',shape_text)
    aligned_rows=[]
    for r in aligned:
        q=r.get('quality',{})
        status='PENDING' if r.get('status')=='pending' else 'PASS' if r['shape_target_met'] else 'UNMET / 未通过'
        aligned_rows.append('<tr><td>'+esc(r['id'])+'</td>'+''.join('<td>'+fmt(v)+'</td>' for v in
            (r.get('nodes'),r.get('tetrahedra'),q.get('radius_edge_max'),q.get('minimum_dihedral_degrees'),
             r.get('source_voxel_cell_centroids_checked'),r.get('maximum_source_voxel_phase_volume_error')))+
            '<td>'+str(len(r.get('format_checks',[])))+'/5</td><td class="'+('good' if status=='PASS' else 'bad')+'">'+status+'</td></tr>')
    aligned_section='<section><h2>Exact voxel-aligned route / 精确体素边界分区</h2><p><a href="voxel_aligned.html">Open paired source/Tet4 views / 打开体素与真实单元切面对照</a></p><table><tr><th>Structure</th><th>Nodes</th><th>Tet4</th><th>Max R/edge</th><th>Min angle °</th><th>All centroids checked</th><th>Source phase-volume error</th><th>Formats</th><th>Shape request</th></tr>'+''.join(aligned_rows)+'</table><p>Original voxel labels and geometry are unchanged. Canonical integer-lattice caps retain physical material jumps; every cell centroid is cross-checked against the source. This route is distinct from the smooth shared surfaces above. 单元中心检查不是相标签赋值方法，也不是严格逐点几何误差界；原始体素几何、光滑共享曲面与形状请求分别验收。</p></section>'
    doc=doc.replace('<section><h2>Rejected or incomplete candidates',aligned_section+'<section><h2>Rejected or incomplete candidates')
    if scalar_verification:
        verified=scalar_verification['passed_error_reduction']
        scalar_section='<section><h2>Independent scalar transmission verification / 独立导电验证</h2><p><a href="../geometry_transmission/index.html">Analytic sphere solution and measured convergence / 球形夹杂解析解与实测收敛</a></p><p>'+('Error reduction checked / 误差下降已检查' if verified else 'Error-reduction gate UNMET / 误差下降门槛未通过')+'</p><p>This benchmark includes voxel-boundary and P1 error, using quadrature estimates. The Dirichlet functional is half the dissipation, not stored free energy. It is not a calibrated battery discharge or a universal full-electrode PDE certificate. 此处独立验证材料传输问题，不冒充电池放电、严格误差界或所有电极的物理认证。</p></section>'
        doc=doc.replace('<section><h2>Rejected or incomplete candidates',scalar_section+'<section><h2>Rejected or incomplete candidates')
    if output_benchmark:
        b=output_benchmark
        compared='PASS / 哈希一致' if b['passed'] else 'FAIL / 哈希不一致'
        benchmark='<section><h2>Same volume mesh, smaller temporary output / 同一体网格，更少临时输出</h2><p>'+compared+' · '+fmt(b['output_cells'])+' Tet4</p><table><tr><th>Output mode</th><th>Raw returned faces</th><th>Native + validation s</th><th>Peak group GiB</th></tr>'
        for i,(name,key) in enumerate((('All faces / 全部面','all_faces'),('Constrained faces / 约束界面','constrained_faces'))):
            benchmark+='<tr><td>'+name+'</td><td>'+fmt(b[key])+'</td><td>'+fmt(b['native_and_validation_seconds'][i])+'</td><td>'+fmt(b['peak_rss_gib'][i])+'</td></tr>'
        benchmark+='</table><p>Native PLC output already includes material interfaces. Removing -f avoids exporting ordinary interior faces; input, nodes, volume connectivity, material labels and canonical marked-face sets are checked by hashes. This does not improve cell shape. 原生PLC默认保留材料界面；取消普通内部面输出不改变体网格，也不改善形状。这里是一个横贯截面块的单次对照，不能冒充完整电池或统计加速结论。</p><details><summary>Evidence / 证据</summary><pre>'+esc(json.dumps(b,ensure_ascii=False,indent=2))+'</pre></details></section>'
        doc=doc.replace('<section><h2>Rejected or incomplete candidates',benchmark+'<section><h2>Rejected or incomplete candidates')
    doc=doc.replace('__ROWS__',''.join(rows)).replace('__STAGES__',''.join(stage_rows)).replace('__CANDIDATES__',esc(json.dumps(candidates,indent=2)))
    font=work/'data/report_font/NotoSansCJKsc-Regular.otf'
    if font.exists():
        from morqex.visualization.report_font import embed_font
        doc=embed_font(doc,font,(font.parent/'LICENSE').read_text())
    (out/'acceptance.html').write_text(doc)
    print(json.dumps({'report':str(out/'acceptance.html'),'shape_targets_unmet':unmet,'regression':tests}),flush=True)


if __name__=='__main__':main()
