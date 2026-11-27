"""Generate ten complete resolved-feature electrodes and weighted ACVD meshes.

Controller runs one externally frozen, resource-limited child at a time. Run
this controller outside run_local; each numerical child uses run_local itself.
"""
import argparse,json,os,sys,time,subprocess,tarfile,hashlib,html
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
NAMES=['granular','rods','platelets','dendritic','hollow','agglomerates','fibres','bicontinuous','lamellar','channels']


def run_case(args):
    import numpy as np
    from morqex.modeling import resolved_electrode,adaptive_mesh,AdaptiveMeshOptions,TetGenOptions,export_comsol,read_comsol
    out=Path(args.output)/args.case/args.precision;out.mkdir(parents=True,exist_ok=True)
    os.environ['MORQEX_MESH_PROGRESS']='1';start=time.perf_counter()
    model=resolved_electrode(args.case,size=args.size,seed=args.seed)
    model.save(out/'geometry.h5');model.viewer(maximum_3d_size=48,title=args.case+' / complete source').save_html(out/'voxels.html')
    config={'preset':args.case,'size':args.size,'seed':args.seed,'precision':args.precision,'regions':[0,1,2],
        'source_snapshot':json.loads(os.environ.get('MORQEX_SOURCE_SNAPSHOT','{}'))}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    result=adaptive_mesh(model,regions=[0,1,2],options=AdaptiveMeshOptions.precision(args.precision,maximum_facets=4000000),
                         tetgen_options=TetGenOptions.for_adaptive(maximum_cells=3000000))
    artifact=result.save_hdf5(out/'mesh.h5')
    comsol=export_comsol(result,out/'mesh.mphtxt',region_names=model.region_names)
    back=read_comsol(comsol)
    np.testing.assert_array_equal(back['points'],result.points)
    np.testing.assert_array_equal(back['elements']['tet']['cells'],result.cells)
    mapping=json.loads(comsol.with_suffix('.regions.json').read_text())['region_to_domain']
    np.testing.assert_array_equal(back['elements']['tet']['entities'],[mapping[str(int(k))] for k in result.phases])
    del back
    result.adaptive_viewer(title=args.case+' / weighted ACVD + TetGen').save_html(out/'mesh.html')
    ids=np.unique(result.facets);surface=result.points[ids]
    grid=(surface-np.asarray(model.origin)[::-1])/np.asarray(model.spacing)[::-1]
    nonlattice=float(np.mean(np.any(abs(grid*2-np.rint(grid*2))>1e-5,axis=1)))
    assert nonlattice>.2,'Adaptive boundary must contain real relocated vertices'
    assert set(map(int,np.unique(result.phases)))=={0,1,2}
    report={'passed_geometry_export':True,'case':args.case,'precision':args.precision,
        'shape_zyx':model.regions.shape,'source_voxels':int(model.regions.size),
        'fractions':model.volume().fractions,'selected_regions':[0,1,2],
        'non_half_voxel_lattice_boundary_fraction':nonlattice,'report':result.report,
        'comsol_readback':True,'wall_seconds':time.perf_counter()-start,
        'comsol_application_tested':False,
        'geometry_scope':'complete synthetic resolved-feature electrode including active, CBD and pore; not CT or a battery simulation',
        'shape_quality_passed':result.report['quality_limits_satisfied']}
    (out/'validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'complete':args.case,'precision':args.precision,'tetrahedra':len(result.cells),'wall_seconds':report['wall_seconds']}),flush=True)


def write_index(out,status,size):
    cards=[]
    for name in NAMES:
        cases=[]
        for report in sorted((out/name).glob('*/validation.json')):
            r=json.loads(report.read_text());mode=report.parent.name;link=f'{name}/{mode}';g=r['report']['geometry']
            cases.append(f'''<section><h3>{mode} · {r['report']['tetrahedra']:,} Tet4</h3>
<p>{r['source_voxels']:,} source voxels · full {size}³ · phases0/1/2 ·
ACVD surface {g['input_surface_vertices']:,} → {g['output_surface_vertices']:,} vertices.
Max phase-volume difference: {max(g['phase_voxel_volume_relative_errors'].values()):.2%}.
Requested element shape quality passed: {r['shape_quality_passed']}.</p>
<p><a href="{link}/mesh.mphtxt">COMSOL mesh</a> · <a href="{link}/mesh.regions.json">Domains / 域映射</a> ·
<a href="{link}/mesh.xdmf">XDMF</a> · <a href="{link}/mesh.h5">HDF5</a> · <a href="{link}/validation.json">Validation</a></p>
<div class="views"><iframe loading="lazy" title="{name} source" src="{link}/voxels.html"></iframe>
<iframe loading="lazy" title="{name} mesh" src="{link}/mesh.html"></iframe></div></section>''')
        cards.append(f'<article id="{name}"><h2>{name}</h2>'+(''.join(cases) or '<p>Pending / 等待计算</p>')+'</article>')
    refresh='<meta http-equiv="refresh" content="30">' if status!='complete' else ''
    body=f'''<!doctype html><meta charset="utf-8">{refresh}<title>MORQEX · GEMsim adaptive meshing</title>
<style>body{{font:15px system-ui;background:#eef3f6;color:#18374a;margin:24px}}article{{background:white;border-radius:14px;padding:18px;margin:20px 0}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}iframe{{width:100%;height:840px;border:1px solid #d6e1e7;border-radius:8px}}a{{color:#136e7c}}@media(max-width:1000px){{.views{{grid-template-columns:1fr}}}}</style>
<main><h1>GEMsim residual-weighted ACVD → TetGen</h1><p>状态 / Status: {html.escape(status)}. Ten different full {size}³ structures;
no cropped SVE, no voxel replication. Active material + CBD + pore domains are included in every completed mesh.</p>
<p>左：原始体素与精确切片。右：真正重构的非规则曲面网格，以及真实体单元的精确平面截面，可查看内部疏密。
每个体素场完整保存；体网格没有“每体素六个单元”的固定对应。质量请求是否达标独立列出。</p>
<p>These are new synthetic resolved-feature recipes using the integrated generator; the earlier dense recipes and meshes are retained.
Geometric adaptation is not certified PDE-error adaptation. COMSOL format/read-back is tested; the COMSOL application is unavailable.</p>
<nav>'''+ ' · '.join(f'<a href="#{n}">{n}</a>' for n in NAMES)+'</nav>'+''.join(cards)+'</main></html>'
    workspace=Path(os.environ.get('MORQEX_WORKSPACE',str(out.parent)))
    font=workspace/'data/report_font/NotoSansCJKsc-Regular.otf'
    if font.exists():
        try:
            from morqex.visualization.report_font import embed_font
            body=embed_font(body,font,(font.parent/'LICENSE').read_text())
        except ImportError:pass
    (out/'index.html').write_text(body)


def controller(args):
    workspace=Path(args.workspace).resolve();out=Path(args.output or workspace/'adaptive_gallery256_gemsim').resolve()
    out.mkdir(parents=True,exist_ok=True)
    # Freeze source before numerical work so edits to tutorials cannot affect a run.
    sys.path.insert(0,str(ROOT/'scripts'));from source_snapshot import snapshot
    snap=snapshot(ROOT,workspace);frozen=workspace/'frozen_geometry'/snap['tree_sha256'][:20]
    if not frozen.exists():
        frozen.mkdir(parents=True)
        with tarfile.open(snap['archive']) as archive:archive.extractall(frozen,filter='data')
    python=args.python or str(workspace/'runtime/bin/python');completed=[]
    (out/'source.json').write_text(json.dumps(snap,indent=2));write_index(out,'running',args.size)
    for name in args.cases or NAMES:
        for mode in args.precisions:
            target=out/name/mode/'validation.json'
            if target.exists():
                report=json.loads(target.read_text());assert report['passed_geometry_export'] and report['shape_zyx']==[args.size]*3
                completed.append((name,mode));continue
            if target.parent.exists():
                rejected=out/'rejected_cases'/f'{name}_{mode}_{time.time_ns()}'
                rejected.parent.mkdir(exist_ok=True);target.parent.rename(rejected)
            cmd=[sys.executable,str(frozen/'scripts/run_local.py'),'--workspace',str(workspace),'--mesh-output',
                 '--label',f'adaptive256_{name}_{mode}','--',python,str(frozen/'scripts/adaptive_geometry_gallery.py'),
                 '--case',name,'--precision',mode,'--size',str(args.size),'--seed',str(args.seed),'--output',str(out)]
            if args.reference_memory:cmd.insert(5,'--reference-memory')
            result=subprocess.run(cmd)
            if result.returncode or not target.exists():
                write_index(out,f'needs correction: {name}/{mode}',args.size)
                raise RuntimeError(f'Case {name}/{mode} did not pass. Original results retained; inspect resource manifest and correct before --resume.')
            completed.append((name,mode));write_index(out,'running',args.size)
    reports=[json.loads(p.read_text()) for p in out.glob('*/*/validation.json')]
    complete=set(NAMES)<=set(r['case'] for r in reports)
    write_index(out,'complete' if complete else 'requested cases complete; ten-family gallery pending',args.size)
    (out/'validation.json').write_text(json.dumps({'all_geometry_export_checks_passed':all(r['passed_geometry_export'] for r in reports),
        'ten_morphologies_complete':complete,'cases':reports,'size':args.size,'scope':'synthetic geometry/meshing, not battery or PDE validation'},indent=2))
    print(json.dumps({'html':str(out/'index.html'),'completed':completed}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',default=str(ROOT.parent/'morph_workspace/microelectrode_gallery'))
    p.add_argument('--output');p.add_argument('--python');p.add_argument('--size',type=int,default=256);p.add_argument('--seed',type=int,default=20261127)
    p.add_argument('--cases',nargs='+',choices=NAMES);p.add_argument('--precisions',nargs='+',default=['balanced'],choices=['coarse','balanced','fine'])
    p.add_argument('--case',choices=NAMES);p.add_argument('--precision',default='balanced',choices=['coarse','balanced','fine'])
    p.add_argument('--reference-memory',action='store_true',help='6GiB process-tree RSS cap, one CPU,8GiB host reserve')
    args=p.parse_args();run_case(args) if args.case else controller(args)
