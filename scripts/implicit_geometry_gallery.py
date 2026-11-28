"""Full-array geometry conditioning and FEM qualification, one bounded stage.

Original external CT/synthetic geometry is retained; preparation is separate
from explicit resampling. Run through run_local.py. No automatic data download.
"""
import argparse,hashlib,html,json,os,time
from pathlib import Path
CASES=('nmc','granular','rods','platelets','agglomerates','bicontinuous','cell_stack')


def json_read(path):return json.loads(path.read_text()) if path.is_file() else None


def source_folder(work,case):return work/'nmc_complex_gallery'/('nmc_tiff' if case=='nmc' else 'native/'+case)


def gallery_index(work):
    root=work/'implicit_geometry';root.mkdir(exist_ok=True);items=[];rows=[]
    for case in CASES:
        base=root/case;state=json_read(base/'preparation.json');entries=[]
        for precision in ('coarse','balanced','fine','very_fine'):
            target=base/precision;r=json_read(target/'validation.json')
            if r:
                entries.append({'precision':precision,'directory':str(target.relative_to(root)),
                    'report':r,'preview_ready':(target/'mesh.html').is_file()})
        items.append({'id':case,'preparation':state,'entries':entries})
        for e in entries:
            r=e['report'];q=r['quality'];rows.append('<tr><td>'+html.escape(case)+'</td><td>'+e['precision']+'</td><td>'+f"{r['nodes']:,} / {r['tetrahedra']:,}"+'</td><td>'+f"{q['radius_edge_max']:.5f} / {q['minimum_dihedral_degrees']:.4f}°"+'</td><td>'+f"{max(r['processed_voxel_phase_volume_errors'].values()):.3%}"+'</td></tr>')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><title>MORQEX · fixed-grid conditioning</title>
<style>body{font:15px system-ui;background:#edf3f6;color:#173b4b;margin:20px}main{max-width:1850px;margin:auto}section{background:white;border-radius:14px;padding:20px;margin:18px 0}.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}iframe{height:960px;width:100%;border:1px solid #dce7ec}nav{display:flex;gap:18px;flex-wrap:wrap}select{padding:8px;font-size:16px;margin:5px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dce7ec;text-align:left}pre{white-space:pre-wrap}@media(max-width:1100px){.views{grid-template-columns:1fr}}</style>
<main><h1>Full-grid smoothing and adaptive FEM / 原尺寸平滑与自适应有限元</h1><nav><a href="../nmc_complex_gallery/index.html">Existing reference gallery / 原图库</a><a href="../nmc_complex_gallery/voxel_aligned.html">Exact staircase FEM / 精确体素网格</a><a href="catalogue.json">Complete measured catalogue / 完整实测目录</a></nav>
<section><p>Original physical box and full arrays are retained. Smoothing changes small geometric details; explicit lower_resolution is a separate API. All shown FEM files contain actual tetrahedra. / 原物理范围与完整数组保留；平滑只规则化几何细节，降低分辨率独立；所示有限元是真实单元。</p>
<label>Structure / 结构<select id="case"></select></label><label>FEM precision / 网格精度<select id="precision"></select></label><label>Compare / 对照<select id="compare"><option value="smooth">Original → smoothed / 原始→平滑</option><option value="mesh">Smoothed → FEM / 平滑→有限元</option></select></label><p id="status"></p><nav id="files"></nav><div class="views" id="views"></div></section>
<section><table><tr><th>Case</th><th>Precision</th><th>Nodes / Tet4</th><th>Max R/edge / min angle</th><th>Max phase volume error</th></tr>__ROWS__</table><p>Shape/volume checks do not certify PDE accuracy or complete topology equivalence. Source and processed component/contact changes remain in each preparation report. / 形状体积验收不等于物理解认证或完整拓扑等价，准备报告保留连通性及接触变化。</p></section></main>
<script>const items=__ITEMS__,$=id=>document.getElementById(id);function show(change=false){const d=items.find(x=>x.id===$('case').value)||items[0];if(change){$('precision').replaceChildren(...d.entries.map(e=>new Option(e.precision,e.precision)));if(d.entries.some(e=>e.precision==='balanced'))$('precision').value='balanced';}const e=d.entries.find(x=>x.precision===$('precision').value);$('status').textContent=!d.preparation?'Not prepared / 尚未准备':JSON.stringify({shape:d.preparation.processed_shape_zyx,changed_fraction:d.preparation.fidelity.changed_fraction,quality_passed:e?.report.quality_limits_satisfied});const paths=$('compare').value==='smooth'&&d.preparation?[d.id+'/source_voxels.html',d.id+'/prepared_voxels.html']:e?.preview_ready?[d.id+'/prepared_voxels.html',e.directory+'/mesh.html#view=2']:[];$('views').replaceChildren(...paths.map(src=>{let f=document.createElement('iframe');f.src=src;return f;}));let links=[[d.id+'/preparation.json','Preparation / 准备报告'],[d.id+'/progress.html','Progress / 进度']];if(e)links.push([e.directory+'/mesh.h5','HDF5'],[e.directory+'/mesh.xdmf','XDMF'],[e.directory+'/validation.json','Validation / 网格报告']);$('files').replaceChildren(...links.map(([href,name])=>{let a=document.createElement('a');a.href=href;a.textContent=name;return a;}));document.body.dataset.case=d.id;document.body.dataset.rendered='true';} $('case').replaceChildren(...items.map(d=>new Option(d.id,d.id)));$('case').onchange=()=>show(true);$('precision').onchange=()=>show();$('compare').onchange=()=>show();show(true);</script></html>'''
    from morqex.visualization.report_font import embed_workspace_font
    (root/'index.html').write_text(embed_workspace_font(doc.replace('__ROWS__',''.join(rows)).replace('__ITEMS__',json.dumps(items,separators=(',',':')).replace('<','\\u003c')),work))
    (root/'catalogue.json').write_text(json.dumps({'scope':'complete source arrays; geometry qualification, not PDE/publication certification','cases':items},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'))
    parser.add_argument('--case',choices=CASES);parser.add_argument('--stage',required=True,choices=['prepare','mesh','preview','exchange','index'])
    parser.add_argument('--precision',default='balanced',choices=['coarse','balanced','fine','very_fine']);parser.add_argument('--smoothing-config',type=Path)
    parser.add_argument('--mesh-config',type=Path);parser.add_argument('--format',choices=['comsol','vtu','gmsh','medit','abaqus'])
    args=parser.parse_args()
    if args.workspace is None:parser.error('External workspace required')
    from morqex.modeling import VoxelModel,PreparedGeometry,smooth_voxels,SmoothingOptions,implicit_mesh,ImplicitMeshOptions,ProgressBar,HTMLProgress,MeshArtifact,export_bundle
    from morqex.modeling.voxel import external_path
    work=external_path(args.workspace)
    if args.stage=='index':gallery_index(work);return
    if args.case is None:parser.error('--case required')
    folder=work/'implicit_geometry'/args.case;folder.mkdir(parents=True,exist_ok=True)
    live=HTMLProgress(folder/'progress.html');console=ProgressBar(json_lines=True)
    def callback(event):live(event);console(event)
    try:
        if args.stage=='prepare':
            model=VoxelModel.load(source_folder(work,args.case)/'geometry.h5')
            if args.smoothing_config:
                import tomllib
                settings=SmoothingOptions(**tomllib.loads(args.smoothing_config.read_text())['preparation'])
            else:settings=SmoothingOptions()
            if (folder/'prepared.h5').exists():
                from dataclasses import asdict
                p=PreparedGeometry.load(folder/'prepared.h5')
                if p.report['options']!=json.loads(json.dumps(asdict(settings))):raise ValueError('Different preparation recipe; select a new external experiment directory')
            else:
                p=smooth_voxels(model,options=settings,progress=callback);p.save(folder/'prepared.h5',progress=callback)
            (folder/'preparation.json').write_text(json.dumps(p.report,indent=2))
            if not (folder/'source_voxels.html').exists():model.viewer(maximum_3d_size=64,slice_origin='lower').save_html(folder/'source_voxels.html')
            if not (folder/'prepared_voxels.html').exists():p.model.viewer(maximum_3d_size=64,slice_origin='lower').save_html(folder/'prepared_voxels.html')
        else:
            target=folder/args.precision;target.mkdir(exist_ok=True)
            if args.stage=='mesh':
                values={}
                if args.mesh_config:
                    import tomllib
                    values=dict(tomllib.loads(args.mesh_config.read_text())['meshing']['options'])
                values['precision']=args.precision;settings=ImplicitMeshOptions(**values)
                if (target/'mesh.h5').exists():raise FileExistsError('Validated mesh checkpoint already exists; use preview/exchange')
                p=PreparedGeometry.load(folder/'prepared.h5')
                result=implicit_mesh(p,options=settings,workspace=work,progress=callback)
                result.save_hdf5(target/'mesh.h5');(target/'validation.json').write_text(json.dumps(result.report,indent=2))
            elif args.stage=='preview':
                artifact=MeshArtifact.load(target/'mesh.h5')
                artifact.adaptive_viewer(maximum_embedded_faces=350000,node_cache_bytes=256*2**20).save_html(target/'mesh.html')
            elif args.stage=='exchange':
                if args.format is None:parser.error('--format required')
                model=PreparedGeometry.load(folder/'prepared.h5').model
                result=export_bundle(MeshArtifact.load(target/'mesh.h5'),target/'software'/args.format,formats=[args.format],
                    region_names=model.region_names,length_unit='um',selections=model.metadata.get('selection_groups'))
                if not result['passed']:raise ArithmeticError('Exchange readback failed')
        live.finish(message='Requested stage complete / 本阶段完成；验收结果见报告');gallery_index(work)
    except BaseException as error:
        live.finish(message='Rejected / 未通过: '+str(error)[:1500]);(folder/'last_failure.json').write_text(json.dumps({'stage':args.stage,'precision':args.precision,'error':str(error)},indent=2));raise

if __name__=='__main__':main()
