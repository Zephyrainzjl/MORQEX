"""Full-resolution CT/reference/dense-native geometry qualification controller.

Run this controller directly. It freezes source and serializes bounded jobs.
Generated files and native libraries remain in the external workspace.
"""
import argparse,json,os,sys,time,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
NAMES=['granular','rods','platelets','agglomerates','bicontinuous','cell_stack']


def write_index(out):
    out=Path(out).resolve()
    def exchange_links(folder):
        links=[]
        for fmt in ('vtu','gmsh','medit','abaqus','comsol'):
            manifest=folder/'software'/fmt/'bundle.json'
            if manifest.exists():
                bundle=json.loads(manifest.read_text())
                if bundle.get('passed'):
                    for check in bundle['checks']:links.append([fmt+' mesh ('+bundle['length_unit']+')',str(Path(check['path']).relative_to(out))])
        return links
    items=[]
    for name in NAMES:
        folder=out/'native'/name;vox=folder/'geometry.h5';mesh_folder='partitioned' if (folder/'partitioned/validation.json').exists() else 'mesh'
        report=folder/mesh_folder/'validation.json'
        ready=report.exists();text='Voxel generation pending / 待生成体素'
        if vox.exists():text='Complete source saved; volume mesh pending / 完整体素已保存，体网格待验收'
        if ready:
            r=json.loads(report.read_text());text=f"{r['nodes']:,} nodes · {r['tetrahedra']:,} Tet4 · {r.get('domain_count',len(r.get('phase_volumes_m3',{})))} separate domains"
        comsol='mesh.mphtxt' if mesh_folder=='partitioned' else 'software/mesh.mphtxt'
        links=[['Source voxels / 完整体素',f'native/{name}/geometry.h5']] if vox.exists() else []
        if ready:
            links += [['COMSOL',f'native/{name}/{mesh_folder}/{comsol}'],['HDF5',f'native/{name}/{mesh_folder}/mesh.h5'],
                ['XDMF',f'native/{name}/{mesh_folder}/mesh.xdmf'],['Checks / 检查',f'native/{name}/{mesh_folder}/validation.json']]
            links += exchange_links(folder/mesh_folder)
        items.append({'id':name,'label':name,'text':text,'voxel':f'native/{name}/voxels.html' if vox.exists() else None,
            'mesh':f'native/{name}/{mesh_folder}/mesh.html' if ready else None,'links':links,
            'shape_quality':r.get('quality_limits_satisfied') if ready else None})
    if (out/'reference_comsol/mesh.html').exists():
        items.insert(0,{'id':'reference','label':'Independent COMSOL / 独立参考','text':'158,385 nodes · 934,288 Tet4 · user-confirmed ≈1 m extent; independent of TIFF',
            'voxel':None,'mesh':'reference_comsol/mesh.html','shape_quality':None,
            'links':[['Original selections / 命名选择','reference_comsol/reference_with_selections.mphtxt'],['HDF5','reference_comsol/mesh.h5'],['XDMF','reference_comsol/mesh.xdmf']]+exchange_links(out/'reference_comsol')})
    if (out/'nmc_tiff/exact_voxel_partitioned/mesh.html').exists():
        r=json.loads((out/'nmc_tiff/exact_voxel_partitioned/validation.json').read_text());base='nmc_tiff/exact_voxel_partitioned/'
        items.insert(1,{'id':'nmc','label':'Original NMC TIFF / 原始NMC','text':f"252×252×253 original voxels · {r['nodes']:,} nodes · {r['tetrahedra']:,} Tet4; exact voxel boundary route",
            'voxel':'nmc_tiff/voxels.html','mesh':base+'mesh.html','shape_quality':r.get('quality_limits_satisfied'),
            'links':[['Source voxels','nmc_tiff/geometry.h5'],['COMSOL',base+'mesh.mphtxt'],['HDF5',base+'mesh.h5'],['XDMF',base+'mesh.xdmf'],['Checks',base+'validation.json']]+exchange_links(out/base)})
    for item in items:
        if item.get('mesh'):
            detail=Path(item['mesh']).with_name('roi_mesh.html')
            if (out/detail).exists():item['detail']=str(detail)
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>MORQEX · complex geometry gallery</title>
<style>body{font:15px system-ui;background:#edf3f6;color:#1c3b4d;margin:20px}main{max-width:1900px;margin:auto}article{background:white;border-radius:14px;padding:16px;margin:18px 0}.views{display:grid;grid-template-columns:1fr 1fr;gap:10px}iframe{width:100%;height:940px;border:1px solid #d9e3e8;border-radius:8px}a{color:#126d80}nav{display:flex;gap:12px;flex-wrap:wrap}select{padding:9px;font-size:16px}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #e2e9ec;padding:9px}.notice{color:#8b471e}button{padding:5px 12px;cursor:pointer}@media(max-width:1100px){.views{grid-template-columns:1fr}}</style>
<main><h1>MORQEX · complex electrode geometry / 复杂电极几何</h1>
<p>完整原始体素 → 共享多相界面 → 非结构有限元网格 → 软件交换。只加载当前结构的两个视图；切换时释放旧视图，减少浏览器内存。</p>
<p>The TIFF and COMSOL references are independent. COMSOL extent is ≈1 m as confirmed by its owner; TIFF spacing is an explicitly chosen 0.5 µm example.
Five native 256³ structures are generated independently, without tiling or deforming the CT. Source and mesh files stay complete; display LOD is labelled.</p>
<nav><a href="acceptance.html">Measured qualification report / 实测验收报告</a><a href="voxel_aligned.html">Exact voxel-aligned route / 精确体素边界分区</a><a href="exchange_validation.json">40 full-model software round trips / 40组全量格式回读</a><a href="browser_validation.json">Offline 3D/section checks / 离线显示检查</a></nav>
<article><nav><label>Structure / 结构 <select id="case"></select></label><a id="surface" target="_blank">Open whole 3D interface sample / 打开全域三维界面抽样</a><a id="detail" target="_blank">Complete actual 3D ROI faces / 完整真实网格三维局部</a><a id="section" target="_blank">Open exact Tet4 section / 打开真实单元切面</a></nav>
<h2 id="label"></h2><p id="description"></p><p id="quality" class="notice"></p><nav id="files"></nav>
<p>Voxel and mesh controls are independent; compare their physical plane coordinates. / 体素与网格控件独立，以各自显示的物理切面坐标为准。</p>
<div class="views" id="views"></div></article><article><h2>Complete data catalogue / 完整数据目录</h2><table><thead><tr><th>Structure / 结构</th><th>Full model / 全量模型</th><th>Inspect / 查看</th></tr></thead><tbody id="catalogue"></tbody></table>
<p>Conformity, phase volume, shape quality and software readback are separate checks. A positive-volume mesh is not a certified converged FE solution. Numerical reports retain failed shape requests.</p>
<p>几何共形、相体积、单元形状与软件回读分别验收；形状未达请求上限的模型不会标成完全合格。三维大模型显示的是明确抽样的实际界面面片；切面由完整Tet4网格求交得到。</p></article></main>
<script>const items=__ITEMS__,$=id=>document.getElementById(id);
function choose(id){const d=items.find(x=>x.id===id)||items[0];$('case').value=d.id;$('label').textContent=d.label;$('description').textContent=d.text;
$('quality').textContent=!d.mesh?'Mesh qualification pending / 网格待验收':d.shape_quality===false?'Geometry conformity checked; requested shape bound NOT met / 几何共形已检查；形状请求上限未通过':d.shape_quality===true?'Measured shape request met / 实测形状请求通过':'No shape limit requested for the imported reference / 导入参考未指定形状上限';
$('surface').hidden=$('section').hidden=!d.mesh;$('surface').href=d.mesh?d.mesh+'#view=surface':'#';$('section').href=d.mesh?d.mesh+'#view=2':'#';$('files').replaceChildren(...d.links.map(([name,href])=>{let a=document.createElement('a');a.textContent=name;a.href=href;if(!href.endsWith('.json'))a.download='';return a;}));
$('detail').hidden=!d.detail;$('detail').href=d.detail||'#';
$('views').replaceChildren(...[d.voxel||(d.mesh?d.mesh+'#view=surface':null),d.mesh?d.mesh+'#view=2':null].filter(Boolean).map((src,i)=>{let f=document.createElement('iframe');f.src=src;f.title=d.label+(i?' actual Tet4 section':' complete source');return f;}));location.hash='case='+d.id;document.body.dataset.rendered='true';document.body.dataset.case=d.id;}
$('case').replaceChildren(...items.map(d=>new Option(d.label,d.id)));$('case').onchange=()=>choose($('case').value);
for(const d of items){let row=document.createElement('tr'),a=document.createElement('td'),b=document.createElement('td'),c=document.createElement('td'),button=document.createElement('button');a.textContent=d.label;b.textContent=d.text;button.textContent='View / 查看';button.onclick=()=>{choose(d.id);scrollTo({top:0,behavior:'smooth'});};c.append(button);row.append(a,b,c);$('catalogue').append(row);}
choose(new URLSearchParams(location.hash.slice(1)).get('case')||'granular');</script></html>'''
    html=html.replace('__ITEMS__',json.dumps(items,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    font=out.parent/'data/report_font/NotoSansCJKsc-Regular.otf'
    if font.exists():
        try:
            from morqex.visualization.report_font import embed_font
            html=embed_font(html,font,(font.parent/'LICENSE').read_text())
        except ImportError:pass
    (out/'index.html').write_text(html)


def statistics(model):
    import numpy as np
    from scipy import ndimage
    result={}
    for k in np.unique(model.regions):
        mask=model.regions==k;components,n=ndimage.label(mask);count=np.bincount(components.ravel())[1:]
        area=sum(np.count_nonzero(np.diff(mask.astype(np.int8),axis=axis)) for axis in range(3))
        result[int(k)]={'name':model.region_names[int(k)],'voxels':int(mask.sum()),'fraction':float(mask.mean()),
            'components_6':int(n),'largest_component_voxels':int(count.max()),'internal_interface_voxel_faces':int(area)}
    return result


def worker(args):
    import numpy as np
    from morqex.modeling import (coated_electrode,porous_layer,stack_layers,from_regions,VoxelModel,
        AdaptiveMeshOptions,TetGenOptions,adaptive_mesh,export_bundle)
    folder=Path(args.output)/'native'/args.case;folder.mkdir(parents=True,exist_ok=True)
    if args.stage=='generate':
        if args.case=='cell_stack':
            cathode=VoxelModel.load(Path(args.output)/'native/granular/geometry.h5')
            n=cathode.regions.shape[0];spacing=cathode.spacing
            anode=from_regions(np.ones((n,n,48),np.uint8),spacing=spacing,region_names={1:'anode_lithium_metal'})
            electrolyte=from_regions(np.ones((n,n,4),np.uint8),spacing=spacing,region_names={1:'negative_electrolyte'})
            separator=from_regions(np.ones((n,n,24),np.uint8),spacing=spacing,region_names={1:'separator_effective_porous'})
            neg=from_regions(np.ones((n,n,4),np.uint8),spacing=spacing,region_names={1:'negative_collector'})
            pos=from_regions(np.ones((n,n,4),np.uint8),spacing=spacing,region_names={1:'positive_collector'})
            model=stack_layers([('negative_collector',neg),('anode',anode),('negative_electrolyte',electrolyte),('separator',separator),('cathode',cathode),('positive_collector',pos)],axis=2)
        else:model=coated_electrode(args.case,size=256,seed=20261127,active_fraction=.5)
        model.save(folder/'geometry.h5');model.viewer(maximum_3d_size=64,slice_origin='lower',title=args.case+' / complete native voxels').save_html(folder/'voxels.html')
        (folder/'generation.json').write_text(json.dumps({'shape':model.regions.shape,'metadata':model.metadata,
            'statistics':statistics(model),'source_snapshot':json.loads(os.environ.get('MORQEX_SOURCE_SNAPSHOT','{}'))},indent=2))
        print(json.dumps({'generated':args.case,'shape':model.regions.shape,'metadata':model.metadata}),flush=True);return
    model=VoxelModel.load(folder/'geometry.h5');out=folder/'mesh.pending';out.mkdir(exist_ok=True)
    start=time.perf_counter();os.environ['MORQEX_MESH_PROGRESS']='1'
    options=AdaptiveMeshOptions(surface_method='surface_nets' if args.case=='cell_stack' else 'nested_contours',vertex_fraction=.12,levels=3,iterations=12,
        conserve_phase_volumes=args.case=='cell_stack',surface_partition_voxels=128,maximum_facets=16000000,topology_retries=2,
        maximum_phase_volume_error=.03)
    native=TetGenOptions.for_adaptive(maximum_cells=5000000,maximum_steiner=800000,native_timeout_seconds=240,
        maximum_volume=float(model.regions.size*np.prod(model.spacing)/1000000))
    mesh=adaptive_mesh(model,options=options,tetgen_options=native)
    artifact=mesh.save_hdf5(out/'mesh.h5');mesh.adaptive_viewer(title=args.case+' / actual multiphase TetGen mesh').save_html(out/'mesh.html')
    bundle=export_bundle(mesh,out/'software',region_names=model.region_names,
        selections=model.metadata.get('selection_groups'),length_unit='um')
    report={'case':args.case,'nodes':len(mesh.points),'tetrahedra':len(mesh.cells),'domain_count':len(np.unique(mesh.phases)),
        'shape':model.regions.shape,'geometry_export_passed':True,'source_model_unchanged':True,
        'shape_quality_passed':mesh.report['quality_limits_satisfied'],'report':mesh.report,'bundle':bundle,'seconds':time.perf_counter()-start,
        'source_snapshot':json.loads(os.environ.get('MORQEX_SOURCE_SNAPSHOT','{}'))}
    (out/'validation.json').write_text(json.dumps(report,indent=2));out.rename(folder/'mesh')
    print(json.dumps({'meshed':args.case,'tetrahedra':len(mesh.cells),'seconds':report['seconds']}),flush=True)


def controller(args):
    work=Path(args.workspace).resolve();out=Path(args.output or work/'nmc_complex_gallery').resolve();out.mkdir(exist_ok=True)
    sys.path.insert(0,str(ROOT/'scripts'));from source_snapshot import snapshot
    snap=snapshot(ROOT,work);frozen=work/'frozen_geometry'/snap['tree_sha256'][:20]
    if not frozen.exists():
        frozen.mkdir(parents=True)
        with tarfile.open(snap['archive']) as archive:archive.extractall(frozen,filter='data')
    write_index(out)
    for name in args.cases or NAMES:
        target=out/'native'/name/('generation.json' if args.stage=='generate' else 'mesh/validation.json')
        if target.exists():continue
        pending=out/'native'/name/'mesh.pending'
        if pending.exists():pending.rename(pending.with_name('rejected_'+str(time.time_ns())))
        cmd=[sys.executable,str(frozen/'scripts/run_local.py'),'--workspace',str(work),'--reference-memory','--mesh-output',
            '--label',f'complex_{name}_{args.stage}','--',str(work/'runtime/bin/python'),str(frozen/'scripts/complex_geometry_gallery.py'),
            '--case',name,'--stage',args.stage,'--output',str(out)]
        result=subprocess.run(cmd);write_index(out)
        if result.returncode:raise RuntimeError(f'{name}/{args.stage} requires correction; original source/results retained')
    print(json.dumps({'html':str(out/'index.html'),'stage':args.stage}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',default=str(ROOT.parent/'morph_workspace/microelectrode_gallery'))
    p.add_argument('--output');p.add_argument('--stage',choices=['generate','mesh'],default='generate');p.add_argument('--case',choices=NAMES)
    p.add_argument('--cases',nargs='+',choices=NAMES);args=p.parse_args()
    worker(args) if args.case else controller(args)
