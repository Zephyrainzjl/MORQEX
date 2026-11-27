"""Qualify the exact voxel-aligned route on full external gallery structures.

Run one numerical stage through run_local.py; all source arrays and results
remain external. This does not replace the independent smooth/GEM workflows.
"""
import argparse,hashlib,html,json,os
from pathlib import Path

CASES=('nmc','granular','rods','platelets','agglomerates','bicontinuous','cell_stack')


def location(root,case):
    return root/('nmc_tiff' if case=='nmc' else 'native/'+case)


def read(path):return json.loads(path.read_text()) if path.exists() else None


def index(root):
    rows=[];items=[]
    for case in CASES:
        base=location(root,case);folder=base/'voxel_aligned';r=read(folder/'validation.json')
        state='PENDING / 未完成' if r is None else f"{r['nodes']:,} nodes · {r['tetrahedra']:,} Tet4"
        if r and not (folder/'mesh.html').exists():state+=' · HTML preview pending / HTML待生成'
        files=[]
        if r:
            for name in ('mesh.h5','mesh.xdmf','validation.json'):
                files.append([name,str((folder/name).relative_to(root))])
            if (folder/'roi_mesh.html').exists():
                files.append(['Complete actual 3D ROI faces / 真实三维局部网格（非新SVE）',str((folder/'roi_mesh.html').relative_to(root))])
            for fmt in ('comsol','vtu','gmsh','medit','abaqus'):
                bundle=read(folder/'software'/fmt/'bundle.json')
                if bundle and bundle.get('passed'):
                    files.extend([[fmt+' ('+bundle['length_unit']+')',str(Path(c['path']).relative_to(root))] for c in bundle['checks']])
        item={'id':case,'state':state,'files':files,'ready':bool(r and (folder/'mesh.html').exists()),
              'voxel':str((base/'voxels.html').relative_to(root)),
              'mesh':str((folder/'mesh.html').relative_to(root)),
              'quality':r.get('quality') if r else None,'shape_passed':r.get('quality_limits_satisfied') if r else None}
        items.append(item)
        rows.append('<tr><td>'+html.escape(case)+'</td><td>'+html.escape(state)+'</td><td>'+('PASS' if item['shape_passed'] is True else 'UNMET / 未通过' if r else 'PENDING')+'</td></tr>')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><title>MORQEX · exact voxel-aligned TetGen</title>
<style>body{font:15px system-ui;background:#edf3f6;color:#173b4b;margin:20px}main{max-width:1800px;margin:auto}section{background:white;padding:18px;border-radius:14px;margin:16px 0}select{padding:8px;font-size:16px}.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}iframe{height:960px;width:100%;border:1px solid #dde6eb;border-radius:9px}nav{display:flex;gap:15px;flex-wrap:wrap}a{color:#096b83}.notice{color:#a03d29}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce7ec}pre{white-space:pre-wrap}@media(max-width:1100px){.views{grid-template-columns:1fr}}</style>
<main><h1>Exact voxel-aligned TetGen / 精确体素边界分区</h1>
<nav><a href="index.html">Original smooth/reference gallery / 原有光滑及参考图库</a><a href="acceptance.html">Measured qualification / 实测验收</a><a href="voxel_aligned_catalogue.json">Full catalogue / 完整目录</a></nav>
<section><p>Original labels → exact voxel quads → canonical lattice cap diagonals → source-integrated TetGen → one conforming volume mesh. Different material labels across a cut remain physical interfaces. Source phase volumes and every cell-centroid material are checked. / 原始相标签、相体积及全部单元中心标签均检查，分区交界处真实材料界面保留。</p>
<p>This exact stair-step route differs from smoothed Surface Nets and weighted ACVD. It does not certify PDE convergence or guarantee a requested shape bound. Original TIFF gray0/128/255 roles remain unassigned; spacing0.5µm is an explicit modelling choice. / 精确阶梯、光滑共享曲面与加权ACVD分别保留；不混淆几何检查与物理收敛。</p>
<label>Structure / 结构 <select id="case"></select></label><h2 id="title"></h2><p id="state"></p><p class="notice" id="quality"></p><nav id="files"></nav><p>Physical section planes and phase colors agree by default. Display sampling is explicitly labelled; all computational arrays stay complete. / 默认切面物理坐标及颜色一致；显示抽样不改变完整计算数据。</p><div id="views" class="views"></div></section>
<section><table><thead><tr><th>Structure / 结构</th><th>Complete mesh / 完整网格</th><th>Shape request / 形状请求</th></tr></thead><tbody>__ROWS__</tbody></table></section></main>
<script>const items=__ITEMS__,$=id=>document.getElementById(id);function choose(id){let d=items.find(x=>x.id===id)||items[0];$('case').value=d.id;$('title').textContent=d.id;$('state').textContent=d.state;$('quality').textContent=!d.quality?'Pending / 待验收':(d.shape_passed?'Shape request met / 形状请求通过':'Shape request UNMET / 形状请求未通过')+' · max R/edge '+d.quality.radius_edge_max+' · min dihedral '+d.quality.minimum_dihedral_degrees+'°';$('files').replaceChildren(...d.files.map(([name,href])=>{let a=document.createElement('a');a.textContent=name;a.href=href;return a;}));$('views').replaceChildren(...(d.ready?[d.voxel,d.mesh+'#view=2']:[]).map(src=>{let f=document.createElement('iframe');f.src=src;return f;}));document.body.dataset.case=d.id;document.body.dataset.rendered='true';location.hash='case='+d.id;}$('case').replaceChildren(...items.map(d=>new Option(d.id,d.id)));$('case').onchange=()=>choose($('case').value);choose(new URLSearchParams(location.hash.slice(1)).get('case')||'granular');</script></html>'''
    doc=doc.replace('__ROWS__',''.join(rows)).replace('__ITEMS__',json.dumps(items,separators=(',',':')).replace('<','\\u003c'))
    font=root.parent/'data/report_font/NotoSansCJKsc-Regular.otf'
    if font.exists():
        from morqex.visualization.report_font import embed_font
        doc=embed_font(doc,font,(font.parent/'LICENSE').read_text())
    (root/'voxel_aligned.html').write_text(doc)
    (root/'voxel_aligned_catalogue.json').write_text(json.dumps({'scope':'geometry qualification; shape requests separate','cases':items},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'))
    p.add_argument('--stage',choices=('mesh','preview','detail','exchange','vtk','index'),required=True)
    p.add_argument('--case',choices=CASES);p.add_argument('--slab-depth',type=int,default=32)
    p.add_argument('--format',choices=('comsol','vtu','gmsh','medit','abaqus'))
    p.add_argument('--length-unit',choices=('m','cm','mm','um','nm'),default='um')
    p.add_argument('--roi-voxels',type=int,default=48)
    p.add_argument('--maximum-roi-faces',type=int,default=500000)
    p.add_argument('--center-fraction',nargs=3,type=float,default=(.5,.5,.5),help='Display ROI centre fractions in XYZ')
    a=p.parse_args()
    if a.workspace is None:p.error('Explicit external workspace required')
    from morqex.modeling import VoxelModel,MeshArtifact,partitioned_tetgen,export_bundle
    from morqex.modeling.voxel import external_path
    root=external_path(a.workspace)/'nmc_complex_gallery'
    if a.stage=='index':index(root);return
    if a.case is None:p.error('--case required for numerical stages')
    base=location(root,a.case);folder=base/'voxel_aligned';model=VoxelModel.load(base/'geometry.h5')
    if a.stage=='mesh':
        artifact=partitioned_tetgen(model,folder,surface_method='voxel_aligned',slab_depth=a.slab_depth)
        print(json.dumps({'case':a.case,'nodes':artifact.nodes,'tetrahedra':artifact.tetrahedra,'mesh':str(artifact.hdf5)}),flush=True)
        return
    artifact=MeshArtifact.load(folder/'mesh.h5')
    if a.stage=='detail':
        import numpy as np
        centre=np.asarray(a.center_fraction)
        if a.roi_voxels<2 or not np.isfinite(centre).all() or np.any(centre<=0) or np.any(centre>=1):
            raise ValueError('Positive ROI voxel width and XYZ centre fractions strictly within0..1 required')
        shape=np.array(model.regions.shape[::-1]);spacing=np.array(model.spacing[::-1]);origin=np.array(model.origin[::-1])
        middle=origin+shape*spacing*centre;half=spacing*a.roi_voxels/2
        bounds=np.array([np.maximum(origin,middle-half),np.minimum(origin+shape*spacing,middle+half)])
        stamps=[(p.stat().st_size,p.stat().st_mtime_ns) for p in (base/'geometry.h5',artifact.hdf5)]
        view=artifact.adaptive_viewer(sections=False,surface_bounds_m=bounds,maximum_embedded_faces=a.maximum_roi_faces,
            title=a.case+' / actual boundary detail; full source computation retained / 真实网格局部，完整计算不裁剪')
        if view.selected_surface_faces!=view.embedded_surface_faces:
            raise ValueError('Complete display ROI exceeds face budget; reduce roi-voxels or explicitly increase maximum-roi-faces')
        view.save_html(folder/'roi_mesh.html')
        if stamps!=[(p.stat().st_size,p.stat().st_mtime_ns) for p in (base/'geometry.h5',artifact.hdf5)]:
            raise ArithmeticError('ROI rendering changed a computational source file')
        (folder/'roi_display.json').write_text(json.dumps({'surface_bounds_xyz_m':bounds.tolist(),
            'source_mesh':str(artifact.hdf5),'source_tetrahedra':artifact.tetrahedra,
            'complete_roi_actual_faces':view.selected_surface_faces,'embedded_actual_faces':view.embedded_surface_faces,
            'source_files_unmodified':True,'scope':'display selection of complete actual triangles only; not a new SVE or numerical solve'},indent=2))
    elif a.stage=='preview':
        import numpy as np
        shape=np.array(model.regions.shape[::-1]);fractions=(shape//2+.5)/shape
        paths=(base/'geometry.h5',artifact.hdf5);stamps=[(f.stat().st_size,f.stat().st_mtime_ns) for f in paths]
        artifact.adaptive_viewer(maximum_embedded_faces=450000,node_cache_bytes=256*2**20,
            section_fractions=fractions,title=a.case+' / exact voxel-aligned TetGen').save_html(folder/'mesh.html')
        if stamps!=[(f.stat().st_size,f.stat().st_mtime_ns) for f in paths]:raise ArithmeticError('Visualization changed computational files')
        (folder/'comparison_alignment.json').write_text(json.dumps({'source':str(paths[0]),'mesh':str(paths[1]),
            'section_fractions_xyz':fractions.tolist(),'source_and_mesh_files_unmodified':True},indent=2))
    elif a.stage=='exchange':
        if a.format is None:p.error('Choose one --format so each full-model readback has its own bounded process')
        destination=folder/'software'/a.format
        with artifact.hdf5.open('rb') as stream:
            source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
        if (destination/'bundle.json').exists():
            old=read(destination/'bundle.json')
            if old.get('passed') and old['length_unit']==a.length_unit and old.get('source_mesh_sha256')==source_hash:
                print(json.dumps(old));return
            raise ValueError('Existing exchange differs; archive it explicitly')
        report=export_bundle(artifact,destination,region_names=model.region_names,
            selections=model.metadata.get('selection_groups'),length_unit=a.length_unit,formats=(a.format,))
        report['source_mesh_sha256']=source_hash
        (destination/'bundle.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    elif a.stage=='vtk':
        from morqex.modeling.vtu_stream import verify_with_vtk
        folder=folder/'software/vtu';bundle=read(folder/'bundle.json')
        if not bundle or not bundle.get('passed'):raise ValueError('First qualify the whole VTU export')
        result=verify_with_vtk(folder/'mesh.vtu',artifact.hdf5,length_unit=bundle['length_unit'])
        (folder/'native_vtk_readback.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
