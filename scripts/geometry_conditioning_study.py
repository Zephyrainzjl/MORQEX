"""Compare fixed-grid smoothing and explicit resampling on a complete source."""
import argparse,html,json,os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'));p.add_argument('--case',choices=['nmc','granular','rods','platelets','agglomerates','bicontinuous'],default='rods');a=p.parse_args()
    from morqex.modeling import VoxelModel,SmoothingOptions,smooth_voxels,PreparedGeometry,lower_resolution,HTMLProgress,ProgressBar,voxel_fingerprint
    from morqex.modeling.voxel import external_path
    work=external_path(a.workspace);source=work/'nmc_complex_gallery'/('nmc_tiff' if a.case=='nmc' else 'native/'+a.case)/'geometry.h5'
    model=VoxelModel.load(source);signature=voxel_fingerprint(model);root=work/'conditioning_study'/a.case;root.mkdir(parents=True,exist_ok=True)
    live=HTMLProgress(root/'progress.html');bar=ProgressBar(json_lines=True)
    def callback(e):live(e);bar(e)
    entries=[{'id':'original','title':'Original / 原始','shape':list(model.regions.shape),'spacing':list(model.spacing),'report':None}]
    if not (root/'original.html').exists():model.viewer(maximum_3d_size=64,slice_origin='lower',title=a.case+' / original complete geometry').save_html(root/'original.html')
    try:
        for name in ('very_fine','fine','balanced','coarse'):
            directory=root/name;directory.mkdir(exist_ok=True);saved=directory/'prepared.h5'
            if saved.exists():result=PreparedGeometry.load(saved)
            else:
                result=smooth_voxels(model,options=SmoothingOptions.precision(name,acceptance='report'),progress=callback)
                result.save(saved,progress=callback)
            assert result.model.regions.shape==model.regions.shape and result.model.spacing==model.spacing and result.model.origin==model.origin
            (directory/'report.json').write_text(json.dumps(result.report,indent=2))
            if not (root/(name+'.html')).exists():result.model.viewer(maximum_3d_size=64,slice_origin='lower',title=a.case+' / '+name+' SAME full grid').save_html(root/(name+'.html'))
            entries.append({'id':name,'title':name+' smoothing / 原尺寸平滑','shape':list(result.model.regions.shape),'spacing':list(result.model.spacing),'report':result.report})
            del result
        target=root/'reduced';target.mkdir(exist_ok=True)
        if (target/'geometry.h5').exists():
            reduced=VoxelModel.load(target/'geometry.h5');report=json.loads((target/'report.json').read_text())
        else:
            reduced,report=lower_resolution(model,factor=2,progress=callback);reduced.save(target/'geometry.h5');(target/'report.json').write_text(json.dumps(report,indent=2))
        if not (root/'reduced.html').exists():reduced.viewer(maximum_3d_size=64,slice_origin='lower',title=a.case+' / LOWER RESOLUTION, same physical box').save_html(root/'reduced.html')
        entries.append({'id':'reduced','title':'Lower resolution / 降低分辨率','shape':list(reduced.regions.shape),'spacing':list(reduced.spacing),'report':report})
        assert voxel_fingerprint(model)==signature
        record={'source':str(source),'original_fingerprint':signature,'source_array_unmodified':True,'cases':entries,'scope':'complete arrays, comparison only; rejected fidelity candidates remain visibly unaccepted'}
        (root/'comparison.json').write_text(json.dumps(record,indent=2));live.finish(message='Comparison complete / 对照完成；红色候选不表示已通过保真验收')
        doc='''<!doctype html><html lang="en"><meta charset="utf-8"><title>MORQEX · smoothing versus resolution</title><style>body{font:16px system-ui;background:#edf3f6;color:#173b4b;margin:25px}main{max-width:1850px;margin:auto}section{background:white;padding:22px;border-radius:12px;margin:18px 0}.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}iframe{height:950px;width:100%;border:1px solid #ddd}select{font-size:16px;padding:10px}pre{white-space:pre-wrap}td,th{padding:12px;border-bottom:1px solid #ddd}table{border-collapse:collapse;width:100%}@media(max-width:1100px){.views{grid-template-columns:1fr}}</style><main><h1>Fixed-grid smoothing ≠ lower resolution / 原尺寸平滑 ≠ 降低分辨率</h1><section><p>All smoothing candidates keep the original array shape, spacing, origin and physical box. Only the explicit lower-resolution case reduces voxel counts. / 所有平滑保持原数组尺寸、间距、原点和范围，只有明确降低分辨率的选项减少体素数。</p><label>Compare with original / 与原始对照 <select id="case"></select></label><p id="state"></p><div class="views" id="views"></div></section><section><h2>Measured sizes and fidelity / 实测尺寸与保真度</h2><table id="table"></table><p>Red fidelity failures are exploratory comparisons, not accepted geometry. Complete HDF5 arrays and JSON reports remain available. 3D display LOD does not reduce computational arrays. / 红色未通过项仅为对照，不是合格几何；三维显示LOD不改变完整计算数据。</p><a href="comparison.json">All measured data / 全部实测</a> · <a href="progress.html">Progress / 进度</a></section></main><script>const data=__DATA__,$=id=>document.getElementById(id);$('case').replaceChildren(...data.cases.filter(x=>x.id!=='original').map(x=>new Option(x.title,x.id)));$('case').value='balanced';function show(){let d=data.cases.find(x=>x.id===$('case').value);$('state').textContent=JSON.stringify({shape:d.shape,spacing_m:d.spacing,operation:d.report.operation,fidelity_passed:d.report.fidelity_limits_satisfied,changed_fraction:d.report.fidelity?.changed_fraction});$('views').replaceChildren(...['original.html',d.id+'.html'].map(src=>{let f=document.createElement('iframe');f.src=src;return f;}));document.body.dataset.case=d.id;document.body.dataset.rendered='true';}$('case').onchange=show;let header='<tr><th>Operation</th><th>ZYX shape</th><th>Spacing[m]</th><th>Changed fraction</th><th>Fidelity</th></tr>';$('table').innerHTML=header+data.cases.map(d=>'<tr style="color:'+(d.report?.fidelity_limits_satisfied===false?'#b33':'inherit')+'"><td>'+d.title+'</td><td>'+d.shape.join(' × ')+'</td><td>'+d.spacing.join(', ')+'</td><td>'+(d.report?.fidelity?.changed_fraction??'—')+'</td><td>'+(d.report?.fidelity_limits_satisfied??'separate operation')+'</td></tr>').join('');show();</script></html>'''
        from morqex.visualization.report_font import embed_workspace_font
        (root/'index.html').write_text(embed_workspace_font(doc.replace('__DATA__',json.dumps(record,separators=(',',':')).replace('<','\\u003c')),work))
        print(json.dumps({'html':str(root/'index.html'),'full_shape':model.regions.shape,'original_unmodified':True}))
    except BaseException as error:live.finish(message='FAILED / 失败: '+str(error));raise

if __name__=='__main__':main()
