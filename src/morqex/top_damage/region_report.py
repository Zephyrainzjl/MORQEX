"""Actual full-voxel field display for the regional finite-mechanics probe."""

def build_region_report(cfg,workspace):
    import os,json,base64
    from pathlib import Path
    import numpy as np,h5py
    from morqex import Workspace
    from morqex.top_damage.polycrystal import PolycrystalData
    from morqex.top_damage.morphology import measure_geometry
    from morqex.visualization.report_font import embed_font
    from .region_probe import _context
    ws,out=_context(cfg,workspace,True);root=Path(__file__).resolve().parent
    meta=json.loads((out/'manifest.json').read_text());feedback=json.loads((out/'feedback_probe.json').read_text())
    data=PolycrystalData.load_hdf5(ws.path('results/top_damage/microstructures',cfg.geometry,'initial.h5'));shape=data.structure.labels.shape;nodes=[];bounds={k:[np.inf,-np.inf] for k in ['sto','damage','principal','mu','fatigue']}
    with h5py.File(out/'trial_macro.h5') as f:c=f['voxel_concentration_mol_m3'][:]
    for node in range(cfg.nodes):
        with h5py.File(out/f'trial_node_{node:02d}.h5') as f:
            ids=f['voxel_ids'][:];cells=f['tetrahedra'][:];count=np.bincount(ids,minlength=np.prod(shape));record=json.loads(f.attrs['record'])
            def voxel(v):return (np.bincount(ids,weights=v,minlength=np.prod(shape))/np.maximum(count,1)).reshape(shape)
            # Average the stress tensor before computing the displayed voxel invariant.
            stress=f['cauchy_stress_Pa'][:];tensor=np.stack([voxel(stress[:,i,j]) for i in range(3) for j in range(3)],axis=-1).reshape(shape+(3,3))
            fields={'sto':c[node]/meta['cmax_mol_m3'],'damage':voxel(f['damage'][:][cells].mean(axis=1)),
              'principal':np.linalg.eigvalsh(tensor)[...,-1]/1e6,'mu':voxel(f['mechanical_chemical_potential_J_mol'][:]),'fatigue':voxel(f['fatigue_J_m3'][:])/1e6}
            encoded={}
            for key,a in fields.items():
                selected=a[data.structure.labels==1];bounds[key]=[min(bounds[key][0],float(selected.min())),max(bounds[key][1],float(selected.max()))]
                encoded[key]=base64.b64encode(np.asarray(a,dtype='<f4').tobytes()).decode()
            nodes.append({'record':record,'fields':encoded,'tetrahedra':len(cells),'mean_sto':float(c[node][data.structure.labels==1].mean()/meta['cmax_mol_m3'])})
    payload={'shape':list(shape),'spacing':(np.asarray(data.structure.spacing_zyx_m)[::-1]*1e6).tolist(),'origin':[0,0,0],
     'phase':data.structure.labels.ravel().tolist(),'particle':data.particle_ids.ravel().tolist(),'grain':data.grain_ids.ravel().tolist(),'kinds':data.structure.phase_kinds,
     'regions':meta['regions'],'metrics':measure_geometry(data),'nodes':nodes,'bounds':bounds,'feedback':feedback}
    page=(root/'morphology_view.html').read_text().replace('__MORQEX_DATA__',json.dumps(payload,separators=(',',':')).replace('<','\\u003c'))
    page=page.replace('少量大颗粒，看清内部晶粒与 CBD','六个 DFN 节点 → 六个 32³ 全体素微结构状态')
    start=page.index('<div class="note">');end=page.index('</div>',start)+6
    page=page[:start]+'''<div class="note">本页为实际 30 秒 DFN 区域浓度驱动的六组三维有限变形/疲劳微状态，局部化学势再反馈到 30–60 秒 DFN。每节点拥有独立浓度和力学状态，共用初始几何模板。使用 64 个连接区域替代径向颗粒状态，未将平均 SOC 复制成真实浓度场。当前为显式滞后耦合探针：没有完成时间步收敛、裂纹侵液/开孔拓扑、长期循环和材料标定，不能当作全部验收通过。</div>'''+page[end:]
    start=page.index('<p><a href="initial.h5">');end=page.index('</p>',start)+4
    page=page[:start]+'<p>完整 FP64 HDF5 位于 morph_workspace/results/top_damage/region_finite_32_probe；页面为 FP32 显示副本。浓度/化学势来自区域变分映射，应力来自实际 Tet4 求解。</p>'+page[end:]
    controls='''<section class="panel"><h2>正极厚度：隔膜侧 → 集流体侧（点击节点）</h2><div id="nodebuttons" class="controls" style="padding:18px"></div><p id="mapping" style="padding:0 18px"></p></section><div class="controls"><label>物理场 <select id="field"><option value="sto">嵌锂分数</option><option value="principal">最大主应力 / MPa</option><option value="damage">总损伤 d</option><option value="fatigue">疲劳历史 / MJ m⁻³</option><option value="mu">力学化学势 / J mol⁻¹</option><option value="geometry">初始颗粒 / 晶粒</option></select></label><span id="scale"></span></div>'''
    page=page.replace('<div class="views">',controls+'<div class="views">',1)
    page=page.replace('</style>','.panel{min-width:0}p,footer,pre{overflow-wrap:anywhere}label{max-width:100%}button.active{border:2px solid #54d7c7}#nodebuttons button{flex:1;min-width:75px}</style>')
    js='''<script>
    let selectedNode=0, physical={};const originalColor=color;
    function decode(s){let a=atob(s),b=new Uint8Array(a.length);for(let i=0;i<a.length;i++)b[i]=a.charCodeAt(i);return new Float32Array(b.buffer)}
    color=function(i,shade=1){const key=$('field').value;if(key==='geometry')return originalColor(i,shade);if(kind(i)!=='active')return `hsl(209 13% ${54*shade}%)`;const [lo,hi]=DATA.bounds[key],v=Math.max(0,Math.min(1,(physical[key][i]-lo)/(hi-lo||1)));return `hsl(${235-235*v} 80% ${(36+20*v)*shade}%)`};
    function showNode(n){selectedNode=n;const node=DATA.nodes[n];physical=Object.fromEntries(Object.entries(node.fields).map(([k,v])=>[k,decode(v)]));$('nodebuttons').innerHTML=DATA.nodes.map((v,i)=>`<button class="${i===n?'active':''}" onclick="showNode(${i})">节点 ${i+1}<br>平均嵌锂 ${v.mean_sto.toFixed(5)}</button>`).join('');$('mapping').textContent=`DFN 厚度控制体 ${n+1}/6 → 同模板 32768 体素，独立 ${DATA.regions} 浓度区域 → 127872 Tet4 力学状态 → 本节点 ${DATA.regions} 个投影力学化学势。平衡残差 ${node.record.equilibrium_residual.toExponential(2)}，最小 J ${node.record.minimum_jacobian.toFixed(6)}。`;const key=$('field').value;$('scale').textContent=key==='geometry'?'原始几何编号':`六节点共用色标：${DATA.bounds[key][0].toPrecision(5)} … ${DATA.bounds[key][1].toPrecision(5)}`;mesh();draw();document.body.dataset.node=n;document.body.dataset.regions=DATA.regions;}
    $('field').onchange=()=>showNode(selectedNode);$('cut').value=24;$('plane').value=23;showNode(0);if(document.fonts)document.fonts.ready.then(()=>showNode(selectedNode));
    </script>'''
    page=page.replace('使用 64 个连接区域','使用 '+str(meta['regions'])+' 个连接区域（请求 '+str(cfg.requested_regions)+'，必要时拆分不连通区域）')
    page=page.replace('127872 Tet4','${node.tetrahedra} Tet4').replace("$('cut').value=24;$('plane').value=23;","$('cut').value=Math.max(1,Math.round(.75*nz));$('plane').value=Math.max(0,Math.round(.75*nz)-1);")
    page=page.replace('</html>',js+'</html>')
    font=ws.path('data/report_font');page=embed_font(page,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
    page=page.replace('六个 DFN 节点 → 六个 32³ 全体素微结构状态',str(cfg.nodes)+' 个 DFN 节点 → '+str(cfg.nodes)+' 个 '+ '×'.join(map(str,shape))+' 全体素微结构状态')
    page=page.replace('实际 30 秒 DFN','实际 '+str(cfg.time_step_s)+' 秒 DFN').replace('到 30–60 秒 DFN','到下一时间段 DFN').replace('六组三维','各节点三维').replace('六节点','全部节点').replace('六个节点','全部节点').replace('/6 →','/${DATA.nodes.length} →').replace('32768 体素','${DATA.phase.length} 体素')
    page=page.replace('morph_workspace/results/top_damage/region_finite_32_probe','morph_workspace/results/top_damage/'+cfg.output)
    path=ws.path('reports',cfg.output+'.html');temporary=path.with_suffix('.pending');temporary.write_text(page);temporary.replace(path)
    return {'report':str(path),'nodes':cfg.nodes,'full_voxels_per_node':int(np.prod(shape)),'scope':feedback['scope']}
