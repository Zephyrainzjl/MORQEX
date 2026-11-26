"""Read accepted dense-cycle fields; publish exact records and FP32 display data."""
import base64,json
from pathlib import Path
import numpy as np
import h5py
from morqex import Workspace
from morqex.mechanics.tensors import MANDEL_BASIS
from morqex.top_damage.polycrystal import PolycrystalData
from morqex.top_damage.morphology import measure_geometry
from .dense_fields import DisplacementRecovery


def _encoded(a):
    return base64.b64encode(np.asarray(a,dtype='<f4').tobytes()).decode()


def build_dense_report(workspace,case_names=('gentle_v2','fast_v2','released_v2'),output_name=None):
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    if len(case_names)>1:
        summaries=[];links=[]
        for name in case_names:
            if not ws.path('results/top_damage/dense_cycles',name,'manifest.json').exists():continue
            path=build_dense_report(ws,(name,),output_name='top_damage_dense_cases/'+name+'.html')
            summaries.append(json.loads(path.with_suffix('.json').read_text()))
            links.append(('top_damage_dense_cases/'+name+'.html',name))
        if not summaries:raise ValueError('No accepted comparison cases')
        summary={'cases':[c for s in summaries for c in s['cases']],
            'all_numerical_protocols_complete':all(s['all_numerical_protocols_complete'] for s in summaries),
            'physical_topology_qualified':False,'publication_qualified':False}
        names={'gentle':'0.5C / 约束边界','fast':'2C / 约束边界','released':'0.5C / 自由边界'}
        labels_by_case={c['name']:names.get(c['manifest'].get('environment'),c['name']) for c in summary['cases']}
        options=''.join('<option value="'+url+'">'+labels_by_case.get(name,name)+'</option>' for url,name in links)
        rows=''
        for c in summary['cases']:
            r=c['records'][-1]
            rows+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in [labels_by_case[c['name']],
                len(c['records']),f"{r['mean_active_damage']:.6f}",f"{max(a['max_principal_stress_P99_MPa'] for a in c['records']):.2f}",
                f"{r['next_feedback']['electronic_conductivity_S_m']:.6f}",f"{max(a['inventory_relative_defect'] for a in c['records']):.2e}",format(c['validation']['maximum_cell_total_strain_frobenius'],'.2%') if c.get('validation') else '待核验'] )+'</tr>'
        page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MORQEX 三维疲劳工况对照</title><style>body{margin:0;background:#0b1220;color:#dce9f5;font:16px/1.7 system-ui}main{max-width:1460px;margin:auto;padding:25px}h1{color:#64d8cd}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #30425a;padding:9px}.note{background:#292638;border-left:4px solid #e5b15a;padding:16px}select{font:inherit;background:#172b42;color:white;padding:10px}a{color:#6ce0d1}iframe{width:100%;height:1700px;border:1px solid #344b65;margin-top:18px}</style><main><h1>MORQEX / TOP-DAMAGE：实际三维工况对照</h1><p>32³ 体素，8 个大颗粒，孔隙率 34.96%。所有场来自实际接受的计算状态。为降低本地内存占用，每次仅加载一个工况的三维场。</p><p class="note">当前为数值候选闭环，尚未通过物理裂缝开度、侵液、颗粒失联及论文验收。不能将损伤阈值图当作真实孔隙；完整窗口算例的局部小应变筛查未通过，不能视为物理参考解。各工况页明确标注协议、材料假设和已接受步。</p><table><tr><th>工况</th><th>已保存状态</th><th>末态平均损伤</th><th>过程最大 P99 主应力 / MPa</th><th>末态电导 / S m⁻¹</th><th>最大锂守恒误差</th><th>最大局部总应变</th></tr>'''+rows+'''</table><p><label>查看工况 <select id="case">'''+options+'''</select></label> · <a id="direct" href="'''+links[0][0]+'''">单独打开完整交互页</a></p><iframe id="fields" title="逐步三维场" src="'''+links[0][0]+'''"></iframe></main><script>document.getElementById('case').onchange=e=>{document.getElementById('fields').src=e.target.value;document.getElementById('direct').href=e.target.value};document.body.dataset.comparison='true';</script></html>'''
        page=page.replace('32³ 体素，8 个大颗粒，孔隙率 34.96%。','结构尺寸、颗粒数和孔隙率见各工况页。')
        page=page.replace('完整窗口算例的局部小应变筛查未通过，不能视为物理参考解。','各工况分别报告小变形适用性或有限变形雅可比检查，均不自动构成材料标定。')
        page=page.replace('<table>','<div class="table-scroll"><table>').replace('</table>','</table></div>')
        page=page.replace('</style>','*{box-sizing:border-box}.table-scroll{overflow:auto}main{min-width:0}iframe{display:block;max-width:100%}a,footer{overflow-wrap:anywhere}@media(max-width:650px){main{padding:14px}h1{font-size:24px}select{max-width:100%}}</style>')
        from morqex.visualization.report_font import embed_font
        font=ws.path('data/report_font');page=embed_font(page,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
        target=ws.path('reports',output_name or 'MORQEX_TOP_DAMAGE_dense_cycles.html');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(page)
        summary['report']=str(target);target.with_suffix('.json').write_text(json.dumps(summary,indent=2));return target
    cases=[];bounds={key:[float('inf'),float('-inf')] for key in ['damage','increment','sto','principal','hydrostatic','fatigue','strain_percent']}
    geometry=None
    for name in case_names:
        directory=ws.path('results/top_damage/dense_cycles',name)
        if not (directory/'manifest.json').exists():continue
        manifest=json.loads((directory/'manifest.json').read_text())
        data=PolycrystalData.load_hdf5(directory/'initial_geometry.h5')
        if geometry is None:geometry=data
        elif geometry.fingerprint!=data.fingerprint:raise ValueError('Environment comparison requires identical initial geometry')
        frames=[];curves=[];initial_damage=None;previous_damage=None
        recovery=DisplacementRecovery(directory,data.structure.labels.shape)
        for path in sorted(directory.glob('step_*.h5')):
            with h5py.File(path,'r') as f:
                if not f.attrs['complete'] or f.attrs['identity']!=manifest['identity']:raise ValueError('Only accepted matching snapshots can be plotted')
                record=json.loads(f.attrs['record']);d=f['voxel/damage'][:];stress=f['voxel/stress_Mandel_Pa'][:]
                if initial_damage is None:initial_damage=d
                if previous_damage is not None and np.min(d-previous_damage)<-1e-12:raise ValueError('Irreversibility failed in accepted fields')
                previous_damage=d
                c=f['voxel/concentration_mol_m3'][:]
                # Model c_max is required from explicit config/material record, never fit to this snapshot.
                import pybamm as pb
                from morqex.electrochemistry.open_materials import HOSTS
                cmax=float(pb.ParameterValues(HOSTS[manifest['config']['host']])['Maximum concentration in positive electrode [mol.m-3]'])
                fields={'damage':d,'increment':d-initial_damage,'sto':c/cmax,
                    'principal':np.linalg.eigvalsh(np.einsum('...k,kij->...ij',stress,MANDEL_BASIS))[...,-1]/1e6,
                    'hydrostatic':stress[...,:3].mean(axis=-1)/1e6,'fatigue':f['voxel/fatigue_J_m3'][:]/1e6}
                strain,_=recovery.fields(f['displacement_m'][:],finite=manifest.get('mechanics_model')=='finite-strain')
                fields['strain_percent']=100*np.linalg.norm(strain,axis=(-2,-1))
                # Increment is decoded as d-d0 in the browser, avoiding duplicate display storage.
                encoded={key:_encoded(value) for key,value in fields.items() if key!='increment'}
                for key,value in fields.items():
                    selected=value[data.structure.labels!=0] if key=='strain_percent' else value[data.structure.labels==1]
                    bounds[key][0]=min(bounds[key][0],float(selected.min()))
                    bounds[key][1]=max(bounds[key][1],float(selected.max()))
                frames.append({'record':record,'fields':encoded,'file':str(path)})
                if 'curve' in f:
                    curves.append({key:f['curve/'+key][:].tolist() for key in f['curve']})
        if not frames:continue
        capacities=[];previous_q=0.
        per_half=manifest['config']['increments_per_half_cycle']
        for cycle in range(manifest['config']['cycles'] if manifest['config'].get('protocol','partial-depth')=='partial-depth' else 0):
            end_dis=(2*cycle+1)*per_half;end_chg=(2*cycle+2)*per_half
            if end_chg>=len(frames):break
            dis=frames[end_dis]['record']['net_capacity_Ah'];chg=frames[end_chg]['record']['net_capacity_Ah']
            capacities.append({'cycle':cycle+1,'discharged_Ah':dis-previous_q,'charged_Ah':dis-chg})
            previous_q=chg
        if manifest['config'].get('protocol')=='voltage-window':
            discharged=None
            for frame in frames:
                r=frame['record'];q=r.get('completed_phase_capacity_Ah')
                if q is None:continue
                if r['protocol_phase']=='discharge':discharged=q
                elif r['protocol_phase']=='charge' and discharged is not None:
                    capacities.append({'cycle':r['completed_cycles'],'discharged_Ah':discharged,'charged_Ah':q});discharged=None
        validation=json.loads((directory/'numerical_validation.json').read_text()) if (directory/'numerical_validation.json').exists() else None
        cases.append({'name':name,'manifest':manifest,'frames':frames,'curves':curves,'capacities':capacities,'validation':validation})
    if not cases:raise ValueError('No accepted dense-cycle data')
    root=Path(__file__).resolve().parents[0]
    # When installed this module sits next to the shared voxel-view template.
    template=root/'morphology_view.html'
    if not template.exists():
        import morqex.top_damage.morphology as morphology
        template=Path(morphology.__file__).with_name('morphology_view.html')
    data=geometry
    payload={'shape':list(data.structure.labels.shape),'spacing':(np.asarray(data.structure.spacing_zyx_m)[::-1]*1e6).tolist(),
        'origin':(np.asarray(data.structure.origin_zyx_m)[::-1]*1e6).tolist(),
        'phase':data.structure.labels.ravel().tolist(),'particle':data.particle_ids.ravel().tolist(),'grain':data.grain_ids.ravel().tolist(),
        'kinds':data.structure.phase_kinds,'metrics':measure_geometry(data),'cases':cases,'bounds':bounds}
    page=template.read_text().replace('__MORQEX_DATA__',json.dumps(payload,separators=(',',':')).replace('<','\\u003c'))
    page=page.replace('少量大颗粒，看清内部晶粒与 CBD','较密大颗粒：DFN 驱动的三维疲劳候选计算')
    page=page.replace('这里展示真实生成的三维体素及其编号。旋转视角、剖开颗粒，或隐藏 CBD 检查内部结构；没有将体素重新画成光滑球面。',
        '同一 32³ 初始结构，比较倍率和机械约束。下方每一帧均来自实际接受的 DFN—三维力学—AT2 疲劳求解，不是动画插值。相场、浓度、应力和疲劳历史可逐步剖切。')
    page=page.replace('同一 32³ 初始结构','同一 '+'×'.join(map(str,geometry.structure.labels.shape[::-1]))+' 初始结构')
    page=page.replace('</style>','.panel{min-width:0}pre,footer,a{overflow-wrap:anywhere}td{overflow-wrap:anywhere}</style>')
    start=page.index('<div class="note">');end=page.index('</div>',start)+6
    page=page[:start]+'''<div class="note"><b>候选模型，尚未通过物理拓扑与论文验收。</b>这是 2 次、每半循环 30% 名义容量的 CC/CC 部分深度协议，不是全电压窗口容量衰减测试。NMC811 电化学参数与数值力学/疲劳参数分别记录；有明确初始预裂纹。局部二次损伤电导律未经裂缝开度/接触标定；没有将高损伤体素删除为孔隙，也没有验证裂纹侵液和颗粒失联。径向浓度映射只用于本组等粒径球形颗粒及所选正极节点。图中的场变化不能直接作为实际材料寿命结论。</div>'''+page[end:]
    audit=cases[0].get('validation')
    if audit and audit['small_strain_screen_passed'] is False:
        warning='<div class="note"><b>小应变适用性筛查未通过。</b>全过程最大单元总应变 Frobenius 范数为 '+format(audit['maximum_cell_total_strain_frobenius'],'.2%')+'，超过本研究采用的 2% 保守筛查线。此筛查不是严格误差界；应检查 CBD/裂尖有限变形与网格收敛，不能将方程残差通过视作材料预测通过。</div>'
        page=page.replace('<div class="views">',warning+'<div class="views">',1)
    if cases[0]['manifest']['config'].get('protocol')=='voltage-window':
        page=page.replace('这是 2 次、每半循环 30% 名义容量的 CC/CC 部分深度协议，不是全电压窗口容量衰减测试。','这是先预充、再按上下截止电压进行的 CC/CC 协议，没有 CV 阶段；实际完成状态见记录。短循环不能确立容量衰减规律。不同倍率的容量差异含极化和截止电压影响，不能直接归因于退化。')
    controls='''<div class="controls"><label>工况 <select id="case"></select></label><label>场 <select id="physical"><option value="increment">损伤增量 Δd</option><option value="damage">总相场 d</option><option value="sto">局部嵌锂分数</option><option value="principal">最大主应力 / MPa</option><option value="hydrostatic">静水应力 / MPa</option><option value="fatigue">疲劳历史 / MJ m⁻³</option><option value="strain_percent">体素平均总应变范数 / %（含 CBD）</option><option value="geometry">颗粒 / 晶粒几何</option></select></label><label>接受步 <input id="accepted" type="range" min="0"><span id="frameText"></span></label></div><p id="fieldlegend"></p>'''
    page=page.replace('<div class="views">',controls+'<div class="views">',1)
    if cases[0]['manifest'].get('mechanics_model')=='finite-strain':
        page=page.replace('体素平均总应变范数 / %（含 CBD）','Green–Lagrange 应变范数 / %（含 CBD）')
        page=page.replace('<div class="views">','<div class="note">有限变形参考：乘法化学变形、MFEM 超弹性解析切线；应力为 Cauchy，应变为 Green–Lagrange，统计使用参考体积。原始 HDF5 保存 F、化学 F、第一 Piola 应力及力学化学势。正雅可比通过不等于裂面接触、材料标定或寿命验证通过。</div><div class="views">',1)
    dashboard='''<section class="panel"><h2>实际计算记录与协议</h2><pre id="state" style="padding:20px;white-space:pre-wrap"></pre><a id="hdf5" style="padding:20px">下载当前完整 HDF5 场</a><div class="views"><canvas id="voltage"></canvas><canvas id="evolution"></canvas></div></section>'''
    page=page.replace('<table><thead>',dashboard+'<table><thead>',1)
    links_start=page.index('<p><a href="initial.h5">');links_end=page.index('</p>',links_start)+4
    page=page[:links_start]+'''<p>完整 FP64 节点位移/相场、四面体网格、体素场及 DFN 状态均保存在各工况结果目录。页面显示副本为 FP32，不参与计算或误差验收。</p>'''+page[links_end:]
    extension='''<script>
const oldColor=color;let currentFields={},initialD=null;
const titles=Object.fromEntries(DATA.cases.map(c=>[c.name,({gentle:'0.5C · 外边界约束',fast:'2C · 外边界约束',released:'0.5C · 自由外边界'})[c.manifest.environment]||c.name]));
$('case').innerHTML=DATA.cases.map((c,i)=>`<option value="${i}">${titles[c.name]||c.name}</option>`).join('');
function decode(s){const a=atob(s),u=new Uint8Array(a.length);for(let i=0;i<a.length;i++)u[i]=a.charCodeAt(i);return new Float32Array(u.buffer)}
color=function(i,shade=1){const field=$('physical').value;if(field==='geometry')return oldColor(i,shade);if(kind(i)!=='active'&&(field==='sto'||field==='increment'))return `hsl(209 13% ${54*shade}%)`;let value=field==='increment'?currentFields.damage[i]-initialD[i]:currentFields[field][i],lo=DATA.bounds[field][0],hi=DATA.bounds[field][1],v=Math.max(0,Math.min(1,(value-lo)/(hi-lo||1)));return `hsl(${235-235*v} 80% ${(36+20*v)*shade}%)`};
function curve(id,series,title){let [ctx,w,h]=canvas(id);ctx.font='13px MORQEXReportCJK,system-ui';ctx.fillStyle='#b9cadc';ctx.fillText(title,18,25);let xs=series.flatMap(s=>s.x),ys=series.flatMap(s=>s.y);if(!xs.length)return;let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);const X=x=>55+(w-85)*(x-xmin)/(xmax-xmin||1),Y=y=>h-50-(h-100)*(y-ymin)/(ymax-ymin||1);ctx.strokeStyle='#536781';ctx.strokeRect(55,50,w-85,h-100);series.forEach((s,k)=>{ctx.strokeStyle=['#54d7c7','#f4bc67','#8c9dff'][k%3];ctx.beginPath();s.x.forEach((x,j)=>j?ctx.lineTo(X(x),Y(s.y[j])):ctx.moveTo(X(x),Y(s.y[j])));ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.fillText(s.name,65,70+k*20)});ctx.fillStyle='#b9cadc';ctx.fillText(ymax.toPrecision(4),4,50);ctx.fillText(ymin.toPrecision(4),4,h-50);ctx.fillText(xmax.toPrecision(4),w-65,h-25)}
function selectFrame(reset=false){const c=DATA.cases[+$('case').value];$('accepted').max=c.frames.length-1;if(reset)$('accepted').value=c.frames.length-1;const f=c.frames[+$('accepted').value],r=f.record;initialD=decode(c.frames[0].fields.damage);currentFields=Object.fromEntries(Object.entries(f.fields).map(([k,v])=>[k,decode(v)]));$('frameText').textContent=`${r.step}/${c.frames.length-1}（已保存） · ${r.time_s.toFixed(0)} s`;let field=$('physical').value;$('fieldlegend').textContent=field==='geometry'?'原始材料/颗粒/晶粒编号':`当前工况全部时刻采用相同色标：${DATA.bounds[field][0].toPrecision(5)} … ${DATA.bounds[field][1].toPrecision(5)}；场值为体素体积平均。`;$('state').textContent=JSON.stringify({environment:titles[c.name],protocol_complete:c.manifest.protocol_complete,step:r.step,time_s:r.time_s,cycle:r.mechanical_cycle,phase:r.protocol_phase,completed_cycles:r.completed_cycles,voltage_V:r.voltage_V,mean_active_damage:r.mean_active_damage,P99_principal_MPa:r.max_principal_stress_P99_MPa,electronic_conductivity_S_m:r.next_feedback.electronic_conductivity_S_m,mechanical_potential_J_mol:r.next_feedback.mechanical_potential_J_mol,lithium_error:r.inventory_relative_defect,mechanics_residual:r.equilibrium_residual,damage_KKT:r.damage_kkt,diagnostic_components:r.diagnostic_intact_components,capacities:c.capacities},null,2);$('hdf5').href=f.file;mesh();draw();curve('voltage',[{name:'电压 / V 对时间 / s',x:c.curves.flatMap(p=>p.time_s),y:c.curves.flatMap(p=>p.voltage_V)}],'求解得到的充放电电压');curve('evolution',DATA.cases.map(a=>({name:titles[a.name]||a.name,x:a.frames.map(f=>f.record.time_s),y:a.frames.map(f=>f.record.mean_active_damage)})),'活性材料平均损伤对时间 / s');document.body.dataset.accepted=r.step;document.body.dataset.environment=c.name;}
$('case').onchange=()=>selectFrame(true);$('accepted').oninput=()=>selectFrame();$('physical').onchange=()=>selectFrame();$('cut').value=Math.max(1,Math.round(.75*nz));$('plane').value=Math.max(0,Math.round(.75*nz)-1);selectFrame(true);if(document.fonts)document.fonts.ready.then(()=>selectFrame());
</script>'''
    page=page.replace('</html>',extension+'</html>')
    font=ws.path('data/report_font')
    from morqex.visualization.report_font import embed_font
    page=embed_font(page,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
    target=ws.path('reports',output_name or 'MORQEX_TOP_DAMAGE_dense_cycles.html');target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix('.pending');temporary.write_text(page);temporary.replace(target)
    summary={'cases':[{k:c[k] for k in ('name','manifest','capacities','validation')}|{'records':[f['record'] for f in c['frames']]} for c in cases],
             'all_numerical_protocols_complete':all(c['manifest'].get('protocol_complete',False) for c in cases),
             'physical_topology_qualified':False,'publication_qualified':False,'report':str(target)}
    target.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    return target


if __name__=='__main__':
    import os
    print(build_dense_report(os.environ['MORQEX_WORKSPACE']))
