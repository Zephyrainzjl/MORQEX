"""Evidence dashboard of actual network-cycle checkpoints, never synthetic cycles."""
import json,os,html,csv
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from morqex.workspace import Workspace
from morqex.top_damage.network_report import audit_network_history


def first_discharge(directory,records):
    rows=[]
    for r in records[1:]:
        if r['protocol_phase']!='discharge':continue
        with h5py.File(directory/f"step_{r['step']:06d}.h5") as f:
            rows.extend(np.column_stack([f['curve/time_s'][:],f['curve/net_capacity_Ah'][:],f['curve/voltage_V'][:]]).tolist())
        if r['completed_phase_capacity_Ah'] is not None:break
    if not rows:return None
    a=np.array(rows);a=a[np.r_[True,np.diff(a[:,0])>0]];a[:,1]=(a[:,1]-a[0,1])*1000
    return a


def main():
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);root=ws.path('results/top_damage/network_cycles')
    names=['network_nmc811_05C_v4','dual_rate01_v1','dual_rate2_v1','dual_rate5_v1','dual_rate8_v1',
        'dual_rate8_symmetric_v1','dual_rate8_no_mechanics_v1','dual_seed105_v1','dual_seed106_v1',
        'dual_nmc532_v1','dual_lfp_v1','dual_lfp_matched_v1','dual_time48_v1','dual_time96_v1',
        'dual_regions_grain_local_v1','dual_regions64_local_v1','dual_regions128_local_v1',
        'dual_regions256_local_v1','dual_regions_full_local_v1','dual_regions_full_batched_v1',
        'dual_full8_time24_v1','dual_full8_time48_v1','dual_full8_time96_v1',
        'dual_full8_projected_v1','dual_full8_projected48_v1','dual_full8_projected96_v1',
        'dual_long20_256_v1','dual_charge_gate_probe_v1','dual_long20_checked_v1',
        'dual_no_feedback_v1','dual_exact_graph_v1']
    output=ws.path('reports/network_study');output.mkdir(exist_ok=True)
    summaries=[];curves={};long=[]
    for name in names:
        directory=root/name
        if not (directory/'manifest.json').exists():continue
        meta=json.loads((directory/'manifest.json').read_text());cfg=meta['config']
        entry={'case':name,'host':cfg['host'],'discharge_C':cfg['C_rate'],'charge_C':cfg['charge_C_rate'],
            'regions':meta['regions'],'nodes':meta['nodes'],'shape':meta['geometry'].get('shape_zyx'),
            'dfn_rtol':cfg.get('dfn_rtol',1e-8),'configuration':cfg}
        try:
            summary,records,_=audit_network_history(directory);entry.update(summary)
            curves[name]=first_discharge(directory,records)
            if name=='dual_long20_checked_v1':
                long=[(r['completed_cycles'],r['time_s'],max((d.get('fracture_energy_J',0.) for d in r['node_diagnostics']),default=0.)) for r in records if r['completed_phase_capacity_Ah'] is not None and r['protocol_phase']=='charge']
        except Exception as exc:entry['audit_error']=str(exc)
        matches=list(ws.path('reports').glob('*'+name.removeprefix('dual_').removesuffix('_v1')+'*.html'))
        entry['html']=str(matches[0]) if len(matches)==1 else None
        status=ws.path('experiments/fatigue_network',name.removeprefix('dual_').removesuffix('_v1')+'_status.json')
        if status.exists():entry['run_status']=json.loads(status.read_text()).get('state')
        if name in ('dual_long20_256_v1','dual_charge_gate_probe_v1'):entry['failure_note']='Rejected subsequent step: dense-output charge residual. Accepted checkpoints only; strict-tolerance replacement is separate.'
        summaries.append(entry)
    evidence={'schema':1,'scope':'Numerical candidate with fixed ionic geometry; no calibrated physical crack wetting/topology or lifetime claim.',
        'publication_qualified':False,'physical_topology_qualified':False,'cases':summaries}
    (output/'evidence.json').write_text(json.dumps(evidence,indent=2))
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6.4,4.2),layout='constrained')
    for name,label in [('dual_rate01_v1','0.1C'),('network_nmc811_05C_v4','0.5C'),('dual_rate2_v1','2C'),('dual_rate5_v1','5C'),('dual_rate8_v1','8C')]:
        a=curves.get(name)
        if a is not None:ax.plot(a[:,1],a[:,2],label=label,lw=1.4)
    ax.set(xlabel='First discharge capacity [mAh]',ylabel='Voltage [V]',title='Actual cutoff discharge; common 0.5C conditioning charge')
    ax.legend(title='Discharge rate');ax.grid(alpha=.15)
    fig.savefig(output/'rate_curves.svg');fig.savefig(output/'rate_curves.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,3.6),layout='constrained')
    for name,label in [('dual_rate8_symmetric_v1','8C / 8C, mechanics'),('dual_rate8_no_mechanics_v1','8C / 8C, no mechanics'),('dual_long20_checked_v1','0.5C / 0.5C, 257 regions, strict DAE')]:
        matching=[e for e in summaries if e['case']==name]
        if matching:
            q=matching[0].get('capacity_retention_percent',[])
            if q:ax.plot(np.arange(1,len(q)+1),q,'o-',ms=3,label=label)
    ax.set(xlabel='Actually integrated discharge number',ylabel='Capacity / first discharge [%]',title='Conditioning and damage effects are not yet separated')
    ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(output/'capacity_history.svg');plt.close(fig)
    rows=[]
    for e in summaries:
        scope='协议完成' if e.get('protocol_complete') else '接受断点 / 协议未完成'
        if e.get('failure_note'):scope='后续步拒绝（失败保留）'
        if e.get('audit_error'):scope='历史核查失败：'+e['audit_error']
        charge=f"{e['max_charge_relative_defect']:.2e}" if e.get('charge_and_local_port_history_checked') else '旧记录未检查'
        link=html.escape(e['case'])
        if e.get('html'):link='<a href="../'+html.escape(Path(e['html']).name)+'">'+link+'</a>'
        rows.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in [link,e['host'],f"{e['charge_C']}/{e['discharge_C']}",e['regions'],
            f"{e.get('completed_cycles',0)}/{e.get('requested_cycles','?')}",f"{e.get('max_inventory_relative_defect',float('nan')):.2e}",charge,
            f"{e.get('max_damage',float('nan')):.4g}",scope])+'</tr>')
    refinement=[]
    for name in ['concentration_refinement','concentration256_accepted','network_time_48_96','full8_time24_48','full8_time48_96','full8_projection_accuracy','graph_port_benchmark']:
        file=ws.path('reports',name+'.json')
        if file.exists():refinement.append('<details><summary>'+name+'</summary><pre>'+html.escape(file.read_text())+'</pre></details>')
    title='MORQEX · 疲劳相场—网络—DFN 数值证据'
    document='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+title+'''</title><style>
body{font:15px/1.65 system-ui,sans-serif;margin:0;background:#eef3f6;color:#152c3c}main{max-width:1250px;margin:auto;padding:32px}h1{font-size:28px}section{background:white;border-radius:12px;padding:24px;margin:20px 0;box-shadow:0 3px 14px #132c3c09}.notice{border-left:5px solid #b97419}table{width:100%;border-collapse:collapse;font-size:12px}td,th{text-align:left;border-bottom:1px solid #dde5eb;padding:9px}a{color:#146e89}img{max-width:100%;display:block}pre{overflow:auto;font-size:12px}.scroll{overflow:auto}.plots{display:grid;grid-template-columns:1fr 1fr;gap:15px}@media(max-width:800px){.plots{display:block}main{padding:10px}}
</style><main><h1>'''+title+'''</h1><p>仅使用实际接受的充放电步与三维状态。源码内编译 MFEM、源码内 PyBaMM；计算、断点和图表位于外部工作区。</p>
<section class="notice"><b>数值候选，尚未达到论文全部验收</b><p>当前疲劳使用 Carrara 型正变差历史和不可逆 AT2，电子/固相传输使用声明的弥散损伤导率。孔隙/CBD 离子几何固定；尚没有经验证的裂纹开度—侵液—新生反应面闭环。每个 SVE 独立固定外边界，未替代整电芯力平衡。迁移率参数、材料疲劳与网格仍需独立验证。</p><p>容量由实际电流积分取得。单纯早期容量变化不能归因于疲劳，损伤相场也不能直接等同开放裂缝。未积分的循环不绘制曲线。</p></section>
<section class="plots"><div><img src="rate_curves.svg" alt="真实倍率放电曲线"></div><div><img src="capacity_history.svg" alt="实际循环容量"></div></section>
<section><h2>逐工况核查</h2><p>倍率列为充电/放电；节点和区域是宏观厚度单元与微结构浓度空间。旧记录未检查的电荷门槛不会标为通过。</p><div class="scroll"><table><thead><tr>'''+''.join('<th>'+s+'</th>' for s in ['工况 / 场 HTML','材料接口','倍率 C','区域数','循环','Li 相对缺陷','电荷相对缺陷','最大节点 d','状态'])+'''</tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div></section>
<section><h2>误差与加速证据</h2><p>257 区域的一循环对照通过既定浓度空间门槛；64/128 区域的疲劳误差失败保留。它们均不是网格收敛证明。约 201 倍为冻结被动图的双端口查询加速，不能作为整电池加速。</p>'''+''.join(refinement)+'''</section><section><h2>可复现数据</h2><p><a href="../topology_study/index.html">相同失效面积、不同拓扑的 DFN 与反应端口对照</a></p><a href="evidence.json">完整配置与核查 JSON</a><p>HDF5 与逐节点 VTI/VTU/PVD 保留完整计算精度。均匀化单位载荷电势不是运行电池中的局部电势。</p></section></main>'''
    font=ws.path('data/report_font')
    if (font/'NotoSansCJKsc-Regular.otf').exists():
        from morqex.visualization.report_font import embed_font
        document=embed_font(document,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
    (output/'index.html').write_text(document)
    print(json.dumps({'html':str(output/'index.html'),'cases':len(summaries),'publication_qualified':False}))

if __name__=='__main__':main()
