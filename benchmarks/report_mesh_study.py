"""Static spatial study: observed convergence, costs and retained failures."""
import html,json,os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from morqex import Workspace

ws=Workspace(os.environ['MORQEX_WORKSPACE']);out=ws.path('reports/mesh_study');out.mkdir(exist_ok=True)
studies={'static_mesh_face_contacts_v3':'Tet4 uniform','static_mesh_adaptive_v1':'Tet4 adaptive',
         'static_mesh_quadratic_v1':'P2/P1 uniform','static_mesh_quadratic_storage_v2':'P2/P1 bounded storage',
         'static_mesh_quadratic_adaptive_v1':'P2/P1 adaptive candidate',
         'static_mesh_quadratic_adaptive_budget_v1':'P2/P1 bounded refinement branch'}
data={}
for name,label in studies.items():
    entries=[json.loads(p.read_text()) for p in ws.path('results/top_damage',name).glob('level*/result.json')]
    if entries:data[name]=sorted(entries,key=lambda x:x['refinement'])
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for name,rows in data.items():
    axes[0].plot([r.get('mechanical_points',r['points'])*3 for r in rows],[(r['elastic_energy_J']+r['fracture_energy_J'])*1e12 for r in rows],'o-',label=studies[name])
    axes[1].plot([r['tetrahedra'] for r in rows],[r['seconds'] for r in rows],'o-',label=studies[name])
axes[0].set(xscale='log',xlabel='Mechanical nodal degrees of freedom',ylabel='Total energy [pJ]',title='Same physical staircase geometry and phase-field length')
axes[1].set(xscale='log',yscale='log',xlabel='Parent tetrahedra',ylabel='Measured solve/setup time [s]',title='One CPU; different interpolation orders')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(out/'spatial_study.svg');fig.savefig(out/'spatial_study.png',dpi=160);plt.close(fig)
comparisons=[]
for pattern in ('static_mesh_*.json','finite_storage_equivalence.json','quadratic_*equivalence.json'):
    for p in sorted(ws.path('reports').glob(pattern)):
        comparisons.append({'file':str(p),'record':json.loads(p.read_text())})
evidence={'scope':'Static fixed geometry, fixed ell, fixed chemical loading. No cyclic-fatigue qualification.',
          'publication_qualified':False,'spatial_qualification_requires_review':True,'studies':data,'comparisons':comparisons}
resource_failures=[]
for pattern in ('*quadratic_adaptive_level3','*quadratic_adaptive_budget3'):
    for run in sorted(ws.path('runs').glob(pattern)):
        manifest=run/'manifest.json'
        if manifest.exists():
            record=json.loads(manifest.read_text())
            if record.get('exit_code')!=0:resource_failures.append(record)
evidence['retained_resource_rejections']=resource_failures
(out/'evidence.json').write_text(json.dumps(evidence,indent=2))
rows=''
for name,entries in data.items():
    for r in entries:
        vals=[studies[name],r['refinement'],r['tetrahedra'],r.get('mechanical_points',r['points']),
              f"{r['seconds']:.2f}",f"{r['equilibrium_residual']:.2e}",f"{r['damage_kkt']:.2e}"]
        rows+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>'
details=''.join('<details><summary>'+html.escape(Path(x['file']).name)+'</summary><pre>'+html.escape(json.dumps(x['record'],indent=2))+'</pre></details>' for x in comparisons)
details+=''.join('<details><summary>资源拒绝 / 未形成参考解</summary><pre>'+html.escape(json.dumps(r,indent=2))+'</pre></details>' for r in resource_failures)
document='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MORQEX 空间误差验证</title><style>body{font:16px/1.7 system-ui;background:#f1f5f7;color:#173044;margin:0}main{max-width:1150px;margin:auto;padding:30px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}.note{border-left:5px solid #bc8124}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:left;border-bottom:1px solid #d6e0e8}img{width:100%}pre{overflow:auto;font-size:12px}a{color:#176e86}</style><main><h1>三维力学—相场：空间误差与求解成本</h1><section class="note"><b>空间验收尚未通过，所有失败比较保留</b><p>保持几何、晶粒、材料、外边界、化学变形和相场长度不变。线性位移与二次位移采用相同自由能；二次位移配合一次损伤及正权重积分。局部加密由源码内 MFEM 执行，残差指标不是严格误差界。</p><p>能量下降与线性残差很小均不能单独证明场收敛。预设门槛：共同体素损伤差≤1%、应力差≤2%、平均应力差≤1%、总能量差≤1%。均匀与自适应历史均保留；较少单元不自动等于更快。</p></section><section><img src="spatial_study.svg" alt="相同物理问题的空间细化与成本曲线"></section><section><h2>实际计算记录</h2><table><thead><tr>'''+''.join('<th>'+x+'</th>' for x in ['离散','层级','母四面体','力学节点','秒','平衡残差','损伤 KKT'])+'''</tr></thead><tbody>'''+rows+'''</tbody></table></section><section><h2>独立比较与失败记录</h2>'''+details+'''</section><section><a href="evidence.json">完整证据 JSON</a> · <a href="../network_study/index.html">DFN 循环与场报告</a></section></main>'''
font=ws.path('data/report_font')
if (font/'NotoSansCJKsc-Regular.otf').exists():
    from morqex.visualization.report_font import embed_font
    document=embed_font(document,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
(out/'index.html').write_text(document);print(str(out/'index.html'))
