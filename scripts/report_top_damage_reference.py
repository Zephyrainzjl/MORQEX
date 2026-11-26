"""Publish measured reference evidence and full-field links; no synthetic data."""
from html import escape
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from morqex import Workspace
from morqex.top_damage.reference_run import read_records
from morqex.visualization.report_font import embed_font


ws=Workspace(os.environ['MORQEX_WORKSPACE'])
reference=json.loads(ws.path('reports','top_damage_reference_vendored.json').read_text())
study=json.loads(ws.path('reports','top_damage_path_vendored.json').read_text())
if not reference['passed'] or not study['cache_field_gate_passed']:raise RuntimeError('Do not publish failed mathematical gates as passed')
cases=study['cases'];convergence=reference['crack_profile_convergence']
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
axes[0].loglog([r['h_over_ell'] for r in convergence],[r['relative_nodal_L2_error'] for r in convergence],'o-',label='Damage profile')
axes[0].loglog([r['h_over_ell'] for r in convergence],[r['fracture_energy_relative_error'] for r in convergence],'s-',label='Fracture energy')
axes[0].set(xlabel='h / ell (X direction)',ylabel='Relative error',title='Analytic stationary crack');axes[0].legend()
for case in cases:
    records=read_records(case['directory'])
    axes[1].plot([r['mechanical_cycle'] for r in records],[r['mean_damage'] for r in records],label=str(case['increments_per_cycle'])+' increments/cycle')
    axes[2].plot([r['strain_yy'] for r in records],[r['mean_stress_yy_MPa'] for r in records],label=str(case['increments_per_cycle']))
axes[1].set(xlabel='Mechanical load cycle',ylabel='Volume mean d',title='History path refinement');axes[1].legend(fontsize=8)
axes[2].set(xlabel='Imposed yy strain',ylabel='Mean yy stress / MPa',title='Computed mechanical response')
for ax in axes:ax.grid(alpha=.2)
figure=ws.path('reports','top_damage_reference_validation.svg');fig.savefig(figure);plt.close(fig)
rows=''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [r['increments_per_cycle'],f"{r['final_mean_damage']:.6f}",f"{100*r['final_mean_damage_relative_difference']:.4f}%",f"{100*r['peak_stress_relative_difference']:.4f}%",f"{r['max_kkt']:.2e}",f"{r['solver_seconds']:.3f}"] )+'</tr>' for r in cases)
links=''.join('<li><a href="'+escape(os.path.relpath(Path(r['directory'])/'index.html',ws.path('reports')))+'">'+str(r['increments_per_cycle'])+' 增量/机械循环：逐步场分布与 VTU</a></li>' for r in cases)
metrics=reference['assembly_timing']
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MORQEX TOP-DAMAGE 源码后端与参考验收</title><style>
body{margin:0;background:#edf3f6;color:#1b3444;font:16px/1.7 system-ui,sans-serif}main{max-width:1220px;margin:auto;padding:28px}header,section{background:white;border-radius:16px;padding:24px;margin:18px 0}h1{color:#067c7c}.notice{background:#fff1da;border-left:5px solid #b67612;padding:14px}a{color:#087b7e}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccd9e0;text-align:left}svg{max-width:100%;height:auto}.cards{display:flex;gap:15px;flex-wrap:wrap}.card{flex:1;background:#edf8f7;padding:18px;border-radius:12px}.card strong{display:block;font-size:24px}</style>
<main><header><h1>MORQEX / TOP-DAMAGE</h1><p>可编译源码、TOML、统一 API 与三维相场参考 · 实测报告</p>
<p class="notice">当前通过的是参考方程和数值子模块门槛，以及三维机械试件的完整加载协议。不是电池充放电循环；多晶正极标定、物理裂纹开度、拓扑—动态均匀化—DFN 长期反馈尚未验收。不能据此宣称顶刊成果或寿命预测已完成。</p>
<div class="cards"><div class="card">完整张量装配相对误差<strong>'''+f"{reference['assembly_relative_frobenius_error']:.2e}"+'''</strong></div><div class="card">均匀损伤最大绝对误差<strong>'''+f"{reference['uniform_damage_max_abs_error']:.2e}"+'''</strong></div><div class="card">缓存与MFEM最大体素损伤差<strong>'''+f"{study['cached_vs_MFEM_max_voxel_damage_difference']:.2e}"+'''</strong></div></div></header>
<section><h2>打开实际三维结果</h2><p>下列页面可以选择每个接受步、Z切片、损伤、单元平均应力和疲劳变量。数据来自实际弱式求解。完整三维结果在各页面旁的 HDF5 和 ParaView VTU 文件中。</p><ul>'''+links+'''</ul></section>
<section><h2>误差与加载路径检查</h2>'''+figure.read_text().split('?>',1)[-1]+'''<table><tr><th>每循环增量</th><th>末态平均损伤</th><th>相对64增量差</th><th>峰应力向量差</th><th>最大KKT</th><th>求解秒数</th></tr>'''+rows+'''</table><p>64增量是较细离散对照，仍不是真解。解析裂纹图的X方向细化不等于一般三维传播网格已收敛。三维缺口试件本身仍需更细网格、长度参数及材料标定验证。</p></section>
<section><h2>源码与性能</h2><p>MFEM 4.10 核心库源码和构建输入位于 MORQEX/src/morqex/vendor/mfem；许可证、逐文件哈希和省略的数据目录均有记录。单核分段编译输出在外部工作区，调用方统一为 MORQEX API。未启用此疲劳求解器的GPU路径。</p>
<p>缓存保存固定几何及稀疏装配映射，每步仍重装配疲劳韧度和损伤驱动系数。缓存单次装配相对MFEM实测比值：<b>'''+f"{metrics['mfem_over_cache_ratio']:.3f}"+'''</b>；同32增量协议的求解耗时比值：<b>'''+f"{study['cached_solver_speedup_at_32_increments']:.3f}"+'''</b>。这些比值小于1时表示变慢，不是加速。后者不含一次性缓存及HDF5/VTU/HTML输出成本。</p><pre>'''+escape(json.dumps(metrics,indent=2,ensure_ascii=False))+'''</pre></section>
<section><h2>理论来源与边界</h2><p>AT2、体积/偏分裂、正变差疲劳历史及有限元装配复用均为已有方法。新增代码的目标是可验证地集成与后续拓扑反馈，不将这些基线改名作为创新。当前是冻结历史的交替变分求解，历史只在接受步提交；没有把相场宽度直接作为液体裂缝。</p>
<p><a href="https://doi.org/10.1016/j.cma.2019.112731">Carrara 等疲劳基线</a>；<a href="https://doi.org/10.1016/j.jmps.2009.04.011">Amor 等张压分裂</a>；<a href="https://doi.org/10.1016/j.jpowsour.2022.231805">Ai 等电池疲劳参考</a>；<a href="https://mfem.org">MFEM</a>。</p></section></main></html>'''
font=ws.path('data','report_font');html=embed_font(html,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
target=ws.path('reports','MORQEX_TOP_DAMAGE_reference.html');target.write_text(html)
print(target)
