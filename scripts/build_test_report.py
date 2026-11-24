"""Build one portable, offline HTML report from actual numerical/JUnit evidence."""
import base64
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WS = Path(os.environ["MORQEX_WORKSPACE"])
METHODS = {"spectral": "扩散谱", "mass-POD": "质量 POD", "boundary-resolvent": "原通量响应空间",
           "resolvent-rebuilt-with-hotspot": "观测热点后重新建基"}
DOMAINS = {"held-out-time": "新时间波形", "held-out-surface-hotspot": "新增表面热点"}


def latest(label, filename=None):
    manifest = json.loads((WS/"reports"/(label+"_latest.json")).read_text())
    if manifest["exit_code"] != 0 or manifest["stopped_because"]:
        raise ValueError(f"Latest {label} run failed; report must not silently select an older success")
    path = Path(manifest["run"])
    data = None if filename is None else json.loads((path/filename).read_text())
    if data is not None:
        for relative, digest in data.get("source_sha256", {}).items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Stale numerical source: {label}: {relative}")
    return manifest, path, data


def pct(x): return f"{100*x:.3g}%"
def num(x): return f"{x:.4g}"
def badge(status):
    labels = {"pass": "通过 · 限定范围", "fail": "未通过", "pending": "待完成"}
    return f'<span class="badge {status}">{labels[status]}</span>'


def table(headers, rows, *, ident="", classes=""):
    head = "".join(f"<th scope=\"col\">{escape(x)}</th>" for x in headers)
    body = "".join("<tr>"+"".join(f"<td>{x}</td>" for x in row)+"</tr>" for row in rows)
    identity = f' id="{ident}"' if ident else ''
    return f'<div class="table-wrap"><table{identity} class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def image(path, caption):
    payload = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img loading="lazy" src="data:image/png;base64,{payload}" alt="{escape(caption, quote=True)}"><figcaption>{escape(caption)}</figcaption></figure>'


def pytest_data(path):
    root = ET.parse(path).getroot()
    tests = []
    for case in root.iter("testcase"):
        status = "passed"
        detail = ""
        for tag, outcome in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
            node = case.find(tag)
            if node is not None:
                status, detail = outcome, node.text or node.get("message", "")
                break
        tests.append({"name": case.get("name", ""), "class": case.get("classname", ""),
            "seconds": float(case.get("time", "0")), "status": status, "detail": detail})
    if not tests: raise ValueError("JUnit contains no executed tests")
    return tests


def gates(controlled, reference, tests, adaptive):
    measured = [r for c in controlled["cases"] for r in c["comparison"] if r["status"] == "measured"]
    old = [r for r in measured if r["method"] != "resolvent-rebuilt-with-hotspot"]
    rebuilt = [r for r in measured if r["method"] == "resolvent-rebuilt-with-hotspot" and r["domain"] == "held-out-surface-hotspot"]
    reliable = all(r["bound_coverage"] == 1 for r in measured)
    in_pass = all(any(r["status"] == "measured" and r["domain"] == "held-out-time" and r["method"] == "mass-POD" and r["certified_absolute_tolerance_passed"] for r in c["comparison"]) for c in controlled["cases"])
    rebuilt_pass = all(any(r["status"] == "measured" and r["method"] == "resolvent-rebuilt-with-hotspot" and r["domain"] == "held-out-surface-hotspot" and r["certified_absolute_tolerance_passed"] for r in c["comparison"]) for c in controlled["cases"])
    tails = [c["quantile_queries"][-1] for c in controlled["cases"]]
    sphere = reference["fitted_sphere"]["mesh_sequence"][-1]["stress_L2_intersection"]
    rows = [
        ("独立数值测试", "pass" if all(t["status"] == "passed" for t in tests) else "fail", f"{sum(t['status']=='passed' for t in tests)}/{len(tests)} 项通过；测试通过不等于论文验收"),
        ("浓度后验界覆盖", "pass" if reliable else "fail", "所有本轮离散轨迹覆盖；固定 Fick / 后向 Euler；FP64 含裕量，未使用区间浮点运算"),
        ("已知通量方向：2 mol/m³ 绝对误差界", "pass" if in_pass else "fail", "逐结构存在合格 POD 阶数；按误差界选阶，不借真实应力判断是否接受"),
        ("新热点：原空间普适外推", "fail", "原空间仍明显失准；误差界给出拒绝信号，不能把域内精度推广到新载荷空间"),
        ("观测热点后重新离线建基", "pass" if rebuilt_pass else "fail", "使用新通量几何方向，未使用验证浓度/应力；是有额外输入信息的重建，不是零样本对照"),
        ("离散 P99 区间宽度", "pass" if all(t["converged"] for t in tails) else "fail", "逐查询增加节点后满足 abs+rel 宽度；仅相同低维输入下的采样误差"),
        ("P99 统计加速", "pass" if all(t["query_seconds"] < t["exact_full_P99_seconds"] for t in tails) else "fail", "计入内部节点与查询开销；当前 Python 树比小网格向量化全统计更慢"),
        ("球解析全域应力 ≤10%", "pass" if sphere <= .1 else "fail", f"默认 n24 当前实测 {pct(sphere)}；残差小不代表空间误差小"),
        ("原任意初始浓度工况", "fail", "原协议完整保留；增加已知通量方向不能解决任意初态的低秩保证"),
        ("完整 DFN / 电荷 / 电解液 / 端电压", "pending", "当前试验是受控 Fick 扩散与单向力学恢复；原二次储库耦合另有测试"),
        ("独立创新、先进基线、CT与真实材料设计", "pending", "未完成；静态凝聚、POD、响应空间、残差界和区间细分均不单独宣称原创"),
    ]
    selected = [m for c in adaptive["cases"] for result in c["results"] for m in result["methods"] if m["method"] == "adaptive-spectral-certificate"]
    rows.insert(1,("固定二次耦合：自适应全时域接受", "pass" if all(m["steps_completed"] == adaptive["config"]["steps"] and m["qualified_steps"] == adaptive["config"]["steps"] and m["bound_coverage"] == 1 for m in selected) else "fail", "三结构×两工况；全生成元H误差界、质量/能量转移及校验重启；不覆盖非线性BV/DFN/空间误差"))
    rows.insert(2,("耦合步进整体加速", "fail", "16维误差动力学减少纠正查询，但包含纠正/证书/重建后仍慢于N=80复用LU全阶；不报告近DFN加速"))
    return rows


CSS = """
:root{--ink:#17253a;--muted:#5b6d80;--line:#dce4ed;--paper:#fff;--bg:#f2f5f9;--teal:#086c64;--blue:#235ea1;--red:#a63332}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.75 system-ui,-apple-system,'Noto Sans CJK SC','Microsoft YaHei',sans-serif}a{color:var(--blue)}header{background:#112b43;color:white;padding:42px max(24px,calc((100vw - 1180px)/2)) 32px}header p{color:#ccdae6;max-width:900px}h1{font-size:34px;line-height:1.3;margin:8px 0 16px}h2{font-size:23px;margin:0 0 18px}h3{font-size:17px;margin:22px 0 8px}.eyebrow{font-size:12px;letter-spacing:.15em;color:#8cdad0}.tag{display:inline-block;border:1px solid #66859b;border-radius:20px;padding:2px 12px;font-size:12px}.layout{max-width:1240px;margin:auto;padding:24px;display:grid;grid-template-columns:170px minmax(0,1fr);gap:26px}nav{position:sticky;top:20px;align-self:start}nav a{display:block;text-decoration:none;color:var(--muted);padding:8px 10px;border-left:2px solid var(--line)}nav a:hover{border-color:var(--teal);color:var(--teal)}main{min-width:0}section{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:26px;margin-bottom:22px;scroll-margin-top:20px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 22px}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:17px}.card strong{display:block;font-size:29px;color:var(--teal);line-height:1.4}.card span{font-size:12px;color:var(--muted)}.note{border-left:3px solid #a16e13;background:#fff7e7;padding:12px 16px}.note.info{border-color:var(--blue);background:#f0f6fd}.muted,figcaption{color:var(--muted);font-size:13px}code,pre{font-family:ui-monospace,monospace;font-size:12px}code{overflow-wrap:anywhere}pre{overflow:auto;background:#edf2f7;padding:14px;border-radius:6px;white-space:pre-wrap}figure{margin:22px 0}img{display:block;width:100%;height:auto;border:1px solid #e6ebf2;border-radius:6px}figcaption{margin-top:8px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th{background:#edf3f8;white-space:nowrap;text-align:left;font-weight:650}td,th{border-bottom:1px solid var(--line);padding:10px 12px;vertical-align:top}tbody tr:hover{background:#f8fbfd}.badge{display:inline-block;border-radius:5px;padding:1px 7px;font-size:11px;white-space:nowrap}.pass,.passed{background:#e0f2ed;color:#126347}.fail,.failed,.error{background:#fbe7e5;color:#a12b27}.pending,.skipped{background:#fff1d9;color:#87570c}.toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:14px 0}.toolbar label{font-size:12px;color:var(--muted)}select,input,button{display:block;padding:8px 11px;border:1px solid #bac9d7;border-radius:6px;background:white;color:var(--ink);font:inherit}button{cursor:pointer;background:var(--teal);color:white;border-color:var(--teal)}input{min-width:240px}details summary{cursor:pointer;color:var(--blue)}.formula{padding:14px;background:#f4f7fa;border-radius:6px;font-family:ui-monospace,monospace;font-size:13px;overflow:auto}footer{max-width:1180px;margin:0 auto 30px;padding:0 24px;color:var(--muted);font-size:12px}[hidden]{display:none!important}
@media(max-width:850px){.layout{grid-template-columns:1fr;padding:12px;gap:14px}nav{position:static;display:flex;flex-wrap:wrap;gap:4px}nav a{font-size:12px;padding:4px 8px}section{padding:18px}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:27px}}
@media print{nav,.toolbar,button{display:none}.layout{display:block;padding:0}body{background:white;font-size:11px}section{break-inside:avoid;border:0;padding:12px 0}header{background:white;color:#17253a;padding:12px}header p{color:#526477}.cards{grid-template-columns:repeat(4,1fr)}img{max-height:550px;object-fit:contain}details{display:block}}
"""

JS = """
function applyMethods(){
 const method=document.getElementById('method-filter').value;
 const domain=document.getElementById('domain-filter').value;
 const seed=document.getElementById('seed-filter').value;
 let count=0;
 document.querySelectorAll('.method-row').forEach(row=>{
  row.hidden=(method!=='all'&&row.dataset.method!==method)||(domain!=='all'&&row.dataset.domain!==domain)||(seed!=='all'&&row.dataset.seed!==seed);
  if(!row.hidden)count++;
 });
 document.getElementById('method-count').textContent=String(count)+' 行';
}
function applyTests(){
 const word=document.getElementById('test-search').value.toLowerCase();
 const status=document.getElementById('test-status').value;
 let count=0;
 document.querySelectorAll('.test-row').forEach(row=>{
  row.hidden=!row.dataset.search.includes(word)||(status!=='all'&&row.dataset.status!==status);
  if(!row.hidden)count++;
 });
 document.getElementById('test-count').textContent=String(count)+' 项';
}
['method-filter','domain-filter','seed-filter'].forEach(id=>document.getElementById(id).addEventListener('change',applyMethods));
['test-search','test-status'].forEach(id=>document.getElementById(id).addEventListener('input',applyTests));
document.getElementById('download-evidence').addEventListener('click',()=>{
 const data=document.getElementById('evidence-json').textContent;
 const url=URL.createObjectURL(new Blob([data],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='MORQEX_evidence.json';a.click();URL.revokeObjectURL(url);
});
applyMethods();applyTests();
"""


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=WS/"reports/MORQEX_test_report.html")
    args = parser.parse_args(); destination = args.output.resolve()
    if WS.resolve() not in destination.parents: raise ValueError("Report output must be in external workspace")
    manifest_c, directory_c, controlled = latest("controlled_study", "controlled_study.json")
    manifest_l, directory_l, linear = latest("linear_study", "linear_study.json")
    manifest_r, directory_r, reference = latest("reference_convergence", "qualification.json")
    manifest_a, directory_a, adaptive = latest("energy_adaptivity", "energy_adaptivity.json")
    manifest_t, directory_t, _ = latest("final_tests")
    tests = pytest_data(directory_t/"test_results.xml")
    gate_rows = gates(controlled, reference, tests, adaptive)
    passed = sum(t["status"] == "passed" for t in tests)
    rows = [r for c in controlled["cases"] for r in c["comparison"] if r["status"] == "measured"]
    coverage = min(r["bound_coverage"] for r in rows)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    sections = []
    sections.append('<section id="gates"><h2>研究门槛：测试通过与方法达标分开判断</h2><p>本轮推进固定二次能量双向耦合的全生成元认证、状态转移和误差动力学；保留前一版固定 Fick 证据。浓度误差、既定浓度模式内的机械误差、风险采样误差分别处理。</p>'+
        table(["验证门槛", "状态", "证据与适用范围"], [[escape(name), badge(status), escape(detail)] for name,status,detail in gate_rows])+ '</section>')
    adaptive_rows, summary_rows, offline_rows = [],[],[]
    names = {"static-mass-projection":"静态质量投影", "static-energy-projection":"静态能量投影", "adaptive-snapshot":"图界快照富集", "adaptive-resolvent-block":"完整移位响应块", "adaptive-compressed-block":"压缩移位响应块", "adaptive-state-restart":"24维状态重启", "adaptive-dynamics-restart":"24维驱动重启", "adaptive-spectral-certificate":"16维误差动力学+快照"}
    for case in adaptive["cases"]:
        offline_rows.append([str(case["seed"]),str(case["active_voxels"]),num(case["full_mechanical_response_offline_s"]),num(case["full_concentration_hessian_setup_s"]),num(case["shared_spectral_certificate_offline_s"]),str(case["shared_spectral_certificate_array_bytes"])])
        for result in case["results"]:
            for row in result["methods"]:
                complete = row["status"] == "measured"
                qualified = complete and row["qualified_steps"] == adaptive["config"]["steps"]
                label = badge("pass" if qualified else "fail")
                accuracy = pct(row["relative_stress_trajectory_error"]) if complete else "— 前缀不作全时域比较"
                tail_accuracy = pct(row["worst_P99_relative_error"]) if complete else "—"
                ratio = num(row["online_seconds_including_enrichment"]/result["reference_online_total_s"])+"×" if complete else "—"
                reason = '' if not row["failure"] else '<details><summary>预算拒绝原因</summary>'+escape(row["failure"])+'</details>'
                cells = [str(case["seed"]),escape(result["protocol"]),names[row["method"]],label+reason,str(row["steps_completed"])+"/"+str(adaptive["config"]["steps"]),str(row["qualified_steps"]),str(row["initial_modes"])+"→"+str(row["final_modes"]),str(row["retained_error_modes"]),str(row["high_fidelity_correction_queries"]),str(row["resolvent_right_hand_sides"])+" / "+str(row["resolvent_factorizations"]),accuracy,tail_accuracy,num(1000*row["online_seconds_including_enrichment"]),ratio]
                adaptive_rows.append('<tr class="adaptive-row" data-method="'+row["method"]+'" data-seed="'+str(case["seed"])+'" data-complete="'+str(complete).lower()+'">'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
                if row["method"] == "adaptive-spectral-certificate":
                    summary_rows.append([str(case["seed"]),escape(result["protocol"]),str(row["final_modes"])+" + "+str(row["retained_error_modes"]),str(row["high_fidelity_correction_queries"]),accuracy,num(row["maximum_energy_norm_bound"]),num(row["maximum_energy_balance_J_m3"]),num(row["maximum_component_mass_error_mol"]),num(row["maximum_transfer_energy_jump_J_m3"]),num(row["maximum_transfer_mass_jump_mol"])])
    adaptive_headers = ["种子","工况","候选","容限/预算","完成步","合格步","浓度总阶","额外误差阶","全阶纠正","移位RHS/分解","全轨迹应力L2","最坏P99误差","在线总ms","在线/全阶"]
    sections.append('<section id="adaptive"><h2>新增：同能量双向耦合、自适应富集与误差动力学</h2><p>固定几何/刚度、二次化学能、指定摩尔通量与宏观应变。完整浓度生成元与既有CoupledTransport独立交叉，化学膨胀为零退化至FV；各结构末时刻应力另用完整FEM核对。新热点端口已在残差库声明，未进入初始基。</p><div class="formula">F(c,E) = ½cᵀHc + cᵀ(GE+g₀) + F_macro(E)，H = κM + H_el<br>ċ = −A(Hc+GE+g₀) + bu；质量Riesz方向 H⁻¹Mχ 显式保留<br>耦合误差单位 sqrt(J/m³)；容限 sqrt(0.1)×2 ≈ 0.63246</div><p>保留误差的带符号低维演化，避免每步仅累计残差模长；未解析余项采用Gershgorin/图下界及跨块耦合上界。嵌套扩阶继承旧误差，非嵌套重启包含当前物理场并计入转移缺陷；未读取真实轨迹来决定接受。</p><p class="note">该候选减少自身图界版本的纠正，但整体仍慢于小全阶。稠密H/谱空间仅离线、硬限制N≤512；本实验N=80。额外16个误差坐标、核坐标/界状态及离线/重建/全阶查询全部计费。不能据此宣称大规模或近DFN性能。</p>'+table(["种子","工况","浓度+误差阶","纠正查询","应力轨迹误差","最大H误差界","最大能量账误差 J/m³","最大分量质量误差 mol","最大转移能量跳变","最大转移库存跳变"],summary_rows)+image(directory_a/"adaptive_bound_and_modes.png","三独立结构×两工况：实际耦合H误差、误差上界和浓度模式数；额外16维误差状态单独列入表格。")+image(directory_a/"energy_projection_comparison.png","八候选消融：全时域误差和包含纠正/重建/证书的在线耗时比。缺线表示某结构预算失败，不是零误差；不把失败前缀和完整80步轨迹平均。")+'<h3>完整逐例对照 · 48条候选记录</h3><p>静态质量/能量投影使用同V且都保持降阶耗散；未合格的静态trial轨迹只用于分析，不由生产step返回。完整移位块、压缩块和更昂贵的重启候选保留。合格步依据误差/物理包络，事后应力不参与接受。</p><div class="table-wrap"><table id="adaptive-table"><thead><tr>'+''.join('<th>'+h+'</th>' for h in adaptive_headers)+'</tr></thead><tbody>'+''.join(adaptive_rows)+'</tbody></table></div><h3>共享离线成本另列</h3>'+table(["种子","活性N","机械响应s","全浓度H构建s","误差谱空间s","误差空间数组bytes"],offline_rows)+'<p class="muted">所有方法参考使用相同输入/步长/网格，在线统一输出浓度步进+能量+逐分量库存；完整场验证不计入任何在线时长。全阶复用LU，预计算宏观驱动，不每步重新装配。低维模型setup另存在内嵌JSON，以上共享离线成本不能隐藏。该误差界仅相同离散线性问题，FP64裕量不是区间浮点证明。</p><p>方法归属：<a href="https://epubs.siam.org/doi/10.1137/15M1055085">结构保持pH投影（2016）</a>、<a href="https://link.springer.com/article/10.1007/s10444-024-10195-8">辅助误差问题与改进pH误差界（2024）</a>均有直接先例。本次是有出处的适配与检验，未认定原创；D类贡献仍为空。</p></section>')
    picks = []
    tolerance = controlled["config"]["concentration_absolute_tolerance_mol_m3"]
    for case in controlled["cases"]:
        for method, domain in (("mass-POD", "held-out-time"), ("boundary-resolvent", "held-out-surface-hotspot"), ("resolvent-rebuilt-with-hotspot", "held-out-surface-hotspot")):
            options = [r for r in case["comparison"] if r["status"] == "measured" and r["method"] == method and r["domain"] == domain and r["certified_absolute_tolerance_passed"]]
            chosen = min(options, key=lambda r:r["modes"]) if options else None
            picks.append([str(case["seed"]), METHODS[method], DOMAINS[domain], str(chosen["modes"]) if chosen else "无可接受阶数",
                num(chosen["worst_concentration_bound"]) if chosen else "超过容限 → 拒绝/富集",
                pct(chosen["stress_trajectory_L2"]) if chosen else "不接受模型"])
    sections.append('<section id="controlled"><h2>通量驱动扩散：三结构持出验证</h2><p>同一微结构分别建基；训练与验证时间波形分开。新增热点不参与原基构造，但其通量方向在残差库中显式声明。量纲为 mol/m³ 的误差范数按完整 RVE 体积归一。</p><p class="note">“观测热点后重新建基”使用了额外已知通量空间方向。没有读取验证浓度/应力，但不能将其当作同信息预算下的零样本外推优势。</p><h3>只依据后验界选择阶数</h3><p>绝对容限 '+num(tolerance)+' mol/m³；真实应力仅用于事后对照。没有合格阶数时明确拒绝。</p>'+
        table(["种子", "候选", "留出域", "最低合格总阶", "最大浓度误差界", "事后应力轨迹误差"], picks)+
        image(directory_c/"controlled_accuracy.png", "相同非恒定模式数下的应力轨迹误差，含新增通量后的重建；三结构均值，完整逐结构数字见下表。")+
        image(directory_c/"controlled_fields.png", "实际三维数据的中间切片：相、浓度和主应力。展示原通量响应空间8个非恒定模式的末时刻，未选取最优阶数；不是CT或材料标定结果。")+
        '<div class="toolbar"><label>方法<select id="method-filter"><option value="all">全部</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in METHODS.items())+'</select></label><label>工况<select id="domain-filter"><option value="all">全部</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in DOMAINS.items())+'</select></label><label>种子<select id="seed-filter"><option value="all">全部</option>'+''.join(f'<option>{c["seed"]}</option>' for c in controlled["cases"])+'</select></label><span id="method-count" aria-live="polite"></span></div>')
    method_rows = []
    for case in controlled["cases"]:
        for row in case["comparison"]:
            if row["status"] != "measured": continue
            cells = [str(case["seed"]), METHODS[row["method"]], DOMAINS[row["domain"]], str(row["modes"]), pct(row["stress_trajectory_L2"]), pct(row["worst_P99_relative"]), num(row["worst_concentration_bound"]), num(row["bound_median_effectivity"]), f'{row["online_step_median_s"]*1e6:.1f} / {row["certificate_step_median_s"]*1e6:.1f}', badge("pass" if row["certified_absolute_tolerance_passed"] else "fail")]
            method_rows.append(f'<tr class="method-row" data-method="{row["method"]}" data-domain="{row["domain"]}" data-seed="{case["seed"]}">'+''.join(f'<td>{v}</td>' for v in cells)+'</tr>')
    headers = ["种子", "方法", "留出域", "总阶", "应力轨迹 L2", "最坏 P99 误差", "最大浓度界", "界/真误差 中位数", "步进/证书 µs", "2 mol/m³ 门槛"]
    sections[-1] += '<div class="table-wrap"><table id="method-table"><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join(method_rows)+'</tbody></table></div><p class="muted">P99误差为全时间点最坏相对误差，近零分母以预设 1000 Pa 尺度处理。轨迹误差按全时间应力平方和归一；两者不可互换。离线建基/算子/证书成本保存在原始 JSON。</p></section>'
    tail_rows = []
    for case in controlled["cases"]:
        for row in case["quantile_queries"]:
            tail_rows.append([str(case["seed"]), str(row["node_budget"]), str(row["evaluated_nodes"]), str(row["evaluated_leaves"]), pct(row["interval_width_relative"]), badge("pass" if row["converged"] else "fail"), f'{row["query_seconds"]*1e3:.2f} / {row["exact_full_P99_seconds"]*1e3:.3f}', num(row["query_seconds"]/row["exact_full_P99_seconds"])+'×'])
    sections.append('<section id="bounds"><h2>后验界与 P99：可靠性、宽度、代价</h2><div class="formula">rₙ = MΦ(aₙ−aₙ₋₁) + Δt KΦaₙ − Δt BΓF uₙ<br>δₙ,⊥ ≤ (δₙ₋₁,⊥ + ‖rₙ,⊥‖M⁻¹)/(1+Δt λtree)<br>λgap ≥ λtree := W / Σtree[Wleft Wright / gedge]</div><p>每个扩散连通分量的常数误差单独累计；树路径 Cauchy 不等式提供可计算下界，不把数值特征值当未经证明的强制性下界。残差离线 QR 平方根形式，在线不重构体场。</p><p class="note info">覆盖范围是相同网格、相同步长、固定扩散算子的全阶离散解。未覆盖时间/空间离散、非线性反应、耦合 DFN 或真实材料不确定性。FP64 加裕量的计算并非定向舍入的浮点区间证明。</p>'+
        image(directory_c/"certificate_and_tail.png", "左：实际浓度误差与后验界；右：包含内部节点的实际查询工作与 P99 区间宽度。")+
        table(["种子", "节点预算", "计算节点", "叶节点", "P99区间宽度", "宽度门槛", "树/全统计 ms", "耗时比"], tail_rows)+
        '<p class="note">树查询的区间已收窄，但当前小网格 Python 查询明显更慢。节点统计包含内部区间工作，不能只报叶节点数宣传加速。默认小网格应使用向量化精确统计；树仍是研究候选。</p></section>')
    sphere = reference["fitted_sphere"]["mesh_sequence"]
    sections.append('<section id="reference"><h2>独立参考与保留的失败工况</h2><p>球体采用独立径向浓度解析位移、应力及能量；几何和离散误差随网格变化。以下参考收敛是先前真实运行的保留证据，已核对涉及的源码 hash。</p>'+
        table(["拟合细化级", "节点", "Tet", "全域应力误差", "能量误差", "平衡残差"], [[str(r["n"]),str(r["nodes"]),str(r["elements"]),pct(r["stress_L2_intersection"]),pct(r["energy_relative_error"]),num(r["equilibrium_residual"])] for r in sphere])+
        image(directory_r/"reference_convergence.png", "独立制造解、阶梯球及拟合球的收敛结果；只截取内域不能代表全域准确。")+
        image(directory_l/"candidate_comparison.png", "原任意初始场协议重新执行；加权 POD 的失败继续显示，不用新通量试验替代。")+
        '<p>原协议仍保留高频初始浓度、新种子与同阶比较。已知通量驱动可达子空间的成功，不意味着任意浓度分布都可低秩表示。</p></section>')
    test_rows = []
    for test in tests:
        label = test["class"]+'::'+test["name"]
        detail = '' if not test["detail"] else '<details><summary>执行信息</summary><pre>'+escape(test["detail"])+'</pre></details>'
        test_rows.append(f'<tr class="test-row" data-status="{test["status"]}" data-search="{escape(label.lower(), quote=True)}"><td><span class="badge {test["status"]}">{test["status"]}</span></td><td><code>{escape(label)}</code>{detail}</td><td>{test["seconds"]:.4f}</td></tr>')
    sections.append('<section id="tests"><h2>实际测试记录</h2><p>来自本次 pytest JUnit XML，逐项展示状态与耗时。原 GEMsim 的 9 个可选后端历史跳过不计入本轮通过数；没有安装的后端仍未认证。</p><div class="toolbar"><label>搜索<input id="test-search" type="search" placeholder="输入测试名或模块" aria-label="搜索测试"></label><label>状态<select id="test-status"><option value="all">全部</option><option value="passed">passed</option><option value="failed">failed</option><option value="skipped">skipped</option><option value="error">error</option></select></label><span id="test-count" aria-live="polite"></span></div><div class="table-wrap"><table id="test-table"><thead><tr><th>结果</th><th>测试</th><th>秒</th></tr></thead><tbody>'+''.join(test_rows)+'</tbody></table></div></section>')
    commands = ["python scripts/run_local.py --label final_tests -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python -m pytest /home/zephyrain/lw/MORQEX/tests -q --junitxml=test_results.xml --basetemp=/home/zephyrain/lw/morph_workspace/tmp/final_certified_tests",
        "python scripts/run_local.py --label controlled_study -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/study_controlled_transport.py --config /home/zephyrain/lw/MORQEX/configs/controlled_transport.toml",
        "python scripts/run_local.py --label test_html -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/build_test_report.py"]
    commands.insert(1,"python scripts/run_local.py --label energy_adaptivity -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/study_energy_adaptivity.py --config /home/zephyrain/lw/MORQEX/configs/energy_adaptivity.toml")
    provenance = []
    for label, manifest, directory, filename in [("energy_adaptivity",manifest_a,directory_a,"energy_adaptivity.json"),("controlled_study",manifest_c,directory_c,"controlled_study.json"),("linear_study",manifest_l,directory_l,"linear_study.json"),("reference_convergence",manifest_r,directory_r,"qualification.json"),("final_tests",manifest_t,directory_t,"test_results.xml")]:
        artifact = directory/filename
        provenance.append({"label":label,"path":str(artifact),"sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),"manifest":manifest})
    sections.append('<section id="reproduce"><h2>复现与来源</h2><p>全部实验默认单 CPU 线程、nice +15、GPU 隐藏；组 RSS 上限 1.5 GiB、虚拟空间 3 GiB、每任务 180 s。源码内不放数据、算子、图像或构建产物。当前研究使用合成参数，不能导出定量材料设计结论。</p>'+
        table(["运行", "UTC", "Wall s", "采样组RSS MiB", "数据 SHA256"], [[p["label"],p["manifest"]["started_utc"],num(p["manifest"]["wall_seconds"]),num(p["manifest"]["peak_group_rss_bytes_sampled"]/2**20),'<code>'+p["sha256"]+'</code>'] for p in provenance])+
        '<h3>运行命令</h3><pre>'+escape('\n\n'.join(commands))+'</pre><details><summary>本轮完整配置</summary><pre>'+escape(json.dumps({"controlled":controlled["config"],"geometry":controlled["geometry_config"],"energy_adaptivity":adaptive["config"]}, ensure_ascii=False, indent=2))+'</pre></details>'+
        '<h3>方法先例与后续门槛</h3><p>残差认证参照 <a href="https://numdam.org/item/M2AN_2005__39_1_157_0/">Grepl–Patera (2005)</a>；响应空间属于 <a href="https://epubs.siam.org/doi/10.1137/1.9780898718713.ch11">Krylov / moment-matching</a> 已有思想。本实现的树 Poincaré 推导与区间细分需和先进目标降阶方法继续比较，不称原创。下一步仍包括完整 DFN、独立双向参考、有效尾部加速、SVE/CT、先进基线与真实参数设计。</p><button id="download-evidence" type="button">下载内嵌原始证据 JSON</button><p class="muted">报告不依赖 CDN、服务器或网络连接；正文与表格在关闭 JavaScript 时仍可阅读。JS仅用于筛选和导出。</p></section>')
    payload = {"generated_utc":now,"source_commit":commit,"source_dirty":dirty,"provenance":provenance,
        "tests":tests,"research_gates":[{"name":n,"status":s,"detail":d} for n,s,d in gate_rows],
        "controlled":controlled,"linear":linear,"reference":reference,"energy_adaptivity":adaptive}
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    failed_gates = sum(s == "fail" for _,s,_ in gate_rows)
    cards = f'<div class="cards"><div class="card"><strong>{passed}/{len(tests)}</strong><span>实际测试通过</span></div><div class="card"><strong>{len(controlled["cases"])}</strong><span>独立 GEMsim 结构</span></div><div class="card"><strong>{pct(coverage)}</strong><span>本轮离散误差界覆盖</span></div><div class="card"><strong>{failed_gates}</strong><span>研究门槛仍未通过</span></div></div>'
    links = [('gates','验收门槛'),('adaptive','新增耦合适应'),('controlled','浓度与应力'),('bounds','误差与 P99'),('reference','参考与负结果'),('tests','测试明细'),('reproduce','复现与来源')]
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'; script-src \'unsafe-inline\'; connect-src \'none\'; base-uri \'none\'; object-src \'none\'"><title>MORQEX · 测试与研究验证报告</title><style>'+CSS+'</style></head><body><header><div class="eyebrow">MORQEX / RESEARCH VERIFICATION</div><h1>测试、误差与计算证据</h1><span class="tag">研究原型 · 尚未达到论文验收</span><p>三维微结构 · 双向二次耦合 · 状态保持富集 · 辅助误差动力学 · 应力尾部<br>本次报告展示真实结果，失败项与待完成项保留。</p><div class="muted" style="color:#ccdae6">'+escape(now)+' · commit '+escape(commit[:12])+(' · 含未提交工作树' if dirty else ' · 工作树干净')+'</div></header><div class="layout"><nav aria-label="报告导航">'+''.join(f'<a href="#{ident}">{label}</a>' for ident,label in links)+'</nav><main>'+cards+''.join(sections)+'</main></div><footer>本地可复现实验报告。算法可靠性、模型适用性、材料真实性与发表标准分别验证；数值测试数量不替代方法学贡献。</footer><script type="application/json" id="evidence-json">'+embedded+'</script><script id="report-interactions">'+JS+'</script></body></html>'
    destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(document, encoding="utf-8")
    record = {"html":str(destination),"bytes":destination.stat().st_size,"sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_commit":commit,"source_dirty":dirty,"tests":len(tests),"passed":passed,"research_failed_gates":failed_gates,
        "coupled_candidate_rows":len(adaptive_rows),"self_contained":True,"renderer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (WS/"reports/test_report_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
