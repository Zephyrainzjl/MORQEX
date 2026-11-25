"""Offline scientific DFN/thermal report from hash-checked external evidence."""
import base64
from datetime import datetime,timezone
import hashlib
from html import escape
import io
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
WS=Path(os.environ['MORQEX_WORKSPACE'])
REQUIRED=['comsol','comsol_fine','ai05','ai1','ai2','raw1','convex1','oneway1','coupled05','coupled1','coupled2','coupled_coarse','coupled_fine','coupled_fixed','coupled_full_expansion','coupled_cycles','coupled_crosscheck','coupled_crosscheck_idaklu','casadi_loose_ic']


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{escape(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def figure(title,draw):
    fig,ax=plt.subplots(figsize=(10,4.8));draw(ax);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title(title);fig.tight_layout()
    # Small PNG previews stay in the resource-limited external run directory.
    fig.savefig(Path.cwd()/f'figure_{figure.count:02d}.png',dpi=130)
    figure.count+=1
    stream=io.BytesIO();fig.savefig(stream,format='svg');plt.close(fig)
    return '<figure><img alt="'+escape(title)+'" src="data:image/svg+xml;base64,'+base64.b64encode(stream.getvalue()).decode()+'"></figure>'


figure.count=1


def load():
    cases={}
    for name in REQUIRED:
        path=WS/'results/dfn'/name/'result.json';r=json.loads(path.read_text())
        for relative,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:raise ValueError(f'Stale source: {name}: {relative}')
        for relative,digest in r['solver']['source_sha256'].items():
            if hashlib.sha256((ROOT/'src/pybamm'/relative).read_bytes()).hexdigest()!=digest:raise ValueError(f'Stale backend: {name}')
        if not r['protocol']['completed']:raise ValueError(f'Incomplete protocol: {name}')
        r['artifact_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();cases[name]=r
    manifest=json.loads((WS/'reports/dfn_reference_data.json').read_text())
    for entry in manifest['files']:
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Reference data hash mismatch')
    return cases,manifest


def common_error(a,b,field):
    da,db=a['data'],b['data'];lo=max(da['time_s'][0],db['time_s'][0]);hi=min(da['time_s'][-1],db['time_s'][-1]);t=np.linspace(lo,hi,1001)
    error=np.interp(t,da['time_s'],da[field])-np.interp(t,db['time_s'],db[field])
    return {'rmse':float(np.mean(error**2)**.5),'maximum':float(max(abs(error))),'common_end_s':hi,'end_time_difference_s':da['time_s'][-1]-db['time_s'][-1]}


def main():
    cases,reference_manifest=load();parts=[];comparisons={};experimental=[];localization={}
    parts.append('<h1>MORQEX · 完整 DFN / 热—力耦合验证</h1><p>0.0.6 · '+datetime.now(timezone.utc).isoformat()+' · 实际本地计算，全部数据内嵌，可离线打开。</p>')
    parts.append('<p class="notice">完成范围：全DFN放电、三次可逆CC-CV循环、集总温度反馈、平均浓度SVE与独立球扩散应力。原始参数参照与凸化/Ω缩放验证轨分开展示。尚未达到顶刊验收：任意微结构非均匀浓度闭合、独立三维热参照、全局热力学证书、DFN误差认证和原创性证明仍缺。</p>')
    parts.append('<h2>1 · 独立 COMSOL 参照</h2>')
    parts.append(table(['网格 x/r','电压 RMSE / mV','最大误差 / mV','结束 / s'],[[f"{cases[n]['arguments']['nx']}/{cases[n]['arguments']['nr']}",f"{cases[n]['metrics']['comsol_voltage_rmse_V']*1000:.4g}",f"{cases[n]['metrics']['comsol_voltage_max_error_V']*1000:.4g}",cases[n]['metrics']['end_time_s']] for n in ['comsol','comsol_fine']]))
    def comsol(ax):
        for n in ['comsol','comsol_fine']:
            d=cases[n]['data'];ax.plot(np.array(d['time_s'])/3600,d['voltage_V'],label=n)
        ax.plot(np.array(d['reference_time_s'])[::8]/3600,np.array(d['reference_voltage_V'])[::8],'o',ms=3,label='independent COMSOL');ax.set(xlabel='Time / h',ylabel='Voltage / V')
    parts.append(figure('DFN / COMSOL: same physical time, no alignment',comsol))
    if 'comsol_field_errors' in cases['comsol_fine']['metrics']:
        parts.append(table(['独立COMSOL内部场','相对 L2','绝对最大误差'],[[k,v['relative_L2'],v['absolute_max']] for k,v in cases['comsol_fine']['metrics']['comsol_field_errors'].items()]))
    parts.append('<h2>2 · 不改参数的 Ai2020 / 公开实验</h2><p>电压与温升按原始时间比较，未拟合时间/电压偏置。热数据为相对初温的 ΔT。重叠窗口误差与覆盖率同时报告；常规DFN并非原论文应力算法的完整复现。</p>')
    for field,suffix,ylabel in [('voltage_V','U','Voltage / V'),('temperature_K','T','Temperature rise / K')]:
        def experimental_plot(ax,field=field,suffix=suffix):
            for name,rate in [('ai05','0.5'),('ai1','1'),('ai2','2')]:
                d=cases[name]['data'];exp=np.loadtxt(WS/f'cache/pybamm/v1.0.1/{rate}C_discharge_{suffix}.txt');pred=np.array(d[field]);pred=pred-(pred[0] if suffix=='T' else 0)
                mask=(exp[:,0]>=d['time_s'][0])&(exp[:,0]<=d['time_s'][-1]);difference=np.interp(exp[mask,0],d['time_s'],pred)-exp[mask,1]
                experimental.append({'case':name,'observable':suffix,'rmse':float(np.mean(difference**2)**.5),'maximum':float(max(abs(difference))),'samples':int(mask.sum()),'sample_coverage':float(mask.mean()),'experiment_end_s':float(exp[-1,0]),'simulation_end_s':d['time_s'][-1]})
                line,=ax.plot(np.array(d['time_s'])/3600,pred,label=rate+'C DFN');ax.plot(exp[::max(1,len(exp)//100),0]/3600,exp[::max(1,len(exp)//100),1],'.',color=line.get_color(),alpha=.5,label=rate+'C experiment')
            ax.set(xlabel='Time / h',ylabel=ylabel)
        parts.append(figure('Published Enertech data: '+ylabel,experimental_plot))
    parts.append(table(['工况','量','RMSE (V/K)','最大误差 (V/K)','样本覆盖率','实验 / 仿真终点 s'],[[r['case'],r['observable'],f"{r['rmse']:.5g}",f"{r['maximum']:.5g}",f"{100*r['sample_coverage']:.2f}%",f"{r['experiment_end_s']:.2f} / {r['simulation_end_s']:.2f}"] for r in experimental]))
    errors=[r['rmse']*1000 for r in experimental if r['observable']=='U']
    parts.append(f'<p class="notice">实验准确性尚未通过：原始参数常规DFN的电压RMSE为 {min(errors):.1f}–{max(errors):.1f} mV。温升文件还包含放电后的冷却段，本轮仿真只覆盖放电，未复现完整冷却轨迹。需要进一步核对原论文模型选项、初始状态与实验条件，不能据当前结果宣称实验级验证完成。</p>')
    parts.append('<h2>3 · 双向热—力 DFN：0.5 / 1 / 2C 完整放电</h2><p>下列为显式修改的 OKane2022 验证轨：严格正曲率OCP，默认 Ω×0.1，小应变数值敏感性工况，非标定材料。全区间负极OCP最终最大改动约21.43mV，不能称为未改参数复现。</p>')
    for field,ylabel in [('voltage_V','Voltage / V'),('temperature_K','Temperature / K'),('negative_hoop_stress_Pa','Negative surface hoop stress / MPa'),('positive_hoop_stress_Pa','Positive surface hoop stress / MPa')]:
        def rate_plot(ax,field=field,ylabel=ylabel):
            scale=1e-6 if 'stress' in field else 1
            for n in ['coupled05','coupled1','coupled2']:
                d=cases[n]['data'];ax.plot(np.array(d['time_s'])/3600,np.array(d[field])*scale,label=f"{cases[n]['arguments']['rate']}C")
            ax.set(xlabel='Time / h',ylabel=ylabel)
        parts.append(figure('Coupled full discharge: '+ylabel,rate_plot))
    parts.append('<h2>4 · 三次 CC-CV 完整循环</h2><p>每循环：放电至2.5V → 静置600s → 1C充至4.2V → 4.2V恒压至C/20 → 静置600s。无退化状态，不把可逆循环重复解释为寿命/衰减预测。</p>')
    for field,ylabel in [('voltage_V','Voltage / V'),('current_A','Current / A'),('temperature_K','Temperature / K'),('thickness_m','Sandwich thickness change / micrometre')]:
        def cycling(ax,field=field,ylabel=ylabel):
            d=cases['coupled_cycles']['data'];ax.plot(np.array(d['time_s'])/3600,np.array(d[field])*(1e6 if field=='thickness_m' else 1),label='3 reversible cycles');ax.set(xlabel='Time / h',ylabel=ylabel)
        parts.append(figure('Complete cycling: '+ylabel,cycling))
    cycle_rows=[]
    for i,cycle in enumerate(cases['coupled_cycles']['protocol']['cycles']):
        for step in cycle:cycle_rows.append([i+1,step['step']+1,f"{step['duration_s']:.4f}",f"{step['voltage_end_V']:.6f}",f"{step['current_end_A']:.6f}",step['termination']])
    parts.append(table(['循环','步骤','持续 s','末电压 V','末电流 A','终止条件'],cycle_rows))
    parts.append('<h2>5 · 网格、反馈、约束及第二求解器</h2>')
    for a,b in [('raw1','convex1'),('convex1','oneway1'),('oneway1','coupled1'),('coupled_coarse','coupled1'),('coupled1','coupled_fine'),('coupled_crosscheck','coupled_crosscheck_idaklu')]:
        comparisons[a+'/'+b]={field:common_error(cases[a],cases[b],field) for field in ['voltage_V','temperature_K']}
    parts.append(table(['比较（共同物理时间）','电压 RMSE / mV','温度 RMSE / K','截止时差 / s'],[[key,f"{v['voltage_V']['rmse']*1000:.6g}",f"{v['temperature_K']['rmse']:.6g}",f"{v['voltage_V']['end_time_difference_s']:.5g}"] for key,v in comparisons.items()]))
    parts.append('<p>CasADi完整放电曾达到CPU时间上限；第二求解器交叉限制为120s同一工况。不能声称完整循环已通过两套求解器。网格误差是同一离散族相互比较，非严格连续体误差界。</p>')
    parts.append('<p class="notice">保留反例 casadi_loose_ic：默认代数初值步长停止阈值1e-4导致起点积分电荷残差约5.86e-4，未过5e-5门槛。仅收紧root_tol或关闭随机扰动不足；将代数步长阈值改为1e-12后通过。该反例虽完成120s协议，仍明确判为数值检查失败，不计作通过工况。</p>')
    def constraint(ax):
        for n in ['coupled1','coupled_fixed']:
            d=cases[n]['data'];ax.plot(np.array(d['time_s'])/3600,np.array(d['stack_stress_Pa'])*1e-6,label=n)
        ax.set(xlabel='Time / h',ylabel='Common axial stress / MPa')
    parts.append(figure('Macroscopic mechanical constraint',constraint))
    from morqex.energy import EnergyResponse
    for domain in ['negative','positive']:
        path=WS/f'results/dfn/coupled1/{domain}_mean_SVE_snapshots.npz'
        with np.load(path,allow_pickle=False) as a:
            stored={k:a[k].copy() for k in a.files}
        response=EnergyResponse.load(WS/f'operators/coupled1_{domain}')
        info=cases['coupled1']['parameters']['mechanics'][domain]
        stress=response.stress(stored['q'])*info['dfn_active_fraction']/info['actual_active_fraction']
        np.testing.assert_allclose(stress,stored['stress_Mandel_Pa'],rtol=1e-12,atol=1e-6)
        localization[domain]={'snapshot_npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'time_s':stored['time_s'].tolist(),
            'stress_Mandel_Pa':stress.tolist(),'volumes_m3':stored['volumes_m3'].tolist(),
            'scope':'uniform intrinsic mean concentration and DeltaT SVE only; isolated-sphere radial stress is separate'}
    def distribution(ax):
        for domain,r in localization.items():
            s=np.array(r['stress_Mandel_Pa'])[-1,:,0]/1e6;v=np.array(r['volumes_m3']);ax.hist(s,bins=35,weights=v/v.sum(),alpha=.5,label=domain+' mean-concentration SVE')
        ax.set(xlabel='Local axial stress / MPa',ylabel='Conditional solid-volume fraction per bin')
    parts.append(figure('GEM SVE stress distribution at discharge endpoint',distribution))
    parts.append('<p>局部应力由外置SVE算子重新计算并与保存快照逐项核对。这里是5³固定种子SVE的平均浓度/温度响应，按固体体积归一化；未声称具有RVE代表性或复原真实球内梯度。局部张量/体积数据同时内嵌在证据下载中。</p>')
    parts.append('<h2>6 · 守恒、温度账本与资源</h2><p>热残差是集总温度ODE的积分一致性，尚非完整化学/电/弹性全能量证书。电荷指标为正/负极积分反应电流与外部电流的相对残差，包含并联电极数量。</p>')
    rows=[]
    for n,r in cases.items():
        m=r['metrics'];rows.append([n,'通过限定数值检查' if all(r['numerical_checks'].values()) else '未通过数值检查',f"{m['relative_lithium_drift']:.3e}",f"{max(m['relative_electrode_charge_balance_residual'].values()):.3e}",f"{m.get('thermal_ODE_relative_ledger_residual',float('nan')):.3e}",f"{m['temperature_max_K']:.3f}",f"{m['build_and_solve_s']:.3f}"])
    parts.append(table(['工况','限定数值检查','锂库存相对漂移','积分电荷残差','温度账本相对残差','最高温度 K','装配+求解 s'],rows))
    failures=[];resources=[]
    for path in sorted((WS/'runs').glob('*_dfn_*/manifest.json')):
        m=json.loads(path.read_text());row={'run':str(path.parent),'exit_code':m['exit_code'],'reason':m['stopped_because'],'wall_s':m['wall_seconds'],'sampled_RSS_MiB':m['peak_group_rss_bytes_sampled']/2**20}
        if m['exit_code']!=0 or m['stopped_because']:failures.append(row)
    for n in REQUIRED:
        p=WS/f'reports/dfn_{n}_latest.json'
        if p.exists():
            m=json.loads(p.read_text());resources.append([n,f"{m['wall_seconds']:.3f}",f"{m['peak_group_rss_bytes_sampled']/2**20:.2f}",m['resources']['cpu_threads'],m['gpu_visible']])
    parts.append(table(['工况','进程耗时 s','采样峰值RSS MiB','CPU线程','GPU可见'],resources))
    parts.append('<details><summary>保留的失败/预算拒绝记录</summary>'+table(['外部运行目录','退出码','原因','秒','采样RSS MiB'],[[r['run'],r['exit_code'],r['reason'],f"{r['wall_s']:.3f}",f"{r['sampled_RSS_MiB']:.2f}"] for r in failures])+'</details>')
    parts.append('<h2>7 · 测试、来源与适用域</h2>')
    junit=ET.parse(WS/'reports/dfn_full_junit.xml').getroot();tests=list(junit.iter('testcase'));bad=[t for t in tests if t.find('failure') is not None or t.find('error') is not None]
    if bad:raise ValueError('Failed tests cannot produce a passing report')
    parts.append(f'<p>自身测试 {len(tests)} 项；逐项结果：</p><details><summary>展开测试名称</summary>'+table(['测试'],[[t.attrib.get('name','')] for t in tests])+'</details>')
    install=json.loads((WS/'reports/dfn_installed_latest.json').read_text())
    if install['exit_code']!=0 or install['stopped_because']:raise ValueError('Installed-wheel probe failed')
    installed_proof=json.loads((Path(install['run'])/'stdout.txt').read_text())
    if installed_proof['upstream_distribution_installed']:raise ValueError('Independent bundled installation not established')
    parts.append('<p>独立安装验证通过：新venv仅安装我们的0.0.6 wheel及外部二进制依赖，没有安装上游PyBaMM发行包，清除源码sys.path后完成60s热—力DFN；参数CSV、BSD许可证及补丁随wheel提供。</p>')
    parts.append('<p>平均浓度SVE局部Mandel应力场与算子存于外部NPZ，未假装恢复真实不规则颗粒的径向浓度。所有定量曲线和结果JSON以SHA绑定数值源码，HTML生成时逐一核对。整个页面没有CDN、遥测或外部脚本。</p>')
    for limitation in cases['coupled_cycles']['limitations']:parts.append('<p>'+escape(limitation)+'</p>')
    parts.append('<p>方法/许可/理论：<a href="https://doi.org/10.5334/jors.309">PyBaMM</a>；<a href="https://doi.org/10.1149/2.0122001JES">Ai2020</a>；<a href="https://arxiv.org/abs/2104.01590">Richardson–Korotkin2021</a>；<a href="https://pubs.rsc.org/en/content/articlehtml/2022/cp/d2cp00417h">O’Kane2022</a>；<a href="https://github.com/pybamm-team/pybamm-data/releases/tag/v1.0.1">公开数据 v1.0.1</a>。</p>')
    evidence={'cases':cases,'reference_data':reference_manifest,'experimental_comparisons':experimental,'numerical_comparisons':comparisons,'localization':localization,'installed_wheel_proof':installed_proof,'failures':failures,'test_count':len(tests)}
    encoded=base64.b64encode(json.dumps(evidence,allow_nan=False).encode()).decode()
    parts.append('<p><a download="MORQEX_DFN_evidence.json" href="data:application/json;base64,'+encoded+'">下载内嵌原始曲线与证据JSON</a></p>')
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MORQEX DFN 热—力验证</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1160px;margin:32px auto;padding:0 18px;color:#152632;background:#fafcfd}h1,h2{color:#123e53}h2{margin-top:48px}.notice{padding:18px;background:#fff1d9;border-left:5px solid #b67720}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #cddce4;padding:9px;text-align:left}th{background:#e8f0f5}figure{margin:24px 0}img{width:100%;height:auto}details{margin:20px 0}a{color:#09678a}</style><body>'+''.join(parts)+'</body></html>'
    out=WS/'reports/MORQEX_DFN_report.html';out.write_text(text)
    (WS/'reports/MORQEX_test_report.html').write_text(text)
    (WS/'reports/dfn_summary.json').write_text(json.dumps({k:v for k,v in evidence.items() if k!='cases'}|{'case_metrics':{n:r['metrics'] for n,r in cases.items()}},indent=2))
    (WS/'reports/dfn_html_manifest.json').write_text(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'cases':REQUIRED,'source_sha256':cases['coupled_cycles']['source_sha256'],'test_count':len(tests),'bytes':out.stat().st_size},indent=2))
    print(json.dumps({'path':str(out),'cases':len(cases),'tests':len(tests),'bytes':out.stat().st_size}))


if __name__=='__main__':main()
