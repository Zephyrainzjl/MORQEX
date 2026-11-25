"""Reproducible offline microstructure/DFN report, with actual field localization."""
import base64
import hashlib
from html import escape
import io
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from morqex.electrochemistry.microstructure_cache import LocalizationMap
from morqex.mechanics.tensors import from_mandel
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters

ROOT=Path(__file__).resolve().parents[1];WS=Path(os.environ['MORQEX_WORKSPACE'])
sys.path.insert(0,str(ROOT/'scripts'))
from run_microstructure_suite import CASES

def verify(result):
    for name,value in result['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,('stale source',name)
    for name,value in result['backend']['source_sha256'].items():
        assert hashlib.sha256((ROOT/'src/pybamm'/name).read_bytes()).hexdigest()==value,('stale backend',name)
    assert all(result['checks'].values())

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def figure(title,draw):
    fig,ax=plt.subplots(figsize=(9,3.8));draw(ax);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title(title);fig.tight_layout()
    stream=io.BytesIO();fig.savefig(stream,format='svg');plt.close(fig)
    return '<figure><img alt="'+escape(title)+'" src="data:image/svg+xml;base64,'+base64.b64encode(stream.getvalue()).decode()+'"></figure>'

def history(result,domain,t):
    m=result['microstructure'][domain];source=result['curves']['time_s'];c=np.asarray(m['concentration_regions'])
    concentrations=np.array([[np.interp(t,source,c[x,:,k]) for k in range(c.shape[2])] for x in range(c.shape[0])]).transpose(0,2,1)
    strain=np.array([np.interp(t,source,row) for row in m['axial_strain']]);theta=np.interp(t,source,result['curves']['temperature_K'])-298.15
    return concentrations,strain,theta

def quantile(values,weights,p=.99):
    order=np.argsort(values,axis=-1);v=np.take_along_axis(values,order,axis=-1);w=weights[order]
    at=np.argmax(np.cumsum(w,axis=-1)>=p*w.sum(axis=-1,keepdims=True),axis=-1)
    return np.take_along_axis(v,at[...,None],axis=-1)[...,0]

def field_comparison(candidate,reference):
    t=np.linspace(0,min(candidate['curves']['time_s'][-1],reference['curves']['time_s'][-1]),81)
    errors={}
    for d in ['negative','positive']:
        local=LocalizationMap(candidate['microstructure'][d]['operator_path']);full=LocalizationMap(reference['microstructure'][d]['operator_path'])
        assert local.online.descriptor['geometry_fingerprint']==full.online.descriptor['geometry_fingerprint']
        a=local.recover(*history(candidate,d,t));b=full.recover(*history(reference,d,t))
        w=full.arrays['volumes']/full.arrays['volumes'].sum();wc=full.arrays['weights_full']/full.arrays['weights_full'].sum()
        stress_error=float(np.sqrt(np.einsum('...ek,e->...', (a['stress']-b['stress'])**2,w).sum()/max(1e-30,np.einsum('...ek,e->...', b['stress']**2,w).sum())))
        # The full concentration is nonzero: also report cmax normalization.
        cmax=float(PARAMETERS[f'Maximum concentration in {d} electrode [mol.m-3]'])
        concentration=float(np.sqrt(np.einsum('...e,e->...',(a['concentration']-b['concentration'])**2,wc).sum()/(len(t)*a['concentration'].shape[0]))/cmax)
        tension_a=np.maximum(0,np.linalg.eigvalsh(from_mandel(a['stress']))[...,-1]);tension_b=np.maximum(0,np.linalg.eigvalsh(from_mandel(b['stress']))[...,-1])
        pa=quantile(tension_a,w);pb=quantile(tension_b,w)
        errors[d]={'stress_relative_L2':stress_error,'concentration_RMS_over_cmax':concentration,
            'P99_tension_RMS_Pa':float(np.sqrt(np.mean((pa-pb)**2))),
            'P99_tension_normalized_RMS':float(np.sqrt(np.mean((pa-pb)**2))/max(1e3,np.sqrt(np.mean(pb**2)))),
            'P99_reference_RMS_Pa':float(np.sqrt(np.mean(pb**2)))}
    ca,cb=candidate['curves'],reference['curves'];dv=np.interp(t,ca['time_s'],ca['voltage_V'])-np.interp(t,cb['time_s'],cb['voltage_V'])
    return {'common_end_s':float(t[-1]),'voltage_RMSE_mV':float(1000*np.sqrt(np.mean(dv**2))),
        'cutoff_time_difference_s':ca['time_s'][-1]-cb['time_s'][-1],'fields':errors}

def curve_difference(a,b):
    t=np.linspace(0,min(a['curves']['time_s'][-1],b['curves']['time_s'][-1]),1001);result={}
    for field in ['voltage_V','temperature_K','thickness_m','stack_stress_Pa']:
        diff=np.interp(t,a['curves']['time_s'],a['curves'][field])-np.interp(t,b['curves']['time_s'],b['curves'][field])
        result[field]={'RMS':float(np.sqrt(np.mean(diff**2))),'maximum':float(np.max(abs(diff)))}
    result['common_end_s']=float(t[-1]);return result

def viewer_data(cases):
    output={}
    for name in ['full4_1c','rom16_1c','large16','familyB16','rom4_cycles']:
        r=cases[name];ids=np.unique(np.linspace(0,len(r['curves']['time_s'])-1,70).astype(int));data={
            'time':np.asarray(r['curves']['time_s'])[ids].tolist(),'voltage':np.asarray(r['curves']['voltage_V'])[ids].tolist(),
            'temperature':np.asarray(r['curves']['temperature_K'])[ids].tolist(),'micro':{}}
        for d in ['negative','positive']:
            local=LocalizationMap(r['microstructure'][d]['operator_path']);a=local.arrays;P=a['P'];N=P.shape[0]
            # Display volume-averaged voxel stress, derived from the actual Tet map.
            S=np.zeros((N,6,P.shape[1]+2));selected=a['element_active_cell']>=0
            for k in range(6):
                for j in range(S.shape[2]):
                    S[:,k,j]=np.bincount(a['element_active_cell'][selected],weights=a['volumes'][selected]*a['stress'][selected,k,j],minlength=N)/a['active_volumes']
            m=r['microstructure'][d]
            data['micro'][d]={'shape':list(a['labels'].shape),'ids':a['voxel_ids'].tolist(),'P':P.tolist(),'S':S.tolist(),
                'c':np.asarray(m['concentration_regions'])[:,ids,:].tolist(),'strain':np.asarray(m['axial_strain'])[:,ids].tolist(),
                'j':np.asarray(m['reaction_currents_regions_A_m2'])[:,ids,:].tolist(),'area':local.online.reaction_area.tolist(),
                'jv':np.asarray(m['volumetric_reaction_current_A_m3'])[:,ids].tolist(),'weights':local.online.weights.tolist(),
                'H':local.online.H.tolist(),'reference':local.online.reference_concentration,
                'cmax':float(PARAMETERS[f'Maximum concentration in {d} electrode [mol.m-3]']),'omega':local.online.descriptor['material']['molar_volume_m3_mol']}
        output[name]=data
    return output

def main():
    global PARAMETERS
    PARAMETERS,_=microstructure_parameters()
    cases={};artifacts={}
    for name,_ in CASES:
        path=WS/'results/microstructure_dfn'/name/'result.json';result=json.loads(path.read_text());verify(result);cases[name]=result
        artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    path=WS/'results/microstructure_runtime/result.json';live=json.loads(path.read_text());verify(live);artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    testmeta=json.loads((WS/'reports/micro_all_tests_latest.json').read_text());assert testmeta['exit_code']==0
    junit=ET.parse(Path(testmeta['run'])/'test_results.xml');suites=list(junit.getroot().iter('testsuite'))
    tests=sum(int(s.get('tests',0)) for s in suites);assert tests>=149 and not any(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    comparisons={name:field_comparison(cases[name],cases['full4_1c']) for name in ['mean4_1c','rom8_1c','rom16_1c','rom24_1c','mechanics16_1c']}
    cycle_comparison=field_comparison(cases['rom3_cycles'],cases['full3_cycles'])
    curve_comparisons={a+'/'+b:curve_difference(cases[a],cases[b]) for a,b in [('rom16_1c','oneway16_1c'),('rom16_1c','rom16_nx6'),('rom16_nx6','rom16_nx12'),('rom16_1c','familyB16')]}
    parts=['<h2>双向映射已经闭合，但不是单值SOC查表</h2><p>DFN的局部电势/电解液浓度/温度 → 有历史的区域浓度演化 → 同源FEM能量/力学化学势 → 界面反应与DFN守恒方程。原径向颗粒状态已替换，锂不重复记账。RP=I；PR是投影。仅给平均SOC无法唯一恢复真实微结构。</p>',
        '<div class="flow"><b>DFN · Δφ, cₑ, T, E</b><span>⇄</span><b>微结构状态 a(t) · 质量与功率对偶端口</b><span>→</span><b>应力 / 浓度 / 局部恢复</b></div>',
        '<h2>实际场恢复 · 可交互</h2><p>滑块重放真实耦合求解轨迹，在线执行P与S映射。图为ZYX体素切片；应力为各体素内Tet体积平均，von Mises基于该平均张量。孔隙灰色。它不是浏览器内新负载求解器；实时变流演算由Python运行核完成。</p>',
        '<div class="controls"><label>工况 <select id="case"></select></label><label>电极 <select id="domain"><option value="negative">负极</option><option value="positive">正极</option></select></label><label>厚度积分点 <select id="macro"><option>0</option><option>1</option><option>2</option></select></label><label>场 <select id="field"><option value="soc">局部浓度 / cmax</option><option value="xx">轴向应力 / MPa</option><option value="vm">von Mises / MPa</option><option value="mu">局部力学化学势 / J mol⁻¹</option></select></label><label>时间 <input id="time" type="range" min="0" value="0"></label><label>深度Z <input id="depth" type="range" min="0" value="0"></label></div><canvas id="map" width="720" height="390"></canvas><p id="readout"></p>',
        '<h2>完整放电与热—力响应</h2>']
    for field,ylabel,scale in [('voltage_V','Voltage / V',1),('temperature_K','Temperature / K',1),('thickness_m','Sandwich thickness change / um',1e6),('stack_stress_Pa','Axial stack stress / MPa',1e-6)]:
        def draw(ax,field=field,ylabel=ylabel,scale=scale):
            names=['rom16_halfC','full4_1c','rom16_1c','rom16_2C'] if field!='stack_stress_Pa' else ['rom16_1c','rom16_fixed']
            for n in names:
                d=cases[n]['curves'];ax.plot(np.asarray(d['time_s'])/3600,np.asarray(d[field])*scale,label=n)
            ax.set(xlabel='Time / h',ylabel=ylabel)
        parts.append(figure(ylabel,draw))
    parts.append('<h2>三次完整CC-CV循环</h2><p>每次：放电到2.5 V → 静置600 s → 1C充到4.2 V → 恒压到C/20 → 静置600 s。全部区域状态逐步连续性检查；无退化模型，不能解释为寿命预测。3³完整固相参照与8区降阶相互比较；4³另有12区三循环。</p>')
    for field,ylabel,scale in [('voltage_V','Voltage / V',1),('current_A','Current / A',1),('temperature_K','Temperature / K',1),('thickness_m','Thickness change / um',1e6)]:
        def draw(ax,field=field,ylabel=ylabel,scale=scale):
            for n in ['full3_cycles','rom3_cycles','rom4_cycles']:
                d=cases[n]['curves'];ax.plot(np.asarray(d['time_s'])/3600,np.asarray(d[field])*scale,label=n,alpha=.8)
            ax.set(xlabel='Time / h',ylabel=ylabel)
        parts.append(figure('Three reversible cycles: '+ylabel,draw))
    rows=[]
    for i,cycle in enumerate(cases['full3_cycles']['protocol']['cycles']):
        for step in cycle:rows.append([i+1,step['step']+1,f"{step['duration_s']:.3f}",f"{step['voltage_end_V']:.6f}",f"{step['current_end_A']:.6f}",step['termination']])
    parts.append(table(['循环','步骤','持续 s','末电压 V','末电流 A','终止'],rows))
    parts.append(f"<p class=\"notice\">3³八区对完整18状态三循环，在相同物理时间上电压RMSE {cycle_comparison['voltage_RMSE_mV']:.2f} mV，总时差{cycle_comparison['cutoff_time_difference_s']:.2f} s；正极应力L2 {100*cycle_comparison['fields']['positive']['stress_relative_L2']:.2f}%、P99归一化RMS {100*cycle_comparison['fields']['positive']['P99_tension_normalized_RMS']:.2f}%。协议完成与守恒通过，不能替代降阶精度验收。</p>")
    parts.append('<h2>精度审查 · 相同物理时间，未对齐SOC或拟合电压</h2><p>全阶每活性体素一个状态（4³有42个）；降阶为区域常数。应力误差在81个共同时间、全部宏观点、全部Tet上体积加权；P99用固体Tet体积加权主拉应力。P99归一化尺度为参照RMS与1 kPa中的较大者。浓度另以cmax归一化。这里评估同一离散族截断误差，不是连续体或实验准确性。</p>')
    rows=[]
    for n,v in comparisons.items():
        rows.append([n,f"{v['voltage_RMSE_mV']:.4g}",f"{v['cutoff_time_difference_s']:.3f}",*[f"{100*v['fields'][d]['stress_relative_L2']:.3f}% / {100*v['fields'][d]['P99_tension_normalized_RMS']:.3f}%" for d in ['negative','positive']]])
    parts.append(table(['候选','电压 RMSE mV','截止时差 s','负极应力L2 / P99 RMS','正极应力L2 / P99 RMS'],rows))
    def pareto(ax):
        for d in ['negative','positive']:
            names=['mean4_1c','rom8_1c','rom16_1c','rom24_1c'];r=[cases[n]['geometry'][d]['online_regions'] for n in names]
            ax.plot(r,[100*comparisons[n]['fields'][d]['stress_relative_L2'] for n in names],'o-',label=d+' stress L2')
        ax.set(xlabel='Online regions per electrode',ylabel='Relative stress L2 error / %')
    parts.append(figure('Concentration voltage accuracy does not certify local stress',pareto))
    parts.append('<h2>双向反馈、宏观离散与独立微结构</h2><p>λ=1与λ=0采用相同几何/区域/参数，差别来自力学反馈；nx3→6→12是宏观离散敏感性，不能混为ROM误差。familyB16改为独立seed108、相同实际孔隙率，差别是模型预测的结构效应，尚缺该结构全阶参照和材料标定。</p>')
    parts.append(table(['比较','电压RMS / 最大 mV','温度RMS K','厚度RMS µm','共同终点 s'],[[name,f"{1000*v['voltage_V']['RMS']:.5g} / {1000*v['voltage_V']['maximum']:.5g}",f"{v['temperature_K']['RMS']:.5g}",f"{1e6*v['thickness_m']['RMS']:.5g}",f"{v['common_end_s']:.3f}"] for name,v in curve_comparisons.items()]))
    parts.append('<h2>真实逐步演算、反向端口上界与富集</h2><p>变流序列包含放电、静置、充电。计时包括状态查询；冷编译、暖步进、场恢复、富集分别计费。软实时实测不构成硬实时期限保证。BV上界只覆盖同一Pa处细力学势与区域平均势造成的瞬时反应差，不是非线性轨迹或应力误差界。</p>')
    warm=np.asarray([r['wall_seconds'] for r in live['steps'][1:]])
    parts.append(table(['指标','实测'],[['首步含编译 s',f"{live['steps'][0]['wall_seconds']:.5g}"],['暖步进中位 / 最大 ms',f'{1000*np.median(warm):.4g} / {1000*warm.max():.4g}'],['完整场暖恢复中位 ms',f"{1000*np.median(live['recovery_warm_seconds']):.4g}"],['嵌套富集构建 s',f"{live['adaptivity']['enrichment_seconds']:.5g}"],['富集后首步 s',f"{live['enriched_steps'][0]['wall_seconds']:.5g}"],['检查点重启电压 / 温度差',f"{live['restart_voltage_difference_V']:.3e} V / {live['restart_temperature_difference_K']:.3e} K"],['富集内能 / 熵跳差',f"{live['energy_jump_J']:.3e} J / {live['entropy_jump_J_K']:.3e} J K⁻¹"]]))
    parts.append(table(['电极','X点','实际反应差 A/m³','计算上界 A/m³','经验浓度缺陷'],[[b['domain'],b['x'],f"{b['actual_defect']:.5g}",f"{b['bound']:.5g}",f"{b['empirical_concentration_defect']:.5g}"] for b in live['bounds'] if 'actual_defect' in b]))
    parts.append('<h2>17工况守恒、独立状态能量与资源</h2><p>内能由浓度的OCV原函数及Fel−T∂TFel独立计算，核对外电功和换热。熵另含电解液混合项。各算例耗散非负已检查，但未证明原生宏观电解液FV对所有工况普遍熵稳定。单CPU/线程、nice15、GPU隐藏、组RSS1.5GiB、单任务锁；不干扰其他训练。</p>')
    rows=[];resources={}
    for n,r in cases.items():
        m=r['metrics'];manifest=json.loads((WS/'reports'/f'micro_{n}_latest.json').read_text());resources[n]=manifest
        rows.append([n,f"{r['geometry']['negative']['micro_cells']} → {r['geometry']['negative']['online_regions']}",f"{m['relative_lithium_drift']:.2e}",f"{m['relative_internal_energy_residual']:.2e}",f"{m['relative_entropy_residual']:.2e}",f"{m['build_and_solve_seconds']:.3f}",f"{manifest['peak_group_rss_bytes_sampled']/2**20:.1f}"])
    parts.append(table(['工况','固相N → r','锂漂移','内能残差','熵残差','编译+求解 s','采样组RSS MiB'],rows))
    failures=[]
    for path in sorted((WS/'runs').glob('*micro*/manifest.json')):
        m=json.loads(path.read_text())
        if m['exit_code']!=0:failures.append({'run':str(path.parent),'reason':m['stopped_because'],'exit_code':m['exit_code'],'sampled_RSS_MiB':m['peak_group_rss_bytes_sampled']/2**20})
    parts.append(table(['保留失败','退出','原因','采样RSS MiB'],[[f['run'].split('/')[-1],f['exit_code'],f['reason'],f"{f['sampled_RSS_MiB']:.1f}"] for f in failures]))
    parts.append(f'<h2>当前研究结论与未过门槛</h2><p>{tests}项自身测试通过。已形成真实非均匀固相状态替换、能量共轭双向反馈、完整可逆循环、按需场恢复与嵌套连续富集。区域投影、线性叠加、静态凝聚本身均属已有方法，力学探针分区是否优于几何分区由表中数据决定。</p><p class="notice">尚未达到顶刊完整验收：当前仅低分辨率合成SVE，凸化OCP与Ω×0.1，热膨胀/隔膜参数未标定；12³的1134状态只验证直接区域离线/在线生产路径，未取得其全阶参照；三维孔隙电解液、真实CT/RVE收敛、实验校准、NTFA/EHM先进基线、非线性轨迹认证与严格尾部应力认证仍缺。旧二次系统100倍加速不能移用至本DFN。</p>')
    parts.append('<p>论文候选主线：可恢复微观观测量的能量/功率对偶DFN闭合，以及可计算局部非线性反应误差驱动的状态连续富集。成熟先例：<a href="https://onlinelibrary.wiley.com/doi/10.1002/nme.70313">VCH/POD电化学降阶</a>、<a href="https://doi.org/10.1149/1945-7111/adb684">CT—DFN桥接</a>。原创性暂为待验证C类。</p>')
    installed=json.loads((WS/'reports/microstructure_installed_wheel.json').read_text())
    assert installed['source_imported'] is False and installed['upstream_distribution_installed'] is False and installed['version']=='0.0.7'
    parts.append('<h2>复现与独立安装</h2><p>0.0.7 wheel在外部独立venv安装、去掉源码导入路径后，60s微结构—DFN/热/力步进及局部恢复通过；未安装上游PyBaMM发行包。原GEMsim274源文件与捆绑PyBaMM355源文件及许可证保持来源追踪。</p><p>串行正式实验：<code>scripts/run_microstructure_suite.py</code>；实际JSONL变流/场恢复入口：<code>scripts/run_microstructure_live.py</code>；均通过<code>scripts/run_local.py</code>启动，运行目录为外部workspace。源码/数据hash变动拒绝复用。报告滑块仅重放真实求解轨迹。</p>')
    summary={'schema':'morqex-microstructure-report-1','version':'0.0.7','test_count':tests,'comparisons':comparisons,'cycle_comparison':cycle_comparison,'curve_comparisons':curve_comparisons,
        'case_checks':{n:r['checks'] for n,r in cases.items()},'runtime':live,'artifacts_sha256':artifacts,
        'resources':resources,'failures':failures,'installed_wheel':installed,'viewer':viewer_data(cases)}
    report=WS/'reports';(report/'microstructure_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    download=base64.b64encode(json.dumps({k:v for k,v in summary.items() if k!='viewer'},allow_nan=False).encode()).decode()
    parts.append('<p><a download="MORQEX_microstructure_evidence.json" href="data:application/json;base64,'+download+'">下载指标、资源及数据SHA256</a> · <a href="MORQEX_DFN_report.html">0.0.6独立DFN/实验对照历史报告</a></p>')
    template=(ROOT/'scripts/microstructure_report_template.html').read_text()
    text=template.replace('<!-- CONTENT -->',''.join(parts)).replace('/* VIEWER_DATA */',json.dumps(summary['viewer'],separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    path=report/'MORQEX_microstructure_report.html';path.write_text(text);(report/'MORQEX_test_report.html').write_text(text)
    (report/'microstructure_report_manifest.json').write_text(json.dumps({'html_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'cases':len(cases),'tests':tests,'artifacts':artifacts},indent=2))
    print(json.dumps({'html':str(path),'bytes':path.stat().st_size,'tests':tests,'comparisons':comparisons},indent=2))

if __name__=='__main__':main()
