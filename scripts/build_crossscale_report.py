"""Source-bound 0.0.9 crossscale/cathode/degradation evidence and field viewer."""
import base64
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from morqex.electrochemistry.aging_cache import AgingLocalization,load_cohesive_certificate
from morqex.cohesive_reduction import cohesive_certificate
from morqex.mechanics.tensors import from_mandel
import build_upgrade_report as br
from run_crossscale_suite import CASES
from run_generated_suite import CASES as GENERATED_CASES

ROOT=Path(__file__).resolve().parents[1];WS=Path(os.environ['MORQEX_WORKSPACE']);FIG=WS/'results/crossscale_figures';FIG.mkdir(parents=True,exist_ok=True);br.FIG=FIG

def history(case,d,t):
    m=case['microstructure'][d];source=case['curves']['time_s']
    c=interp1d(source,np.asarray(m['concentration']),axis=1)(t)
    E=interp1d(source,np.asarray(m['strain']),axis=1)(t);damage=interp1d(source,np.asarray(m['damage']),axis=1)(t)
    theta=np.interp(t,source,case['curves']['temperature_K'])-case['geometry'][d]['operator']['reference_temperature_K']
    return c,E,np.broadcast_to(theta,E.shape),damage

def compare(case,reference):
    end=min(case['curves']['time_s'][-1],reference['curves']['time_s'][-1]);extra=[]
    for r in [case,reference]:
        if 'cycles' in r['protocol']:
            elapsed=0.
            for cycle in r['protocol']['cycles']:
                for step in cycle:elapsed+=step['duration_s'];extra.append(elapsed)
    t=np.unique(np.r_[np.linspace(0,end,321),np.asarray(extra)[np.asarray(extra)<=end]])
    dt=np.diff(t);tw=np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2]
    result={'common_end_s':float(end),'field_time_samples':len(t),'metric':'trapezoid physical-time weighting; uniform samples plus protocol transitions'}
    for d in ['negative','positive']:
        a=AgingLocalization(case['microstructure'][d]['kernel_path']);b=AgingLocalization(reference['microstructure'][d]['kernel_path'])
        assert a.kernel.operator.metadata()['geometry_fingerprint']==b.kernel.operator.metadata()['geometry_fingerprint']
        fa=a.recover(*history(case,d,t));fb=b.recover(*history(reference,d,t));w=b.arrays['volumes']/b.arrays['volumes'].sum()
        norm=lambda x:float(np.einsum('xtek,e,t->',x*x,w,tw))
        pa=br.weighted_quantile(np.maximum(0,np.linalg.eigvalsh(from_mandel(fa['stress']))[...,-1]),w)
        pr=br.weighted_quantile(np.maximum(0,np.linalg.eigvalsh(from_mandel(fb['stress']))[...,-1]),w)
        tail=float(np.sqrt(np.einsum('xt,t->',(pa-pr)**2,tw)/(pa.shape[0]*end)))
        tail_scale=float(np.sqrt(np.einsum('xt,t->',pr*pr,tw)/(pr.shape[0]*end)))
        wa=b.arrays['interface_areas']/b.arrays['interface_areas'].sum()
        opening=lambda x:float(np.einsum('xtik,i,t->',x*x,wa,tw))
        result[d]={'stress_relative_L2':float(np.sqrt(norm(fa['stress']-fb['stress'])/max(1e-30,norm(fb['stress'])))),
            'P99_normalized_RMS':tail/max(1000,tail_scale),'P99_absolute_RMS_Pa':tail,'P99_reference_RMS_Pa':tail_scale,
            'opening_relative_L2':float(np.sqrt(opening(fa['opening']-fb['opening'])/max(1e-30,opening(fb['opening']))))}
    tv=np.unique(np.r_[case['curves']['time_s'],reference['curves']['time_s']]);tv=tv[tv<=end]
    dv=np.interp(tv,case['curves']['time_s'],case['curves']['voltage_V'])-np.interp(tv,reference['curves']['time_s'],reference['curves']['voltage_V'])
    result['voltage_RMSE_mV']=float(np.sqrt(np.trapezoid(dv*dv,tv)/end)*1000);result['voltage_maximum_error_mV']=float(np.max(abs(dv))*1000)
    for field in ['temperature_K','thickness_m']:
        end=result['common_end_s'];t=np.linspace(0,end,301)
        a=np.interp(t,case['curves']['time_s'],case['curves'][field]);b=np.interp(t,reference['curves']['time_s'],reference['curves'][field])
        result[field+'_RMS_difference']=float(np.sqrt(np.mean((a-b)**2)))
    result['end_time_difference_s']=case['metrics']['end_time_s']-reference['metrics']['end_time_s']
    if 'cycles' in case['protocol'] and 'cycles' in reference['protocol']:
        square=duration=maximum=stage_duration=0.;sa=sb=0.
        for ca,cb in zip(case['protocol']['cycles'],reference['protocol']['cycles']):
            for j,(aa,bb) in enumerate(zip(ca,cb)):
                da,db=aa['duration_s'],bb['duration_s'];dt=min(da,db);tau=np.linspace(0,dt,301);values=[]
                for r,start,length in [(case,sa,da),(reference,sb,db)]:
                    time=np.asarray(r['curves']['time_s']);current=np.asarray(r['curves']['current_A']);v=np.asarray(r['curves']['voltage_V'])
                    mask=(time>=start-1e-6)&(time<=start+length+1e-6)
                    if j in [1,4]:mask&=abs(current)<1e-7
                    elif j==0:mask&=current>1e-7
                    elif j in [2,3]:mask&=current<-1e-7
                    if mask.sum()<2:raise ValueError('Not enough protocol-conditioned samples')
                    values.append(np.interp(tau,time[mask]-start,v[mask]))
                difference=values[0]-values[1];square+=np.trapezoid(difference*difference,tau);duration+=dt
                maximum=max(maximum,float(np.max(abs(difference))));stage_duration=max(stage_duration,abs(da-db));sa+=da;sb+=db
        result['protocol_conditioned_voltage']={'RMSE_mV':float(np.sqrt(square/duration)*1000),'maximum_mV':maximum*1000,'maximum_stage_duration_difference_s':stage_duration,
            'scope':'elapsed time within corresponding control segments; timing discrepancy separately reported; common physical-clock metrics retained above'}
    return result

def frames(case,d,ids):
    local=AgingLocalization(case['microstructure'][d]['kernel_path']);a=local.arrays;m=case['microstructure'][d];t=np.asarray(case['curves']['time_s'])[ids]
    c,E,theta,damage=history(case,d,t);f=local.recover(c,E,theta,damage);n=a['labels'].size
    stress=f['stress'];dev=stress.copy();dev[...,:3]-=stress[...,:3].mean(axis=-1,keepdims=True);vm=np.sqrt(1.5*np.sum(dev*dev,axis=-1))
    volume=np.bincount(a['element_voxel_ids'],weights=a['volumes'],minlength=n);v=np.zeros((*E.shape,n))
    for x in range(E.shape[0]):
        for j in range(len(ids)):v[x,j]=np.bincount(a['element_voxel_ids'],weights=a['volumes']*vm[x,j],minlength=n)/np.maximum(volume,1e-30)/1e6
    active=np.zeros(n,bool);active[a['voxel_ids']]=True
    soc=np.zeros_like(v);soc[...,a['voxel_ids']]=f['concentration']/case['parameter_audit']['host_scales'][d]['maximum_concentration_mol_m3']
    output={'soc':soc,'vm':v,'damage':np.broadcast_to(damage[...,None],v.shape).copy()}
    for name in ['sei','plating','dead_lithium']:
        value=np.zeros_like(v)
        if name in m:
            stock=np.asarray(m[name])[:,ids,:];density=stock/np.maximum(a['reaction_area'],1e-30)
            fine=(density@a['P'].T)*a['area_full']
            np.testing.assert_allclose(fine.sum(axis=-1),stock.sum(axis=-1),rtol=1e-10,atol=1e-12)
            value[...,a['voxel_ids']]=5.+4.7925e-5*(density@a['P'].T)*1e9 if name=='sei' else fine/np.maximum(a['weights_full'],1e-30)
        output[name]=value
    doping=np.zeros_like(v);profile=getattr(local.kernel.operator,'regional_chemistry',None)
    if profile is not None:doping[...,a['voxel_ids']]=profile.fraction@a['P'].T
    output['dopant']=doping
    return {'shape':list(a['labels'].shape),'labels':a['labels'].ravel().tolist(),'active_mask':active.tolist(),
        'fields':{name:values.tolist() for name,values in output.items()},'ranges':{name:[float(values.min()),float(values.max())] for name,values in output.items()},'active_voxels':int(active.sum()),'regions':local.kernel.operator.dimension,
        'opening_P99_nm':br.weighted_quantile(np.linalg.norm(f['opening'],axis=-1)*1e9,a['interface_areas']).tolist()}

def certificate_comparison(case,reference):
    records=[]
    for d in ['negative','positive']:
        small=AgingLocalization(case['microstructure'][d]['kernel_path']);full=AgingLocalization(reference['microstructure'][d]['kernel_path'])
        data=load_cohesive_certificate(case['microstructure'][d]['kernel_path']);cohesive=small.kernel.cohesive
        t=np.linspace(0,case['curves']['time_s'][-1],21);c,E,theta,damage=history(case,d,t)
        for x in range(E.shape[0]):
            for j in range(len(t)):
                q=np.r_[E[x,j],c[x,j]-small.kernel.operator.reference_concentration,theta[x,j]];dd=float(damage[x,j]);bound=cohesive_certificate(cohesive,data,q,dd)
                a=.5*q@cohesive.hessian(dd)@q;b=.5*q@full.kernel.cohesive.hessian(dd)@q
                fs=small.recover(c[x,j],E[x,j],theta[x,j],dd)['stress'];ff=full.recover(c[x,j],E[x,j],theta[x,j],dd)['stress']
                error=float(np.max(np.linalg.norm(fs-ff,axis=-1)));rounding=256*np.finfo(float).eps*max(1.,abs(a),abs(b))
                assert a-b>=-rounding and a-b<=bound['energy_error_upper_J_m3']+rounding
                assert error<=bound['stress_point_upper_Pa']+1e-3
                records.append({'electrode':d,'time_s':float(t[j]),'X':x,'energy_error':float(a-b),'energy_bound':bound['energy_error_upper_J_m3'],
                    'stress_point_error_Pa':error,'stress_point_bound_Pa':bound['stress_point_upper_Pa'],'arithmetic_energy_allowance':rounding})
    return records

def main():
    cases={};resources={};artifacts={}
    for name in [n for n,_ in CASES]+[n for n,_,_ in GENERATED_CASES]:
        path=WS/'results/aging'/name/'result.json'
        if not path.exists():continue
        data=json.loads(path.read_text());br.verify(data)
        if not all(data['checks'].values()):raise ValueError('Failed result: '+name)
        cases[name]=data;resources[name]=json.loads((WS/'reports'/f'{name}_latest.json').read_text());assert resources[name]['exit_code']==0
        artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    required=['closure_ct4_exact_cycles','closure_ct4_rom8_cycles','closure_ct4_raw8_cycles','closure_ct4_rom16_cycles',
        'closure_ct4_full_pulse','closure_ct4_rom8_pulse','closure_nmc_mid','closure_lfp_mid','closure_lco_mid',
        'closure_doped_uniform','closure_doped_gradient','closure_doped_shell','closure_lfp_doped','closure_lco_doped']
    assert all(n in cases for n in required),[n for n in required if n not in cases]
    checks=json.loads((WS/'reports/crossscale_tests_latest.json').read_text());assert checks['exit_code']==0
    suites=list(ET.parse(Path(checks['run'])/'test_results.xml').getroot().iter('testsuite'));count=sum(int(s.get('tests',0)) for s in suites)
    assert not any(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    errors={name:compare(cases[name],cases['closure_ct4_exact_cycles']) for name in ['closure_ct4_rom8_cycles','closure_ct4_rom16_cycles','closure_ct4_raw8_cycles','closure_ct4_geo8_cycles']}
    errors['closure_ct4_rom8_pulse']=compare(cases['closure_ct4_rom8_pulse'],cases['closure_ct4_full_pulse'])
    for name,reference in [('closure_lfp_cycle','closure_lfp_exact_cycle'),('closure_lfp_rom16_cycle','closure_lfp_exact_cycle'),('closure_lco_cycle','closure_lco_exact_cycle'),('closure_doped_gradient','closure_doped_exact_cycles')]:
        if name in cases and reference in cases:errors[name]=compare(cases[name],cases[reference])
    native={}
    for path in sorted((WS/'results/native_dfn').glob('*.json')):
        r=json.loads(path.read_text());assert r['driver_sha256']==hashlib.sha256((ROOT/'benchmarks/study_geometry_dfn_control.py').read_bytes()).hexdigest()
        assert r['source_case_sha256']==hashlib.sha256(Path(r['source_case']).read_bytes()).hexdigest()
        assert all(hashlib.sha256((ROOT/'src/pybamm'/n).read_bytes()).hexdigest()==h for n,h in r['backend']['source_sha256'].items())
        assert r['protocol']['completed'] and r['relative_lithium_drift_including_side_reservoirs']<1e-8
        native[path.stem]=r;artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for family in ['isotropic','directional','CBD-rich']:
        a=f'gem_{family}_rom8';b=f'gem_{family}_full';assert a in cases and b in cases
        errors[a]=compare(cases[a],cases[b])
    precision={}
    if 'gem_directional_full_tight_cycle' in cases:
        precision['directional_first_cycle']=compare(cases['gem_directional_full'],cases['gem_directional_full_tight_cycle'])
        precision['directional_first_cycle']['scope']='common first-cycle physical time only; 1e-7/1e-8 vs 1e-9/1e-10 integration, same full discretization; end-time difference spans different cycle counts and is not an accuracy metric'
    a=native['isotropic_aging_16'];b=native['isotropic_aging_32'];end=min(a['curves']['time_s'][-1],b['curves']['time_s'][-1]);t=np.linspace(0,end,1201)
    dv=np.interp(t,a['curves']['time_s'],a['curves']['voltage_V'])-np.interp(t,b['curves']['time_s'],b['curves']['voltage_V'])
    precision['native_radial_16_32']={'voltage_RMSE_mV':float(np.sqrt(np.trapezoid(dv*dv,t)/end)*1000),'maximum_voltage_difference_mV':float(np.max(abs(dv))*1000),'capacity_difference_Ah':(np.asarray(a['cycle_discharge_Ah'])-b['cycle_discharge_Ah']).tolist(),'scope':'two radial resolutions; native spherical DFN aging; not continuum convergence'}
    certificates=certificate_comparison(cases['closure_ct4_rom8_cycles'],cases['closure_ct4_raw8_cycles'])
    parts=['<p>目标：微结构与成分决定DFN输运/反应/能量端口，退化反馈端口，宏观状态驱动局部场恢复。本文所有图来自实际积分。CT仍是极小SVE，掺杂系数是声明的合成灵敏度输入。</p>',
        '<h2>完整固相、变化损伤与副反应的三循环参照</h2><p>同一4³CT裁剪、相同3点宏观网格、相同FP64/容差、3.5–4.0V窗口、同一变化界面与副反应。完整参照保留全部活性固相状态和完整界面谱，电解液仍是DFN尺度；这不是全孔隙三维电解液参照或连续体误差。</p>']
    def curves(fig,axes):
        for name in ['closure_ct4_exact_cycles','closure_ct4_rom8_cycles','closure_ct4_rom16_cycles']:
            r=cases[name];t=np.asarray(r['curves']['time_s'])/3600
            axes[0,0].plot(t,r['curves']['voltage_V'],label=name);axes[0,1].plot(t,np.asarray(r['microstructure']['positive']['damage'])[1],label=name)
        for ax,ylabel in zip(axes.ravel(),['Voltage / V','Positive SVE contact damage']):ax.set(xlabel='Time / h',ylabel=ylabel);ax.legend(fontsize=7);ax.grid(alpha=.2)
    parts.append(br.figure('cycles','Evolving degradation: full solid reference and reduced trajectories',curves,(1,2)))
    parts.append('<h2>循环退化与机制对照</h2><p>无退化、仅SEI、仅接触损伤和全部机制用同一8区空间与协议。首次初始SOC与后续不同，不能把首次容量差解释为衰减；这里单独列循环2→3。阈值与掺杂物性未标定，不能当寿命预测。</p>')
    control_names=['closure_no_degradation','closure_sei_only','closure_contact_only','closure_ct4_rom8_cycles']
    def capacities(fig,axes):
        for name in control_names:
            r=cases[name];q=[row['discharge_Ah'] for row in r['metrics']['cycle_metrics']]
            axes[0,0].plot([2,3],np.array(q[1:])/q[1]*100,'o-',label=name)
            stock=r['reservoirs_mol'].get('negative_sei',np.zeros(len(r['curves']['time_s'])))
            axes[0,1].plot(np.asarray(r['curves']['time_s'])/3600,np.asarray(stock)*96485.33212/3600*1000,label=name)
        for ax,xlabel,ylabel in zip(axes.ravel(),['Cycle','Time / h'],['Capacity relative to cycle 2 / %','SEI lithium equivalent / mAh']):ax.set(xlabel=xlabel,ylabel=ylabel);ax.legend(fontsize=7);ax.grid(alpha=.2)
    parts.append(br.figure('capacity_mechanisms','Discharge capacity and SEI inventory are distinct observables',capacities,(1,2)))
    def metal(fig,axes):
        for name in ['closure_ct4_full_pulse','closure_ct4_rom8_pulse']:
            r=cases[name];t=r['curves']['time_s']
            for ax,key in zip(axes.ravel(),['negative_plating','negative_dead_lithium']):ax.plot(t,np.asarray(r['reservoirs_mol'][key])*96485.33212/3600*1e6,label=name)
        for ax,ylabel in zip(axes.ravel(),['Reversible deposited Li / uAh','Dead lithium / uAh']):ax.set(xlabel='Time / s',ylabel=ylabel);ax.legend(fontsize=7);ax.grid(alpha=.2)
    parts.append(br.figure('plated_dead_lithium','Cold thin-electrode pulse: metal generated from zero, then dead lithium',metal,(1,2)))
    parts.append(br.table(['候选','电压RMSE mV','正极应力L2','正极主拉应力P99归一RMS','正极界面跳量L2'],[[name,f"{v['voltage_RMSE_mV']:.6g}",*[f'{100*v["positive"][k]:.5g}%' for k in ['stress_relative_L2','P99_normalized_RMS','opening_relative_L2']]] for name,v in errors.items()]))
    parts.append('<p>P99是固相体积加权的max(0,最大主应力)分位数，误差取时间和宏观点RMS，以参照P99的RMS归一；不是单点峰值误差或最大时间误差。应力场L2包含六个Mandel分量及固相体积/时间权重，界面跳量使用真实界面面积权重。</p>')
    parts.append('<p>主指标按共同物理时间、固相体积权重比较。场误差在321个均匀时刻加控制切换时刻以梯形时间权重求积；电压使用存储时间点并集，最大误差另给出。电流阶跃会使任意小的切换时间差产生有限瞬时电压差，因此证据同时附对应控制阶段内的经过时间比较及阶段时长差，二者与原时钟指标并存。模式增加的收益与同阶算法收益分开；16区在正极已等于全部16个活性体素，不能把正极接近精确当作低阶突破。</p>')
    parts.append('<p class="notice">LFP原8区应力场误差13.75%，未过10%门槛，保留为失败精度候选；增加到16区后重新对照。电压控制切换时刻存在电流阶跃，不同轨迹触发时刻不同，因此共同物理时钟上的最大误差与时间积分RMSE同时保留，不能只挑RMSE。</p>')
    parts.append('<h2>独立生成微结构 × 标准DFN × 全固相参照</h2><p>GEMsim固定种子三组3³合成SVE：各向同性随机场、方向性随机场、提高CBD比例。每组两极独立生成，采用5点宏观网格、相同3.5–3.9V三循环协议。方向性描述生成器设定，3³样本不足以证明统计各向异性；不是RVE。标准DFN匹配体积分数、几何输运效率和初始反应面积，R=3ε/a；其球扩散/原生副反应/原生热模型与MORQEX不同，用于模型形式对照，不作为降阶误差参照。</p>')
    def morphologies(fig,axes):
        for col,family in enumerate(['isotropic','directional','CBD-rich']):
            for kind in ['full','rom8']:
                r=cases[f'gem_{family}_{kind}'];axes[0,col].plot(np.asarray(r['curves']['time_s'])/3600,r['curves']['voltage_V'],label='MORQEX '+kind)
            for kind in ['bare','aging']:
                r=native[f'{family}_{kind}_16'];axes[0,col].plot(np.asarray(r['curves']['time_s'])/3600,r['curves']['voltage_V'],linestyle='--',label='Spherical DFN '+kind)
            axes[0,col].set(title=family,xlabel='Time / h',ylabel='Voltage / V');axes[0,col].legend(fontsize=6);axes[0,col].grid(alpha=.2)
    parts.append(br.figure('morphology_dfn_cycles','Independent morphologies: discharge / rest / CC / CV / rest',morphologies,(1,3)))
    def morphology_degradation(fig,axes):
        for family in ['isotropic','directional','CBD-rich']:
            r=cases[f'gem_{family}_rom8'];t=np.asarray(r['curves']['time_s'])/3600
            axes[0,0].plot(t,np.asarray(r['microstructure']['positive']['damage'])[2],label=family)
            axes[0,1].plot(t,np.asarray(r['reservoirs_mol']['negative_sei'])*96485.33212/3600*1e3,label=family)
        for ax,ylabel in zip(axes.ravel(),['Positive contact damage at middle X','SEI lithium equivalent / mAh']):ax.set(xlabel='Time / h',ylabel=ylabel);ax.legend(fontsize=7);ax.grid(alpha=.2)
    parts.append(br.figure('morphology_degradation','Degradation responds to each geometry; uncalibrated constitutive inputs',morphology_degradation,(1,2)))
    parts.append(br.table(['微结构','模型','循环1放电Ah','循环2放电Ah','循环3放电Ah','总墙钟s'],[[family,'MORQEX '+kind,*[f"{v['discharge_Ah']:.7g}" for v in cases[f'gem_{family}_{kind}']['metrics']['cycle_metrics']],f"{resources[f'gem_{family}_{kind}']['wall_seconds']:.4g}"] for family in ['isotropic','directional','CBD-rich'] for kind in ['full','rom8']]+[[family,'DFN '+kind,*[f'{q:.7g}' for q in native[f'{family}_{kind}_16']['cycle_discharge_Ah']],f"{json.loads((WS/'reports'/f'native_{family}_{kind}_16_latest.json').read_text())['wall_seconds']:.4g}"] for family in ['isotropic','directional','CBD-rich'] for kind in ['bare','aging']]))
    parts.append('<p>DFN原生SEI与析锂量、库存漂移和径向16→32单元敏感性随证据下载。不同方法容量差是物理表示差异，不能称为本方法对真实材料更准确。三循环可检测状态演化，不能替代长期寿命验证。</p>')
    def efficiency(fig,axes):
        names=['closure_ct4_exact_cycles','closure_ct4_raw8_cycles','closure_ct4_rom8_cycles','closure_ct4_rom16_cycles'];x=np.arange(len(names))
        bottom=np.zeros(len(names))
        for key,label in [('offline_seconds','Offline'),('symbolic_model_seconds','Symbolic model'),('build_solve_seconds','Discretize / compile / integrate')]:
            value=np.array([cases[n]['metrics'][key] for n in names]);axes[0,0].bar(x,value,bottom=bottom,label=label);bottom+=value
        axes[0,0].set(ylabel='Measured seconds (audit/output separate)');axes[0,0].legend(fontsize=7)
        axes[0,1].bar(x,[resources[n]['peak_group_rss_bytes_sampled']/2**20 for n in names]);axes[0,1].axhline(1536,color='r',linestyle='--');axes[0,1].set(ylabel='Sampled process-group RSS / MiB')
        for ax in axes.ravel():ax.set_xticks(x,['Full','8 raw poles','8 reduced poles','16 reduced poles'],rotation=20,ha='right');ax.grid(axis='y',alpha=.2)
    parts.append(br.figure('efficiency','Same physics / tolerance / hardware / output contract',efficiency,(1,2)))
    speed={n:resources['closure_ct4_exact_cycles']['wall_seconds']/resources[n]['wall_seconds'] for n in ['closure_ct4_rom8_cycles','closure_ct4_rom16_cycles']}
    parts.append(br.table(['方案','含审计/输出端到端墙钟s','相对完整参照','离线s','编译及积分s'],[[n,f"{resources[n]['wall_seconds']:.4g}",f'{speed[n]:.3f}x',f"{cases[n]['metrics']['offline_seconds']:.4g}",f"{cases[n]['metrics']['build_solve_seconds']:.4g}"] for n in speed]))
    parts.append('<p>上述端到端包含各自状态轨迹导出及相同宏观/守恒审计，原始状态维数随模型不同；未计画图或完整场恢复。完整固相参照也复用离线力学响应，不是每步重装配的低效FEM。相同输入、相同完整场输出的预热恢复成本在下表另测；本轮不把编译加积分的总计冒充纯积分耗时。</p>')
    macro_check=None
    if 'closure_macro5_discharge' in cases:
        a=cases['closure_macro3_discharge'];b=cases['closure_macro5_discharge'];end=min(a['curves']['time_s'][-1],b['curves']['time_s'][-1]);tt=np.linspace(0,end,1201)
        delta=np.interp(tt,a['curves']['time_s'],a['curves']['voltage_V'])-np.interp(tt,b['curves']['time_s'],b['curves']['voltage_V'])
        macro_check={'voltage_RMSE_mV':float(np.sqrt(np.trapezoid(delta*delta,tt)/end)*1000),'scope':'3 vs 5 macro cells per domain at fixed 8-region closure; sensitivity, not continuum error bound'}
        parts.append(f"<p>固定8区闭合、6000s放电宏观3→5点的电压差RMS为{macro_check['voltage_RMSE_mV']:.5g}mV。原0.1C、4.0V上限的5点循环在约18703.86s/3.9888V达到负极局部0.999浓度上限；收紧容差100倍仍触发，端点证据保留，未算作完整循环或收敛通过。该项与微观浓度降阶误差分开，两级放电网格不能证明整个循环连续体收敛。</p>")
    if 'closure_macro3_safe_current' in cases and 'closure_macro5_safe_current' in cases:
        a=cases['closure_macro3_safe_current'];b=cases['closure_macro5_safe_current'];end=min(a['curves']['time_s'][-1],b['curves']['time_s'][-1]);tt=np.unique(np.r_[a['curves']['time_s'],b['curves']['time_s']]);tt=tt[tt<=end]
        delta=np.interp(tt,a['curves']['time_s'],a['curves']['voltage_V'])-np.interp(tt,b['curves']['time_s'],b['curves']['voltage_V'])
        macro_check['safe_006C_cycles']={'voltage_RMSE_mV':float(np.sqrt(np.trapezoid(delta*delta,tt)/end)*1000),'maximum_difference_mV':float(np.max(abs(delta))*1000),'scope':'same 0.06C full three-cycle protocol, 3 vs 5 macro points; two-grid sensitivity, not continuum convergence'}
        parts.append(f"<p>明确改为0.06C后，3点和5点均完成3.5–4.0V的三个循环，原浓度上限不变；两网格电压RMSE为{macro_check['safe_006C_cycles']['voltage_RMSE_mV']:.5g}mV。0.05C曾因CC电流等于固定C/20停止阈值而使CV初始事件不可行，已改正该协议配置，未删事件或伪造零长度CV。</p>")
    kernel_benchmark=json.loads((WS/'reports/crossscale_kernel_benchmark.json').read_text())
    assert kernel_benchmark['driver_sha256']==hashlib.sha256((ROOT/'benchmarks/benchmark_crossscale_kernels.py').read_bytes()).hexdigest()
    parts.append(br.table(['同输入/同完整场输出','完整谱极点数','压缩后','七次预热恢复中位加速','应力L2误差'],[[d,r['raw_modes'],r['compressed_modes'],f"{r['median_speedup']:.4g}x",f"{100*r['stress_relative_L2']:.5g}%"] for d,r in kernel_benchmark['electrodes'].items()]))
    parts.append('<p>上表单独测完整场恢复，不包含离线、磁盘或DFN演化。大裁剪6³同8浓度区的完整谱/压缩谱端到端成本也单列，避免把减少浓度自由度的收益全归因于谱压缩。</p>')
    def certplot(fig,axes):
        for ax,key,bound,label in zip(axes.ravel(),['energy_error','stress_point_error_Pa'],['energy_bound','stress_point_bound_Pa'],['Energy error / J m-3','Point stress error / Pa']):
            a=np.maximum([abs(r[key]) for r in certificates],1e-15);b=np.maximum([r[bound] for r in certificates],1e-15)
            ax.semilogy(a,label='Observed at same q,d');ax.semilogy(b,label='Residual bound');ax.set(xlabel='Fixed-state query index',ylabel=label);ax.legend(fontsize=7);ax.grid(alpha=.2)
    parts.append('<h2>界面压缩：同状态的可计算界</h2><p>Galerkin投影保持有理能量结构，离线QR残差给出机械能、广义力及应力的保守界。训练仅用算子。下面核对相同q,d下压缩核与完整谱；非线性轨迹误差另由上方全阶对照测量，二者不混用。能量近舍入误差时保留浮点比较余量。</p>')
    parts.append(br.figure('certificate','Fixed-state resolvent compression bounds, not a DFN trajectory certificate',certplot,(1,2)))
    family_names=['closure_nmc_mid','closure_lfp_mid','closure_lco_mid']
    parts.append('<h2>NMC / LFP / LCO：独立材料参数路径</h2><p>分别来自固定OKane2022、Prada2013和Ai2020包，保留投影/补充参数审计。LFP是有效单相自由能，尚未解析两相前沿/滞回；LCO化学OCP熵被显式关闭。使用同一CT形貌是虚拟材料替换，不是真实LFP/LCO CT或寿命标定。</p>')
    def hosts(fig,axes):
        for name in family_names:
            r=cases[name];t=r['curves']['time_s'];label=r['arguments']['cathode']
            axes[0,0].plot(t,r['curves']['voltage_V'],label=label);axes[0,1].plot(t,np.asarray(r['microstructure']['positive']['release'])[1],label=label)
        for ax,ylabel in zip(axes.ravel(),['Voltage / V','Positive energy release / J m-3']):ax.set(xlabel='Time / s',ylabel=ylabel);ax.legend();ax.grid(alpha=.2)
    parts.append(br.figure('cathodes','Three distinct host parameterizations in the same crossscale framework',hosts,(1,2)))
    parts.append(br.table(['母体','原参数','cmax正极 mol/m³','正极OCP最大投影改动V','附加补充项数'],[[r['arguments']['cathode'],r['parameter_audit']['source_parameter_set'],r['parameter_audit']['host_scales']['positive']['maximum_concentration_mol_m3'],f"{r['parameter_audit']['electrodes']['positive']['max_ocp_change_V']:.5g}",len(r['parameter_audit']['supplemented_from_OKane2022'])] for r in [cases[n] for n in family_names]]))
    parts.append('<h2>掺杂场确实进入微结构与DFN</h2><p>示例X是合成灵敏度参数，不指代已测定的真实元素。局部OCP/扩散/主副反应倍率和分相弹性/化学膨胀同时参与计算；成分边界处分区，不隐式平均。共用宿主cmax、固定掺杂；容量变化/新相/掺杂迁移需对应新闭合。</p>')
    def doped(fig,axes):
        for name in ['closure_ct4_rom8_cycles','closure_doped_uniform','closure_doped_gradient','closure_doped_shell']:
            r=cases[name];axes[0,0].plot(np.asarray(r['curves']['time_s'])/3600,r['curves']['voltage_V'],label=name)
            axes[0,1].plot(np.asarray(r['curves']['time_s'])/3600,np.asarray(r['microstructure']['positive']['damage'])[1],label=name)
        for ax,ylabel in zip(axes.ravel(),['Voltage / V','Positive contact damage']):ax.set(xlabel='Time / h',ylabel=ylabel);ax.legend(fontsize=7);ax.grid(alpha=.2)
    parts.append(br.figure('doping_coupling','Prescribed dopant-property laws: coupled sensitivity, not material prediction',doped,(1,2)))
    viewer={}
    selected=['closure_ct4_rom8_cycles','closure_ct4_rom8_pulse','closure_doped_gradient','closure_lfp_doped','closure_lco_doped','closure_ct6_compressed']
    if 'closure_ct8_compressed' in cases:selected.append('closure_ct8_compressed')
    selected.extend(f'gem_{family}_rom8' for family in ['isotropic','directional','CBD-rich'])
    for n in selected:
        r=cases[n];ids=np.unique(np.linspace(0,len(r['curves']['time_s'])-1,49).astype(int))
        viewer[n]={'time':np.asarray(r['curves']['time_s'])[ids].tolist(),'voltage':np.asarray(r['curves']['voltage_V'])[ids].tolist(),'temperature':np.asarray(r['curves']['temperature_K'])[ids].tolist(),'micro':{d:frames(r,d,ids) for d in ['negative','positive']}}
    def fields(fig,axes):
        for row,n in enumerate(['closure_doped_uniform','closure_doped_gradient','closure_doped_shell']):
            r=cases[n];ids=np.unique(np.rint(np.linspace(0,len(r['curves']['time_s'])-1,3)).astype(int));m=frames(r,'positive',ids);shape=m['shape'];mask=np.array(m['active_mask']).reshape(shape);z=shape[0]//2
            values=np.asarray(m['fields']['vm'])[1].reshape(len(ids),*shape);limit=values.max()
            for col in range(3):
                image=axes[row,col].imshow(np.ma.array(values[col,z],mask=~mask[z]),origin='lower',cmap='magma',vmin=0,vmax=limit)
                axes[row,col].set(title=f'{n.rsplit("_",1)[-1]} / {r["curves"]["time_s"][ids[col]]/3600:.2f} h');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            fig.colorbar(image,ax=list(axes[row]),label='Active voxel VM / MPa',shrink=.65)
    parts.append(br.figure('doping_field_evolution','Actual stress snapshots; fixed scale per scenario through time',fields,(3,3)))
    parts.append('<h2>场演化查看器</h2><p>滑块重放求解轨迹；新输入由Python DAE积分。SEI为亚体素状态，d为每SVE一个标量，不伪造逐界面裂纹。浓度和掺杂仅在活性域着色，应力包括CBD。</p><div class="controls"><label>工况<select id="case"></select></label><label>电极<select id="domain"><option value="negative">负极</option><option value="positive">正极</option></select></label><label>X点<select id="macro"><option>0</option><option selected>1</option><option>2</option></select></label><label>场<select id="field"><option value="soc">c/cmax</option><option value="vm">von Mises / MPa</option><option value="dopant">掺杂位点分数</option><option value="sei">SEI / nm</option><option value="plating">析锂 / mol m⁻³活性相</option><option value="dead_lithium">死锂 / mol m⁻³活性相</option><option value="damage">SVE统一损伤d</option></select></label><label>时间<input id="time" type="range" min="0" value="0"></label><label>Z<input id="depth" type="range" min="0" value="0"></label></div><canvas id="map" width="760" height="390"></canvas><p id="readout"></p>')
    parts.append('<h2>守恒、资源与验收范围</h2>')
    parts.append(br.table(['工况','总锂漂移','内能残差','熵残差','墙钟s','采样RSS MiB'],[[n,f"{r['metrics']['relative_lithium_including_reservoirs_drift']:.2e}",f"{r['metrics']['relative_total_internal_energy_residual']:.2e}",f"{r['metrics']['relative_entropy_residual']:.2e}",f"{resources[n]['wall_seconds']:.3f}",f"{resources[n]['peak_group_rss_bytes_sampled']/2**20:.1f}"] for n,r in cases.items()]))
    failures=[]
    for p in sorted((WS/'runs').glob('*/manifest.json')):
        m=json.loads(p.read_text())
        if any(tag in p.parent.name for tag in ['closure_','gem_','native_','diagnose_']) and m['exit_code']!=0:failures.append({'run':str(p.parent),'reason':m['stopped_because'] or ('CPU time limit' if m['exit_code']==-24 else 'see stderr and endpoint diagnostics'),'exit':m['exit_code']})
    parts.append(f'<p class="notice">{count}项测试通过；{len(cases)}个本版工况通过限定数值门禁。所有作业仍1CPU、nice15、GPU隐藏、RSS1.5GiB，失败日志保留。尚缺CT代表性/连续体收敛、材料与掺杂及退化标定、全部先进基线和非线性全过程误差认证；界面压缩及共享控制属于已知数学/工程思想的应用，独立原创仍需审查。没有宣称全部微结构机理或顶刊验收完成。</p>')
    installed=json.loads((WS/'reports/crossscale_installed_wheel.json').read_text());assert installed['version']=='0.0.9' and not installed['source_imported'] and not installed['upstream_distribution_installed']
    diagnosis=json.loads((WS/'reports/protocol_diagnosis.json').read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in diagnosis['source_sha256'].items())
    assert all(hashlib.sha256((ROOT/'src/pybamm'/n).read_bytes()).hexdigest()==h for n,h in diagnosis['backend']['source_sha256'].items())
    wheel=WS/'dist/morqex_research-0.0.9-py3-none-any.whl'
    for p in [WS/'reports/crossscale_kernel_benchmark.json',WS/'reports/crossscale_installed_wheel.json',WS/'reports/protocol_diagnosis.json',Path(checks['run'])/'test_results.xml',wheel]:artifacts[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    evidence={'version':'0.0.9','tests':count,'artifacts':artifacts,'errors':errors,'certificates':certificates,'resources':resources,'speed_end_to_end':speed,'macro_grid_sensitivity':macro_check,
        'release_source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(ROOT/'src/morqex').rglob('*.py'),*(ROOT/'tests').glob('*.py'),Path(__file__)]},'installed_wheel':installed,
        'case_checks':{n:r['checks'] for n,r in cases.items()},'metrics':{n:r['metrics'] for n,r in cases.items()},'parameter_audits':{n:r['parameter_audit'] for n,r in cases.items()},
        'geometry':{n:r['geometry'] for n,r in cases.items()},'precision_sensitivity':precision,'native_geometry_DFN':native,'localization_benchmark':kernel_benchmark,'protocol_diagnosis':diagnosis,'backend':next(iter(cases.values()))['backend'],'source_sha256':next(iter(cases.values()))['source_sha256'],'failures':failures}
    (WS/'reports/crossscale_summary.json').write_text(json.dumps(evidence,indent=2,allow_nan=False))
    encoded=base64.b64encode(json.dumps(evidence,allow_nan=False).encode()).decode()
    parts.append('<p><a download="MORQEX_crossscale_evidence.json" href="data:application/json;base64,'+encoded+'">下载全部数值证据与hash</a></p>')
    template=(ROOT/'scripts/upgrade_template.html').read_text().replace('0.0.8','0.0.9').replace('分相CBD与退化跨尺度闭环','微结构—DFN退化、掺杂与三类正极')
    template=template.replace("(p===1&&key!=='vm')","(m.active_mask[i]&&key!=='vm')")
    template=template.replace("const whole=m.fields[key].flat(2);let lo=Infinity,hi=-Infinity;for(const row of whole){for(const x of Array.isArray(row)?row:[row]){lo=Math.min(lo,x);hi=Math.max(hi,x)}}",'let [lo,hi]=m.ranges[key];')
    html=template.replace('<!-- CONTENT -->',''.join(parts)).replace('/* VIEWER_DATA */',json.dumps(viewer,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    html=html.replace('<select id="macro"><option>0</option><option selected>1</option><option>2</option></select>','<input id="macro" type="range" min="0" max="2" value="1">')
    html=html.replace("el('time').max=d.time.length-1;", "el('macro').max=m.fields[key].length-1;el('time').max=d.time.length-1;")
    html=html.replace("xi=+el('macro').value", "xi=Math.min(+el('macro').value,m.fields[key].length-1)")
    html=html.replace('`t=${d.time[ti].toFixed(2)} s', '`X index=${xi} | t=${d.time[ti].toFixed(2)} s')
    path=WS/'reports/MORQEX_crossscale_cathodes_report.html';path.write_text(html);(WS/'reports/MORQEX_test_report.html').write_text(html)
    manifest={'html_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tests':count,'cases':len(cases),'viewer_cases':len(viewer),'artifacts':artifacts,'figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FIG.glob('*')}}
    (WS/'reports/crossscale_report_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'html':str(path),'tests':count,'cases':len(cases),'errors':errors,'speed':speed},indent=2))

if __name__=='__main__':main()
