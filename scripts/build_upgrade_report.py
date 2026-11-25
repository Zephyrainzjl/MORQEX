"""Scientific CT/degradation/evolution report from frozen external evidence."""
import base64
from html import escape
import hashlib
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
import tifffile
from morqex.electrochemistry.aging_cache import AgingLocalization
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
from morqex.mechanics.tensors import from_mandel

ROOT=Path(__file__).resolve().parents[1];WS=Path(os.environ['MORQEX_WORKSPACE']);FIG=WS/'results/upgrade_figures';FIG.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
from run_upgrade_suite import CASES
import build_microstructure_report as old_report

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def figure(name,title,draw,shape=None):
    if shape is None:fig,axes=plt.subplots(figsize=(9,3.8),layout='constrained')
    else:fig,axes=plt.subplots(*shape,figsize=(4.2*shape[1],3.5*shape[0]),squeeze=False,layout='constrained')
    draw(fig,axes);fig.suptitle(title,fontsize=12);fig.savefig(FIG/(name+'.svg'));fig.savefig(FIG/(name+'.png'),dpi=160)
    stream=io.BytesIO();fig.savefig(stream,format='svg');plt.close(fig)
    return '<figure><img alt="'+escape(title)+'" src="data:image/svg+xml;base64,'+base64.b64encode(stream.getvalue()).decode()+'"></figure>'

def verify(data):
    for name,value in data['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,('stale',name)
    if 'backend' in data:
        for name,value in data['backend']['source_sha256'].items():assert hashlib.sha256((ROOT/'src/pybamm'/name).read_bytes()).hexdigest()==value

def weighted_quantile(x,w,p=.99):
    order=np.argsort(x,axis=-1);values=np.take_along_axis(x,order,axis=-1);weights=w[order]
    idx=np.argmax(np.cumsum(weights,axis=-1)>=p*weights.sum(axis=-1,keepdims=True),axis=-1)
    return np.take_along_axis(values,idx[...,None],axis=-1)[...,0]

def interp_history(case,d,t):
    micro=case['microstructure'][d];source=case['curves']['time_s']
    c=interp1d(source,np.asarray(micro['concentration']),axis=1)(t)
    E=interp1d(source,np.asarray(micro['strain']),axis=1)(t);damage=interp1d(source,np.asarray(micro['damage']),axis=1)(t)
    theta=np.interp(t,source,case['curves']['temperature_K'])-298.15
    return c,E,np.broadcast_to(theta,E.shape),damage

def ct_errors(case,reference):
    end=min(case['curves']['time_s'][-1],reference['curves']['time_s'][-1]);t=np.linspace(0,end,81);result={}
    for d in ['negative','positive']:
        la=AgingLocalization(case['microstructure'][d]['kernel_path']);lb=AgingLocalization(reference['microstructure'][d]['kernel_path'])
        assert la.kernel.operator.metadata()['geometry_fingerprint']==lb.kernel.operator.metadata()['geometry_fingerprint']
        a=la.recover(*interp_history(case,d,t));b=lb.recover(*interp_history(reference,d,t));w=lb.arrays['volumes']/lb.arrays['volumes'].sum()
        norm=lambda x:float(np.einsum('...ek,e->...',x*x,w).sum())
        pa=weighted_quantile(np.maximum(0,np.linalg.eigvalsh(from_mandel(a['stress']))[...,-1]),w)
        pb=weighted_quantile(np.maximum(0,np.linalg.eigvalsh(from_mandel(b['stress']))[...,-1]),w)
        result[d]={'stress_relative_L2':np.sqrt(norm(a['stress']-b['stress'])/max(1e-30,norm(b['stress']))),
            'P99_normalized_RMS':float(np.sqrt(np.mean((pa-pb)**2))/max(1000,np.sqrt(np.mean(pb*pb)))),
            'opening_relative_L2':float(np.linalg.norm(a['opening']-b['opening'])/max(1e-30,np.linalg.norm(b['opening'])))}
    dv=np.interp(t,case['curves']['time_s'],case['curves']['voltage_V'])-np.interp(t,reference['curves']['time_s'],reference['curves']['voltage_V'])
    result['voltage_RMSE_mV']=float(np.sqrt(np.mean(dv*dv))*1000);result['common_end_s']=float(end);return result

def reconstruct_frames(case,d,indices):
    m=case['microstructure'][d];local=AgingLocalization(m['kernel_path']);a=local.arrays;op=local.kernel.operator
    c=np.asarray(m['concentration'])[:,indices,:];E=np.asarray(m['strain'])[:,indices];damage=np.asarray(m['damage'])[:,indices];theta=np.asarray(case['curves']['temperature_K'])[indices]-298.15
    field=local.recover(c,E,theta[None,:],damage);labels=a['labels'];size=labels.size
    # Voxel VM is the Tet-volume average of VM, rather than VM of averaged stress.
    s=field['stress'];dev=s.copy();dev[...,:3]-=s[...,:3].mean(axis=-1,keepdims=True);vm=np.sqrt(1.5*np.sum(dev*dev,axis=-1))
    voxel_weight=np.bincount(a['element_voxel_ids'],weights=a['volumes'],minlength=size);vmvox=np.zeros((*E.shape,size))
    for x in range(E.shape[0]):
        for ti in range(len(indices)):vmvox[x,ti]=np.bincount(a['element_voxel_ids'],weights=a['volumes']*vm[x,ti],minlength=size)/np.maximum(voxel_weight,1e-30)
    cube=np.zeros((*E.shape,size));cube[...,a['voxel_ids']]=field['concentration']/PARAMETERS[f'Maximum concentration in {d} electrode [mol.m-3]']
    fields={'soc':cube,'vm':vmvox/1e6,'damage':np.broadcast_to(damage[...,None],cube.shape).copy()}
    for name in ['sei','plating','dead_lithium']:
        output=np.zeros_like(cube)
        if name in m:
            n=np.asarray(m[name])[:,indices,:]
            # Conservative allocation by true reactive area, not by voxel count.
            density=n/np.maximum(a['reaction_area'],1e-30)
            fine=(density@a['P'].T)*a['area_full']
            np.testing.assert_allclose(fine.sum(axis=-1),n.sum(axis=-1),rtol=1e-11,atol=1e-12)
            if name=='sei':value=5.+4.7925e-5*(density@a['P'].T)*1e9
            else:value=fine/np.maximum(a['weights_full'],1e-30)
            output[...,a['voxel_ids']]=value
        fields[name]=output
    opening=np.linalg.norm(field['opening'],axis=-1)*1e9
    p99=weighted_quantile(opening,a['interface_areas'])
    return {'shape':list(labels.shape),'labels':labels.ravel().tolist(),'fields':{n:v.tolist() for n,v in fields.items()},
        'opening_P99_nm':p99.tolist(),'active_voxels':len(a['voxel_ids']),'regions':op.dimension,
        'cohesive_modes':len(local.kernel.cohesive.eigenvalues)}

def main():
    global PARAMETERS
    PARAMETERS,_=microstructure_parameters();old_report.PARAMETERS=PARAMETERS
    cases={};goals={};rejected={};artifacts={};resources={}
    for name,kind,_,reason in CASES:
        folder=WS/'results'/('aging' if kind=='aging' else 'microstructure_dfn')/name
        path=folder/('rejected_protocol.json' if reason else 'result.json');r=json.loads(path.read_text());verify(r);artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        resources[name]=json.loads((WS/'reports'/f'upgrade_{name}_latest.json').read_text())
        if reason:assert reason in r['termination'];rejected[name]=r
        else:
            assert all(r['checks'].values())
            (cases if kind=='aging' else goals)[name]=r
    tests_manifest=json.loads((WS/'reports/upgrade_tests_latest.json').read_text());assert tests_manifest['exit_code']==0
    suites=list(ET.parse(Path(tests_manifest['run'])/'test_results.xml').getroot().iter('testsuite'));tests=sum(int(s.get('tests',0)) for s in suites)
    assert tests>=166 and not any(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    goal={n:old_report.field_comparison(goals[n],goals['goal_ref42']) for n in ['goal_geo16','goal_port16','goal_port24','goal_port32','goal_tail16','goal_tail32','goal_tail40']}
    ct_compare={n:ct_errors(cases[n],cases['ct_exact']) for n in ['ct_rom4','ct_rom8']}
    parts=['<h2>分相物理与可替换接口</h2><p>本版把几何、材料/拓扑、基函数、算子/能量核、界面动力学、SEI、析锂/剥离/死锂、损伤、热模型和求解器作为显式组合入口。CBD可以同时承载、导电子和传输微孔电解液；每相可指定固定弹性张量、化学/热膨胀与对角扩散倍率。接口可扩展不代表所有物理组合已验证。</p>',
        table(['已实现','限制'],[['真实CT相标签与轴序/hash导入','原CBD由作者数值重构，示例微孔参数未标定'],['分相各向异性固定弹性、线性化学膨胀','混合宿主化学、非对角扩散需相应新插件'],['变化刚度的活性材料—CBD界面弹簧','每SVE一个损伤标量，无逐界面裂纹路径/摩擦/闭合'],['SEI、析锂/剥离、死锂库存与膜电阻','声明的被动动力学变体，不冒充所有已发表退化模型'],['热—电化学—力学单体DAE','理想电解液、集总热、固定bulk C、零轴向外压'],['自定义Python工厂/子模型、缓存/能力检查','错误单位/库存/耗散需独立资格，不静默套用不兼容模式']]),
        '<h2>公开CT来源与实际计算尺度</h2><p>官方NMC532 1-CAL（398 nm）与石墨5-CAL（126 nm）数据。活性形貌源于X射线CT，CBD由数据作者重构；当前假设CBD微孔率0.47、相对离子效率0.05。裁剪不补孔、不重标、不降采样，按固定种子与可达性顺序选择。4³只验证CT来源微小SVE的接口与离散闭环，不能代表真实电极或RVE。</p>']
    source_manifest=json.loads((WS/'data/ct/sources.json').read_text())
    def source_plot(fig,axes):
        for ax,(d,r) in zip(axes.ravel(),cases['ct_all']['geometry'].items()):
            src=r['source'];assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
            with tifffile.TiffFile(src['path']) as f:volume=f.asarray(maxworkers=1).transpose(1,2,0)
            crop=r['data']['actual_crop'];z=(crop[0][0]+crop[0][1])//2
            ax.imshow(volume[z],origin='lower',cmap=ListedColormap(['#e7edf1','#40776b','#bb873e']),vmin=0,vmax=2)
            ax.add_patch(Rectangle((crop[2][0]-.5,crop[1][0]-.5),crop[2][1]-crop[2][0],crop[1][1]-crop[1][0],fill=False,edgecolor='red',linewidth=1.5))
            ax.set(title=d+' / CT slice, red = simulated crop',xlabel='X voxel / through plane',ylabel='Y voxel')
    parts.append(figure('ct_sources','Actual segmented CT sources; CBD is reconstructed',source_plot,(1,2)))
    parts.append('<p>数据来源：<a href="https://www.nlr.gov/transportation/battery-microstructure-library-data">官方微结构库</a>；关联论文 <a href="https://doi.org/10.1149/2.0731814jes">Usseglio-Viretta等2018</a>。源文件SHA256、裁剪坐标、过滤拒绝数均在证据下载中。</p>')
    parts.append('<h2>三次退化循环与单独机制对照</h2><p>有效窗口3.5–4.0 V，0.5 A（参数集中名义0.1C），每循环含静置300s与CV至0.25 A。不是原始2.5–4.2V全电压容量复现。全部机制案例采用未标定的低损伤阈值100 J/m³以观察接触损失；默认20 kJ/m³在该协议未激活，二者不能混写。首次循环初始SOC与后续不同，容量衰减需对照同窗口、同周期及不退化基线。</p>')
    for field,title,ylabel,scale in [('voltage_V','cycling_voltage','Voltage / V',1),('temperature_K','cycling_temperature','Temperature / K',1)]:
        def draw(fig,ax,field=field,ylabel=ylabel,scale=scale):
            for n in ['ct_none','ct_sei','ct_damage','ct_all']:
                c=cases[n]['curves'];ax.plot(np.asarray(c['time_s'])/3600,np.asarray(c[field])*scale,label=n,alpha=.8)
            ax.set(xlabel='Time / h',ylabel=ylabel);ax.legend(fontsize=8);ax.grid(alpha=.2)
        parts.append(figure(title,'CT-derived SVE / '+ylabel,draw))
    capacity_rows=[]
    for n in ['ct_none','ct_sei','ct_damage','ct_all']:
        m=cases[n]['metrics'];qs=[x['discharge_Ah'] for x in m['cycle_metrics']]
        capacity_rows.append([n,*[f'{q:.7f}' for q in qs],f'{100*(qs[2]/qs[1]-1):+.5f}%',f"{m['reservoir_end_mol'].get('negative_sei',0)*96485.33212/3600:.6g}",f"{m['damage_end']['positive']:.5f}"])
    parts.append(table(['模式','循环1 Ah','循环2 Ah','循环3 Ah','3/2容量变化','SEI等价LLI Ah','正极d最大'],capacity_rows))
    def capacity(fig,ax):
        for n in ['ct_none','ct_sei','ct_damage','ct_all']:
            q=[m['discharge_Ah'] for m in cases[n]['metrics']['cycle_metrics']];ax.plot([2,3],np.asarray(q[1:])/q[1]*100,'o-',label=n)
        ax.set(xlabel='Cycle (cycle 2 baseline)',ylabel='Discharged capacity relative to cycle 2 / %');ax.legend(fontsize=8);ax.grid(alpha=.2)
    parts.append(figure('capacity','Capacity is measured from the discharge step, not inferred from LLI',capacity))
    def degradation(fig,axes):
        r=cases['ct_all'];t=np.asarray(r['curves']['time_s'])/3600
        for d in ['negative','positive']:
            for x in range(3):axes[0,0].plot(t,np.asarray(r['microstructure'][d]['damage'])[x],label=d+' X'+str(x))
        for n,c in r['reservoirs_mol'].items():axes[0,1].plot(t,np.asarray(c)*96485.33212/3600*1000,label=n)
        axes[0,0].set(xlabel='Time / h',ylabel='Uniform SVE cohesive damage d');axes[0,1].set(xlabel='Time / h',ylabel='Lithium-equivalent inventory / mAh')
        for ax in axes.ravel():ax.legend(fontsize=7);ax.grid(alpha=.2)
    parts.append(figure('degradation_modes','Contact damage and separate lithium reservoirs',degradation,(1,2)))
    parts.append('<h2>析锂、剥离与死锂：独立激活测试</h2><p>薄电极10µm/10µm、263.15K、中间SOC参考态，充电2s → 静置1200s → 放电60s。此为明确改动条件的机制测试，不是标定电芯快充预测；初始金属锂为零。普通低倍率循环未出现析锂时如实显示零。</p>')
    def plating(fig,axes):
        r=cases['ct_pulse'];t=np.asarray(r['curves']['time_s'])
        axes[0,0].plot(t,r['curves']['current_A'],label='Current');axes[0,0].set(xlabel='Time / s',ylabel='Current / A')
        for n in ['negative_plating','negative_dead_lithium']:
            axes[0,1].plot(t,np.asarray(r['reservoirs_mol'][n])*96485.33212/3600*1e6,label=n)
        axes[0,1].set(xlabel='Time / s',ylabel='Li-equivalent inventory / uAh')
        for ax in axes.ravel():ax.legend(fontsize=7);ax.grid(alpha=.2)
    parts.append(figure('plating_dead_lithium','Generated plated Li and subsequent irreversible dead Li',plating,(1,2)))
    viewer={}
    for n in ['ct_all','ct_pulse','ct_larger']:
        r=cases[n];ids=np.unique(np.linspace(0,len(r['curves']['time_s'])-1,65).astype(int))
        viewer[n]={'time':np.asarray(r['curves']['time_s'])[ids].tolist(),'voltage':np.asarray(r['curves']['voltage_V'])[ids].tolist(),
            'temperature':np.asarray(r['curves']['temperature_K'])[ids].tolist(),'micro':{d:reconstruct_frames(r,d,ids) for d in ['negative','positive']}}
    parts.append('<h2>DFN → CT微结构：演化场与界面位移跳量</h2><p>由求得的区域浓度、宏观应变、温度和变化的d，通过同一有理算子恢复场；SEI/金属锂/死锂按真实反应面积守恒分配。薄膜是亚体素状态，不伪造CT分辨的SEI形貌。损伤每SVE仅一个标量，不能把颜色理解为已预测逐界面裂纹路径。应力是Tet体积平均von Mises；时间全程固定色标。</p><div class="controls"><label>工况<select id="case"></select></label><label>电极<select id="domain"><option value="negative">负极</option><option value="positive">正极</option></select></label><label>X积分点<select id="macro"><option>0</option><option selected>1</option><option>2</option></select></label><label>场<select id="field"><option value="soc">浓度 / cmax</option><option value="vm">von Mises / MPa</option><option value="sei">SEI厚度 / nm</option><option value="plating">析出锂 / mol m⁻³活性相</option><option value="dead_lithium">死锂 / mol m⁻³活性相</option><option value="damage">SVE损伤 d</option></select></label><label>时间<input id="time" type="range" value="0" min="0"></label><label>Z切片<input id="depth" type="range" value="0" min="0"></label></div><canvas id="map" width="760" height="390"></canvas><p id="readout"></p>')
    r=cases['ct_all'];idx=np.unique(np.rint(np.linspace(0,len(r['curves']['time_s'])-1,4)).astype(int));snapshots={d:reconstruct_frames(r,d,idx) for d in ['negative','positive']}
    def evolution(fig,axes):
        d='positive';m=snapshots[d];shape=m['shape'];z=shape[0]//2;values=np.asarray(m['fields']['vm'])[1].reshape(len(idx),*shape);soc=np.asarray(m['fields']['soc'])[1].reshape(len(idx),*shape)
        labels=np.asarray(m['labels']).reshape(shape)
        for row,(v,name,cmap) in enumerate([(soc,'Concentration / cmax','viridis'),(values,'von Mises / MPa','magma')]):
            valid=labels==1 if row==0 else labels>0
            lo=float(v[:,valid].min());hi=float(v[:,valid].max())
            for col,j in enumerate(idx):
                axes[row,col].imshow(labels[z],origin='lower',cmap=ListedColormap(['#e7edf1','#40776b','#bb873e']),vmin=0,vmax=2)
                im=axes[row,col].imshow(np.ma.array(v[col,z],mask=~valid[z]),origin='lower',vmin=lo,vmax=hi,cmap=cmap);axes[row,col].set_title(f"t={r['curves']['time_s'][j]/3600:.2f} h",fontsize=9);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            fig.colorbar(im,ax=list(axes[row]),label=name,shrink=.6)
    parts.append(figure('ct_field_evolution','Positive CT SVE / fixed DFN position / fixed scale through cycling',evolution,(2,4)))
    def evolution3d(fig,axes):
        m=snapshots['positive'];shape=m['shape'];mask=np.flatnonzero(np.asarray(m['labels'])>0)
        z,y,x=np.unravel_index(mask,shape);spacing=cases['ct_all']['geometry']['positive']['source']['voxel_size_m']*1e6
        values=np.asarray(m['fields']['vm'])[1];lo=float(values.min());hi=float(values.max())
        for panel,ti in enumerate([0,len(idx)//2,len(idx)-1]):
            fig.delaxes(axes[0,panel]);ax=fig.add_subplot(1,3,panel+1,projection='3d')
            point=ax.scatter((x+.5)*spacing,(y+.5)*spacing,(z+.5)*spacing,c=values[ti,mask],s=70,marker='s',cmap='magma',vmin=lo,vmax=hi,depthshade=False)
            ax.set(xlabel='X / um',ylabel='Y / um',zlabel='Z / um',title=f"t={r['curves']['time_s'][idx[ti]]/3600:.2f} h")
        fig.colorbar(point,ax=fig.axes,label='Voxel-averaged von Mises / MPa',shrink=.6)
    parts.append(figure('ct_3d_evolution','3D voxel-center samples of the actual CT SVE stress evolution',evolution3d,(1,3)))
    def opening(fig,ax):
        for d in ['negative','positive']:
            m=viewer['ct_all']['micro'][d];ax.plot(np.asarray(viewer['ct_all']['time'])/3600,np.asarray(m['opening_P99_nm'])[1],label=d)
        ax.set(xlabel='Time / h',ylabel='Area-weighted P99 displacement jump / nm');ax.legend();ax.grid(alpha=.2)
    parts.append(figure('interface_opening','Actual separated-node cohesive-interface displacement jump',opening))
    parts.append('<h2>误差审查：同一离散参照、同一物理时间</h2><p>合成4³基准用于隔离基函数改动，完整42状态作为参照。端口基只用几何/机械响应和面积信息，未使用验证轨迹训练。不同模式数分开列出，增加模式的收益不能算作同阶创新；全阶精确回退也不等于获得加速。</p>')
    parts.append(table(['候选','电压RMSE mV','负极应力L2 / P99','正极应力L2 / P99'],[[n,f"{v['voltage_RMSE_mV']:.5g}",*[f"{100*v['fields'][d]['stress_relative_L2']:.3f}% / {100*v['fields'][d]['P99_tension_normalized_RMS']:.3f}%" for d in ['negative','positive']]] for n,v in goal.items()]))
    def error_curve(fig,axes):
        for prefix,label,marker in [('geo','Geometry','s'),('port','Energy-port','o'),('tail','Tail-targeted','^')]:
            selected=[(int(n.rsplit(prefix,1)[1]),v) for n,v in goal.items() if n.startswith('goal_'+prefix)]
            selected.sort(key=lambda item:item[0])
            for ax,key in zip(axes.ravel(),['stress_relative_L2','P99_tension_normalized_RMS']):
                ax.plot([n for n,v in selected],[100*v['fields']['positive'][key] for n,v in selected],marker+'-',label=label)
        for ax,ylabel in zip(axes.ravel(),['Positive stress relative L2 / %','Positive tensile P99 normalized RMS / %']):
            ax.axhline(5,color='#888888',linestyle='--',label='5% research target')
            ax.set(xlabel='Concentration regions (full reference: 42)',ylabel=ylabel,yscale='log');ax.legend(fontsize=8);ax.grid(alpha=.2)
    parts.append(figure('basis_error_tradeoff','Independent synthetic validation: accuracy versus model size, not a speedup claim',error_curve,(1,2)))
    parts.append('<p>CT来源4³另用完整固相状态与同一cohesive界面模型对照6000s：比较应力、主拉应力P99及界面位移跳量。它仍是同一空间离散的降阶比较，不是连续体、代表性或实验误差。</p>')
    parts.append(table(['CT候选','电压RMSE mV','负极应力 / P99 / 位移跳量L2','正极应力 / P99 / 位移跳量L2'],[[n,f"{v['voltage_RMSE_mV']:.5g}",*[f"{100*v[d]['stress_relative_L2']:.3f}% / {100*v[d]['P99_normalized_RMS']:.3f}% / {100*v[d]['opening_relative_L2']:.3f}%" for d in ['negative','positive']]] for n,v in ct_compare.items()]))
    a,b=cases['ct_short'],cases['ct_tight'];t=np.linspace(0,120,101);dv=np.interp(t,a['curves']['time_s'],a['curves']['voltage_V'])-np.interp(t,b['curves']['time_s'],b['curves']['voltage_V']);tight={'voltage_RMS_V':float(np.sqrt(np.mean(dv*dv)))}
    parts.append(f"<p>120s求解容差加严交叉，电压RMS差{tight['voltage_RMS_V']*1000:.5g}mV；这是容差敏感性，不替代时间/空间连续体收敛。</p>")
    parts.append('<h2>守恒、能量、稳定性与资源</h2><p>全库存包括宿主/电解液、SEI、析出锂与死锂。主/副反应、膜热与扩散满足局部功率核验；动态cohesive能量、温度/损伤交叉导数进入热方程。每次构建用直接FEM核对算子。保持单CPU、nice15、GPU隐藏、组RSS1.5GiB、CPU120s和单任务锁。</p>')
    parts.append(table(['工况','总锂漂移','总内能残差','熵残差','电荷残差','墙钟s / 采样RSS MiB'],[[n,f"{r['metrics']['relative_lithium_including_reservoirs_drift']:.2e}",f"{r['metrics']['relative_total_internal_energy_residual']:.2e}",f"{r['metrics']['relative_entropy_residual']:.2e}",f"{max(r['metrics']['charge_residual'].values()):.2e}",f"{resources[n]['wall_seconds']:.2f} / {resources[n]['peak_group_rss_bytes_sampled']/2**20:.1f}"] for n,r in cases.items()]))
    parts.append('<h2>明确保留的拒绝与不足</h2>')
    parts.append(table(['工况','末时间s','末电压V','拒绝原因'],[[n,f"{r['end_time_s']:.3f}",f"{r['end_voltage_V']:.6f}",r['termination']] for n,r in rejected.items()]))
    failures=[]
    for path in sorted((WS/'runs').glob('*/manifest.json')):
        m=json.loads(path.read_text())
        if m['exit_code']!=0 and ('aging' in path.parent.name or 'upgrade' in path.parent.name):failures.append({'run':str(path.parent),'exit_code':m['exit_code'],'reason':m['stopped_because']})
    parts.append(f'<p class="notice">{tests}项自身测试通过，接口、CT来源、退化库存与变化界面闭环已有数值证据；尚未达到顶刊完整验收。仍缺标定材料/副反应/界面参数、CT尺度与代表性收敛、大型界面谱压缩、全程误差认证、先进NTFA/EHM等完整基线，以及实验退化验证。不是所有微结构或所有退化机制已解决；复杂塑性/黏弹CBD、真实裂纹传播与三维电芯热场均未验证。源码接口不能代替这些证据。</p>')
    parts.append('<p>复现：<code>scripts/run_upgrade_suite.py</code>；接口与数学范围见源码文档<code>docs/EXTENSIBLE_DEGRADATION.md</code>。已有局部化、凝聚、cohesive与端口方法均是成熟先例，不宣称首次。旧报告仅对应各自冻结提交。</p>')
    installed_path=WS/'reports/upgrade_installed_wheel.json';installed=json.loads(installed_path.read_text());assert installed['source_imported'] is False and installed['version']=='0.0.8'
    summary={'schema':'morqex-upgrade-report-1','tests':tests,'goal_errors':goal,'ct_errors':ct_compare,'tolerance_comparison':tight,'artifacts':artifacts,
        'case_checks':{n:r['checks'] for n,r in cases.items()},'metrics':{n:r['metrics'] for n,r in cases.items()},'rejections':rejected,
        'resources':resources,'historical_failures':failures,'CT_sources':source_manifest,'installed_wheel':installed,'viewer':viewer}
    (WS/'reports/upgrade_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    download=base64.b64encode(json.dumps({k:v for k,v in summary.items() if k!='viewer'},allow_nan=False).encode()).decode()
    parts.append('<p><a download="MORQEX_upgrade_evidence.json" href="data:application/json;base64,'+download+'">下载数值证据、来源、参数与hash</a></p>')
    template=(ROOT/'scripts/upgrade_template.html').read_text();html=template.replace('<!-- CONTENT -->',''.join(parts)).replace('/* VIEWER_DATA */',json.dumps(viewer,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    path=WS/'reports/MORQEX_degradation_CT_report.html';path.write_text(html);(WS/'reports/MORQEX_test_report.html').write_text(html)
    manifest={'html_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'tests':tests,'successful_cases':len(cases)+len(goals),'physical_rejections':len(rejected),
        'artifacts':artifacts,'figure_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FIG.glob('*') if p.suffix in ['.svg','.png']}}
    (WS/'reports/upgrade_report_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'html':str(path),'bytes':path.stat().st_size,'tests':tests,'goal':goal,'CT':ct_compare},indent=2))

if __name__=='__main__':main()
