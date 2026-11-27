"""Offline report of prescribed dry-contact topology and actual DFN results."""
import os,json,html
from pathlib import Path
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from morqex.workspace import Workspace


def main():
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);root=ws.path('results/top_damage/topology_counterexample/v1')
    data=json.loads(ws.path('reports/topology_counterexample_v1.json').read_text())
    work=json.loads(ws.path('reports/reactive_graph_v1.json').read_text())
    output=ws.path('reports/topology_study');output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(10,4),layout='constrained')
    for k,mode in enumerate(('redundant','isolate'),1):
        with h5py.File(root/f'{mode}_8C/phase_005.h5') as f:
            phase=f['phase_labels'][:];reach=f['electron_accessible'][:]
        filled=(phase>0).transpose(2,1,0);colors=np.empty(filled.shape+(4,))
        colors[:]=matplotlib.colors.to_rgba('#729391')
        colors[(phase==1).transpose(2,1,0)]=matplotlib.colors.to_rgba('#e5a24c')
        colors[((phase==1)&~reach).transpose(2,1,0)]=matplotlib.colors.to_rgba('#874a9c')
        ax=fig.add_subplot(1,2,k,projection='3d');ax.voxels(filled,facecolors=colors,edgecolor='#344052',linewidth=.3)
        patch=[[5,0,0],[5,1,0],[5,1,1],[5,0,1]] if mode=='redundant' else [[1,2,2],[2,2,2],[2,3,2],[1,3,2]]
        ax.add_collection3d(Poly3DCollection([patch],facecolor='#dc3037',edgecolor='#ff0000',linewidth=2,alpha=1))
        ax.set(xlabel='x [voxel]',ylabel='y [voxel]',zlabel='z [voxel]',title='Redundant CBD edge cut' if mode=='redundant' else 'Sole particle contact cut',xlim=(0,10),ylim=(0,6),zlim=(0,6))
        ax.view_init(elev=27,azim=-70);ax.set_box_aspect((10,6,6))
    fig.savefig(output/'topology.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    colors={'intact':'#1e5e82','redundant':'#398475','isolate':'#b55752'}
    for mode in colors:
        rows=[d for d in data if d['mode']==mode]
        axes[0].plot([d['C_rate'] for d in rows],[100*d['second_to_first_capacity_ratio'] for d in rows],'o-',color=colors[mode],label=mode)
        curve=np.loadtxt(root/f'{mode}_8C/curve.csv',delimiter=',',skiprows=1)
        a=curve[(curve[:,0]==2)&(curve[:,3]>0)]
        keep=np.r_[True,np.diff(a[:,1])>0];a=a[keep]
        axes[1].plot((a[:,4]-a[0,4])*1000,a[:,2],color=colors[mode],label=mode)
    axes[0].set(xlabel='Rate [C]',ylabel='Second / first discharge [%]',ylim=(0,105),title='Actual current-integrated capacity')
    axes[1].set(xlabel='Second discharge capacity [mAh]',ylabel='Voltage [V]',title='8C discharge after prescribed contact cut')
    for ax in axes:ax.legend();ax.grid(alpha=.15)
    fig.savefig(output/'response.svg');fig.savefig(output/'response.png',dpi=180);plt.close(fig)
    rows=[]
    for d in data:
        vals=[d['mode'],d['C_rate'],d['prescribed_dry_crack_area_m2'],d['accessible_active_voxels_after'],
            f"{d['electronic_conductivity_after_S_m']:.6f}",f"{100*d['second_to_first_capacity_ratio']:.6f}",f"{d['trapped_inventory_relative_drift']:.2e}"]
        rows.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in vals)+'</tr>')
    notes='''这是预设几何失效的独立验证，不是相场自动生成的裂纹，也不是材料寿命预测。两个失效工况都移除一个 1 μm² 的干式接触面，固体体积和锂库存保留；没有把损伤区伪造为开放孔隙。模型为两个小颗粒及 CBD/孔隙的稀疏数值结构，不代表真实电极的体积分数。'''
    document='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MORQEX 拓扑与反应端口验证</title><style>body{font:15px/1.7 system-ui;background:#edf3f6;color:#163044;margin:0}main{max-width:1100px;margin:auto;padding:28px}section{background:#fff;padding:24px;border-radius:12px;margin:20px 0}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dde5eb;text-align:left}.note{border-left:5px solid #c28225}pre{font-size:12px;overflow:auto}a{color:#176e86}</style><main><h1>相同接触失效面积，为什么容量不同？</h1><section class="note">'''+notes+'''</section><section><img src="topology.png" alt="真实体素和失效接触位置"><p>黄色：电子可达活性材料；紫色：电子失联活性材料；绿灰色：CBD；红色：移除的实际接触面。体素边长 1 μm。</p></section><section><img src="response.svg" alt="DFN容量和放电曲线"><p>失联工况的等效通过厚度电导率与完整结构相同，但一半活性材料失去集流体连接。冗余路径工况的电导率反而下降，容量变化却很小。仅用平均电导率或失效面积不能区分这两种情况。</p></section><section><h2>实际求解数据</h2><table><tr>'''+''.join('<th>'+s+'</th>' for s in ['工况','倍率 C','干裂面面积 m²','可达活性体素','电子导率 S/m','第二/首次容量 %','失联锂库存漂移'])+'''</tr>'''+''.join(rows)+'''</table></section><section><h2>进一步核对非线性反应端口</h2><p>同一几何还求解了准静态电子/离子网络与 Butler–Volmer 反应，电压由指定总电流反求；保留所有反应顶点，仅消去无源内部顶点。全图和 Schur 端口解单独对照，HDF5 保存的是工作电势。该验证给定 SOC 与电解液浓度，尚不包含浓溶液扩散电势和时间演化，也未默认替代循环 DFN。</p><details><summary>完整数值与性能</summary><pre>'''+html.escape(json.dumps(work,indent=2))+'''</pre></details><p>电子浮动颗粒仍可有局部双极反应，但其净电子电流必须为零；全停反应的失联近似与这个更细的模型不能混为一谈。</p></section><section><a href="../topology_counterexample_v1.json">DFN对照 JSON</a> · <a href="../reactive_graph_v1.json">反应端口 JSON</a> · <a href="../network_study/index.html">疲劳循环总览</a></section></main>'''
    bounds=ws.path('reports/reactive_voltage_bounds.json')
    if bounds.exists():
        result=json.loads(bounds.read_text())
        rows=result['records']
        evidence='<section><h2>非线性电压事件检查</h2><p>12组固定反应条件的边导率扰动均通过独立全图对照。最大实际电压误差 '+f"{max(r['actual_voltage_error_V'] for r in rows)*1000:.6f}"+' mV；最大解析上界 '+f"{max(r['voltage_bound_V'] for r in rows)*1000:.6f}"+' mV。上界较保守，不代表长期容量误差已获保证。</p><a href="../reactive_voltage_bounds.json">完整上界、实际误差及计算时间</a></section>'
        document=document.replace('</main>',evidence+'</main>')
    changed=ws.path('reports/reactive_state_voltage_bounds_v2.json')
    if changed.exists():
        result=json.loads(changed.read_text());records=result['records'];x=np.arange(len(records))
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,error,bound,title in [(axes[0],'actual_voltage_error_V','voltage_bound_V','Cached old voltage: state changes included'),
                (axes[1],'rebound_voltage_error_V','rebound_voltage_bound_V','New reaction solve: passive factorization reused')]:
            ax.semilogy(x,[max(r[error],1e-14) for r in records],'o',ms=3,label='Actual error')
            ax.semilogy(x,[max(r[bound],1e-14) for r in records],'+',label='Bound')
            ax.axhline(.001,color='#a96528',ls='--',label='1 mV event threshold')
            ax.set(xlabel='Controlled case index',ylabel='Voltage [V]',title=title);ax.legend(fontsize=8);ax.grid(alpha=.15)
        fig.savefig(output/'changed_state_bounds.svg');plt.close(fig)
        accepted=sum(r['rebound_below_1mV_event_tolerance'] for r in records)
        text='<section><h2>反应状态变化与守恒复用</h2><p>24组NMC811/NMC532/LFP对照覆盖完整/失联结构、0.5C/8C、小变化与电流反转。OCP、交换电流、界面覆盖和温度显式更新；全部与新鲜完整图比较。复用被动分解后仍重新求解BV电流，保持总电流约束；'+str(accepted)+'/24组复用后的电压界低于1mV。</p><p>这是已知化学状态下的准静态欧姆/BV问题，尚未作为长期DFN浓溶液求解器的替代。LFP舍入停滞通过64位有效尾数的扩展精度电位/残差与FP64切线修正，未放宽Newton容差；不是IEEE binary128，也不是GPU加速。</p><img src="changed_state_bounds.svg" alt="状态变化和重新求解反应后的误差与上界"><a href="../reactive_state_voltage_bounds_v2.json">全部状态、误差、守恒和计时</a></section>'
        document=document.replace('</main>',text+'</main>')
    events=ws.path('reports/reactive_event_sequences_v1.json')
    if events.exists():
        result=json.loads(events.read_text());records=result['records']
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for host,colour in [('NMC811','#1e5e82'),('NMC532','#398475'),('LFP','#b55752')]:
            chosen=[r for r in records if r['host']==host]
            x=np.arange(len(chosen))
            axes[0].semilogy(x,[max(r['voltage_error_V'],1e-15) for r in chosen],'.',color=colour,label=host)
            axes[1].semilogy(x,[max(r['relative_face_error_L1'],1e-15) for r in chosen],'.',color=colour,label=host)
        axes[0].axhline(.001,color='#a96528',ls='--');axes[1].axhline(.01,color='#a96528',ls='--')
        axes[0].set(xlabel='Supplied-state query index per material',ylabel='Voltage error [V]')
        axes[1].set(xlabel='Supplied-state query index per material',ylabel='Relative face-current L1 error')
        for ax in axes:ax.legend();ax.grid(alpha=.15)
        fig.savefig(output/'event_sequences.svg');plt.close(fig)
        rebuilds=sum(r['rebuilt'] for r in records)
        elapsed=sum(r['cache_update_seconds'] for r in records);reference=sum(r['fresh_full_seconds'] for r in records)
        text='<section><h2>自动事件更新：电压与局部锂通量同时约束</h2><p>'+str(len(records))+'次状态查询全部通过独立完整图比较：三种合成结构、NMC811/NMC532/LFP、0.1C/2C/8C，包含电流反转、导率改变、接触断开/恢复和反应支持集变化。'+str(rebuilds)+'次重建、'+str(len(records)-rebuilds)+'次复用；每次均重新求解反应电流。重建失败不会提交新状态。</p><p>该查询序列总成本 '+f'{elapsed:.3f}'+' s，对照 '+f'{reference:.3f}'+' s（包含本序列初次建图与全部重建，约 '+f'{reference/elapsed:.2f}'+' 倍）。这不是连续充放电或长期寿命加速；给定浓度、温度和几何变化没有由相场自动生成。浮点数评估的解析界不等于区间算术认证。</p><img src="event_sequences.svg" alt="所有查询的电压和反应面电流误差"><a href="../reactive_event_sequences_v1.json">逐查询事件、误差上界、守恒与计时</a></section>'
        document=document.replace('</main>',text+'</main>')
    font=ws.path('data/report_font')
    if (font/'NotoSansCJKsc-Regular.otf').exists():
        from morqex.visualization.report_font import embed_font
        document=embed_font(document,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
    (output/'index.html').write_text(document);print(json.dumps({'html':str(output/'index.html')}))

if __name__=='__main__':main()
