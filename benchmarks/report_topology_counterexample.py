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
    font=ws.path('data/report_font')
    if (font/'NotoSansCJKsc-Regular.otf').exists():
        from morqex.visualization.report_font import embed_font
        document=embed_font(document,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
    (output/'index.html').write_text(document);print(json.dumps({'html':str(output/'index.html')}))

if __name__=='__main__':main()
