"""Portable scientific HTML exports. Large figures belong in a Workspace."""
from pathlib import Path
import html
import json
import numpy as np

def structure_html(volume,path,title='GEM v1 三维微结构',phase_names=None,max_triangles=120000,reference=None):
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes
    colors=['#48b9df','#f7ae45','#b39aff','#6bdbaf','#ff7f8a']
    fig=go.Figure()
    phase_names=phase_names or {0:'孔隙',128:'活性颗粒',255:'CBD / 黏结相'}
    for i,phase in enumerate(np.unique(volume.data)):
        mask=np.pad((volume.data==phase).astype(np.float32),1)
        if not mask.any():continue
        step=1
        vertices,faces,_,_=marching_cubes(mask,.5,spacing=volume.spacing,step_size=step)
        if len(faces)>max_triangles:
            # Visualization simplification only; original voxel data remain untouched.
            step=2
            vertices,faces,_,_=marching_cubes(mask,.5,spacing=volume.spacing,step_size=step)
        vertices=(vertices-np.asarray(volume.spacing)+volume.origin)[:,::-1]
        label=phase_names.get(int(phase),f'相位 {phase}')
        fraction=float(np.mean(volume.data==phase))
        fig.add_trace(go.Mesh3d(x=vertices[:,0].tolist(),y=vertices[:,1].tolist(),z=vertices[:,2].tolist(),
            i=faces[:,0].tolist(),j=faces[:,1].tolist(),k=faces[:,2].tolist(),name=f'{label} ({fraction:.1%})',
            color=colors[i%len(colors)],opacity=.78 if i else .13,
            visible='legendonly' if phase==0 else True,flatshading=False,
            lighting=dict(ambient=.5,diffuse=.8,specular=.25,roughness=.5),
            hovertemplate=html.escape(label)+'<br>x=%{x:.3g}<br>y=%{y:.3g}<br>z=%{z:.3g}<extra></extra>'))
    fig.update_layout(template='plotly_dark',paper_bgcolor='#101827',plot_bgcolor='#101827',
        title=dict(text=title,x=.04),height=800,margin=dict(l=0,r=0,t=70,b=0),
        scene=dict(aspectmode='data',xaxis_title='x',yaxis_title='y',zaxis_title='z',
                   camera=dict(eye=dict(x=1.5,y=1.5,z=1.2))),
        legend=dict(x=.02,y=.95,bgcolor='rgba(10,18,30,.7)'),
        updatemenus=[dict(type='buttons',direction='right',x=.02,y=1.02,buttons=[
            dict(label='所有相位',method='restyle',args=[{'visible':True}]),
            dict(label='仅活性颗粒',method='restyle',args=[{'visible':[p==128 for p in np.unique(volume.data)]}]),
            dict(label='仅黏结相',method='restyle',args=[{'visible':[p==255 for p in np.unique(volume.data)]}]),
        ])])
    note='拖动旋转，滚轮缩放；单击图例切换相位，双击图例仅保留该相。孔隙默认隐藏以便查看内部。'
    if volume.metadata.get('material_prior') or volume.metadata.get('quota_postprocessing'):
        note+=' 此结果包含显式组成或材料先验约束，不代表未经约束的网络输出。'
    metadata=html.escape(json.dumps(volume.metadata,ensure_ascii=False,indent=2))
    reference_block=''
    if reference is not None:
        import io,base64
        from PIL import Image
        if isinstance(reference,(str,Path)):
            import tifffile
            original=tifffile.imread(reference)
        else:original=np.asarray(reference)
        buffer=io.BytesIO();Image.fromarray(original.astype(np.uint8)).save(buffer,format='PNG')
        encoded=base64.b64encode(buffer.getvalue()).decode()
        reference_block=f'<section style="display:flex;gap:22px;align-items:center;margin:22px 32px;padding:18px;background:#17263b;border-radius:12px"><img alt="原始 NMC 二维训练图" src="data:image/png;base64,{encoded}" style="width:250px;max-width:45%;image-rendering:pixelated"><div><b>原始 NMC 二维训练图</b><p>尺寸 {original.shape}；分割标签保留。</p><p>下方是模型生成的三维结构，不是对原图的堆叠或插值放大。</p></div></section>'
    content=fig.to_html(full_html=False,include_plotlyjs=True,config={'scrollZoom':True,'displaylogo':False})
    cut_script=r'''<script>
const gemGraph=document.querySelector('.plotly-graph-div');
const gemOriginal=gemGraph.data.map(t=>({i:[...t.i],j:[...t.j],k:[...t.k],z:[...t.z]}));
let gemCutScheduled=false;
function gemApplyCut(){gemCutScheduled=false;const percent=Number(document.getElementById('gemCut').value);document.getElementById('gemCutLabel').textContent=percent+'%';gemOriginal.forEach((t,index)=>{let min=Infinity,max=-Infinity;for(let z of t.z){min=Math.min(min,z);max=Math.max(max,z)}let threshold=min+(max-min)*percent/100;let i=[],j=[],k=[];for(let f=0;f<t.i.length;f++){if((t.z[t.i[f]]+t.z[t.j[f]]+t.z[t.k[f]])/3<=threshold){i.push(t.i[f]);j.push(t.j[f]);k.push(t.k[f])}}Plotly.restyle(gemGraph,{i:[i],j:[j],k:[k]},[index])})}
document.getElementById('gemCut').addEventListener('input',()=>{if(!gemCutScheduled){gemCutScheduled=true;requestAnimationFrame(gemApplyCut)}});
</script>'''
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{margin:0;background:#101827;color:#dce6f4;font:15px system-ui,"Microsoft YaHei"}}header{{padding:20px 32px 0}}p{{color:#9eb0c8}}details{{margin:20px 32px}}pre{{white-space:pre-wrap;font-size:12px}}</style>
<header><b>GEM · MICROSTRUCTURE EXPLORER</b><p>{note}</p><p>体素尺寸：{volume.data.shape} · 原始体素未因展示简化而改变</p></header>{reference_block}<header><label>Z 方向剖切：<input id="gemCut" type="range" min="5" max="100" value="100" style="width:260px;vertical-align:middle"> <span id="gemCutLabel">100%</span></label></header>{content}<details><summary>生成与约束元数据</summary><pre>{metadata}</pre></details>{cut_script}</html>'''
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(page,encoding='utf8')
    return path
