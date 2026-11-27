"""Actual accepted particle fields and explicitly unpassed reproduction gates."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .backend import require_local_budget
from .paper_particle import PaperParticle, ParticleMaterial, ParticleState
from morqex.workspace import Workspace


def build_paper_report(directory, workspace, *, reference=None, output='MORQEX_Ai2022_reproduction.html'):
    import h5py
    from skfem import MeshTri
    require_local_budget()
    ws = workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ws.root.resolve()):
        raise ValueError('Results must be inside the external workspace')
    meta = json.loads((directory/'run.json').read_text())
    solver = Path(__file__).with_name('paper_particle.py')
    if hashlib.sha256(solver.read_bytes()).hexdigest()!=meta['source'][solver.name]:
        raise ValueError('Use the recorded solver revision to reconstruct derived fields')
    cfg=meta['config']; exp=cfg['experiment']
    data=np.load(directory/'mesh.npz'); mesh=MeshTri(data['points'],data['cells'])
    options={k:exp[k] for k in ('geometry','fatigue','stress_feedback','damage_constraint','linearization',
        'boundary_policy','concentration_domain','phase_form','toughness_floor','seed_width_over_length') if k in exp}
    model=PaperParticle(mesh,ParticleMaterial(**cfg['material']),**options)
    frames=[]; initial=None
    paths=sorted(directory.glob('frame_*.h5'))
    # Include the newest committed state even when the save stride was not hit.
    paths.append(directory/'checkpoint.h5')
    for path in paths:
        with h5py.File(path,'r') as h:
            if not h.attrs['complete'] or h.attrs['identity']!=meta['identity']:
                raise ValueError('Frame identity/commit mismatch')
            state=ParticleState(**{k:h[k][:] for k in ('u','c','d','history','fatigue','previous_alpha')},
                                time_s=float(h.attrs['time_s']))
            step=int(h.attrs['step'])
        if frames and step==frames[-1]['step']:continue
        if initial is None:initial=state.d
        f=model.fields(state.u,state.c,state.d,state)
        frames.append(dict(step=step,time_s=state.time_s,cycle=state.time_s/(1.4*3600/exp['c_rate']),
            cracked_fraction=model.cracked_fraction(state.d),
            fields=dict(soc=state.c.tolist(),damage=state.d.tolist(),increment=(state.d-initial).tolist(),
                hydrostatic_MPa=(f['sh']*model.material.young_Pa/1e6).tolist(),
                fatigue_ratio=(model.projection@(state.fatigue/model.threshold)).tolist(),
                toughness_factor=(model.projection@f['factor']).tolist())))
    records=[json.loads(s) for s in (directory/'history.jsonl').read_text().splitlines()]
    if not records or records[-1]['step']!=frames[-1]['step']:
        raise ValueError('History does not match accepted checkpoint')
    ref=json.loads(Path(reference).read_text()) if reference else None
    if ref and (ref.get('schema_version')!=1 or ref.get('case')!=dict(geometry='sphere',radius_m=5e-6,c_rate=5.,crack_fraction=.02)):
        raise ValueError('Unexpected reference schema/case')
    if ref and any(exp.get(k)!=v for k,v in {'geometry':'sphere','c_rate':5.,'crack_fraction':.02}.items()):
        raise ValueError('Reference is specific to Figure 5 sphere/5C/a0=0.02R')
    if ref and model.material.radius_m!=ref['case']['radius_m']:raise ValueError('Reference radius mismatch')
    summary=dict(accepted_steps=records[-1]['step'],cycles=records[-1]['cycle'],
        max_mass_error=max(r['mass_error'] for r in records),
        max_residual=max(r['residual'] for r in records),
        max_soc=max(r['max_soc'] for r in records),min_soc=min(r['min_soc'] for r in records),
        physical_concentration_passed=all(r['physical_concentration_passed'] for r in records),
        full_paper_reproduction_passed=False,full_3d_ct_passed=False,dfn_feedback_passed=False,
        source_identity=meta['identity'],directory=str(directory),reference_is_raw_data=False)
    if ref:
        curve=ref['curves']['cracked_fraction']
        value=float(np.interp(0,curve['cycle'],curve['value']))
        summary['initial_crack_relative_read_off_difference']=abs(frames[0]['cracked_fraction']-value)/value
    payload=dict(summary=summary,metadata=meta,points=mesh.p.T.tolist(),cells=mesh.t.T.tolist(),
                 frames=frames,records=records,reference=ref)
    text=_HTML.replace('__PAYLOAD__',json.dumps(payload,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
    font=ws.path('data','report_font')
    if (font/'NotoSansCJKsc-Regular.otf').exists() and (font/'LICENSE').exists():
        from morqex.visualization.report_font import embed_font
        text=embed_font(text,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
    target=ws.path('reports',output)
    if target.suffix!='.html':raise ValueError('HTML output required')
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    target.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    return dict(html=str(target),**summary)


_HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MORQEX · Ai 2022 独立复现审查</title><style>
*{box-sizing:border-box}body{margin:0;background:#0c1422;color:#e5edf7;font:15px system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:26px}h1{font-size:27px;margin:8px 0}h2{font-size:18px}p{line-height:1.65;color:#b8c7db}a{color:#70daca}.tag{color:#72e1cc;font-size:12px;letter-spacing:2px}.warning{border-left:4px solid #f9ae65;background:#33261e;padding:16px;margin:20px 0;line-height:1.65}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,.panel{background:#142135;border:1px solid #293b51;border-radius:12px;padding:16px}.big{font-size:25px;margin-top:8px}.small{font-size:12px;color:#9fb4ce}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}canvas{display:block;width:100%;height:340px;background:#101b2c;border-radius:8px}.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:20px 0}select,button{background:#223954;color:white;padding:10px;border:1px solid #49617c;border-radius:6px}input[type=range]{flex:1;min-width:150px}.scale{display:flex;justify-content:space-between;font-size:12px;margin:10px 0}.bar{height:7px;background:linear-gradient(90deg,#18346e,#167a9c,#26bcac,#b9d56b,#ffae5f)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}details{padding:14px;background:#142135;border-radius:8px}@media(max-width:650px){main{padding:14px}.cards{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}h1{font-size:22px}canvas{height:300px}}
</style><main><div class="tag">MORQEX / TOP-DAMAGE · INDEPENDENT REPRODUCTION</div>
<h1>扩散—力学—疲劳相场：逐项对照原文</h1>
<p>Ai, Wu & Martínez-Pañeda, JPS 544 (2022), 231805 · <a href="https://doi.org/10.1016/j.jpowsour.2022.231805">原文</a> · 独立有限元，不需要 COMSOL。</p>
<div class="warning" id="warning"></div><div class="cards" id="cards"></div>
<div class="controls"><button id="play">播放接受态</button><input id="time" type="range" min="0" step="1" value="0"><span id="stamp"></span>
<select id="field"><option value="damage">相场 d</option><option value="increment">相对初态 Δd</option><option value="soc">c / cmax</option><option value="hydrostatic_MPa">静水应力 / MPa</option><option value="fatigue_ratio">疲劳 ᾱ / αT</option><option value="toughness_factor">局部 Gd / Gc</option></select></div>
<div class="grid"><section class="panel"><h2>球形颗粒的轴对称子午截面</h2><canvas id="global"></canvas><p class="small">横轴 r/R，纵轴 z/R；这是轴对称求解，不是任意三维 CT 求解。颜色为真实接受态的 P1 单元均值。</p></section>
<section class="panel"><h2>初始环裂纹附近放大</h2><canvas id="tip"></canvas><p class="small">r/R∈[0.94,1]，z/R∈[0,0.06]。d&gt;0.95 是原文的裂纹域指标，不能直接认作开放孔隙。</p></section></div>
<div class="scale"><span id="low"></span><span id="high"></span></div><div class="bar"></div>
<div class="controls"><label><input type="checkbox" id="whole">显示原文完整 300 循环范围</label><span class="small">青色：本次计算；紫色：出版图矢量读数（不是作者原始数据）</span></div>
<div class="grid"><section class="panel"><h2>归一化裂纹域 / %</h2><canvas id="crack"></canvas></section><section class="panel"><h2>体积平均韧度因子</h2><canvas id="toughness"></canvas></section></div>
<details><summary>模型差异、数据来源与未通过门槛</summary><pre id="audit"></pre></details>
<p class="small">所有图来自保存的接受态；保留浓度越界标志，不裁剪浓度、不将短时解标为长期复现通过。初始裂纹匹配不代表疲劳演化匹配。两个时间步之间的差异也不是实验预测误差。</p>
</main><script>const D=__PAYLOAD__;const $=id=>document.getElementById(id);let playing=null;
const S=D.summary;$('warning').textContent=S.physical_concentration_passed?'当前接受态浓度检查通过，但完整原文复现仍未通过：长期曲线、空间/时间收敛和三维 CT 尚需验证。':'未通过物理浓度门槛：存在 c/cmax > 1。当前结果只用于独立方程诊断，不能作为合格电池寿命或完整论文复现结果。';
$('cards').innerHTML=[['实际循环',S.cycles.toFixed(3)],['最大锂守恒缺陷',S.max_mass_error.toExponential(2)],['最高 c/cmax',S.max_soc.toFixed(4)],['完整复现','未通过']].map(a=>'<div class="card"><div class="small">'+a[0]+'</div><div class="big">'+a[1]+'</div></div>').join('');
$('audit').textContent=JSON.stringify({summary:S,differences:D.metadata.differences,config:D.metadata.config,reference:D.reference?{doi:D.reference.doi,kind:D.reference.kind,pdf_sha256:D.reference.pdf_sha256,uncertainty:D.reference.curves.cracked_fraction.read_off_allowance}:null},null,2);
$('time').max=D.frames.length-1;
function setup(id){let c=$(id),r=c.getBoundingClientRect(),s=window.devicePixelRatio||1;c.width=r.width*s;c.height=r.height*s;let x=c.getContext('2d');x.scale(s,s);return[x,r.width,r.height]}
function color(v){let stops=[[24,52,110],[22,122,156],[38,188,172],[185,213,107],[255,174,95]],a=Math.max(0,Math.min(1,v))*4,i=Math.min(3,Math.floor(a)),t=a-i;return 'rgb('+stops[i].map((n,k)=>Math.round(n*(1-t)+stops[i+1][k]*t)).join(',')+')'}
function field(id,values,lo,hi,box){let [ctx,w,h]=setup(id),sz=Math.min(w-70,h-50),ox=(w-sz)/2,oy=h-28;let X=x=>ox+(x-box[0])/(box[1]-box[0])*sz,Y=y=>oy-(y-box[2])/(box[3]-box[2])*sz;ctx.save();ctx.beginPath();ctx.rect(ox,oy-sz,sz,sz);ctx.clip();for(let t of D.cells){let p=t.map(i=>D.points[i]);if(p.every(v=>v[0]<box[0])||p.every(v=>v[0]>box[1])||p.every(v=>v[1]>box[3]))continue;let v=t.reduce((a,i)=>a+values[i],0)/3;ctx.fillStyle=color((v-lo)/(hi-lo||1));ctx.beginPath();ctx.moveTo(X(p[0][0]),Y(p[0][1]));ctx.lineTo(X(p[1][0]),Y(p[1][1]));ctx.lineTo(X(p[2][0]),Y(p[2][1]));ctx.closePath();ctx.fill()}ctx.restore();ctx.strokeStyle='#6f829c';ctx.strokeRect(ox,oy-sz,sz,sz);ctx.fillStyle='#a9bdd5';ctx.font='12px MORQEXReportCJK,system-ui';ctx.fillText(box[0].toFixed(2),ox,oy+19);ctx.fillText(box[1].toFixed(2)+' r/R',ox+sz-35,oy+19);ctx.fillText(box[3].toFixed(2),ox-35,oy-sz+8);ctx.fillText('z/R',ox-35,oy-sz+25)}
function plot(id,key,mult){let [ctx,w,h]=setup(id),left=62,top=18,r=w-18,b=h-35,xmax=$('whole').checked?300:Math.max(S.cycles,.1);let actual=D.frames.map(f=>[f.cycle,f.cracked_fraction*mult]);if(key==='mean_toughness_factor')actual=[[0,1],...D.records.map(v=>[v.cycle,v[key]])];let ref=D.reference?D.reference.curves[key]:null;let rr=ref?ref.cycle.map((v,i)=>[v,ref.value[i]*mult]).filter(v=>v[0]<=xmax):[];if(ref){let j=ref.cycle.findIndex(v=>v>xmax);if(j>0&&ref.cycle[j-1]<xmax){let t=(xmax-ref.cycle[j-1])/(ref.cycle[j]-ref.cycle[j-1]);rr.push([xmax,((1-t)*ref.value[j-1]+t*ref.value[j])*mult])}}let ys=[...actual,...rr].map(v=>v[1]);let ymin=key==='cracked_fraction'?0:Math.min(...ys),ymax=Math.max(...ys);if(ymax-ymin<1e-5){ymin-=.00001;ymax+=.00001}if(key==='mean_toughness_factor'){ymin=.5;ymax=1.01}let span=ymax-ymin;ymax+=span*.08;if(key!=='cracked_fraction')ymin-=span*.08;let X=v=>left+v/xmax*(r-left),Y=v=>b-(v-ymin)/(ymax-ymin)*(b-top);ctx.font='11px MORQEXReportCJK,system-ui';for(let i=0;i<5;i++){let y=top+(b-top)*i/4;ctx.strokeStyle='#293a50';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(r,y);ctx.stroke();ctx.fillStyle='#9fb4ce';ctx.fillText((ymax-(ymax-ymin)*i/4).toPrecision(3),2,y+4);ctx.fillText((xmax*i/4).toFixed(xmax<10?1:0),X(xmax*i/4)-8,b+18)}for(let [points,col] of [[rr,'#e88ee4'],[actual,'#60e5d0']]){ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();points.forEach((v,i)=>{if(i)ctx.lineTo(X(v[0]),Y(v[1]));else ctx.moveTo(X(v[0]),Y(v[1]))});ctx.stroke()}ctx.fillText('循环次数',r-52,h-2)}
function draw(){let f=D.frames[+$('time').value],key=$('field').value,v=f.fields[key],lo=Infinity,hi=-Infinity;for(let a of D.frames)for(let x of a.fields[key]){lo=Math.min(lo,x);hi=Math.max(hi,x)}if(key==='damage'||key==='toughness_factor'){lo=0;hi=1}if(key==='increment')lo=0;$('stamp').textContent='N='+f.cycle.toFixed(3)+' · t='+f.time_s.toFixed(1)+' s';$('low').textContent='固定跨帧色标 '+lo.toPrecision(5);$('high').textContent=hi.toPrecision(5);field('global',v,lo,hi,[0,1,0,1]);field('tip',v,lo,hi,[.94,1,0,.06]);plot('crack','cracked_fraction',100);plot('toughness','mean_toughness_factor',1)}
$('time').oninput=draw;$('field').onchange=draw;$('whole').onchange=draw;$('play').onclick=()=>{if(playing){clearInterval(playing);playing=null;$('play').textContent='播放接受态'}else{playing=setInterval(()=>{$('time').value=(+$('time').value+1)%D.frames.length;draw()},1200);$('play').textContent='暂停'}};window.onresize=draw;draw();if(document.fonts)document.fonts.ready.then(draw);</script></html>'''
