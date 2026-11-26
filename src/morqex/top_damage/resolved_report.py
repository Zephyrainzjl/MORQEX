"""Self-contained actual full-lattice electrochemistry/finite-fatigue report."""
from pathlib import Path
import base64,json
import numpy as np
import h5py
from morqex import Workspace
from .polycrystal import PolycrystalData
from .morphology import measure_geometry
from ..visualization.report_font import embed_font


def build_resolved_report(directory,workspace,*,output='MORQEX_resolved_3D_phase_field.html'):
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    directory=ws.path(str(directory));data=PolycrystalData.load_hdf5(directory/'initial.h5');shape=data.structure.labels.shape
    files=sorted(directory.glob('mechanics_*.h5'))
    if not files:raise ValueError('Actual accepted finite-phase-field states required')
    names={'solid':'固相浓度 / mol m⁻³','liquid':'电解液浓度 / mol m⁻³','potential':'电势 / V','principal':'最大主应力 / MPa','damage':'连续损伤 d','damage_increment':'相对初态损伤增量 Δd','fatigue':'疲劳历史 / J m⁻³','mu':'力学化学势 / J mol⁻¹'}
    with h5py.File(files[0]) as f:initial_damage=f['damage'][:]
    frames=[];bounds={k:[np.inf,-np.inf] for k in names}
    for path in files:
        with h5py.File(path) as f:
            ids=f['voxel_ids'][:];cells=f['tetrahedra'][:];counts=np.bincount(ids,minlength=np.prod(shape))
            def avg(a):return (np.bincount(ids,weights=a,minlength=np.prod(shape))/np.maximum(counts,1)).reshape(shape)
            stress=f['cauchy_stress_Pa'][:]
            tensor=np.stack([avg(stress[:,i,j]) for i in range(3) for j in range(3)],axis=-1).reshape(shape+(3,3))
            c=f['voxel_concentration_mol_m3'][:]
            values={'solid':c,'liquid':c,'potential':f['voxel_potential_V'][:],'principal':np.linalg.eigvalsh(tensor)[...,-1]/1e6,'damage':avg(f['damage'][:][cells].mean(axis=1)),'damage_increment':avg((f['damage'][:]-initial_damage)[cells].mean(axis=1)),'fatigue':avg(f['fatigue_J_m3'][:]),'mu':f['voxel_mechanical_potential_J_mol'][:]}
            encoded={}
            for key,a in values.items():
                mask=data.structure.labels==0 if key=='liquid' else (np.ones(shape,bool) if key=='potential' else data.structure.labels==1)
                if not np.isfinite(a).all():raise ValueError('Nonfinite report field')
                bounds[key]=[min(bounds[key][0],float(a[mask].min())),max(bounds[key][1],float(a[mask].max()))]
                encoded[key]=base64.b64encode(np.asarray(a,dtype='<f4').tobytes()).decode()
            frames.append({'time':float(f.attrs['time_s']),'fields':encoded,'residual':float(f.attrs['equilibrium_residual']),'kkt':float(f.attrs['damage_kkt']),'transfer_error':float(f.attrs['inventory_transfer_relative_error']),'tetrahedra':len(cells)})
    curve=json.loads((directory/'probe.json').read_text())
    payload={'shape':list(shape),'spacing':(np.asarray(data.structure.spacing_zyx_m)[::-1]*1e6).tolist(),'origin':[0,0,0],'phase':data.structure.labels.ravel().tolist(),'particle':data.particle_ids.ravel().tolist(),'grain':data.grain_ids.ravel().tolist(),'kinds':data.structure.phase_kinds,'metrics':measure_geometry(data),'frames':frames,'bounds':bounds,'names':names,'curve':curve}
    page=Path(__file__).with_name('morphology_view.html').read_text().replace('__MORQEX_DATA__',json.dumps(payload,separators=(',',':')).replace('<','\\u003c'))
    page=page.replace('少量大颗粒，看清内部晶粒与 CBD','全三维浓度、电势 → 三维有限变形与疲劳相场')
    page=page.replace('FULL VOXEL GEOMETRY','RESOLVED FIELDS · QUALIFICATION REFERENCE')
    start=page.index('<div class="note">');end=page.index('</div>',start)+6
    page=page[:start]+'''<div class="note">真实求解：32³ 正极、8 个大颗粒、32 个晶粒、显式惰性 CBD；外接隔膜和理想锂负极，总计 45056 体素。电化学为 NMC532 参数卡，浓度逐体素传给 127554 Tet4 有限变形/AT2，相场历史保持不可逆。本页只有短时三维参考，不是完整循环或长期寿命：材料疲劳参数未经实验标定，微观力学化学势尚未反馈原生三维电化学，连续损伤尚未转换为物理开孔裂纹。三维几何没有伪造裂纹删除。总损伤包含预置裂纹；请选“损伤增量 Δd”区别初始缺陷与计算产生的增长。</div>'''+page[end:]
    start=page.index('<p><a href="initial.h5">');end=page.index('</p>',start)+4
    page=page[:start]+f'<p>完整 FP64 数据：{directory}。本页 FP32 仅用于显示。<a href="MORQEX_TOP_DAMAGE_2S_finite_cycles.html">已有 DFN 完整充放电/有限变形案例</a> · <a href="region_finite_32_toml.html">六节点区域 DFN 双向探针</a>。这些是不同验收案例，尚未组成长期物理拓扑闭环。</p>'+page[end:]
    actual_metrics=payload['metrics']
    page=page.replace('32³ 正极、8 个大颗粒、32 个晶粒', '×'.join(map(str,shape))+' 正极、'+str(actual_metrics['particle_count'])+' 个颗粒、'+str(actual_metrics['grain_count'])+' 个晶粒')
    page=page.replace('127554 Tet4',str(frames[0]['tetrahedra'])+' Tet4')
    with h5py.File(directory/'fields_00000.h5') as f:total_voxels=int(np.prod(f['Concentration'].shape))
    page=page.replace('45056 体素',str(total_voxels)+' 体素')
    options=''.join(f'<option value="{k}">{v}</option>' for k,v in names.items())
    controls='<div class="controls"><label>时间 <select id="frame"></select></label><label>物理场 <select id="field">'+options+'<option value="geometry">几何编号</option></select></label><span id="scale"></span></div><p id="verification"></p>'
    page=page.replace('<div class="views">',controls+'<div class="views">',1).replace("kind(i)==='pore'?'#213349':color(i)",'color(i)')
    page=page.replace('</style>','.panel{min-width:0}p,footer{overflow-wrap:anywhere}select{max-width:100%}</style>')
    script='''<script>
let physical={};const initialColor=color;
function decode(s){let b=Uint8Array.from(atob(s),x=>x.charCodeAt(0));return new Float32Array(b.buffer)}
color=function(i,shade=1){const key=$('field').value;if(key==='geometry')return kind(i)==='pore'?'#213349':initialColor(i,shade);if(key==='liquid'?kind(i)!=='pore':key!=='potential'&&kind(i)!=='active')return kind(i)==='pore'?'#213349':`hsl(209 13% ${54*shade}%)`;let [a,b]=DATA.bounds[key],v=Math.max(0,Math.min(1,(physical[key][i]-a)/(b-a||1)));return `hsl(${235-235*v} 80% ${(36+20*v)*shade}%)`};
$('frame').innerHTML=DATA.frames.map((v,i)=>`<option value="${i}">${v.time} s</option>`).join('');
function showFrame(){const n=+$('frame').value,f=DATA.frames[n],k=$('field').value;physical=Object.fromEntries(Object.entries(f.fields).map(([k,v])=>[k,decode(v)]));$('scale').textContent=k==='geometry'?'材料初始拓扑':`所有时刻统一色标 ${DATA.bounds[k][0].toPrecision(5)} … ${DATA.bounds[k][1].toPrecision(5)}`;$('verification').textContent=`t = ${f.time} s；${f.tetrahedra} Tet4；力学残差 ${f.residual.toExponential(2)}；相场 KKT ${f.kkt.toExponential(2)}；体素→有限元库存误差 ${f.transfer_error.toExponential(2)}。`;mesh();draw();document.body.dataset.frame=n;document.body.dataset.field=k;}
$('frame').onchange=showFrame;$('field').onchange=showFrame;$('cut').value=Math.round(.75*nz);$('plane').value=Math.round(.75*nz)-1;showFrame();if(document.fonts)document.fonts.ready.then(showFrame);
const oldInspect=$('slice').onclick;$('slice').onclick=e=>{oldInspect(e);const r=$('slice').getBoundingClientRect(),{s,ox,oy}=sliceBounds,x=Math.floor((e.clientX-r.left-ox)/(hx*s)),y=ny-1-Math.floor((e.clientY-r.top-oy)/(hy*s)),z=+$('plane').value,k=$('field').value;if(x>=0&&x<nx&&y>=0&&y<ny&&k!=='geometry')$('inspect').textContent+=' | '+DATA.names[k]+' = '+physical[k][index(x,y,z)].toPrecision(7)};
</script>'''
    t=np.asarray(curve['time_s']);v=np.asarray(curve['voltage_V']);x=30+540*(t-t.min())/max(np.ptp(t),1.);y=170-140*(v-v.min())/max(np.ptp(v),1e-9)
    points=' '.join(f'{a:.3f},{b:.3f}' for a,b in zip(x,y))
    chart=f'<section class="panel"><h2>本次三维求解电压：{t[0]:g}–{t[-1]:g} s（短时段）</h2><svg viewBox="0 0 600 200" style="width:100%;max-height:220px"><polyline points="{points}" fill="none" stroke="#54d7c7" stroke-width="3"/><text x="30" y="22" fill="#b9cadc">{v.max():.7f} V</text><text x="30" y="195" fill="#b9cadc">{v.min():.7f} V · 时间向右</text></svg></section>'
    page=page.replace('<table>',chart+'<table>',1).replace('</html>',script+'</html>')
    font=ws.path('data/report_font');page=embed_font(page,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
    target=ws.path('reports',output);target.write_text(page)
    return {'report':str(target),'frames':len(frames),'full_voxels':int(np.prod(shape)),'lifetime_qualified':False}


def export_resolved_history(directory,workspace):
    """Lossless full-lattice VTI plus temporal FEM XDMF; damage is not a void."""
    import xml.etree.ElementTree as ET
    from .dense_export import write_field_vti
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace);directory=ws.path(str(directory))
    data=PolycrystalData.load_hdf5(directory/'initial.h5');shape=data.structure.labels.shape
    xr=ET.Element('Xdmf',Version='3.0');temporal=ET.SubElement(ET.SubElement(xr,'Domain'),'Grid',Name='Resolved electrochemical finite-fatigue states',GridType='Collection',CollectionType='Temporal')
    vr=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian');collection=ET.SubElement(vr,'Collection');records=[]
    for path in sorted(directory.glob('mechanics_*.h5')):
        with h5py.File(path) as f:
            time=float(f.attrs['time_s']);cells=f['tetrahedra'][:];ids=f['voxel_ids'][:];counts=np.bincount(ids,minlength=np.prod(shape))
            def avg(a):return (np.bincount(ids,weights=a,minlength=np.prod(shape))/np.maximum(counts,1)).reshape(shape)
            stress=f['cauchy_stress_Pa'][:];u=f['displacement_m'][:];tensor=np.stack([avg(stress[:,i,j]) for i in range(3) for j in range(3)],axis=-1)
            fields={'phase_labels':data.structure.labels,'particle_ids':data.particle_ids,'grain_ids':data.grain_ids,'concentration_mol_m3':f['voxel_concentration_mol_m3'][:],'potential_V':f['voxel_potential_V'][:],'mechanical_chemical_potential_J_mol':f['voxel_mechanical_potential_J_mol'][:],'damage':avg(f['damage'][:][cells].mean(axis=1)),'fatigue_J_m3':avg(f['fatigue_J_m3'][:]),'cauchy_stress_tensor_Pa':tensor,'displacement_m':np.stack([avg(u[cells,i].mean(axis=1)) for i in range(3)],axis=-1)}
            vti=path.with_suffix('.vti');write_field_vti(vti,shape,data.structure.spacing_zyx_m,data.structure.origin_zyx_m,fields)
            ET.SubElement(collection,'DataSet',timestep=str(time),group='',part='0',file=vti.name)
            grid=ET.SubElement(temporal,'Grid',Name=path.stem,GridType='Uniform');ET.SubElement(grid,'Time',Value=str(time))
            def item(parent,key):
                a=f[key];node=ET.SubElement(parent,'DataItem',Dimensions=' '.join(map(str,a.shape)),NumberType='Int' if np.issubdtype(a.dtype,np.integer) else 'Float',Precision='8',Format='HDF');node.text=path.name+':/'+key
            item(ET.SubElement(grid,'Topology',TopologyType='Tetrahedron',NumberOfElements=str(len(cells))),'tetrahedra')
            item(ET.SubElement(grid,'Geometry',GeometryType='XYZ'),'points_m')
            for key,kind,centre in [('displacement_m','Vector','Node'),('damage','Scalar','Node'),('cauchy_stress_Pa','Tensor','Cell'),('fatigue_J_m3','Scalar','Cell')]:item(ET.SubElement(grid,'Attribute',Name=key,AttributeType=kind,Center=centre),key)
            records.append({'time_s':time,'hdf5':path.name,'vti':vti.name,'full_voxels':int(np.prod(shape))})
    if not records:raise ValueError('No accepted field history')
    ET.ElementTree(xr).write(directory/'resolved_history.xdmf',encoding='utf-8',xml_declaration=True)
    ET.ElementTree(vr).write(directory/'resolved_history.pvd',encoding='utf-8',xml_declaration=True)
    (directory/'resolved_exports.json').write_text(json.dumps(records,indent=2));return records
