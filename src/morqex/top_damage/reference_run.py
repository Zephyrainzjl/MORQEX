"""Checkpointed 3D notched-specimen qualification, not a DFN lifetime run."""
from dataclasses import dataclass
from html import escape
import hashlib
import json
from pathlib import Path
import time
import tomllib
import numpy as np
from morqex.workspace import Workspace
from morqex.mechanics import TetMesh
from morqex.microstructure import Volume
from .backend import NativeConfig, MFEMScalarBackend
from .phase_field import AT2Damage, FatigueHistory
from .coupled_reference import SplitElasticity, staggered_step


@dataclass(frozen=True)
class ReferenceConfig:
    name: str
    shape_zyx: tuple
    spacing_m: float
    young_Pa: float
    poisson: float
    fracture_toughness_J_m2: float
    length_m: float
    residual_stiffness: float
    maximum_yy_strain: float
    notch_fraction: float
    cycles: int
    increments_per_cycle: int
    fatigue: bool = True
    stagger_tolerance: float = 1e-8
    max_stagger_iterations: int = 100
    cache_forms: bool = False

    def __post_init__(self):
        if not self.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in self.name):raise ValueError('Invalid run name')
        if len(self.shape_zyx)!=3 or any(type(n) is not int or n<2 for n in self.shape_zyx):raise ValueError('Three mesh sizes >=2 required')
        if self.shape_zyx[1]%2:raise ValueError('Even Y size required to resolve the initial notch plane')
        positive=[self.spacing_m,self.young_Pa,self.fracture_toughness_J_m2,self.length_m,self.maximum_yy_strain,self.stagger_tolerance]
        if not np.isfinite(positive).all() or min(positive)<=0:raise ValueError('Positive finite SI parameters required')
        if not 0<self.notch_fraction<1 or not 0<self.residual_stiffness<1 or not -1<self.poisson<.5:raise ValueError('Invalid specimen constitutive parameters')
        if self.maximum_yy_strain>.02:raise ValueError('Reference small-strain qualification limited to 2% imposed strain')
        if type(self.cycles) is not int or self.cycles<1 or type(self.increments_per_cycle) is not int or self.increments_per_cycle<4 or self.increments_per_cycle%2:raise ValueError('Positive cycles and even path resolution >=4 required')
        if type(self.fatigue) is not bool or type(self.cache_forms) is not bool or type(self.max_stagger_iterations) is not int or self.max_stagger_iterations<1:raise ValueError('Invalid solver controls')
        if self.spacing_m>self.length_m/2:raise ValueError('Initial reference requires h <= ell/2; further mesh convergence still required')
        # Avoid accidental workstation-sized jobs before scaling qualification.
        if np.prod(self.shape_zyx)>32768:raise ValueError('Reference mesh exceeds the qualified local budget')


def write_vtu(path, mesh, point_fields, cell_fields):
    def array(name,a,kind='Float64'):
        a=np.asarray(a);components=a.shape[1] if a.ndim>1 else 1
        content=' '.join(str(v) for v in a.reshape(-1))
        return f'<DataArray type="{kind}" Name="{escape(name)}" NumberOfComponents="{components}" format="ascii">{content}</DataArray>'
    body=f'<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian"><UnstructuredGrid><Piece NumberOfPoints="{len(mesh.points)}" NumberOfCells="{len(mesh.cells)}">'
    body+='<Points>'+array('points_m',mesh.points)+'</Points><Cells>'
    body+=array('connectivity',mesh.cells.reshape(-1),'Int64')+array('offsets',4*np.arange(1,len(mesh.cells)+1),'Int64')+array('types',np.full(len(mesh.cells),10),'UInt8')+'</Cells>'
    body+='<PointData>'+''.join(array(k,v) for k,v in point_fields.items())+'</PointData>'
    body+='<CellData>'+''.join(array(k,v) for k,v in cell_fields.items())+'</CellData></Piece></UnstructuredGrid></VTKFile>'
    temp=Path(path).with_suffix('.pending');temp.write_text(body);temp.replace(path)


def read_records(directory):
    import h5py
    records=[]
    for p in sorted(Path(directory).glob('step_*.h5')):
        with h5py.File(p,'r') as f:
            if not f.attrs.get('complete',False):raise ValueError('Incomplete accepted checkpoint')
            record=json.loads(f.attrs['record'])
            record['fields']={k:f['voxel/'+k][:].tolist() for k in f['voxel']}
            records.append(record)
    if [r['step'] for r in records]!=list(range(len(records))):raise ValueError('Missing or reordered accepted steps')
    return records


def write_live_report(directory, config, records):
    data=json.dumps(records,allow_nan=False).replace('</','<\\/')
    shape=json.dumps(list(config.shape_zyx))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MORQEX TOP-DAMAGE | 3D reference</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#1c3443;background:#eef3f6;margin:0}main{max-width:1240px;margin:auto;padding:24px}header,section{background:white;padding:22px;margin:18px 0;border-radius:14px}h1{color:#087b7e;margin:0}.note{padding:12px;background:#fff2da;border-left:4px solid #b87711}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}canvas{width:100%;height:auto;background:#f8fafc}label{display:inline-block;margin:8px}select,input{font:inherit}.metric{font-variant-numeric:tabular-nums}a{color:#087b7e}@media(max-width:760px){.grid{display:block}main{padding:10px}}</style>
<main><header><h1>MORQEX / TOP-DAMAGE</h1><p>Compiled MFEM + variational mechanics + irreversible AT2 + fatigue history</p>
<p class="note">3D NOTCHED SPECIMEN BENCHMARK. These are mechanical load cycles, not battery charge/discharge cycles. No DFN capacity, pore opening, or NMC lifetime prediction is claimed. Damage is not pore volume.</p>
<p id="status"></p><p>Reload this file to see newly accepted steps. Full nodal/cell fields are in the adjacent HDF5 and VTU files.</p></header>
<section><label>Accepted step <input id="step" type="range" min="0" value="0"></label><label>Field <select id="field"><option value="damage">Mean damage d</option><option value="stress_yy_MPa">Cell-mean stress yy / MPa</option><option value="fatigue_MJ_m3">Fatigue / MJ m-3</option></select></label><label>Z slice <input id="slice" type="range" min="0" value="0"></label>
<div class="grid"><div><canvas id="map" width="540" height="460"></canvas><p id="range"></p></div><div><h2>Accepted state</h2><pre class="metric" id="values"></pre><a id="vtk">Open full 3D VTU</a><p>Rendered voxel values are volume averages of the tetrahedral reference fields.</p></div></div></section>
<section class="grid"><div><h2>Stress vs imposed strain</h2><canvas id="curve" width="540" height="340"></canvas></div><div><h2>Mean damage vs load step</h2><canvas id="damageCurve" width="540" height="340"></canvas></div></section>
<section><h2>Model and limits</h2><pre id="config"></pre><p>Natural damage flux at outer boundaries; precrack d=1 on a partial internal plane. KUBC: all outer displacements follow the imposed affine yy strain, xx=zz=0. Isotropic bulk/shear split; no crack-face contact. Fatigue toughness is held fixed during each staggered solve and updated once after acceptance. Mesh, path-resolution, and regularization sensitivity remain separate validation gates.</p>
<a href="https://doi.org/10.1016/j.cma.2019.112731">Carrara et al. fatigue baseline</a> · <a href="https://doi.org/10.1016/j.jpowsour.2022.231805">Ai et al. battery phase-field reference</a> · <a href="https://mfem.org">MFEM</a></section></main>
<script>const records=__DATA__,shape=__SHAPE__,config=__CONFIG__;
const step=document.getElementById('step'),slice=document.getElementById('slice'),field=document.getElementById('field');step.max=records.length-1;step.value=records.length-1;slice.max=shape[0]-1;
document.getElementById('config').textContent=JSON.stringify(config,null,2);document.getElementById('status').textContent=`${records.length-1} accepted increments; ${shape.join(' x ')} voxel cells; reference solver, FP64 CPU.`;
function chart(id,xkey,ykey){let c=document.getElementById(id),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);let xs=records.map(r=>r[xkey]),ys=records.map(r=>r[ykey]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(0,...ys),ymax=Math.max(...ys);let X=v=>45+(v-xmin)/(xmax-xmin||1)*460,Y=v=>295-(v-ymin)/(ymax-ymin||1)*250;g.strokeStyle='#9aaeb9';g.strokeRect(45,30,460,265);g.strokeStyle='#087b7e';g.lineWidth=2;g.beginPath();xs.forEach((v,i)=>i?g.lineTo(X(v),Y(ys[i])):g.moveTo(X(v),Y(ys[i])));g.stroke();g.fillStyle='#1c3443';g.font='13px sans-serif';g.fillText(xkey,220,330);g.fillText(ymax.toPrecision(4),2,22);g.fillText(xmax.toPrecision(4),455,315);}
function draw(){const r=records[+step.value],v=r.fields[field.value],z=+slice.value,[nz,ny,nx]=shape,lo=Math.min(...v),hi=Math.max(...v),c=document.getElementById('map'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);const w=400/Math.max(nx,ny);for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){let a=v[(z*ny+y)*nx+x],t=(a-lo)/(hi-lo||1);g.fillStyle=`hsl(${230-220*t},75%,${35+15*t}%)`;g.fillRect(50+x*w,20+(ny-1-y)*w,w+.3,w+.3);}g.fillStyle='#1c3443';g.font='16px sans-serif';g.fillText('X →    Y ↑    Z slice '+z,50,445);document.getElementById('range').textContent=`Range ${lo.toPrecision(5)} … ${hi.toPrecision(5)}`;let clean={...r};delete clean.fields;document.getElementById('values').textContent=JSON.stringify(clean,null,2);let a=document.getElementById('vtk');a.href=`step_${String(r.step).padStart(5,'0')}.vtu`;document.body.dataset.rendered='true';}step.oninput=draw;slice.oninput=draw;field.onchange=draw;draw();chart('curve','strain_yy','mean_stress_yy_MPa');chart('damageCurve','step','mean_damage');
</script></html>'''
    page=page.replace('__DATA__',data).replace('__SHAPE__',shape).replace('__CONFIG__',json.dumps(config.__dict__))
    target=Path(directory)/'index.html';temp=target.with_suffix('.pending');temp.write_text(page);temp.replace(target)
    return target


def run_reference(config_path, workspace, *, steps=8, resume=False, output=None):
    import h5py
    if type(steps) is not int or steps<1:raise ValueError('Positive step budget required')
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    content=Path(config_path).read_bytes();config=ReferenceConfig(**tomllib.loads(content.decode())['reference'])
    native=NativeConfig.from_toml(config_path);backend=MFEMScalarBackend(native,ws)
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    identity=hashlib.sha256(json.dumps({'config':content.decode(),'backend':backend.manifest['identity'],'code':code},sort_keys=True).encode()).hexdigest()
    directory=ws.path('results','top_damage',output or (config.name+'_'+identity[:12]));directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'run.json'
    if manifest.exists():
        if not resume:raise ValueError('Run exists; explicit resume required')
        if json.loads(manifest.read_text())['identity']!=identity:raise ValueError('Checkpoint/config/backend/code identity mismatch')
    else:
        if resume:raise ValueError('No run to resume')
        manifest.write_text(json.dumps({'identity':identity,'config':config.__dict__,'backend':backend.manifest,'code':code,'scope':'3D numerical mechanical specimen, not DFN or calibrated battery lifetime'},indent=2))
        (directory/'config.toml').write_bytes(content)
    mesh=TetMesh.from_volume(Volume(np.ones(config.shape_zyx,dtype=np.uint8),spacing=(config.spacing_m,)*3))
    if config.cache_forms:
        from .assembly_cache import CachedP1ScalarBackend
        backend=CachedP1ScalarBackend(mesh,backend)
    model=AT2Damage(mesh,backend,config.fracture_toughness_J_m2,config.length_m,config.residual_stiffness)
    mechanical=SplitElasticity(model,config.young_Pa,config.poisson)
    records=read_records(directory)
    if records:
        with h5py.File(directory/f'step_{records[-1]["step"]:05d}.h5','r') as f:
            if f.attrs['identity']!=identity or f.attrs['mesh_sha256']!=mesh.fingerprint:raise ValueError('Checkpoint identity mismatch')
            d=f['damage'][:];history=FatigueHistory(f['fatigue_accumulated_J_m3'][:],f['fatigue_previous_J_m3'][:])
        first=records[-1]['step']+1
    else:
        d=np.zeros(len(mesh.points));history=FatigueHistory.zero(len(mesh.cells));first=0
        plane=np.isclose(mesh.points[:,1],mesh.box[1,1]/2,rtol=0,atol=config.spacing_m*1e-8)
        d[plane&(mesh.points[:,0]<=config.notch_fraction*mesh.box[1,0]+config.spacing_m*1e-8)]=1
    total=config.cycles*config.increments_per_cycle
    for step in range(first,min(total+1,first+steps)):
        phase=(step%config.increments_per_cycle)/config.increments_per_cycle
        strain=config.maximum_yy_strain*(1-abs(2*phase-1));macro=np.array([0.,strain,0.,0.,0.,0.])
        begin=time.perf_counter()
        trial=staggered_step(mechanical,d,history,macro_strain=macro,fatigue=config.fatigue and step>0,
                             tolerance=config.stagger_tolerance,max_iterations=config.max_stagger_iterations)
        old_d=d;d=trial.damage;history=trial.fatigue
        if np.min(d-old_d)<-1e-12:raise RuntimeError('Irreversibility violation')
        cells_damage=d[mesh.cells].mean(axis=1)
        voxel=lambda a:np.bincount(mesh.voxel_ids,weights=mesh.volumes*a)/np.bincount(mesh.voxel_ids,weights=mesh.volumes)
        fields={'damage':voxel(cells_damage),'stress_yy_MPa':voxel(trial.mechanics.mean_cell_stress_Pa[:,1])*1e-6,'fatigue_MJ_m3':voxel(history.accumulated_J_m3)*1e-6}
        record={'step':step,'mechanical_cycle':step/config.increments_per_cycle,'strain_yy':strain,
                'mean_stress_yy_MPa':float(mesh.volumes@trial.mechanics.mean_cell_stress_Pa[:,1]/mesh.rve_volume*1e-6),
                'mean_damage':float(mesh.volumes@cells_damage/mesh.volumes.sum()),
                'elastic_energy_J':trial.mechanics.elastic_energy_J,
                'fixed_history_total_energy_J':float(trial.fixed_history_energy_trace_J[-1]),
                'fatigue_fracture_energy_change_J':trial.fatigue_fracture_energy_change_J,
                'damage_kkt':trial.damage_kkt,'mechanical_residual':trial.mechanics.equilibrium_residual,
                'stagger_iterations':trial.stagger_iterations,'solve_seconds':time.perf_counter()-begin}
        target=directory/f'step_{step:05d}.h5';temp=target.with_suffix('.pending')
        with h5py.File(temp,'w') as f:
            f.attrs.update(identity=identity,mesh_sha256=mesh.fingerprint,record=json.dumps(record),complete=True)
            arrays={'points_m':mesh.points,'cells':mesh.cells,'voxel_ids':mesh.voxel_ids,'damage':d,
                    'fatigue_accumulated_J_m3':history.accumulated_J_m3,'fatigue_previous_J_m3':history.previous_alpha_J_m3,
                    'displacement_m':trial.mechanics.displacement_m,'strain_mandel':trial.mechanics.strain,
                    'mean_cell_stress_Pa':trial.mechanics.mean_cell_stress_Pa,'reaction_N':trial.mechanics.reaction_N,
                    'fixed_history_energy_trace_J':trial.fixed_history_energy_trace_J}
            for name,a in arrays.items():f.create_dataset(name,data=a,compression='gzip')
            for name,a in fields.items():f.create_dataset('voxel/'+name,data=a)
            f.flush()
        temp.replace(target)
        write_vtu(target.with_suffix('.vtu'),mesh,{'damage':d,'displacement_m':trial.mechanics.displacement_m},
                  {'mean_stress_Mandel_Pa':trial.mechanics.mean_cell_stress_Pa,'fatigue_J_m3':history.accumulated_J_m3,'phase':mesh.phases})
        record['fields']={k:a.tolist() for k,a in fields.items()};records.append(record)
        write_live_report(directory,config,records)
        print(json.dumps({k:v for k,v in record.items() if k!='fields'}),flush=True)
    return {'directory':str(directory),'html':str(directory/'index.html'),'last_accepted_step':records[-1]['step'],
            'requested_steps':total,'complete_specimen_protocol':records[-1]['step']==total,'battery_lifetime_qualified':False}
