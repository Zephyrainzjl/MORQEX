"""Build an offline report from full-volume evidence, without invented fields."""
import datetime as dt
import base64
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pybamm as pb
from morqex.electrochemistry.open_materials import HOSTS
from morqex.visualization import render_report

def main():
    ws=Path(os.environ['MORQEX_WORKSPACE']);structures=[];resource=[]
    for name in ['nmc1uncal','nmc1cal','nmc2uncal','nmc2cal','nmc3uncal']:
        path=ws/'operators/full_volume'/name;meta=json.loads((path/'manifest.json').read_text());basis=json.loads((path/'concentration.json').read_text())
        if hashlib.sha256((path/'geometry.npz').read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Geometry checksum mismatch')
        with np.load(path/'geometry.npz') as a:phase=a['phase']
        step=max(1,int(np.ceil(max(phase.shape)/17)));small=phase[::step,::step,::step];z,y,x=np.nonzero(small);p=small[z,y,x]
        lod=np.column_stack([(x*step+.5)/phase.shape[2],(y*step+.5)/phase.shape[1],(z*step+.5)/phase.shape[0],p]).tolist()
        transports=[json.loads((path/(kind+'_transport.json')).read_text()) for kind in ['electrolyte','electrode']]
        for kind,value in zip(['electrolyte','electrode'],transports):
            if value['geometry_sha256']!=meta['sha256']:raise ValueError('Transport belongs to a different geometry')
            resource.append(json.loads((ws/'reports'/f'full_{name}_{kind}_latest.json').read_text()))
        structures.append({'id':name,'shape':meta['shape'],'voxels':meta['voxels'],'active':meta['phase_fractions']['1'],'cbd':meta['phase_fractions']['2'],
            'lod':lod,'lod_step':step,'spatial_ports':json.loads((path/'spatial_ports_5.json').read_text()) if (path/'spatial_ports_5.json').exists() else None,'transport':transports,'concentration_dimension':basis['dimension'],'source':meta['source']})
    cases=[]
    for path in sorted((ws/'results/full_ct_dfn').glob('*/result.json')):
        value=json.loads(path.read_text());hashes=value.get('source_sha256',{})
        if not hashes or any(not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in hashes.items()):continue
        value['id']=path.parent.name;value['maximum_concentration']=float(pb.ParameterValues(HOSTS[value['arguments']['host']])['Maximum concentration in positive electrode [mol.m-3]']);cases.append(value)
    if not cases:raise ValueError('No current-source DFN cases; do not substitute stale curves')
    accuracy=[]
    for rate in [.1,3.,5.]:
        candidates=[c for c in cases if c['arguments']['dataset']=='nmc1cal' and c['arguments']['host']=='NMC532' and not c['arguments']['cycles'] and c['arguments']['rate']==rate]
        if not candidates:continue
        reference=max(candidates,key=lambda c:c['arguments']['nx']*c['arguments']['nr']);qref=np.asarray(reference['curves']['discharge_capacity_Ah']);vref=np.asarray(reference['curves']['voltage_V'])
        for c in candidates:
            if c is reference:continue
            q=np.asarray(c['curves']['discharge_capacity_Ah']);v=np.asarray(c['curves']['voltage_V']);grid=np.linspace(max(q[0],qref[0]),min(q[-1],qref[-1]),1001);delta=np.interp(grid,q,v)-np.interp(grid,qref,vref)
            accuracy.append({'rate':rate,'grid':f"({c['arguments']['nx']},{c['arguments']['nr']})",'reference':f"({reference['arguments']['nx']},{reference['arguments']['nr']})",'rmse_V':float(np.mean(delta*delta)**.5),'maximum_V':float(np.max(abs(delta))),
                'relative_capacity_error':float(abs(q[-1]-qref[-1])/max(abs(qref[-1]),1e-12)),'scope':'sampled common-capacity comparison, interpolated output; no uniform time-continuous bound'})
    mechanics=[json.loads(p.read_text()) for p in sorted((ws/'results/full_voxel').glob('*/weak_0e+00/result.json'))]
    trials=[]
    for path in sorted((ws/'runs').glob('*/manifest.json')):
        record=json.loads(path.read_text());command=record.get('command',[])
        if any(Path(arg).name in {'study_full_voxel_response.py','smooth_voxel_reference.py'} for arg in command):
            trials.append({'run':str(path.parent),'wall_seconds':record.get('wall_seconds',0.),'exit_code':record.get('exit_code'),'stopped_because':record.get('stopped_because'),'source_changed_during_run':record.get('source_changed_during_run'),'command':command})
    evidence={}
    for name in ['open_full_tests','open_coupling_tests','phase_coarse_tests','voxel_gpu_probe','voxel_gpu_odd_spacing','voxel_transport_probe','open_wheel_probe']:
        file=ws/'reports'/(name+'_latest.json')
        if file.exists():evidence[name]=json.loads(file.read_text())
    qualification=ws/'reports/voxel_reference_qualification.json'
    reference_figure=None;figure_record=ws/'reports/full_reference_figure.json'
    if figure_record.exists():
        reference_figure=json.loads(figure_record.read_text());figure_path=Path(reference_figure['image'])
        if hashlib.sha256(figure_path.read_bytes()).hexdigest()!=reference_figure['image_sha256']:raise ValueError('Reference figure checksum mismatch')
        matching=next((m for m in mechanics if m['dataset']==reference_figure['dataset']),None)
        if matching is None or matching.get('fields',{}).get('files',{}).get('von_mises_Pa',{}).get('sha256')!=reference_figure['field_sha256']:raise ValueError('Reference figure is stale relative to the mechanical field')
        reference_figure['data_url']='data:image/png;base64,'+base64.b64encode(figure_path.read_bytes()).decode()
    payload={'schema':'morqex-full-volume-report-1','generated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'structures':structures,'cases':cases,'accuracy':accuracy,'mechanics':mechanics,'mechanics_research_trials':trials,'evidence':evidence,'reference_field_figure':reference_figure,'reference_qualification':json.loads(qualification.read_text()) if qualification.exists() else None,
        'resource_summary':f"最大采样进程显存 {max(r['peak_gpu_process_mib_sampled'] for r in resource)} MiB；10 项合计墙钟 {sum(r['wall_seconds'] for r in resource):.2f} 秒。"}
    font=ws/'data/report_font/NotoSansCJKsc-Regular.otf';license_file=ws/'data/report_font/LICENSE'
    destination=render_report(payload,ws/'reports/MORQEX_full_volume_report.html',font_path=font if font.exists() else None,font_license=license_file.read_text() if font.exists() else None)
    metrics={'html':str(destination),'structures':len(structures),'voxels':sum(s['voxels'] for s in structures),'dfn_cases':len(cases),'dfn_protocol_passed':sum(c['completed'] for c in cases),'accuracy':accuracy}
    (ws/'reports/full_volume_report_metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
