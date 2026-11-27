"""Independent phase-aligned temporal/cache comparisons; no curve fitting.

Uses actual current integrals and accepted 3D states. A failed convergence gate
is saved before returning nonzero; results and tolerances are never relabelled.
"""
import argparse,json,os
from pathlib import Path
import h5py
import numpy as np
from morqex.workspace import Workspace
from morqex.top_damage.network_report import audit_network_history


def read(directory):
    summary,records,_=audit_network_history(directory)
    if not summary['protocol_complete']:raise ValueError('Complete protocol required')
    phases=[];part=[];states=[]
    for path,record in zip(sorted(Path(directory).glob('step_*.h5')),records):
        if record['step']==0:continue
        with h5py.File(path) as f:
            t=f['curve/time_s'][:];v=f['curve/voltage_V'][:]
            part.extend(np.column_stack([t,v]).tolist())
            if record['completed_phase_capacity_Ah'] is not None:
                a=np.array(part);keep=np.r_[True,np.diff(a[:,0])>0];a=a[keep];a[:,0]-=a[0,0]
                phases.append((record['protocol_phase'],a));part=[]
                states.append({k:f[k][:] for k in ['damage','fatigue_J_m3','voxel/concentration_mol_m3','next/electronic_S_m','next/ionic_efficiency']})
    return summary,phases,states


def compare(coarse,fine):
    # A numerical convergence claim requires the same material geometry and
    # protocol. Numerical controls may differ; physical models may not.
    with h5py.File(Path(coarse)/'initial.h5') as f,h5py.File(Path(fine)/'initial.h5') as g:
        if f.attrs['fingerprint']!=g.attrs['fingerprint']:raise ValueError('Identical physical microstructure required')
    ma=json.loads((Path(coarse)/'manifest.json').read_text())['config']
    mb=json.loads((Path(fine)/'manifest.json').read_text())['config']
    numerical={'output','increments_per_half_cycle','concentration_regions','graph_update_tolerance','stagger_tolerance',
        'dfn_rtol','dfn_atol','project_algebraic_output'}
    defaults={'diffusivity_interpretation':'tracer_reference'}
    for key in (ma.keys()|mb.keys())-numerical:
        if ma.get(key,defaults.get(key))!=mb.get(key,defaults.get(key)):
            raise ValueError('Physical/protocol mismatch: '+key)
    a,pa,sa=read(coarse);b,pb,sb=read(fine)
    if len(pa)!=len(pb) or [p[0] for p in pa]!=[p[0] for p in pb]:raise ValueError('Matched complete cycling histories required')
    voltage=[]
    for (name,x),(_,y) in zip(pa,pb):
        t=np.unique(np.r_[x[:,0],y[:,0]]);t=t[t<=min(x[-1,0],y[-1,0])]
        error=np.interp(t,x[:,0],x[:,1])-np.interp(t,y[:,0],y[:,1])
        voltage.append({'phase':name,'rmse_mV':float(np.sqrt(np.trapezoid(error**2,t)/t[-1])*1000),
            'max_error_mV':float(np.max(abs(error))*1000),'end_time_relative_difference':float(abs(x[-1,0]-y[-1,0])/y[-1,0])})
    fields={}
    for k in sa[0]:
        fields[k]=max(float(np.linalg.norm(x[k]-y[k])/max(np.linalg.norm(y[k]),1e-30)) for x,y in zip(sa,sb))
    capacity=float(np.max(abs(np.array(a['discharge_capacity_Ah'])/np.array(b['discharge_capacity_Ah'])-1)))
    gates={'capacity_below_0p2percent':capacity<=.002,'voltage_rmse_below_2mV':max(v['rmse_mV'] for v in voltage)<=2.,
        'damage_below_1percent':fields['damage']<=.01,'fatigue_below_1percent':fields['fatigue_J_m3']<=.01,
        'concentration_below_0p2percent':fields['voxel/concentration_mol_m3']<=.002}
    return {'coarse':str(coarse),'fine':str(fine),'capacity_relative_error':capacity,'phase_voltage':voltage,
        'phase_end_field_relative_L2':fields,'gates':gates,'passed':all(gates.values()),
        'scope':'Matched-geometry temporal/cache/concentration-space comparison; not continuum mesh convergence or material validation.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('coarse');p.add_argument('fine');p.add_argument('--name',required=True)
    args=p.parse_args();ws=Workspace(os.environ['MORQEX_WORKSPACE'])
    root=ws.path('results/top_damage/network_cycles');result=compare(root/args.coarse,root/args.fine)
    target=ws.path('reports',args.name+'.json');target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
