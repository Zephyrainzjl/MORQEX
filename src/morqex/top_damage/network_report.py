"""Independent history checks and offline HTML of actual accepted states."""
import json
from pathlib import Path
import numpy as np
from morqex.workspace import Workspace


def audit_network_history(directory):
    import h5py
    directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
    paths=sorted(directory.glob('step_*.h5'))
    if not paths:raise ValueError('No accepted network states')
    records=[];curves=[];previous_d=previous_f=None;identity=meta['identity'];max_mass=max_jump=max_kkt=max_equilibrium=0.
    capacities=[];last_time=-1.;max_damage=0.;min_concentration=float('inf');max_concentration=0.
    with h5py.File(directory/'initial.h5') as f:
        labels=f['phase_labels'][:];h=np.asarray(f.attrs['spacing_zyx_m'])
    with h5py.File(directory/'mesh.h5') as f:partition=f['partition'][:];voxel_ids=f['material_voxel_ids'][:]
    for index,path in enumerate(paths):
        with h5py.File(path) as f:
            r=json.loads(f.attrs['record'])
            if f.attrs['identity']!=identity or r['step']!=index:raise ValueError('Non-contiguous or incompatible accepted history')
            d=f['damage'][:];fatigue=f['fatigue_J_m3'][:];c=f['concentration_regions_mol_m3'][:]
            if not all(np.isfinite(v).all() for v in (d,fatigue,c)) or np.any((d<0)|(d>1)) or np.any(c<=0) or np.any(c>=meta['maximum_concentration_mol_m3']):raise ArithmeticError('Invalid state range')
            if previous_d is not None and (np.min(d-previous_d)<-1e-12 or np.min(fatigue-previous_f)<-1e-8):raise ArithmeticError('Damage/fatigue irreversibility failed')
            full=f['voxel/concentration_mol_m3'][:].reshape(len(c),-1)[:,voxel_ids]
            np.testing.assert_allclose(full,c[:,partition],rtol=1e-13,atol=1e-12)
            np.testing.assert_allclose(full.sum(axis=1)*np.prod(h),r['solid_lithium_mol_per_SVE'],rtol=1e-12)
            previous_d=d;previous_f=fatigue;max_damage=max(max_damage,float(d.max()))
            min_concentration=min(min_concentration,float(c.min()));max_concentration=max(max_concentration,float(c.max()))
            max_mass=max(max_mass,r['inventory_relative_defect']);max_jump=max(max_jump,r['relative_state_jump'])
            for diag in r['node_diagnostics']:
                max_kkt=max(max_kkt,diag.get('damage_kkt',0));max_equilibrium=max(max_equilibrium,diag.get('equilibrium_residual',0))
            if r['time_s']<=last_time:raise ArithmeticError('Accepted times are not strictly increasing')
            last_time=r['time_s']
            if index:
                t=f['curve/time_s'][:];v=f['curve/voltage_V'][:];I=f['curve/current_A'][:];q=f['curve/net_capacity_Ah'][:]
                if not all(np.isfinite(a).all() for a in (t,v,I,q)) or np.any(np.diff(t)<=0):raise ArithmeticError('Invalid electrochemical curve')
                np.testing.assert_allclose(q[-1]-q[0],np.trapezoid(I,t)/3600,rtol=1e-7,atol=1e-13)
                curves.extend(np.column_stack([t,v,I,q]).tolist())
            if r['completed_phase_capacity_Ah'] is not None:
                capacities.append({'phase':r['protocol_phase'],'completed_cycles':r['completed_cycles'],
                    'capacity_Ah':r['completed_phase_capacity_Ah'],'time_s':r['time_s']})
            records.append(r)
    if max_mass>1e-6 or max_jump>1e-9 or max_equilibrium>1e-7 or max_kkt>meta['config']['stagger_tolerance']:
        raise ArithmeticError('History violates recorded numerical acceptance gates')
    dis=[a['capacity_Ah'] for a in capacities if a['phase']=='discharge']
    summary={'accepted_steps':len(paths)-1,'completed_cycles':records[-1]['completed_cycles'],
        'requested_cycles':meta['config']['cycles'],'protocol_complete':records[-1]['completed_cycles']>=meta['config']['cycles'],
        'max_inventory_relative_defect':max_mass,'max_state_jump':max_jump,'max_damage_kkt':max_kkt,
        'max_equilibrium_residual':max_equilibrium,'max_damage':max_damage,
        'min_concentration_mol_m3':min_concentration,'max_concentration_mol_m3':max_concentration,
        'discharge_capacity_Ah':dis,'capacity_retention_percent':([100*a/dis[0] for a in dis] if dis else []),
        'history_consistency_passed':True,'physical_topology_qualified':False,'publication_qualified':False,
        'mesh_time_convergence_qualified':False,'long_term_prediction_qualified':False,'identity':identity,
        'phase_fractions':{str(k):float(np.mean(labels==k)) for k in (0,1,2)}}
    charge=[r.get('charge_relative_defect') for r in records[1:]]
    port=[r.get('local_port_relative_defect') for r in records[1:]]
    summary['charge_and_local_port_history_checked']=bool(charge) and all(v is not None for v in charge+port)
    if summary['charge_and_local_port_history_checked']:
        summary.update(max_charge_relative_defect=max(charge),max_local_port_relative_defect=max(port))
        if max(charge)>1e-6 or max(port)>1e-8:raise ArithmeticError('Charge/local energy port check failed')
    return summary,records,curves


def build_network_report(directory,workspace,*,output=None,max_frames=25):
    import h5py
    from .backend import require_local_budget
    require_local_budget();ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ws.root.resolve()):raise ValueError('External workspace results required')
    summary,records,curves=audit_network_history(directory)
    meta=json.loads((directory/'manifest.json').read_text());paths=sorted(directory.glob('step_*.h5'))
    selected=np.unique(np.linspace(0,len(paths)-1,min(max_frames,len(paths))).astype(int));frames=[]
    with h5py.File(directory/'initial.h5') as f:labels=f['phase_labels'][:];particles=f['particle_ids'][:];grains=f['grain_ids'][:]
    for index in selected:
        with h5py.File(paths[index]) as f:
            stress=f['voxel/stress_Pa'][:].reshape((meta['config']['nodes'],)+labels.shape+(3,3))
            fields={k:f['voxel/'+v][:].reshape(meta['config']['nodes'],-1).tolist() for k,v in
                [('c','concentration_mol_m3'),('d','damage'),('fatigue','fatigue_J_m3'),('electron','electronic_potential_unit_load'),('ion','ionic_potential_unit_load')]}
            fields['stress']=(np.trace(stress,axis1=-2,axis2=-1)/3e6).reshape(meta['config']['nodes'],-1).tolist()
            frames.append({'step':int(index),'time_s':records[index]['time_s'],'cycles':records[index]['completed_cycles'],'fields':fields})
    payload={'summary':summary,'metadata':meta,'records':records,'curves':curves,'frames':frames,
        'shape':list(labels.shape),'phase':labels.ravel().tolist(),'particle':particles.ravel().tolist(),'grain':grains.ravel().tolist()}
    html=Path(__file__).with_name('network_view.html').read_text().replace('__PAYLOAD__',json.dumps(payload,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
    font=ws.path('data/report_font')
    if (font/'NotoSansCJKsc-Regular.otf').exists():
        from morqex.visualization.report_font import embed_font
        html=embed_font(html,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
    target=ws.path('reports',output or meta['config']['output']+'.html');target.write_text(html)
    target.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    np.savetxt(directory/'electrochemical_curve.csv',np.asarray(curves).reshape(-1,4),delimiter=',',header='time_s,voltage_V,current_A,net_capacity_Ah',comments='')
    (directory/'validation.json').write_text(json.dumps(summary,indent=2))
    return {'html':str(target),**summary}
