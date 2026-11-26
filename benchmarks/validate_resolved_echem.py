"""Compiled 3D electrochemical time/restart/field regression; not a lifetime claim."""
import argparse,json,time,sys,os
from pathlib import Path
import numpy as np,h5py
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.top_damage.resolved_echem import ResolvedElectrochemistry
parser=argparse.ArgumentParser();parser.add_argument('--output',default='resolved_planar_validation_source');args=parser.parse_args()
ws=Workspace(os.environ['MORQEX_WORKSPACE']);backend=ResolvedElectrochemistry(ws)
root=ws.path('results/top_damage',args.output);root.mkdir(parents=True,exist_ok=True)
labels=np.full((8,8,20),4,np.uint8);labels[:,:,:8]=5;labels[:,:,12:]=1
volume=Volume(labels,(1e-6,)*3)
base={'prob.Crate':-.5,'prob.soc_anode_t0':.8,'prob.soc_cathode_t0':.5,'prob.Temperature':298.15,
 'prob.Sp_corr_anode':1.,'prob.Sp_corr_cathode':1.,'prob.Cphi_electrolyte_coupling':1,
 'echemamr.bv_nonlinear_iters':100,'echemamr.bv_nonlinear_reltol':1e-8,'echemamr.bv_nonlinear_abstol':1e-12,
 'echemamr.linsolve_reltol':1e-10,'echemamr.linsolve_abstol':1e-12,'echemamr.linsolve_max_coarsening_level':0,
 'amr.plot_int':1,'amr.chk_int':1,'amr.line_plot_int':-1}
fields=['Concentration','Potential','Cathode_active_material','Anode_active_material','Separator','Nanoporosity']
def case(name,dt,stop=2.,restart=None,current=-.5,grid=4096,iters=100):
 out=root/name;p=base|{'max_step':int(round(stop/dt)),'stop_time':stop,'echemamr.dtmin':dt,'echemamr.dtmax':dt,'prob.Crate':current,'amr.max_grid_size':grid,'echemamr.bv_nonlinear_iters':iters}
 curve=backend.run(volume,out,{i:i for i in [1,4,5]},p,restart=restart)
 final=out/f'plt{int(round(stop/dt)):05d}'
 state=out/'fields.h5';backend.export_fields(final,state,fields)
 with h5py.File(state) as f:arrays={k:f[k][:] for k in fields};t=float(f.attrs['time_s']);spacing=f.attrs['spacing_zyx_m']
 assert t==stop and arrays['Concentration'].shape==labels.shape
 assert np.array_equal(arrays['Cathode_active_material'],labels==5)
 assert np.array_equal(arrays['Anode_active_material'],labels==1)
 assert np.array_equal(arrays['Separator'],labels==4)
 assert np.allclose(spacing,volume.spacing,rtol=0,atol=1e-20)
 assert np.min(arrays['Concentration'])>0
 result={'voltage_V':curve.voltage_V.tolist(),'time_s':curve.time_s.tolist(),'current_A':curve.current_A.tolist(),'diagnostics':curve.diagnostics,'fields':str(state)}
 return result,arrays
tick=time.perf_counter();results={};states={}
for name,dt in [('dt1',1.),('dt05',.5),('dt025',.25)]:results[name],states[name]=case(name,dt)
results['first'],_=case('first',1.,stop=1.)
results['restart'],states['restart']=case('restart',1.,restart=root/'first/chk00001')
results['charge'],states['charge']=case('charge',1.,stop=4.,restart=root/'dt1/chk00002',current=.5)
results['multibox'],states['multibox']=case('multibox',1.,grid=4)
results['restart_relative_field_error']={k:float(np.max(abs(states['restart'][k]-states['dt1'][k]))/max(np.max(abs(states['dt1'][k])),1.)) for k in fields}
results['multibox_relative_field_error']={k:float(np.max(abs(states['multibox'][k]-states['dt1'][k]))/max(np.max(abs(states['dt1'][k])),1.)) for k in fields}
results['time_refinement_voltage_changes_V']=[abs(results[a]['voltage_V'][-1]-results[b]['voltage_V'][-1]) for a,b in [('dt1','dt05'),('dt05','dt025')]]
try:case('forced_failure',1.,iters=1)
except Exception as exc:results['forced_failure_rejected']=True;results['forced_failure_reason']=str(exc)
else:results['forced_failure_rejected']=False
dv=np.prod(volume.spacing);F=96485.33212;I=results['dt1']['current_A'][-1]
initial=np.where(labels==1,.8*28000,np.where(labels==5,.5*49600,1200.))
results['current_inventory_check']={}
for name in ['dt1','charge']:
 a=states[name]['Concentration'];netcharge=I*2 if name=='dt1' else 0.
 electrolyte=abs(float((a[labels==4]-initial[labels==4]).sum())/float(initial[labels==4].sum()))
 anode=abs(float((a[labels==1]-initial[labels==1]).sum()*dv)-netcharge/F)
 cathode=abs(float((a[labels==5]-initial[labels==5]).sum()*dv)+netcharge/F)
 results['current_inventory_check'][name]={'anode_defect_mol':anode,'cathode_defect_mol':cathode,'electrolyte_relative_defect':electrolyte}
results['seconds']=time.perf_counter()-tick
results['passed']=all(d['anode_defect_mol']<1e-20 and d['cathode_defect_mol']<1e-20 and d['electrolyte_relative_defect']<1e-8 for d in results['current_inventory_check'].values()) and results['time_refinement_voltage_changes_V'][1]<.75*results['time_refinement_voltage_changes_V'][0] and results['forced_failure_rejected'] and max(results['restart_relative_field_error'].values())<1e-8 and max(results['multibox_relative_field_error'].values())<1e-6
results['scope']='Extruded planar 3D electrochemical solver/restart reference, not material-calibrated fatigue or physical topology qualification'
(root/'validation.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));assert results['passed']
