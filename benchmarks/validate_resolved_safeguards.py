"""Compiled residual-budget equivalence and concentration rejection checks.

The rejected large-step continuation is a regression fixture, not an accepted
cycle. It requires the separately retained failed 10 s timestep experiment.
"""
import argparse,json,os,time
from pathlib import Path
import h5py,numpy as np
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.top_damage.polycrystal import PolycrystalData
from morqex.top_damage.resolved_echem import ResolvedElectrochemistry

p=argparse.ArgumentParser();p.add_argument('--output',default='resolved_safeguards');args=p.parse_args()
ws=Workspace(os.environ['MORQEX_WORKSPACE']);backend=ResolvedElectrochemistry(ws)
reference=ws.path('results/top_damage/resolved_particles_reference')
root=ws.path('results/top_damage',args.output)
if root.exists():raise ValueError('Use a new result directory')
root.mkdir(parents=True)
data=PolycrystalData.load_hdf5(reference/'initial.h5');labels=data.structure.labels
full=np.concatenate([np.choose(labels,[6,5,7]),np.full((32,32,4),4),np.full((32,32,8),1)],axis=2).astype(np.uint8)
volume=Volume(full,spacing=data.structure.spacing_zyx_m)
parameters=json.loads((reference/'discharge/morqex_segment.json').read_text())['parameters']
parameters.update({'echemamr.morqex_linear_budget_fraction':.1,'echemamr.potential_initial_guess':1})
tick=time.perf_counter();curve=backend.run(volume,root/'budget_warm',{i:i for i in [1,4,5,6,7]},parameters)
backend.export_fields(root/'budget_warm/plt00002',root/'fields.h5',['Concentration','Potential'])
result={'seconds':time.perf_counter()-tick,'diagnostics':curve.diagnostics,'scope':'Short-time inner accuracy allocation and invalid-trial rejection; not long-time stability or lifetime qualification'}
with h5py.File(root/'fields.h5') as f,h5py.File(reference/'fields_00002.h5') as r:
    result['relative_field_differences']={k:float(np.linalg.norm(f[k][:]-r[k][:])/max(np.linalg.norm(r[k][:]),1e-30)) for k in ['Concentration','Potential']}
old=json.loads((reference/'probe.json').read_text())
result['maximum_voltage_difference_V']=float(np.max(abs(curve.voltage_V-np.asarray(old['voltage_V']))))
def rejected(name,params,restart=None):
    try:backend.run(volume,root/name,{i:i for i in [1,4,5,6,7]},params,restart=restart)
    except Exception as exc:
        log=(root/name/'solver.log').read_text()
        if 'concentration bounds violated' not in log:raise RuntimeError('Unexpected failure instead of concentration rejection') from exc
        manifest=json.loads((root/name/'morqex_segment.json').read_text())
        return {'concentration_rejected':True,'manifest_state':manifest['state'],'clipping_enabled':manifest['parameters']['echemamr.bound_specden']}
    raise AssertionError('Invalid state was accepted')
result['invalid_initial']=rejected('invalid_initial',parameters|{'prob.soc_cathode_t0':1.1})
restart=ws.path('results/top_damage/resolved_particles_dt10_300_budget/discharge/chk00008')
if not restart.exists():raise ValueError('Generate the retained failed 10 s benchmark before this regression')
result['invalid_large_step']=rejected('invalid_large_step',parameters|{'max_step':9,'stop_time':90.,'echemamr.dtmin':10.,'echemamr.dtmax':10.},restart)
result['passed']=max(result['relative_field_differences'].values())<1e-7 and result['maximum_voltage_difference_V']<1e-7 and all(result[k]['manifest_state']=='failed' and result[k]['clipping_enabled']==0 for k in ['invalid_initial','invalid_large_step'])
(root/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['passed']
