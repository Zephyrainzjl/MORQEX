"""Compare same-geometry resolved trajectories; report evolution-relative error.

This is a temporal discretization comparison, never an experimental error or
a proof of spatial convergence. No automatic publication/lifetime gate.
"""
import argparse,json,os
from pathlib import Path
import h5py,numpy as np
from morqex import Workspace
from morqex.top_damage.backend import require_local_budget
require_local_budget()
p=argparse.ArgumentParser();p.add_argument('coarse');p.add_argument('fine');p.add_argument('--output',default='resolved_time_comparison.json');args=p.parse_args()
ws=Workspace(os.environ['MORQEX_WORKSPACE']);roots=[ws.path(args.coarse),ws.path(args.fine)]
curves=[json.loads((r/'probe.json').read_text()) for r in roots]
settings=[json.loads((r/'discharge/morqex_segment.json').read_text())['parameters'] for r in roots]
numerical={'max_step','echemamr.dtmin','echemamr.dtmax','echemamr.potential_initial_guess',
           'echemamr.morqex_linear_budget_fraction','echemamr.linsolve_maxiter'}
if {k:v for k,v in settings[0].items() if k not in numerical}!={k:v for k,v in settings[1].items() if k not in numerical}:
    raise ValueError('Constitutive parameters, forcing and boundary settings must match')
fields=[sorted(r.glob('fields_*.h5')) for r in roots]
if not all(fields):raise ValueError('Complete exported field histories required')
initial=[];final=[];times=[];masks=[];spacing=[]
for files in fields:
    with h5py.File(files[0]) as f:initial.append(f['Concentration'][:]);times.append(float(f.attrs['time_s']))
    with h5py.File(files[-1]) as f:
        final.append(f['Concentration'][:]);times.append(float(f.attrs['time_s']));spacing.append(f.attrs['spacing_zyx_m'])
        masks.append({'solid':f['Cathode_active_material'][:]>0,'electrolyte':(f['Cathode_electrolyte'][:]+f['Separator'][:])>0})
if times[:2]!=times[2:] or not np.array_equal(initial[0],initial[1]) or not np.array_equal(spacing[0],spacing[1]) or any(not np.array_equal(masks[0][k],masks[1][k]) for k in masks[0]):raise ValueError('Matching time interval, lattice, initial concentration and phases required')
a,b=final;rows={}
for key,mask in masks[0].items():
    error=float(np.linalg.norm((a-b)[mask]));evolution=float(np.linalg.norm((b-initial[1])[mask]))
    rows[key]={'absolute_L2':error,'full_concentration_relative_L2':error/float(np.linalg.norm(b[mask])),
               'evolution_relative_L2':error/evolution if evolution>1e-30 else None,'evolution_L2':evolution}
coarse,fine=curves;dv=np.asarray(coarse['voltage_V'])-np.interp(coarse['time_s'],fine['time_s'],fine['voltage_V'])
result={'coarse':str(roots[0]),'fine':str(roots[1]),'time_interval_s':times[:2],
        'voltage_RMSE_V':float(np.sqrt(np.mean(dv**2))),'voltage_max_difference_V':float(np.max(abs(dv))),
        'final_concentration':rows,'scope':__doc__,'publication_qualified':False}
target=ws.path('results/top_damage',args.output)
if target.exists():raise ValueError('Choose a new comparison output')
target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
