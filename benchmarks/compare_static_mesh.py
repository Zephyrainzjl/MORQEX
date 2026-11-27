"""Predeclared same-geometry static FE gates, retaining failed comparisons."""
import argparse,json,os
from pathlib import Path
import h5py,numpy as np
from morqex import Workspace
p=argparse.ArgumentParser();p.add_argument('coarse',type=int);p.add_argument('fine',type=int)
p.add_argument('--study',choices=['static_mesh_v1','static_mesh_storage_v2','static_mesh_face_contacts_v3','static_mesh_adaptive_v1','static_mesh_quadratic_v1','static_mesh_quadratic_adaptive_v1','static_mesh_quadratic_adaptive_budget_v1'],default='static_mesh_v1')
p.add_argument('--fine-study',choices=['static_mesh_quadratic_adaptive_budget_v1']);a=p.parse_args()
ws=Workspace(os.environ['MORQEX_WORKSPACE']);root=ws.path('results/top_damage',a.study)
fine_root=ws.path('results/top_damage',a.fine_study) if a.fine_study else root
records=[json.loads((r/f'level{k}/result.json').read_text()) for r,k in ((root,a.coarse),(fine_root,a.fine))]
x,y=records
if x.get('separate_point_contacts',False)!=y.get('separate_point_contacts',False):raise ValueError('Solid contact policy mismatch')
for key in ('geometry_fingerprint','length_m','mechanical_boundary','chemical_log_stretch_change','fatigue_enabled'):
    if x[key]!=y[key]:raise ValueError('Physical model mismatch: '+key)
errors={}
with h5py.File(root/f'level{a.coarse}/fields.h5') as f,h5py.File(fine_root/f'level{a.fine}/fields.h5') as g:
    for key in ('damage','stress_Pa'):
        u=f['common_voxel/'+key][:];v=g['common_voxel/'+key][:]
        errors[key+'_relative_L2']=float(np.linalg.norm(u-v)/max(np.linalg.norm(v),1e-30))
    errors['mean_stress_relative_L2']=float(np.linalg.norm(f['common_voxel/stress_Pa'][:].mean(axis=(0,1,2))-g['common_voxel/stress_Pa'][:].mean(axis=(0,1,2)))/max(np.linalg.norm(g['common_voxel/stress_Pa'][:].mean(axis=(0,1,2))),1e-30))
errors['total_energy_relative']=abs((x['elastic_energy_J']+x['fracture_energy_J'])/(y['elastic_energy_J']+y['fracture_energy_J'])-1)
gates={'damage_below1percent':errors['damage_relative_L2']<=.01,'common_voxel_stress_below2percent':errors['stress_Pa_relative_L2']<=.02,
    'mean_stress_below1percent':errors['mean_stress_relative_L2']<=.01,'energy_below1percent':errors['total_energy_relative']<=.01}
r={'levels':[a.coarse,a.fine],'studies':[a.study,a.fine_study or a.study],'errors':errors,'gates':gates,'passed':all(gates.values()),
    'scope':'Common original-voxel volume-averaged fields, static fixed geometry and fixed ell; excludes pointwise maxima and fatigue history'}
name=f'static_mesh_{a.coarse}_{a.fine}' if a.study=='static_mesh_v1' else f'{a.fine_study or a.study}_{a.coarse}_{a.fine}'
ws.path('reports',name+'.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if not r['passed']:raise SystemExit(1)
