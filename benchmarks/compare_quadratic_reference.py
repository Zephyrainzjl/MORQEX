"""Independent stored P2 storage/quadrature comparisons; not a spatial gate."""
import argparse,json,os
import h5py,numpy as np
from morqex import Workspace
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['storage','quadrature'],required=True);p.add_argument('--level',type=int,default=1);a=p.parse_args()
ws=Workspace(os.environ['MORQEX_WORKSPACE']);root=ws.path('results/top_damage')
old=root/'static_mesh_quadratic_v1'/f'level{a.level}'
new=root/('static_mesh_quadratic_storage_v2' if a.kind=='storage' else 'static_mesh_quadratic_q7_v1')/f'level{a.level}'
errors={}
with h5py.File(old/'fields.h5') as f,h5py.File(new/'fields.h5') as g:
    for key in ('points_m','cells','mechanics_points_m','mechanics_element_nodes'):
        if not np.array_equal(f[key][:],g[key][:]):raise ValueError('Physical discretization changed')
    for key in ('nodal_damage','nodal_displacement_m','voxel/stress_Pa'):
        x=f[key][:];y=g[key][:];errors[key]=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
x=json.loads((old/'result.json').read_text());y=json.loads((new/'result.json').read_text())
for key in ('geometry_fingerprint','length_m','mechanical_boundary','chemical_log_stretch_change','fatigue_enabled','separate_point_contacts'):
    if x[key]!=y[key]:raise ValueError('Physical model changed')
errors['energy']=abs((x['elastic_energy_J']+x['fracture_energy_J'])/(y['elastic_energy_J']+y['fracture_energy_J'])-1)
threshold=1e-8 if a.kind=='storage' else 1e-4
result={'kind':a.kind,'level':a.level,'relative_errors':errors,'threshold':threshold,'passed':max(errors.values())<=threshold,
        'scope':'Same spatial discretization; storage or quadrature check only, not spatial/cyclic convergence'}
if a.kind=='storage':
    matches=[]
    for study in ('static_mesh_quadratic_v1','static_mesh_quadratic_storage_v2'):
        candidate=[]
        for file in ws.path('runs').glob('*/manifest.json'):
            record=json.loads(file.read_text());cmd=record.get('command',[])
            if '--study' in cmd and cmd[cmd.index('--study')+1]==study and '--refinement' in cmd and cmd[cmd.index('--refinement')+1]==str(a.level) and record.get('exit_code')==0:candidate.append(record)
        if len(candidate)!=1:raise ValueError('Require one unambiguous successful timing per study')
        matches.append(candidate[0])
    result['performance']={name:[r[key] for r in matches] for name,key in [('wall_seconds','wall_seconds'),('peak_RSS_bytes','peak_group_rss_bytes_sampled')]}
    result['performance']['scope']='Single-run same-output comparison, not lifetime acceleration'
target=ws.path('reports',f'quadratic_{a.kind}_level{a.level}_equivalence.json');target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit(1)
