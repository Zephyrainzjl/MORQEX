"""Same 406272-Tet4 physics/output, independent stored baseline and new storage."""
import json,os
import h5py,numpy as np
from morqex import Workspace
ws=Workspace(os.environ['MORQEX_WORKSPACE']);root=ws.path('results/top_damage')
errors={}
with h5py.File(root/'static_mesh_v1/level4/fields.h5') as a,h5py.File(root/'static_mesh_storage_v2/level4/fields.h5') as b:
    for key in ('points_m','cells'):
        if not np.array_equal(a[key][:],b[key][:]):raise ValueError('Reference discretization changed')
    for key in ('nodal_damage','nodal_displacement_m','voxel/stress_Pa'):
        x=a[key][:];y=b[key][:];errors[key]=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30))
old=json.loads(ws.path('runs/20261127T041925.907485Z_static_mesh_level4/manifest.json').read_text())
new=json.loads(ws.path('runs/20261127T042756.523974Z_static_mesh_storage4/manifest.json').read_text())
result={'passed':max(errors.values())<1e-8,'relative_L2_errors':errors,
        'old_seconds':old['wall_seconds'],'new_seconds':new['wall_seconds'],
        'old_peak_RSS_bytes':old['peak_group_rss_bytes_sampled'],'new_peak_RSS_bytes':new['peak_group_rss_bytes_sampled'],
        'speed_ratio':old['wall_seconds']/new['wall_seconds'],
        'memory_reduction_fraction':1-new['peak_group_rss_bytes_sampled']/old['peak_group_rss_bytes_sampled'],
        'scope':'Single measured static same-problem comparison including setup, solve and field output; not lifetime acceleration'}
ws.path('reports/finite_storage_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit(1)
