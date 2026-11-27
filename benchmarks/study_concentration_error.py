"""All declared concentration refinements against the full active-voxel run."""
import json,os
from compare_network_cycles import compare
from morqex.workspace import Workspace
ws=Workspace(os.environ['MORQEX_WORKSPACE'])
root=ws.path('results/top_damage/network_cycles');results={}
for name in ('regions_full_local','regions128_local','regions64_local','regions_grain_local'):
    result=compare(root/('dual_'+name+'_v1'),root/'dual_regions_full_batched_v1')
    results[name]=result
    ws.path('reports','concentration_'+name+'.json').write_text(json.dumps(result,indent=2))
ws.path('reports/concentration_refinement.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
if not all(r['passed'] for r in results.values()):raise SystemExit(1)
