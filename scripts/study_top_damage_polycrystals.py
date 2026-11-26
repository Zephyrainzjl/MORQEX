"""Run independent seeds of the sparse large-particle geometry check."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from morqex import Workspace
from morqex.top_damage.morphology import PolycrystalSpec, create_polycrystal

ws=Workspace(os.environ['MORQEX_WORKSPACE'])
root=Path(__file__).resolve().parents[1]
base=PolycrystalSpec.from_toml(root/'configs/top_damage_few_particles.toml')
cases=[]
for seed in (104,105,106):
    spec=replace(base,seed=seed,name=base.name if seed==104 else base.name+'_seed'+str(seed))
    config=ws.path('configs','top_damage',spec.name+'.toml');config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text('schema_version = 1\n[microstructure]\n'+
                      '\n'.join(key+' = '+json.dumps(value) for key,value in asdict(spec).items())+'\n')
    result=create_polycrystal(config,ws)
    metrics=json.loads((Path(result['directory'])/'metrics.json').read_text())
    cases.append({'seed':seed,**metrics,'directory':result['directory']})
summary={'scope':'Three independent initial numerical morphologies; not three calibrated material chemistries or degradation histories.',
         'cases':cases,'passed':all(c['particle_count']==4 and c['grain_count']>=32 and
             c['minimum_grain_voxels']>0 and c['networks']['CBD']['spans_xyz'][0] and
             max(abs(v) for v in c['sphere_volume_signed_relative_errors'].values())<.025 for c in cases)}
ws.path('reports/top_damage_polycrystal_study.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'passed':summary['passed'],'directories':[c['directory'] for c in cases]},indent=2))
if not summary['passed']:raise RuntimeError('Initial polycrystal study failed')
