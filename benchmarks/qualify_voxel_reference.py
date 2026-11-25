import json
import os
from pathlib import Path
from morqex.acceleration.verification import manufactured_periodic

if __name__=='__main__':
    rows=[manufactured_periodic(n) for n in [8,16,32,64]]
    orders=[__import__('math').log(a['displacement_relative_L2']/b['displacement_relative_L2'],2) for a,b in zip(rows,rows[1:])]
    record={'results':rows,'observed_displacement_orders':orders,'passed':all(1.9<p<2.1 for p in orders) and max(r['stress_relative_L2'] for r in rows)<1e-8 and max(r['relative_energy_error'] for r in rows)<1e-8}
    (Path(os.environ['MORQEX_WORKSPACE'])/'reports/voxel_reference_qualification.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
    if not record['passed']:raise RuntimeError('Manufactured reference qualification failed')
