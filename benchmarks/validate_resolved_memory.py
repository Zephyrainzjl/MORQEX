"""Repeat full finite-fatigue states in one process; compare isolated reference."""
import argparse,json,os,resource,time,subprocess,sys
from pathlib import Path
import h5py,numpy as np
from morqex import Workspace
from morqex.top_damage.resolved_phase_field import advance_resolved_phase_field
p=argparse.ArgumentParser();p.add_argument('--reference',default='resolved_particles_dt025');p.add_argument('--output',default='resolved_memory_validation');p.add_argument('--frames',type=int,default=9);p.add_argument('--seeded-isolated',action='store_true');args=p.parse_args()
if not 1<=args.frames<=20:raise ValueError('Bounded 1..20 frame validation required')
ws=Workspace(os.environ['MORQEX_WORKSPACE']);ref=ws.path('results/top_damage',args.reference);root=ws.path('results/top_damage',args.output)
if root.exists():raise ValueError('Choose a new memory-validation directory')
root.mkdir(parents=True);tick=time.perf_counter();rows=[]
for i in range(args.frames):
    path=root/f'mechanics_{i:05d}.h5'
    original_path=ref/path.name
    if args.seeded_isolated:
        baseline=root/'isolated';baseline.mkdir(exist_ok=True);original_path=baseline/path.name
        # PyAMG uses NumPy's RNG in its spectral-radius estimate. Control the
        # same seed in each fresh process and each in-process construction, so
        # an object-lifetime comparison is not confounded by another AMG basis.
        code='import numpy as np,sys;np.random.seed(int(sys.argv[6]));from morqex.top_damage.resolved_phase_field import advance_resolved_phase_field;advance_resolved_phase_field(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],previous=sys.argv[5] or None)'
        with (baseline/f'mechanics_{i:05d}.solver.log').open('w') as log:
            subprocess.run([sys.executable,'-c',code,str(ref/'initial.h5'),str(ref/f'fields_{i:05d}.h5'),str(original_path),str(ws.root),str(baseline/f'mechanics_{i-1:05d}.h5') if i else '',str(104+i)],stdout=log,stderr=subprocess.STDOUT,check=True)
        np.random.seed(104+i)
    r=advance_resolved_phase_field(ref/'initial.h5',ref/f'fields_{i:05d}.h5',path,ws,previous=root/f'mechanics_{i-1:05d}.h5' if i else None)
    errors={}
    with h5py.File(path) as f,h5py.File(original_path) as original:
        for key in ['displacement_m','damage','cauchy_stress_Pa','fatigue_J_m3','voxel_mechanical_potential_J_mol']:
            a=f[key][:];b=original[key][:];errors[key]=float(np.max(abs(a-b))/max(float(np.max(abs(b))),1.))
    rows.append({'time_s':r['time_s'],'relative_or_unit_scaled_errors':errors,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)})
    print(json.dumps(rows[-1]),flush=True)
result={'frames':rows,'seconds':time.perf_counter()-tick,'seeded_isolated':args.seeded_isolated,'passed':max(max(r['relative_or_unit_scaled_errors'].values()) for r in rows)<1e-7,'scope':'Full-lattice in-process versus isolated-process history. Seeded mode controls the AMG spectral initialization; no manual garbage collection or numerical model reduction.'}
(root/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['passed']
