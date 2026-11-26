"""Carry every saved resolved concentration through accepted finite-fatigue history."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from morqex import Workspace

p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--start',type=int,default=0);p.add_argument('--steps',type=int,default=64);args=p.parse_args()
ws=Workspace(os.environ['MORQEX_WORKSPACE']);root=ws.path(args.directory);fields=sorted(root.glob('fields_*.h5'))
if not fields or not 0<=args.start<len(fields) or not 1<=args.steps<=64:raise ValueError('Existing field sequence and bounded step count required')
records=[]
for i in range(args.start,min(len(fields),args.start+args.steps)):
    # Isolate each native/AMG lifetime: accepted history is in HDF5, not retained
    # Python solver closures or allocator arenas. One child at a time, same
    # inherited CPU affinity, nice level, resource watchdog and source snapshot.
    code="from morqex.top_damage.resolved_phase_field import advance_resolved_phase_field;import sys,json;print(json.dumps(advance_resolved_phase_field(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],previous=sys.argv[5] or None)))"
    log=root/f'mechanics_{i:05d}.solver.log'
    with log.open('w') as handle:
        subprocess.run([sys.executable,'-c',code,str(root/'initial.h5'),str(fields[i]),str(root/f'mechanics_{i:05d}.h5'),str(ws.root),str(root/f'mechanics_{i-1:05d}.h5') if i else ''],check=True,stdout=handle,stderr=subprocess.STDOUT,text=True)
    record=json.loads(log.read_text().strip().splitlines()[-1])
    records.append(record);print(json.dumps(record),flush=True)
(root/f'phase_bridge_records_{args.start:05d}.json').write_text(json.dumps(records,indent=2))
