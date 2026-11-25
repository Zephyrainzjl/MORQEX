"""Full-volume ionic/electronic cell problems, one budgeted GPU job at a time."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    for dataset in ['nmc1uncal','nmc1cal','nmc2uncal','nmc2cal','nmc3uncal']:
        for kind in ['electrolyte','electrode']:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_local.py'),'--label',f'full_{dataset}_{kind}','--gpu','--',sys.executable,str(ROOT/'benchmarks/study_full_voxel_transport.py'),'--dataset',dataset,'--kind',kind],cwd=ROOT,capture_output=True,text=True)
            print(json.dumps({'dataset':dataset,'kind':kind,'exit_code':result.returncode,'launcher':result.stdout[-2000:],'stderr':result.stderr[-2000:]}),flush=True)
            if result.returncode:raise SystemExit(result.returncode)
