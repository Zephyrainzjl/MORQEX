"""Serial bounded jobs for five distinct complete NLR CT volumes."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    for dataset in ['nmc1uncal','nmc1cal','nmc2uncal','nmc2cal','nmc3uncal']:
        for action,script in [('geometry','prepare_full_electrode.py'),('concentration','project_full_concentration.py')]:
            command=[sys.executable,str(ROOT/'scripts/run_local.py'),'--label',f'full_{dataset}_{action}','--',sys.executable,str(ROOT/'scripts'/script),'--dataset',dataset]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            print(json.dumps({'dataset':dataset,'action':action,'exit_code':result.returncode,'launcher':result.stdout[-2000:],'stderr':result.stderr[-2000:]}),flush=True)
            if result.returncode:raise SystemExit(result.returncode)
