"""Serial native DFN controls for the exact same generated geometries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
from run_crossscale_suite import ROOT,WS

def main():
    records=[]
    for family,aging,radial in [(f,a,16) for f in ['isotropic','directional','CBD-rich'] for a in [False,True]]+[('isotropic',True,32)]:
        name=f'native_{family}_{"aging" if aging else "bare"}_{radial}'
        result=WS/'results/native_dfn'/f'{family}_{"aging" if aging else "bare"}_{radial}.json';manifest=WS/'reports'/f'{name}_latest.json'
        command=[sys.executable,str(ROOT/'benchmarks/study_geometry_dfn_control.py'),'--family',family,'--radial-cells',str(radial)]+(['--aging'] if aging else [])
        reuse=False
        if result.exists() and manifest.exists():
            r=json.loads(result.read_text());m=json.loads(manifest.read_text())
            reuse=m['exit_code']==0 and m['command']==command and r['driver_sha256']==hashlib.sha256((ROOT/'benchmarks/study_geometry_dfn_control.py').read_bytes()).hexdigest() and r['source_case_sha256']==hashlib.sha256(Path(r['source_case']).read_bytes()).hexdigest() and all(hashlib.sha256((ROOT/'src/pybamm'/n).read_bytes()).hexdigest()==h for n,h in r['backend']['source_sha256'].items())
        if not reuse:
            print('RUN',name,flush=True);run=subprocess.run([sys.executable,str(ROOT/'scripts/run_local.py'),'--label',name,'--',*command],cwd=ROOT,capture_output=True,text=True)
            if run.returncode:print(run.stdout[-1000:],run.stderr[-1000:]);raise SystemExit(name+' failed; diagnose')
        m=json.loads(manifest.read_text());records.append({'case':name,'result':str(result),'manifest':m,'reused':reuse})
        (WS/'reports/native_geometry_suite.json').write_text(json.dumps(records,indent=2));print(name,'PASS',round(m['wall_seconds'],2),flush=True)

if __name__=='__main__':main()
