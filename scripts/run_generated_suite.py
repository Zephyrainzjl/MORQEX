"""Serial independent GEMsim morphology qualification, with strict resume."""
import json
import subprocess
import sys
sys.dont_write_bytecode=True
from run_crossscale_suite import ROOT,WS,valid

PROTOCOL='--protocol cycles --cycles 3 --rate 0.1 --lower-voltage 3.5 --upper-voltage 3.9 --damage-threshold 100 --nx 5'
CASES=[(f'gem_{family}_{basis}',family,('--basis exact' if basis=='full' else '--regions 8 --compression 1e-6')+' '+PROTOCOL+(' --rtol 1e-9 --atol 1e-10' if family=='isotropic' else ' --rtol 1e-7 --atol 1e-8'))
    for family in ['isotropic','directional','CBD-rich'] for basis in ['full','rom8']]
CASES.append(('gem_directional_full_tight_cycle','directional','--basis exact '+PROTOCOL.replace('--cycles 3','--cycles 1')+' --rtol 1e-9 --atol 1e-10'))

def main():
    records=[]
    for name,family,arguments in CASES:
        result=WS/'results/aging'/name/'result.json';manifest=WS/'reports'/f'{name}_latest.json'
        command=[sys.executable,str(ROOT/'benchmarks/study_generated_degradation.py'),'--geometry-family',family,'--case',name,*arguments.split()]
        reuse=valid(result,manifest,command)
        if not reuse:
            print('RUN',name,flush=True)
            run=subprocess.run([sys.executable,str(ROOT/'scripts/run_local.py'),'--label',name,'--',*command],cwd=ROOT,capture_output=True,text=True)
            if run.returncode:
                print(run.stdout[-1200:],run.stderr[-1200:],flush=True);raise SystemExit(name+' failed; diagnose before proceeding')
        m=json.loads(manifest.read_text());records.append({'case':name,'reused':reuse,'result':str(result),'manifest':m})
        (WS/'reports/generated_suite.json').write_text(json.dumps(records,indent=2))
        print(name,'PASS',round(m['wall_seconds'],2),round(m['peak_group_rss_bytes_sampled']/2**20,1),flush=True)

if __name__=='__main__':main()
