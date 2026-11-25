"""Serial, source-bound upgrade cases; valid rejections are never counted passes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];WS=ROOT.parent/'morph_workspace'
WINDOW='--protocol cycles --cycles 3 --rate 0.1 --lower-voltage 3.5 --upper-voltage 4.0'
CASES=[
 ('goal_ref42','micro','--regions 0',None),
 ('goal_geo16','micro','--regions 16',None),
 ('goal_port16','micro','--regions 16 --strategy port',None),
 ('goal_port24','micro','--regions 24 --strategy port',None),
 ('goal_port32','micro','--regions 32 --strategy port',None),
 ('goal_tail16','micro','--regions 16 --strategy tail',None),
 ('goal_tail32','micro','--regions 32 --strategy tail',None),
 ('goal_tail40','micro','--regions 40 --strategy tail',None),
 ('ct_none','aging',WINDOW+' --mechanisms none',None),
 ('ct_sei','aging',WINDOW+' --mechanisms sei',None),
 ('ct_damage','aging',WINDOW+' --mechanisms damage --damage-threshold 100',None),
 ('ct_all','aging',WINDOW+' --damage-threshold 100',None),
 ('ct_pulse','aging','--protocol pulse --duration 2 --temperature 263.15 --initial-state mid --thin-electrodes',None),
 ('ct_short','aging','--duration 120 --damage-threshold 100',None),
 ('ct_tight','aging','--duration 120 --damage-threshold 100 --rtol 3e-8 --atol 3e-9',None),
 ('ct_exact','aging','--basis exact --mechanisms none --duration 6000 --rate 0.1',None),
 ('ct_rom4','aging','--regions 4 --mechanisms none --duration 6000 --rate 0.1',None),
 ('ct_rom8','aging','--regions 8 --mechanisms none --duration 6000 --rate 0.1',None),
 ('ct_larger','aging','--crop-size 6 --regions 8 --basis geometry --duration 120 --rate 0.1 --mechanisms none',None),
 ('ct_high_current','aging','--protocol cycles --cycles 1','electrolyte validity floor'),
 ('ct_original_window','aging','--protocol cycles --cycles 1 --rate 0.1','upper concentration'),
]

def current_hashes(data):return all((ROOT/n).is_file() and hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in data['source_sha256'].items())

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cases',nargs='*');args=parser.parse_args();outcomes=[]
    for name,kind,arguments,rejection in CASES:
        if args.cases and name not in args.cases:continue
        script=ROOT/'benchmarks'/('study_degradation.py' if kind=='aging' else 'study_microstructure_dfn.py')
        folder=WS/'results'/('aging' if kind=='aging' else 'microstructure_dfn')/name
        artifact=folder/('rejected_protocol.json' if rejection else 'result.json');latest=WS/'reports'/f'upgrade_{name}_latest.json'
        command=[sys.executable,str(script),'--case',name,*arguments.split()];reuse=False
        if artifact.exists() and latest.exists():
            data=json.loads(artifact.read_text());m=json.loads(latest.read_text())
            reuse=m['command']==command and current_hashes(data) and (rejection in data.get('termination','') if rejection else m['exit_code']==0 and all(data['checks'].values()))
        if not reuse:
            print('RUN',name,flush=True)
            run=subprocess.run([sys.executable,str(ROOT/'scripts/run_local.py'),'--label','upgrade_'+name,'--',*command],cwd=ROOT,capture_output=True,text=True)
            m=json.loads(latest.read_text())
            if rejection:
                if not artifact.exists() or rejection not in json.loads(artifact.read_text()).get('termination','') or run.returncode==0:
                    print(run.stdout[-2000:],run.stderr[-3000:],flush=True);raise SystemExit(name+': expected physical rejection was not demonstrated')
            elif run.returncode:
                print(run.stdout[-2000:],run.stderr[-3000:],flush=True);raise SystemExit(name+': numerical gate/resource failure; stop suite')
        row={'case':name,'kind':kind,'status':'expected-physical-rejection' if rejection else 'numerical-gates-pass','reused':reuse,'manifest':m,'artifact':str(artifact)}
        outcomes.append(row);(WS/'reports/upgrade_suite.json').write_text(json.dumps(outcomes,indent=2))
        print(name,row['status'],round(m['wall_seconds'],3),flush=True)

if __name__=='__main__':main()
