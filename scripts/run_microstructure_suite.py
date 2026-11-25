"""Serial, resumable microstructure/DFN study; children retain local caps."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WS=ROOT.parent/'morph_workspace'
CASES=[
 ('full4_1c','--regions 0'),
 ('mean4_1c','--regions 2'),
 ('rom8_1c','--regions 8'),
 ('rom16_1c','--regions 16'),
 ('rom24_1c','--regions 24'),
 ('mechanics16_1c','--regions 16 --strategy mechanics'),
 ('oneway16_1c','--regions 16 --feedback 0'),
 ('rom16_halfC','--regions 16 --rate 0.5'),
 ('rom16_2C','--regions 16 --rate 2'),
 ('rom16_fixed','--regions 16 --constraint fixed-thickness'),
 ('rom16_nx6','--regions 16 --nx 6'),
 ('rom16_nx12','--regions 16 --nx 12'),
 ('large16','--regions 16 --refine 3'),
 ('familyB16','--regions 16 --seed 108'),
 ('full3_cycles','--shape 3 --regions 0 --protocol cycles --cycles 3'),
 ('rom3_cycles','--shape 3 --regions 8 --protocol cycles --cycles 3'),
 ('rom4_cycles','--regions 12 --protocol cycles --cycles 3'),
]

def reusable(path,latest,arguments):
    if not path.exists() or not latest.exists():return False
    data=json.loads(path.read_text());manifest=json.loads(latest.read_text())
    if manifest.get('exit_code')!=0 or not all(data['checks'].values()):return False
    expected=[str(ROOT/'benchmarks/study_microstructure_dfn.py'),'--case',data['case'],*arguments.split()]
    if manifest['command'][1:]!=expected:return False
    return all((ROOT/name).is_file() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in data['source_sha256'].items())

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cases',nargs='*');args=parser.parse_args()
    outcomes=[]
    for name,arguments in CASES:
        if args.cases and name not in args.cases:continue
        latest=WS/'reports'/f'micro_{name}_latest.json';path=WS/'results/microstructure_dfn'/name/'result.json'
        cached=reusable(path,latest,arguments)
        if not cached:
            print('RUN',name,flush=True)
            command=[sys.executable,str(ROOT/'scripts/run_local.py'),'--label','micro_'+name,'--',sys.executable,
                str(ROOT/'benchmarks/study_microstructure_dfn.py'),'--case',name,*arguments.split()]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            if result.returncode:
                print(result.stdout[-4000:],result.stderr[-2000:],flush=True)
                raise SystemExit(f'{name} failed; retained evidence, no further numerical jobs')
        manifest=json.loads(latest.read_text())
        outcomes.append({'case':name,'success':True,'reused':cached,'manifest':manifest})
        (WS/'reports/microstructure_suite.json').write_text(json.dumps(outcomes,indent=2))
        print(name,'REUSED' if cached else 'COMPLETED',round(manifest['wall_seconds'],3),flush=True)

if __name__=='__main__':main()
