"""Lightweight sequential driver; every numerical child uses run_local limits."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WS=ROOT.parent/'morph_workspace'
CASES=[
 ('comsol', '--mode stock --protocol comsol --parameters Marquis2019 --ocp raw --nx 31 --nr 11'),
 ('comsol_fine', '--mode stock --protocol comsol --parameters Marquis2019 --ocp raw --nx 48 --nr 30'),
 ('ai05', '--mode stock --parameters Ai2020 --ocp raw --rate 0.5'),
 ('ai1', '--mode stock --parameters Ai2020 --ocp raw --rate 1'),
 ('ai2', '--mode stock --parameters Ai2020 --ocp raw --rate 2'),
 ('raw1', '--mode stock --ocp raw'),
 ('convex1', '--mode stock'),
 ('oneway1', '--mode oneway'),
 ('coupled05', '--rate 0.5'),
 ('coupled1', '--rate 1'),
 ('coupled2', '--rate 2'),
 ('coupled_coarse', '--nx 8 --nr 12'),
 ('coupled_fine', '--nx 18 --nr 24'),
 ('coupled_crosscheck', '--backend casadi --protocol crosscheck'),
 ('coupled_crosscheck_idaklu', '--protocol crosscheck'),
 ('casadi_loose_ic', '--backend casadi --protocol crosscheck --algebraic-step-tol 0.0001'),
 ('coupled_fixed', '--constraint fixed-thickness'),
 ('coupled_full_expansion', '--omega-scale 1'),
 ('coupled_cycles', '--protocol cycles'),
]


def main():
    outcomes=[]
    for name,arguments in CASES:
        command=[sys.executable,str(ROOT/'scripts/run_local.py'),'--label','dfn_'+name,'--',sys.executable,str(ROOT/'benchmarks/study_dfn.py'),'--case',name,*arguments.split()]
        print('RUN',name,flush=True)
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        latest=json.loads((WS/'reports'/f'dfn_{name}_latest.json').read_text())
        outcomes.append({'case':name,'success':result.returncode==0,'manifest':latest})
        (WS/'reports/dfn_suite.json').write_text(json.dumps(outcomes,indent=2))
        print(name,'COMPLETED' if result.returncode==0 else 'FAILED',latest['wall_seconds'],flush=True)
        if result.returncode:
            print(result.stderr[-1000:],flush=True)
            raise SystemExit('Numerical job failed; stop before spending the remaining suite budget')
    if not all(r['success'] for r in outcomes):raise SystemExit(1)


if __name__=='__main__':main()
