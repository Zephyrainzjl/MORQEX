"""Sequential bounded slices; fail immediately instead of skipping a failed case."""
import argparse,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
workspace=root.parent/'morph_workspace'
parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(root/'configs/top_damage_dense_cycle.toml'));parser.add_argument('--suffix',default='v2');parser.add_argument('--steps',type=int,default=2);args=parser.parse_args()
config=Path(args.config)
for environment in ('gentle','fast','released'):
    output=environment+'_'+args.suffix
    path=workspace/'results/top_damage/dense_cycles'/output/'manifest.json'
    for part in range(120):
        if path.exists() and json.loads(path.read_text()).get('protocol_complete'):break
        command=[sys.executable,str(root/'scripts/run_local.py'),'--label',f'dense_{environment}_{part:02d}','--',
                 sys.executable,'-m','morqex.top_damage','dense-cycle',str(config),'--environment',environment,'--steps',str(args.steps),'--output',output]
        run=subprocess.run(command,capture_output=True,text=True)
        if run.returncode:
            print(run.stdout,flush=True);print(run.stderr,flush=True);raise SystemExit(run.returncode)
        status=json.loads(path.read_text())
        print(json.dumps({'environment':environment,'accepted_step':status['last_accepted_step'],
                          'total_steps':status['requested_steps'],'complete':status['protocol_complete']}),flush=True)
    else:raise RuntimeError('Slice budget exhausted without protocol completion')
print('All requested numerical protocols completed; physical topology/publication gates remain separate.',flush=True)
