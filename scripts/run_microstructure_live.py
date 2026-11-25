"""JSONL interface to the actual monolithic microstructure/DFN runtime.

Run through run_local. Default demo: discharge, rest, charge, recover and save.
With --interactive, send one JSON command per line, e.g.
{"action":"advance","dt":20,"current_A":5}
{"action":"recover","domain":"negative","x_index":1}
{"action":"checkpoint"}
{"action":"quit"}
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib
import numpy as np
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
from morqex.electrochemistry.microstructure_runtime import MicrostructureDFNRuntime

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',default='rom16_1c');parser.add_argument('--interactive',action='store_true');args=parser.parse_args()
    ws=Path(os.environ['MORQEX_WORKSPACE']);record=json.loads((ws/'results/microstructure_dfn'/args.case/'result.json').read_text())
    for name,value in record['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value:raise ValueError('Rebuild case after physics changes')
    cfg=tomllib.loads((ROOT/'configs/microstructure_dfn.toml').read_text());p,_=microstructure_parameters()
    for d in ['Negative','Positive']:
        name=f'{d} electrode partial molar volume [m3.mol-1]';p.update({name:p[name]*cfg['partial_molar_volume_scale']})
    runtime=MicrostructureDFNRuntime({d:m['operator_path'] for d,m in record['microstructure'].items()},p,nx=record['arguments']['nx'],
        feedback=record['arguments']['feedback'],constraint=record['arguments']['constraint'])
    output=Path.cwd();counter=0
    def execute(command):
        nonlocal counter
        action=command['action']
        if action=='advance':result=runtime.advance(float(command['dt']),float(command['current_A']))
        elif action=='recover':
            d=command.get('domain','negative');field=runtime.recover(d,x_index=int(command.get('x_index',0)),selection=command.get('selection'))
            counter+=1;path=output/f'live_field_{counter:03d}.npz';np.savez_compressed(path,**field)
            result={'field_npz':str(path),'wall_seconds':field['wall_seconds'],'concentration_range':np.quantile(field['concentration'],[0,1]).tolist()}
        elif action=='checkpoint':result={'checkpoint':str(runtime.checkpoint(output/'checkpoint'))}
        elif action=='bound':result=runtime.reaction_bound(command.get('domain','negative'),x_index=int(command.get('x_index',0)))
        else:raise ValueError('Unknown action')
        print(json.dumps({'action':action,**result},default=lambda x:np.asarray(x).tolist()),flush=True)
    if args.interactive:
        for line in sys.stdin:
            command=json.loads(line)
            if command['action']=='quit':break
            execute(command)
    else:
        for dt,current in [(600,5),(20,5),(20,0),(20,-2.5),(20,5)]:execute({'action':'advance','dt':dt,'current_A':current})
        execute({'action':'bound'});execute({'action':'recover','domain':'negative','x_index':1});execute({'action':'checkpoint'})

if __name__=='__main__':main()
