"""Serial independent DFN controls for complete CTs, materials and rates."""
import json
import hashlib
import fcntl
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    workspace=ROOT.parent/'morph_workspace';(workspace/'locks').mkdir(parents=True,exist_ok=True)
    lock=(workspace/'locks/full_ct_dfn_suite.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A full CT DFN suite is already running')
    cases=[]
    for data in ['nmc1uncal','nmc1cal','nmc2uncal','nmc2cal','nmc3uncal']:
        for rate in [.1,1.,3.,5.]:cases.append((data,'NMC532',rate,10,20,0))
        cases.append((data,'NMC532',1.,10,20,2))
    for host in ['NMC811','LFP','LCO']:
        for rate in [.1,1.,3.]:cases.append(('nmc1cal',host,rate,10,20,0))
        cases.append(('nmc1cal',host,1.,10,20,2))
    for rate in [.1,3.,5.]:
        for nx,nr in [(20,40),(40,80)]:cases.append(('nmc1cal','NMC532',rate,nx,nr,0))
    records=[]
    for data,host,rate,nx,nr,cycles in cases:
        label=f'fullct_{data}_{host}_{rate:g}C_x{nx}_r{nr}_cy{cycles}'.replace('.','p')
        artifact=workspace/'results/full_ct_dfn'/f'{data}_{host}_{rate:g}C_x{nx}_r{nr}_cycles{cycles}'/'result.json'
        if artifact.exists():
            old=json.loads(artifact.read_text())
            if old.get('completed') and all(Path(path).exists() and hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in old.get('source_sha256',{}).items()) and old.get('source_sha256'):
                records.append({'case':label,'exit_code':0,'reused_current_source_artifact':str(artifact)});print(json.dumps(records[-1]),flush=True);continue
        command=[sys.executable,str(ROOT/'scripts/run_local.py'),'--label',label,'--',sys.executable,str(ROOT/'benchmarks/study_full_ct_dfn.py'),'--dataset',data,'--host',host,'--rate',str(rate),'--nx',str(nx),'--nr',str(nr),'--cycles',str(cycles)]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        record={'case':label,'exit_code':result.returncode,'launcher':result.stdout[-2000:],'stderr':result.stderr[-2000:]};records.append(record);print(json.dumps(record),flush=True)
        if result.returncode and 'Another MORQEX local job' in result.stderr:raise SystemExit('Stop suite: another numerical job holds the resource lock')
    destination=ROOT.parent/'morph_workspace/reports/full_ct_dfn_suite.json';destination.write_text(json.dumps(records,indent=2))
    if any(r['exit_code'] for r in records):raise SystemExit(1)
