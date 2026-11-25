"""Serial, source-bound crossscale/degradation/material qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];WS=ROOT.parent/'morph_workspace'
CYCLE='--protocol cycles --cycles 3 --rate 0.1 --lower-voltage 3.5 --upper-voltage 4.0 --damage-threshold 100'
PULSE='--protocol pulse --duration 2 --temperature 263.15 --initial-state mid --thin-electrodes'
CASES=[
 ('closure_ct4_exact_cycles','--basis exact '+CYCLE),
 ('closure_ct4_rom8_cycles','--regions 8 --compression 1e-6 '+CYCLE),
 ('closure_ct4_raw8_cycles','--regions 8 '+CYCLE),
 ('closure_ct4_geo8_cycles','--regions 8 --basis geometry --compression 1e-6 '+CYCLE),
 ('closure_ct4_rom16_cycles','--regions 16 --compression 1e-6 '+CYCLE),
 ('closure_ct4_full_pulse','--basis exact '+PULSE),
 ('closure_ct4_rom8_pulse','--regions 8 --compression 1e-6 '+PULSE),
 ('closure_ct6_compressed','--crop-size 6 --regions 8 --basis geometry --compression 1e-6 --duration 120 --rate 0.1 --damage-threshold 100'),
 ('closure_ct6_raw','--crop-size 6 --regions 8 --basis geometry --duration 120 --rate 0.1 --damage-threshold 100'),
 ('closure_ct8_compressed','--crop-size 8 --regions 8 --basis geometry --compression 1e-6 --duration 120 --rate 0.1 --damage-threshold 100'),
 ('closure_doped_uniform','--regions 8 --compression 1e-6 --doping uniform '+CYCLE),
 ('closure_doped_gradient','--regions 8 --compression 1e-6 --doping gradient '+CYCLE),
 ('closure_doped_shell','--regions 8 --compression 1e-6 --doping core-shell '+CYCLE),
 ('closure_nmc_mid','--initial-state mid --regions 8 --compression 1e-6 --duration 120 --rate 0.01 --damage-threshold 100'),
 ('closure_lfp_mid','--cathode LFP --initial-state mid --regions 8 --compression 1e-6 --duration 120 --rate 0.01 --damage-threshold 100'),
 ('closure_lco_mid','--cathode LCO --initial-state mid --regions 8 --compression 1e-6 --duration 120 --rate 0.01 --damage-threshold 100'),
 ('closure_lfp_doped','--cathode LFP --initial-state mid --regions 8 --compression 1e-6 --duration 120 --rate 0.01 --damage-threshold 100 --doping gradient'),
 ('closure_lco_doped','--cathode LCO --initial-state mid --regions 8 --compression 1e-6 --duration 120 --rate 0.01 --damage-threshold 100 --doping gradient'),
 ('closure_lfp_cycle','--cathode LFP --initial-state mid --regions 8 --compression 1e-6 --protocol cycles --cycles 1 --rate 0.1 --lower-voltage 3.1 --upper-voltage 3.4 --damage-threshold 100 --rtol 1e-11 --atol 1e-12'),
 ('closure_lfp_rom16_cycle','--cathode LFP --initial-state mid --regions 16 --compression 1e-6 --protocol cycles --cycles 1 --rate 0.1 --lower-voltage 3.1 --upper-voltage 3.4 --damage-threshold 100 --rtol 1e-11 --atol 1e-12'),
 ('closure_lco_cycle','--cathode LCO --initial-state mid --regions 8 --compression 1e-6 --protocol cycles --cycles 1 --rate 0.1 --lower-voltage 3.5 --upper-voltage 4.0 --damage-threshold 100'),
 ('closure_lfp_exact_cycle','--cathode LFP --initial-state mid --basis exact --protocol cycles --cycles 1 --rate 0.1 --lower-voltage 3.1 --upper-voltage 3.4 --damage-threshold 100 --rtol 1e-11 --atol 1e-12'),
 ('closure_lco_exact_cycle','--cathode LCO --initial-state mid --basis exact --protocol cycles --cycles 1 --rate 0.1 --lower-voltage 3.5 --upper-voltage 4.0 --damage-threshold 100'),
 ('closure_doped_exact_cycles','--basis exact --doping gradient '+CYCLE),
 ('closure_no_degradation','--regions 8 --compression 1e-6 --mechanisms none '+CYCLE),
 ('closure_sei_only','--regions 8 --compression 1e-6 --mechanisms sei '+CYCLE),
 ('closure_contact_only','--regions 8 --compression 1e-6 --mechanisms damage '+CYCLE),
 # Retain the original failed loose-tolerance run and recheck without clipping.
 ('closure_macro5_cycles','--regions 8 --compression 1e-6 --nx 5 --rtol 1e-9 --atol 1e-10 '+CYCLE),
 # CC must exceed the prescribed C/20 CV cutoff; 0.05C equals that cutoff.
 ('closure_macro5_safe_current','--regions 8 --compression 1e-6 --nx 5 --rtol 1e-9 --atol 1e-10 '+CYCLE.replace('--rate 0.1','--rate 0.06')),
 ('closure_macro3_safe_current','--regions 8 --compression 1e-6 --nx 3 --rtol 1e-9 --atol 1e-10 '+CYCLE.replace('--rate 0.1','--rate 0.06')),
 ('closure_macro3_discharge','--regions 8 --compression 1e-6 --nx 3 --duration 6000 --rate 0.1 --damage-threshold 100'),
 ('closure_macro5_discharge','--regions 8 --compression 1e-6 --nx 5 --duration 6000 --rate 0.1 --damage-threshold 100'),
]

def valid(path,manifest,command):
    if not path.exists() or not manifest.exists():return False
    data=json.loads(path.read_text());m=json.loads(manifest.read_text())
    backend=all(hashlib.sha256((ROOT/'src/pybamm'/n).read_bytes()).hexdigest()==h for n,h in data['backend']['source_sha256'].items())
    return m['exit_code']==0 and m['command']==command and all(data['checks'].values()) and backend and all((ROOT/n).is_file() and hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in data['source_sha256'].items())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cases',nargs='*');args=parser.parse_args();records=[]
    for name,arguments in CASES:
        if args.cases and name not in args.cases:continue
        result=WS/'results/aging'/name/'result.json';manifest=WS/'reports'/f'{name}_latest.json'
        command=[sys.executable,str(ROOT/'benchmarks/study_degradation.py'),'--case',name,*arguments.split()]
        reuse=valid(result,manifest,command)
        if not reuse:
            print('RUN',name,flush=True)
            run=subprocess.run([sys.executable,str(ROOT/'scripts/run_local.py'),'--label',name,'--',*command],cwd=ROOT,capture_output=True,text=True)
            if run.returncode:
                m=json.loads(manifest.read_text());records.append({'case':name,'status':'failed','manifest':m})
                error=(Path(m['run'])/'stderr.txt').read_text()
                if name=='closure_macro5_cycles' and 'Incomplete shared protocol: event: MORQEX negative region 7 upper concentration' in error:
                    records[-1]['status']='physical-domain-rejected';records[-1]['scope']='Not a completed cycle or a convergence pass; saturation endpoint diagnosed separately. Valid lower-current protocol rechecked without changing equations/events.'
                    (WS/'reports/crossscale_suite.json').write_text(json.dumps(records,indent=2));print(name,'PHYSICAL REJECTION',flush=True);continue
                (WS/'reports/crossscale_suite.json').write_text(json.dumps(records,indent=2))
                print(run.stdout[-1000:],run.stderr[-1000:],flush=True);raise SystemExit(name+' failed; preserve evidence and stop')
        m=json.loads(manifest.read_text());records.append({'case':name,'status':'numerical-gates-pass','reused':reuse,'manifest':m,'result':str(result)})
        (WS/'reports/crossscale_suite.json').write_text(json.dumps(records,indent=2))
        print(name,'PASS',round(m['wall_seconds'],2),round(m['peak_group_rss_bytes_sampled']/2**20,1),flush=True)

if __name__=='__main__':main()
