"""Reproduce the qualified matrix and offline report with serial bounded jobs.

Requires the documented external environment and staged, checksummed CT data.
The 0.1C five-point saturation case is an explicit rejected applicability test.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
sys.dont_write_bytecode=True

ROOT=Path(__file__).resolve().parents[1];WS=ROOT.parent/'morph_workspace'
def execute(command):
    print(' '.join(map(str,command)),flush=True);subprocess.run(list(map(str,command)),cwd=ROOT,check=True)
def limited(label,command):execute([sys.executable,ROOT/'scripts/run_local.py','--label',label,'--',*command])
def main():
    for name in ['run_crossscale_suite.py','run_generated_suite.py','run_geometry_dfn_controls.py']:execute([sys.executable,ROOT/'scripts'/name])
    diagnostic=WS/'reports/protocol_diagnosis.json';valid=False
    if diagnostic.exists():
        record=json.loads(diagnostic.read_text());valid='source_sha256' in record and all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in record['source_sha256'].items())
    if not valid:
        command=json.loads((WS/'reports/closure_macro5_cycles_latest.json').read_text())['command']
        command[1]=str(ROOT/'benchmarks/diagnose_protocol.py');command[command.index('--case')+1]='diagnose_macro5'
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_local.py'),'--label','diagnose_macro5','--',*command],cwd=ROOT)
        if result.returncode==0 or not diagnostic.exists():raise RuntimeError('Expected saturation diagnostic was not reproduced; review physical qualification')
        record=json.loads(diagnostic.read_text())
        if record['termination']!='event: MORQEX negative region 7 upper concentration':raise RuntimeError('Unexpected diagnostic termination')
        if not all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in record['source_sha256'].items()):raise RuntimeError('Stale saturation diagnostic after failed retry')
    limited('crossscale_tests',[sys.executable,'-m','pytest',ROOT/'tests','-q','--junitxml=test_results.xml',f'--basetemp={WS}/tmp/crossscale_tests'])
    limited('crossscale_kernel_benchmark',[sys.executable,ROOT/'benchmarks/benchmark_crossscale_kernels.py'])
    wheel=WS/'dist/morqex_research-0.0.9-py3-none-any.whl';independent=WS/'venvs/morqex_crossscale/bin/python'
    limited('crossscale_wheel',[sys.executable,'-m','pip','wheel',ROOT,'--no-deps','--no-build-isolation','--wheel-dir',WS/'dist'])
    limited('crossscale_venv',[sys.executable,'-m','venv','--system-site-packages',independent.parent.parent])
    execute([sys.executable,ROOT/'scripts/run_local.py','--label','crossscale_install','--dependency-install','--',independent,'-m','pip','install',str(wheel)+'[dfn,ct]','--no-build-isolation'])
    execute([sys.executable,ROOT/'scripts/run_local.py','--label','crossscale_install_source','--dependency-install','--',independent,'-m','pip','install','--force-reinstall','--no-deps',wheel])
    limited('crossscale_wheel_probe',[independent,ROOT/'scripts/probe_crossscale_installation.py'])
    limited('crossscale_report',[sys.executable,ROOT/'scripts/build_crossscale_report.py'])
    limited('crossscale_report_check',[sys.executable,ROOT/'scripts/check_crossscale_report.py'])
    limited('crossscale_integrity',[sys.executable,ROOT/'scripts/check_integrity.py'])

if __name__=='__main__':main()
