"""Construct all five complete-CT spatial interfaces as serial limited jobs."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    for dataset in ['nmc1uncal','nmc1cal','nmc2uncal','nmc2cal','nmc3uncal']:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_local.py'),'--label','spatial_ports_'+dataset,'--',sys.executable,str(ROOT/'scripts/build_full_spatial_ports.py'),'--dataset',dataset],check=True)
