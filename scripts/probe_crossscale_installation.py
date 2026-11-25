"""Independent wheel proof for the LFP/dopant/aging API and field recovery."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import numpy as np
source=Path(__file__).resolve().parents[1]/'src';sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=source]
import morqex
import pybamm as pb
from morqex.electrochemistry.cathodes import cathode_parameters,HOSTS
from morqex.electrochemistry.aging_cache import load_aging_kernel,AgingLocalization
from morqex.electrochemistry.aging_dfn import build_aging_dfn,age
from morqex.electrochemistry.microstructure_dfn import MicrostructureStack,key
from morqex.degradation import DegradationOptions,EnergyDrivenDebonding
from morqex.solvers.pybamm_backend import solver

assert source not in Path(morqex.__file__).resolve().parents and source not in Path(pb.__file__).resolve().parents
assert metadata.version('morqex-research')==morqex.__version__=='0.0.9'
try:metadata.distribution('pybamm')
except metadata.PackageNotFoundError:pass
else:raise AssertionError('A separate upstream distribution must not overwrite bundled source')
w=Path(os.environ['MORQEX_WORKSPACE']);paths={d:w/'operators/aging/closure_lfp_doped'/d for d in ['negative','positive']}
kernels={d:load_aging_kernel(p) for d,p in paths.items()};p,audit=cathode_parameters('LFP')
p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15,'Current function [A]':.023})
for d in ['negative','positive']:
    volume=f'{d.capitalize()} electrode partial molar volume [m3.mol-1]'
    p.update({volume:p[volume]*.1,f'Initial concentration in {d} electrode [mol.m-3]':kernels[d].operator.reference_concentration})
stack=MicrostructureStack(kernels['negative'].operator,kernels['positive'].operator,p);stack.update_parameters(p)
model=build_aging_dfn(stack,kernels,DegradationOptions(damage=EnergyDrivenDebonding(threshold_J_m3=100.)))
sim=pb.Simulation(model,parameter_values=p,solver=solver(),var_pts={**model.default_var_pts,'x_n':3,'x_s':3,'x_p':3,'r_n':2,'r_p':2})
s=sim.solve([0,30],t_interp=np.linspace(0,30,4));assert s.t[-1]==30.
op=kernels['positive'].operator;c=np.array([s[key('positive',k,'concentration [mol.m-3]')].entries[1,-1] for k in range(op.dimension)])
local=AgingLocalization(paths['positive']);f=local.recover(c,np.array(s['MORQEX positive axial strain'].entries[1,-1]),np.array(s['Volume-averaged cell temperature [K]'].entries[-1]-op.metadata()['reference_temperature_K']),np.array(s[age('positive','damage')].entries[1,-1]))
assert np.isfinite(f['stress']).all() and op.regional_chemistry.fraction.max()>0
result={'version':'0.0.9','source_imported':False,'upstream_distribution_installed':False,'host_packs':HOSTS,'tested_host':'LFP',
    'dopant_profile_fingerprint':op.regional_chemistry.fingerprint,'seconds':30.,'active_cells':len(f['concentration']),'pybamm_path':pb.__file__}
(w/'reports/crossscale_installed_wheel.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
