"""Installed-wheel proof with no upstream PyBaMM distribution or source path."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import numpy as np

# Launcher uses source for ordinary research jobs; remove it for this probe.
source=Path(__file__).resolve().parents[1]/'src'
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=source]
import morqex
import pybamm as pb
assert source not in Path(pb.__file__).resolve().parents
assert source not in Path(morqex.__file__).resolve().parents
try:metadata.distribution('pybamm')
except metadata.PackageNotFoundError:pass
else:raise AssertionError('Probe requires no upstream PyBaMM distribution')
assert metadata.version('morqex-research')=='0.0.6'
assert pb.MORQEX_PATCH_LEVEL=='scalar-1d-interpolant-derivatives-v1'
assert (Path(pb.__file__).parent/'LICENSE.txt').is_file()
from morqex import Workspace
from morqex.electrochemistry.parameters import convex_parameters
from morqex.electrochemistry.mechanical_closure import ElectrodeClosure,AxialStackClosure
from morqex.electrochemistry.dfn import build_model
from morqex.solvers.pybamm_backend import solver
p,audit=convex_parameters()
for domain in ['Negative','Positive']:
    k=f'{domain} electrode partial molar volume [m3.mol-1]';p.update({k:p[k]*.1})
ws=Workspace(Path(os.environ['MORQEX_WORKSPACE']))
a=ElectrodeClosure.from_gemsim(ws,p,'negative',shape=(3,3,3));b=ElectrodeClosure.from_gemsim(ws,p,'positive',shape=(3,3,3),seed=205)
model=build_model(AxialStackClosure(a,b,p))
sim=pb.Simulation(model,parameter_values=p,solver=solver(),var_pts={**model.default_var_pts,'x_n':4,'x_s':4,'x_p':4,'r_n':6,'r_p':6})
sol=sim.solve([0,60],t_interp=np.linspace(0,60,7))
assert sol.t[-1]==60
assert np.ptp(sol['Total lithium [mol]'].entries)<1e-12
assert max(abs(sol['MORQEX thermal balance residual [J.m-3]'].entries))<.1
# Parameter entry points and interpolation CSVs must come from our wheel.
assert pb.ParameterValues('Ai2020')['Nominal cell capacity [A.h]']>0
print(json.dumps({'framework':morqex.__version__,'pybamm_module':pb.__file__,'upstream_distribution_installed':False,'compiled_patch':pb.MORQEX_PATCH_LEVEL,'DFN_coupled_seconds':60,'end_voltage_V':float(sol['Voltage [V]'].entries[-1]),'external_GEM_checkout_required':False}))
