"""Installed-wheel CT/degradation proof, without source imports or upstream dist."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import numpy as np
source=Path(__file__).resolve().parents[1]/'src'
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=source]
import morqex
import pybamm as pb
from morqex.electrochemistry.aging_cache import load_aging_kernel,AgingLocalization
from morqex.electrochemistry.aging_dfn import build_aging_dfn,age
from morqex.electrochemistry.microstructure_dfn import MicrostructureStack,key
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
from morqex.degradation import DegradationOptions,standard_registry
from morqex.solvers.pybamm_backend import solver
assert source not in Path(morqex.__file__).resolve().parents and source not in Path(pb.__file__).resolve().parents
assert metadata.version('morqex-research')==morqex.__version__=='0.0.8'
try:metadata.distribution('pybamm')
except metadata.PackageNotFoundError:pass
else:raise AssertionError('Independent wheel proof requires no upstream PyBaMM distribution')
w=Path(os.environ['MORQEX_WORKSPACE']);kernels={d:load_aging_kernel(w/'operators/aging/ct_all'/d) for d in ['negative','positive']};p,_=microstructure_parameters()
for d in ['Negative','Positive']:
    name=f'{d} electrode partial molar volume [m3.mol-1]';p.update({name:p[name]*.1})
stack=MicrostructureStack(kernels['negative'].operator,kernels['positive'].operator,p);stack.update_parameters(p);model=build_aging_dfn(stack,kernels,DegradationOptions())
sim=pb.Simulation(model,parameter_values=p,solver=solver(),var_pts={**model.default_var_pts,'x_n':3,'x_s':3,'x_p':3,'r_n':2,'r_p':2});s=sim.solve([0,60],t_interp=np.linspace(0,60,7));assert s.t[-1]==60
c=np.array([s[key('negative',k,'concentration [mol.m-3]')].entries[1,-1] for k in range(kernels['negative'].operator.dimension)])
local=AgingLocalization(w/'operators/aging/ct_all/negative');fields=local.recover(c,np.array(s['MORQEX negative axial strain'].entries[1,-1]),np.array(s['Volume-averaged cell temperature [K]'].entries[-1]-298.15),np.array(s[age('negative','damage')].entries[1,-1]))
assert np.isfinite(fields['stress']).all() and fields['opening'].shape[-1]==3
result={'version':morqex.__version__,'source_imported':False,'upstream_distribution_installed':False,'seconds':60.,
    'CT_active_voxels':len(fields['concentration']),'interface_samples':len(fields['opening']),'registry_kinds':list(standard_registry().manifest()),'pybamm_path':pb.__file__}
(w/'reports/upgrade_installed_wheel.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
