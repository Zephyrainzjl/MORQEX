"""Independent installed-wheel compilation, parameter and mapping proof."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

source=Path(__file__).resolve().parents[1]/'src';sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=source]
import numpy as np
import morqex
import pybamm as pb
from morqex.acceleration.verification import manufactured_periodic
from morqex.electrochemistry.open_materials import HOSTS,OpenMaterialCard,TabulatedOCP
from morqex.electrochemistry.conservative_ports import nodal_ports
from morqex.solvers.pybamm_backend import solver

assert source not in Path(morqex.__file__).resolve().parents and source not in Path(pb.__file__).resolve().parents
assert metadata.version('morqex-research')==morqex.__version__=='0.1.0.dev0'
try:metadata.distribution('pybamm')
except metadata.PackageNotFoundError:pass
else:raise AssertionError('Bundled source must not be overwritten by a separate PyBaMM distribution')
root=Path(morqex.__file__).parent
for asset in ['acceleration/voxel.cu','acceleration/poisson.cu','visualization/full_volume.html','visualization/full_volume.js']:
    assert (root/asset).is_file(),asset
reference=manufactured_periodic(8);assert reference['stress_relative_L2']<1e-8
cards={name:OpenMaterialCard.host_card(name).build()[1] for name in HOSTS}
ports=nodal_ports([.2,.3],np.array([[10.,20.],[30.,10.]]),[.4,.6],provenance='independent wheel algebraic proof')
audit=ports.audit(np.array([100.,150.]),np.array([.01,.02]),np.array([.1,.2,.3,.4]));assert abs(audit['power_residual_W_m3'])<1e-12
curve=TabulatedOCP([.001,.3,.6,.999],[.5,.2,.25,.1],source='independent nonmonotone interpolation check')
np.testing.assert_allclose(curve(pb.Array([.3,.6])).evaluate().ravel(),[.2,.25],atol=1e-14)
model=pb.lithium_ion.DFN();p=pb.ParameterValues('Chen2020');sim=pb.Simulation(model,parameter_values=p,solver=solver(),var_pts={**model.default_var_pts,'x_n':3,'x_s':3,'x_p':3,'r_n':3,'r_p':3})
solution=sim.solve([0,10]);assert solution.t[-1]==10
result={'version':morqex.__version__,'morqex_path':morqex.__file__,'pybamm_path':pb.__file__,'source_imported':False,'upstream_distribution_installed':False,
    'host_cards':cards,'native_compiled_manufactured':reference,'reaction_port_audit':audit,'native_DFN_seconds':10.,'scope':'installed wheel assets, CPU compilation, all host entries, PCHIP, port identities and a native DFN smoke test; not full CT coupled qualification'}
w=Path(os.environ['MORQEX_WORKSPACE']);(w/'reports/open_installed_wheel.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
