"""Standalone 0.0.7 wheel proof: no source path or upstream distribution."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
source=Path(__file__).resolve().parents[1]/'src'
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=source]
import morqex
import pybamm as pb
from morqex.electrochemistry.microstructure_runtime import MicrostructureDFNRuntime
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
from morqex.electrochemistry.microstructure_validation import thermodynamic_observables
assert source not in Path(pb.__file__).resolve().parents
assert source not in Path(morqex.__file__).resolve().parents
try:metadata.distribution('pybamm')
except metadata.PackageNotFoundError:pass
else:raise AssertionError('Upstream distribution must not be installed')
assert metadata.version('morqex-research')==morqex.__version__=='0.0.7'
ws=Path(os.environ['MORQEX_WORKSPACE']);case=json.loads((ws/'results/microstructure_dfn/rom16_1c/result.json').read_text())
p,_=microstructure_parameters()
for d in ['Negative','Positive']:
    name=f'{d} electrode partial molar volume [m3.mol-1]';p.update({name:p[name]*.1})
runtime=MicrostructureDFNRuntime({d:m['operator_path'] for d,m in case['microstructure'].items()},p,nx=3)
assert not runtime._localizers
step=runtime.advance(60.,5.);assert step['completed']
bound=runtime.reaction_bound('negative');assert not runtime._localizers
energy=thermodynamic_observables(runtime.solution,runtime.stack,runtime.parameters)['metrics']
assert energy['relative_internal_energy_residual']<1e-6 and energy['relative_entropy_residual']<1e-5
field=runtime.recover('negative');assert field['concentration'].shape==(42,)
assert pb.ParameterValues('Ai2020')['Nominal cell capacity [A.h]']>0
result={'version':morqex.__version__,'pybamm_module':pb.__file__,'upstream_distribution_installed':False,
    'source_imported':False,'completed_seconds':step['time_s'],'micro_cells':len(field['concentration']),
    'online_regions':runtime.stack.electrodes['negative'].dimension,'energy':energy}
(ws/'reports/microstructure_installed_wheel.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':pass
