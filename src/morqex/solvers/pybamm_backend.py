"""Bundled pinned PyBaMM/IDAKLU backend with a local derivative conversion fix."""
import hashlib
from pathlib import Path
import pybamm

SUPPORTED_VERSION='26.3.1'


def solver(*,rtol=1e-7,atol=1e-8):
    if pybamm.__version__!=SUPPORTED_VERSION:raise RuntimeError(f'Qualified PyBaMM version is {SUPPORTED_VERSION}, got {pybamm.__version__}')
    if getattr(pybamm,'MORQEX_PATCH_LEVEL',None)!='scalar-1d-interpolant-derivatives-v1':raise RuntimeError('The qualified bundled derivative conversion patch is required')
    return pybamm.IDAKLUSolver(rtol=rtol,atol=atol,options={'num_threads':1,'num_solvers':1})


def provenance():
    root=Path(pybamm.__file__).resolve().parent
    files=['models/submodels/particle/fickian_diffusion.py','models/submodels/particle/base_particle.py',
        'models/submodels/interface/open_circuit_potential/single_ocp.py','models/submodels/thermal/base_thermal.py','solvers/idaklu_solver.py',
        'solvers/casadi_solver.py','solvers/casadi_algebraic_solver.py']
    files.append('expression_tree/operations/convert_to_casadi.py')
    files.append('models/full_battery_models/lithium_ion/dfn.py')
    files.extend('input/parameters/lithium_ion/'+name+'.py' for name in ['Ai2020','Marquis2019','OKane2022'])
    files.extend(str(path.relative_to(root)) for path in sorted((root/'input/parameters/lithium_ion/data').glob('*.csv')))
    return {'version':pybamm.__version__,'module':str(root),'patch':getattr(pybamm,'MORQEX_PATCH_LEVEL',None),'implementation':'bundled upstream Python source plus qualified 1D derivative conversion fix; binary dependencies external',
        'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}}
