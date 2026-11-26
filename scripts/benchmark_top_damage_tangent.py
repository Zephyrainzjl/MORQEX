"""Same FP64 MFEM tangent, geometry and vectors: C++ versus NumPy action."""
import os,time,json,gc
from pathlib import Path
import h5py
import numpy as np
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.mechanics.mesh import TetMesh
from morqex.top_damage import HyperelasticBackend,Hyperelasticity,NativeConfig
from morqex.top_damage.backend import require_local_budget
require_local_budget();ws=Workspace(os.environ['MORQEX_WORKSPACE'])
source=ws.path('results/top_damage/dense_cycles/finite_gentle_v2')
with h5py.File(source/'initial_geometry.h5') as f:labels=f['phase_labels'][:];h=f.attrs['spacing_zyx_m']
mesh=TetMesh.from_volume(Volume(labels,spacing=h),solid_phases=[1,2])
backend=HyperelasticBackend(NativeConfig(),ws)
model=Hyperelasticity(mesh,backend,np.where(mesh.phases==1,93e9,1e9),.3,linear_solver='matrix-free-amg-minres')
with h5py.File(source/'step_00003.h5') as f:u=f['displacement_m'][:];fc=f['finite/chemical_deformation_gradient'][:]
compiled=model.assemble(u.ravel(),fc)[2]
model.compiled_action=False
numpy_action=model.assemble(u.ravel(),fc)[2]
rng=np.random.default_rng(20261126);v=rng.normal(size=3*len(mesh.points))
a=compiled@v;b=numpy_action@v
error=float(np.linalg.norm(a-b)/np.linalg.norm(b));assert error<1e-12
samples={'compiled':[],'numpy':[]}
for repeat in range(9):
    sequence=[('compiled',compiled),('numpy',numpy_action)]
    if repeat%2:sequence.reverse()
    for name,action in sequence:
        start=time.perf_counter();action@v;samples[name].append(time.perf_counter()-start)
result={'cells':len(mesh.cells),'degrees_of_freedom':len(v),'relative_action_difference':error,
        'seconds':samples,'median_action_cost_ratio':float(np.median(samples['numpy'])/np.median(samples['compiled'])),
        'scope':'Single tangent application only; same CPU, FP64 inputs, stored MFEM tangent, interleaved timings. Not full DFN, cycle-jump or GPU speedup.'}
path=ws.path('reports/top_damage_compiled_tangent_benchmark.json');path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
