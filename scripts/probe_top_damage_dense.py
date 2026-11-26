"""Bounded full 32^3 dense-electrode mechanics profiling before cycle scaling."""
import json
import os
import time
import numpy as np
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.mechanics import TetMesh
from morqex.top_damage import PolycrystalData, NativeConfig, MFEMScalarBackend, AT2Damage, SplitElasticity
from morqex.top_damage.assembly_cache import CachedP1ScalarBackend

ws=Workspace(os.environ['MORQEX_WORKSPACE']);started=time.perf_counter()
data=PolycrystalData.load_hdf5(ws.path('results/top_damage/microstructures/dense_large_particles_32_g4/initial.h5'))
mesh=TetMesh.from_volume(Volume(data.structure.labels,spacing=data.structure.spacing_zyx_m),solid_phases=[1,2])
print(json.dumps({'stage':'mesh','seconds':time.perf_counter()-started,'cells':len(mesh.cells),'nodes':len(mesh.points)}),flush=True)
backend=CachedP1ScalarBackend(mesh,MFEMScalarBackend(NativeConfig(),ws))
damage=AT2Damage(mesh,backend,np.where(mesh.phases==1,10.,20.),1e-6)
mechanics=SplitElasticity(damage,np.where(mesh.phases==1,93e9,1e9),.3,linear_solver='amg-cg')
print(json.dumps({'stage':'assembled','seconds':time.perf_counter()-started}),flush=True)
strain=np.zeros((len(mesh.cells),6));strain[mesh.phases==1,:3]=.001
state=mechanics.solve(np.zeros(len(mesh.points)),strain)
result={'cells':len(mesh.cells),'nodes':len(mesh.points),'seconds':time.perf_counter()-started,
        'residual':state.equilibrium_residual,'energy_J':state.elastic_energy_J,'iterations':mechanics.linear_iterations,
        'scope':'Undamaged full-grid numerical probe; E=93/1 GPa is not a calibrated NMC composite'}
ws.path('reports/top_damage_dense_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
