import numpy as np
from morqex.mechanics import TetMesh
from morqex.microstructure import Volume
from morqex.top_damage import NativeConfig, MFEMScalarBackend, AT2Damage, SplitElasticity
from morqex import Workspace
import os


def test_amg_matches_direct_heterogeneous_damaged_equilibrium():
    mesh=TetMesh.from_volume(Volume(np.ones((3,4,4),dtype=np.uint8),spacing=(1e-6,)*3))
    backend=MFEMScalarBackend(NativeConfig(),Workspace(os.environ['MORQEX_WORKSPACE']))
    model=AT2Damage(mesh,backend,10.,2e-6)
    E=np.where(mesh.centers[:,0]<2e-6,90e9,2e9)
    direct=SplitElasticity(model,E,.3)
    fast=SplitElasticity(model,E,.3,linear_solver='amg-cg')
    d=.3+.2*np.sin(mesh.points[:,0]/1e-6)
    eps=np.zeros((len(mesh.cells),6));eps[:,0]=.003*np.sin(mesh.centers[:,0]/1e-6)
    a=direct.solve(d,eps);b=fast.solve(d,eps)
    assert b.equilibrium_residual<1e-9
    np.testing.assert_allclose(a.displacement_m,b.displacement_m,atol=1e-18,rtol=1e-8)
    np.testing.assert_allclose(a.mean_cell_stress_Pa,b.mean_cell_stress_Pa,atol=.02,rtol=1e-7)
    assert len(fast.linear_iterations)>0
