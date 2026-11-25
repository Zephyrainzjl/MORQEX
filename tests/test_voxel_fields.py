import numpy as np
from morqex.acceleration.voxel import NativeStencil
from morqex.mechanics import isotropic_stiffness
from morqex.visualization.voxel_fields import export_reference_fields

def test_streamed_complete_fields_match_native_stress_and_analytic_energy(tmp_path):
    phase=np.ones((4,4,4),np.uint8);C=np.stack([np.zeros((6,6)),isotropic_stiffness(100.,.25)]);beta=np.zeros((2,6));beta[1,:3]=.01
    model=NativeStencil(phase,C,beta);c=np.ones(phase.shape);u,metrics=model.solve(c)
    fields=export_reference_fields(model,u,c,tmp_path,chunk=7)
    np.testing.assert_array_equal(np.load(tmp_path/'stress_Pa.npy'),model.stress(u,c));np.testing.assert_allclose(np.load(tmp_path/'strain.npy'),0,atol=1e-12)
    expected=.5*beta[1]@C[1]@beta[1]
    assert abs(fields['elastic_energy_J_m3']-expected)<1e-13
    assert fields['relative_stationarity_energy_defect']<1e-12
    assert np.load(tmp_path/'von_mises_Pa.npy').max()<1e-12

def test_solver_displacement_has_explicit_conversion_to_metres():
    phase=np.ones((3,3,3),np.uint8);model=NativeStencil(phase,np.stack([np.zeros((6,6)),isotropic_stiffness(1.,.2)]),spacing=(2e-6,3e-6,4e-6));scaled=np.ones((3,*phase.shape))
    np.testing.assert_array_equal(model.displacement_m(scaled),scaled*2e-6)

def test_exported_total_displacement_includes_macroscopic_affine_part(tmp_path):
    phase=np.ones((3,3,3),np.uint8);model=NativeStencil(phase,np.stack([np.zeros((6,6)),isotropic_stiffness(100.,.2)]),spacing=(2.,3.,4.));macro=(.01,.02,.03,0.,0.,0.)
    c=np.zeros(phase.shape);u,_=model.solve(c,macro);export_reference_fields(model,u,c,tmp_path,macro=macro,chunk=4)
    physical=np.load(tmp_path/'displacement_m.npy');np.testing.assert_allclose(physical[0],np.broadcast_to(.01*np.arange(3)[None,None,:]*4.,phase.shape),atol=1e-14)
    np.testing.assert_allclose(physical[1],np.broadcast_to(.02*np.arange(3)[None,:,None]*3.,phase.shape),atol=1e-14)
