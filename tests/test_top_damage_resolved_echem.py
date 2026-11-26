from pathlib import Path
import pytest
from morqex.top_damage.resolved_echem import ResolvedBuildConfig
from morqex.top_damage.resolved_echem import check_potential_connectivity
from morqex.top_damage.resolved_echem import check_restart_geometry
import numpy as np
from morqex.microstructure.simulate.electrochemistry import audit_solver

def test_resolved_backend_budget_is_explicit():
    assert ResolvedBuildConfig.read(Path(__file__).parents[1]/'configs/top_damage_resolved_build.toml').jobs==1
    for args in [{'device':'cuda'},{'jobs':4},{'precision':'float32'},{'slice_seconds':100},{'hypre':'false'}]:
        with pytest.raises(ValueError):ResolvedBuildConfig(**args)

def test_floating_cbd_is_rejected_without_deleting_material():
    a=np.full((5,5,12),4,np.uint8);a[:,:,:3]=5;a[:,:,9:]=1
    assert check_potential_connectivity(a)['potential_components']==1
    a[2,2,6]=7;original=a.copy()
    with pytest.raises(ValueError,match='floating potential'):check_potential_connectivity(a)
    np.testing.assert_array_equal(a,original)

def test_native_audit_rejects_missing_earlier_acceptance(tmp_path):
    (tmp_path/'inputs').write_text('echemamr.bv_nonlinear_reltol = 1e-8\nechemamr.bv_nonlinear_abstol = 1e-12\n')
    text='MORQEX BV BEGIN time=0\nBV NON-LINEAR RESIDUAL (rel,abs): 1e-10 0.01\nMORQEX BV ACCEPT 1e-10 0.01\n'
    log=tmp_path/'solver.log';log.write_text(text)
    assert audit_solver(tmp_path)['nonlinear_convergence_confirmed']
    log.write_text('MORQEX BV BEGIN time=-1\n'+text)
    assert not audit_solver(tmp_path)['nonlinear_convergence_confirmed']
    log.write_text(text+'amrex::Abort:: failed\n')
    assert not audit_solver(tmp_path)['nonlinear_convergence_confirmed']
    log.write_text(text.replace('MORQEX BV ACCEPT 1e-10 0.01','MORQEX BV ACCEPT nan 0.01'))
    assert not audit_solver(tmp_path)['nonlinear_convergence_confirmed']

def test_native_audit_supports_absolute_convergence_but_not_old_last_only_logs(tmp_path):
    log=tmp_path/'solver.log';log.write_text('BV NON-LINEAR RESIDUAL (rel,abs): 1e-10 1e-13\n')
    assert not audit_solver(tmp_path)['nonlinear_convergence_confirmed']
    log.write_text('MORQEX BV BEGIN time=0\nBV NON-LINEAR RESIDUAL (rel,abs): 1 1e-13\nMORQEX BV ACCEPT 1 1e-13\n')
    assert audit_solver(tmp_path)['nonlinear_convergence_confirmed']

def test_native_restart_cannot_silently_change_topology_or_voxel_size(tmp_path):
    from morqex.microstructure import Volume
    a=np.full((2,2,3),4,np.uint8);a[:,:,0]=5;a[:,:,-1]=1
    v=Volume(a,spacing=(1e-6,)*3);v.save(tmp_path/'input_volume.npz')
    with (tmp_path/'geometry.txt').open('w') as f:
        f.write('3 2 2\n');np.savetxt(f,a.ravel(),fmt='%d')
    checkpoint=tmp_path/'chk00001';checkpoint.mkdir();(checkpoint/'Header').write_text('fixture')
    assert check_restart_geometry(v,a,checkpoint)==checkpoint
    different=a.copy();different[0,0,0]=7
    with pytest.raises(ValueError,match='topology changed'):check_restart_geometry(v,different,checkpoint)
    with pytest.raises(ValueError,match='lattice changed'):check_restart_geometry(Volume(a,spacing=(2e-6,)*3),a,checkpoint)
