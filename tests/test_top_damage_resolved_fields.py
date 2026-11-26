from types import SimpleNamespace
import numpy as np
import h5py
import pytest
from morqex.top_damage.resolved_phase_field import read_resolved_cathode,ResolvedMechanicsConfig


def test_exact_field_port_preserves_values_and_rejects_phase_and_coordinate_aliasing(tmp_path):
    labels=np.zeros((2,3,4),int);labels[:,:,:2]=1;labels[:,:,2]=2
    data=SimpleNamespace(structure=SimpleNamespace(labels=labels,spacing_zyx_m=(1e-6,)*3,origin_zyx_m=(0.,)*3))
    path=tmp_path/'fields.h5';c=np.arange(24).reshape(labels.shape)+100.
    with h5py.File(path,'w') as f:
        f.attrs.update(schema='morqex-resolved-fields-1',axis_order='ZYX',spacing_zyx_m=(1e-6,)*3,origin_zyx_m=(0.,)*3,time_s=4.)
        for key,value in [('Concentration',c),('Potential',c/100),('Cathode_active_material',labels==1),('Cathode_CBD',labels==2)]:f[key]=value
    actual,phi,time=read_resolved_cathode(path,data,cmax_mol_m3=1000.)
    np.testing.assert_array_equal(actual,c);np.testing.assert_array_equal(phi,c/100);assert time==4.
    with h5py.File(path,'a') as f:f['Cathode_CBD'][0,0,0]=True
    with pytest.raises(ValueError,match='phase mismatch'):read_resolved_cathode(path,data,cmax_mol_m3=1000.)
    with h5py.File(path,'a') as f:f['Cathode_CBD'][0,0,0]=False;f.attrs['spacing_zyx_m']=(2e-6,)*3
    with pytest.raises(ValueError,match='coordinates'):read_resolved_cathode(path,data,cmax_mol_m3=1000.)


def test_resolved_mechanics_rejects_nonphysical_materials():
    for args in [{'cmax_mol_m3':0},{'poisson':.5},{'active_Gc_J_m2':-1},{'boundary':'unconstrained-silently'},{'chemical_strain_per_stoichiometry':[1,2]}]:
        with pytest.raises(ValueError):ResolvedMechanicsConfig(**args)
