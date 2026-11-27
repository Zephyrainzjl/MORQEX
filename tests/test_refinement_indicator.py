import numpy as np
import pytest
from morqex.mechanics import TetMesh
from morqex.microstructure import Volume
from morqex.top_damage.refinement_indicator import coupled_indicator,mark_bulk


def test_constant_equilibrated_fields_have_zero_indicator():
    mesh=TetMesh.from_volume(Volume(np.ones((2,2,2),np.uint8),spacing=(1e-6,)*3))
    psi=2e6;Gc=5.;ell=2e-6;k=1e-5
    value=2*(1-k)*psi/(2*(1-k)*psi+Gc/ell)
    result=coupled_indicator(mesh,np.broadcast_to(np.diag([2e8,-1e8,3e8]),(len(mesh.cells),3,3)),
        np.full(len(mesh.points),value),psi,90e9,Gc,ell,k)
    assert result['total_J'].sum()<1e-37
    P=np.broadcast_to(np.eye(3)*1e8,(len(mesh.cells),3,3)).copy();P[0]*=2
    result=coupled_indicator(mesh,P,np.full(len(mesh.points),value),psi,90e9,Gc,ell,k)
    assert result['mechanical_J'].sum()>0


def test_mark_bulk_covers_requested_mass_and_is_deterministic():
    a=np.array([1.,3.,2.,3.,0.])
    selected=mark_bulk(a,.5)
    np.testing.assert_array_equal(selected,[1,3])
    assert a[selected].sum()>=.5*a.sum()
    assert len(mark_bulk(np.zeros(3)))==0
    with pytest.raises(ValueError):mark_bulk([-1.])
    with pytest.raises(ValueError):mark_bulk([1.],0)
