import numpy as np
import pytest
from morqex.top_damage.voxel_conductivity import effective_conductivity


def test_uniform_and_series_layers_include_boundary_half_cells():
    for axis in range(3):
        field=np.ones((4,6,8))*2
        result=effective_conductivity(field,(1e-6,2e-6,3e-6),axis=axis)
        assert result['effective']==pytest.approx(2.,rel=1e-9)
        cut=[slice(None)]*3;cut[axis]=slice(0,field.shape[axis]//2);field[tuple(cut)]=6.
        result=effective_conductivity(field,(1e-6,2e-6,3e-6),axis=axis)
        assert result['effective']==pytest.approx(3.,rel=1e-8)


def test_true_disconnection_and_floating_material_no_leakage():
    field=np.ones((5,5,7));field[:,:,2:5]=0;field[2,2,3]=10
    result=effective_conductivity(field,(1e-6,)*3)
    assert result['effective']<1e-14
    assert not result['collector_accessible'][2,2,3]
    assert not result['collector_accessible'][:,:,0].any()
    assert result['collector_accessible'][:,:,-1].all()


def test_damage_cg_matches_direct_obstacle():
    from scipy.sparse import diags
    from morqex.top_damage.phase_field import solve_box_quadratic
    A=diags([-np.ones(19),3*np.ones(20),-np.ones(19)],[-1,0,1]).tocsr()
    b=np.linspace(.05,2.5,20);lower=np.linspace(.7,.05,20)
    a,_,_=solve_box_quadratic(A,b,lower)
    c,res,_=solve_box_quadratic(A,b,lower,linear_solver='jacobi-cg')
    np.testing.assert_allclose(a,c,atol=1e-10);assert res<1e-10
