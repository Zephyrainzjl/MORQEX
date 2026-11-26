import numpy as np
from morqex.top_damage.epoch_dfn import EpochDFN,DFNFeedback


def make_runtime():
    return EpochDFN(active_fraction=.420379638671875,porosity=.349609375,radius_m=3.725e-6,nx=6,nr=20)


def test_exterior_flux_and_feedback_preserve_lithium_and_restart():
    runtime=make_runtime();current=.5*runtime.capacity_Ah
    first=runtime.advance(540.,current,DFNFeedback(10.,.15))
    assert first['inventory_relative_defect']<1e-10
    flux=np.asarray(runtime.last['Electrolyte flux [mol.m-2.s-1]'].entries)
    np.testing.assert_allclose(flux[0],current/(1e-4*96485.33212331001),rtol=1e-12)
    saved=np.asarray(runtime.last.last_state.y).ravel();time=float(runtime.last.t[-1]);inputs=runtime.last_inputs
    second=runtime.advance(270.,-current,DFNFeedback(5.,.12,.8,15.))
    assert second['relative_state_jump']<1e-12
    assert second['inventory_relative_defect']<1e-10
    resumed=make_runtime();resumed.restore_state(time,saved,inputs)
    repeated=resumed.advance(270.,-current,DFNFeedback(5.,.12,.8,15.))
    assert abs(second['voltage_V']-repeated['voltage_V'])<1e-8
    np.testing.assert_allclose(runtime.last.last_state.y,resumed.last.last_state.y,rtol=1e-8,atol=1e-9)
    r,c=runtime.radial_profile()
    assert np.all(np.diff(r)>0) and r[0]==0 and r[-1]==1
    assert c.min()>0 and c.max()<runtime.cmax


def test_full_voltage_window_current_reversal_after_cutoffs():
    runtime=make_runtime();phase=0;cutoffs=[]
    for step in range(40):
        current=(-1 if phase in (0,2) else 1)*2*runtime.capacity_Ah
        result=runtime.advance(300.,current,DFNFeedback(10.,.15))
        assert result['inventory_relative_defect']<1e-9
        if result['termination']!='final time':
            name='Lower' if phase==1 else 'Upper'
            assert abs(result['voltage_V']-runtime.parameters[name+' voltage cut-off [V]'])<1e-4
            cutoffs.append(result);phase+=1
            if phase==3:break
    assert phase==3
    assert cutoffs[1]['net_capacity_Ah']>cutoffs[0]['net_capacity_Ah']
    assert cutoffs[2]['net_capacity_Ah']<cutoffs[1]['net_capacity_Ah']
