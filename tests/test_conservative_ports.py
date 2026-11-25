import numpy as np
import pytest
from morqex.electrochemistry.conservative_ports import ConservativePorts,nodal_ports

def test_general_signed_modes_preserve_lithium_and_power_with_mass_projection():
    rng=np.random.default_rng(9);T=np.column_stack([np.ones(17),rng.normal(size=(17,3))]);A=rng.normal(size=(4,4));M=A.T@A+np.eye(4);L=rng.random((17,5));L/=L.sum(axis=1,keepdims=True)
    port=ConservativePorts(M,T,L,rng.uniform(10,100,17),np.full(5,.2),[1,0,0,0],'synthetic signed reduced basis / independent port test')
    g=rng.normal(size=4)*100;delta=rng.normal(size=5)*.01;mu=port.site_chemical_potential(g)
    np.testing.assert_allclose(mu,T@np.linalg.solve(M,g))
    eta=port.site_macro_potential(delta)+mu/96485.33212;j=2*np.sinh(96485.33212*eta/(2*8.314462618*298.15));audit=port.audit(g,delta,j)
    assert abs(audit['lithium_rate_residual'])<1e-15 and abs(audit['power_residual_W_m3'])<1e-10
    assert audit['reaction_dissipation_W_m3']>=0

def test_nodal_spatial_ports_are_adjoint_not_independent_copies():
    port=nodal_ports([.1,.2],np.array([[2.,4.,0.],[0.,1.,3.]]),[.2,.3,.5],provenance='one global modal inventory and spatial ports')
    j=np.arange(1,len(port.area_weights)+1,dtype=float);rate=port.modal_reaction_rate(j)
    assert port.lithium_inventory(rate)==pytest.approx(-port.area_weights@j/96485.33212,abs=1e-16)
    assert np.dot(port.macro_volumes,port.macro_current(j))==pytest.approx(port.area_weights@j)
    with pytest.raises(ValueError,match='constant'):ConservativePorts(np.eye(1),np.zeros((2,1)),np.ones((2,1)),np.ones(2),np.ones(1),np.ones(1),'invalid')
