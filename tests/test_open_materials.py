import numpy as np
import pybamm as pb
import pytest
from morqex.electrochemistry.open_materials import TabulatedOCP,OpenMaterialCard,HOSTS

def test_arbitrary_nonmonotone_ocp_preserves_knots_and_energy_primitive():
    x=np.linspace(.01,.99,31);y=3.6-.2*x+.1*np.sin(4*np.pi*x);s=1e-4*(x-.5)
    curve=TabulatedOCP(x,y,'declared synthetic nonmonotone regression',s)
    np.testing.assert_allclose(curve(pb.Array(x)).evaluate().ravel(),y,rtol=0,atol=1e-14)
    assert curve.metadata()['minimum_sampled_negative_slope_V']<0
    z=np.linspace(.02,.98,61);h=1e-7
    derivative=(curve.primitive(z+h,310)-curve.primitive(z-h,310))/(2*h)
    actual=curve(pb.Array(z)).evaluate().ravel()+(310-298.15)*curve.entropy(pb.Array(z)).evaluate().ravel()
    np.testing.assert_allclose(derivative,actual,rtol=2e-8,atol=1e-8)

@pytest.mark.parametrize('host',['NMC532','NMC811','LFP','LCO'])
def test_open_host_cards_do_not_replace_native_ocp(host):
    p,audit=OpenMaterialCard.host_card(host).build();native=pb.ParameterValues(HOSTS[host]);sto=pb.Scalar(.5)
    assert p['Positive electrode OCP [V]'](sto).evaluate()==native['Positive electrode OCP [V]'](sto).evaluate()
    assert not audit['supplemented_keys']

def test_custom_dictionary_overrides_and_tabulated_entropy_are_explicit():
    curve=TabulatedOCP([.01,.5,.99],[4.3,3.9,3.4],'user lab dataset',[.0001,0.,-.0001])
    p,a=OpenMaterialCard('custom',{'Reference temperature [K]':298.15},overrides={'custom coefficient':7},ocps={'positive':curve}).build()
    assert p['custom coefficient']==7 and p['Positive electrode OCP [V]'] is curve
    with pytest.raises(ValueError):OpenMaterialCard('custom',{},supplements={'unknown constant':2}).build()

def test_exact_compact_pchip_casadi_values_and_first_derivatives():
    import casadi as ca
    from scipy.interpolate import PchipInterpolator
    x=np.linspace(.01,.99,21);y=3.7+.1*np.sin(9*x)-.3*x;curve=TabulatedOCP(x,y,'CasADi regression',extrapolate=True)
    state=pb.StateVector(slice(0,1));expr=curve(state);z=ca.MX.sym('z');test=np.r_[x,[-.02,1.02],(x[1:]+x[:-1])/2]
    value=ca.Function('pchip_value',[z],[expr.to_casadi(y=z)]);first=ca.Function('pchip_first',[z],[expr.diff(state).to_casadi(y=z)])
    reference=PchipInterpolator(x,y)
    np.testing.assert_allclose(np.array(value(test)).ravel(),reference(test),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.array(first(test)).ravel(),reference(test,1),rtol=1e-10,atol=1e-10)
