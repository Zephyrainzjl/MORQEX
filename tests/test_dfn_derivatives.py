import numpy as np
import pytest
pb=pytest.importorskip('pybamm')
ca=pytest.importorskip('casadi')


@pytest.mark.parametrize('kind',['cubic','pchip'])
@pytest.mark.parametrize('order',[0,1,2,3])
def test_bundled_interpolant_derivatives_agree_python_casadi_and_finite_difference(kind,order):
    x=np.linspace(.1,.9,15);y=np.exp(x)+.2*x**3
    variable=pb.InputParameter('x');symbol=pb.Interpolant(x,y,variable,interpolator=kind)
    for _ in range(order):symbol=symbol.diff(variable)
    sx=ca.MX.sym('x');converted=pb.CasadiConverter().convert(symbol,None,None,None,{'x':sx})
    function=ca.Function('derivative',[sx],[converted])
    for point in [.133,.253,.453,.753,.873]:
        python=np.asarray(symbol.evaluate(inputs={'x':point})).item();compiled=float(function(point))
        assert compiled==pytest.approx(python,rel=1e-9,abs=1e-8)
        if order==1:
            parent=pb.Interpolant(x,y,variable,interpolator=kind);h=1e-6
            fd=(np.asarray(parent.evaluate(inputs={'x':point+h})).item()-np.asarray(parent.evaluate(inputs={'x':point-h})).item())/(2*h)
            assert compiled==pytest.approx(fd,rel=1e-8,abs=1e-8)


def test_ai2020_physical_chemical_curvature_compilation_regression():
    params=pb.ParameterValues('Ai2020');model=pb.lithium_ion.DFN()
    c=pb.InputParameter('c');sx=ca.MX.sym('c')
    for domain in ['negative','positive']:
        phase=model.param.domain_params[domain].prim
        curvature=params.process_symbol(-model.param.F*phase.U(c/phase.c_max,298.15).diff(c))
        converted=pb.CasadiConverter().convert(curvature,None,None,None,{'c':sx});function=ca.Function('curvature',[sx],[converted])
        value=params[f'Initial concentration in {domain} electrode [mol.m-3]']
        expected=np.asarray(curvature.evaluate(inputs={'c':value})).item()
        assert expected>0 and float(function(value))==pytest.approx(expected,rel=1e-10)
    assert pb.MORQEX_PATCH_LEVEL=='scalar-1d-interpolant-derivatives-v1'


@pytest.mark.parametrize('order',[0,1,2])
def test_pchip_vector_compilation_and_ad_including_knots_and_extrapolation(order):
    x=np.linspace(.1,.9,15);y=np.exp(x)+.2*x**3
    points=np.r_[.05,x,.95]
    v=pb.InputParameter('x',expected_size=len(points));symbol=pb.Interpolant(x,y,v,interpolator='pchip')
    for _ in range(order):symbol=symbol.diff(v)
    sx=ca.MX.sym('x',len(points));compiled=pb.CasadiConverter().convert(symbol,None,None,None,{'x':sx})
    f=ca.Function('vector',[sx],[compiled,ca.jacobian(compiled,sx)])
    value,jac=f(points)
    expected=np.asarray(symbol.evaluate(inputs={'x':points})).ravel()
    derivative=np.asarray(symbol.diff(v).evaluate(inputs={'x':points})).ravel()
    np.testing.assert_allclose(np.asarray(value).ravel(),expected,rtol=1e-10,atol=1e-9)
    np.testing.assert_allclose(np.diag(np.asarray(jac)),derivative,rtol=1e-10,atol=1e-8)


@pytest.mark.parametrize('order',[1,2,3])
def test_cubic_vector_derivatives_at_original_knots(order):
    x=np.linspace(.1,.9,15);y=np.exp(x)+.2*x**3
    variable=pb.InputParameter('x',expected_size=len(x))
    symbol=pb.Interpolant(x,y,variable,interpolator='cubic')
    for _ in range(order):symbol=symbol.diff(variable)
    sx=ca.MX.sym('x',len(x));converted=pb.CasadiConverter().convert(symbol,None,None,None,{'x':sx})
    function=ca.Function('cubic_vector',[sx],[converted])
    np.testing.assert_allclose(np.asarray(function(x)).ravel(),np.asarray(symbol.evaluate(inputs={'x':x})).ravel(),rtol=1e-10,atol=1e-8)
