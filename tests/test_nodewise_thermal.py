import numpy as np
import pytest
from morqex.geometry import MicrostructureData
from morqex.nodewise.heat import ThermalPhase,ThermalCell


def test_voxel_heat_homogeneous_linear_profile_and_source_power():
    a=np.ones((4,5,6),dtype=np.uint8);h=2e-6
    geometry=MicrostructureData(a,(h,)*3,(0.,)*3,{1:'active'},{'source':'analytic regression'})
    cell=ThermalCell(geometry,{1:ThermalPhase(2.,2e6,'analytic')})
    assert cell.effective_conductivity==pytest.approx(2.,rel=1e-12)
    x=(np.arange(6)+.5)*h-3*h
    np.testing.assert_allclose(cell.gradient_map.reshape(a.shape),np.broadcast_to(x,a.shape),rtol=1e-12,atol=1e-19)
    recovered=cell.recover(300.,1e4,source_W_m3=np.ones(a.shape)*1e7)
    assert recovered['source_power_W']==pytest.approx(recovered['outward_boundary_power_W'],rel=1e-8)
    assert recovered['absolute_equilibrium_residual_W']<1e-15


def test_voxel_heat_series_parallel_and_contact_resistance():
    h=1e-6;a=np.ones((3,3,8),dtype=np.uint8);a[:,:,4:]=2
    phases={1:ThermalPhase(2.,2e6,'analytic'),2:ThermalPhase(.5,1e6,'analytic')}
    geometry=MicrostructureData(a,(h,)*3,(0.,)*3,{1:'active',2:'cbd'},{'source':'analytic regression'})
    cell=ThermalCell(geometry,phases)
    assert cell.effective_conductivity==pytest.approx(1/(.5/2+.5/.5),rel=1e-12)
    resistance=1e-6;broken=ThermalCell(geometry,phases,contact_resistance_m2_K_W=resistance,damage=.4)
    retention=.001+.999*.6**2
    exact=8*h/(4*h/2+4*h/.5+resistance/retention)
    assert broken.effective_conductivity==pytest.approx(exact,rel=1e-12)
    transverse=np.ones((4,4,8),dtype=np.uint8);transverse[:2]=2
    parallel=ThermalCell(MicrostructureData(transverse,(h,)*3,(0.,)*3,geometry.phase_kinds,geometry.provenance),phases)
    assert parallel.effective_conductivity==pytest.approx(1.25,rel=1e-12)


def test_geometry_thermal_damage_table_and_symbolic_derivative():
    import casadi as ca
    import pybamm as pb
    from morqex.nodewise.thermal_damage import build_thermal_damage_table,SpatialThermalConductivity
    a=np.ones((3,3,8),dtype=np.uint8);a[:,:,4:]=2
    geometry=MicrostructureData(a,(1e-6,)*3,(0.,)*3,{1:'active',2:'cbd'},{'source':'analytic'})
    phases={1:ThermalPhase(2.,2e6,'analytic'),2:ThermalPhase(.5,1e6,'analytic')}
    table=build_thermal_damage_table(geometry,phases,contact_resistance_m2_K_W=1e-6,tolerance=.003)
    assert table.metadata['sampled_midpoint_maximum_relative_error']<=.003
    assert min(table.conductivity_W_m_K)>0 and max(np.diff(table.conductivity_W_m_K))<=0
    state=pb.StateVector(slice(0,2),domain='positive electrode')
    field=SpatialThermalConductivity(table.damage,np.tile(table.conductivity_W_m_K,(2,1)),state)
    x=ca.MX.sym('d',2);fn=field.function
    compiled=ca.Function('thermal',[x],[fn(x),ca.jacobian(fn(x),x)])
    y=np.array([.27,.77]);value,jac=compiled(y)
    expected=field.evaluate(y=y[:,None]);np.testing.assert_allclose(value,expected,rtol=1e-13)
    eps=1e-6
    for i in range(2):
        delta=np.zeros(2);delta[i]=eps
        fd=(field.evaluate(y=(y+delta)[:,None])-field.evaluate(y=(y-delta)[:,None]))/(2*eps)
        np.testing.assert_allclose(fd,np.array(jac)[:,i:i+1],rtol=1e-7,atol=1e-9)


def test_conservative_face_power_and_analytic_jacobian():
    import pybamm as pb
    from morqex.electrochemistry.power_flux import FacePotential
    n=4;d='positive electrode';state=pb.StateVector(slice(0,n),domain=d)
    left=pb.Vector(np.full(n,1.),domain=d);right=pb.Vector(np.full(n,5.),domain=d)
    face=FacePotential(state,left,right)
    y=np.array([1.2,2.,3.,4.8])[:,None]
    np.testing.assert_allclose(face.evaluate(y=y).ravel(),[1.,1.6,2.5,3.9,5.])
    jac=face.jac(state).evaluate()
    if hasattr(jac,'toarray'):jac=jac.toarray()
    for j in range(n):
        delta=np.zeros_like(y);delta[j]=1e-6
        fd=(face.evaluate(y=y+delta)-face.evaluate(y=y-delta))/2e-6
        np.testing.assert_allclose(fd.ravel(),jac[:,j],atol=1e-9)
