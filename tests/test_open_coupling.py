import numpy as np
import pytest
import pybamm as pb
from morqex import Workspace
from morqex.gemsim import GEMsim
from morqex.electrochemistry.open_materials import TabulatedOCP,OpenMaterialCard
from morqex.electrochemistry.open_particle import TransportLaw,build_open_dfn,OpenParticle,OpenOCP
from morqex.electrochemistry.open_validation import chemical_primitives,open_observables
from morqex.concentration_modes.pou_operator import build_pou_operator
from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode
from morqex.electrochemistry.microstructure_dfn import MicrostructureStack,key
from morqex.electrochemistry.aging_dfn import EnergyKernel
from morqex.degradation import DegradationOptions,NoDamage,DiffusionLimitedSEI,PlatingStrippingDeadLi
from morqex.solvers.pybamm_backend import solver

@pytest.fixture(scope='module')
def open_case(tmp_path_factory):
    x=np.linspace(.001,.999,65)
    curves={'negative':TabulatedOCP(x,.45-.35*x,'synthetic regression',1e-5*(x-.5)),
        'positive':TabulatedOCP(x,4.25-.7*x+.15*np.sin(8*x),'synthetic nonconvex regression',2e-5*(x-.5))}
    p,_=OpenMaterialCard('synthetic',parameter_source='OKane2022',ocps=curves).build()
    ws=Workspace(tmp_path_factory.mktemp('open_coupling'));g=GEMsim(ws);ops={};fulls={}
    for d in ['negative','positive']:
        data=g.random_field(shape=(4,)*3,fractions=(.35,.65),length_scale=(.6,)*3,seed=105,spacing=(2e-6,)*3,phase_kinds={0:'pore',1:'active'})
        full=MicrostructureElectrode(data,g,p,d);fulls[d]=full
        ops[d]=build_pou_operator(full,target=8,transport_law=TransportLaw(gradient_coefficient_J_m5_mol2=1e-14),relax_transverse=True)
    return p,ops,fulls

def test_open_chemical_primitive_derivatives_and_linear_entropy(open_case):
    p,_,_=open_case;phase=pb.lithium_ion.DFN().param.p.prim;cmax=float(p['Maximum concentration in positive electrode [mol.m-3]'])
    c=np.linspace(.02,.98,30)*cmax;T=310.;h=cmax*1e-7
    a=chemical_primitives(p,'positive',c+h,T)['free_energy_J_m3'];b=chemical_primitives(p,'positive',c-h,T)['free_energy_J_m3']
    mu=-float(pb.constants.F.value)*p.process_symbol(phase.U(pb.Array(c/cmax),pb.Scalar(T))).evaluate().ravel()
    np.testing.assert_allclose((a-b)/(2*h),mu,rtol=1e-7,atol=.01)
    dt=.001;parts=chemical_primitives(p,'positive',c,T)
    slope=(chemical_primitives(p,'positive',c,T+dt)['free_energy_J_m3']-chemical_primitives(p,'positive',c,T-dt)['free_energy_J_m3'])/(2*dt)
    np.testing.assert_allclose(slope,-parts['entropy_J_m3_K'],rtol=1e-6,atol=1e-4)
    np.testing.assert_allclose(parts['free_energy_J_m3']+T*parts['entropy_J_m3_K'],parts['internal_energy_J_m3'],atol=1e-5)

def test_positive_pou_energy_and_relaxed_transverse_stress(open_case):
    _,ops,_=open_case;op=ops['positive'];rng=np.random.default_rng(12);c=op.reference_concentration+rng.normal(size=op.dimension)*100
    field=op.recover(c,.001,2.);v=op.full.mesh.volumes/op.full.mesh.rve_volume
    np.testing.assert_allclose(v@field['stress'][:,1:],0,atol=1e-5)
    assert field['concentration'].min()>=c.min()-1e-9 and field['concentration'].max()<=c.max()+1e-9
    q=np.r_[.001,c-op.reference_concentration,2.]
    for k in range(op.dimension):
        z=np.zeros(len(q));z[k+1]=.01
        fd=(.5*(q+z)@op.H@(q+z)-.5*(q-z)@op.H@(q-z))/.02
        assert fd==pytest.approx(op.weights[k]*op.chemical_potential(c,.001,2.)[k],rel=1e-6,abs=1e-6)
    online=op.detach()
    with pytest.raises(NotImplementedError,match='POU'):online.reaction_port_bound(q,np.ones(op.dimension),np.zeros(op.dimension),298.15)

def test_open_nonconvex_thermal_degradation_dfn_conservation(open_case):
    p,ops,_=open_case;p=p.copy();online={d:o.detach() for d,o in ops.items()};stack=MicrostructureStack(online['negative'],online['positive'],p,allow_linear_ocp_entropy=True);stack.update_parameters(p)
    config=DegradationOptions(negative=(DiffusionLimitedSEI(),PlatingStrippingDeadLi(initial_plated_mol_m3=.01)),damage=NoDamage())
    model=build_open_dfn(stack,{d:EnergyKernel(o) for d,o in online.items()},config,parameters=p)
    sim=pb.Simulation(model,parameter_values=p,solver=solver(rtol=1e-8,atol=1e-9),var_pts={**model.default_var_pts,'x_n':3,'x_s':3,'x_p':3,'r_n':2,'r_p':2})
    sol=sim.solve(np.linspace(0,30,7));assert sol.t[-1]==pytest.approx(30.)
    audit=open_observables(sol,stack,p,config)['metrics']
    assert audit['relative_lithium_drift']<1e-9
    assert audit['relative_internal_energy_residual']<1e-5
    assert audit['relative_entropy_residual']<1e-4
    assert max(audit['port_identity_relative_residual'].values())<1e-10
    assert audit['minimum_dissipation_W']>=-1e-7
