import numpy as np
import pybamm as pb
from morqex import Workspace
from morqex.gemsim import GEMsim
from morqex.geometry import MicrostructureData
from morqex.topology import PorousCBD
from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode
from morqex.electrochemistry.open_particle import build_open_dfn,TransportLaw
from morqex.electrochemistry.aging_dfn import EnergyKernel
from morqex.electrochemistry.open_validation import open_observables
from morqex.nodewise.halfcell import HalfCellStack,lithium_halfcell_parameters
from morqex.degradation import DegradationOptions,NoDamage
from morqex.solvers.pybamm_backend import solver
from dataclasses import asdict

def test_six_node_planar_li_mass_energy_and_no_negative_particles(tmp_path):
 p,_=lithium_halfcell_parameters(active_fraction=.5)
 a=np.ones((4,)*3,np.uint8);a[0]=0;a[1]=2
 geometry=MicrostructureData(a,(1e-6,)*3,(0.,)*3,{0:'pore',1:'active',2:'cbd'},{'source':'synthetic qualification'})
 f=MicrostructureElectrode(geometry,GEMsim(Workspace(tmp_path)),p,'positive',topology=PorousCBD());op=f.partition(4)
 op.transport_law=asdict(TransportLaw(gradient_coefficient_J_m5_mol2=1e-14))
 stack=HalfCellStack([EnergyKernel(op.detach())]*6,p,constraint='fixed-thickness');stack.update_parameters(p)
 config=DegradationOptions(negative=(),positive=(),damage=NoDamage())
 model=build_open_dfn(stack,stack.kernels,config,parameters=p)
 assert not [n for n in model.submodels if 'negative' in n and 'particle' in n]
 assert set(stack.electrodes)=={'positive'} and stack.var_pts(model)['x_p']==6
 sim=pb.Simulation(model,parameter_values=p,solver=solver(rtol=1e-8,atol=1e-9),var_pts=stack.var_pts(model))
 sol=sim.solve([0,30]);metrics=open_observables(sol,stack,p,config)['metrics']
 assert metrics['relative_lithium_drift']<1e-9
 assert metrics['relative_internal_energy_residual']<1e-5
 assert metrics['relative_entropy_residual']<1e-4
 assert np.max(abs(sol['MORQEX sandwich thickness change [m]'].entries))<1e-12

def test_composed_jacobian_and_directional_action():
 model=pb.BaseModel();a=pb.Variable('a');b=pb.Variable('b')
 model.rhs={a:-2*a+b*b};model.algebraic={b:a+b-1};model.initial_conditions={a:.2,b:.8}
 sim=pb.Simulation(model,solver=solver(rtol=1e-9,atol=1e-10));sim.solve([0,.1]);m=sim.built_model
 y=np.array([.21,.76]);expected=np.array([[-2,1.52],[1,1]])
 np.testing.assert_allclose(np.asarray(m.jac_rhs_algebraic_eval(0,y,[])),expected,rtol=1e-13,atol=1e-13)
 v=np.array([.3,-.7]);np.testing.assert_allclose(np.asarray(m.jac_rhs_algebraic_action_eval(0,y,[],v)).ravel(),expected@v,rtol=1e-13,atol=1e-13)
