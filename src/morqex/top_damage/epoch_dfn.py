"""Persistent DFN half-cell with explicit, bounded microstructure input ports.

No remeshing or material deletion: differential lithium states persist when
effective transport and area-averaged kinetics change. The scalar mechanical
potential port is a mean-field closure, not a resolved stress-diffusion model.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pybamm as pb
from pybamm.models.submodels.transport_efficiency.base_transport_efficiency import BaseModel
from morqex.nodewise.halfcell import lithium_halfcell_parameters
from morqex.solvers.pybamm_backend import solver
from .electrolyte_boundary import ConservativeHalfCellElectrolyte


@dataclass(frozen=True)
class DFNFeedback:
    electronic_conductivity_S_m: float
    electrolyte_efficiency: float
    reaction_fraction: float = 1.
    mechanical_potential_J_mol: float = 0.

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():raise ValueError('Finite physical feedback required')
        if self.electronic_conductivity_S_m<=0 or not 0<self.electrolyte_efficiency<=1 or not 0<self.reaction_fraction<=1:
            raise ValueError('Disconnected electrode must terminate or change the state model, not use invalid feedback')

    def inputs(self,current_A):
        return {'Current function [A]':float(current_A),
                'TOP-DAMAGE electronic conductivity [S.m-1]':self.electronic_conductivity_S_m,
                'TOP-DAMAGE electrolyte efficiency':self.electrolyte_efficiency,
                'TOP-DAMAGE reaction fraction':self.reaction_fraction,
                'TOP-DAMAGE mechanical potential [J.mol-1]':self.mechanical_potential_J_mol}


class GeometryEfficiency(BaseModel):
    def get_coupled_variables(self,v):
        parts={}
        for domain in self.options.whole_cell_domains:
            if domain=='positive electrode':
                value=pb.InputParameter('TOP-DAMAGE electrolyte efficiency') if self.component=='Electrolyte' else pb.Scalar(1.)
                parts[domain]=pb.FullBroadcast(value,domain,'current collector')
            elif self.component=='Electrolyte':
                parts[domain]=v[domain.capitalize()+' porosity']**self.param.domain_params[domain.split()[0]].b_e
            else:parts[domain]=pb.FullBroadcast(0,domain,'current collector')
        v.update(self._get_standard_transport_efficiency_variables(parts));return v


class EpochDFN:
    def __init__(self,*,host='NMC811',active_fraction,porosity,radius_m,temperature_K=298.15,nx=6,nr=20,initial_stoichiometry=.5):
        if not 0<active_fraction<1 or not 0<porosity<1 or active_fraction+porosity>=1:raise ValueError('Explicit active/CBD/pore volumes required')
        if not np.isfinite([radius_m,temperature_K]).all() or radius_m<=0 or temperature_K<=0:raise ValueError('Positive radius and temperature required')
        if not .25<=initial_stoichiometry<.9:raise ValueError('Initial stoichiometry outside the declared half-cell window')
        self.parameters,self.audit=lithium_halfcell_parameters(host=host,active_fraction=active_fraction)
        p=self.parameters;ocp=p['Positive electrode OCP [V]'];j0=p['Positive electrode exchange-current density [A.m-2]']
        p.update({'Positive electrode porosity':porosity,'Positive particle radius [m]':radius_m,
            'Initial concentration in positive electrode [mol.m-3]':initial_stoichiometry*float(p['Maximum concentration in positive electrode [mol.m-3]']),
            'Current function [A]':'[input]','Initial temperature [K]':temperature_K,'Ambient temperature [K]':temperature_K,
            'Positive electrode conductivity [S.m-1]':lambda T:pb.InputParameter('TOP-DAMAGE electronic conductivity [S.m-1]'),
            'Positive electrode OCP [V]':lambda sto:ocp(sto)-pb.InputParameter('TOP-DAMAGE mechanical potential [J.mol-1]')/pb.constants.F,
            'Positive electrode exchange-current density [A.m-2]':lambda ce,cs,cmax,T:pb.InputParameter('TOP-DAMAGE reaction fraction')*j0(ce,cs,cmax,T)})
        model=pb.lithium_ion.DFN({'working electrode':'positive','thermal':'isothermal'},build=False)
        model.submodels['electrolyte diffusion']=ConservativeHalfCellElectrolyte(model.param,model.options)
        model.submodels['electrolyte transport efficiency']=GeometryEfficiency(model.param,'Electrolyte',model.options)
        model.submodels['electrode transport efficiency']=GeometryEfficiency(model.param,'Electrode',model.options)
        model.build_model();self.model=model
        self.simulation=pb.Simulation(model,parameter_values=p,solver=solver(rtol=1e-8,atol=1e-9),
            var_pts={**model.default_var_pts,'x_p':nx,'x_s':5,'r_p':nr})
        self.simulation.build();self.last=None;self.last_inputs=None
        self.capacity_Ah=float(p['Nominal cell capacity [A.h]']);self.cmax=float(p['Maximum concentration in positive electrode [mol.m-3]'])
        self.reference_c=float(p['Initial concentration in positive electrode [mol.m-3]'])
        area=float(p['Electrode width [m]'])*float(p['Electrode height [m]'])
        self.initial_inventory=area*(float(p['Positive electrode thickness [m]'])*active_fraction*self.reference_c+
            float(p['Initial concentration in electrolyte [mol.m-3]'])*(float(p['Positive electrode thickness [m]'])*porosity+
            float(p['Separator thickness [m]'])*float(p['Separator porosity'])))

    def advance(self,dt,current_A,feedback):
        if not np.isfinite([dt,current_A]).all() or dt<=0:raise ValueError('Finite positive duration and current required')
        inputs=feedback.inputs(current_A)
        previous=self.last
        if previous is not None and previous.termination!='final time':
            if 'voltage' not in previous.termination.lower() or self.last_inputs['Current function [A]']*current_A>=0:
                raise RuntimeError('Only a voltage cutoff followed by current reversal can restart this protocol')
            previous=pb.Solution(np.array([float(previous.t[-1])]),np.asarray(previous.last_state.y),
                self.simulation.built_model,self.last_inputs,check_solution=False)
        sol=self.simulation.step(dt,starting_solution=previous,inputs=inputs,save=False,
                                 t_interp=np.linspace(0.,dt,max(3,int(np.ceil(dt/30))+1)))
        c=np.asarray(sol['Positive particle concentration [mol.m-3]'].entries)
        if not np.isfinite(c).all() or c.min()<=0 or c.max()>=self.cmax:raise RuntimeError('Particle concentration bounds failed')
        q=np.asarray(sol['Discharge capacity [A.h]'].entries)
        lithium=np.asarray(sol['Total lithium [mol]'].entries)
        inventory_defect=float(np.max(np.abs(lithium-self.initial_inventory-q*3600/96485.33212))/self.initial_inventory)
        if inventory_defect>1e-6:raise RuntimeError(f'Half-cell lithium ledger failed: {inventory_defect}')
        jump=0.
        if previous is not None:
            for key in ['Positive particle concentration [mol.m-3]','Electrolyte concentration [mol.m-3]']:
                a=np.asarray(previous[key].entries)[...,-1];b=np.asarray(sol[key].entries)[...,0]
                jump=max(jump,float(np.max(abs(a-b))/max(1.,np.max(abs(a)))))
            if jump>1e-9:raise RuntimeError('Feedback update changed a differential concentration state')
        self.last=sol;self.last_inputs=inputs
        return {'time_s':float(sol.t[-1]),'voltage_V':float(sol['Voltage [V]'].entries[-1]),
                'termination':sol.termination,'inventory_relative_defect':inventory_defect,
                'relative_state_jump':jump,'net_capacity_Ah':float(q[-1]),'feedback':asdict(feedback)}

    def restore_state(self,time_s,state,inputs):
        state=np.asarray(state,float)
        if state.shape!=(self.simulation.built_model.len_rhs_and_alg,) or not np.isfinite(state).all():raise ValueError('Invalid DFN state layout')
        self.last=pb.Solution(np.array([time_s]),state[:,None],self.simulation.built_model,inputs,check_solution=False)
        self.last_inputs=dict(inputs)

    def radial_profile(self,x_index=2):
        if self.last is None:raise ValueError('Advance DFN before microstructure reconstruction')
        c=np.asarray(self.last['Positive particle concentration [mol.m-3]'].entries)[..., -1]
        mesh=self.simulation.mesh['positive particle']
        r=np.asarray(mesh.nodes)/float(self.parameters['Positive particle radius [m]'])
        surface=float(np.asarray(self.last['Positive particle surface concentration [mol.m-3]'].entries)[x_index,-1])
        return np.r_[0.,r,1.],np.r_[c[0,x_index],c[:,x_index],surface]
