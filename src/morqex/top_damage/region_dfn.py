"""DFN with geometric region states and spatial finite-mechanics input ports.

Uses the existing MORQEX particle/kinetics API. The positive radial states are
replaced, not duplicated. Mechanical chemical potentials are frozen external
inputs over an accepted segment; coupling convergence is a separate gate.
"""
import numpy as np
import pybamm as pb
from morqex.electrochemistry.microstructure_dfn import (
    MicrostructureParticle, MicrostructureOCP, MicrostructureKinetics,
    GeometryActiveMaterial,key,dot)
from morqex.electrochemistry.open_particle import TransportLaw
from morqex.nodewise.halfcell import lithium_halfcell_parameters
from morqex.solvers.pybamm_backend import solver
from .epoch_dfn import EpochDFN,GeometryEfficiency
from .electrolyte_boundary import ConservativeHalfCellElectrolyte


def mu_input(region):return f'TOP-DAMAGE region {region} mechanical potential [J.mol-1]'


class RegionParticle(MicrostructureParticle):
    def __init__(self,*args,transport_law,**kwargs):
        super().__init__(*args,**kwargs);self.law=transport_law

    def get_coupled_variables(self,v):
        d,D=self.domain_Domain;op=self.op;phase=self.phase_param;law=self.law
        T=v[f'{D} electrode temperature [K]']
        c=[v[key(d,k,'concentration [mol.m-3]')] for k in range(op.dimension)]
        chemical=[-self.param.F*phase.U(a/phase.c_max,T) for a in c]
        grad=[law.gradient_coefficient_J_m5_mol2*dot(row,c)/float(w) for row,w in zip(op.stiffness.toarray(),op.weights)]
        mu=[a+v[key(d,k,'elastic chemical potential [J.mol-1]')]+grad[k] for k,a in enumerate(chemical)]
        mean=dot(op.weights,c)/op.active_fraction
        mobility=law.mobility_multiplier*phase.D(mean,T)*phase.c_max/(self.param.R*T)
        if law.mobility=='lattice':mobility*=mean/phase.c_max*(1-mean/phase.c_max)
        diffusive=[-mobility*dot(row,mu) for row in op.stiffness.toarray()]
        j=[v[key(d,k,'reaction current [A.m-2]')] for k in range(op.dimension)]
        massdot=[a-float(A)*b/self.param.F for a,A,b in zip(diffusive,op.reaction_area,j)]
        diss=-sum((f*m for f,m in zip(diffusive,mu)),pb.Scalar(0))
        reaction=sum((float(A)*a*v[key(d,k,'reaction overpotential [V]')] for k,(A,a) in enumerate(zip(op.reaction_area,j))),pb.Scalar(0))
        power=sum((a*b for a,b in zip(massdot,mu)),pb.Scalar(0))
        port=v[f'{D} electrode surface potential difference [V]']*dot(op.reaction_area,j)
        for k,rate in enumerate(massdot):v[key(d,k,'rate [mol.m-3.s-1]')]=rate/float(op.weights[k])
        v.update({f'MORQEX {d} diffusion dissipation [W.m-3]':diss,
                  f'MORQEX {d} reaction dissipation [W.m-3]':reaction,
                  f'MORQEX {d} port identity residual [W.m-3]':power+diss+reaction-port})
        v.update(self._get_standard_diffusivity_variables(phase.D(mean,T)))
        v.update(self._get_standard_flux_variables(0*mean));return v

    def add_events_from(self,v):
        lo,hi=self.law.stoichiometry_bounds
        for k in range(self.op.dimension):
            c=v[key(self.domain,k,'concentration [mol.m-3]')]
            self.events.extend([pb.Event(f'TOP-DAMAGE region {k} lower bound',pb.min(c)-lo*self.phase_param.c_max),
                                pb.Event(f'TOP-DAMAGE region {k} upper bound',hi*self.phase_param.c_max-pb.max(c))])


class RegionMechanics(pb.BaseSubModel):
    def __init__(self,param,options,op,nx):super().__init__(param,options=options);self.op=op;self.nx=nx
    def get_fundamental_variables(self):
        return {key('positive',k,'elastic chemical potential [J.mol-1]'):
                pb.SecondaryBroadcast(pb.InputParameter(mu_input(k),domain='positive electrode'),'current collector') for k in range(self.op.dimension)} | {
                key('positive',k,'elastic chemical potential dT [J.mol-1.K-1]'):pb.Scalar(0) for k in range(self.op.dimension)}


class RegionOCP(MicrostructureOCP):
    def __init__(self,*args,transport_law,**kwargs):super().__init__(*args,**kwargs);self.law=transport_law
    def get_coupled_variables(self,v):
        # The same gradient-energy derivative enters diffusion and reaction.
        d,D=self.domain_Domain;op=self.op;T=v[f'{D} electrode temperature [K]']
        c=[v[key(d,k,'concentration [mol.m-3]')] for k in range(op.dimension)]
        grad=[self.law.gradient_coefficient_J_m5_mol2*dot(row,c)/float(w) for row,w in zip(op.stiffness.toarray(),op.weights)]
        U=[]
        for k,a in enumerate(c):
            value=self.phase_param.U(a/self.phase_param.c_max,T)-(v[key(d,k,'elastic chemical potential [J.mol-1]')]+grad[k])/self.param.F
            U.append(value);v[key(d,k,'open circuit potential [V]')]=value
        surface=dot(op.reaction_area,U)/op.area_density;bulk=pb.x_average(dot(op.weights,U)/op.active_fraction)
        v.update(self._get_standard_ocp_variables(surface,bulk,pb.Scalar(0)))
        v[f'{D} electrode equilibrium open-circuit potential [V]']=surface
        v[f'X-averaged {d} electrode equilibrium open-circuit potential [V]']=pb.x_average(surface)
        return v


class RegionEpochDFN(EpochDFN):
    """Isothermal half-cell; shared geometry, independent states at all nx nodes.

    Fixed topology and reaction mask within this model. The returned concentrations
    have shape (nx, regions), and `port.field` recovers every active voxel.
    Existing scalar-epoch restore/advance performs mass and state-jump checks.
    No finite elastic H is assumed; external potentials must be projected from
    the current nonlinear microstate. Input updates do not integrate fatigue.
    """
    def __init__(self,port,*,host='NMC811',porosity,temperature_K=298.15,nx=6,transport_law=None):
        law=transport_law or TransportLaw();self.port=port;self.nx=nx;self.transport_law=law
        if type(nx) is not int or nx<3:raise ValueError('At least three DFN thickness cells required by boundary extrapolation')
        # Reuse parameter/source/inventory bookkeeping. Replace its radial model
        # before any advance; no radial state participates in this solver.
        p,_=lithium_halfcell_parameters(host=host,active_fraction=port.active_fraction)
        cmax=float(p['Maximum concentration in positive electrode [mol.m-3]'])
        super().__init__(host=host,active_fraction=port.active_fraction,porosity=porosity,
            radius_m=3*port.active_fraction/port.area_density,temperature_K=temperature_K,nx=nx,
            initial_stoichiometry=port.reference_concentration/cmax)
        p0,_=lithium_halfcell_parameters(host=host,active_fraction=port.active_fraction)
        curve=p0['Positive electrode OCP [V]']
        lo,hi=law.stoichiometry_bounds
        if lo<curve.stoichiometry[0] or hi>curve.stoichiometry[-1]:raise ValueError('Region concentration bounds exceed OCP table')
        if curve.slope_bounds(temperature_K)[1]>1e-12 and law.gradient_coefficient_J_m5_mol2<=0:
            raise ValueError('Nonconvex OCP requires declared gradient energy and spatial convergence')
        self.parameters.update({'Positive electrode OCP [V]':curve,
            'Positive electrode exchange-current density [A.m-2]':p0['Positive electrode exchange-current density [A.m-2]']})
        model=pb.lithium_ion.DFN({'working electrode':'positive','thermal':'isothermal'},build=False)
        model.submodels['electrolyte diffusion']=ConservativeHalfCellElectrolyte(model.param,model.options)
        model.submodels['electrolyte transport efficiency']=GeometryEfficiency(model.param,'Electrolyte',model.options)
        model.submodels['electrode transport efficiency']=GeometryEfficiency(model.param,'Electrode',model.options)
        model.submodels['positive primary active material']=GeometryActiveMaterial(model.param,'positive',model.options,operator=port)
        model.submodels['positive primary particle']=RegionParticle(model.param,'positive',model.options,operator=port,transport_law=law)
        model.submodels['positive primary open-circuit potential']=RegionOCP(model.param,'positive','lithium-ion main',model.options,operator=port,transport_law=law)
        model.submodels['positive primary interface']=MicrostructureKinetics(model.param,'positive','lithium-ion main',model.options,operator=port)
        model.submodels['TOP-DAMAGE finite mechanics input']=RegionMechanics(model.param,model.options,port,nx)
        model.build_model();self.model=model
        self.simulation=pb.Simulation(model,parameter_values=self.parameters,solver=solver(rtol=1e-8,atol=1e-9),
            var_pts={**model.default_var_pts,'x_p':nx,'x_s':5,'r_p':2})
        self.simulation.build();self.last=None;self.last_inputs=None

    def advance_regions(self,dt,current_A,feedback,mechanical_potential_J_mol):
        mu=np.asarray(mechanical_potential_J_mol,float)
        if mu.shape!=(self.nx,self.port.dimension) or not np.isfinite(mu).all():raise ValueError('Finite node-by-region mechanical chemical potentials required')
        if feedback.reaction_fraction!=1 or feedback.mechanical_potential_J_mol!=0:
            raise ValueError('Region reaction geometry and local mu replace scalar reaction/mu feedback')
        base=feedback.inputs(current_A)
        base.update({mu_input(k):mu[:,k] for k in range(self.port.dimension)})
        previous_solution=self.simulation._solution
        try:
            return self._advance_regions(dt,current_A,feedback,base)
        except Exception:
            self.simulation._solution=previous_solution
            raise

    def _advance_regions(self,dt,current_A,feedback,inputs):
        # Independent accepted-state transaction; never commits on failed gates.
        if not np.isfinite([dt,current_A]).all() or dt<=0:raise ValueError('Finite positive duration required')
        previous=self.last
        if previous is not None and previous.termination!='final time':
            if 'voltage' not in previous.termination.lower() or self.last_inputs['Current function [A]']*current_A>=0:
                raise RuntimeError('Voltage cutoff requires current reversal')
            previous=pb.Solution(np.array([float(previous.t[-1])]),np.asarray(previous.last_state.y),self.simulation.built_model,self.last_inputs,check_solution=False)
        sol=self.simulation.step(dt,starting_solution=previous,inputs=inputs,save=False,t_interp=np.linspace(0,dt,5))
        concentrations=np.stack([sol[key('positive',k,'concentration [mol.m-3]')].entries for k in range(self.port.dimension)],axis=1)
        lo,hi=self.transport_law.stoichiometry_bounds
        if not np.isfinite(concentrations).all() or concentrations.min()<lo*self.cmax or concentrations.max()>hi*self.cmax:raise RuntimeError('Regional concentration bounds failed')
        q=np.asarray(sol['Discharge capacity [A.h]'].entries);li=np.asarray(sol['Total lithium [mol]'].entries)
        defect=float(np.max(abs(li-self.initial_inventory-q*3600/96485.33212))/self.initial_inventory)
        if defect>1e-6:raise RuntimeError('Regional DFN lithium ledger failed')
        jump=0.
        if previous is not None:
            for k in range(self.port.dimension):
                before=np.asarray(previous[key('positive',k,'concentration [mol.m-3]')].entries)[...,-1]
                after=concentrations[:,k,0];jump=max(jump,float(np.max(abs(after-before))/self.cmax))
            before=np.asarray(previous['Electrolyte concentration [mol.m-3]'].entries)[...,-1]
            after=np.asarray(sol['Electrolyte concentration [mol.m-3]'].entries)[...,0]
            jump=max(jump,float(np.max(abs(after-before))/max(1.,np.max(abs(before)))))
            if jump>1e-9:raise RuntimeError('Mechanical input update changed regional lithium state')
        self.last=sol;self.last_inputs=inputs
        return {'time_s':float(sol.t[-1]),'voltage_V':float(sol['Voltage [V]'].entries[-1]),'termination':sol.termination,
                'inventory_relative_defect':defect,'relative_state_jump':jump,'net_capacity_Ah':float(q[-1])}

    def concentrations(self):
        if self.last is None:return np.full((self.nx,self.port.dimension),self.port.reference_concentration)
        return np.column_stack([np.asarray(self.last[key('positive',k,'concentration [mol.m-3]')].entries)[...,-1] for k in range(self.port.dimension)])

    def radial_profile(self,*args,**kwargs):raise TypeError('Region DFN has no radial states; use concentrations() and port.field()')
