"""Full PyBaMM DFN with MORQEX mean-RVE/spherical energy feedback and heat.

Only the declared additive closure is implemented: RVE mean concentration plus
free-sphere zero-mean diffusion energy. No arbitrary-geometry radial mapping.
Public submodel composition uses the bundled, derivative-patched PyBaMM solver.
"""
import os
os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY','true')
import pybamm as pb
from pybamm.models.submodels.particle.fickian_diffusion import FickianDiffusion
from pybamm.models.submodels.interface.open_circuit_potential.single_ocp import SingleOpenCircuitPotential
from pybamm.models.submodels.thermal.lumped import Lumped


class RVEParticleMechanics(pb.BaseSubModel):
    def __init__(self,param,options,stack,feedback=1.):
        super().__init__(param,options=options);self.stack=stack;self.feedback=feedback

    def get_coupled_variables(self,variables):
        T=variables['Volume-averaged cell temperature [K]'];theta=T-self.param.T_ref
        deviations={d:variables[f'R-averaged {d} particle concentration [mol.m-3]']-e.reference_concentration for d,e in self.stack.electrodes.items()}
        traction=self.stack.traction({d:pb.x_average(v) for d,v in deviations.items()},theta)
        energy=0;thickness=self.stack.separator_length*traction/self.stack.separator_C
        # Match PyBaMM's heat-source normalization by actual thermal cell volume,
        # including all parallel electrode pairs (not nominal stack thickness).
        volume_curvature=self.stack.energy_TT_per_area*self.param.n_electrodes_parallel*self.param.L_y*self.param.L_z/self.param.V_cell
        outputs={'MORQEX stack axial stress [Pa]':traction,'MORQEX elastic heat-capacity coefficient [J.m-3.K-2]':volume_curvature}
        for d,e in self.stack.electrodes.items():
            D=d.capitalize();h=e.dfn_H;dc=deviations[d]
            strain=(traction-h[0,1]*dc-h[0,2]*theta)/h[0,0]
            mu=(h[1,0]*strain+h[1,1]*dc+h[1,2]*theta)/e.dfn_active_fraction
            W=.5*h[0,0]*strain**2+strain*(h[0,1]*dc+h[0,2]*theta)+.5*h[1,1]*dc**2+h[1,2]*dc*theta+.5*h[2,2]*theta**2
            c=variables[f'{D} particle concentration [mol.m-3]'];average=variables[f'R-averaged {d} particle concentration [mol.m-3]'];surface=variables[f'{D} particle surface concentration [mol.m-3]']
            phase=self.param.domain_params[d].prim
            particle_T=pb.PrimaryBroadcast(variables[f'{D} electrode temperature [K]'],f'{d} particle')
            chemical_curvature=-self.param.F*phase.U(c/phase.c_max,particle_T).diff(c)
            fluctuation=c-pb.PrimaryBroadcast(average,f'{d} particle')
            particle_energy=.5*e.dfn_active_fraction*e.sphere_curvature*pb.r_average(fluctuation**2)
            mu_surface=mu+e.sphere_curvature*(surface-average)
            tangential=e.molar_volume*e.young/(3*(1-e.poisson))*(average-surface)
            radius=self.param.domain_params[d].prim.R
            displacement=e.molar_volume*radius*(average-e.reference_concentration)/3
            energy+=self.stack.lengths[d]*pb.x_average(W+particle_energy)
            thickness+=self.stack.lengths[d]*pb.x_average(strain)
            outputs.update({f'MORQEX {d} mean mechanical chemical potential [J.mol-1]':mu,
                f'MORQEX {d} surface mechanical chemical potential [J.mol-1]':mu_surface,
                f'MORQEX {d} mechanical chemical-potential temperature derivative [J.mol-1.K-1]':pb.Scalar(self.stack.mu_T[d]),
                f'MORQEX {d} axial strain':strain,f'MORQEX {d} RVE energy density [J.m-3]':W,
                f'MORQEX {d} particle gradient energy density [J.m-3]':particle_energy,
                f'MORQEX {d} particle surface tangential stress [Pa]':tangential,
                f'MORQEX {d} particle surface radial stress [Pa]':0*surface,
                f'MORQEX {d} particle surface displacement [m]':displacement,
                f'MORQEX {d} chemical curvature [J.m3.mol-2]':chemical_curvature,
                f'MORQEX {d} diffusional elastic curvature [J.m3.mol-2]':pb.Scalar(e.sphere_curvature)})
        energy+=.5*self.stack.separator_length*traction**2/self.stack.separator_C
        outputs.update({'MORQEX mechanical energy per area [J.m-2]':energy,'MORQEX sandwich thickness change [m]':thickness})
        variables.update(outputs);return variables

    def add_events_from(self,variables):
        if self.feedback:
            for d in self.stack.electrodes:
                self.events.append(pb.Event(f'MORQEX {d} nonconvex chemical branch',pb.min(variables[f'MORQEX {d} chemical curvature [J.m3.mol-2]'])-1e-8))


class EnergeticDiffusion(FickianDiffusion):
    def __init__(self,*args,curvature,feedback=1.,**kwargs):
        super().__init__(*args,**kwargs);self.curvature=curvature;self.feedback=feedback

    def _get_effective_diffusivity(self,c,T,current):
        D=super()._get_effective_diffusivity(c,T,current)
        if self.feedback==0:return D
        chemical_curvature=-self.param.F*self.phase_param.U(c/self.phase_param.c_max,T).diff(c)
        # D is chemical diffusivity. M=D/f'' recovers stock Fick at zero coupling.
        return D*(1+self.feedback*self.curvature/chemical_curvature)


class EnergeticOCP(SingleOpenCircuitPotential):
    def __init__(self,*args,feedback=1.,**kwargs):
        super().__init__(*args,**kwargs);self.feedback=feedback

    def get_coupled_variables(self,variables):
        if self.feedback==0:return super().get_coupled_variables(variables)
        d,D=self.domain_Domain
        sto,bulk,T,Tbulk=self._get_stoichiometry_and_temperature(variables)
        mu=variables[f'MORQEX {d} surface mechanical chemical potential [J.mol-1]']
        mu_bulk=pb.x_average(variables[f'MORQEX {d} mean mechanical chemical potential [J.mol-1]'])
        mu_T=variables[f'MORQEX {d} mechanical chemical-potential temperature derivative [J.mol-1.K-1]']
        surface=self.phase_param.U(sto,T)-self.feedback*mu/self.param.F
        ocp_bulk=self.phase_param.U(bulk,Tbulk)-self.feedback*mu_bulk/self.param.F
        entropy=self.phase_param.dUdT(sto)-self.feedback*mu_T/self.param.F
        variables.update(self._get_standard_ocp_variables(surface,ocp_bulk,entropy))
        variables[f'{D} electrode equilibrium open-circuit potential [V]']=surface
        variables[f'X-averaged {d} electrode equilibrium open-circuit potential [V]']=pb.x_average(surface)
        return variables


class EnergeticThermal(Lumped):
    def __init__(self,*args,stack,feedback=1.,**kwargs):
        super().__init__(*args,**kwargs);self.stack=stack;self.feedback=feedback

    def get_fundamental_variables(self):
        variables=super().get_fundamental_variables()
        for name in ['heating','cooling']:
            variables[f'MORQEX accumulated thermal {name} [J.m-3]']=pb.Variable(f'MORQEX accumulated thermal {name} [J.m-3]',scale=1e6)
        return variables

    def _heat_of_mixing(self,variables):
        if self.feedback==0:return super()._heat_of_mixing(variables)
        parts=[]
        for d,e in self.stack.electrodes.items():
            D=d.capitalize();phase=self.param.domain_params[d].prim
            c=variables[f'{D} particle concentration [mol.m-3]']
            T=pb.PrimaryBroadcast(variables[f'{D} electrode temperature [K]'],f'{d} particle')
            curvature=-self.param.F*phase.U(c/phase.c_max,T).diff(c)
            elastic=self.feedback*e.sphere_curvature
            mobility=phase.D(c,T)/curvature
            gradient2=pb.inner(pb.grad(c),pb.grad(c))
            # Irreversible chemical+elastic diffusion loss for this declared mobility.
            parts.append(e.dfn_active_fraction*pb.r_average(mobility*(curvature+elastic)**2*gradient2))
        separator=pb.FullBroadcast(0,'separator','current collector')
        return parts[0],separator,parts[1]

    def set_rhs(self,variables):
        super().set_rhs(variables)
        temperature=variables['Volume-averaged cell temperature [K]']
        cp=variables['Volume-averaged effective heat capacity [J.K-1.m-3]']
        volume_curvature=variables['MORQEX elastic heat-capacity coefficient [J.m-3.K-2]']
        correction=self.feedback*temperature*volume_curvature
        effective=cp-correction
        self.rhs[temperature]=self.rhs[temperature]*cp/effective
        variables['MORQEX thermoelastic effective heat capacity [J.K-1.m-3]']=effective
        variables['MORQEX thermoelastic heat capacity correction [J.K-1.m-3]']=correction
        self.rhs[variables['MORQEX accumulated thermal heating [J.m-3]']]=variables['Volume-averaged total heating [W.m-3]']+variables['Lumped contact resistance heating [W.m-3]']
        self.rhs[variables['MORQEX accumulated thermal cooling [J.m-3]']]=variables['Surface total cooling [W.m-3]']
        initial=pb.x_average(self.param.T_init)
        # Qualified ledger for the selected constant heat-capacity parameter sets.
        storage=cp*(temperature-initial)-.5*self.feedback*volume_curvature*(temperature**2-initial**2)
        variables['MORQEX thermal storage change [J.m-3]']=storage
        variables['MORQEX thermal balance residual [J.m-3]']=storage-variables['MORQEX accumulated thermal heating [J.m-3]']-variables['MORQEX accumulated thermal cooling [J.m-3]']

    def set_initial_conditions(self,variables):
        super().set_initial_conditions(variables)
        for name in ['heating','cooling']:self.initial_conditions[variables[f'MORQEX accumulated thermal {name} [J.m-3]']]=pb.Scalar(0)

    def add_events_from(self,variables):
        if self.feedback:self.events.append(pb.Event('MORQEX nonpositive thermoelastic heat capacity',variables['MORQEX thermoelastic effective heat capacity [J.K-1.m-3]']-1.))


def build_model(stack,*,feedback=1.,thermal=True):
    if not 0<=feedback<=1:raise ValueError('Feedback strength must lie in [0,1]')
    options={'thermal':'lumped' if thermal else 'isothermal','heat of mixing':'true','stress-induced diffusion':'false'}
    model=pb.lithium_ion.DFN(options=options,build=False,name='MORQEX full DFN thermo-chemo-mechanical closure')
    if not thermal:raise ValueError('Coupled model currently requires lumped thermal; use stock DFN for isothermal reference')
    model.submodels['MORQEX energy mechanics']=RVEParticleMechanics(model.param,model.options,stack,feedback=feedback)
    for d,e in stack.electrodes.items():
        model.submodels[f'{d} primary particle']=EnergeticDiffusion(model.param,d,model.options,curvature=e.sphere_curvature,feedback=feedback)
        model.submodels[f'{d} primary open-circuit potential']=EnergeticOCP(model.param,d,'lithium-ion main',model.options,feedback=feedback)
    model.submodels['thermal']=EnergeticThermal(model.param,model.options,stack=stack,feedback=feedback)
    model.build_model()
    return model
