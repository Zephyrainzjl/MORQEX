"""Monolithic DFN coupled to conservative, energy-conjugate microstructure ports.

State: intrinsic region concentrations at every DFN electrode location; these
replace radial particles. Geometry supplies mass, reactive area, graph transport
and the full nonuniform FEM energy. The thermal model uses electrical boundary
power minus interface port power, with explicit thermoelastic entropy coupling.
Qualified chemistry: ideal electrolyte, constant t+, zero chemical OCP entropy,
constant heat capacity, fixed geometry/C, zero external mechanical power.
"""
import numpy as np
import pybamm as pb
from pybamm.models.submodels.particle.base_particle import BaseParticle
from pybamm.models.submodels.active_material.constant_active_material import Constant
from pybamm.models.submodels.interface.kinetics.butler_volmer import SymmetricButlerVolmer
from pybamm.models.submodels.interface.open_circuit_potential.base_ocp import BaseOpenCircuitPotential
from pybamm.models.submodels.transport_efficiency.base_transport_efficiency import BaseModel as TransportBase
from pybamm.models.submodels.thermal.lumped import Lumped


def key(domain,region,quantity):return f'MORQEX {domain} region {region} {quantity}'
def dot(a,b):return sum((float(x)*y for x,y in zip(a,b)),pb.Scalar(0))


class MicrostructureStack:
    def __init__(self,negative,positive,parameters,*,constraint='pressure',separator_modulus_Pa=1e8):
        if constraint not in {'pressure','fixed-thickness'}:raise ValueError('Zero pressure or fixed total thickness required')
        if parameters['Thermodynamic factor']!=1 or not isinstance(parameters['Cation transference number'],(float,int)):
            raise ValueError('Ideal electrolyte with constant transference number required for the entropy closure')
        if not np.isfinite(separator_modulus_Pa) or separator_modulus_Pa<=0:raise ValueError('Positive separator stiffness required')
        for d in ['Negative','Positive']:
            if parameters[f'{d} electrode OCP entropic change [V.K-1]']!=0:raise ValueError('Zero chemical OCP entropy required by this energy primitive')
        for material in ['Negative electrode','Positive electrode','Negative current collector','Positive current collector','Separator']:
            for quantity in ['density [kg.m-3]','specific heat capacity [J.kg-1.K-1]']:
                if not isinstance(parameters[f'{material} {quantity}'],(float,int)):raise ValueError('Constant thermal density/heat capacity required')
        self.electrodes={'negative':negative,'positive':positive};self.constraint=constraint
        self.lengths={d:float(parameters[f'{d.capitalize()} electrode thickness [m]']) for d in self.electrodes}
        self.separator_length=float(parameters['Separator thickness [m]']);self.separator_C=separator_modulus_Pa
        self.compliance=self.separator_length/self.separator_C+sum(self.lengths[d]/op.H[0,0] for d,op in self.electrodes.items())
        numerator=sum(self.lengths[d]*op.H[0,-1]/op.H[0,0] for d,op in self.electrodes.items())
        self.stress_T=numerator/self.compliance if constraint=='fixed-thickness' else 0.
        self.energy_TT_per_area=sum(self.lengths[d]*(op.H[-1,-1]-op.H[0,-1]**2/op.H[0,0]) for d,op in self.electrodes.items())
        if constraint=='fixed-thickness':self.energy_TT_per_area+=numerator**2/self.compliance
        self.mu_T={d:(op.H[1:-1,-1]+op.H[1:-1,0]*(self.stress_T-op.H[0,-1])/op.H[0,0])/op.weights for d,op in self.electrodes.items()}

    def update_parameters(self,parameters):
        for d,op in self.electrodes.items():
            D=d.capitalize()
            if float(parameters[f'Initial concentration in {d} electrode [mol.m-3]'])!=op.reference_concentration:raise ValueError('Operator reference concentration mismatch')
            material=op.metadata()['material']
            if float(parameters['Reference temperature [K]'])!=op.metadata()['reference_temperature_K']:raise ValueError('Operator thermal reference mismatch')
            for parameter,field in [(f"{D} electrode Young's modulus [Pa]",'young_Pa'),(f"{D} electrode Poisson's ratio",'poisson'),
                (f'{D} electrode partial molar volume [m3.mol-1]','molar_volume_m3_mol')]:
                if float(parameters[parameter])!=material[field]:raise ValueError('Mechanical parameter changes require a rebuilt microstructure operator')
            parameters.update({f'{D} electrode porosity':op.porosity,f'{D} electrode active material volume fraction':op.active_fraction})
        return parameters


class GeometryActiveMaterial(Constant):
    def __init__(self,*args,operator,**kwargs):super().__init__(*args,**kwargs);self.op=operator
    def get_fundamental_variables(self):
        variables=super().get_fundamental_variables();d,D=self.domain_Domain
        area=pb.FullBroadcast(self.op.area_density,f'{d} electrode','current collector')
        variables[f'{D} electrode surface area to volume ratio [m-1]']=area
        variables[f'X-averaged {d} electrode surface area to volume ratio [m-1]']=pb.x_average(area)
        return variables


class GeometryTransport(TransportBase):
    def __init__(self,*args,stack,**kwargs):super().__init__(*args,**kwargs);self.stack=stack
    def get_coupled_variables(self,variables):
        tor={}
        for domain in self.options.whole_cell_domains:
            d=domain.split()[0]
            if d=='separator':value=variables['Separator porosity']**self.param.s.b_e if self.component=='Electrolyte' else 0
            else:value=self.stack.electrodes[d].transport[self.component.lower()]['efficiency']
            tor[domain]=value if isinstance(value,pb.Symbol) else pb.FullBroadcast(float(value),domain,'current collector')
        variables.update(self._get_standard_transport_efficiency_variables(tor));return variables


class MicrostructureMechanics(pb.BaseSubModel):
    def __init__(self,param,options,stack):super().__init__(param,options=options);self.stack=stack
    def get_coupled_variables(self,variables):
        T=variables['Volume-averaged cell temperature [K]'];theta=T-self.param.T_ref
        dc={d:[variables[key(d,k,'concentration [mol.m-3]')]-op.reference_concentration for k in range(op.dimension)] for d,op in self.stack.electrodes.items()}
        numerator=sum((self.stack.lengths[d]*pb.x_average(dot(op.H[0,1:-1],dc[d])+op.H[0,-1]*theta)/op.H[0,0]
            for d,op in self.stack.electrodes.items()),pb.Scalar(0))
        traction=numerator/self.stack.compliance if self.stack.constraint=='fixed-thickness' else 0*T
        energy=.5*self.stack.separator_length*traction**2/self.stack.separator_C
        entropy_force=pb.Scalar(0);thickness=self.stack.separator_length*traction/self.stack.separator_C
        for d,op in self.stack.electrodes.items():
            h=op.H;eps=(traction-dot(h[0,1:-1],dc[d])-h[0,-1]*theta)/h[0,0]
            forces=[h[k+1,0]*eps+dot(h[k+1,1:-1],dc[d])+h[k+1,-1]*theta for k in range(op.dimension)]
            force_T=h[-1,0]*eps+dot(h[-1,1:-1],dc[d])+h[-1,-1]*theta
            W=.5*(eps*traction+sum((c*f for c,f in zip(dc[d],forces)),pb.Scalar(0))+theta*force_T)
            for k,force in enumerate(forces):
                variables[key(d,k,'elastic chemical potential [J.mol-1]')]=force/op.weights[k]
                variables[key(d,k,'elastic chemical potential dT [J.mol-1.K-1]')]=pb.Scalar(self.stack.mu_T[d][k])
            variables[f'MORQEX {d} axial strain']=eps
            variables[f'MORQEX {d} elastic energy density [J.m-3]']=W
            variables[f'MORQEX {d} elastic entropy force [J.m-3.K-1]']=force_T
            energy+=self.stack.lengths[d]*pb.x_average(W);entropy_force+=self.stack.lengths[d]*pb.x_average(force_T)
            thickness+=self.stack.lengths[d]*pb.x_average(eps)
        variables.update({'MORQEX stack axial stress [Pa]':traction,'MORQEX sandwich thickness change [m]':thickness,
            'MORQEX elastic energy per area [J.m-2]':energy,'MORQEX elastic entropy force per area [J.m-2.K-1]':entropy_force})
        return variables


class MicrostructureParticle(BaseParticle):
    def __init__(self,*args,operator,feedback=1.,**kwargs):super().__init__(*args,**kwargs);self.op=operator;self.feedback=feedback
    def get_fundamental_variables(self):
        d,D=self.domain_Domain;op=self.op
        states=[pb.Variable(key(d,k,'concentration [mol.m-3]'),domain=f'{d} electrode',
            auxiliary_domains={'secondary':'current collector'},scale=self.phase_param.c_max,bounds=(0,self.phase_param.c_max)) for k in range(op.dimension)]
        mean=dot(op.weights,states)/op.active_fraction;surface=dot(op.reaction_area,states)/op.area_density
        # Compatibility aliases are explicitly means; physical microfields have
        # named region states and a geometric localization map, never radius r.
        variables=self._get_standard_concentration_variables(mean,c_s_xav=pb.x_average(mean),c_s_rav=mean,c_s_av=pb.x_average(mean),c_s_surf=surface)
        variables.update({key(d,k,'concentration [mol.m-3]'):c for k,c in enumerate(states)})
        return variables

    def get_coupled_variables(self,variables):
        d,D=self.domain_Domain;op=self.op;phase=self.phase_param;T=variables[f'{D} electrode temperature [K]']
        c=[variables[key(d,k,'concentration [mol.m-3]')] for k in range(op.dimension)]
        chemical=[-self.param.F*phase.U(v/phase.c_max,T) for v in c]
        mu=[chem+self.feedback*variables[key(d,k,'elastic chemical potential [J.mol-1]')] for k,chem in enumerate(chemical)]
        current=[variables[key(d,k,'reaction current [A.m-2]')] for k in range(op.dimension)]
        massdot=[-float(area)*j/self.param.F for area,j in zip(op.reaction_area,current)]
        diffusion_diss=pb.Scalar(0)
        upper=np.triu(op.stiffness.toarray(),1)
        for i,j in zip(*np.nonzero(upper)):
            g=-upper[i,j]
            if g<=0:raise RuntimeError('Coarse diffusion graph lost its sign structure')
            mid=(c[i]+c[j])/2;delta=c[i]-c[j];regularizer=(1e-7*phase.c_max)**2
            # Differentiate an actual state: expression simplification can
            # remove a composite `mid` subtree before parameter substitution.
            curvature=-2*self.param.F*phase.U(mid/phase.c_max,T).diff(c[i])
            secant=((chemical[i]-chemical[j])*delta+regularizer*curvature)/(delta**2+regularizer)
            mobility=phase.D(mid,T)/secant
            flux=g*mobility*(mu[i]-mu[j]);massdot[i]-=flux;massdot[j]+=flux
            diffusion_diss+=flux*(mu[i]-mu[j])
        rate=[m/float(w) for m,w in zip(massdot,op.weights)]
        reaction_diss=sum((float(a)*j*variables[key(d,k,'reaction overpotential [V]')] for k,(a,j) in enumerate(zip(op.reaction_area,current))),pb.Scalar(0))
        port=variables[f'{D} electrode surface potential difference [V]']*dot(op.reaction_area,current)
        energy_rate=sum((m*f for m,f in zip(massdot,mu)),pb.Scalar(0))
        reversible=T*self.feedback*sum((m*variables[key(d,k,'elastic chemical potential dT [J.mol-1.K-1]')] for k,m in enumerate(massdot)),pb.Scalar(0))
        for k,derivative in enumerate(rate):variables[key(d,k,'rate [mol.m-3.s-1]')]=derivative
        variables.update({f'MORQEX {d} diffusion dissipation [W.m-3]':diffusion_diss,
            f'MORQEX {d} reaction dissipation [W.m-3]':reaction_diss,
            f'MORQEX {d} interface port power [W.m-3]':port,
            f'MORQEX {d} reversible elastic heating [W.m-3]':reversible,
            f'MORQEX {d} port identity residual [W.m-3]':energy_rate+diffusion_diss+reaction_diss-port})
        mean=variables[f'R-averaged {d} particle concentration [mol.m-3]']
        variables.update(self._get_standard_diffusivity_variables(phase.D(mean,T)))
        variables.update(self._get_standard_flux_variables(0*mean))
        return variables

    def set_rhs(self,variables):
        d=self.domain
        for k in range(self.op.dimension):self.rhs[variables[key(d,k,'concentration [mol.m-3]')]]=variables[key(d,k,'rate [mol.m-3.s-1]')]

    def set_initial_conditions(self,variables):
        initial=pb.FullBroadcast(self.op.reference_concentration,f'{self.domain} electrode','current collector')
        for k in range(self.op.dimension):self.initial_conditions[variables[key(self.domain,k,'concentration [mol.m-3]')]]=initial

    def add_events_from(self,variables):
        for k in range(self.op.dimension):
            c=variables[key(self.domain,k,'concentration [mol.m-3]')]
            self.events.extend([pb.Event(f'MORQEX {self.domain} region {k} lower concentration',pb.min(c)-.001*self.phase_param.c_max),
                pb.Event(f'MORQEX {self.domain} region {k} upper concentration',.999*self.phase_param.c_max-pb.max(c))])


class MicrostructureOCP(BaseOpenCircuitPotential):
    def __init__(self,*args,operator,feedback=1.,**kwargs):super().__init__(*args,**kwargs);self.op=operator;self.feedback=feedback
    def get_coupled_variables(self,variables):
        d,D=self.domain_Domain;op=self.op;T=variables[f'{D} electrode temperature [K]']
        U=[];entropy=[]
        for k in range(op.dimension):
            c=variables[key(d,k,'concentration [mol.m-3]')]
            value=self.phase_param.U(c/self.phase_param.c_max,T)-self.feedback*variables[key(d,k,'elastic chemical potential [J.mol-1]')]/self.param.F
            U.append(value);entropy.append(-self.feedback*variables[key(d,k,'elastic chemical potential dT [J.mol-1.K-1]')]/self.param.F)
            variables[key(d,k,'open circuit potential [V]')]=value
        surface=dot(op.reaction_area,U)/op.area_density;bulk=pb.x_average(dot(op.weights,U)/op.active_fraction)
        variables.update(self._get_standard_ocp_variables(surface,bulk,dot(op.reaction_area,entropy)/op.area_density))
        variables[f'{D} electrode equilibrium open-circuit potential [V]']=surface
        variables[f'X-averaged {d} electrode equilibrium open-circuit potential [V]']=pb.x_average(surface)
        return variables


class MicrostructureKinetics(SymmetricButlerVolmer):
    def __init__(self,*args,operator,**kwargs):super().__init__(*args,**kwargs);self.op=operator
    def get_coupled_variables(self,variables):
        d,D=self.domain_Domain;op=self.op;T=variables[f'{D} electrode temperature [K]'];ce=variables[f'{D} electrolyte concentration [mol.m-3]']
        delta=variables[f'{D} electrode surface potential difference [V]'];currents=[];exchange=[];eta=[]
        for k,area in enumerate(op.reaction_area):
            c=variables[key(d,k,'concentration [mol.m-3]')];U=variables[key(d,k,'open circuit potential [V]')]
            j0=self.phase_param.j0(ce,c,T);overpotential=delta-U
            j=2*j0*pb.sinh(self.param.F*overpotential/(2*self.param.R*T)) if area>0 else 0*c
            currents.append(j);exchange.append(j0);eta.append(overpotential)
            variables[key(d,k,'reaction current [A.m-2]')]=j
            variables[key(d,k,'exchange current [A.m-2]')]=j0
            variables[key(d,k,'reaction overpotential [V]')]=overpotential
        averaged=dot(op.reaction_area,currents)/op.area_density
        jtot,ajtot=self._get_average_total_interfacial_current_density(variables)
        variables.update(self._get_standard_interfacial_current_variables(averaged))
        variables.update(self._get_standard_total_interfacial_current_variables(jtot,ajtot))
        variables.update(self._get_standard_exchange_current_variables(dot(op.reaction_area,exchange)/op.area_density))
        variables.update(self._get_standard_overpotential_variables(dot(op.reaction_area,eta)/op.area_density))
        variables.update(self._get_standard_volumetric_current_density_variables(variables))
        variables.update(self._get_standard_sei_film_overpotential_variables(pb.Scalar(0)))
        return variables


class PortThermal(Lumped):
    def __init__(self,*args,stack,feedback=1.,**kwargs):super().__init__(*args,**kwargs);self.stack=stack;self.feedback=feedback
    def get_fundamental_variables(self):
        variables=super().get_fundamental_variables()
        for quantity in ['electrical work [J]','cooling work [J]','entropy production [J.K-1]','entropy heat exchange [J.K-1]']:
            variables[f'MORQEX accumulated {quantity}']=pb.Variable(f'MORQEX accumulated {quantity}',scale=100.)
        return variables

    def get_coupled_variables(self,variables):
        variables=super().get_coupled_variables(variables)
        T=variables['Volume-averaged cell temperature [K]'];A=self.param.L_y*self.param.L_z*self.param.n_electrodes_parallel;V=self.param.V_cell
        diffusion=pb.Scalar(0);reaction=pb.Scalar(0);port=pb.Scalar(0);reversible=pb.Scalar(0)
        for d in self.stack.electrodes:
            scale=A*self.stack.lengths[d]
            diffusion+=scale*pb.x_average(variables[f'MORQEX {d} diffusion dissipation [W.m-3]'])
            reaction+=scale*pb.x_average(variables[f'MORQEX {d} reaction dissipation [W.m-3]'])
            port+=scale*pb.x_average(variables[f'MORQEX {d} interface port power [W.m-3]'])
            reversible+=scale*pb.x_average(variables[f'MORQEX {d} reversible elastic heating [W.m-3]'])
        incoming=-variables['Current [A]']*variables['Voltage [V]']
        macro=incoming-port;heating=macro+diffusion+reaction+reversible
        ce=variables['Electrolyte concentration [mol.m-3]'];eps=variables['Porosity']
        electrolyte_rhs=-pb.div(variables['Electrolyte flux [mol.m-2.s-1]'])+variables['Sum of electrolyte reaction source terms [A.m-3]']/self.param.F-ce*variables['Transverse volume-averaged acceleration [m.s-2]']
        mixing_power=A*self.param.L_x*pb.x_average(2*self.param.R*T*pb.log(ce/self.param.c_e_init)*electrolyte_rhs)
        electrolyte_entropy=-A*self.param.L_x*pb.x_average(eps*2*self.param.R*(ce*pb.log(ce/self.param.c_e_init)-ce+self.param.c_e_init))
        dissipation=macro+diffusion+reaction-mixing_power
        variables.update({'MORQEX electrical incoming power [W]':incoming,'MORQEX interface port power [W]':port,
            'MORQEX micro diffusion dissipation [W]':diffusion,'MORQEX micro reaction dissipation [W]':reaction,
            'MORQEX macro electric conversion heat [W]':macro,'MORQEX reversible elastic heat [W]':reversible,
            'MORQEX total dissipation [W]':dissipation,'MORQEX electrolyte mixing entropy [J.K-1]':electrolyte_entropy,
            'MORQEX native ohmic heating [W]':variables['Ohmic heating [W]'],
            'MORQEX native diagnostic total heating [W]':variables['Total heating [W]'],
            'MORQEX total heating [W]':heating})
        # Only the cell-integrated heating enters this 0D thermal model. Native
        # local heat profiles remain named diagnostics, not the port heat model.
        variables['Volume-averaged total heating [W.m-3]']=heating/V
        variables['Total heating [W]']=heating
        return variables

    def set_rhs(self,variables):
        T=variables['Volume-averaged cell temperature [K]'];Cp=variables['Volume-averaged effective heat capacity [J.K-1.m-3]']
        V=self.param.V_cell;A=self.param.L_y*self.param.L_z*self.param.n_electrodes_parallel
        coefficient=self.feedback*A*self.stack.energy_TT_per_area/V
        effective=Cp-T*coefficient;cooling=variables['Surface total cooling [W]']
        self.rhs[T]=(variables['MORQEX total heating [W]']+cooling)/(V*effective)
        for name,value in [('electrical work [J]',variables['MORQEX electrical incoming power [W]']),('cooling work [J]',cooling),
            ('entropy production [J.K-1]',variables['MORQEX total dissipation [W]']/T),('entropy heat exchange [J.K-1]',cooling/T)]:
            self.rhs[variables[f'MORQEX accumulated {name}']]=value
        variables['MORQEX effective heat capacity [J.m-3.K-1]']=effective
        variables['MORQEX thermal entropy [J.K-1]']=Cp*V*pb.log(T/self.param.T_init)
        variables['MORQEX elastic entropy [J.K-1]']=-self.feedback*A*variables['MORQEX elastic entropy force per area [J.m-2.K-1]']

    def set_initial_conditions(self,variables):
        super().set_initial_conditions(variables)
        for quantity in ['electrical work [J]','cooling work [J]','entropy production [J.K-1]','entropy heat exchange [J.K-1]']:
            self.initial_conditions[variables[f'MORQEX accumulated {quantity}']]=pb.Scalar(0)

    def add_events_from(self,variables):
        self.events.append(pb.Event('MORQEX nonpositive heat capacity',variables['MORQEX effective heat capacity [J.m-3.K-1]']-1.))


def build_microstructure_dfn(stack,*,feedback=1.,thermal=True):
    if not 0<=feedback<=1:raise ValueError('Feedback lies in [0,1]')
    model=pb.lithium_ion.DFN(options={'thermal':'lumped' if thermal else 'isothermal','heat of mixing':'false','stress-induced diffusion':'false'},build=False,
        name='MORQEX DFN with geometric microstructure states and energy-conjugate ports')
    for d,op in stack.electrodes.items():
        model.submodels[f'{d} primary active material']=GeometryActiveMaterial(model.param,d,model.options,operator=op)
        model.submodels[f'{d} primary particle']=MicrostructureParticle(model.param,d,model.options,operator=op,feedback=feedback)
        model.submodels[f'{d} primary open-circuit potential']=MicrostructureOCP(model.param,d,'lithium-ion main',model.options,operator=op,feedback=feedback)
        model.submodels[f'{d} primary interface']=MicrostructureKinetics(model.param,d,'lithium-ion main',model.options,operator=op)
    for component in ['Electrolyte','Electrode']:
        model.submodels[f'{component.lower()} transport efficiency']=GeometryTransport(model.param,component,model.options,stack=stack)
    model.submodels['MORQEX microstructure mechanics']=MicrostructureMechanics(model.param,model.options,stack)
    if thermal:model.submodels['thermal']=PortThermal(model.param,model.options,stack=stack,feedback=feedback)
    model.build_model()
    return model
