"""Extensible DFN with explicit side reservoirs and evolving cohesive mechanics.

Uniform scalar cohesive damage per SVE changes actual interface spring stiffness
and accessible reaction area. Bulk host inventory remains present; this is contact
loss, not an untracked deletion of active material or resolved crack propagation.
"""
from dataclasses import dataclass,replace
import numpy as np
import pybamm as pb
from pybamm.models.submodels.interface.total_interfacial_current import TotalInterfacialCurrent
from pybamm.models.submodels.porosity.base_porosity import BaseModel as BasePorosity
from pybamm.models.submodels.thermal.lumped import Lumped
from .microstructure_dfn import (MicrostructureParticle,MicrostructureOCP,MicrostructureKinetics,GeometryActiveMaterial,
    GeometryTransport,MicrostructureStack,PortThermal,key,dot)
from ..degradation import ReactionContext
from .linear_port import linear_combination as dot

def age(d,name,k=None):return f'MORQEX aging {d} '+(f'region {k} ' if k is not None else '')+name

@dataclass
class EnergyKernel:
    operator:object
    cohesive:object=None
    fixed_damage:float|None=None
    def parts(self):
        if self.fixed_damage is not None:
            return self.hessian(self.fixed_damage),np.zeros((0,self.operator.dimension+2)),np.empty(0),.001
        if self.cohesive is None:return self.operator.H,np.zeros((0,self.operator.dimension+2)),np.empty(0),.001
        c=self.cohesive;return c.direct,c.factors,c.eigenvalues,c.residual
    def hessian(self,d):return self.operator.H if self.cohesive is None else self.cohesive.hessian(d)

class AgingStates(pb.BaseSubModel):
    def __init__(self,param,options,stack,config):super().__init__(param,options=options);self.stack=stack;self.config=config
    def get_fundamental_variables(self):
        v={}
        for d,op in self.stack.electrodes.items():
            v[age(d,'damage')]=pb.Variable(age(d,'damage'),domain=f'{d} electrode',auxiliary_domains={'secondary':'current collector'},bounds=(0,1))
            for k in range(op.dimension):
                v[age(d,'film current [A.m-2]',k)]=pb.Variable(age(d,'film current [A.m-2]',k),domain=f'{d} electrode',auxiliary_domains={'secondary':'current collector'},scale=1.)
                for law in getattr(self.config,d):
                    for r in law.reservoirs():v[age(d,r.name+' [mol.m-3]',k)]=pb.Variable(age(d,r.name+' [mol.m-3]',k),domain=f'{d} electrode',auxiliary_domains={'secondary':'current collector'},scale=r.scale,bounds=(0,np.inf))
        return v
    def set_rhs(self,v):
        for d,op in self.stack.electrodes.items():
            self.rhs[v[age(d,'damage')]]=v[age(d,'damage rate [s-1]')]
            for k in range(op.dimension):
                for law in getattr(self.config,d):
                    for r in law.reservoirs():self.rhs[v[age(d,r.name+' [mol.m-3]',k)]]=v[age(d,r.name+' rate [mol.m-3.s-1]',k)]
    def set_algebraic(self,v):
        for d,op in self.stack.electrodes.items():
            for k in range(op.dimension):self.algebraic[v[age(d,'film current [A.m-2]',k)]]=v[age(d,'film current residual [A.m-2]',k)]
    def set_initial_conditions(self,v):
        for d,op in self.stack.electrodes.items():
            self.initial_conditions[v[age(d,'damage')]]=pb.Scalar(self.config.initial_damage)
            for k in range(op.dimension):
                self.initial_conditions[v[age(d,'film current [A.m-2]',k)]]=pb.Scalar(0.)
                for law in getattr(self.config,d):
                    for r in law.reservoirs():self.initial_conditions[v[age(d,r.name+' [mol.m-3]',k)]]=pb.Scalar(r.initial*float(op.reaction_area[k])/op.area_density)
    def add_events_from(self,v):
        for d in self.stack.electrodes:self.events.append(pb.Event(age(d,'damage validity limit'),.95-pb.max(v[age(d,'damage')])))
        self.events.append(pb.Event('MORQEX aging electrolyte validity floor',pb.min(v['Electrolyte concentration [mol.m-3]'])-self.config.minimum_electrolyte_mol_m3))

class AgingMechanics(pb.BaseSubModel):
    modal_variables=False
    def __init__(self,param,options,stack,kernels,config,feedback):super().__init__(param,options=options);self.stack=stack;self.kernels=kernels;self.config=config;self.feedback=feedback
    def get_fundamental_variables(self):
        variables={};self._equations={}
        for d,op in self.stack.electrodes.items():
            names={f'MORQEX {d} axial strain':.001,f'MORQEX {d} elastic entropy force [J.m-3.K-1]':1000.,
                age(d,'energy release [J.m-3]'):1e5,age(d,'elastic TT [J.m-3.K-2]'):100.,
                age(d,'elastic damage-T [J.m-3.K-1]'):1000.,age(d,'strain-T [K-1]'):1e-5}
            for k in range(op.dimension):
                names[key(d,k,'elastic chemical potential [J.mol-1]')]=1000.
                names[key(d,k,'elastic chemical potential dT [J.mol-1.K-1]')]=10.
            for k in range(len(self.kernels[d].parts()[1]) if self.modal_variables else 0):
                names[age(d,'cohesive amplitude',k)]=100.
                names[age(d,'cohesive amplitude dT',k)]=1.
            for name,scale in names.items():variables[name]=pb.Variable(name,domain=f'{d} electrode',auxiliary_domains={'secondary':'current collector'},scale=scale)
        self._owned=variables;return variables
    def _equation(self,v,name,target):
        variable=v[name];self._equations[variable]=(variable-target)/variable.scale
    def set_algebraic(self,v):self.algebraic.update(self._equations)
    def set_initial_conditions(self,v):
        for variable in self._owned.values():self.initial_conditions[variable]=pb.Scalar(0.)
    def get_coupled_variables(self,v):
        T=v['Volume-averaged cell temperature [K]'];theta=T-self.param.T_ref;energy=0.;entropy=0.;thickness=0.;TT=0.
        for d,op in self.stack.electrodes.items():
            D,F,rho,residual=self.kernels[d].parts();damage=v[age(d,'damage')]
            retention=residual+(1-residual)*(1-damage)**2;derivative=-2*(1-residual)*(1-damage)
            den=[1+(retention-1)*float(r) for r in rho];dp=[derivative*float(r) for r in rho]
            dc=[v[key(d,k,'concentration [mol.m-3]')]-op.reference_concentration for k in range(op.dimension)]
            b=[dot(row[1:-1],dc)+float(row[-1])*theta for row in F]
            h00=float(D[0,0])-sum((float(row[0]**2)/z for row,z in zip(F,den)),pb.Scalar(0))
            h0T=float(D[0,-1])-sum((float(row[0]*row[-1])/z for row,z in zip(F,den)),pb.Scalar(0))
            hTT=float(D[-1,-1])-sum((float(row[-1]**2)/z for row,z in zip(F,den)),pb.Scalar(0))
            eps=v[f'MORQEX {d} axial strain'];epsT=v[age(d,'strain-T [K-1]')]
            q=[eps,*dc,theta];s=[float(row[0])*eps+bb for row,bb in zip(F,b)]
            amplitudes=[];thermal_amplitudes=[]
            if self.modal_variables:
                for k,(f,ss,z) in enumerate(zip(F,s,den)):
                    a=v[age(d,'cohesive amplitude',k)];at=v[age(d,'cohesive amplitude dT',k)]
                    self._equation(v,age(d,'cohesive amplitude',k),ss/z)
                    self._equation(v,age(d,'cohesive amplitude dT',k),(float(f[-1])+float(f[0])*epsT)/z)
                    amplitudes.append(a);thermal_amplitudes.append(at)
                self._equation(v,f'MORQEX {d} axial strain',(-dot(D[0,1:-1],dc)-float(D[0,-1])*theta+dot(F[:,0],amplitudes))/float(D[0,0]))
                self._equation(v,age(d,'strain-T [K-1]'),(-float(D[0,-1])+dot(F[:,0],thermal_amplitudes))/float(D[0,0]))
            else:
                amplitudes=[ss/z for ss,z in zip(s,den)]
                thermal_amplitudes=[(float(f[-1])+float(f[0])*epsT)/z for f,z in zip(F,den)]
                self._equation(v,f'MORQEX {d} axial strain',(-dot(D[0,1:-1],dc)-float(D[0,-1])*theta+sum((float(f[0])*bb/z for f,bb,z in zip(F,b,den)),pb.Scalar(0)))/h00)
                self._equation(v,age(d,'strain-T [K-1]'),-h0T/h00)
            forces=[dot(row,q)-dot(F[:,i],amplitudes) for i,row in enumerate(D)]
            W=.5*(sum((qq*dot(row,q) for qq,row in zip(q,D)),pb.Scalar(0))-sum((ss*a for ss,a in zip(s,amplitudes)),pb.Scalar(0)))
            self._equation(v,age(d,'energy release [J.m-3]'),-.5*sum((a**2*p for a,p in zip(amplitudes,dp)),pb.Scalar(0)))
            self._equation(v,age(d,'elastic damage-T [J.m-3.K-1]'),sum((a*at*p for a,at,p in zip(amplitudes,thermal_amplitudes,dp)),pb.Scalar(0)))
            self._equation(v,age(d,'elastic TT [J.m-3.K-2]'),float(D[-1,-1])+float(D[-1,0])*epsT-dot(F[:,-1],thermal_amplitudes))
            self._equation(v,f'MORQEX {d} elastic entropy force [J.m-3.K-1]',forces[-1])
            release=v[age(d,'energy release [J.m-3]')];wdT=v[age(d,'elastic damage-T [J.m-3.K-1]')]
            rate=self.config.damage.evaluate(damage,self.feedback*release)
            for k in range(op.dimension):
                self._equation(v,key(d,k,'elastic chemical potential [J.mol-1]'),forces[k+1]/op.weights[k])
                self._equation(v,key(d,k,'elastic chemical potential dT [J.mol-1.K-1]'),(float(D[k+1,-1])+float(D[k+1,0])*epsT-
                    dot(F[:,k+1],thermal_amplitudes))/op.weights[k])
            v.update({f'MORQEX {d} elastic energy density [J.m-3]':W,age(d,'damage rate [s-1]'):rate,
                age(d,'damage dissipation [W.m-3]'):self.feedback*release*rate,
                age(d,'damage entropy heat [W.m-3]'):self.feedback*T*wdT*rate})
            length=self.stack.lengths[d];energy+=length*pb.x_average(W);entropy+=length*pb.x_average(v[f'MORQEX {d} elastic entropy force [J.m-3.K-1]']);thickness+=length*pb.x_average(eps);TT+=length*pb.x_average(v[age(d,'elastic TT [J.m-3.K-2]')])
        v.update({'MORQEX elastic energy per area [J.m-2]':energy,'MORQEX elastic entropy force per area [J.m-2.K-1]':entropy,
            'MORQEX sandwich thickness change [m]':thickness,'MORQEX stack axial stress [Pa]':0*T,'MORQEX aging elastic TT per area':TT})
        return v

class AgingNetwork(MicrostructureKinetics):
    def __init__(self,*args,config,**kwargs):super().__init__(*args,**kwargs);self.config=config
    def get_coupled_variables(self,v):
        d,D=self.domain_Domain;op=self.op;T=v[f'{D} electrode temperature [K]'];ce=v[f'{D} electrolyte concentration [mol.m-3]'];delta=v[f'{D} electrode surface potential difference [V]']
        h=1-v[age(d,'damage')];laws=getattr(self.config,d);main=[];exchange=[];eta=[]
        side_total,side_D,film_D,reservoir_power,lithium_rate=[pb.Scalar(0) for _ in range(5)]
        for k,A in enumerate(op.reaction_area):
            stock={r.name:v[age(d,r.name+' [mol.m-3]',k)] for law in laws for r in law.reservoirs()}
            film=sum((law.resistance(stock,float(A)) for law in laws),pb.Scalar(0))
            for law in laws:
                if hasattr(law,'thickness'):stock['sei_thickness_m']=law.thickness(stock,float(A))
            z=v[age(d,'film current [A.m-2]',k)];c=v[key(d,k,'concentration [mol.m-3]')];U=v[key(d,k,'open circuit potential [V]')]
            chemistry=getattr(op,'regional_chemistry',None)
            e=delta-film*z-U;j0=self.phase_param.j0(ce,c,T)*(1. if chemistry is None else float(chemistry.kinetic_factor[k]));j=2*j0*pb.sinh(self.param.F*e/(2*self.param.R*T)) if A>0 else 0*c
            v[key(d,k,'reaction current [A.m-2]')]=j;v[key(d,k,'exchange current [A.m-2]')]=j0;v[key(d,k,'reaction overpotential [V]')]=e
            main.append(h*j);exchange.append(h*j0);eta.append(e);j_side=0*c;rate_sum=0*c
            context=ReactionContext(stock,delta,T,ce,float(A),float(A)*h,self.param.F,self.param.R,z,film)
            for law in laws:
                contribution=law.evaluate(context);factor=1. if chemistry is None else float(chemistry.side_factor[k])
                jlaw=contribution.current*factor if A>0 else 0*c;j_side+=jlaw
                for r in law.reservoirs():
                    rate=contribution.rates[r.name]*factor if A>0 else 0*c;v[age(d,r.name+' rate [mol.m-3.s-1]',k)]=rate
                    reservoir_power+=float(r.chemical_potential_J_mol)*rate;rate_sum+=rate
                side_D+=float(A)*h*jlaw*(delta-film*z-float(contribution.equilibrium_V))
            v[age(d,'film current residual [A.m-2]',k)]=z-j-j_side
            v[age(d,'side current [A.m-2]',k)]=j_side
            v[age(d,'reservoir balance residual [mol.m-3.s-1]',k)]=rate_sum+float(A)*h*j_side/self.param.F
            side_total+=float(A)*h*j_side;film_D+=float(A)*h*film*z**2;lithium_rate+=rate_sum
        averaged=dot(op.reaction_area,main)/op.area_density;jtot,ajtot=self._get_average_total_interfacial_current_density(v)
        v.update(self._get_standard_interfacial_current_variables(averaged));v.update(self._get_standard_total_interfacial_current_variables(jtot,ajtot))
        v.update(self._get_standard_exchange_current_variables(dot(op.reaction_area,exchange)/op.area_density));v.update(self._get_standard_overpotential_variables(dot(op.reaction_area,eta)/op.area_density))
        v.update(self._get_standard_volumetric_current_density_variables(v));v.update(self._get_standard_sei_film_overpotential_variables(0*delta))
        v.update({age(d,'side current [A.m-3]'):side_total,age(d,'side dissipation [W.m-3]'):side_D,
            age(d,'film dissipation [W.m-3]'):film_D,age(d,'reservoir power [W.m-3]'):reservoir_power,
            age(d,'reservoir rate [mol.m-3.s-1]'):lithium_rate})
        return v

class AgingParticle(MicrostructureParticle):
    def get_coupled_variables(self,v):
        d,D=self.domain_Domain;op=self.op;phase=self.phase_param;T=v[f'{D} electrode temperature [K]'];h=1-v[age(d,'damage')]
        chemistry=getattr(op,'regional_chemistry',None)
        c=[v[key(d,k,'concentration [mol.m-3]')] for k in range(op.dimension)]
        base_chemical=[-self.param.F*phase.U(a/phase.c_max,T) for a in c]
        chemical=[base if chemistry is None else base-self.param.F*chemistry.ocp_shift(k,a/phase.c_max) for k,(a,base) in enumerate(zip(c,base_chemical))]
        mu=[f+self.feedback*v[key(d,k,'elastic chemical potential [J.mol-1]')] for k,f in enumerate(chemical)]
        j=[v[key(d,k,'reaction current [A.m-2]')] for k in range(op.dimension)];massdot=[-float(A)*h*jj/self.param.F for A,jj in zip(op.reaction_area,j)];diffusion=pb.Scalar(0)
        i,jj=np.nonzero(np.triu(op.stiffness.toarray(),1))
        for a,b in zip(i,jj):
            g=-float(op.stiffness[a,b]);mid=(c[a]+c[b])/2;dc=c[a]-c[b];reg=(1e-7*phase.c_max)**2
            curvature=-2*self.param.F*phase.U(mid/phase.c_max,T).diff(c[a])
            # Use the SAME host secant at all compositions: offsets belong in
            # the driving potential, not in the mobility denominator. This
            # also guarantees continuity as a nonuniform dopant field -> 0.
            secant=((base_chemical[a]-base_chemical[b])*dc+reg*curvature)/(dc**2+reg)
            if chemistry is not None:secant-=self.param.F*float((chemistry.slope_V[a]+chemistry.slope_V[b])/2)/phase.c_max
            if chemistry is None:factor=1.
            else:
                da,db=chemistry.diffusion_factor[a],chemistry.diffusion_factor[b];factor=float(2*da*db/(da+db))
            # CBD contact loss reduces accessible reaction area and electronic
            # transport. It does not change intrinsic diffusion inside an
            # intact active host; changes of D require a constitutive law.
            flux=g*factor*phase.D(mid,T)/secant*(mu[a]-mu[b]);massdot[a]-=flux;massdot[b]+=flux;diffusion+=flux*(mu[a]-mu[b])
        reaction=sum((float(A)*h*current*v[key(d,k,'reaction overpotential [V]')] for k,(A,current) in enumerate(zip(op.reaction_area,j))),pb.Scalar(0))
        reaction+=v[age(d,'side dissipation [W.m-3]')]+v[age(d,'film dissipation [W.m-3]')]
        main_current=sum((float(A)*h*current for A,current in zip(op.reaction_area,j)),pb.Scalar(0));total=main_current+v[age(d,'side current [A.m-3]')]
        port=v[f'{D} electrode surface potential difference [V]']*total
        power=sum((rate*m for rate,m in zip(massdot,mu)),pb.Scalar(0))+v[age(d,'reservoir power [W.m-3]')]
        reversible=T*self.feedback*sum((rate*v[key(d,k,'elastic chemical potential dT [J.mol-1.K-1]')] for k,rate in enumerate(massdot)),pb.Scalar(0))
        for k,rate in enumerate(massdot):v[key(d,k,'rate [mol.m-3.s-1]')]=rate/float(op.weights[k])
        v.update({f'MORQEX {d} diffusion dissipation [W.m-3]':diffusion,f'MORQEX {d} reaction dissipation [W.m-3]':reaction,
            f'MORQEX {d} interface port power [W.m-3]':port,f'MORQEX {d} reversible elastic heating [W.m-3]':reversible,
            f'MORQEX {d} port identity residual [W.m-3]':power+diffusion+reaction-port})
        mean=v[f'R-averaged {d} particle concentration [mol.m-3]'];v.update(self._get_standard_diffusivity_variables(phase.D(mean,T)));v.update(self._get_standard_flux_variables(0*mean));return v

class AgingTotalCurrent(TotalInterfacialCurrent):
    def get_coupled_variables(self,v):
        v=super().get_coupled_variables(v)
        for d in ['negative','positive']:
            side=v[age(d,'side current [A.m-3]')]
            for quantity in ['volumetric interfacial current densities','electrolyte reaction source terms']:
                v[f'Sum of {d} electrode {quantity} [A.m-3]']+=side
                v[f'Sum of x-averaged {d} electrode {quantity} [A.m-3]']+=pb.x_average(side)
        v.update(self._get_whole_cell_coupled_variables(v));return v

class AgingPorosity(BasePorosity):
    def __init__(self,param,options,stack,config):super().__init__(param,options);self.stack=stack;self.config=config
    def get_coupled_variables(self,v):
        eps={}
        for domain in self.options.whole_cell_domains:
            d=domain.split()[0];value=self.param.domain_params[d].epsilon_init
            if d in self.stack.electrodes and self.config.porosity_change:
                for k in range(self.stack.electrodes[d].dimension):
                    for law in getattr(self.config,d):
                        for r in law.reservoirs():
                            op=self.stack.electrodes[d]
                            value-=float(r.volume_m3_mol)*(v[age(d,r.name+' [mol.m-3]',k)]-r.initial*float(op.reaction_area[k])/op.area_density)
            eps[domain]=value if isinstance(value,pb.Symbol) and value.domain else pb.FullBroadcast(value,domain,'current collector')
        v.update(self._get_standard_porosity_variables(eps));return v
    def add_events_from(self,v):self.events.append(pb.Event('MORQEX aging minimum pore fraction',pb.min(v['Porosity'])-.01))

class AgingPorosityRates(AgingPorosity):
    def get_coupled_variables(self,v):
        rates={}
        for domain in self.options.whole_cell_domains:
            d=domain.split()[0];rate=0.
            if d in self.stack.electrodes and self.config.porosity_change:
                for k in range(self.stack.electrodes[d].dimension):
                    for law in getattr(self.config,d):
                        for r in law.reservoirs():rate-=float(r.volume_m3_mol)*v[age(d,r.name+' rate [mol.m-3.s-1]',k)]
            rates[domain]=rate if isinstance(rate,pb.Symbol) and rate.domain else pb.FullBroadcast(rate,domain,'current collector')
        v.update(self._get_standard_porosity_change_variables(rates));return v
    def add_events_from(self,v):pass

class AgingTransport(GeometryTransport):
    def get_coupled_variables(self,v):
        tor={}
        for domain in self.options.whole_cell_domains:
            d=domain.split()[0]
            if d=='separator':value=v['Separator porosity']**self.param.s.b_e if self.component=='Electrolyte' else 0.
            else:
                op=self.stack.electrodes[d];value=op.transport[self.component.lower()]['efficiency']
                value=value*(v[f'{d.capitalize()} electrode porosity']/op.porosity)**1.5 if self.component=='Electrolyte' else value*(1-v[age(d,'damage')])
            tor[domain]=value if isinstance(value,pb.Symbol) and value.domain else pb.FullBroadcast(value,domain,'current collector')
        v.update(self._get_standard_transport_efficiency_variables(tor));return v

class AgingThermal(PortThermal):
    def get_coupled_variables(self,v):
        v=Lumped.get_coupled_variables(self,v);T=v['Volume-averaged cell temperature [K]'];A=self.param.L_y*self.param.L_z*self.param.n_electrodes_parallel;V=self.param.V_cell
        diffusion=pb.Scalar(0);reaction=pb.Scalar(0);port=pb.Scalar(0);reversible=pb.Scalar(0);damage=pb.Scalar(0)
        for d in self.stack.electrodes:
            scale=A*self.stack.lengths[d]
            diffusion+=scale*pb.x_average(v[f'MORQEX {d} diffusion dissipation [W.m-3]']);reaction+=scale*pb.x_average(v[f'MORQEX {d} reaction dissipation [W.m-3]'])
            port+=scale*pb.x_average(v[f'MORQEX {d} interface port power [W.m-3]']);reversible+=scale*pb.x_average(v[f'MORQEX {d} reversible elastic heating [W.m-3]']+v[age(d,'damage entropy heat [W.m-3]')])
            damage+=scale*pb.x_average(v[age(d,'damage dissipation [W.m-3]')])
        incoming=-v['Current [A]']*v['Voltage [V]'];macro=incoming-port;heating=macro+diffusion+reaction+damage+reversible
        ce=v['Electrolyte concentration [mol.m-3]'];eps=v['Porosity'];epsdot=v['Porosity change']
        rhs=-pb.div(v['Electrolyte flux [mol.m-2.s-1]'])+v['Sum of electrolyte reaction source terms [A.m-3]']/self.param.F-ce*v['Transverse volume-averaged acceleration [m.s-2]']
        mixing=A*self.param.L_x*pb.x_average(2*self.param.R*T*(pb.log(ce/self.param.c_e_init)*rhs+(self.param.c_e_init-ce)*epsdot))
        entropy=-A*self.param.L_x*pb.x_average(eps*2*self.param.R*(ce*pb.log(ce/self.param.c_e_init)-ce+self.param.c_e_init))
        v.update({'MORQEX electrical incoming power [W]':incoming,'MORQEX interface port power [W]':port,'MORQEX micro diffusion dissipation [W]':diffusion,
            'MORQEX micro reaction dissipation [W]':reaction,'MORQEX macro electric conversion heat [W]':macro,'MORQEX total dissipation [W]':macro+diffusion+reaction+damage-mixing,
            'MORQEX electrolyte mixing entropy [J.K-1]':entropy,'MORQEX total heating [W]':heating,'MORQEX aging damage dissipation [W]':damage,
            'MORQEX native ohmic heating [W]':v['Ohmic heating [W]'],'MORQEX native diagnostic total heating [W]':v['Total heating [W]']})
        v['Volume-averaged total heating [W.m-3]']=heating/V;v['Total heating [W]']=heating;return v
    def set_rhs(self,v):
        # Parent uses a fixed TT coefficient: supply the actual evolving one.
        class DynamicStack:
            energy_TT_per_area=v['MORQEX aging elastic TT per area']
        original=self.stack;self.stack=DynamicStack()
        try:super().set_rhs(v)
        finally:self.stack=original

def build_aging_dfn(stack,kernels,config,*,feedback=1.,factories=None,control=None):
    if not np.isfinite(feedback) or not 0<=feedback<=1:raise ValueError('Feedback must lie in [0,1]')
    if stack.constraint!='pressure':raise ValueError('Aging cohesive closure currently requires zero axial pressure; no unverified fixed-thickness extension')
    if set(kernels)!=set(stack.electrodes):raise ValueError('One energy kernel per electrode required')
    for d,op in stack.electrodes.items():
        if kernels[d].operator.metadata()['fingerprint']!=op.metadata()['fingerprint']:raise ValueError('Energy kernel/geometry/concentration basis mismatch')
    if config.damage.name!='none' and any(k.cohesive is None for k in kernels.values()):raise ValueError('Debonding needs actual cohesive operators, not fixed intact stress multipliers')
    roles={'states':AgingStates,'mechanics':AgingMechanics,'kinetics':AgingNetwork,'particle':AgingParticle,'ocp':MicrostructureOCP,'porosity':AgingPorosity,'transport':AgingTransport,'thermal':AgingThermal}
    if any(hasattr(op,'regional_chemistry') for op in stack.electrodes.values()):
        from ..doping import DopedOCP
        roles['ocp']=DopedOCP
    if factories:
        if not set(factories)<=roles.keys():raise ValueError('Unknown submodel role')
        roles.update(factories)
    native_options={'thermal':'lumped','heat of mixing':'false','stress-induced diffusion':'false'}
    if control is not None:native_options['operating mode']=control
    model=pb.lithium_ion.DFN(options=native_options,build=False,name='MORQEX extensible cohesive degradation DFN')
    for d,op in stack.electrodes.items():
        model.submodels[f'{d} primary active material']=GeometryActiveMaterial(model.param,d,model.options,operator=op)
        model.submodels[f'{d} primary particle']=roles['particle'](model.param,d,model.options,operator=op,feedback=feedback)
        model.submodels[f'{d} primary open-circuit potential']=roles['ocp'](model.param,d,'lithium-ion main',model.options,operator=op,feedback=feedback)
        model.submodels[f'{d} primary interface']=roles['kinetics'](model.param,d,'lithium-ion main',model.options,operator=op,config=config)
    model.submodels['MORQEX aging states']=roles['states'](model.param,model.options,stack,config)
    from ..degradation import NoDamage
    mechanical_kernels={d:replace(k,fixed_damage=config.initial_damage) if isinstance(config.damage,NoDamage) else k for d,k in kernels.items()}
    model.submodels['MORQEX aging mechanics']=roles['mechanics'](model.param,model.options,stack,mechanical_kernels,config,feedback)
    model.submodels['porosity']=roles['porosity'](model.param,model.options,stack,config)
    model.submodels['MORQEX aging porosity rates']=AgingPorosityRates(model.param,model.options,stack,config)
    for component in ['Electrolyte','Electrode']:model.submodels[f'{component.lower()} transport efficiency']=roles['transport'](model.param,component,model.options,stack=stack)
    model.submodels['total interface']=AgingTotalCurrent(model.param,'lithium-ion',model.options)
    model.submodels['thermal']=roles['thermal'](model.param,model.options,stack=stack,feedback=feedback)
    model.build_model();return model
