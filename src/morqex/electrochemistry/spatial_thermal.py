"""Through-thickness heat transport with microstructure thermodynamic ports.

At fixed total thickness the thermoelastic heat capacity is a diagonal operator
plus one global rank-one coupling. Eliminate the stress rate analytically; do
not replace this coupling by independent local heat capacities. Thermal geometry
is the repeated n/separator/p/collector stack, with native Fourier/Robin fluxes.
"""
import pybamm as pb
from dataclasses import dataclass,asdict
from pybamm.models.submodels.thermal.pouch_cell.x_full import OneDimensionalX
from .aging_dfn import age
from .microstructure_dfn import key


@dataclass(frozen=True)
class PouchThermalBoundary:
    negative_face_W_m2_K:float=5.
    positive_face_W_m2_K:float=5.
    edge_W_m2_K:float=0.
    negative_tab_width_m:float=0.
    positive_tab_width_m:float=0.
    negative_tab_W_m2_K:float=0.
    positive_tab_W_m2_K:float=0.
    def apply(self,parameters):
        import numpy as np
        if not np.isfinite(list(asdict(self).values())).all() or min(asdict(self).values())<0:raise ValueError('Finite nonnegative boundary coefficients/widths required')
        changes={'Edge heat transfer coefficient [W.m-2.K-1]':self.edge_W_m2_K}
        for d in ['negative','positive']:
            D=d.capitalize();changes.update({f'{D} tab width [m]':getattr(self,d+'_tab_width_m'),
                f'{D} tab heat transfer coefficient [W.m-2.K-1]':getattr(self,d+'_tab_W_m2_K'),
                f'{D} current collector surface heat transfer coefficient [W.m-2.K-1]':getattr(self,d+'_face_W_m2_K')})
        parameters.update(changes,check_already_exists=False)
        return parameters


class SpatialAgingThermal(OneDimensionalX):
    def __init__(self,*args,stack,feedback=1.,**kwargs):
        super().__init__(*args,**kwargs);self.stack=stack;self.feedback=feedback
    def get_fundamental_variables(self):
        v=super().get_fundamental_variables()
        for name in ['electrical work [J]','cooling work [J]','entropy production [J.K-1]','entropy heat exchange [J.K-1]']:
            v['MORQEX accumulated '+name]=pb.Variable('MORQEX accumulated '+name,scale=100.)
        return v
    def get_coupled_variables(self,v):
        v=super().get_coupled_variables(v);A=self.param.L_y*self.param.L_z*self.param.n_electrodes_parallel
        extra=[];reversible=[];port=[];diss=[]
        for domain in self.options.whole_cell_domains:
            d=domain.split()[0];T=v[f'{domain.capitalize()} temperature [K]']
            if d=='separator':extra.append(0*T);reversible.append(0*T);port.append(0*T);diss.append(0*T)
            else:
                irreversible=v[f'MORQEX {d} diffusion dissipation [W.m-3]']+v[f'MORQEX {d} reaction dissipation [W.m-3]']+v[age(d,'damage dissipation [W.m-3]')]
                rev=v[f'MORQEX {d} reversible elastic heating [W.m-3]']+v[age(d,'damage entropy heat [W.m-3]')]
                extra.append(irreversible+rev);reversible.append(rev);diss.append(irreversible);port.append(v[f'MORQEX {d} interface port power [W.m-3]'])
        from .power_flux import conservative_macro_heat
        ohmic=conservative_macro_heat(v,port);local_port=pb.concatenation(*port);irreversible=ohmic+pb.concatenation(*diss)
        Q=ohmic+pb.concatenation(*extra);v['Total heating [W.m-3]']=Q
        def integral(a):return A*self.param.L_x*pb.x_average(a)
        incoming=-v['Current [A]']*v['Voltage [V]'];v['MORQEX electrical incoming power [W]']=incoming
        v['MORQEX interface port power [W]']=integral(local_port)
        v['MORQEX distributed electric power closure residual [W]']=incoming-integral(local_port)-integral(ohmic)
        v['MORQEX total heating [W]']=integral(Q);v['Total heating [W]']=integral(Q)
        T=v['Cell temperature [K]'];ce=v['Electrolyte concentration [mol.m-3]'];eps=v['Porosity'];epsdot=v['Porosity change']
        rhs=-pb.div(v['Electrolyte flux [mol.m-2.s-1]'])+v['Sum of electrolyte reaction source terms [A.m-3]']/self.param.F-ce*v['Transverse volume-averaged acceleration [m.s-2]']
        mixing_rate=2*self.param.R*(pb.log(ce/self.param.c_e_init)*rhs+(self.param.c_e_init-ce)*epsdot)
        v['MORQEX distributed irreversible entropy source [W.K-1]']=integral(irreversible/T-mixing_rate)
        v['MORQEX distributed nonthermal dissipation [W]']=integral(irreversible-T*mixing_rate)
        v['MORQEX electrolyte mixing entropy [J.K-1]']=-integral(eps*2*self.param.R*(ce*pb.log(ce/self.param.c_e_init)-ce+self.param.c_e_init))
        v['MORQEX distributed heat over T [W.K-1]']=integral(Q/T)
        return v
    def set_rhs(self,v):
        super().set_rhs(v)
        A=self.param.L_y*self.param.L_z*self.param.n_electrodes_parallel
        T=v['Cell temperature [K]'];surface=v['Surface temperature [K]']
        if getattr(self.stack,'thermal_damage_tables',None):
            from ..nodewise.thermal_damage import conductivity_field
            before=[];after=[]
            for domain in self.options.whole_cell_domains:
                d=domain.split()[0];temp=v[domain.capitalize()+' temperature [K]'];lam=self.param.domain_params[d].lambda_(temp)
                before.append(lam*pb.ones_like(temp))
                after.append(lam*pb.ones_like(temp) if d=='separator' else conductivity_field(self.stack.thermal_damage_tables[d],v[age(d,'damage')]))
            previous=pb.concatenation(*before);updated=pb.concatenation(*after)
            capacities=pb.concatenation(*[self.param.domain_params[domain.split()[0]].rho_c_p(v[domain.capitalize()+' temperature [K]'])*pb.ones_like(v[domain.capitalize()+' temperature [K]']) for domain in self.options.whole_cell_domains])
            self.rhs[T]+=(pb.div(updated*pb.grad(T))-pb.div(previous*pb.grad(T)))/capacities
            for d,i,side,sign in [('negative',0,'left',1.),('positive',2,'right',-1.)]:
                temp=v[d.capitalize()+' electrode temperature [K]'];collector=v[d.capitalize()+' current collector temperature [K]'];p=self.param.domain_params[d]
                delta=pb.boundary_value(after[i]-before[i],side)*pb.boundary_gradient(temp,side)
                self.rhs[collector]+=sign*delta/(p.L_cc*p.rho_c_p_cc(collector))
                v[f'MORQEX {d} thermal conductivity [W.m-1.K-1]']=after[i]
        Cp=[];D=[];k=[];thermal=[];internal=pb.Scalar(0.);entropy=pb.Scalar(0.);native_power=pb.Scalar(0.);native_entropy_rate=pb.Scalar(0.)
        R=pb.Scalar(0.);C=self.stack.separator_length/self.stack.separator_C
        for domain in self.options.whole_cell_domains:
            d=domain.split()[0];temp=v[f'{domain.capitalize()} temperature [K]'];param=self.param.domain_params[d];cp=param.rho_c_p(temp)
            if d=='separator':modulus=1.;cross=0*temp;a=0*temp;length=self.stack.separator_length
            else:
                op=self.stack.electrodes[d];length=self.stack.lengths[d]
                modulus=v[age(d,'axial modulus [Pa]')];cross=v[age(d,'axial thermal coupling [Pa.K-1]')]/modulus
                a=v[age(d,'elastic TT [J.m-3.K-2]')]
                sigma_rate=sum((v[key(d,j,'axial concentration coupling [Pa.m3.mol-1]')]*v[key(d,j,'rate [mol.m-3.s-1]')] for j in range(op.dimension)),pb.Scalar(0.))
                sigma_rate+=v[age(d,'axial damage force [Pa]')]*v[age(d,'damage rate [s-1]')]
                R+=length*pb.x_average(sigma_rate/modulus);C+=length*pb.x_average(1/modulus)
            Cp.append(cp*pb.ones_like(temp));D.append(cp-self.feedback*temp*a);k.append(cross);thermal.append(temp)
            internal+=A*length*pb.x_average(cp*(temp-self.param.T_init))
            entropy+=A*length*pb.x_average(cp*pb.log(temp/self.param.T_init))
        cp=pb.concatenation(*Cp);effective=pb.concatenation(*D);cross=pb.concatenation(*k)
        native=self.rhs[T];uncoupled=native*cp/effective
        native_power=A*self.param.L_x*pb.x_average(native*cp)
        native_entropy_rate=A*self.param.L_x*pb.x_average(native*cp/T)
        for d in ['negative','positive']:
            domain=f'{d} current collector';temp=v[domain.capitalize()+' temperature [K]'];param=self.param.domain_params[d];length=param.L_cc;capacity=param.rho_c_p_cc(temp)
            native_power+=A*length*capacity*self.rhs[temp];native_entropy_rate+=A*length*capacity*self.rhs[temp]/temp
            internal+=A*length*capacity*(temp-self.param.T_init);entropy+=A*length*capacity*pb.log(temp/self.param.T_init)
        cooling=native_power-v['MORQEX total heating [W]']
        numerator=R+self.param.L_x*pb.x_average(cross*uncoupled)
        denominator=C-self.param.L_x*pb.x_average(self.feedback*T*cross**2/effective)
        stress_rate=numerator/denominator if self.stack.constraint=='fixed-thickness' else pb.Scalar(0.)
        self.rhs[T]=uncoupled+self.feedback*T*cross*stress_rate/effective
        conduction=native_entropy_rate-v['MORQEX distributed heat over T [W.K-1]']-cooling/surface
        entropy_production=v['MORQEX distributed irreversible entropy source [W.K-1]']+conduction
        v.update({'MORQEX thermal internal energy [J]':internal,'MORQEX thermal entropy [J.K-1]':entropy,
            'MORQEX elastic entropy [J.K-1]':-self.feedback*A*v['MORQEX elastic entropy force per area [J.m-2.K-1]'],
            'MORQEX distributed conduction entropy production [W.K-1]':conduction,
            'MORQEX total dissipation [W]':v['MORQEX distributed nonthermal dissipation [W]'],
            'MORQEX effective heat capacity [J.m-3.K-1]':pb.min(effective),
            'MORQEX distributed heat-capacity Schur ratio':denominator/C,
            'MORQEX thermal boundary cooling [W]':cooling})
        for name,value in [('electrical work [J]',v['MORQEX electrical incoming power [W]']),('cooling work [J]',cooling),
            ('entropy production [J.K-1]',entropy_production),('entropy heat exchange [J.K-1]',cooling/surface)]:self.rhs[v['MORQEX accumulated '+name]]=value
    def set_initial_conditions(self,v):
        super().set_initial_conditions(v)
        for name in ['electrical work [J]','cooling work [J]','entropy production [J.K-1]','entropy heat exchange [J.K-1]']:self.initial_conditions[v['MORQEX accumulated '+name]]=pb.Scalar(0.)
    def add_events_from(self,v):
        self.events.extend([pb.Event('MORQEX nonpositive spatial heat capacity',v['MORQEX effective heat capacity [J.m-3.K-1]']-1.),
            pb.Event('MORQEX nonpositive thermoelastic Schur complement',v['MORQEX distributed heat-capacity Schur ratio']-1e-5)])
