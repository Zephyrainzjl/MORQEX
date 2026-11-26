"""Mass-conservative Onsager/gradient-energy particles for general OCP tables.

Nonconvex OCPs are retained. Positive mobility and a declared gradient coefficient
replace the old division by chemical curvature. Nodal/lumped quadrature is an
explicit approximation; it is not a claim that eight nodes resolve every front.
"""
from dataclasses import dataclass
import numpy as np
import pybamm as pb
from .aging_dfn import AgingParticle,age
from .microstructure_dfn import MicrostructureOCP,key
from .linear_port import linear_combination as dot,coefficient

@dataclass(frozen=True)
class TransportLaw:
    mobility:str='lattice'
    gradient_coefficient_J_m5_mol2:float=0.
    mobility_multiplier:float=1.
    stoichiometry_bounds:tuple=(.001,.999)
    def __post_init__(self):
        if self.mobility not in {'lattice','constant'} or self.gradient_coefficient_J_m5_mol2<0 or self.mobility_multiplier<=0 or not np.isfinite([self.gradient_coefficient_J_m5_mol2,self.mobility_multiplier]).all():raise ValueError('Positive mobility and nonnegative gradient coefficient required')
        if not 0<self.stoichiometry_bounds[0]<self.stoichiometry_bounds[1]<1:raise ValueError('Explicit physical/table concentration bounds required')

def law_for(op):
    value=op.metadata().get('transport_law')
    return TransportLaw(**value) if value else TransportLaw()

def gradient_potential(op,c):
    law=law_for(op);matrix=op.stiffness.toarray();kappa=law.gradient_coefficient_J_m5_mol2
    return [kappa*dot(row,c)/coefficient(w) for row,w in zip(matrix,getattr(op,'division_weights',op.weights))]

class OpenOCP(MicrostructureOCP):
    def get_coupled_variables(self,v):
        d,D=self.domain_Domain;op=self.op;T=v[f'{D} electrode temperature [K]'];phase=self.phase_param
        c=[v[key(d,k,'concentration [mol.m-3]')] for k in range(op.dimension)];grad=gradient_potential(op,c);values=[];entropy=[]
        for k,concentration in enumerate(c):
            mu=v[key(d,k,'elastic chemical potential [J.mol-1]')]
            value=phase.U(concentration/phase.c_max,T)-(self.feedback*mu+grad[k])/self.param.F
            ut=phase.dUdT(concentration/phase.c_max)-self.feedback*v[key(d,k,'elastic chemical potential dT [J.mol-1.K-1]')]/self.param.F
            v[key(d,k,'open circuit potential [V]')]=value;values.append(value);entropy.append(ut)
        surface=dot(op.reaction_area,values)/op.area_density;bulk=pb.x_average(dot(op.weights,values)/op.active_fraction)
        v.update(self._get_standard_ocp_variables(surface,bulk,dot(op.reaction_area,entropy)/op.area_density))
        v[f'{D} electrode equilibrium open-circuit potential [V]']=surface;v[f'X-averaged {d} electrode equilibrium open-circuit potential [V]']=pb.x_average(surface);return v

class OpenParticle(AgingParticle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if hasattr(self.op,'regional_chemistry'):raise ValueError('Open POU transport requires a projected spatial mobility law for nonuniform doping; the legacy region mobility cannot be silently reused')
    def add_events_from(self,v):
        lo,hi=law_for(self.op).stoichiometry_bounds
        for k in range(self.op.dimension):
            c=v[key(self.domain,k,'concentration [mol.m-3]')]
            self.events.extend([pb.Event(f'MORQEX {self.domain} node {k} lower concentration',pb.min(c)-lo*self.phase_param.c_max),
                pb.Event(f'MORQEX {self.domain} node {k} upper concentration',hi*self.phase_param.c_max-pb.max(c))])
    def get_coupled_variables(self,v):
        d,D=self.domain_Domain;op=self.op;phase=self.phase_param;law=law_for(op);T=v[f'{D} electrode temperature [K]'];h=1-v[age(d,'damage')]
        c=[v[key(d,k,'concentration [mol.m-3]')] for k in range(op.dimension)];gradient=gradient_potential(op,c)
        mu=[-self.param.F*phase.U(a/phase.c_max,T)+self.feedback*v[key(d,k,'elastic chemical potential [J.mol-1]')]+g for k,(a,g) in enumerate(zip(c,gradient))]
        mean=dot(op.weights,c)/op.active_fraction;Dref=phase.D(mean,T)
        # Dref is the explicitly selected tracer/mobility reference here; it
        # is not silently equated to a measured Fick chemical diffusivity.
        mobility=law.mobility_multiplier*Dref*phase.c_max/(self.param.R*T)
        if law.mobility=='lattice':mobility*=mean/phase.c_max*(1-mean/phase.c_max)
        diffusive=[-mobility*dot(row,mu) for row in op.stiffness.toarray()]
        j=[v[key(d,k,'reaction current [A.m-2]')] for k in range(op.dimension)]
        massdot=[f-coefficient(A)*h*current/self.param.F for f,A,current in zip(diffusive,op.reaction_area,j)]
        diffusion=-sum((f*m for f,m in zip(diffusive,mu)),pb.Scalar(0))
        reaction=sum((coefficient(A)*h*current*v[key(d,k,'reaction overpotential [V]')] for k,(A,current) in enumerate(zip(op.reaction_area,j))),pb.Scalar(0))
        reaction+=v[age(d,'side dissipation [W.m-3]')]+v[age(d,'film dissipation [W.m-3]')]
        total=sum((coefficient(A)*h*current for A,current in zip(op.reaction_area,j)),pb.Scalar(0))+v[age(d,'side current [A.m-3]')]
        port=v[f'{D} electrode surface potential difference [V]']*total
        power=sum((rate*m for rate,m in zip(massdot,mu)),pb.Scalar(0))+v[age(d,'reservoir power [W.m-3]')]
        muT=[-self.param.F*phase.dUdT(a/phase.c_max)+self.feedback*v[key(d,k,'elastic chemical potential dT [J.mol-1.K-1]')] for k,a in enumerate(c)]
        reversible=T*sum((rate*m for rate,m in zip(massdot,muT)),pb.Scalar(0))
        for k,rate in enumerate(massdot):v[key(d,k,'rate [mol.m-3.s-1]')]=rate/coefficient(getattr(op,'division_weights',op.weights)[k])
        v.update({f'MORQEX {d} diffusion dissipation [W.m-3]':diffusion,f'MORQEX {d} reaction dissipation [W.m-3]':reaction,
            f'MORQEX {d} interface port power [W.m-3]':port,f'MORQEX {d} reversible elastic heating [W.m-3]':reversible,
            f'MORQEX {d} gradient energy density [J.m-3]':.5*sum((coefficient(w)*a*g for w,a,g in zip(op.weights,c,gradient)),pb.Scalar(0)),
            f'MORQEX {d} port identity residual [W.m-3]':power+diffusion+reaction-port})
        mean=v[f'R-averaged {d} particle concentration [mol.m-3]'];v.update(self._get_standard_diffusivity_variables(phase.D(mean,T)));v.update(self._get_standard_flux_variables(0*mean));return v

def build_open_dfn(stack,kernels,config,*,parameters,feedback=1.,factories=None,control=None,temperature_interval_K=(273.15,333.15),thermal_mode='lumped'):
    """Compose raw OCP/positive mobility particles with the degradation ports.

    Negative chemical curvature requires an explicitly positive gradient energy.
    This is a model admissibility gate, not a proof that the chosen basis resolves
    a phase interface. That requires a separate mode/mesh convergence study.
    """
    from .aging_dfn import build_aging_dfn
    from .open_materials import TabulatedOCP
    if not stack.linear_ocp_entropy:raise ValueError('Open DFN requires the explicit linear OCP entropy stack contract')
    if not 0<temperature_interval_K[0]<temperature_interval_K[1]:raise ValueError('Ordered positive temperature interval required')
    for d,op in stack.electrodes.items():
        curve=parameters[f'{d.capitalize()} electrode OCP [V]'];law=law_for(op)
        if not isinstance(curve,TabulatedOCP):raise ValueError('Tabulate and audit each raw OCP before constructing the open solver')
        lo,hi=law.stoichiometry_bounds
        if lo<curve.stoichiometry[0] or hi>curve.stoichiometry[-1]:raise ValueError('Concentration validity bounds exceed the supplied OCP table')
        if max(curve.slope_bounds(T)[1] for T in temperature_interval_K)>1e-12 and law.gradient_coefficient_J_m5_mol2<=0:raise ValueError('Nonconvex OCP requires an explicitly positive gradient-energy coefficient and interface-resolution validation')
    roles={'particle':OpenParticle,'ocp':OpenOCP}
    roles.update(factories or {})
    model=build_aging_dfn(stack,kernels,config,feedback=feedback,factories=roles,control=control,thermal_mode=thermal_mode)
    T=model.variables['Volume-averaged cell temperature [K]'];lo,hi=temperature_interval_K
    model.events.extend([pb.Event('MORQEX open minimum temperature',T-lo),pb.Event('MORQEX open maximum temperature',hi-T)])
    return model
