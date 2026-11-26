"""Composable symbolic degradation laws with explicit lithium reservoirs.

Defaults are numerical research values, not calibrated material parameters.
Current is positive for oxidation; reservoir concentrations are mol Li/m3 RVE.
"""
from dataclasses import dataclass,field
from typing import Protocol
import pybamm as pb
import math

def _finite_parameters(instance):
    if any(not math.isfinite(v) for v in vars(instance).values() if isinstance(v,(float,int))):raise ValueError('Finite constitutive parameters required')

@dataclass(frozen=True)
class Reservoir:
    name:str
    initial:float=0.
    scale:float=100.
    chemical_potential_J_mol:float=0.
    volume_m3_mol:float=0.

@dataclass(frozen=True)
class SideContribution:
    current:object
    rates:dict
    equilibrium_V:float
    film_resistance:object=0.

@dataclass(frozen=True)
class ReactionContext:
    states:dict
    delta:object
    temperature:object
    electrolyte_concentration:object
    area:float
    accessible_area:object
    faraday:object
    gas_constant:object
    film_current:object
    film_resistance:object

class SideLaw(Protocol):
    name:str
    def reservoirs(self)->tuple[Reservoir,...]:...
    def resistance(self,states,area):...
    def evaluate(self,context:ReactionContext)->SideContribution:...

@dataclass(frozen=True)
class DiffusionLimitedSEI:
    """Cathodic solvent-diffusion limit, with passive-potential cutoff.

    The potential factor tends to one at large negative overpotential and
    prevents negative dissipation outside the SEI stability range. This is a
    declared constitutive variant, not a reproduction of all OKane SEI laws.
    """
    name:str='sei'
    diffusivity_m2_s:float=2e-22
    solvent_concentration_mol_m3:float=2636.
    initial_thickness_m:float=5e-9
    molar_volume_per_Li:float=4.7925e-5
    resistivity_ohm_m:float=2e5
    equilibrium_V:float=.4
    onset_width_V:float=.02

    def __post_init__(self):
        _finite_parameters(self)
        if min(self.diffusivity_m2_s,self.solvent_concentration_mol_m3,self.initial_thickness_m,self.molar_volume_per_Li,self.onset_width_V)<=0 or self.resistivity_ohm_m<0:raise ValueError('Invalid SEI parameters')
    def reservoirs(self):return (Reservoir(self.name,chemical_potential_J_mol=-96485.33212*self.equilibrium_V,volume_m3_mol=self.molar_volume_per_Li),)
    def thickness(self,states,area):return self.initial_thickness_m+self.molar_volume_per_Li*states[self.name]/pb.maximum(area,1e-30)
    def resistance(self,states,area):return self.resistivity_ohm_m*self.thickness(states,area)
    def evaluate(self,c):
        eta=c.delta-c.film_resistance*c.film_current-self.equilibrium_V
        x=pb.maximum(-eta,0)/self.onset_width_V;activation=x/(1+x)
        j=-c.faraday*self.diffusivity_m2_s*self.solvent_concentration_mol_m3/self.thickness(c.states,c.area)*activation
        return SideContribution(j,{self.name:-c.accessible_area*j/c.faraday},self.equilibrium_V,self.resistance(c.states,c.area))

@dataclass(frozen=True)
class PlatingStrippingDeadLi:
    """Passive linear plating/stripping branches with finite stripping stock.

    Both branches obey j*eta>=0, vanish at zero driving potential, and stripping
    vanishes with plated inventory. Not the native asymmetric BV plating law.
    Dead-Li conversion follows the SEI-dependent decay motif of OKane2022.
    """
    name:str='plating'
    conductance_A_m2_V:float=.2
    stripping_rate_per_V_s:float=1e-3
    dead_rate_s:float=1e-6
    sei_reference_m:float=5e-9
    lithium_molar_volume:float=1.3e-5
    initial_plated_mol_m3:float=0.
    def __post_init__(self):
        _finite_parameters(self)
        if min(self.conductance_A_m2_V,self.stripping_rate_per_V_s,self.dead_rate_s,self.initial_plated_mol_m3)<0 or min(self.sei_reference_m,self.lithium_molar_volume)<=0:raise ValueError('Invalid plating parameters')
    def reservoirs(self):return (Reservoir(self.name,initial=self.initial_plated_mol_m3,volume_m3_mol=self.lithium_molar_volume),Reservoir('dead_lithium',volume_m3_mol=self.lithium_molar_volume))
    def resistance(self,states,area):return 0.
    def evaluate(self,c):
        eta=c.delta-c.film_resistance*c.film_current
        plate=-self.conductance_A_m2_V*(c.electrolyte_concentration/1000.)*pb.maximum(-eta,0)
        strip=c.faraday*self.stripping_rate_per_V_s*pb.maximum(c.states[self.name],0)/pb.maximum(c.area,1e-30)*pb.maximum(eta,0)
        j=plate+strip
        L=c.states.get('sei',0.)*0.+self.sei_reference_m
        # Actual SEI thickness is exposed by the network when available.
        L=c.states.get('sei_thickness_m',L)
        dead=self.dead_rate_s*self.sei_reference_m/L*pb.maximum(c.states[self.name],0)
        return SideContribution(j,{self.name:-c.accessible_area*j/c.faraday-dead,'dead_lithium':dead},0.)

@dataclass(frozen=True)
class EnergyDrivenDebonding:
    name:str='cohesive_debonding'
    rate_s:float=2e-6
    threshold_J_m3:float=2e4
    exponent:float=1.
    maximum:float=.85
    def __post_init__(self):
        _finite_parameters(self)
        if self.rate_s<0 or self.threshold_J_m3<=0 or self.exponent<1 or not 0<self.maximum<.999:raise ValueError('Invalid damage law')
    def evaluate(self,damage,release_rate):
        return self.rate_s*pb.maximum(release_rate/self.threshold_J_m3-1,0)**self.exponent*pb.maximum(self.maximum-damage,0)

@dataclass(frozen=True)
class NoDamage:
    name:str='none'
    maximum:float=.85
    def evaluate(self,damage,release_rate):return 0*damage

@dataclass(frozen=True)
class DegradationOptions:
    negative:tuple=field(default_factory=lambda:(DiffusionLimitedSEI(),PlatingStrippingDeadLi()))
    positive:tuple=()
    damage:object=field(default_factory=EnergyDrivenDebonding)
    porosity_change:bool=True
    initial_damage:float=0.
    minimum_electrolyte_mol_m3:float=20.
    def __post_init__(self):
        _finite_parameters(self)
        if not 0<=self.initial_damage<self.damage.maximum:raise ValueError('Initial damage must be within constitutive range')
        if self.minimum_electrolyte_mol_m3<=0:raise ValueError('Positive electrolyte validity floor required')
        for laws in [self.negative,self.positive]:
            names=[r.name for law in laws for r in law.reservoirs()]
            if len(set(names))!=len(names):raise ValueError('Reservoir ownership must be unique')

def standard_registry():
    from .extensions import Registry,ComponentSpec
    from .bases import GeometricBasis,ExactBasis,PortBasis,VariationalOperator
    from .ct import SegmentedCT,ArrayGeometry
    from .topology import PorousCBD
    from .materials import PhaseCatalog
    from .electrochemistry.aging_dfn import EnergyKernel,AgingThermal,AgingNetwork
    from .solvers.pybamm_backend import solver
    r=Registry()
    for kind,name,factory in [('basis','geometry',GeometricBasis),('basis','exact',ExactBasis),('basis','port',PortBasis),
        ('operator','variational',VariationalOperator),('sei','diffusion-limited',DiffusionLimitedSEI),
        ('plating','passive-dead-li',PlatingStrippingDeadLi),('damage','energy-driven',EnergyDrivenDebonding),('damage','none',NoDamage),
        ('geometry','segmented-CT',SegmentedCT),('geometry','array',ArrayGeometry),('topology','porous-CBD',PorousCBD),('materials','phase-catalog',PhaseCatalog),
        ('energy','rational-cohesive',EnergyKernel),('thermal','lumped-ports',AgingThermal),('kinetics','regional-network',AgingNetwork),('solver','idaklu',solver)]:
        r.register(kind,ComponentSpec(name,factory,assumptions=('fixed bulk geometry','SI units','explicit numerical qualification required')))
    return r
