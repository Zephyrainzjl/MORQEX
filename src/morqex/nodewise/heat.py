"""Phase-resolved TPFA thermal cell problems and conditional field recovery."""
from dataclasses import dataclass,asdict
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class ThermalPhase:
    conductivity_W_m_K:float
    volumetric_capacity_J_m3_K:float
    source:str
    def __post_init__(self):
        if not self.source.strip() or not np.isfinite([self.conductivity_W_m_K,self.volumetric_capacity_J_m3_K]).all() or min(self.conductivity_W_m_K,self.volumetric_capacity_J_m3_K)<=0:raise ValueError('Positive SI thermal parameters and source required')


class ThermalCell:
    """Real phase conductivity and optional active/CBD thermal contact loss.

    This solves a conditional linear thermal cell, not an independent empirical
    temperature coloring. Use its effective conductivity in the macro heat PDE.
    Quasisteady fine-temperature recovery must additionally satisfy its timescale
    assumption; it does not silently introduce extra stored heat in the DFN.
    """
    def __init__(self,structure,phases,*,damage=0.,contact_resistance_m2_K_W=0.,residual_contact_fraction=1e-3):
        if set(phases)!=set(structure.phase_kinds):raise ValueError('Thermal properties for every present phase required')
        if not 0<=damage<=1 or contact_resistance_m2_K_W<0 or not 0<residual_contact_fraction<=1:raise ValueError('Admissible damage/contact parameters required')
        self.structure=structure;self.phases=phases;a=structure.labels;h=np.array(structure.spacing_zyx_m);volume=float(np.prod(h));self.volume=volume*a.size
        conductivity=np.empty(a.shape);capacity=conductivity.copy()
        for p,law in phases.items():conductivity[a==p]=law.conductivity_W_m_K;capacity[a==p]=law.volumetric_capacity_J_m3_K
        self.capacity=capacity.ravel()*volume;self.mean_capacity=float(capacity.mean());ids=np.arange(a.size).reshape(a.shape)
        rows=[];cols=[];entries=[];diagonal=np.zeros(a.size);kinds=structure.phase_kinds
        g=residual_contact_fraction+(1-residual_contact_fraction)*(1-damage)**2
        for axis in range(3):
            lo=[slice(None)]*3;hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None);lo=tuple(lo);hi=tuple(hi)
            i=ids[lo].ravel();j=ids[hi].ravel();resistance=h[axis]/2*(1/conductivity[lo].ravel()+1/conductivity[hi].ravel())
            interface=np.zeros(i.shape,bool)
            for pa,ka in kinds.items():
                for pb,kb in kinds.items():
                    if {ka,kb}=={'active','cbd'}:interface|=(a[lo].ravel()==pa)&(a[hi].ravel()==pb)
            resistance+=interface*contact_resistance_m2_K_W/g
            edge=volume/h[axis]/resistance
            np.add.at(diagonal,i,edge);np.add.at(diagonal,j,edge)
            rows.extend([i,j]);cols.extend([j,i]);entries.extend([-edge,-edge])
        self.left=np.zeros(a.shape);self.right=np.zeros(a.shape)
        self.left[:,:,0]=2*volume/h[2]**2*conductivity[:,:,0];self.right[:,:,-1]=2*volume/h[2]**2*conductivity[:,:,-1]
        self.left=self.left.ravel();self.right=self.right.ravel();diagonal+=self.left+self.right
        self.K=(sparse.coo_matrix((np.concatenate(entries),(np.concatenate(rows),np.concatenate(cols))),shape=(a.size,)*2)+sparse.diags(diagonal)).tocsc()
        self.factor=splu(self.K);L=a.shape[2]*h[2]
        self.gradient_map=self.factor.solve((self.right-self.left)*L/2)
        current=self.right@(L/2-self.gradient_map)
        self.effective_conductivity=float(current*L/self.volume)
        # Maximum principle, series/parallel bounds and source-free energy.
        if self.gradient_map.min()<-L/2-1e-12*L or self.gradient_map.max()>L/2+1e-12*L:raise ValueError('Thermal maximum principle violated')
        self.metadata={'geometry':structure.fingerprint,'phases':{p:asdict(v) for p,v in phases.items()},'damage':damage,
            'contact_resistance_m2_K_W':contact_resistance_m2_K_W,'effective_conductivity_W_m_K':self.effective_conductivity,
            'mean_volumetric_capacity_J_m3_K':self.mean_capacity,'boundary':'Dirichlet X faces, insulated Y/Z faces',
            'scope':'conditional TPFA cell; local-source/gradient-driven recovery, not a resolved nonlinear electrochemical reaction solve'}
    def response(self,source_modes_W_m3):
        B=np.asarray(source_modes_W_m3,float)
        if B.shape[0]!=self.structure.labels.size or B.ndim!=2 or not np.isfinite(B).all():raise ValueError('Finite voxel-by-mode volumetric heat sources required')
        return self.factor.solve(B*np.prod(self.structure.spacing_zyx_m))
    def recover(self,mean_boundary_temperature_K,gradient_K_m,*,source_W_m3=None):
        source=np.zeros(self.K.shape[0]) if source_W_m3 is None else np.asarray(source_W_m3,float).ravel()
        if source.shape!=(self.K.shape[0],) or not np.isfinite(source).all():raise ValueError('Finite heat source at every voxel required')
        theta=self.factor.solve(source*np.prod(self.structure.spacing_zyx_m))
        temperature=mean_boundary_temperature_K+gradient_K_m*self.gradient_map+theta
        if temperature.min()<=0:raise ValueError('Thermal recovery outside positive absolute temperature')
        flux=self.left@(temperature-mean_boundary_temperature_K+gradient_K_m*self.structure.labels.shape[2]*self.structure.spacing_zyx_m[2]/2)+self.right@(temperature-mean_boundary_temperature_K-gradient_K_m*self.structure.labels.shape[2]*self.structure.spacing_zyx_m[2]/2)
        residual=self.K@(temperature-mean_boundary_temperature_K)-((self.right-self.left)*gradient_K_m*self.structure.labels.shape[2]*self.structure.spacing_zyx_m[2]/2+source*np.prod(self.structure.spacing_zyx_m))
        return {'temperature_K':temperature.reshape(self.structure.labels.shape),
            'outward_boundary_power_W':float(flux),'source_power_W':float(source.sum()*np.prod(self.structure.spacing_zyx_m)),
            'absolute_equilibrium_residual_W':float(np.linalg.norm(residual)),'metadata':self.metadata}


def apply_thermal_homogenization(stack,parameters,cells):
    """Bind computed phase-resolved thermal conductivity/capacity to each x."""
    from .constitutive import field
    stack.thermal_cells=cells
    for d,op in stack.electrodes.items():
        bank=cells[d]
        if len(bank)!=op.nodes:raise ValueError('One thermal cell per DFN location required')
        conductivity=field([c.effective_conductivity for c in bank],d)
        capacity=field([c.mean_capacity for c in bank],d)
        rho=float(parameters[d.capitalize()+' electrode density [kg.m-3]'])
        # Bind d as a default argument: callbacks survive the next loop iteration.
        def constitutive(value,average,domain=d):
            if isinstance(value,(int,float)):return value
            return lambda temperature:value if temperature.domain==[domain+' electrode'] else average
        parameters.update({d.capitalize()+' electrode thermal conductivity [W.m-1.K-1]':constitutive(conductivity,float(np.mean([c.effective_conductivity for c in bank]))),
            d.capitalize()+' electrode specific heat capacity [J.kg-1.K-1]':constitutive(capacity/rho,float(np.mean([c.mean_capacity for c in bank]))/rho)})
    return parameters
