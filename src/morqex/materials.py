"""Phase-resolved fixed material catalog; a phase may have overlapping roles.

Elastic tensors and chemical/thermal strains use orthonormal Mandel notation.
Electronic and ionic factors are relative to the corresponding DFN reference
conductivity. No physical property is inferred from a color or phase label.
"""
from dataclasses import dataclass,field
import numpy as np
from .mechanics.tensors import validate_stiffness,isotropic_stiffness

@dataclass(frozen=True)
class Phase:
    name:str
    kind:str
    elasticity:object=None
    chemical_strain_per_concentration:tuple=(0.,)*6
    thermal_strain_per_K:tuple=(0.,)*6
    ionic_efficiency:float=0.
    electronic_efficiency:float=0.
    electrolyte_fraction:float=0.
    diffusion_factor_XYZ:tuple=(1.,1.,1.)
    chemistry:str='single-host'
    source:str='user-supplied, uncalibrated'

    def __post_init__(self):
        if self.kind not in {'pore','active','cbd','inert'} or not self.name:raise ValueError('Explicit phase name and supported geometry semantic required')
        if not np.isfinite([self.ionic_efficiency,self.electronic_efficiency,self.electrolyte_fraction]).all():raise ValueError('Finite phase coefficients required')
        if not 0<=self.ionic_efficiency<=self.electrolyte_fraction<=1 or not 0<=self.electronic_efficiency<=1:raise ValueError('Transport factors outside physical relative bounds')
        for name in ['chemical_strain_per_concentration','thermal_strain_per_K']:
            values=np.asarray(getattr(self,name),float)
            if values.shape!=(6,) or not np.isfinite(values).all():raise ValueError('Six finite Mandel strain components required')
        if self.kind!='active' and np.any(self.chemical_strain_per_concentration):raise ValueError('Only declared host phases have chemical eigenstrain')
        d=np.asarray(self.diffusion_factor_XYZ)
        if d.shape!=(3,) or not np.isfinite(d).all() or np.any(d<=0):raise ValueError('Positive diagonal XYZ diffusion factors required')
        if self.kind=='pore' and self.elasticity is not None:raise ValueError('True pores do not have an elastic tensor')
        if self.kind!='pore' and self.elasticity is None:raise ValueError('Solid phase requires a fixed elastic tensor')
        if self.elasticity is not None:
            C=np.array(validate_stiffness(self.elasticity),copy=True);C.flags.writeable=False;object.__setattr__(self,'elasticity',C)

    @classmethod
    def isotropic(cls,name,kind,young,poisson,**kwargs):return cls(name,kind,isotropic_stiffness(young,poisson),**kwargs)

    def metadata(self):
        return {name:(value.tolist() if isinstance(value,np.ndarray) else value) for name,value in vars(self).items()}

@dataclass(frozen=True)
class PhaseCatalog:
    phases:dict[int,Phase]
    def validate(self,structure):
        if set(self.phases)!=set(structure.phase_kinds):raise ValueError('One material record per present geometry label required')
        if any(self.phases[p].kind!=kind for p,kind in structure.phase_kinds.items()):raise ValueError('Material and geometry phase semantics disagree')
        chemistries={p.chemistry for p in self.phases.values() if p.kind=='active'}
        if len(chemistries)!=1:raise ValueError('Current regional DFN requires a common host chemistry; use a multi-species particle plugin for mixed chemistries')
    def ionic_phases(self,structure):self.validate(structure);return [p for p,v in self.phases.items() if v.ionic_efficiency>0]
    def ionic_efficiencies(self,structure):self.validate(structure);return {p:v.ionic_efficiency for p,v in self.phases.items()}
    def electronic_phases(self,structure):self.validate(structure);return [p for p,v in self.phases.items() if v.electronic_efficiency>0]
    def electronic_efficiencies(self,structure):self.validate(structure);return {p:v.electronic_efficiency for p,v in self.phases.items()}
    def porosity(self,structure):self.validate(structure);return sum(float(np.mean(structure.labels==p))*v.electrolyte_fraction for p,v in self.phases.items())
    def metadata(self):return {'schema':'morqex-phase-catalog-1','phases':{str(p):v.metadata() for p,v in self.phases.items()},
        'scope':'fixed tensors; separable diagonal solid diffusion; single host chemistry; transport factors relative to DFN reference properties'}
