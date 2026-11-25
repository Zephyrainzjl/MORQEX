"""Composition-resolved constitutive inputs for a common cathode host.

No dopant-property relationship is inferred. Profiles are explicit measured,
computed or hypothetical inputs with provenance. Different host chemistries and
capacity changes require a multi-species closure and are deliberately rejected.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pybamm as pb
from .electrochemistry.microstructure_dfn import MicrostructureOCP,key,dot


@dataclass(frozen=True)
class DopantLaw:
    species:str
    source:str
    offset_V_per_fraction:float=0.
    slope_V_per_fraction:float=0.
    log_diffusion_per_fraction:float=0.
    log_kinetics_per_fraction:float=0.
    log_side_reaction_per_fraction:float=0.
    log_young_per_fraction:float=0.
    log_chemical_strain_per_fraction:float=0.
    site:str='declared substitutional site'
    def __post_init__(self):
        values=[v for v in vars(self).values() if isinstance(v,(int,float))]
        if not self.species or not self.source or not np.isfinite(values).all():raise ValueError('Dopant species, property provenance and finite coefficients required')
        if self.slope_V_per_fraction>0:raise ValueError('This convex host extension requires a nonpositive OCP slope correction')


@dataclass(frozen=True)
class RegionalChemistry:
    fraction:object
    law:DopantLaw
    def __post_init__(self):
        value=np.array(self.fraction,float,copy=True)
        if value.ndim!=1 or not len(value) or not np.isfinite(value).all() or np.any((value<0)|(value>1)):raise ValueError('One atomic/site fraction in [0,1] per region required')
        value.flags.writeable=False;object.__setattr__(self,'fraction',value)
        for name,coefficient in [('offset_V',self.law.offset_V_per_fraction),('slope_V',self.law.slope_V_per_fraction)]:
            array=value*coefficient;array.flags.writeable=False;object.__setattr__(self,name,array)
        for name,coefficient in [('diffusion_factor',self.law.log_diffusion_per_fraction),('kinetic_factor',self.law.log_kinetics_per_fraction),('side_factor',self.law.log_side_reaction_per_fraction)]:
            if np.max(abs(value*coefficient))>20:raise ValueError('Constitutive factor outside qualification range')
            array=np.exp(value*coefficient);array.flags.writeable=False;object.__setattr__(self,name,array)
    def ocp_shift(self,k,sto):return float(self.offset_V[k])+float(self.slope_V[k])*(sto-.5)
    def primitive_shift(self,k,c,cmax):
        c=np.asarray(c);return self.offset_V[k]*c+self.slope_V[k]*(c*c/(2*cmax)-.5*c)
    def metadata(self):return {'schema':'morqex-common-host-doping-1','fraction':self.fraction.tolist(),'law':vars(self.law),
        'scope':'common host/cmax; fixed prescribed dopant; no species migration, capacity change or inferred dopant physics'}
    @property
    def fingerprint(self):return hashlib.sha256(json.dumps(self.metadata(),sort_keys=True).encode()).hexdigest()


def composition_partition(electrode,base_partition,fractions):
    """Split regions at composition boundaries without losing inventories.

    Continuous profiles may produce many regions. No hidden averaging or binning
    is performed; a caller can explicitly prescribe a coarser composition model.
    """
    from .extensions import validate_partition
    partition=validate_partition(electrode,base_partition);f=np.asarray(fractions,float)
    if f.shape!=(electrode.n,) or not np.isfinite(f).all() or np.any((f<0)|(f>1)):raise ValueError('Valid active-cell dopant fractions required')
    from scipy.sparse.csgraph import connected_components
    labels=np.empty(electrode.n,int);means=[];next_id=0
    for region in np.unique(partition):
        region_ids=np.flatnonzero(partition==region)
        for composition in np.unique(f[region_ids]):
            ids=region_ids[f[region_ids]==composition]
            count,components=connected_components(electrode.diffusion.stiffness[ids][:,ids],directed=False)
            for k in range(count):
                selected=ids[components==k];labels[selected]=next_id;means.append(float(composition));next_id+=1
    return validate_partition(electrode,labels),np.asarray(means)


class DopedOCP(MicrostructureOCP):
    def get_coupled_variables(self,v):
        v=super().get_coupled_variables(v);d,D=self.domain_Domain;op=self.op
        if not hasattr(op,'regional_chemistry'):return v
        chem=op.regional_chemistry
        U=[];ent=[]
        for k in range(op.dimension):
            c=v[key(d,k,'concentration [mol.m-3]')];name=key(d,k,'open circuit potential [V]')
            v[name]+=chem.ocp_shift(k,c/self.phase_param.c_max);U.append(v[name])
            ent.append(-self.feedback*v[key(d,k,'elastic chemical potential dT [J.mol-1.K-1]')]/self.param.F)
        surface=dot(op.reaction_area,U)/op.area_density;bulk=pb.x_average(dot(op.weights,U)/op.active_fraction)
        v.update(self._get_standard_ocp_variables(surface,bulk,dot(op.reaction_area,ent)/op.area_density))
        v[f'{D} electrode equilibrium open-circuit potential [V]']=surface
        v[f'X-averaged {d} electrode equilibrium open-circuit potential [V]']=pb.x_average(surface)
        return v


def attach_doping(stack,profiles):
    if not set(profiles)<=set(stack.electrodes):raise ValueError('Unknown electrode for doping profile')
    # Install zero profiles on the other host so the OCP factory has one
    # consistent contract. All profiles use a common host cmax per electrode.
    for d,op in stack.electrodes.items():
        profile=profiles.get(d,RegionalChemistry(np.zeros(op.dimension),DopantLaw('none','zero dopant limit')))
        if len(profile.fraction)!=op.dimension:raise ValueError('Dopant profile/region dimension mismatch')
        op.regional_chemistry=profile
    stack.regional_chemistries={d:op.regional_chemistry for d,op in stack.electrodes.items()}
    return {'ocp':DopedOCP}


@dataclass
class CompositionBasis:
    base:object
    fractions:object
    def build(self,electrode,target):return composition_partition(electrode,self.base.build(electrode,target),self.fractions)[0]


@dataclass
class DopedOperator:
    fractions:object
    law:DopantLaw
    def build(self,electrode,partition):
        labels,values=composition_partition(electrode,partition,self.fractions)
        if not np.array_equal(labels,partition):
            # Label permutation is harmless, actual splitting is explicit.
            if len(values)!=int(np.max(partition))+1:raise ValueError('Use CompositionBasis to avoid averaging distinct dopant compositions')
        profile=np.array([np.asarray(self.fractions)[partition==k][0] for k in range(int(np.max(partition))+1)])
        result=electrode.reduce(partition,'composition-preserving');result.regional_chemistry=RegionalChemistry(profile,self.law)
        return result


def composition_materials(structure,active_fractions,law,base_catalog):
    """Refine active phase labels and their fixed mechanical properties.

    This is a declared constitutive map, not an atomistic dopant predictor.
    The source segmentation and its hash remain in provenance; only this derived
    composition-labelled geometry is changed. Other phases retain their records.
    """
    from dataclasses import replace
    from .geometry import MicrostructureData
    from .materials import PhaseCatalog
    base_catalog.validate(structure)
    selected=np.isin(structure.labels,list(p for p,k in structure.phase_kinds.items() if k=='active'))
    fractions=np.asarray(active_fractions,float)
    if fractions.shape!=(int(selected.sum()),) or not np.isfinite(fractions).all() or np.any((fractions<0)|(fractions>1)):raise ValueError('One valid fraction per active voxel required')
    labels=structure.labels.astype(np.int64).copy();flat=labels.ravel();ids=np.flatnonzero(selected)
    phases={p:v for p,v in base_catalog.phases.items() if v.kind!='active'};next_id=int(labels.max())+1
    # Keep different source active phases separate even if composition matches.
    for original in np.unique(structure.labels[selected]):
        original_mask=structure.labels.ravel()[ids]==original
        for value in np.unique(fractions[original_mask]):
            if max(abs(value*law.log_young_per_fraction),abs(value*law.log_chemical_strain_per_fraction))>20:raise ValueError('Mechanical dopant factor outside qualification range')
            chosen=ids[original_mask&(fractions==value)];flat[chosen]=next_id;base=base_catalog.phases[int(original)]
            phases[next_id]=replace(base,name=f'{base.name} / {law.species}={value:g}',
                elasticity=base.elasticity*np.exp(value*law.log_young_per_fraction),
                chemical_strain_per_concentration=tuple(np.asarray(base.chemical_strain_per_concentration)*np.exp(value*law.log_chemical_strain_per_fraction)),
                source=base.source+'; dopant map: '+law.source)
            next_id+=1
    data=MicrostructureData(labels,structure.spacing_zyx_m,structure.origin_zyx_m,{p:v.kind for p,v in phases.items()},
        dict(structure.provenance)|{'composition_parent_fingerprint':structure.fingerprint,'dopant_law':vars(law)})
    catalog=PhaseCatalog(phases);catalog.validate(data);return data,catalog
