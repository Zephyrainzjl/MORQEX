"""Fixed-C thermoelastic RVE energy for a declared mean-concentration closure.

The DFN radial particle field is NOT mapped into irregular particles. RVE input
is the intrinsic active-volume mean only; free-sphere zero-mean diffusion energy
is represented separately. This additive closure is an explicit model assumption.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from morqex.gemsim import GEMsim
from morqex.mechanics import TetMesh,ElasticitySolver,isotropic_stiffness
from morqex.energy import ResponseBuilder


@dataclass
class ElectrodeClosure:
    H: np.ndarray
    response: object
    active_fraction: float
    dfn_active_fraction: float
    reference_concentration: float
    young: float
    poisson: float
    molar_volume: float
    metadata: dict

    @property
    def dfn_H(self): return self.H*self.dfn_active_fraction/self.active_fraction

    @property
    def sphere_curvature(self): return 2*self.young*self.molar_volume**2/(9*(1-self.poisson))

    @classmethod
    def from_gemsim(cls,workspace,parameters,domain,*,shape=(5,5,5),seed=104,thermal_expansion=1e-5,cbd_young=1e9):
        if domain not in {'negative','positive'}:raise ValueError('Electrode domain required')
        Domain=domain.capitalize();p=parameters
        eps=float(p[f'{Domain} electrode active material volume fraction']);pore=float(p[f'{Domain} electrode porosity'])
        young=float(p[f"{Domain} electrode Young's modulus [Pa]"]);nu=float(p[f"{Domain} electrode Poisson's ratio"])
        omega=float(p[f'{Domain} electrode partial molar volume [m3.mol-1]'])
        reference=float(p[f'Initial concentration in {domain} electrode [mol.m-3]'])
        if not 0<eps<=1-pore+1e-12 or not np.isfinite([thermal_expansion,cbd_young]).all() or cbd_young<=0:raise ValueError('Invalid phase fractions/materials')
        provider=GEMsim(workspace)
        cbd=max(0.,1-pore-eps)
        fractions=(pore,eps,cbd) if cbd>1e-12 else (pore,1-pore)
        semantics={0:'pore',1:'active',2:'cbd'} if cbd>1e-12 else {0:'pore',1:'active'}
        data=provider.random_field(shape=shape,fractions=fractions,length_scale=(.8,.9,1.1),seed=seed,
            spacing=(2e-6,)*3,phase_kinds=semantics)
        volume=provider.to_volume(data);mesh=TetMesh.from_volume(volume,solid_phases=[1,2])
        tensors={1:isotropic_stiffness(young,nu),2:isotropic_stiffness(cbd_young,.3)}
        solver=ElasticitySolver(mesh,np.array([tensors[int(ph)] for ph in mesh.phases]),boundary='kubc')
        modes=np.zeros((len(mesh.cells),6,2));active=mesh.phases==1
        modes[active,:3,0]=omega/3
        modes[:,:3,1]=thermal_expansion
        response=ResponseBuilder(solver,modes).exact()
        response.metadata.update(input_units=['1']*6+['mol.m-3','K'],closure='active-volume mean concentration only; not a radial-to-geometry mapping',
            material_sources={'active E/nu/Omega':'selected PyBaMM parameter record','CBD and linear thermal alpha':'declared synthetic sensitivity assumptions'},geometry_seed=seed)
        fraction=float(mesh.volumes[active].sum()/mesh.rve_volume)
        H=response.H[np.ix_([0,6,7],[0,6,7])]
        return cls(H,response,fraction,eps,reference,young,nu,omega,
            {'domain':domain,'geometry_fingerprint':data.fingerprint,'seed':seed,'grid_shape':list(shape),
             'actual_active_fraction':fraction,'dfn_active_fraction':eps,'young_Pa':young,'poisson':nu,
             'molar_volume_m3_mol':omega,'thermal_expansion_per_K':thermal_expansion,'cbd_young_Pa':cbd_young,
             'boundary':'KUBC SVE; plane strain laterally in stack; no RVE representativity claim','model_scope':'uniform intrinsic concentration RVE closure; synthetic geometry/CBD/alpha, no CT/material calibration'})

    def save(self,workspace,name):
        path=self.response.save(workspace,name)
        manifest=json.loads((path/'manifest.json').read_text());manifest['units']['a']='mixed: [mol.m-3, K], metadata.input_units'
        manifest['closure']=self.metadata
        (path/'manifest.json').write_text(json.dumps(manifest,indent=2))
        return path


class AxialStackClosure:
    """Plane-strain electrode/separator stack, common axial traction.

    Each electrode supplies energy in (epsilon_xx, mean-c deviation, DeltaT).
    No extra lithium state is introduced. Fixed thickness and prescribed pressure
    are mutually exclusive mechanical ensembles.
    """
    def __init__(self,negative,positive,parameters,*,constraint='pressure',pressure_Pa=0.,thickness_change_m=0.,separator_modulus_Pa=1e8):
        if constraint not in {'pressure','fixed-thickness'}:raise ValueError('Unknown stack constraint')
        self.electrodes={'negative':negative,'positive':positive};self.constraint=constraint
        self.pressure=float(pressure_Pa);self.target=float(thickness_change_m)
        self.lengths={d:float(parameters[f'{d.capitalize()} electrode thickness [m]']) for d in self.electrodes}
        self.separator_length=float(parameters['Separator thickness [m]']);self.separator_C=float(separator_modulus_Pa)
        self.total_length=float(parameters['Negative current collector thickness [m]']+parameters['Positive current collector thickness [m]']+sum(self.lengths.values())+self.separator_length)
        if not np.isfinite([self.pressure,self.target,self.separator_C]).all() or self.separator_C<=0:raise ValueError('Finite mechanical loading/positive separator stiffness required')
        self.compliance=self.separator_length/self.separator_C+sum(self.lengths[d]/e.dfn_H[0,0] for d,e in self.electrodes.items())
        self.thermal_numerator=sum(self.lengths[d]*e.dfn_H[0,2]/e.dfn_H[0,0] for d,e in self.electrodes.items())
        self.stress_T=self.thermal_numerator/self.compliance if constraint=='fixed-thickness' else 0.
        self.energy_TT_per_area=sum(self.lengths[d]*(e.dfn_H[2,2]-e.dfn_H[0,2]**2/e.dfn_H[0,0]) for d,e in self.electrodes.items())
        if constraint=='fixed-thickness':self.energy_TT_per_area+=self.thermal_numerator**2/self.compliance
        self.mu_T={d:(e.dfn_H[1,2]-e.dfn_H[0,1]*e.dfn_H[0,2]/e.dfn_H[0,0]+e.dfn_H[0,1]/e.dfn_H[0,0]*self.stress_T)/e.dfn_active_fraction for d,e in self.electrodes.items()}

    def traction(self,mean_deviations,temperature_deviation):
        if self.constraint=='pressure':return -self.pressure+0*temperature_deviation
        numerator=self.target+sum(self.lengths[d]*(e.dfn_H[0,1]*mean_deviations[d]+e.dfn_H[0,2]*temperature_deviation)/e.dfn_H[0,0] for d,e in self.electrodes.items())
        return numerator/self.compliance
