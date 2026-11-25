"""Mass- and power-adjoint microstructure/DFN reaction ports.

General mode values may be signed; physical quadrature measures remain positive.
The energy gradient is mass-projected before evaluating site chemical potentials.
This contract is a mapping/interface, not an implementation of every DFN closure.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor,cho_solve

@dataclass
class ConservativePorts:
    mass:np.ndarray
    site_modes:np.ndarray
    macro_interpolation:np.ndarray
    area_weights:np.ndarray
    macro_volumes:np.ndarray
    constant_mode:np.ndarray
    provenance:str
    def __post_init__(self):
        for name in ['mass','site_modes','macro_interpolation','area_weights','macro_volumes','constant_mode']:
            a=np.array(getattr(self,name),dtype=float,copy=True)
            if not np.isfinite(a).all():raise ValueError('Finite port matrices required')
            a.flags.writeable=False;setattr(self,name,a)
        M,T,L,w,v,z=self.mass,self.site_modes,self.macro_interpolation,self.area_weights,self.macro_volumes,self.constant_mode
        m=len(z);q=len(w);p=len(v)
        if M.shape!=(m,m) or T.shape!=(q,m) or L.shape!=(q,p) or min(m,q,p)<1 or not self.provenance.strip():raise ValueError('Compatible mode/site/macro matrices and provenance required')
        if not np.allclose(M,M.T,rtol=1e-12,atol=1e-15) or np.linalg.eigvalsh(M).min()<=0:raise ValueError('State mass matrix must be symmetric positive definite')
        if np.min(w)<=0 or np.min(v)<=0 or not np.isclose(v.sum(),1,rtol=0,atol=1e-12) or np.min(L)<0 or not np.allclose(L.sum(axis=1),1,rtol=0,atol=1e-12):raise ValueError('Positive area/volume measures and conservative macro interpolation required')
        if not np.allclose(T@z,1,rtol=1e-12,atol=1e-12):raise ValueError('Reaction traces must reproduce the constant concentration mode')
        self._factor=cho_factor(M)
    def _solve(self,a):
        a=np.asarray(a,float)
        if a.shape[-1]!=len(self.constant_mode):raise ValueError('Incorrect modal vector dimension')
        return cho_solve(self._factor,a.reshape(-1,a.shape[-1]).T).T.reshape(a.shape)
    def site_chemical_potential(self,energy_gradient):
        """g=dW/da is not a pointwise chemical potential; first solve M mu=g."""
        return self._solve(energy_gradient)@self.site_modes.T
    def site_macro_potential(self,macro_potential):return np.asarray(macro_potential)@self.macro_interpolation.T
    def modal_reaction_rate(self,site_current,faraday=96485.33212):
        if not np.isfinite(faraday) or faraday<=0:raise ValueError('Positive Faraday constant required')
        return -self._solve((np.asarray(site_current)*self.area_weights)@self.site_modes)/faraday
    def macro_current(self,site_current):return (np.asarray(site_current)*self.area_weights)@self.macro_interpolation/self.macro_volumes
    def lithium_inventory(self,coefficients):return np.asarray(coefficients)@(self.mass@self.constant_mode)
    def audit(self,energy_gradient,macro_potential,site_current,faraday=96485.33212):
        g=np.asarray(energy_gradient);j=np.asarray(site_current);delta=np.asarray(macro_potential);mu=self.site_chemical_potential(g);d=self.site_macro_potential(delta)
        rate=self.modal_reaction_rate(j,faraday);electric=float((self.macro_volumes*delta)@self.macro_current(j));chemical=float(g@rate);dissipation=float((self.area_weights*j)@(d+mu/faraday))
        return {'lithium_rate_residual':float(self.lithium_inventory(rate)+self.area_weights@j/faraday),'electric_power_W_m3':electric,
            'chemical_power_W_m3':chemical,'reaction_dissipation_W_m3':dissipation,'power_residual_W_m3':electric-chemical-dissipation,
            'scope':'exact algebraic reaction-port identities; dissipation sign also requires a passive kinetic law; no trajectory/field error bound'}

def nodal_ports(weights,area_by_mode_macro,macro_volumes,*,provenance):
    area=np.asarray(area_by_mode_macro);k,p=np.nonzero(area>0);m=len(weights);P=len(macro_volumes)
    if area.shape!=(m,P) or not np.isfinite(area).all() or np.any(area<0):raise ValueError('Nonnegative finite mode-by-macro reaction measures required')
    return ConservativePorts(np.diag(weights),np.eye(m)[k],np.eye(P)[p],area[k,p],macro_volumes,np.ones(m),provenance)
