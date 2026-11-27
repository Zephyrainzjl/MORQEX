"""Positive, orientation-aware tabulated chemical deformation and exact derivative.

Input stretches are dimensionless lattice-length ratios, not expansion rates.
The table is normalized at the initial stress-free stoichiometry. Nonmonotone
lattice collapse is allowed; no extrapolation or arbitrary strain clipping.
This path is isothermal and history independent, not a hysteresis model.
"""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from .contracts import _owned

@dataclass(frozen=True)
class ChemicalStretchTable:
    stoichiometry: np.ndarray
    principal_stretches: np.ndarray
    reference_stoichiometry: float
    provenance: str

    def __post_init__(self):
        x=np.asarray(self.stoichiometry,float);s=np.asarray(self.principal_stretches,float)
        if x.ndim!=1 or len(x)<3 or s.shape!=(len(x),3) or not np.isfinite(x).all() or not np.isfinite(s).all():
            raise ValueError('Finite stoichiometry and three principal stretch columns required')
        if np.any(np.diff(x)<=0) or x[0]<0 or x[-1]>1 or np.any(s<=0):
            raise ValueError('Ordered physical stoichiometry and positive principal stretches required')
        if not np.isfinite(self.reference_stoichiometry) or not x[0]<=self.reference_stoichiometry<=x[-1] or not self.provenance.strip():
            raise ValueError('Resolved reference state and explicit parameter provenance required')
        object.__setattr__(self,'stoichiometry',_owned(x));object.__setattr__(self,'principal_stretches',_owned(s))
        p=PchipInterpolator(x,s,axis=0,extrapolate=False)
        object.__setattr__(self,'_interpolator',p);object.__setattr__(self,'_reference',p(self.reference_stoichiometry))

    def evaluate(self,theta,rotations,cmax):
        t=np.asarray(theta,float);R=np.asarray(rotations,float)
        if t.ndim!=1 or R.shape!=(len(t),3,3) or not np.isfinite(t).all() or not np.isfinite(R).all():
            raise ValueError('One finite proper rotation per stoichiometry required')
        if np.any(t<self.stoichiometry[0]) or np.any(t>self.stoichiometry[-1]):
            raise ValueError('Chemical stretch outside measured tabulation; extrapolation is forbidden')
        if not np.isfinite(cmax) or cmax<=0:raise ValueError('Positive maximum molar concentration required')
        if not np.allclose(R@R.swapaxes(1,2),np.eye(3),rtol=0,atol=1e-12) or not np.allclose(np.linalg.det(R),1.,rtol=0,atol=1e-12):
            raise ValueError('Proper SO(3) grain rotations required')
        stretches=self._interpolator(t)/self._reference
        derivative=self._interpolator.derivative()(t)/(self._reference*cmax)
        if np.any(stretches<=0):raise ArithmeticError('Interpolated chemical deformation inverted')
        return (np.einsum('eik,ek,ejk->eij',R,stretches,R),
                np.einsum('eik,ek,ejk->eij',R,derivative,R))

    @classmethod
    def load(cls,workspace,path,sha256,host):
        p=Path(path)
        if p.is_absolute() or not (workspace.root/p).resolve().is_relative_to(workspace.root.resolve()):
            raise ValueError('Chemical material data must be in the external workspace')
        p=(workspace.root/p).resolve()
        if p.stat().st_size>1024**2:raise ValueError('Material table exceeds1MiB budget')
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha256:raise ValueError('Chemical stretch table checksum mismatch')
        data=json.loads(raw)
        if set(data)!={'schema','host','stoichiometry','principal_stretches','reference_stoichiometry','provenance'} or data['schema']!='morqex-chemical-stretch-1' or data['host']!=host:
            raise ValueError('Explicit matching chemical stretch material schema required')
        return cls(np.array(data['stoichiometry']),np.array(data['principal_stretches']),data['reference_stoichiometry'],data['provenance'])
