"""Positive geometry-derived contact-loss conductivity for the macro heat PDE."""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.interpolate import PchipInterpolator
import pybamm as pb
from .heat import ThermalCell


@dataclass
class ThermalDamageTable:
    damage:np.ndarray
    conductivity_W_m_K:np.ndarray
    metadata:dict


def build_thermal_damage_table(structure,phases,*,contact_resistance_m2_K_W,tolerance=1e-3,maximum_points=65,residual_contact_fraction=1e-3):
    if contact_resistance_m2_K_W<=0 or not 0<tolerance<1 or maximum_points<9:raise ValueError('Positive declared interface resistance and finite interpolation budget required')
    cache={}
    def evaluate(d):
        d=float(d)
        if d not in cache:cache[d]=ThermalCell(structure,phases,damage=d,contact_resistance_m2_K_W=contact_resistance_m2_K_W,residual_contact_fraction=residual_contact_fraction).effective_conductivity
        return cache[d]
    grid=np.linspace(0,1,9)
    while True:
        values=np.array([evaluate(d) for d in grid]);curve=PchipInterpolator(grid,values)
        mid=(grid[:-1]+grid[1:])/2;exact=np.array([evaluate(d) for d in mid]);error=abs(curve(mid)-exact)/exact
        if error.max()<=tolerance:break
        updated=np.union1d(grid,mid[error>tolerance])
        if len(updated)>maximum_points:raise ValueError('Thermal contact interpolation failed tolerance; increase offline sampling budget')
        grid=updated
    if values.min()<=0 or np.diff(values).max()>1e-10*values.max():raise ValueError('Thermal contact operator lost positivity/monotonicity')
    return ThermalDamageTable(grid,values,{'geometry':structure.fingerprint,
        'contact_resistance_m2_K_W':contact_resistance_m2_K_W,'residual_contact_fraction':residual_contact_fraction,
        'sampled_midpoint_maximum_relative_error':float(error.max()),'target_relative_error':tolerance,
        'full_cell_solves':len(cache),'scope':'geometry-resolved scalar TPFA thermal contact degradation; empirical midpoint interpolation audit, not a uniform mathematical bound'})


class _Table:
    __name__='morqex_spatial_thermal_damage'
    def __init__(self,x,values,order):
        self.x=x;self.values=values;self.order=order;self.polynomials=[PchipInterpolator(x,row).derivative(order) for row in values]
    def __call__(self,d):
        import casadi as ca
        if isinstance(d,(ca.MX,ca.SX,ca.DM)):
            if d.numel()!=len(self.values):raise ValueError('Thermal coefficient grid differs from DFN grid')
            out=[]
            for i,poly in enumerate(self.polynomials):
                z=ca.fmin(1,ca.fmax(0,d[i]));idx=ca.fmin(len(self.x)-2,ca.fmax(0,ca.low(ca.MX(self.x),z,{'lookup_mode':'binary'})))
                shifted=z-ca.MX(self.x)[idx];v=ca.MX(0.)
                for row in poly.c:v=v*shifted+ca.MX(row)[idx]
                out.append(v)
            return ca.vertcat(*out)
        a=np.asarray(d)
        if np.isnan(a).any():return np.full_like(a,np.nan)
        if a.size!=len(self.values):raise ValueError('Thermal coefficient grid differs from DFN grid')
        return np.array([poly(float(np.clip(di,0,1))) for poly,di in zip(self.polynomials,a.ravel())]).reshape(a.shape)


class SpatialThermalConductivity(pb.Function):
    def __init__(self,x,values,damage,order=0):
        self.x=np.asarray(x,float);self.values=np.asarray(values,float);self.order=order
        identity=hashlib.sha256(self.x.tobytes()+self.values.tobytes()).hexdigest()
        super().__init__(_Table(self.x,self.values,order),damage,name=f'function (MORQEX thermal contact {identity} {order})')
    def _function_diff(self,children,idx):return SpatialThermalConductivity(self.x,self.values,children[0],self.order+1) if self.order<3 else pb.Scalar(0.)
    def _function_new_copy(self,children):return SpatialThermalConductivity(self.x,self.values,children[0],self.order)
    def create_copy(self,new_children=None,perform_simplifications=True):return self._function_new_copy(self._children_for_copying(new_children))


def conductivity_field(tables,damage):
    x=np.unique(np.concatenate([t.damage for t in tables]))
    values=np.array([PchipInterpolator(t.damage,t.conductivity_W_m_K)(x) for t in tables])
    return SpatialThermalConductivity(x,values,damage)
