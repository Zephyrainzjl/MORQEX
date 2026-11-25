"""Open material cards and shape-preserving OCPs, with no convex projection.

An arbitrary finite table can be represented. Its transport/phase physics still
has to be selected explicitly: a nonconvex free energy is not ordinary Fickian
diffusion with a negative diffusivity. Linear OCP entropy is supported separately.
"""
from dataclasses import dataclass,field
import hashlib
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
import pybamm as pb

HOSTS={'NMC532':'Mohtat2020','NMC811':'Chen2020','LFP':'Prada2013','LCO':'Ai2020'}
_TABLE_FUNCTIONS={}

class _TableEvaluation:
    __name__='morqex_exact_pchip_piece'
    def __init__(self,curve,entropy=False,order=0):self.curve=curve;self.entropy=entropy;self.order=order
    def __call__(self,value):
        import casadi as ca
        curve=self.curve;polynomial=PchipInterpolator(curve.stoichiometry,curve.entropy_V_K if self.entropy else curve.voltage_V,extrapolate=curve.extrapolate).derivative(self.order)
        if not isinstance(value,(ca.MX,ca.SX,ca.DM)):return polynomial(value)
        key=(curve.fingerprint,self.entropy,self.order)
        if key not in _TABLE_FUNCTIONS:
            z=ca.MX.sym('sto')
            # Binary interval lookup followed by Horner evaluation preserves
            # the SciPy pieces at the knots as well as between them. Repeated
            # full-multiplicity CasADi B-splines returned zero at interior knots
            # in the qualification test and must not be used for this table.
            index=ca.fmin(len(polynomial.x)-2,ca.fmax(0,ca.low(ca.MX(polynomial.x),z,{'lookup_mode':'binary'})))
            shifted=z-ca.MX(polynomial.x)[index];result=ca.MX(0.)
            for row in polynomial.c:result=result*shifted+ca.MX(row)[index]
            def tail(index):
                index=index if index>=0 else len(polynomial.x)-2
                shifted=z-polynomial.x[index];out=0.
                for coefficient in polynomial.c[:,index]:out=out*shifted+float(coefficient)
                return out
            left=tail(0) if curve.extrapolate else ca.MX(np.nan);right=tail(-1) if curve.extrapolate else ca.MX(np.nan)
            result=ca.if_else(z<polynomial.x[0],left,ca.if_else(z>polynomial.x[-1],right,result))
            _TABLE_FUNCTIONS[key]=ca.Function('morqex_ocp_'+curve.fingerprint[:16]+f'_{int(self.entropy)}_{self.order}',[z],[result])
        return _TABLE_FUNCTIONS[key](value)

class PchipFunction(pb.Function):
    def __init__(self,curve,child,*,entropy=False,order=0):
        self.curve=curve;self.entropy=entropy;self.order=order
        super().__init__(_TableEvaluation(curve,entropy,order),child,name=f'function (MORQEX PCHIP {curve.fingerprint} {entropy} {order})')
    def _function_diff(self,children,idx):return pb.Scalar(0.) if self.order>=3 else PchipFunction(self.curve,children[0],entropy=self.entropy,order=self.order+1)
    def create_copy(self,new_children=None,perform_simplifications=True):return PchipFunction(self.curve,self._children_for_copying(new_children)[0],entropy=self.entropy,order=self.order)
    def _function_new_copy(self,children):return self.create_copy(children)

@dataclass(frozen=True)
class TabulatedOCP:
    stoichiometry:object
    voltage_V:object
    source:str
    entropy_V_K:object=None
    reference_temperature_K:float=298.15
    extrapolate:bool=False
    def __post_init__(self):
        x=np.array(self.stoichiometry,dtype=float,copy=True);y=np.array(self.voltage_V,dtype=float,copy=True)
        s=np.zeros_like(x) if self.entropy_V_K is None else np.array(self.entropy_V_K,dtype=float,copy=True)
        if x.ndim!=1 or len(x)<3 or y.shape!=x.shape or s.shape!=x.shape or not np.isfinite(np.r_[x,y,s,self.reference_temperature_K]).all():raise ValueError('Finite matching OCP/entropy arrays and at least three knots required')
        if not self.source.strip() or np.any(np.diff(x)<=0) or x[0]<0 or x[-1]>1 or self.reference_temperature_K<=0:raise ValueError('Ordered stoichiometry in [0,1], source and positive temperature required')
        for name,a in [('stoichiometry',x),('voltage_V',y),('entropy_V_K',s)]:a.flags.writeable=False;object.__setattr__(self,name,a)
    @property
    def fingerprint(self):return hashlib.sha256(self.stoichiometry.tobytes()+self.voltage_V.tobytes()+self.entropy_V_K.tobytes()+json.dumps([self.source,self.reference_temperature_K,self.extrapolate]).encode()).hexdigest()
    def __call__(self,sto):
        return PchipFunction(self,sto)
    def entropy(self,sto):return PchipFunction(self,sto,entropy=True)
    def primitive(self,sto,temperature_K=None):
        temperature=self.reference_temperature_K if temperature_K is None else np.asarray(temperature_K)
        value=PchipInterpolator(self.stoichiometry,self.voltage_V,extrapolate=self.extrapolate).antiderivative()(sto)
        return value+(temperature-self.reference_temperature_K)*self.entropy_primitive(sto)
    def entropy_primitive(self,sto):return PchipInterpolator(self.stoichiometry,self.entropy_V_K,extrapolate=self.extrapolate).antiderivative()(sto)
    def slope_bounds(self,temperature_K=None):
        temperature=self.reference_temperature_K if temperature_K is None else float(temperature_K)
        polynomial=PchipInterpolator(self.stoichiometry,self.voltage_V).derivative()
        polynomial.c[:]+=(temperature-self.reference_temperature_K)*PchipInterpolator(self.stoichiometry,self.entropy_V_K).derivative().c
        a,b,c=polynomial.c;h=np.diff(polynomial.x);values=[c,a*h*h+b*h+c]
        vertex=np.divide(-b,2*a,out=np.zeros_like(a),where=a!=0);selected=(a!=0)&(vertex>0)&(vertex<h)
        values.append((a*vertex*vertex+b*vertex+c)[selected]);all_values=np.concatenate(values)
        return float(all_values.min()),float(all_values.max())
    def metadata(self):
        spline=PchipInterpolator(self.stoichiometry,self.voltage_V);check=np.linspace(self.stoichiometry[0],self.stoichiometry[-1],10001)
        return {'source':self.source,'fingerprint':self.fingerprint,'knots':len(self.stoichiometry),'domain':[float(self.stoichiometry[0]),float(self.stoichiometry[-1])],
            'interpolation':'shape-preserving PCHIP, original knot values unchanged','minimum_sampled_negative_slope_V':float(np.min(-spline(check,1))),
            'slope_extrema_V_exact_piecewise_quadratic':list(self.slope_bounds()),
            'extrapolation':self.extrapolate,'reference_temperature_K':self.reference_temperature_K,'entropy':'linear in temperature; separate PCHIP primitive'}
    @classmethod
    def from_csv(cls,path,*,source,entropy_column=None,**kwargs):
        a=np.loadtxt(path,delimiter=',',skiprows=1)
        return cls(a[:,0],a[:,1],source,None if entropy_column is None else a[:,entropy_column],**kwargs)

@dataclass
class OpenMaterialCard:
    host:str
    parameter_source:object=None
    overrides:dict=field(default_factory=dict)
    ocps:dict=field(default_factory=dict)
    supplements:dict=field(default_factory=dict)
    source_note:str=''
    def build(self):
        source=self.parameter_source if self.parameter_source is not None else HOSTS.get(self.host)
        if source is None:raise ValueError('Custom chemistry needs a named pack or explicit parameter dictionary')
        p=pb.ParameterValues(source);added={k:v for k,v in self.supplements.items() if k not in p.keys()}
        if added and not self.source_note.strip():raise ValueError('Supplemented material properties need an explicit provenance note')
        p.update(added,check_already_exists=False);p.update(self.overrides,check_already_exists=False)
        for domain,curve in self.ocps.items():
            if domain not in {'negative','positive'} or not isinstance(curve,TabulatedOCP):raise ValueError('Explicit negative/positive TabulatedOCP required')
            reference=float(p['Reference temperature [K]'])
            if curve.reference_temperature_K!=reference:curve=TabulatedOCP(curve.stoichiometry,curve.voltage_V+(reference-curve.reference_temperature_K)*curve.entropy_V_K,curve.source+'; reference-temperature conversion',curve.entropy_V_K,reference,curve.extrapolate)
            p.update({f'{domain.capitalize()} electrode OCP [V]':curve,f'{domain.capitalize()} electrode OCP entropic change [V.K-1]':curve.entropy},check_already_exists=False)
        return p,{'host':self.host,'source_parameter_set':source if isinstance(source,str) else 'user dictionary','source_note':self.source_note,
            'overridden_keys':sorted(self.overrides),'supplemented_keys':sorted(added),'ocps':{d:c.metadata() for d,c in self.ocps.items()},
            'scope':'Raw user/native OCP, no isotonic fit or smoothing; transport closure and material calibration are separate contracts'}
    @classmethod
    def host_card(cls,host,**kwargs):
        if host not in HOSTS:raise ValueError('Use NMC532, NMC811, LFP, LCO, or an explicit custom parameter source')
        return cls(host,**kwargs)

def tabulate_parameter_ocps(parameters,*,knots=513,domain=(.001,.999),sampled_tolerance_V=1e-4):
    """Sample native/callable curves with a separate midpoint error audit.

    This is a reported interpolation approximation, not a uniform error proof.
    It never changes values to impose monotonicity or a curvature sign.
    """
    result={};audit={};temperature=float(parameters['Reference temperature [K]'])
    for d in ['negative','positive']:
        fn=parameters[f'{d.capitalize()} electrode OCP [V]'];entropy=parameters[f'{d.capitalize()} electrode OCP entropic change [V.K-1]']
        if isinstance(fn,TabulatedOCP):result[d]=fn;audit[d]=fn.metadata();continue
        for count in [knots,2*knots-1,4*knots-3,8*knots-7]:
            x=np.linspace(*domain,count);mid=(x[1:]+x[:-1])/2
            evaluate=lambda f,s:np.broadcast_to(np.asarray(f(pb.Array(s)).evaluate() if callable(f) else f).ravel(),s.shape)
            y=evaluate(fn,x);s=evaluate(entropy,x);curve=TabulatedOCP(x,y,'sampled native parameter OCP; provenance retained in material card',s,temperature)
            error=float(np.max(abs(PchipInterpolator(x,y)(mid)-evaluate(fn,mid))))
            if error<=sampled_tolerance_V:break
        else:raise ValueError('Raw OCP sampling did not meet its declared midpoint audit tolerance')
        result[d]=curve;audit[d]=curve.metadata()|{'sampled_midpoint_error_V':error,'sampled_tolerance_V':sampled_tolerance_V}
    return result,audit
