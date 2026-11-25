"""Explicit, audited convex chemical-energy branch for mobility D/f''.

This changes the parameterization and is never applied to raw DFN references.
The default qualified set has zero entropic OCP, so monotonicity persists in T.
"""
import hashlib
from functools import partial
import numpy as np
from scipy.optimize import isotonic_regression
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import pybamm as pb


def _ocp(sto, *, x, y, domain):
    """Top-level callable so native PyBaMM checkpoints remain serializable."""
    return pb.Interpolant(x,y,sto,interpolator='cubic',name=f'MORQEX_convex_{domain}_OCP')


def convex_parameters(name='OKane2022', *, slope_floor_V=.01, points=4001,allow_idealization=False):
    if name not in {'OKane2022','Prada2013','Ai2020'}:raise ValueError('Unsupported source parameter set')
    if name!='OKane2022' and not allow_idealization:raise ValueError('Other hosts require explicit consent to the ideal thermodynamic projection')
    if slope_floor_V<=0 or points<101:raise ValueError('Positive slope floor and adequate sampling required')
    native=pb.ParameterValues(name);supplements=[];idealizations={}
    if name=='Prada2013':
        p=pb.ParameterValues('OKane2022');supplements=[k for k in p.keys() if k not in native.keys()]
        p.update(dict(native.items()),check_already_exists=False)
    else:p=native
    if allow_idealization:
        for domain in ['Negative','Positive']:
            field=f'{domain} electrode OCP entropic change [V.K-1]'
            if not isinstance(p[field],(float,int)) or p[field]!=0:idealizations[field]='set to zero for the qualified fixed chemical-energy primitive';p.update({field:0.})
        if not isinstance(p['Cation transference number'],(float,int)):
            import inspect
            fn=p['Cation transference number'];args=[pb.Scalar(p['Initial concentration in electrolyte [mol.m-3]']),pb.Scalar(p['Reference temperature [K]'])]
            value=fn(*args[:len(inspect.signature(fn).parameters)]);value=float(value.evaluate()) if hasattr(value,'evaluate') else float(value)
            p.update({'Cation transference number':value});idealizations['Cation transference number']='frozen at reference concentration and temperature'
        if not isinstance(p['Thermodynamic factor'],(float,int)) or p['Thermodynamic factor']!=1:
            p.update({'Thermodynamic factor':1.});idealizations['Thermodynamic factor']='set to one for ideal electrolyte energy'
    x=np.linspace(.001,.999,points);audit={}
    for domain in ['Negative','Positive']:
        if p[f'{domain} electrode OCP entropic change [V.K-1]']!=0:raise ValueError('Temperature-dependent OCP needs a separate convexity analysis')
        raw=np.asarray(p[f'{domain} electrode OCP [V]'](pb.Array(x)).evaluate()).ravel()
        isotonic=isotonic_regression(raw+slope_floor_V*x,increasing=False).x
        # Smooth the monotone isotonic component before cubic interpolation.
        # The slope floor is added after smoothing; the final dense derivative
        # audit rejects any cubic overshoot rather than clipping mobility.
        fitted=gaussian_filter1d(isotonic,8.,mode='nearest')-slope_floor_V*x
        check=np.linspace(x[0],x[-1],10*points)
        minimum=float(np.min(-CubicSpline(x,fitted).derivative()(check)))
        if minimum<=0:raise ValueError('Convex OCP interpolation audit failed')
        ocp=partial(_ocp,x=x.copy(),y=fitted.copy(),domain=domain)
        p.update({f'{domain} electrode OCP [V]':ocp})
        audit[domain.lower()]={'max_ocp_change_V':float(np.max(abs(raw-fitted))),
            'rms_ocp_change_V':float(np.mean((raw-fitted)**2)**.5),
            'min_negative_slope_V':minimum,
            'table_sha256':hashlib.sha256(np.column_stack([x,raw,fitted]).tobytes()).hexdigest()}
    return p,{'source_parameter_set':name,'supplemented_from_OKane2022':supplements,'idealizations':idealizations,'modification':'explicit isotonic OCP projection, Gaussian smoothing (8 grid samples), strictly positive slope floor and audited cubic interpolation',
        'stoichiometry_interval':[float(x[0]),float(x[-1])],'slope_floor_V':slope_floor_V,'points':points,'electrodes':audit,
        'scope':'modified model, not an unmodified published parameter set; degradation disabled'}
