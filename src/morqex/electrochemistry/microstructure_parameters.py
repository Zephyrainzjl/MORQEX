"""Audited spline compression to bound nonlinear DFN compiler memory.

Both full and reduced microstructure models use the SAME compressed functions.
This is a separately reported parameter approximation, not solver tolerance
relaxation. Convexity is checked at each quadratic-derivative extremum.
"""
from functools import partial
import hashlib
import numpy as np
from scipy.interpolate import CubicSpline
from .parameters import convex_parameters,_ocp


def _derivative_minimum(spline):
    h=np.diff(spline.x);a,b,c=3*spline.c[0],2*spline.c[1],spline.c[2]
    values=[c,a*h*h+b*h+c]
    mask=a!=0;vertex=np.zeros_like(a);vertex[mask]=-b[mask]/(2*a[mask]);valid=mask&(vertex>0)&(vertex<h)
    values.append((a*vertex**2+b*vertex+c)[valid])
    return float(min(np.min(v) for v in values if len(v)))


def _maximum_difference(reference,compressed):
    x=np.union1d(reference.x,compressed.x);left=x[:-1];h=np.diff(x)
    A=.5*(reference(left,3)-compressed(left,3));B=reference(left,2)-compressed(left,2);C=reference(left,1)-compressed(left,1)
    candidates=[x];disc=B*B-4*A*C;quadratic=(A!=0)&(disc>=0)
    for sign in [-1,1]:
        root=np.zeros_like(A);root[quadratic]=(-B[quadratic]+sign*np.sqrt(disc[quadratic]))/(2*A[quadratic])
        mask=quadratic&(root>0)&(root<h);candidates.append(left[mask]+root[mask])
    linear=(A==0)&(B!=0);root=np.zeros_like(A);root[linear]=-C[linear]/B[linear]
    mask=linear&(root>0)&(root<h);candidates.append(left[mask]+root[mask])
    points=np.concatenate(candidates)
    return float(np.max(abs(reference(points)-compressed(points))))


def microstructure_parameters(*,value_tolerance_V=2e-5,relative_slope_tolerance=.05):
    p,audit=convex_parameters();audit['compression']={}
    for domain in ['Negative','Positive']:
        original=p[f'{domain} electrode OCP [V]'];x=original.keywords['x'];y=original.keywords['y']
        reference=CubicSpline(x,y);check=np.linspace(x[0],x[-1],40001)
        indices=np.unique(np.rint(np.linspace(0,len(x)-1,129)).astype(int))
        for _ in range(14):
            xc,yc=x[indices],y[indices];compressed=CubicSpline(xc,yc)
            value_error=_maximum_difference(reference,compressed)
            slope_error=float(np.max(abs(compressed(check,1)-reference(check,1))/np.maximum(abs(reference(check,1)),.01)))
            minimum=_derivative_minimum(CubicSpline(xc,-yc))
            if value_error<=value_tolerance_V and slope_error<=relative_slope_tolerance and minimum>0:break
            score=np.maximum(abs(compressed(check)-reference(check))/value_tolerance_V,
                abs(compressed(check,1)-reference(check,1))/(relative_slope_tolerance*np.maximum(abs(reference(check,1)),.01)))
            bins=np.clip(np.searchsorted(xc,check,side='right')-1,0,len(xc)-2)
            worst=np.zeros(len(xc)-1);np.maximum.at(worst,bins,score)
            intervals=np.flatnonzero(worst>1)
            if not len(intervals):intervals=np.array([int(bins[np.argmax(score)])])
            added=(indices[intervals]+indices[intervals+1])//2
            updated=np.union1d(indices,added)
            indices=np.arange(len(x)) if len(updated)==len(indices) else updated
        else:raise ValueError('OCP compression audit failed')
        p.update({f'{domain} electrode OCP [V]':partial(_ocp,x=xc.copy(),y=yc.copy(),domain=domain)})
        audit['compression'][domain.lower()]={'source_knots':len(x),'retained_knots':len(xc),
            'maximum_value_error_V_polynomial_extrema':value_error,'relative_slope_error_audit_grid':slope_error,
            'minimum_positive_negative_slope_V_polynomial_extrema':minimum,
            'table_sha256':hashlib.sha256(np.column_stack([xc,yc]).tobytes()).hexdigest()}
    return p,audit
