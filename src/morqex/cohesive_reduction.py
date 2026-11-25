"""Galerkin compression of the positive cohesive resolvent.

This is a reduced-basis application, not a claim of a new projection theorem.
Certificates concern a fixed discrete geometry, concentration input and damage;
they do not certify the nonlinear DFN trajectory or continuum discretization.
"""
from dataclasses import replace
import numpy as np
from scipy import linalg


def compress_cohesive(response, *, tolerance=1e-6, maximum_damage=.85,
                      maximum_modes=80, training_points=17):
    """Return a variational response, plus an independent residual certificate.

    A(d)=I+s(d)diag(rho)>0. Orthogonal projection of A onto Q preserves
    symmetry/positive residues and gives H_reduced >= H_full. Training only uses
    the operator, not validation trajectories. Selection is empirical on a grid;
    the residual certificate itself is evaluable at every d in [0,1].
    """
    if not 0<tolerance<1 or not 0<maximum_damage<=1 or maximum_modes<1 or training_points<2:
        raise ValueError('Invalid cohesive compression budget')
    F=response.factors;rho=response.eigenvalues;n,q=F.shape
    if not n:raise ValueError('A nonempty cohesive spectrum is required')
    scale=np.sqrt(np.maximum(np.abs(response.direct.diagonal()),1e-30))
    normalized=F/scale
    samples=np.linspace(0,maximum_damage,training_points)
    snapshots=np.column_stack([normalized/(1+(response.retention(d)-1)*rho[:,None]) for d in samples])
    U,sigma,_=linalg.svd(snapshots,full_matrices=False)
    limit=min(n,maximum_modes,U.shape[1]);chosen=None
    # A normalized force error, unlike the squared energy error, controls all
    # generalized forces in this specified input metric. Not a relative bound
    # against a possibly zero actual stress/chemical potential.
    trials=[]
    for r in range(min(q,limit),limit+1):
        Q=U[:,:r];T=Q.T@(rho[:,None]*Q);B=Q.T@normalized
        worst=0.
        for d in samples:
            s=response.retention(d)-1
            Z=linalg.solve(np.eye(r)+s*T,B,assume_a='pos')
            residual=normalized-Q@Z-s*rho[:,None]*(Q@Z)
            a=1+s*float(rho.max())
            worst=max(worst,float(linalg.norm(normalized,2)*linalg.norm(residual,2)/a))
        trials.append((r,worst))
        if worst<=tolerance:chosen=r;break
    if chosen is None:
        # No false qualification: retain a complete basis when it fits budget,
        # otherwise explicitly refuse the requested tolerance.
        if n<=maximum_modes:Q=np.eye(n);chosen=n
        else:raise ValueError(f'Cohesive compression failed tolerance within {maximum_modes} modes; last={trials[-1]}')
    else:Q=U[:,:chosen]
    T=(Q.T@(rho[:,None]*Q));lam,V=linalg.eigh((T+T.T)/2);Q=Q@V
    if lam.min()<-1e-10 or lam.max()>1+1e-10:raise ValueError('Projected spectrum lost positivity')
    lam=np.clip(lam,0,1);factors=Q.T@F
    # A stable QR root avoids cancellation in R^T R, especially at a training
    # point where the physical residual is close to roundoff.
    _,root=linalg.qr(np.column_stack([normalized,Q,rho[:,None]*Q]),mode='economic')
    meta=dict(response.metadata)
    meta['compression']={'method':'Galerkin cohesive resolvent / operator-only snapshot SVD',
        'original_modes':n,'reduced_modes':chosen,'training_maximum_damage':maximum_damage,
        'training_points':training_points,'target_normalized_force_bound':tolerance,
        'training_trials':trials,'certificate_scope':'fixed discrete q,d; not DFN trajectory, spatial error or RVE representativity'}
    arrays={name+'_modes':np.einsum('eik,kr->eir',getattr(response,name+'_modes'),Q)
            for name in ['displacement','strain','stress']}
    reduced=replace(response,factors=factors,eigenvalues=lam,metadata=meta,**arrays)
    meta['original_direct_solve_scaled_energy_error']=meta.pop('direct_solve_scaled_energy_error',None)
    deviations=[]
    for d in [0.,.63]:
        exact=response.hessian(d);delta=reduced.hessian(d)-exact;normalizer=np.sqrt(np.maximum(abs(exact.diagonal()),1e-30))
        deviations.append(float(linalg.norm(delta/normalizer[:,None]/normalizer[None,:])))
    meta['compression_scaled_energy_matrix_error']=max(deviations)
    volume=response.mesh.volumes/response.mesh.rve_volume
    S=response.stress_modes
    certificate={'residual_root':root,'input_scale':scale,'rho_max':float(rho.max()),
        'force_column_norms':linalg.norm(F,axis=0),
        'stress_L2_lipschitz':float(np.sqrt(np.einsum('eik,eik,e->',S,S,volume))),
        'stress_point_lipschitz':float(np.max(np.linalg.norm(S,axis=(1,2))))}
    reduced.certificate_arrays=certificate
    return reduced


def cohesive_certificate(cohesive, arrays, q, damage):
    """Residual bounds in exact arithmetic, with a declared roundoff allowance.

    r=Fq-A Q z, e=A^-1 r, DeltaW=.5 r^T A^-1 r. The stress
    bound uses Frobenius operator norms and Weyl's eigenvalue inequality.
    """
    if not np.isfinite(damage) or not 0<=damage<=1:raise ValueError('Invalid damage')
    q=np.asarray(q,float)
    if q.shape!=(cohesive.factors.shape[1],) or not np.isfinite(q).all():raise ValueError('Finite q of kernel dimension required')
    s=cohesive.residual+(1-cohesive.residual)*(1-damage)**2-1
    z=cohesive.factors@q/(1+s*cohesive.eigenvalues)
    coefficients=np.r_[q*arrays['input_scale'],-z,-s*z]
    vector=arrays['residual_root']@coefficients
    rounding=64*np.finfo(float).eps*float(np.linalg.norm(arrays['residual_root'],2))*float(np.linalg.norm(coefficients))
    norm=float(np.linalg.norm(vector))+rounding;a=1+s*float(arrays['rho_max'])
    if a<=0:raise ValueError('No positive coercivity lower bound')
    return {'residual_norm':norm,'coercivity_lower_bound':a,'energy_error_upper_J_m3':.5*norm**2/a,
        'generalized_force_absolute_upper':np.asarray(arrays['force_column_norms'])*norm/a,
        'stress_L2_upper_Pa':float(arrays['stress_L2_lipschitz'])*norm/a,
        'stress_point_upper_Pa':float(arrays['stress_point_lipschitz'])*norm/a,
        'scope':'same discrete q and d; excludes concentration truncation and trajectory error',
        'roundoff_allowance':rounding}
