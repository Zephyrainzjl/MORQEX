"""Two-sided displacement extrapolation on explicitly supplied crack patches.

Prior art: Babaoğlu, Kaynar & Göktepe, CMAME 458 (2026) 119016,
DOI 10.1016/j.cma.2026.119016 (two-sided displacement fitted planes).
This independent 3D local-affine implementation is a candidate extension, not
that paper's full crack-path algorithm or an automatically qualified aperture.
It does not infer wetting, contact traction, crack area or transport from d.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .contracts import _owned


@dataclass(frozen=True)
class OpeningResult:
    jump_m: np.ndarray
    current_normals: np.ndarray
    signed_gap_m: np.ndarray
    aperture_m: np.ndarray
    slip_m: np.ndarray
    fit_rms_m: np.ndarray
    support_counts: np.ndarray
    condition_numbers: np.ndarray
    usable: np.ndarray
    reasons: tuple
    scope: str = 'Local fit diagnostic on supplied patches; not a contact or wetting solution.'

    def require_usable(self):
        if not np.all(self.usable):
            raise ValueError('Unresolved crack opening: '+', '.join(sorted(set(r for r in self.reasons if r))))
        return self


def reconstruct_opening(points_m, displacement_m, centres_m, normals, *,
                        inner_distance_m, outer_distance_m, tangential_radius_m,
                        sample_damage=None, maximum_sample_damage=.2,
                        weights=None, minimum_support=8, maximum_condition=1e5,
                        maximum_normal_angle_deg=30., maximum_fit_rms_m=None):
    """Fit u=A(X-Xc)+b independently on both sides; jump=b+ - b-.

    The normal averages the two normalized cof(Fside)*N push-forwards.
    Thus a common affine deformation gives zero opening and a superposed rigid
    rotation rotates the jump and normal together. Positive aperture is a
    diagnostic only: negative gap is retained to expose missing contact.
    Samples must already belong to the relevant material neighbourhood; caller
    must exclude other cracks/materials/particles. Ambiguous/insufficient fits
    return NaNs and usable=False, never an invented closed crack.
    """
    x=np.asarray(points_m,float);u=np.asarray(displacement_m,float)
    centres=np.asarray(centres_m,float);N=np.asarray(normals,float)
    if x.ndim!=2 or x.shape[1]!=3 or u.shape!=x.shape or len(x)<4 or not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError('Finite compatible reference points and displacements required')
    if centres.ndim!=2 or centres.shape[1]!=3 or N.shape!=centres.shape or not len(centres) or not np.isfinite(centres).all() or not np.isfinite(N).all():
        raise ValueError('Finite oriented reference crack patches required')
    lengths=np.linalg.norm(N,axis=1)
    if np.any(lengths<=0):raise ValueError('Nonzero oriented normals required')
    N=N/lengths[:,None]
    inner=float(inner_distance_m);outer=float(outer_distance_m);radius=float(tangential_radius_m)
    if not np.isfinite([inner,outer,radius,maximum_condition,maximum_normal_angle_deg]).all() or not 0<=inner<outer or radius<=0:
        raise ValueError('Finite nested physical support distances required')
    if type(minimum_support) is not int or minimum_support<4 or maximum_condition<=1 or not 0<maximum_normal_angle_deg<90:
        raise ValueError('Valid rank/support/normal agreement controls required')
    if not np.isfinite(maximum_sample_damage) or not 0<=maximum_sample_damage<1:raise ValueError('Valid intact sample threshold required')
    d=np.zeros(len(x)) if sample_damage is None else np.asarray(sample_damage,float)
    w=np.ones(len(x)) if weights is None else np.asarray(weights,float)
    if d.shape!=(len(x),) or w.shape!=d.shape or not np.isfinite(d).all() or not np.isfinite(w).all() or np.any((d<0)|(d>1)) or np.any(w<=0):
        raise ValueError('Bounded damage and positive sample weights required')
    if maximum_fit_rms_m is not None and (not np.isfinite(maximum_fit_rms_m) or maximum_fit_rms_m<=0):raise ValueError('Positive fit tolerance required')
    n=len(centres);jump=np.full((n,3),np.nan);normal=jump.copy();gap=np.full(n,np.nan)
    slip=jump.copy();rms=np.full((n,2),np.nan);count=np.zeros((n,2),np.int64);condition=np.full((n,2),np.inf)
    usable=np.zeros(n,bool);reasons=[];tree=cKDTree(x);scale=max(outer,radius)
    for p,(centre,refnormal) in enumerate(zip(centres,N)):
        ids=np.asarray(tree.query_ball_point(centre,np.hypot(outer,radius)),np.int64)
        local=x[ids]-centre;distance=local@refnormal
        tangent=np.linalg.norm(local-distance[:,None]*refnormal,axis=1)
        fits=[];reason=''
        for side in (0,1):
            sign=-1 if side==0 else 1
            mask=(sign*distance>inner)&(sign*distance<=outer)&(tangent<=radius)&(d[ids]<=maximum_sample_damage)
            chosen=ids[mask];count[p,side]=len(chosen)
            if len(chosen)<minimum_support:reason='insufficient intact two-sided support';break
            A=np.column_stack([np.ones(len(chosen)),local[mask]/scale])
            # Scale weights to avoid conditioning changes from physical units.
            sw=np.sqrt(w[chosen]/np.mean(w[chosen]));Aw=A*sw[:,None]
            # Remove an arbitrary common translation before fitting.
            reference=u[chosen[0]]
            coefficients,_,rank,singular=np.linalg.lstsq(Aw,(u[chosen]-reference)*sw[:,None],rcond=None)
            condition[p,side]=singular[0]/singular[-1] if singular[-1]>0 else np.inf
            if rank<4 or condition[p,side]>maximum_condition:reason='rank deficient or ill-conditioned support';break
            residual=A@coefficients-(u[chosen]-reference)
            rms[p,side]=np.sqrt(np.sum(w[chosen,None]*residual**2)/np.sum(w[chosen]))
            coefficients[0]+=reference
            smoothF=np.eye(3)+coefficients[1:].T/scale
            if np.linalg.det(smoothF)<=0:reason='inverted smooth-side deformation';break
            fits.append((coefficients[0],smoothF))
        if not reason:
            cofactor_normals=[np.linalg.solve(f[1].T,refnormal) for f in fits]
            cofactor_normals=[v/np.linalg.norm(v) for v in cofactor_normals]
            if cofactor_normals[0]@cofactor_normals[1]<np.cos(np.deg2rad(maximum_normal_angle_deg)):
                reason='two sides do not define a common local crack normal'
            elif maximum_fit_rms_m is not None and np.max(rms[p])>maximum_fit_rms_m:
                reason='local affine fit residual exceeds physical tolerance'
            else:
                v=cofactor_normals[0]+cofactor_normals[1];v/=np.linalg.norm(v)
                jump[p]=fits[1][0]-fits[0][0];normal[p]=v;gap[p]=jump[p]@v
                slip[p]=jump[p]-gap[p]*v;usable[p]=True
        reasons.append(reason)
    return OpeningResult(*[_owned(a) for a in (jump,normal,gap,np.maximum(gap,0),slip,rms,count,condition,usable)],tuple(reasons))
