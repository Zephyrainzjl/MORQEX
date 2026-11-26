"""Sparse multi-load reduced-basis cohesive mechanics, qualified against FEM.

No dense eigenproblem in the original displacement dimension. The Galerkin
resolvent has positive residues, so stress, chemical forces and damage energy
remain derivatives of one energy. Qualification is for the specified discrete
loads/damage interval; continuum, concentration and trajectory errors are separate.
"""
import time
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import splu
from .cohesive import assemble_cohesive,CohesiveResponse


def build_sparse_cohesive(electrode,P,*,normal_stiffness=1e15,tangent_stiffness=5e14,
                          residual=1e-3,maximum_damage=.85,training_points=4,
                          maximum_modes=120,tolerance=1e-5,validation_points=(.17,.43,.71),
                          retain_system=False):
    """Build and qualify a scalar-interface damage response with sparse solves.

    Scale input columns by their direct-energy norms before snapshot selection.
    Reuse each sparse factor for all load columns. Held-out full FEM solves test
    force matrices and all stress columns, with an absolute input metric for
    nearly stress-free loads. A failed tolerance raises, never silently qualifies.
    """
    if not 0<maximum_damage<=1 or training_points<2 or maximum_modes<1 or not 0<tolerance<1:
        raise ValueError('Invalid sparse cohesive budget')
    if any(not 0<=d<=maximum_damage for d in validation_points):raise ValueError('Validation damage outside trained interval')
    started=time.perf_counter()
    a=assemble_cohesive(electrode,P,normal_stiffness=normal_stiffness,tangent_stiffness=tangent_stiffness,residual=residual)
    mesh=a['mesh'];bulk=a['bulk'];J=a['projection'];B=a['B'];V=mesh.rve_volume
    K0=(J.T@(bulk.K+a['C'])@J).tocsc();C=(J.T@a['C']@J).tocsc()
    scale=float(K0.diagonal().max());K0=K0/scale;C=C/scale;B=B/scale
    # K0 and C here share one normalization; energy restores scale/V.
    factor=splu(K0);nq=B.shape[1]
    input_scale=np.sqrt(np.maximum(a['direct'].diagonal(),1e-30))
    training=np.linspace(0,maximum_damage,training_points);samples=[]
    for d in training:
        g=residual+(1-residual)*(1-d)**2
        samples.append((factor if d==0 else splu(K0+(g-1)*C)).solve(B)/input_scale)
    X=np.column_stack(samples)
    # Metric whitening through the small Gram matrix; two passes remove weak
    # snapshots without allocating a dense original stiffness matrix.
    gram=X.T@(K0@X);lam,R=linalg.eigh((gram+gram.T)/2)
    keep=lam>max(lam.max()*1e-13,np.finfo(float).tiny)
    if np.count_nonzero(keep)>maximum_modes:raise ValueError('Sparse basis exceeds mode budget; increase maximum_modes explicitly')
    U=(X@R[:,keep])/np.sqrt(lam[keep]);del X,samples
    A=U.T@(K0@U);Cr=U.T@(C@U);Br=U.T@B
    L=linalg.cholesky((A+A.T)/2,lower=True)
    T=linalg.solve_triangular(L,Cr,lower=True)
    T=linalg.solve_triangular(L,T.T,lower=True).T
    rho,R=linalg.eigh((T+T.T)/2)
    if rho.min()<-1e-9 or rho.max()>1+1e-9:raise ValueError('Nonphysical projected cohesive spectrum')
    rho=np.clip(rho,0,1);Q=linalg.solve_triangular(L.T,R,lower=False)
    F=Q.T@Br*np.sqrt(scale/V);modes=np.asarray(J@(U@Q))*np.sqrt(V/scale)
    affine=a['affine'];eigen=a['eigen'];ne=len(mesh.cells)
    strain0=np.asarray(bulk.D@affine).reshape(ne,6,nq)
    strain_modes=np.asarray(bulk.D@modes).reshape(ne,6,len(rho))
    response=CohesiveResponse((a['direct']+a['direct'].T)/2,F,rho,
        affine.reshape(-1,3,nq),modes.reshape(-1,3,len(rho)),strain0,strain_modes,
        np.einsum('eij,ejk->eik',bulk.stiffness,strain0-eigen),
        np.einsum('eij,ejk->eik',bulk.stiffness,strain_modes),mesh,residual,
        a['metadata']|{'method':'sparse FEM multi-right-hand-side reduced basis / Galerkin cohesive resolvent',
            'spring_modes':len(rho),'training_damage':training.tolist(),'maximum_damage':maximum_damage,
            'qualification_tolerance':tolerance,'input_metric':'direct energy diagonal; stress columns absolute under this input scale'})
    response.interface_pairs=a['interface_pairs'];response.interface_areas=a['interface_areas']
    # Stable residual dual-norm root for all q,d (K(d)>=g K0). Unlike a
    # subtraction of large quadratic forms, QR retains near-zero residuals.
    AinvB=linalg.solve(A,Br,assume_a='pos');AinvC=linalg.solve(A,Cr,assume_a='pos')
    R0=B-K0@(U@AinvB);R1=C@U-K0@(U@AinvC)
    res_basis,root=linalg.qr(np.column_stack([R0,R1]),mode='economic')
    dual=res_basis.T@factor.solve(res_basis)
    dual=(dual+dual.T)/2;ev,vec=linalg.eigh(dual)
    residual_root=(np.sqrt(np.maximum(ev,0))[:,None]*vec.T)@root*np.sqrt(scale/V)
    response.sparse_certificate={'root':residual_root,'projected_A':A,'projected_C':Cr,'projected_B':Br,
        'scope':'fixed discrete concentration load and damage; excludes mesh, transport truncation and trajectory error'}
    validation=[]
    for d in sorted(set([0.,maximum_damage,*validation_points])):
        g=response.retention(d);K=K0+(g-1)*C
        exact=(factor if d==0 else splu(K)).solve(B)
        H=a['direct']-B.T@exact*scale/V
        error=float(linalg.norm((response.hessian(d)-H)/input_scale[:,None]/input_scale[None,:],2))
        reduced=U@linalg.solve(A+(g-1)*Cr,Br,assume_a='pos')
        # Energy norm controls the strain/stress energy, including springs.
        delta=(reduced-exact)/input_scale;energy_error=float(np.sqrt(max(0,linalg.eigvalsh(delta.T@(K@delta)).max())*scale/V))
        direct_residual=float(linalg.norm(K@exact-B)/max(linalg.norm(B),1e-300))
        validation.append({'damage':float(d),'scaled_force_matrix_error':error,
            'absolute_energy_norm_per_unit_input':energy_error,'direct_equilibrium_residual':direct_residual})
    response.metadata.update(validation=validation,offline_seconds=time.perf_counter()-started,
        original_DOFs=K0.shape[0],factorizations=training_points+len(set([maximum_damage,*validation_points])-{0.}),
        certificate='residual dual norm with coercivity g(d); fixed discrete input/damage only')
    if max(v['scaled_force_matrix_error'] for v in validation)>tolerance:
        raise ValueError('Sparse cohesive basis failed held-out FEM force tolerance; enrich training points or mode budget')
    if retain_system:response.reference_system=dict(K0=K0,C=C,B=B,U=U,scale=scale,volume=V)
    return response


def sparse_cohesive_certificate(response,q,damage):
    """Coercivity energy bound, for the same discrete q,d in exact arithmetic."""
    q=np.asarray(q,float)
    if q.shape!=(len(response.direct),) or not np.isfinite(q).all() or not 0<=damage<=1:raise ValueError('Finite input and damage in [0,1] required')
    c=response.sparse_certificate;g=float(response.retention(damage))
    z=linalg.solve(c['projected_A']+(g-1)*c['projected_C'],c['projected_B']@q,assume_a='pos')
    coeff=np.r_[q,(1-g)*z];r=c['root']@coeff
    rounding=64*np.finfo(float).eps*linalg.norm(c['root'],2)*linalg.norm(coeff)
    norm=linalg.norm(r)+rounding
    return {'energy_error_upper_J_m3':float(.5*norm**2/g),'dual_residual_norm':float(norm),
        'coercivity_lower_bound':g,'roundoff_allowance':float(rounding),'scope':c['scope']}


def build_sparse_operator(electrode,target=16,**options):
    """Construct the conservative transport and cohesive energy in one pass."""
    from scipy import sparse
    from .electrochemistry.microstructure_operator import MicrostructureOperator
    labels=electrode.partition_labels(target);P=np.eye(labels.max()+1)[labels]
    response=build_sparse_cohesive(electrode,P,**options)
    fv=electrode.diffusion;weights=P.T@fv.weights;R=(P.T*fv.weights)/weights[:,None]
    maps={name:getattr(response,name+'_direct')+getattr(response,name+'_modes')@response.factors for name in ['stress','strain','displacement']}
    op=MicrostructureOperator(electrode,labels,P,R,weights,P.T@electrode.area,
        sparse.csr_matrix(P.T@fv.stiffness@P),response.hessian(0.),maps['stress'],maps['strain'],maps['displacement'],
        'connected geometric partition / sparse cohesive reduced basis')
    return op,response
