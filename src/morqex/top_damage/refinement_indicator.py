"""Residual-based marking for static coupled P1 mechanics/AT2.

Face jumps and damage cell residuals have energy units. They are refinement
indicators, NOT certified nonlinear error bounds or a fatigue-life estimator.
Only zero-traction pores and outer-box KUBC, natural damage BC are supported.
"""
import numpy as np


def coupled_indicator(mesh,first_piola,damage,tensile_energy,young,Gc,length,residual_stiffness):
    ne=len(mesh.cells);P=np.asarray(first_piola,float);d=np.asarray(damage,float)
    E=np.broadcast_to(young,(ne,));gc=np.broadcast_to(Gc,(ne,));ell=np.broadcast_to(length,(ne,))
    psi=np.broadcast_to(tensile_energy,(ne,))
    if P.shape!=(ne,3,3) or d.shape!=(len(mesh.points),) or not all(np.isfinite(x).all() for x in (P,d,E,gc,ell,psi)):
        raise ValueError('Finite compatible fields required')
    if min(E.min(),gc.min(),ell.min())<=0 or np.any((d<0)|(d>1)) or np.any(psi<0):raise ValueError('Invalid physical fields')
    if not 0<residual_stiffness<1:raise ValueError('Invalid residual stiffness')
    local=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    faces=mesh.cells[:,local].reshape(-1,3)
    _,first,inverse,counts=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_inverse=True,return_counts=True)
    if counts.max()>2:raise ValueError('Nonmanifold face')
    xyz=mesh.points[faces];cross=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    area=np.linalg.norm(cross,axis=1)/2;normal=cross/(2*area[:,None]);owner=np.repeat(np.arange(ne),4)
    tractions=np.einsum('fij,fj->fi',P[owner],normal)
    jump=np.zeros((len(first),3));np.add.at(jump,inverse,tractions)
    # Exclude prescribed-displacement exterior faces, not free internal pores.
    tol=np.max(mesh.box[1]-mesh.box[0])*1e-10
    outer=np.any(np.all(np.abs(xyz-mesh.box[0])<tol,axis=1)|np.all(np.abs(xyz-mesh.box[1])<tol,axis=1),axis=1)
    fixed=(counts==1)&outer[first];jump[fixed]=0
    h=(6*mesh.volumes)**(1/3)
    compliance=np.zeros(len(first));np.maximum.at(compliance,inverse,h[owner]/E[owner])
    mechanical_face=area[first]*compliance*np.sum(jump**2,axis=1)
    mechanical=np.bincount(owner,weights=mechanical_face[inverse]/counts[inverse],minlength=ne)
    grad=np.einsum('ei,eij->ej',d[mesh.cells],mesh.gradients)
    flux=np.sum((gc*ell)[owner,None]*grad[owner]*normal,axis=1)
    jump_d=np.bincount(inverse,weights=flux,minlength=len(first))
    mobility=np.zeros(len(first));np.maximum.at(mobility,inverse,h[owner]/(gc*ell)[owner])
    damage_face=area[first]*mobility*jump_d**2
    phase=np.bincount(owner,weights=damage_face[inverse]/counts[inverse],minlength=ne)
    dm=d[mesh.cells].mean(axis=1)
    r=2*(1-residual_stiffness)*psi*(1-dm)-gc/ell*dm
    # This static marking does not certify active-set inequality residuals.
    phase+=mesh.volumes*h*h/(gc*ell)*r*r
    return {'mechanical_J':mechanical,'phase_field_J':phase,'total_J':mechanical+phase}


def mark_bulk(indicator,fraction=.5):
    a=np.asarray(indicator,float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or np.any(a<0) or not 0<fraction<=1:
        raise ValueError('Finite nonnegative indicators and fraction in (0,1] required')
    if a.sum()==0:return np.array([],dtype=np.int64)
    order=np.argsort(-a,kind='stable');n=np.searchsorted(np.cumsum(a[order]),fraction*a.sum())+1
    return np.sort(order[:n])


def quadratic_indicator(geometry,first_piola,damage,tensile_energy,young,Gc,length,residual_stiffness):
    """Cell-average jumps plus unresolved within-cell P2 response variation.

    This is a dimensionally consistent marking heuristic, not a P2 residual
    estimator or certified error bound. Variation terms prevent cancellation
    when the cell mean hides a nonconstant quadratic-displacement response.
    Spatial qualification still requires independent mesh comparisons.
    """
    g=geometry;mesh=g.base;P=np.asarray(first_piola,float);psi=np.asarray(tensile_energy,float)
    if P.shape!=(g.quadrature_count,3,3) or psi.shape!=(g.quadrature_count,) or not np.isfinite(P).all() or not np.isfinite(psi).all() or np.any(psi<0):
        raise ValueError('Finite compatible quadrature response required')
    meanP=g.quadrature_average(P);meanpsi=g.quadrature_average(psi)
    result=coupled_indicator(mesh,meanP,damage,meanpsi,young,Gc,length,residual_stiffness)
    delta=P.reshape(len(mesh.cells),g.nq,9)-meanP.reshape(-1,1,9)
    pvar=np.einsum('q,eqi,eqi->e',g.weights,delta,delta)
    svar=np.einsum('q,eq,eq->e',g.weights,psi.reshape(-1,g.nq)-meanpsi[:,None],psi.reshape(-1,g.nq)-meanpsi[:,None])
    E=np.broadcast_to(young,(len(mesh.cells),));gc=np.broadcast_to(Gc,E.shape);ell=np.broadcast_to(length,E.shape)
    result['mechanical_variation_J']=mesh.volumes*pvar/E
    # Reaction-source variation scaled by the AT2 local reaction coefficient.
    result['phase_source_variation_J']=mesh.volumes*(2*(1-residual_stiffness))**2*svar/(gc/ell+2*(1-residual_stiffness)*meanpsi)
    result['total_J']=result['total_J']+result['mechanical_variation_J']+result['phase_source_variation_J']
    return result
