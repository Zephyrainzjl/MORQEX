"""Bounded multiphase volume projection on one shared surface complex.

This is a geometric constraint solve, not a claim of pointwise CT equivalence.
No vertices/faces/material labels are deleted or reclassified.
"""
import numpy as np


def conserve_phase_volumes(plc,model,*,relative_tolerance=1e-6,maximum_iterations=12,
                           maximum_displacement_voxels=.35):
    from .adaptive import _region_volumes
    phases=list(plc.seeds);pairs=plc.metadata['interface_pairs'];faces=np.asarray(plc.facets)
    original=plc.points.copy();low=np.asarray(model.origin)[::-1]
    extent=np.asarray(model.regions.shape)[::-1]*np.asarray(model.spacing)[::-1]
    scale=float(max(extent));p=(original-low)/scale;reference=p.mean(0);h=min(model.spacing)/scale
    target={k:float((model.regions==k).sum()*np.prod(model.spacing)/scale**3) for k in phases}
    before=_region_volumes(p,faces,plc.markers,pairs,phases)
    chosen=phases[1:] if set(phases)==set(map(int,np.unique(model.regions))) else phases
    pins=np.zeros(p.shape,bool)
    for axis,n in enumerate(extent/scale):pins[:,axis]=np.isclose(p[:,axis],0,atol=1e-8*h,rtol=0)|np.isclose(p[:,axis],n,atol=1e-8*h,rtol=0)
    nearest=np.full(len(p),np.inf)
    for start in range(0,len(faces),50000):
        f=faces[start:start+50000];x=p[f]
        lengths=np.minimum.reduce([np.linalg.norm(x[:,a]-x[:,b],axis=1) for a,b in ((0,1),(1,2),(2,0))])
        np.minimum.at(nearest,f.ravel(),np.repeat(lengths,3))
    nearest[~np.isfinite(nearest)]=0
    mobility=np.minimum(1.,nearest/h)**2
    logs=[];current=before;frozen_vertices=0
    for iteration in range(maximum_iterations):
        error=max(abs(current[k]/target[k]-1) for k in phases)
        logs.append(float(error))
        if error<=relative_tolerance:break
        gradient=np.zeros((len(chosen),len(p),3),float)
        for start in range(0,len(faces),50000):
            f=faces[start:start+50000];x=p[f]-reference;m=plc.markers[start:start+50000]
            v=np.stack((np.cross(x[:,1],x[:,2]),np.cross(x[:,2],x[:,0]),np.cross(x[:,0],x[:,1])),axis=1)/6
            for j,k in enumerate(chosen):
                signs=np.zeros(len(f))
                for marker,pair in pairs.items():
                    if k in pair:signs[m==int(marker)]=1 if k==pair[0] else -1
                np.add.at(gradient[j],f.ravel(),(v*signs[:,None,None]).reshape(-1,3))
        gradient[:,pins]=0
        flat=gradient.reshape(len(chosen),-1);weighted=(gradient*mobility[None,:,None]).reshape(len(chosen),-1)
        gram=flat@weighted.T
        rhs=np.array([target[k]-current[k] for k in chosen])
        coefficients=np.linalg.lstsq(gram,rhs,rcond=1e-12)[0]
        move=np.einsum('k,kij->ij',coefficients,gradient)*mobility[:,None]
        norm=np.linalg.norm(move,axis=1);local=.12*nearest
        move*=np.minimum(1.,local/np.maximum(norm,1e-300))[:,None]
        accepted=False;blocked=[]
        for reduction in range(12):
            trial=p+move*(.5**reduction)
            if np.linalg.norm(trial-(original-low)/scale,axis=1).max()>maximum_displacement_voxels*h:continue
            valid=True
            for start in range(0,len(faces),50000):
                f=faces[start:start+50000];a=p[f];b=trial[f]
                n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);n1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
                invalid=np.einsum('ij,ij->i',n0,n1)<=0
                if invalid.any():valid=False;blocked.append(np.unique(f[invalid]));break
            if not valid:continue
            volume=_region_volumes(trial,faces,plc.markers,pairs,phases)
            if max(abs(volume[k]/target[k]-1) for k in phases)>=error:continue
            p=trial;current=volume;accepted=True;break
        if not accepted:
            if blocked:
                ids=np.unique(np.concatenate(blocked));previous=int(pins.sum());pins[ids]=True
                frozen_vertices+=len(ids)
                if int(pins.sum())>previous:continue
            raise ArithmeticError(f'Bounded phase-volume projection cannot improve without an inverted surface triangle: {logs}')
    else:raise ArithmeticError(f'Phase-volume projection did not converge: {logs}')
    plc.points=p*scale+low
    plc.metadata['volume_projection']={'relative_tolerance':relative_tolerance,'iterations':iteration,
        'error_history':logs,'maximum_displacement_voxels':float(np.linalg.norm(plc.points-original,axis=1).max()/min(model.spacing)),
        'locally_frozen_vertices':frozen_vertices,
        'initial_relative_errors':{str(k):float(before[k]/target[k]-1) for k in phases},
        'final_relative_errors':{str(k):float(current[k]/target[k]-1) for k in phases},
        'source_labels_changed':False,'scope':'global geometric volumes; no pointwise geometry or topology-equivalence certificate'}
    return plc
