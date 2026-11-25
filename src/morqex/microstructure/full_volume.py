"""Whole-volume geometry features; no selection of favorable small crops."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import ndimage
import tifffile

def periodic_components(mask):
    labels,count=ndimage.label(mask,ndimage.generate_binary_structure(3,1));parent=np.arange(count+1)
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a in range(3):
        left=np.take(labels,0,axis=a).ravel();right=np.take(labels,-1,axis=a).ravel();valid=(left>0)&(right>0)
        for i,j in np.unique(np.column_stack([left[valid],right[valid]]),axis=0):
            ri,rj=root(int(i)),root(int(j))
            if ri!=rj:parent[max(ri,rj)]=min(ri,rj)
    representatives=np.array([root(i) for i in range(count+1)]);unique=np.unique(representatives);mapping=np.searchsorted(unique,representatives).astype(np.int32)
    labels[:]=mapping[labels];return labels,len(unique)-1

def periodic_distance(mask,*,halo=32,slab=24):
    """Exact EDT when the returned interior distances stay below halo.

    Each slab includes a periodic halo on all axes. The explicit distance
    check certifies that an omitted image cannot be the nearest background.
    """
    if not (~mask).any():raise ValueError('Distance to surface undefined for an entirely active periodic volume')
    out=np.empty(mask.shape,np.float32);ny,nx=mask.shape[1:]
    iy=np.arange(-halo,ny+halo)%ny;ix=np.arange(-halo,nx+halo)%nx
    for z in range(0,mask.shape[0],slab):
        end=min(z+slab,mask.shape[0]);iz=np.arange(z-halo,end+halo)%mask.shape[0]
        padded=mask[np.ix_(iz,iy,ix)];distance=ndimage.distance_transform_edt(padded)
        interior=distance[halo:halo+end-z,halo:halo+ny,halo:halo+nx]
        if interior.max()>=halo:raise ValueError('EDT halo insufficient; increase halo with a smaller slab under the memory budget')
        out[z:end]=interior
        del padded,distance,interior
    return out

def prepare_full_volume(source,destination,*,halo=32):
    started=time.perf_counter();path=Path(source['path']);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Raw volume checksum mismatch')
    phase=np.ascontiguousarray(tifffile.imread(path).transpose(source['axes_to_zyx']),dtype=np.uint8)
    if not set(np.unique(phase))<={0,1,2}:raise ValueError('Declare the phase map for non-NLR volumes explicitly')
    active=phase==1;components,count=periodic_components(active);counts=np.bincount(components.ravel());counts[0]=0
    distance=periodic_distance(active,halo=halo)
    maxima=np.zeros(count+1,np.float32);np.maximum.at(maxima,components.ravel(),distance.ravel());maxima[0]=1.
    depth=distance/np.maximum(maxima[components],1.);del distance,maxima
    faces=np.zeros(phase.shape,np.uint8)
    for axis in range(3):
        for sign in [-1,1]:faces+=active&(~np.roll(active,sign,axis))
    areas=np.bincount(components.ravel(),weights=faces.ravel(),minlength=count+1);areas[0]=0
    exposure=np.divide(areas,counts,out=np.zeros_like(areas),where=counts>0)
    values=np.log(np.maximum(exposure[components],1e-8));lo,hi=np.quantile(values[active],[.01,.99]);exposure_feature=np.clip((values-lo)/max(hi-lo,1e-8),0,1).astype(np.float32);del values
    # Cyclic angle gives a continuous directional feature on the periodic
    # domain and does not invent a center for particles crossing its boundary.
    directional=np.broadcast_to((.5+.5*np.cos(2*np.pi*(np.arange(phase.shape[2])+.5)/phase.shape[2]))[None,None,:],phase.shape).astype(np.float32)
    features=np.stack([depth,exposure_feature,directional]);del depth,exposure_feature,directional
    features[:,~active]=0
    np.savez_compressed(destination/'geometry.npz',phase=phase,components=components,faces=faces,features=features)
    record={'source':source,'shape':list(phase.shape),'voxels':int(phase.size),'phase_fractions':{str(k):float(np.mean(phase==k)) for k in [0,1,2]},
        'active_components':count,'voxel_size_m':source['voxel_size_m'],'reaction_area_density_m_inv':float(faces.sum()/phase.size/source['voxel_size_m']),
        'feature_names':['periodic surface depth normalized within connected particle','log particle reactive area/volume','periodic through-plane direction'],
        'EDT_halo_voxels':halo,'EDT_interior_guard':'distance < halo in every slab','connectivity':'periodic six-face active connectivity',
        'sha256':hashlib.sha256((destination/'geometry.npz').read_bytes()).hexdigest(),'seconds':time.perf_counter()-started,
        'scope':'Every input voxel retained; surface reaction assumes microporous CBD ionic access. Accessibility and transport are audited separately.'}
    (destination/'manifest.json').write_text(json.dumps(record,indent=2));return record

def positive_basis(features,*,orders=(2,2,2),xp=np):
    """Tensor Bernstein partition of unity; eight nodes for (2,2,2).

    Modes represent population/geometry feature nodes, not eight exact particles.
    Nonnegative rows preserve pointwise concentration bounds and total mass.
    """
    import math
    if len(orders)!=3 or min(orders)<1:raise ValueError('Three positive feature orders required')
    axes=[]
    for axis,order in enumerate(orders):
        f=features[axis];axes.append([math.comb(order-1,j)*f**j*(1-f)**(order-1-j) for j in range(order)])
    return xp.stack([a*b*c for a in axes[0] for b in axes[1] for c in axes[2]],axis=-1)
