"""Chunked conservative POU transport projection of a complete voxel volume.

No dense N-by-m field is stored. Each active component owns a small local
Bernstein block; zero flux across different components is enforced by topology.
The stiffness is the Galerkin projection of the periodic face finite-volume
Laplacian, while nonlinear chemical energy uses declared lumped quadrature.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .positive import ComponentBasis,full_features

def project_volume(directory,*,extra_nodes=8,chunk=32768):
    started=time.perf_counter();directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if hashlib.sha256((directory/'geometry.npz').read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Geometry checksum mismatch')
    with np.load(directory/'geometry.npz') as a:
        components=a['components'];features=full_features(a);faces=a['faces']
    count=int(components.max())
    if not isinstance(extra_nodes,int) or extra_nodes<0:raise ValueError('Nonnegative number of additional concentration nodes required')
    if count+extra_nodes>2048:raise MemoryError('Dense offline concentration prototype capped at 2048 modes; supply a sparse/block projection backend for larger independent inventories')
    basis=ComponentBasis.allocate(components,features,target=count+extra_nodes)
    m=basis.dimension;w=np.zeros(m);area=np.zeros(m);K=np.zeros((m,m));N=components.size;h=float(manifest['voxel_size_m']);shape=components.shape
    import math
    def local(f,orders):
        axes=[]
        for axis,order in enumerate(orders):
            degree=int(order)-1
            axes.append([math.comb(degree,j)*f[:,axis]**j*(1-f[:,axis])**(degree-j) for j in range(degree+1)])
        return np.stack([a*b*c for a in axes[0] for b in axes[1] for c in axes[2]],axis=1)
    flat=components.ravel();feature=features.reshape(3,-1);stride=np.array([shape[1]*shape[2],shape[2],1]);sizes=np.asarray(shape)
    for begin in range(0,N,chunk):
        ids=np.arange(begin,min(N,begin+chunk));ids=ids[flat[ids]>0]
        if not len(ids):continue
        coord=(ids[:,None]//stride)%sizes
        for component in np.unique(flat[ids]):
            selected=flat[ids]==component;i=ids[selected];xyz=coord[selected];start,end=basis.offsets[component-1:component+1]
            P=local(feature[:,i].T.astype(float),basis.orders[component]);w[start:end]+=P.sum(axis=0)/N
            area[start:end]+=P.T@faces.ravel()[i]/(N*h)
            for axis in range(3):
                j=i+np.where(xyz[:,axis]==sizes[axis]-1,-(sizes[axis]-1),1)*stride[axis]
                keep=flat[j]==component
                if not keep.any():continue
                Q=local(feature[:,j[keep]].T.astype(float),basis.orders[component])-P[keep]
                K[start:end,start:end]+=Q.T@Q/(N*h*h)
    if np.any(w<=1e-16):raise ValueError('Degenerate POU node: change the feature basis instead of retaining zero mass')
    scale=max(float(np.linalg.norm(K)),1.)
    if np.linalg.norm(K@np.ones(m))>1e-12*scale or np.linalg.eigvalsh(K).min()<-1e-12*scale:raise RuntimeError('Projected diffusion lost conservation or positive semidefiniteness')
    result={'basis':basis.metadata(),'geometry':manifest,'seconds':time.perf_counter()-started,'dimension':m,
        'scope':'Complete geometry; periodic active diffusion, microporous CBD reaction access, lumped nonlinear chemical quadrature; mechanics is a separate qualified operator'}
    np.savez_compressed(directory/'concentration.npz',weights=w,reaction_area=area,stiffness=K)
    result['sha256']=hashlib.sha256((directory/'concentration.npz').read_bytes()).hexdigest()
    (directory/'concentration.json').write_text(json.dumps(result,indent=2));return result
