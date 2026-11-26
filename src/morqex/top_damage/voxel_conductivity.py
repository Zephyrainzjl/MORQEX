"""Scalar finite-volume transport on the full material lattice.

Prescribed scalar local conductivity is a constitutive input, not inferred from
damage. True zero phases are excluded and floating components are gauged, never
connected with an artificial small background conductivity.
"""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg


def effective_conductivity(field,spacing_zyx_m,*,axis=2,tolerance=1e-10):
    k=np.asarray(field,float);h=np.asarray(spacing_zyx_m,float)
    if k.ndim!=3 or min(k.shape)<2 or not np.isfinite(k).all() or np.any(k<0):raise ValueError('Finite nonnegative 3D scalar conductivity required')
    if h.shape!=(3,) or not np.isfinite(h).all() or min(h)<=0 or axis not in (0,1,2):raise ValueError('Positive spacing and ZYX axis required')
    ids=np.flatnonzero(k>0);n=len(ids);index=np.full(k.shape,-1,dtype=np.int64);index.ravel()[ids]=np.arange(n)
    if n==0:return {'effective':0.,'residual':0.,'energy_relative_defect':0.,'components':0,'potential':np.zeros_like(k),'collector_accessible':np.zeros_like(k,dtype=bool)}
    ei=[];ej=[];eg=[]
    for a in range(3):
        low=[slice(None)]*3;high=low.copy();low[a]=slice(None,-1);high[a]=slice(1,None);low=tuple(low);high=tuple(high)
        valid=(index[low]>=0)&(index[high]>=0);kl=k[low][valid];kr=k[high][valid]
        ei.append(index[low][valid]);ej.append(index[high][valid]);eg.append(2*np.minimum(kl,kr)/(1+np.minimum(kl,kr)/np.maximum(kl,kr))*np.prod(h)/h[a]**2)
    i=np.concatenate(ei);j=np.concatenate(ej);g=np.concatenate(eg)
    K=coo_matrix((np.r_[g,g,-g,-g],(np.r_[i,j,i,j],np.r_[i,j,j,i])),shape=(n,n)).tocsr()
    graph=coo_matrix((np.ones(len(i)),(i,j)),shape=(n,n)).tocsr();count,components=connected_components(graph,directed=False)
    coordinates=np.array(np.unravel_index(ids,k.shape)).T
    left=np.flatnonzero(coordinates[:,axis]==0);right=np.flatnonzero(coordinates[:,axis]==k.shape[axis]-1)
    reachable=np.isin(components,components[right])
    if not np.intersect1d(components[left],components[right]).size:
        full=np.zeros(k.size);full[ids]=reachable.astype(float)
        accessible=np.zeros(k.size,bool);accessible[ids]=reachable
        return {'effective':0.,'residual':0.,'energy_relative_defect':0.,'components':int(count),
                'potential':full.reshape(k.shape),'collector_accessible':accessible.reshape(k.shape)}
    diagonal=np.zeros(n);rhs=np.zeros(n);factor=2*np.prod(h)/h[axis]**2
    gl=k.ravel()[ids[left]]*factor;gr=k.ravel()[ids[right]]*factor
    diagonal[left]+=gl;diagonal[right]+=gr;rhs[right]=gr
    connected=np.unique(components[np.r_[left,right]])
    floating=~np.isin(components,connected)
    _,first=np.unique(components,return_index=True);gauges=first[floating[first]]
    free=np.ones(n,bool);free[gauges]=False
    A=(K+diags(diagonal)).tocsr();reduced=A[free][:,free].tocsr();b=rhs[free]
    potential=np.zeros(n)
    if np.linalg.norm(b)>0:
        import pyamg
        scale=float(reduced.diagonal().max());normalized=reduced/scale
        multilevel=pyamg.smoothed_aggregation_solver(normalized,symmetry='symmetric',max_coarse=40,
            presmoother=('gauss_seidel',{'sweep':'symmetric'}),postsmoother=('gauss_seidel',{'sweep':'symmetric'}))
        potential[free],info=cg(normalized,b/scale,M=multilevel.aspreconditioner(),rtol=tolerance,atol=0.,maxiter=3000)
        if info:raise RuntimeError('Full-volume conductivity failed to converge')
    residual=float(np.linalg.norm((A@potential-rhs)[free])/max(np.linalg.norm(rhs),np.finfo(float).tiny))
    current=float(gr@(1-potential[right]));left_current=float(gl@potential[left])
    energy=float(g@((potential[j]-potential[i])**2)+gl@(potential[left]**2)+gr@((potential[right]-1)**2))
    scale_current=max(abs(current),abs(left_current),abs(energy),float(np.sum(gr))*1e-10,np.finfo(float).tiny)
    defect=max(abs(current-energy),abs(left_current-energy))/scale_current
    if residual>5*tolerance or defect>1e-6:raise RuntimeError(f'Transport residual/energy gate failed: {residual}, {defect}')
    L=k.shape[axis]*h[axis];area=np.prod(np.asarray(k.shape)*h)/L
    full=np.zeros(k.size);full[ids]=potential;accessible=np.zeros(k.size,bool)
    accessible[ids]=np.isin(components,components[right])
    return {'effective':max(0.,energy*L/area),'residual':residual,'energy_relative_defect':float(defect),'components':int(count),
            'potential':full.reshape(k.shape),'collector_accessible':accessible.reshape(k.shape)}
