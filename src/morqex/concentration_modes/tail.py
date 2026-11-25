"""Connected concentration aggregation using actual stress-observable columns."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from morqex.mechanics.tensors import from_mandel

@dataclass(frozen=True)
class TailBasis:
    """Geometry-only offline loads select stress-sensitive observables.

    This is an experimental goal-metric clustering candidate, not an optimal
    basis theorem or a certified P99 bound. Requires full small-SVE response.
    """
    hotspot_fraction:float=.02
    hotspot_weight:float=4.
    def build(self,electrode,target):
        f=electrode;fv=f.diffusion;n=f.n
        if not 0<self.hotspot_fraction<1 or self.hotspot_weight<0 or target<fv.n_components:raise ValueError('Invalid tail-basis options')
        response=f.response;H=f.H
        S=response.stress_operator[:,:,6:6+n]-response.stress_operator[:,:,0,None]*(H[0,1:-1]/H[0,0])[None,None,:]
        xyz=fv.centers;centered=(xyz-xyz.mean(axis=0))/np.ptp(xyz,axis=0).clip(min=1e-30)
        loads=np.column_stack([np.ones(n),centered]);hot=np.zeros(len(S),bool)
        for load in loads.T:
            stress=np.einsum('eik,k->ei',S,load)
            for sign in [-1,1]:
                principal=np.linalg.eigvalsh(from_mandel(sign*stress))[...,-1];order=np.argsort(principal)[::-1]
                count=max(1,int(np.ceil(self.hotspot_fraction*len(S))));hot[order[:count]]=True
        # Dividing columns by physical cell weights makes equal features cancel
        # exactly against volume-zero concentration errors inside a region.
        weights=f.mesh.volumes/f.mesh.rve_volume
        linear=(S*np.sqrt(weights[:,None,None])).reshape(-1,n).T/fv.weights[:,None]
        tail=S[hot].reshape(-1,n).T/fv.weights[:,None]
        linear/=max(np.sqrt(np.mean(linear**2)),1e-30);tail/=max(np.sqrt(np.mean(tail**2)),1e-30)
        area=f.area/fv.weights;area/=max(np.sqrt(np.mean(area**2)),1e-30)
        features=np.column_stack([linear/np.sqrt(linear.shape[1]),self.hotspot_weight*tail/np.sqrt(tail.shape[1]),area])
        adjacency=fv.stiffness.tocoo();off=adjacency.row!=adjacency.col;i,j=adjacency.row[off],adjacency.col[off]
        graph=sparse.coo_matrix((np.linalg.norm(features[i]-features[j],axis=1)+1e-12,(i,j)),shape=(n,n)).tocsr()
        groups=[np.flatnonzero(fv.components==k) for k in range(fv.n_components)]
        while len(groups)<min(n,target):
            scores=[]
            for g in groups:
                mean=np.average(features[g],weights=fv.weights[g],axis=0)
                scores.append(float(np.sum(fv.weights[g,None]*(features[g]-mean)**2)) if len(g)>1 else -1.)
            g=groups.pop(int(np.argmax(scores)));feature=features[g];_,_,v=np.linalg.svd((feature-feature.mean(axis=0))*np.sqrt(fv.weights[g,None]),full_matrices=False)
            order=np.argsort(feature@v[0]);distance=dijkstra(graph[g][:,g],directed=False,indices=[order[0],order[-1]])
            labels=np.argmin(distance,axis=0);groups.extend(g[labels==j] for j in [0,1])
        labels=np.empty(n,int)
        for k,g in enumerate(groups):labels[g]=k
        return labels
