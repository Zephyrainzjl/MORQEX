"""Replaceable concentration partitions, including a geometry-only port metric."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.csgraph import dijkstra

@dataclass(frozen=True)
class GeometricBasis:
    strategy:str='geometry'
    def build(self,electrode,target):return electrode.partition(target,self.strategy).partition

@dataclass(frozen=True)
class ExactBasis:
    def build(self,electrode,target):return np.arange(electrode.n)

@dataclass(frozen=True)
class VariationalOperator:
    def build(self,electrode,partition):return electrode.reduce(partition,'custom-variational')

@dataclass(frozen=True)
class PortBasis:
    """Connected greedy partition from mechanical chemical-potential rows.

    No validation concentration snapshots are used. This candidate requires a
    small full offline response (<=128 active voxels), and is not claimed optimal.
    """
    concentration_scale:float=5000.
    strain_scale:float=.005
    temperature_scale:float=10.
    def build(self,electrode,target):
        fv=electrode.diffusion
        if target<fv.n_components:raise ValueError('At least one mode per lithium inventory component')
        full=electrode.H;scales=np.r_[self.strain_scale,np.full(electrode.n,self.concentration_scale),self.temperature_scale]
        features=(full[1:-1]/fv.weights[:,None])*scales
        norm=max(np.sqrt(np.sum(features**2)/len(features)),1e-30);features/=norm
        # Surface-to-volume sensitivity captures reaction forcing in addition
        # to mechanical feedback. Each channel is dimensionless and documented.
        area=electrode.area/fv.weights;area/=max(np.sqrt(np.mean(area**2)),1e-30)
        features=np.column_stack([features,area])
        weight=fv.weights+electrode.area/max(electrode.area.sum(),1e-30)*fv.weights.sum()
        graph=fv.stiffness.copy();i,j=graph.nonzero();graph=graph.tocoo();off=graph.row!=graph.col
        from scipy import sparse
        cost=np.linalg.norm(features[graph.row[off]]-features[graph.col[off]],axis=1)+1e-12
        graph=sparse.coo_matrix((cost,(graph.row[off],graph.col[off])),shape=graph.shape).tocsr()
        groups=[np.flatnonzero(fv.components==k) for k in range(fv.n_components)]
        while len(groups)<min(target,electrode.n):
            score=[]
            for g in groups:
                mean=np.average(features[g],weights=weight[g],axis=0)
                score.append(float(np.sum(weight[g,None]*(features[g]-mean)**2)) if len(g)>1 else -1.)
            k=int(np.argmax(score));g=groups.pop(k);f=features[g];_,_,v=np.linalg.svd((f-np.average(f,weights=weight[g],axis=0))*np.sqrt(weight[g,None]),full_matrices=False)
            ordered=np.argsort(f@v[0]);dist=dijkstra(graph[g][:,g],directed=False,indices=[ordered[0],ordered[-1]])
            labels=np.argmin(dist,axis=0);groups.extend(g[labels==j] for j in (0,1))
        labels=np.empty(electrode.n,dtype=int)
        for k,g in enumerate(groups):labels[g]=k
        return labels
