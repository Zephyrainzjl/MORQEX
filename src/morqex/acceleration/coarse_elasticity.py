"""P0 Galerkin elastic coarse matrix and symmetric GPU/CPU AMG preconditioner.

Offline coarse assembly visits block faces/intersections, not every fine
stiffness entry. Established multigrid/aggregation ideas are used as a solver
candidate, not claimed as a new homogenization method.
"""
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from .diagonal import Diagonal

def coarse_matrix(phase,C,spacing,block=8):
    phase=np.asarray(phase);shape=phase.shape;coarse=tuple((n+block-1)//block for n in shape);N=int(np.prod(coarse));nphase=len(C)
    if N*nphase*8>128*2**20:raise MemoryError('Coarse material count exceeds the assembly budget; use larger blocks')
    index=np.array([[0,5,4],[5,1,3],[4,3,2]]);ratios=min(spacing)/np.asarray(spacing)[::-1];B=[]
    for a in range(3):
        basis=np.zeros((6,3))
        for k in range(3):basis[index[k,a],k]=ratios[a]*(1. if k==a else 1/np.sqrt(2))
        B.append(basis)
    rows=[];cols=[];values=[];coarse_ids=np.arange(N).reshape(coarse)
    for a in range(3):
        for b in range(a,3):
            selected=phase
            for axis in sorted({2-a,2-b}):
                planes=np.r_[np.arange(block-1,shape[axis]-1,block),shape[axis]-1]
                selected=np.take(selected,planes,axis=axis)
            coords=np.indices(selected.shape)
            for axis in range(3):
                if axis not in {2-a,2-b}:coords[axis]//=block
            groups=np.ravel_multi_index(tuple(coords),coarse)
            count=np.bincount((groups.ravel()*nphase+selected.ravel()),minlength=N*nphase).reshape(N,nphase)
            active=np.flatnonzero(count.sum(axis=1));g=active;ga=np.roll(coarse_ids,-1,2-a).ravel()[active];gb=np.roll(coarse_ids,-1,2-b).ravel()[active]
            tensor=np.einsum('ik,pij,jl->pkl',B[a],C,B[b]);coeff=(count[active]@tensor.reshape(nphase,9)).reshape(-1,3,3)
            for swapped in ([False,True] if a!=b else [False]):
                left,right=(gb,ga) if swapped else (ga,gb);v=coeff.swapaxes(1,2) if swapped else coeff
                for i,j,sign in [(left,right,1),(left,g,-1),(g,right,-1),(g,g,1)]:
                    rr=np.broadcast_to(i[:,None,None]*3+np.arange(3)[None,:,None],v.shape).astype(np.int32)
                    cc=np.broadcast_to(j[:,None,None]*3+np.arange(3)[None,None,:],v.shape).astype(np.int32)
                    mask=v!=0;rows.append(rr[mask]);cols.append(cc[mask]);values.append(sign*v[mask])
    K=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(3*N,3*N)).tocsr();K.eliminate_zeros();return K,coarse

class CoarseElasticity(Diagonal):
    def __init__(self,model,*,block=None,smoothing_steps=4):
        import pyamg
        super().__init__(model);cp=model.xp;self.cp=cp;block=max(2,int(np.ceil(max(model.shape)/28))) if block is None else block
        K,shape=coarse_matrix(model.phase_host,model.C_host,model.spacing,block)
        transfers=Path(__file__).with_name('coarse_transport.cu').read_text();transfers=transfers[transfers.index('extern "C" __global__ void restrict_block'):]
        self._finish(model,K,shape,block,transfers,smoothing_steps=smoothing_steps)
    def _finish(self,model,K,shape,block,transfers,*,extra_arguments=(),smoothing_steps=4):
        import pyamg
        cp=model.xp;self.shape=shape;self.fast=shape;self.extra_arguments=extra_arguments
        graph=sum(abs(K[k::3,k::3]) for k in range(3));_,labels=connected_components(graph,directed=False);_,pins=np.unique(labels,return_index=True)
        free=graph.diagonal()>0;free[pins]=False;self.free=np.repeat(free,3);N=len(free)
        candidates=np.tile(np.eye(3),(N,1))[self.free];state=np.random.get_state();np.random.seed(0)
        try:
            self.multilevel=pyamg.smoothed_aggregation_solver(K[self.free][:,self.free].tobsr(blocksize=(3,3)),B=candidates,symmetry='symmetric',presmoother=('block_gauss_seidel',{'sweep':'symmetric'}),postsmoother=('block_gauss_seidel',{'sweep':'symmetric'}),max_coarse=36) if self.free.any() else None
            self.inverse=self.multilevel.aspreconditioner(cycle='V') if self.multilevel is not None else None
        finally:np.random.set_state(state)
        self.host=np.zeros((N,3));self.coarse=cp.zeros((3,*self.shape));self.work=cp.empty((3,*model.shape),dtype='float32')
        names=['restrict_block','prolong_block'];prefix=f'#define NX {model.shape[2]}\n#define NY {model.shape[1]}\n#define NZ {model.shape[0]}\n#define BX {block}\n#define NPHASE {len(model.C_host)}\n'
        self.module=cp.RawModule(code=prefix+transfers,options=('--std=c++14',),name_expressions=names)
        self.functions={name:self.module.get_function(name) for name in names}
        self.update=cp.ElementwiseKernel('float64 r,float32 d,float32 Au,float64 omega','float32 x','x+=omega*d*(r-Au)','morqex_elastic_chebyshev')
        self.residual=cp.ElementwiseKernel('float64 r','float32 v','v=r-v','morqex_elastic_coarse_residual')
        weights=1/(6.075+5.925*np.cos(np.pi*(np.arange(smoothing_steps)+.5)/smoothing_steps));self.weights=weights[[1,3,0,2]] if smoothing_steps==4 else weights
    def apply(self,r,out):
        cp=self.cp;m=self.model;out.fill(0.)
        for weight in self.weights:m.apply(out,self.work);self.update(r,self.scaling,self.work,float(weight),out)
        m.apply(out,self.work);self.residual(r,self.work)
        nc=int(np.prod(self.shape))
        for k in range(3):self.functions['restrict_block'](((nc+127)//128,),(128,),(self.work[k],self.coarse[k],*self.extra_arguments))
        rhs=cp.asnumpy(self.coarse).reshape(3,-1).T.ravel();self.host.fill(0.)
        if self.inverse is not None:self.host.ravel()[self.free]=self.inverse@rhs[self.free]
        self.coarse[:]=cp.asarray(self.host.T.reshape(3,*self.shape))
        for k in range(3):self.functions['prolong_block'](((m.n+127)//128,),(128,),(self.coarse[k],out[k],*self.extra_arguments))
        for weight in reversed(self.weights):m.apply(out,self.work);self.update(r,self.scaling,self.work,float(weight),out)
        return out
