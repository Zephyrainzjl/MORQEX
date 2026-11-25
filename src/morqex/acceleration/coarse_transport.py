"""Two-level scalar preconditioner with exact P0 Galerkin coarse geometry.

Fine conductivities and pores are unchanged. A CPU AMG V-cycle acts only on
the much smaller P0^T K P0 problem. Symmetric Chebyshev pre/post smoothing
and adjoint injection/summation transfers preserve variational symmetry.
"""
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def coarse_matrix(phase,coefficient,spacing,block=8):
    phase=np.asarray(phase);shape=phase.shape;coarse=tuple((n+block-1)//block for n in shape);N=int(np.prod(coarse));rows=[];cols=[];values=[]
    ratios=min(spacing)/np.asarray(spacing)
    for axis in range(3):
        planes=np.r_[np.arange(block-1,shape[axis]-1,block),shape[axis]-1]
        here=np.asarray(coefficient)[np.take(phase,planes,axis=axis)];other=np.asarray(coefficient)[np.take(phase,(planes+1)%shape[axis],axis=axis)]
        g=np.divide(2*here*other,here+other,out=np.zeros_like(here),where=here+other>0)*ratios[axis]**2
        coordinates=np.indices(g.shape)
        for a in range(3):
            if a!=axis:coordinates[a]//=block
        ids=np.ravel_multi_index(tuple(coordinates),coarse);conductance=np.bincount(ids.ravel(),weights=g.ravel(),minlength=N)
        i=np.arange(N).reshape(coarse);j=np.roll(i,-1,axis).ravel();i=i.ravel()
        rows.extend([i,i,j,j]);cols.extend([i,j,i,j]);values.extend([conductance,-conductance,-conductance,conductance])
    K=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(N,N)).tocsr();K.eliminate_zeros();return K,coarse

class CoarseTransport:
    def __init__(self,model,scaling,*,block=None,smoothing_steps=4):
        import pyamg
        block=max(2,int(np.ceil(max(model.shape)/32))) if block is None else block
        self.model=model;cp=model.xp;self.cp=cp;self.scaling=scaling;self.block=block
        K,self.shape=coarse_matrix(model.phase_host,model.coefficient_host,model.spacing,block)
        _,labels=connected_components(K,directed=False);_,pins=np.unique(labels,return_index=True);free=K.diagonal()>0;free[pins]=False;self.free=free
        # Keep global random state unchanged while making AMG setup reproducible.
        state=np.random.get_state();np.random.seed(0)
        try:
            self.multilevel=pyamg.smoothed_aggregation_solver(K[free][:,free],symmetry='symmetric',presmoother=('gauss_seidel',{'sweep':'symmetric'}),postsmoother=('gauss_seidel',{'sweep':'symmetric'}),max_coarse=32) if free.any() else None
            self.inverse=self.multilevel.aspreconditioner(cycle='V') if free.any() else None
        finally:np.random.set_state(state)
        self.host=np.zeros(int(np.prod(self.shape)));self.coarse=cp.zeros(self.shape);self.work=cp.empty(model.shape,dtype='float32')
        names=['restrict_block','prolong_block','matrix32'];prefix=f'#define NX {model.shape[2]}\n#define NY {model.shape[1]}\n#define NZ {model.shape[0]}\n#define BX {block}\n'
        self.module=cp.RawModule(code=prefix+Path(__file__).with_name('transport.cu').read_text()+Path(__file__).with_suffix('.cu').read_text(),options=('--std=c++14',),name_expressions=names)
        self.functions={name:self.module.get_function(name) for name in names}
        self.update=cp.ElementwiseKernel('float64 r,float32 d,float32 Au,float64 omega','float32 x','x+=omega*d*d*(r-Au)','morqex_chebyshev_smooth')
        self.residual=cp.ElementwiseKernel('float64 r','float32 v','v=r-v','morqex_chebyshev_residual')
        # Spectrum of D^-1/2 K D^-1/2 is contained in [0,2] for the
        # harmonic-face graph Laplacian. The polynomial is a smoother only.
        weights=1/(1.025+.975*np.cos(np.pi*(np.arange(smoothing_steps)+.5)/smoothing_steps))
        self.weights=weights[[1,3,0,2]] if smoothing_steps==4 else weights
    def matvec(self,x):
        m=self.model;self.functions['matrix32'](((m.n+127)//128,),(128,),(x,self.work,m.phase,m.c,m.ratio))
    def apply(self,r,out):
        cp=self.cp;m=self.model;out.fill(0.)
        for weight in self.weights:self.matvec(out);self.update(r,self.scaling,self.work,float(weight),out)
        self.matvec(out);self.residual(r,self.work)
        self.functions['restrict_block'](((self.coarse.size+127)//128,),(128,),(self.work,self.coarse))
        rhs=cp.asnumpy(self.coarse).ravel();self.host.fill(0.)
        if self.inverse is not None:self.host[self.free]=self.inverse@rhs[self.free]
        self.coarse[:]=cp.asarray(self.host.reshape(self.shape))
        self.functions['prolong_block'](((m.n+127)//128,),(128,),(self.coarse,out))
        for weight in reversed(self.weights):self.matvec(out);self.update(r,self.scaling,self.work,float(weight),out)
        return out
