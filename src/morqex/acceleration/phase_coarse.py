"""Material-aware P0 aggregation; exact Galerkin, an AMG solver candidate."""
from pathlib import Path
import numpy as np
from scipy import sparse
from .coarse_elasticity import CoarseElasticity
from .diagonal import Diagonal

def assemble(phase,C,spacing,block,counts=None):
    shape=phase.shape;coarse=tuple((n+block-1)//block for n in shape);N=int(np.prod(coarse));nphase=len(C);cells=np.arange(N).reshape(coarse)
    if N*4*nphase**3*4>128*2**20:raise MemoryError('Material-aware coarse histogram exceeds 128 MiB; choose fewer material groups or another preconditioner')
    B=[];index=np.array([[0,5,4],[5,1,3],[4,3,2]]);ratio=min(spacing)/np.asarray(spacing)[::-1]
    for a in range(3):
        v=np.zeros((6,3))
        for k in range(3):v[index[k,a],k]=ratio[a]*(1. if k==a else 1/np.sqrt(2))
        B.append(v)
    if counts is None:
        if phase.size>32768:raise MemoryError('Use the bounded CUDA histogram for large volumes')
        coords=np.indices(shape);grid_groups=np.ravel_multi_index(tuple(coords//block),coarse)
        def counts(a,b):
            ca=coords[2-a];cb=coords[2-b];flag=((ca%block==block-1)|(ca==shape[2-a]-1)).astype(int)+2*((cb%block==block-1)|(cb==shape[2-b]-1))
            key=(((grid_groups*4+flag)*nphase+phase)*nphase+np.roll(phase,-1,2-a))*nphase+np.roll(phase,-1,2-b)
            valid=np.max(abs(C),axis=(1,2))[phase]>0
            return np.bincount(key[valid],minlength=N*4*nphase**3).reshape(N,4,nphase,nphase,nphase)
    rows=[];cols=[];values=[]
    for a in range(3):
        for b in range(a,3):
            histogram=counts(a,b);cell,flags,p,pa,pb=np.nonzero(histogram);weight=histogram[cell,flags,p,pa,pb].astype(float);del histogram
            g=cell*nphase+p;ga=np.where(flags&1,np.roll(cells,-1,2-a).ravel()[cell],cell)*nphase+pa;gb=np.where(flags&2,np.roll(cells,-1,2-b).ravel()[cell],cell)*nphase+pb
            tensor=np.einsum('ik,pij,jl->pkl',B[a],C,B[b]);coeff=tensor[p]*weight[:,None,None]
            for swapped in ([False,True] if a!=b else [False]):
                left,right=(gb,ga) if swapped else (ga,gb);v=coeff.swapaxes(1,2) if swapped else coeff
                for i,j,sign in [(left,right,1),(left,g,-1),(g,right,-1),(g,g,1)]:
                    rr=np.broadcast_to(i[:,None,None]*3+np.arange(3)[None,:,None],v.shape).astype(np.int32);cc=np.broadcast_to(j[:,None,None]*3+np.arange(3)[None,None,:],v.shape).astype(np.int32);keep=v!=0
                    rows.append(rr[keep]);cols.append(cc[keep]);values.append(sign*v[keep])
    K=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(3*N*nphase,)*2).tocsr();K.eliminate_zeros()
    return K,(*coarse,nphase)

class PhaseCoarse(CoarseElasticity):
    def __init__(self,model,*,block=None):
        Diagonal.__init__(self,model);cp=model.xp;self.cp=cp;block=max(2,int(np.ceil(max(model.shape)/28))) if block is None else block
        coarse=tuple((n+block-1)//block for n in model.shape);size=int(np.prod(coarse))*4*len(model.C_host)**3
        if size*4>128*2**20:raise MemoryError('Material-aware histogram exceeds configured allocation cap')
        prefix=f'#define NX {model.shape[2]}\n#define NY {model.shape[1]}\n#define NZ {model.shape[0]}\n#define BX {block}\n#define NPHASE {len(model.C_host)}\n'
        code=Path(__file__).with_suffix('.cu').read_text();module=cp.RawModule(code=prefix+code,options=('--std=c++14',),name_expressions=['count_pairs']);kernel=module.get_function('count_pairs');nonzero=cp.asarray(np.max(abs(model.C_host),axis=(1,2))>0,dtype='uint8')
        def counts(a,b):
            with model.budget.batch():
                out=cp.zeros(size,dtype='uint32');kernel(((model.n+127)//128,),(128,),(model.phase,nonzero,out,np.int32(a),np.int32(b)));host=cp.asnumpy(out)
            return host.reshape(*[int(np.prod(coarse)),4,len(model.C_host),len(model.C_host),len(model.C_host)])
        K,shape=assemble(model.phase_host,model.C_host,model.spacing,block,counts)
        self._finish(model,K,shape,block,code,extra_arguments=(model.phase,))
