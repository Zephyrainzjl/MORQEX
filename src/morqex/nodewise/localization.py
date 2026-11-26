"""Batched full-grid localization, CPU reference and bounded CUDA backend."""
from pathlib import Path
import time
import numpy as np


def voxel_maps(localizer):
    """One exact volume average of the Tet4 linear localization maps.

    A principal stress computed afterwards is that of the voxel-mean tensor.
    It is not the maximum over sub-voxel elements; those remain in the cache.
    """
    arrays={};labels=localizer.array('labels');shape=labels.shape
    for name in ['stress','strain','displacement']:
        for part in ['direct','modes']:
            a=localizer.array(name+'_'+part)
            if name=='displacement':
                from types import SimpleNamespace
                from .voxel_operator import nodal_to_voxel
                mesh=SimpleNamespace(cells=localizer.array('cells'),voxel_ids=localizer.array('element_voxel_ids'),volumes=localizer.array('volumes'))
                a=nodal_to_voxel(mesh,a,shape)
            elif localizer.manifest.get('field_location')!='voxel':a=localizer.voxel_average(a).reshape(-1,*a.shape[1:])
            arrays[name+'_'+part]=np.nan_to_num(a,copy=False)
    P=np.zeros((labels.size,localizer.kernel.operator.dimension));P[localizer.array('voxel_ids')]=localizer.array('P')
    arrays['concentration']=P
    # Full Tet4 maps are recoverable from their checked cache. They need not
    # remain resident after building the much smaller voxel maps.
    localizer.loaded.clear()
    return arrays


class BatchLocalization:
    """``parallel_nodes`` limits simultaneously evaluated distinct node states.

    Upload once and reuse immutable maps for every node/time. IDAKLU remains a
    CPU sparse DAE solver; this backend accelerates actual full-voxel recovery.
    """
    def __init__(self,localizers,*,backend='cpu',parallel_nodes=2):
        if backend not in {'cpu','cuda'} or not isinstance(parallel_nodes,int) or parallel_nodes<1:raise ValueError('CPU/CUDA backend and positive parallel node budget required')
        self.backend=backend;self.parallel_nodes=parallel_nodes;self.localizers=list(localizers)
        self.shapes=[s.array('labels').shape for s in localizers]
        if len(set(self.shapes))!=1:raise ValueError('One batch requires matching voxel output shapes; group differing grids explicitly')
        self.kernels=[s.kernel for s in localizers]
        self.nq=max(s.kernel.operator.dimension+2 for s in localizers);self.nr=max(len(s.kernel.cohesive.eigenvalues) for s in localizers)
        self.packed={};self.timings={'upload_s':0.,'kernel_and_transfer_s':0.,'calls':0}
        ng=len(localizers);N=int(np.prod(self.shapes[0]));channels={'stress':slice(0,6),'strain':slice(6,12),'displacement':slice(12,15)}
        if backend=='cuda':
            combined=[np.zeros((ng,self.nq,N*15)),np.zeros((ng,self.nr,N*15))]
        else:
            for name,sl in channels.items():
                nc=sl.stop-sl.start
                self.packed[name]=(np.zeros((ng,N*nc,self.nq)),np.zeros((ng,N*nc,self.nr)),nc)
        # Stream one geometry at a time instead of retaining original maps plus
        # a row-major copy plus a transposed copy for every geometry.
        for i,localizer in enumerate(localizers):
            maps=voxel_maps(localizer);k=localizer.kernel.operator.dimension;columns=[0,*range(1,k+1),self.nq-1]
            for name,sl in channels.items():
                nc=sl.stop-sl.start;direct=maps[name+'_direct'];modes=maps[name+'_modes'];r=modes.shape[-1]
                if backend=='cuda':
                    combined[0][i].reshape(self.nq,N,15)[columns,:,sl]=direct.transpose(2,0,1)
                    combined[1][i,:r].reshape(r,N,15)[:,:,sl]=modes.transpose(2,0,1)
                else:
                    self.packed[name][0][i][:,columns]=direct.reshape(N*nc,k+2)
                    self.packed[name][1][i,:,:r]=modes.reshape(N*nc,r)
            del maps
        if backend=='cuda':
            from ..acceleration.budget import GpuBudget
            self.budget=GpuBudget();cp=self.budget.cp;self.xp=cp
            source=Path(__file__).resolve().parents[1]/'acceleration/node_localization.cu'
            self.kernel=cp.RawKernel(source.read_text(),'morqex_node_fields',options=('--std=c++14',))
            started=time.perf_counter();self.kernel.compile();self.timings['compile_s']=time.perf_counter()-started
            self.channel_slices={'stress':slice(0,6),'strain':slice(6,12),'displacement':slice(12,15)}
            started=time.perf_counter()
            with self.budget.batch():
                self.cuda_packed=tuple(cp.asarray(a) for a in combined)
            self.timings['upload_s']=time.perf_counter()-started
        else:self.xp=np
    def recover(self,states,*,fields=('stress','strain','displacement')):
        """States: (geometry_index, local_c, axial_strain, deltaT, damage)."""
        if not states or not set(fields)<={'stress','strain','displacement'}:raise ValueError('Nonempty states and valid output fields required')
        result={name:[] for name in fields};started=time.perf_counter();xp=self.xp
        for start in range(0,len(states),self.parallel_nodes):
            block=states[start:start+self.parallel_nodes];Q=np.zeros((len(block),self.nq));Z=np.zeros((len(block),self.nr));indices=[]
            for i,(index,c,E,theta,d) in enumerate(block):
                if not 0<=index<len(self.kernels) or not 0<=d<=1:raise ValueError('Invalid node geometry or damage')
                kernel=self.kernels[index];m=kernel.operator.dimension;q=np.r_[E,np.asarray(c)-kernel.operator.reference_concentration,theta]
                if q.shape!=(m+2,) or not np.isfinite(q).all():raise ValueError('Finite matching local state required')
                Q[i,[0,*range(1,m+1),self.nq-1]]=q
                coh=kernel.cohesive;g=coh.residual+(1-coh.residual)*(1-d)**2
                Z[i,:len(coh.eigenvalues)]=coh.factors@q/(1+(g-1)*coh.eigenvalues);indices.append(index)
            if self.backend=='cpu':
                for name in fields:
                    D,M,ncomp=self.packed[name]
                    value=np.stack([D[index]@Q[i]+M[index]@Z[i] for i,index in enumerate(indices)])
                    result[name].append(value.reshape(len(block),*self.shapes[0],ncomp))
            else:
                with self.budget.batch():
                    q=xp.asarray(Q);z=xp.asarray(Z);ind=xp.asarray(indices,dtype=xp.int32)
                    D,M=self.cuda_packed;outputs=D.shape[-1];out=xp.empty((len(block),outputs),dtype=xp.float64)
                    self.kernel(((out.size+127)//128,),(128,),(D,M,q,z,ind,out,np.int32(len(block)),np.int32(outputs),np.int32(self.nq),np.int32(self.nr)))
                    host=xp.asnumpy(out).reshape(len(block),*self.shapes[0],15)
                    for name in fields:result[name].append(host[...,self.channel_slices[name]])
        self.timings['kernel_and_transfer_s']+=time.perf_counter()-started;self.timings['calls']+=1
        return {name:np.concatenate(values,axis=0) for name,values in result.items()}
    def metadata(self):
        return {'backend':self.backend,'parallel_nodes':self.parallel_nodes,'precision':'FP64','timings':self.timings,
            'gpu':self.budget.metadata() if self.backend=='cuda' else None,
            'scope':'voxel volume-mean tensor and displacement localization; nonlinear invariants evaluated from those tensors'}
