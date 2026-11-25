"""Compiled periodic voxel elasticity with no explicit global stiffness matrix.

The compatible forward-gradient energy is the discretization definition. It is
not Tet4/Hex8 FEM and is not silently substituted into their archived benchmarks.
Zero pore stiffness is retained. Additional null modes can make CG fail; such
failures are explicit, never replaced by an unreported weak phase.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,cg
from scipy.fft import rfftn,irfftn

SOURCE=Path(__file__).with_suffix('.cu')

class NativeStencil:
    def __init__(self,phase,C,beta=None,alpha=None,spacing=(1.,1.,1.),*,backend='cpu',budget=None,preconditioner='fft'):
        if preconditioner not in {'fft','jacobi','coarse','phase-coarse','geometric'}:raise ValueError('Unknown voxel preconditioner')
        self.preconditioner_kind=preconditioner
        phase=np.asarray(phase)
        if phase.ndim!=3 or min(phase.shape)<2 or phase.dtype.kind not in 'ui':raise ValueError('A three-dimensional integer phase array is required')
        C=np.asarray(C,float);nphase=C.shape[0]
        if nphase>256 or C.shape!=(nphase,6,6) or phase.max()>=nphase or not np.isfinite(C).all() or not np.allclose(C,C.swapaxes(1,2)):raise ValueError('At most 256 symmetric finite phase tensors required')
        if np.linalg.eigvalsh(C).min()<-1e-8*max(1,np.max(abs(C))):raise ValueError('Positive semidefinite elasticity required')
        if len(spacing)!=3 or not np.isfinite(spacing).all() or min(spacing)<=0:raise ValueError('Positive finite ZYX spacing required')
        self.shape=phase.shape;self.n=phase.size;self.backend=backend;self.scale=float(np.max(abs(C))) or 1.
        self.budget=budget;self.spacing=tuple(spacing);self.ratios=np.asarray(spacing).min()/np.asarray(spacing)[::-1]
        self.phase_host=np.ascontiguousarray(phase,dtype=np.uint8);self.C_host=np.ascontiguousarray(C/self.scale)
        template=np.zeros_like(self.C_host)
        for i,cc in enumerate(self.C_host):
            template[i,:3,:3]=cc[0,1];np.fill_diagonal(template[i,:3,:3],cc[0,1]+cc[3,3]);template[i,3:,3:]=np.eye(3)*cc[3,3]
        isotropic=np.allclose(self.C_host,template,rtol=1e-14,atol=1e-14)
        options=['--std=c++14']+(['-DMORQEX_ISOTROPIC=1'] if isotropic else [])
        self.beta_host=np.zeros((nphase,6)) if beta is None else np.ascontiguousarray(beta,dtype=float)
        self.alpha_host=np.zeros((nphase,6)) if alpha is None else np.ascontiguousarray(alpha,dtype=float)
        if self.beta_host.shape!=(nphase,6) or self.alpha_host.shape!=(nphase,6) or not np.isfinite(np.r_[self.beta_host.ravel(),self.alpha_host.ravel()]).all():raise ValueError('One finite Kelvin chemical/thermal strain tensor per phase required')
        if backend=='cuda':
            from .budget import GpuBudget
            if budget is None:self.budget=GpuBudget()
            import cupy as cp
            self.budget.stream.use()
            self.xp=cp;names=['elastic','source','stress','residual','update','direction','elastic_macro','mean_stress']
            prefix=f'#define MORQEX_NX {self.shape[2]}\n#define MORQEX_NY {self.shape[1]}\n#define MORQEX_NZ {self.shape[0]}\n'
            self.module=cp.RawModule(code=prefix+SOURCE.read_text(),options=tuple(options),name_expressions=names)
            self.functions={n:self.module.get_function(n) for n in names}
            self.float_module=cp.RawModule(code=prefix+SOURCE.read_text(),options=tuple(options+['-DMORQEX_FLOAT=1']),name_expressions=names)
            self.float_functions={n:self.float_module.get_function(n) for n in names}
            self.phase=cp.asarray(self.phase_host);self.C=cp.asarray(self.C_host);self.beta=cp.asarray(self.beta_host);self.alpha=cp.asarray(self.alpha_host);self.ratios=cp.asarray(self.ratios)
        elif backend=='cpu':
            self.xp=np;ws=Path(os.environ.get('MORQEX_WORKSPACE',''))
            if not ws.is_absolute():raise RuntimeError('External MORQEX_WORKSPACE required for native builds')
            key=hashlib.sha256(SOURCE.read_bytes()+str(isotropic).encode()).hexdigest()[:20];library=ws/'build/voxel'/f'{key}.so';library.parent.mkdir(parents=True,exist_ok=True)
            if not library.exists():subprocess.run(['g++','-x','c++','-std=c++14','-O3','-shared','-fPIC',*(['-DMORQEX_ISOTROPIC=1'] if isotropic else []),str(SOURCE),'-o',str(library)],check=True)
            self.module=ctypes.CDLL(str(library));self.phase=self.phase_host;self.C=self.C_host;self.beta=self.beta_host;self.alpha=self.alpha_host
        else:raise ValueError('backend must be cpu or cuda')
    def _call(self,name,args,count=None):
        count=self.n if count is None else count
        if self.backend=='cuda':
            converted=tuple(np.int32(v) if isinstance(v,int) else np.float64(v) if isinstance(v,float) else v for v in args)
            functions=self.float_functions if getattr(args[0],'dtype',None)==np.dtype('float32') else self.functions
            functions[name](((count+127)//128,),(128,),converted,stream=self.budget.stream)
            if os.environ.get('MORQEX_CUDA_DEBUG')=='1':self.budget.stream.synchronize()
        else:
            converted=[ctypes.c_void_p(v.ctypes.data) if isinstance(v,np.ndarray) else ctypes.c_double(v) if isinstance(v,float) else ctypes.c_int(v) for v in args]
            getattr(self.module,name)(*converted)
    @property
    def dims(self):return tuple(reversed(self.shape))
    def _displacement(self,u):
        dtype='float32' if self.backend=='cuda' and getattr(u,'dtype',None)==np.dtype('float32') else 'float64'
        u=self.xp.ascontiguousarray(self.xp.asarray(u,dtype=dtype))
        if u.shape!=(3,*self.shape):raise ValueError('Displacement must have shape (3,Z,Y,X)')
        return u
    def _check_output(self,out,shape,dtype):
        if out.shape!=shape or out.dtype!=dtype or not out.flags.c_contiguous:raise ValueError('Output must be contiguous with the expected shape and dtype')
        if self.backend=='cpu' and not out.flags.writeable:raise ValueError('Output buffer must be writable')
    def _concentration(self,c):
        c=self.xp.ascontiguousarray(self.xp.asarray(c,dtype='float64'))
        if c.shape!=self.shape:raise ValueError('Concentration must match the ZYX phase grid')
        return c
    def _macro(self,macro):
        E=self.xp.ascontiguousarray(self.xp.asarray(macro,dtype='float64'))
        if E.shape!=(6,):raise ValueError('Six Kelvin macroscopic strains required')
        return E
    def displacement_m(self,u):
        """Convert u/h_min to the physical periodic displacement fluctuation (m)."""
        return self._displacement(u)*min(self.spacing)
    def apply(self,u,out=None):
        u=self._displacement(u)
        if out is None:out=self.xp.empty_like(u)
        self._check_output(out,u.shape,u.dtype)
        self._call('elastic',[u,out,self.phase,self.C,*self.dims,self.ratios]);return out
    def load(self,concentration,macro=(0.,)*6,temperature=0.,out=None):
        if out is None:out=self.xp.empty((3,*self.shape),dtype='float64')
        concentration=self._concentration(concentration);E=self._macro(macro)
        if out.shape!=(3,*self.shape) or out.dtype not in (np.dtype('float32'),np.dtype('float64')) or not out.flags.c_contiguous:raise ValueError('Invalid load output buffer')
        if self.backend=='cpu' and (out.dtype!=np.dtype('float64') or not out.flags.writeable):raise ValueError('CPU load requires a writable FP64 output')
        self._call('source',[out,self.phase,self.C,self.beta,self.alpha,concentration,E,float(temperature),*self.dims,self.ratios]);return out
    def stress(self,u,concentration,macro=(0.,)*6,temperature=0.,out=None):
        u=self._displacement(u).astype('float64',copy=False)
        if out is None:out=self.xp.empty((6,*self.shape),dtype='float64')
        self._check_output(out,(6,*self.shape),np.dtype('float64'))
        E=self._macro(macro);concentration=self._concentration(concentration)
        self._call('stress',[u,out,self.phase,self.C,self.beta,self.alpha,concentration,E,float(temperature),*self.dims,self.ratios]);out*=self.scale;return out
    def residual(self,u,concentration,macro=(0.,)*6,temperature=0.,out=None):
        u=self._displacement(u)
        if out is None:out=self.xp.empty_like(u)
        self._check_output(out,u.shape,u.dtype)
        E=self._macro(macro);concentration=self._concentration(concentration)
        self._call('residual',[u,out,self.phase,self.C,self.beta,self.alpha,concentration,E,float(temperature),*self.dims,self.ratios]);return out
    def mean_stress(self,u=None,concentration=None,macro=(0.,)*6,temperature=0.):
        xp=self.xp;blocks=(self.n+127)//128 if self.backend=='cuda' else 1
        if concentration is None and temperature:concentration=xp.zeros(self.shape,dtype='float64')
        out=xp.empty((6,blocks),dtype='float64');displacement=int(u is not None);source=1 if concentration is not None else 2
        if u is None:u=xp.zeros(1,dtype='float64')
        else:u=self._displacement(u)
        if concentration is None:concentration=xp.zeros(1,dtype='float64')
        else:concentration=self._concentration(concentration)
        E=self._macro(macro)
        self._call('mean_stress',[u,out,self.phase,self.C,self.beta,self.alpha,concentration,E,float(temperature),source,displacement,*self.dims,self.ratios])
        return out.sum(axis=1)/self.n*self.scale
    def solve(self,concentration,macro=(0.,)*6,temperature=0.,*,rtol=1e-7,maxiter=1500,mixed_preconditioner=True,checkpoint=None,low_memory=None):
        if not np.isfinite(rtol) or not 0<rtol<1 or not 1<=maxiter<=10000:raise ValueError('Positive tolerance and bounded iterations required')
        if self.backend=='cuda' and (low_memory is True or (low_memory is None and self.n>=1_000_000)):
            from .pcg import solve_pcg
            return solve_pcg(self,concentration,macro,temperature,rtol=rtol,maxiter=maxiter,checkpoint=checkpoint)
        if not np.isfinite(rtol) or not 0<rtol<1 or not 1<=maxiter<=10000:raise ValueError('Positive tolerance and bounded iterations required')
        xp=self.xp;start=time.perf_counter();shape=(3,*self.shape);x=xp.zeros(shape);r=self.load(concentration,macro,temperature);norm0=float(xp.linalg.norm(r));history=[]
        identity=hashlib.sha256(self.phase_host.tobytes()+self.C_host.tobytes()+self.beta_host.tobytes()+self.alpha_host.tobytes()+SOURCE.read_bytes()+np.asarray([*macro,temperature,*self.spacing]).tobytes())
        if checkpoint is not None:
            c_host=xp.asnumpy(concentration) if self.backend=='cuda' else np.asarray(concentration);identity.update(np.ascontiguousarray(c_host).tobytes());del c_host
            checkpoint=Path(checkpoint);checkpoint.mkdir(parents=True,exist_ok=True)
            record=checkpoint/'state.json'
            if record.exists():
                import json
                previous=json.loads(record.read_text())
                if previous['fingerprint']!=identity.hexdigest():raise ValueError('Checkpoint geometry/material/load/source mismatch')
                data=np.load(checkpoint/'u.npy',mmap_mode='r');x[:]=xp.asarray(data);del data
                self.apply(x,out=r);r*=-1.;r+=self.load(concentration,macro,temperature)
        if norm0==0:return x,{'iterations':0,'relative_residual':0.,'wall_seconds':time.perf_counter()-start,'converged':True}
        # Homogeneous vector-Laplacian inverse: removes the h^-2 condition
        # factor, while the exact heterogeneous operator stays FP64.
        inv=xp.zeros((self.shape[0],self.shape[1],self.shape[2]//2+1),dtype='float32' if mixed_preconditioner and self.backend=='cuda' else 'float64')
        for a,n in enumerate(self.shape):
            k=xp.arange(inv.shape[a]);term=4*xp.sin(np.pi*k/n)**2*float(np.asarray(self.spacing).min()/self.spacing[a])**2
            view=[1,1,1];view[a]=len(k);inv+=term.reshape(view)
        inv[0,0,0]=1.;xp.reciprocal(inv,out=inv);inv[0,0,0]=0.
        p=xp.empty_like(x);z=xp.empty_like(x);Ap=xp.empty_like(x);rho_old=None;converged=False
        for iteration in range(maxiter):
            context=self.budget.batch() if self.backend=='cuda' else _NullContext()
            with context:
                for k in range(3):
                    channel=r[k].astype(inv.dtype,copy=False)
                    transform=xp.fft.rfftn(channel) if self.backend=='cuda' else rfftn(channel)
                    transform*=inv
                    z[k]=xp.fft.irfftn(transform,s=self.shape,axes=(0,1,2)) if self.backend=='cuda' else irfftn(transform,s=self.shape)
                rho=float(xp.vdot(r,z).real)
                if rho<=0 or not np.isfinite(rho):raise RuntimeError('Nonpositive preconditioned residual; inspect null modes/materials')
                if rho_old is None:p[:]=z
                elif self.backend=='cuda':self._call('direction',[p,z,float(rho/rho_old),np.int64(x.size)],count=x.size)
                else:p*=rho/rho_old;p+=z
                self.apply(p,Ap);curvature=float(xp.vdot(p,Ap).real)
                if curvature<=0 or not np.isfinite(curvature):raise RuntimeError('Nonpositive CG curvature; no silent regularization')
                step=rho/curvature
                if self.backend=='cuda':self._call('update',[x,r,p,Ap,float(step),np.int64(x.size)],count=x.size)
                else:x+=step*p;r-=step*Ap
                rho_old=rho
                # Recompute the true FP64 residual before accepting a solution.
                relative=float(xp.linalg.norm(r))/norm0
                if (iteration+1)%50==0 or relative<rtol:
                    self.apply(x,Ap);self.load(concentration,macro,temperature,out=r);r-=Ap;relative=float(xp.linalg.norm(r))/norm0;rho_old=None
                history.append(relative)
            if iteration%25==0:print({'iteration':iteration+1,'relative_residual':relative},flush=True)
            if relative<rtol:converged=True;break
        metrics={'iterations':len(history),'relative_residual':history[-1],'residual_history':history,'wall_seconds':time.perf_counter()-start,'converged':converged,'backend':self.backend,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
        if self.budget:metrics['gpu']=self.budget.metadata()
        if checkpoint is not None:
            import json
            stored=np.lib.format.open_memmap(checkpoint/'u.npy',mode='w+',dtype='float64',shape=shape)
            for k in range(3):
                for begin in range(0,self.n,262144):
                    value=x[k].ravel()[begin:begin+262144];stored[k].ravel()[begin:begin+262144]=xp.asnumpy(value) if self.backend=='cuda' else value
            stored.flush();del stored
            (checkpoint/'state.json').write_text(json.dumps({'fingerprint':identity.hexdigest(),'metrics':metrics},indent=2))
        if not converged and checkpoint is None:raise RuntimeError(f'Voxel CG did not converge: {metrics["relative_residual"]}')
        return x,metrics

class _NullContext:
    def __enter__(self):return self
    def __exit__(self,*args):return False
