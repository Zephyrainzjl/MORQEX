"""Objective finite-strain chemical elasticity with an upstream MFEM kernel.

The reference energy is defined per initial material volume. Isotropic elastic
moduli, multiplicative (possibly anisotropic) chemical deformation, fixed
material geometry, dead nominal loads. No claim of crack contact, wetting,
calibrated CBD rheology, or general anisotropic crystal elasticity.
"""
from dataclasses import dataclass
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, minres, LinearOperator
from .backend import NativeConfig, MFEMScalarBackend, require_local_budget, _sha, _LOCK
from .phase_field import _field
from morqex.workspace import Workspace
from morqex.mechanics.reference import ElasticitySolver, MechanicalSolveError

_NATIVE = Path(__file__).with_name('native_hyperelastic')


def _identity(base):
    inputs = {'adapter': {p.name: _sha(p) for p in sorted(_NATIVE.iterdir()) if p.is_file()},
              'mfem': base.manifest['dependency_sha256'], 'mfem_build': base.manifest['identity'],
              'compiler': subprocess.check_output(['c++','--version'],text=True).splitlines()[0]}
    return hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(), inputs


def build_hyperelastic(config, workspace):
    """Compile only the small adapter, reusing the verified MFEM dependency."""
    require_local_budget()
    if config.provider != 'vendored':
        raise ValueError('Finite-strain adapter currently uses the vendored MFEM build')
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    base=MFEMScalarBackend(config,ws); identity,inputs=_identity(base)
    out=ws.path('build/top_damage_hyperelastic',identity[:20]);out.mkdir(parents=True,exist_ok=True)
    upstream=Path(__file__).parents[1]/'vendor/mfem'
    dependency=Path(base.manifest['dependency']); started=time.perf_counter()
    ninja=shutil.which('ninja') or str(Path(sys.executable).parent/'ninja')
    with (out/'build.log').open('a') as log:
        subprocess.run(['cmake','-S',str(_NATIVE),'-B',str(out),'-G','Ninja',
                        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_MAKE_PROGRAM='+ninja,'-DMFEM_SOURCE='+str(upstream),
                        '-DMFEM_BUILD='+str(dependency.parent),'-DMFEM_LIBRARY='+str(dependency)],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
        subprocess.run(['cmake','--build',str(out),'--parallel','1'],stdout=log,
                       stderr=subprocess.STDOUT,check=True,timeout=config.slice_seconds)
    lib=out/'libmorqex_hyperelastic.so'
    result={'state':'complete','identity':identity,'inputs':inputs,'library':str(lib),
            'library_sha256':_sha(lib),'dependency':str(dependency),
            'dependency_sha256':_sha(dependency),'seconds':time.perf_counter()-started}
    (out/'manifest.json').write_text(json.dumps(result,indent=2));return result


class HyperelasticBackend:
    def __init__(self, config, workspace):
        require_local_budget()
        ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
        self.base=MFEMScalarBackend(config,ws)
        identity,_=_identity(self.base)
        self.manifest=json.loads(ws.path('build/top_damage_hyperelastic',identity[:20],'manifest.json').read_text())
        m=self.manifest
        if m['identity']!=identity or m['state']!='complete' or _sha(m['library'])!=m['library_sha256'] or _sha(m['dependency'])!=m['dependency_sha256']:
            raise ValueError('Missing or modified finite-strain backend')
        self.lib=ct.CDLL(m['library']);self.lib.mq_hyperelastic_abi.restype=ct.c_int
        if self.lib.mq_hyperelastic_abi()!=2:raise ValueError('Wrong finite-strain ABI')
        ptr=ct.POINTER(ct.c_double)
        self.lib.mq_neo_batch.argtypes=[ct.c_int]+[ptr]*9+[ct.c_char_p,ct.c_int]
        self.lib.mq_neo_batch.restype=ct.c_int
        self.lib.mq_tet_hvp.argtypes=[ct.c_int,ct.c_int,ct.POINTER(ct.c_longlong)]+[ptr]*5
        self.lib.mq_tet_hvp.restype=ct.c_int

    def tangent_action(self, mesh, tangent):
        """Compiled serial FP64 action; uses the same MFEM material tangent."""
        dofs=np.ascontiguousarray(mesh.dofs,dtype=np.int64)
        gradients=np.ascontiguousarray(mesh.gradients,dtype=float)
        volumes=np.ascontiguousarray(mesh.volumes,dtype=float)
        C=np.ascontiguousarray(tangent,dtype=float)
        n=len(mesh.cells);nd=3*len(mesh.points)
        if C.shape!=(n,9,9) or not np.isfinite(C).all():raise ValueError('Finite cell tangents required')
        dp=lambda x:x.ctypes.data_as(ct.POINTER(ct.c_double))
        def action(vector):
            v=np.ascontiguousarray(vector,dtype=float).reshape(-1)
            if v.shape!=(nd,) or not np.isfinite(v).all():raise ValueError('Finite compatible tangent vector required')
            out=np.empty(nd)
            code=self.lib.mq_tet_hvp(n,nd,dofs.ctypes.data_as(ct.POINTER(ct.c_longlong)),
                dp(gradients),dp(C),dp(volumes),dp(v),dp(out))
            if code:raise ValueError('Compiled Tet4 action rejected its dimensions')
            return out
        return action

    def evaluate(self, F, chemical_F, shear, bulk, degradation=1., *, tangent=True):
        require_local_budget()
        f=np.ascontiguousarray(F,dtype=float)
        if f.ndim!=3 or f.shape[1:]!=(3,3) or len(f)==0 or not np.isfinite(f).all():
            raise ValueError('Finite nonempty (cells,3,3) deformation required')
        n=len(f);fc=np.array(np.broadcast_to(chemical_F,f.shape),float,copy=True)
        if not np.isfinite(fc).all() or np.any(np.linalg.det(fc)<=1e-12):
            raise ValueError('Chemical deformation must preserve orientation')
        a=np.ascontiguousarray(np.linalg.inv(fc))
        mu=_field(shear,n,'shear',True);k=_field(bulk,n,'bulk',True);g=_field(degradation,n,'degradation',True)
        if np.any(g>1):raise ValueError('Degradation exceeds one')
        w=np.empty(n);plus=np.empty(n);p=np.empty_like(f)
        c=np.empty((n,9,9)) if tangent else None
        dp=lambda x: x.ctypes.data_as(ct.POINTER(ct.c_double))
        err=ct.create_string_buffer(4096)
        with _LOCK:
            code=self.lib.mq_neo_batch(n,dp(f),dp(a),dp(mu),dp(k),dp(g),dp(w),dp(plus),dp(p),dp(c) if tangent else None,err,len(err))
        if code:raise ValueError(err.value.decode())
        if not all(np.isfinite(v).all() for v in (w,plus,p)) or (tangent and not np.isfinite(c).all()):
            raise FloatingPointError('Nonfinite constitutive response')
        return w,plus,p,c


def chemical_potential(P, F, chemical_F, derivative_chemical_F):
    """Elastic dW/dc0 at fixed total F, frozen history and fixed elastic moduli.

    For concentration-dependent moduli add their energy derivatives explicitly;
    this helper deliberately does not accept or conceal such dependence.
    """
    p,f,fc,dfc=np.broadcast_arrays(P,F,chemical_F,derivative_chemical_F)
    if p.shape[-2:]!=(3,3) or not all(np.isfinite(v).all() for v in (p,f,fc,dfc)):
        raise ValueError('Finite compatible deformation tensors required')
    if np.any(np.linalg.det(fc)<=1e-12):raise ValueError('Invalid chemical deformation')
    # P=Pe A^T; dFe/dc=-F A Fc_prime A; Pe=P Fc^T.
    pe=p@np.swapaxes(fc,-1,-2);a=np.linalg.inv(fc)
    return -np.sum(pe*(f@a@dfc@a),axis=(-2,-1))


@dataclass(frozen=True)
class HyperelasticState:
    displacement_m: np.ndarray
    deformation_gradient: np.ndarray
    first_piola_Pa: np.ndarray
    cauchy_stress_Pa: np.ndarray
    tensile_energy_J_m3: np.ndarray
    elastic_energy_J: float
    potential_energy_J: float
    reaction_N: np.ndarray
    equilibrium_residual: float
    newton_iterations: int
    minimum_jacobian: float
    equilibrium_residual_N: float
    equilibrium_force_scale_N: float


class Hyperelasticity:
    """Total-Lagrangian Tet4 reference; full 3x3 prescribed macro F.

    Sparse Newton is intended for bounded reference cases. Large cases need
    separately profiled matrix-free/preconditioned solves. Linear constraint
    maps are reused, not the old stiffness response or old stress fields.
    """
    def __init__(self, mesh, backend, young_Pa, poisson, *, boundary='kubc', linear_solver='direct'):
        if linear_solver not in ('direct','minres','amg-minres','matrix-free-amg-minres','numpy-matrix-free-amg-minres'):raise ValueError('Unknown finite-strain linear solver')
        self.mesh=mesh;self.backend=backend;self.linear_solver=linear_solver
        self._geometry_fingerprint=mesh.fingerprint
        n=len(mesh.cells);self.young=_field(young_Pa,n,'Young modulus',True)
        nu=np.array(np.broadcast_to(poisson,(n,)),float,copy=True)
        if not np.isfinite(nu).all() or np.any((nu<=-1)|(nu>=.5)):raise ValueError('Poisson ratio must be in (-1,.5)')
        self.shear=self.young/(2*(1+nu));self.bulk=self.young/(3*(1-2*nu))
        identity=np.array([1.,1.,1.,0.,0.,0.])
        C=2*self.shear[:,None,None]*np.eye(6)+(self.bulk-2*self.shear/3)[:,None,None]*np.outer(identity,identity)
        self.base=ElasticitySolver(mesh,C,boundary=boundary,factorize=False)
        self.P=self.base.P
        # B[e,iJ,ak]=delta_ik N_a,J; total F=I+B u.
        self.B=np.einsum('ik,eaJ->eiJak',np.eye(3),mesh.gradients).reshape(n,9,12)
        self.matrix_free='matrix-free' in linear_solver
        self.compiled_action=self.matrix_free and not linear_solver.startswith('numpy-')
        if not self.matrix_free:
            self.rows=np.broadcast_to(mesh.dofs[:,:,None],(n,12,12)).ravel()
            self.cols=np.broadcast_to(mesh.dofs[:,None,:],(n,12,12)).ravel()
        self.preconditioner=None;self.linear_iterations=[];self.solve_diagnostics=[]
        if 'amg' in linear_solver and self.P.shape[1]:
            import pyamg
            from morqex.mechanics.constraints import _rigid
            rigid=_rigid(mesh.points,float(np.max(np.ptp(mesh.points,axis=0)))).reshape(-1,6)
            ml=pyamg.smoothed_aggregation_solver((self.base.Kr/self.base.scale).tocsr(),
                B=np.asarray(self.P.T@rigid),symmetry='symmetric',max_coarse=48,
                presmoother=('gauss_seidel',{'sweep':'symmetric'}),
                postsmoother=('gauss_seidel',{'sweep':'symmetric'}))
            amg=ml.aspreconditioner();self.preconditioner=LinearOperator(amg.shape,matvec=lambda x:amg@x/self.base.scale)

    def assemble(self,u,chemical_F,degradation=1.,*,tangent=True):
        if self.mesh.fingerprint!=self._geometry_fingerprint:
            raise ValueError('Geometry changed: rebuild constraints, gradients and preconditioner')
        u=np.asarray(u,float).reshape(-1)
        if u.shape!=(3*len(self.mesh.points),) or not np.isfinite(u).all():raise ValueError('Invalid displacement')
        F=np.eye(3)+np.einsum('eij,ej->ei',self.B,u[self.mesh.dofs]).reshape(-1,3,3)
        w,plus,p,c=self.backend.evaluate(F,chemical_F,self.shear,self.bulk,degradation,tangent=tangent)
        force=np.zeros(len(u));local=np.einsum('eij,ei,e->ej',self.B,p.reshape(-1,9),self.mesh.volumes)
        np.add.at(force,self.mesh.dofs.ravel(),local.ravel())
        A=None
        if tangent:
            if self.matrix_free:
                def action(v):
                    dv=np.einsum('eij,ej->ei',self.B,v[self.mesh.dofs],optimize=True)
                    dp=np.einsum('eij,ej->ei',c,dv,optimize=True)
                    nodal=np.einsum('eij,ei,e->ej',self.B,dp,self.mesh.volumes,optimize=True)
                    out=np.zeros(len(u));np.add.at(out,self.mesh.dofs.ravel(),nodal.ravel());return out
                if self.compiled_action:action=self.backend.tangent_action(self.mesh,c)
                A=LinearOperator((len(u),len(u)),matvec=action,rmatvec=action,dtype=float)
            else:
                localA=np.einsum('eia,eij,ejb,e->eab',self.B,c,self.B,self.mesh.volumes,optimize=True)
                A=coo_matrix((localA.ravel(),(self.rows,self.cols)),shape=(len(u),len(u))).tocsr()
        return float(self.mesh.volumes@w),force,A,F,p,plus

    def solve(self, chemical_F=np.eye(3), degradation=1., *, macro_F=None, force=None,
              initial_displacement=None,tolerance=1e-9,max_iterations=50):
        n=len(self.mesh.points);nd=3*n
        if macro_F is not None and self.base.constraint.boundary=='neumann':
            raise ValueError('Neumann problem uses nominal forces, not macro F')
        fm=np.eye(3) if macro_F is None else np.asarray(macro_F,float)
        if fm.shape!=(3,3) or not np.isfinite(fm).all() or np.linalg.det(fm)<=1e-12:raise ValueError('Invalid macro F')
        lift=((self.mesh.points-self.mesh.box.mean(axis=0))@(fm-np.eye(3)).T).ravel()
        external=np.zeros(nd) if force is None else np.asarray(force,float).reshape(-1)
        if external.shape!=(nd,) or not np.isfinite(external).all():raise ValueError('Invalid nominal load')
        u=lift.copy()
        if initial_displacement is not None:
            initial=np.asarray(initial_displacement,float).reshape(-1)
            if initial.shape!=(nd,) or not np.isfinite(initial).all():raise ValueError('Invalid warm start')
            norm=np.asarray(self.P.power(2).sum(axis=0)).ravel()
            u+=self.P@(np.asarray(self.P.T@(initial-lift))/norm)
        force_scale=self.young.max()*np.max(np.ptp(self.mesh.points,axis=0))**2
        # A pure traction problem has internal(u) -> 0 at equilibrium. Dividing
        # by that shrinking vector falsely demands near-zero/near-zero accuracy.
        # Freeze the norm using the chemical/macro lift and actual applied load;
        # include the first trial force when unloading from a warm start.
        _,lift_force,*_=self.assemble(lift,chemical_F,degradation,tangent=False)
        load_scale=max(np.linalg.norm(lift_force),np.linalg.norm(external),force_scale*1e-9)
        # EvalW subtracts O(modulus) invariants to obtain the much smaller
        # strain energy. Near equilibrium, Armijo alone can reject a Newton
        # correction whose force residual is already at roundoff. This is an
        # arithmetic noise scale, NOT a physical error bound or a relaxed
        # equilibrium tolerance. Require residual decrease as a second merit
        # only when predicted work is below this scale.
        roundoff_energy=64*np.finfo(float).eps*float(self.mesh.volumes@(self.shear+self.bulk))
        fallback_count=0;backtracks=0;trace=[]
        for iteration in range(max_iterations+1):
            energy,internal,A,F,p,plus=self.assemble(u,chemical_F,degradation)
            r=internal-external;rr=np.asarray(self.P.T@r)
            if iteration==0:load_scale=max(load_scale,np.linalg.norm(internal))
            scale=load_scale
            residual=float(np.linalg.norm(rr)/scale)
            physical=float(np.linalg.norm(self.base.constraint.equilibrium_projection.T@r)/scale)
            potential=energy-float(external@u)
            trace.append({'iteration':iteration,'residual':residual,'potential_J':potential,
                          'residual_N':float(np.linalg.norm(rr)),'force_scale_N':float(scale)})
            if residual<=tolerance:
                if physical>max(tolerance,1e-7):raise MechanicalSolveError('Unbalanced nominal load on gauges')
                self.solve_diagnostics.append({'newton_iterations':iteration,
                    'roundoff_merit_acceptances':fallback_count,'backtracks':backtracks,
                    'roundoff_energy_scale_J':roundoff_energy,'trace':trace})
                J=np.linalg.det(F);sigma=(p@np.swapaxes(F,1,2))/J[:,None,None]
                return HyperelasticState(u.reshape(-1,3),F,p,sigma,plus,energy,potential,r.reshape(-1,3),physical,iteration,float(J.min()),
                    float(np.linalg.norm(self.base.constraint.equilibrium_projection.T@r)),float(scale))
            if iteration==max_iterations:break
            if self.matrix_free:
                reduced=LinearOperator((self.P.shape[1],)*2,matvec=lambda x:np.asarray(self.P.T@(A@(self.P@x))),dtype=float)
            else:reduced=(self.P.T@A@self.P).tocsc()
            if self.linear_solver=='direct':du=spsolve(reduced,-rr)
            else:
                pre=self.preconditioner
                if pre is None:
                    diagonal=np.abs(reduced.diagonal());floor=max(diagonal.max()*1e-12,np.finfo(float).tiny)
                    pre=LinearOperator(reduced.shape,matvec=lambda x:x/np.maximum(diagonal,floor))
                iterations=[0]
                def count(_):iterations[0]+=1
                du,info=minres(reduced,-rr,M=pre,rtol=min(tolerance*.001,1e-12),maxiter=4000,callback=count)
                self.linear_iterations.append(iterations[0])
                if info!=0:raise MechanicalSolveError('Finite-strain MINRES did not converge')
            if not np.isfinite(du).all():raise MechanicalSolveError('Singular finite-strain tangent')
            defect=np.linalg.norm(reduced@du+rr)/max(np.linalg.norm(rr),np.finfo(float).tiny)
            if defect>max(tolerance*.1,1e-7):raise MechanicalSolveError('Finite-strain linear residual failed')
            slope=float(rr@du)
            if slope>=0:raise MechanicalSolveError('Non-descending Newton step; refine load increment')
            direction=np.asarray(self.P@du);step=1.
            for _ in range(35):
                trial=u+step*direction
                try:
                    new_energy,new_force,*_=self.assemble(trial,chemical_F,degradation,tangent=False)
                except ValueError:
                    step*=.5;continue
                new_potential=new_energy-float(external@trial)
                if new_potential<=potential+1e-4*step*slope+1e-13*max(abs(potential),force_scale*np.max(np.ptp(self.mesh.points,axis=0))*1e-8):
                    u=trial;break
                if abs(step*slope)<=roundoff_energy and new_potential<=potential+roundoff_energy:
                    trial_residual=np.linalg.norm(self.P.T@(new_force-external))
                    if trial_residual<(1-1e-4*step)*np.linalg.norm(rr):
                        fallback_count+=1;u=trial;break
                backtracks+=1
                step*=.5
            else:raise MechanicalSolveError('Finite-strain positive-J line search failed')
        raise MechanicalSolveError('Finite-strain Newton did not converge; accepted state unchanged')


@dataclass(frozen=True)
class FiniteCoupledStep:
    damage: np.ndarray
    mechanics: HyperelasticState
    fatigue: object
    stagger_iterations: int
    damage_kkt: float
    fixed_history_energy_trace_J: np.ndarray
    fatigue_fracture_energy_change_J: float


def finite_staggered_step(mechanics, damage_model, accepted_damage, accepted_history,
                          chemical_F=np.eye(3), *, macro_F=None, force=None,
                          fatigue=True, threshold_J_m3=None, tolerance=1e-7,
                          max_iterations=100, initial_displacement=None):
    """Frozen-history variational AT2 step; only accepted endpoints add fatigue.

    Fixed reference topology, finite-strain isotropic split. This does not
    replace a contact/aperture model or solve local lithium diffusion.
    """
    if mechanics.mesh.fingerprint!=damage_model.mesh.fingerprint:
        raise ValueError('Mechanics and phase field must share a material mesh')
    if len(accepted_history.accumulated_J_m3)!=len(mechanics.mesh.cells):raise ValueError('History size mismatch')
    threshold=damage_model.Gc/(12*damage_model.ell) if threshold_J_m3 is None else threshold_J_m3
    factor=accepted_history.toughness_factor(threshold) if fatigue else np.ones(len(mechanics.mesh.cells))
    lower=damage_model._damage(accepted_damage).copy();d=lower.copy();u=initial_displacement;trace=[]
    for iteration in range(1,max_iterations+1):
        state=mechanics.solve(chemical_F,damage_model.mean_degradation(d),macro_F=macro_F,force=force,
                              initial_displacement=u,tolerance=min(tolerance*.01,1e-9))
        psi=np.maximum(state.tensile_energy_J_m3,0.)
        energy=state.potential_energy_J+damage_model.energies(d,psi,factor)[1]
        if trace and energy>trace[-1]+1e-8*max(abs(trace[-1]),np.finfo(float).tiny):
            raise MechanicalSolveError('Finite alternating minimization increased frozen-history energy')
        trace.append(energy)
        solved=damage_model.solve(psi,lower,factor,tolerance=min(tolerance*.01,1e-10))
        if np.max(np.abs(solved.damage-d))<tolerance:
            A,b=damage_model.assemble(psi,factor)
            kkt=float(np.max(np.abs(d-np.clip(d-(A@d-b)/A.diagonal(),lower,1))))
            if kkt<tolerance:
                history=accepted_history.trial(damage_model.mean_degradation(d)*psi) if fatigue else accepted_history
                changed=history.toughness_factor(threshold) if fatigue else factor
                delta=damage_model.energies(d,psi,changed)[1]-damage_model.energies(d,psi,factor)[1]
                d.flags.writeable=False
                return FiniteCoupledStep(d,state,history,iteration,kkt,np.asarray(trace),delta)
        d=solved.damage.copy();u=state.displacement_m
    raise MechanicalSolveError('Finite stagger failed; accepted damage/history unchanged')
