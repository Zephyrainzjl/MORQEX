"""Sparse mixed and matrix-free references for fixed quadratic chemoelasticity.

No N-by-N condensed Hessian or identity concentration response is constructed.
The sparse mechanical factor is still retained: its fill, time and memory must
be measured. These are established Schur/mixed/Krylov ideas, not new methods.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
import numpy as np
from scipy.sparse import coo_matrix, diags, bmat, eye
from scipy.sparse.linalg import splu, LinearOperator, gmres, cg
from .energy import _digest
from .transport import ConcentrationInfeasible
from .gradient_transport import immutable


def sparse_bytes(a):
    return int(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes)


@dataclass(frozen=True)
class SparseCoupledState:
    deviation: np.ndarray
    mechanical_coordinates: np.ndarray
    macro_strain: np.ndarray
    time: float
    energy: float
    inventory: np.ndarray
    system_fingerprint: str

    def __post_init__(self):
        for name in ('deviation','mechanical_coordinates','macro_strain','inventory'):
            object.__setattr__(self,name,immutable(getattr(self,name)))


class SparseChemoelasticSystem:
    def __init__(self, solver, diffusion, expansion, patterns, *, chemical_curvature,
                 reference_concentration, maximum_concentration):
        started = time.perf_counter()
        if solver.constraint.boundary not in {'kubc','periodic'}:
            raise ValueError('Strain-controlled fixed geometry required')
        self.solver,self.diffusion = solver,diffusion
        mesh = solver.mesh; n = len(diffusion.weights)
        if not hasattr(solver.Kr,'tocsc'): raise ValueError('Sparse assembled mechanical reference required')
        if not np.isfinite(chemical_curvature) or chemical_curvature <= 0: raise ValueError('Positive curvature required')
        if not np.isfinite([reference_concentration,maximum_concentration]).all() or not 0 < reference_concentration < maximum_concentration:
            raise ValueError('Physical reference concentration required')
        if mesh.voxel_ids is None: raise ValueError('Parent voxel provenance required')
        expected = np.array([diffusion.volume.origin,np.asarray(diffusion.volume.origin)+np.asarray(diffusion.volume.spacing)*diffusion.volume.data.shape])[:,::-1]
        if not np.allclose(mesh.box,expected,rtol=1e-12,atol=0): raise ValueError('Different reference geometry')
        mapping = np.full(diffusion.volume.data.size,-1,dtype=int);mapping[diffusion.voxel_ids] = np.arange(n)
        if np.max(mesh.voxel_ids) >= len(mapping): raise ValueError('Invalid parent voxel ID')
        element_cells = mapping[mesh.voxel_ids]; active = element_cells >= 0
        represented = np.bincount(element_cells[active],weights=mesh.volumes[active],minlength=n)
        if not np.allclose(represented,diffusion.volumes,rtol=1e-11,atol=0): raise ValueError('FV and mechanical active volumes differ')
        beta = np.broadcast_to(np.asarray(expansion,dtype=float),(len(mesh.cells),6)).copy()
        if not np.isfinite(beta).all(): raise ValueError('Finite Mandel expansion required')
        beta[~active] = 0.
        rows = (6*np.flatnonzero(active)[:,None]+np.arange(6)).ravel()
        cols = np.repeat(element_cells[active],6)
        self.T = coo_matrix((beta[active].ravel(),(rows,cols)),shape=(6*len(mesh.cells),n)).tocsr();self.T.eliminate_zeros()
        weighted_beta = np.einsum('eij,ej,e->ei',solver.stiffness,beta,mesh.volumes)
        ct = coo_matrix((weighted_beta[active].ravel(),(rows,cols)),shape=self.T.shape).tocsr();ct.eliminate_zeros()
        force = (solver.P.T @ (solver.D.T @ ct)).tocsc()
        diagonal = np.bincount(element_cells[active],weights=np.sum(beta[active]*weighted_beta[active],axis=1),minlength=n)/mesh.rve_volume
        self.n,self.kappa,self.reference,self.maximum = n,float(chemical_curvature),float(reference_concentration),float(maximum_concentration)
        self.weights = immutable(diffusion.weights)
        self.scale_c = immutable(np.sqrt(self.kappa*self.weights))
        diagonal_k = solver.Kr.diagonal()
        if len(diagonal_k) and np.any(diagonal_k <= 0): raise ValueError('Positive constrained stiffness diagonal required')
        self.inverse_mechanical_scale = 1/np.sqrt(diagonal_k)
        ds = diags(self.inverse_mechanical_scale)
        self.Km = (ds @ solver.Kr @ ds).tocsc()
        self.J = (ds @ force @ diags(1/self.scale_c)/np.sqrt(mesh.rve_volume)).tocsr()
        self.local_upper = immutable(1+diagonal/(self.kappa*self.weights))
        self.L = (diags(1/np.sqrt(self.weights)) @ diffusion.stiffness @ diags(1/np.sqrt(self.weights))).tocsr()
        pattern = np.asarray(patterns,dtype=float)
        if pattern.ndim != 2 or pattern.shape[0] != len(diffusion.face_areas) or not np.isfinite(pattern).all(): raise ValueError('Finite fixed face channels required')
        self.b = immutable(np.asarray(diffusion.flux_operator @ pattern)*np.sqrt(self.kappa/self.weights)[:,None])
        affine_strain = np.asarray(solver.D @ solver.affine).reshape(len(mesh.cells),6,6)
        affine_stress = np.einsum('eij,ejk->eik',solver.stiffness,affine_strain)
        self.R = immutable(np.asarray(ds @ (solver.P.T @ (solver.K @ solver.affine)))/np.sqrt(mesh.rve_volume))
        self.G0 = immutable(-np.asarray(self.T.T @ (mesh.volumes[:,None,None]*affine_stress).reshape(-1,6))/mesh.rve_volume/self.scale_c[:,None])
        self.macro_clamped = immutable(np.einsum('eik,eil,e->kl',affine_strain,affine_stress,mesh.volumes)/mesh.rve_volume)
        self.mechanical_solves = 0
        self.v_macro = immutable(self.solve_mechanical(-self.R))
        self.G = immutable(self.G0-self.J.T @ self.v_macro)
        self.macro_H = immutable(self.macro_clamped+self.R.T @ self.v_macro)
        self.inventory_rows = np.array([diffusion.volumes*(diffusion.components == j) for j in range(diffusion.n_components)])
        self.baseline_inventory = self.inventory_rows @ np.full(n,self.reference)
        self.fingerprint = _digest(mesh.points,mesh.cells,solver.stiffness,beta,diffusion.voxel_ids,pattern,
            diffusion.stiffness.data,diffusion.stiffness.indices,diffusion.stiffness.indptr,
            np.r_[self.kappa,self.reference,self.maximum],np.frombuffer(Path(__file__).read_bytes(),dtype=np.uint8),
            np.frombuffer(solver.constraint.boundary.encode(),dtype=np.uint8))
        self._mixed_dt,self._mixed_factor,self._mixed_matrix = None,None,None
        self._krylov_dt,self._krylov_preconditioner = None,None
        self.mixed_factorizations,self.krylov_factorizations = 0,0
        self.construction_seconds = time.perf_counter()-started

    def solve_mechanical(self,rhs):
        self.mechanical_solves += 1
        rhs = np.asarray(rhs)
        # Km = Dk^-1/2 Kr Dk^-1/2; reuse the original checked Kr factor.
        return self.solver.solve_reduced(rhs/self.inverse_mechanical_scale[(slice(None),)+(None,)*(rhs.ndim-1)]) / self.inverse_mechanical_scale[(slice(None),)+(None,)*(rhs.ndim-1)]

    def h_action(self,z):
        z = np.asarray(z,dtype=float)
        if z.ndim not in (1,2) or z.shape[0] != self.n or not np.isfinite(z).all(): raise ValueError('Invalid Hessian action')
        weighted = self.local_upper[(slice(None),)+(None,)*(z.ndim-1)]*z
        return weighted-self.J.T @ self.solve_mechanical(self.J @ z)

    def solve_h(self,rhs,*,relative_tolerance=1e-12):
        rhs = np.asarray(rhs,dtype=float)
        matrix = LinearOperator((self.n,self.n),matvec=self.h_action,dtype=float)
        preconditioner = LinearOperator((self.n,self.n),matvec=lambda x:x/self.local_upper,dtype=float)
        result,info = cg(matrix,rhs,M=preconditioner,rtol=relative_tolerance,atol=0.,maxiter=300)
        if info: raise RuntimeError('Hessian action CG failed; no mass-Riesz direction returned')
        if np.linalg.norm(self.h_action(result)-rhs) > 5*relative_tolerance*max(np.linalg.norm(rhs),1e-30): raise RuntimeError('Hessian inverse residual failed')
        return result

    def fields(self,c,e):
        eigen = np.asarray(self.T @ np.asarray(c)).reshape(-1,6)
        return self.solver.solve(eigen,macro_strain=e)

    def energy(self,z,v,e):
        return float(.5*v @ (self.Km @ v)-v @ (self.J @ z)+.5*np.dot(self.local_upper*z,z)+v @ (self.R @ e)+z @ (self.G0 @ e)+.5*e @ self.macro_clamped @ e)

    def initialize(self,c,*,macro_strain=None):
        c = np.asarray(c,dtype=float);e = np.zeros(6) if macro_strain is None else np.asarray(macro_strain,dtype=float)
        if c.shape != (self.n,) or e.shape != (6,) or not np.isfinite(np.r_[c,e]).all(): raise ValueError('Invalid initial state')
        self._physical(c)
        z = self.scale_c*c;v = self.solve_mechanical(self.J @ z)+self.v_macro @ e
        return SparseCoupledState(c,v,e,0.,self.energy(z,v,e),self.baseline_inventory+self.inventory_rows @ c,self.fingerprint)

    def _physical(self,c):
        if np.any(self.reference+c < 0) or np.any(self.reference+c > self.maximum): raise ConcentrationInfeasible('Full coupled concentration outside physical limits')

    def prepare(self,dt,method):
        if not np.isfinite(dt) or dt <= 0: raise ValueError('Positive dt required')
        if method == 'mixed-LU':
            if self._mixed_dt != dt:
                # Retaining the old factor while factoring a new step can
                # nearly double peak fill memory. No old-step factor is used.
                self._mixed_factor = None
                n=self.n
                self._mixed_matrix = bmat([[self.Km,-self.J,None],[-self.J.T,diags(self.local_upper),-eye(n)],[None,eye(n),dt*self.L]],format='csc')
                self._mixed_factor = splu(self._mixed_matrix)
                self._mixed_dt=dt;self.mixed_factorizations += 1
        elif method == 'Schur-GMRES':
            if self._krylov_dt != dt:
                self._krylov_preconditioner = None
                self._krylov_preconditioner = splu((eye(self.n)+dt*self.L @ diags(self.local_upper)).tocsc())
                self._krylov_dt=dt;self.krylov_factorizations += 1
        else: raise ValueError('Unknown full coupled reference method')

    def step(self,previous,dt,u,*,macro_strain=None,method='Schur-GMRES',relative_tolerance=1e-10):
        if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1: raise ValueError('Positive relative solver tolerance below one required')
        e = previous.macro_strain if macro_strain is None else np.asarray(macro_strain,dtype=float)
        u = np.asarray(u,dtype=float)
        if e.shape != (6,) or u.shape != (self.b.shape[1],) or not np.isfinite(np.r_[e,u]).all(): raise ValueError('Invalid full coupled input')
        if previous.deviation.shape != (self.n,) or previous.system_fingerprint != self.fingerprint: raise ValueError('State/system mismatch')
        self.prepare(dt,method)
        old = self.scale_c*previous.deviation
        before=self.mechanical_solves;iterations=0
        if method == 'mixed-LU':
            rhs=np.r_[-self.R @ e,-self.G0 @ e,old+dt*self.b @ u]
            solution=self._mixed_factor.solve(rhs)
            nv=self.Km.shape[0];v,z,mu=solution[:nv],solution[nv:nv+self.n],solution[nv+self.n:]
            algebra=np.linalg.norm(self._mixed_matrix @ solution-rhs)/max(np.linalg.norm(rhs),1e-30)
        else:
            rhs = old+dt*(self.b @ u-self.L @ (self.G @ e))
            operator=LinearOperator((self.n,self.n),matvec=lambda x:x+dt*self.L @ self.h_action(x),dtype=float)
            pre=LinearOperator((self.n,self.n),matvec=self._krylov_preconditioner.solve,dtype=float)
            def count(_):
                nonlocal iterations
                iterations+=1
            z,info=gmres(operator,rhs,x0=old,M=pre,rtol=relative_tolerance,atol=0.,restart=30,maxiter=100,callback=count,callback_type='pr_norm')
            if info: raise RuntimeError('Full Schur GMRES did not converge')
            v=self.solve_mechanical(self.J @ z)+self.v_macro @ e
            mu=self.local_upper*z-self.J.T @ v+self.G0 @ e
            algebra=np.linalg.norm(z+dt*self.L @ mu-rhs-dt*self.L @ (self.G @ e))/max(np.linalg.norm(rhs),1e-30)
        c=z/self.scale_c;self._physical(c)
        energy=self.energy(z,v,e)
        inventory=self.baseline_inventory+self.inventory_rows @ c
        old_at_new_e=previous.mechanical_coordinates+self.v_macro @ (e-previous.macro_strain)
        work=self.energy(old,old_at_new_e,e)-previous.energy
        delta=z-old;delta_v=v-old_at_new_e
        numerical=.5*(delta_v @ (self.Km @ delta_v)-2*delta_v @ (self.J @ delta)+np.dot(self.local_upper*delta,delta))
        dissipation=float(mu @ (self.L @ mu));power=float(mu @ self.b @ u)
        balance=energy-previous.energy-work-dt*(power-dissipation)+numerical
        mass_error=inventory-previous.inventory-dt*(self.inventory_rows/self.scale_c) @ (self.b @ u)
        if algebra > max(5*relative_tolerance,1e-11): raise RuntimeError('Full coupled algebraic residual failed')
        state=SparseCoupledState(c,v,e,previous.time+dt,energy,inventory,self.fingerprint)
        return state,{'relative_algebraic_residual':float(algebra),'energy_balance':float(balance),'component_mass_error':mass_error,
            'iterations':iterations,'mechanical_queries':self.mechanical_solves-before,'diffusion_dissipation':dissipation,
            'external_chemical_power':power,'mechanical_work':work,'numerical_dissipation':float(numerical)}

    def retained_sparse_bytes(self):
        return sum(sparse_bytes(a) for a in (self.T,self.Km,self.J,self.L))

    def coupled_resolvent_basis(self,count,*,shifts=(.01,.1,1.,10.),moments=3):
        """Established rational Krylov responses of the actual coupled generator.

        Fixed declared input directions only, no validation trajectories.
        All block factorizations/RHS and the complete library cost are returned.
        """
        from .concentration import ConcentrationBasis,component_constants,_orthogonal
        shifts=np.asarray(shifts,dtype=float)
        if shifts.ndim != 1 or not len(shifts) or not np.isfinite(shifts).all() or np.any(shifts <= 0) or moments < 1:
            raise ValueError('Positive resolvent shifts and moment count required')
        if not self.diffusion.n_components <= count <= self.n: raise ValueError('Invalid coupled space size')
        columns=[];rhs_count=0;started=time.perf_counter();factor_before=self.mixed_factorizations
        nr=self.Km.shape[0]
        for shift in shifts:
            dt=1/shift;self.prepare(dt,'mixed-LU')
            right=self.b
            for _ in range(moments):
                block=np.vstack([np.zeros((nr+self.n,right.shape[1])),dt*right])
                solution=self._mixed_factor.solve(block)
                right=solution[nr:nr+self.n]
                columns.append(right/self.scale_c[:,None]);rhs_count+=right.shape[1]
        values=_orthogonal(np.column_stack(columns),self.weights,initial=component_constants(self.diffusion),count=count)
        return ConcentrationBasis(values,self.weights.copy(),'coupled-resolvent',self.diffusion.n_components),{
            'complete_library_seconds':time.perf_counter()-started,'mixed_factorizations':self.mixed_factorizations-factor_before,'right_hand_sides':rhs_count}
