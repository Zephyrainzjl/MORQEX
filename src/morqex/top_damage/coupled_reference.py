"""Small-strain variational mechanics/AT2 reference on existing MORQEX Tet4.

Volumetric/deviatoric split, isotropic elastic moduli and potentially anisotropic
chemical eigenstrain. Compression does not drive volumetric fracture. Shear
still contributes to psi+; this model is not a contact or crystal fracture law.
Split baseline: Amor, Marigo, Maurini (2009), doi:10.1016/j.jmps.2009.04.011.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from morqex.mechanics.reference import ElasticitySolver, MechanicalSolveError
from .phase_field import _field, FatigueHistory

_I = np.array([1.,1.,1.,0.,0.,0.])
_VOL = np.outer(_I,_I)
_DEV = np.eye(6)-_VOL/3


@dataclass(frozen=True)
class SplitState:
    displacement_m: np.ndarray
    strain: np.ndarray
    mean_cell_stress_Pa: np.ndarray
    tensile_energy_J_m3: np.ndarray
    elastic_energy_J: float
    potential_energy_J: float
    reaction_N: np.ndarray
    equilibrium_residual: float
    newton_iterations: int


class SplitElasticity:
    def __init__(self, damage_model, young_Pa, poisson, *, boundary='kubc'):
        self.damage_model=damage_model; self.mesh=damage_model.mesh
        n=len(self.mesh.cells)
        self.young=_field(young_Pa,n,'Young modulus',True)
        self.poisson=np.array(np.broadcast_to(poisson,(n,)),float,copy=True)
        if not np.isfinite(self.poisson).all() or np.any((self.poisson<=-1)|(self.poisson>=.5)):
            raise ValueError('Isotropic Poisson ratio must lie in (-1, .5)')
        self.bulk=self.young/(3*(1-2*self.poisson))
        self.shear=self.young/(2*(1+self.poisson))
        stiffness=2*self.shear[:,None,None]*_DEV+self.bulk[:,None,None]*_VOL
        self.base=ElasticitySolver(self.mesh,stiffness,boundary=boundary,factorize=False)
        self.D=self.base.D;self.P=self.base.P
        indices=6*np.arange(n)[:,None]+np.arange(6)
        self.rows=np.broadcast_to(indices[:,:,None],(n,6,6)).ravel()
        self.cols=np.broadcast_to(indices[:,None,:],(n,6,6)).ravel()

    def constitutive(self, elastic_strain, degradation):
        e=np.asarray(elastic_strain,float)
        if e.shape!=(len(self.mesh.cells),6) or not np.isfinite(e).all():
            raise ValueError('Finite cell Mandel strain required')
        g=_field(degradation,len(e),'mean degradation',True)
        if np.any(g>1+1e-14):raise ValueError('Degradation exceeds intact stiffness')
        trace=e@_I;dev=e-trace[:,None]*_I/3
        positive=np.maximum(trace,0);negative=np.minimum(trace,0)
        psi_plus=.5*self.bulk*positive**2+self.shear*np.sum(dev*dev,axis=1)
        psi_minus=.5*self.bulk*negative**2
        stress=g[:,None]*(self.bulk[:,None]*positive[:,None]*_I+2*self.shear[:,None]*dev)+self.bulk[:,None]*negative[:,None]*_I
        tangent=(2*g*self.shear)[:,None,None]*_DEV+(self.bulk*np.where(trace>=0,g,1))[:,None,None]*_VOL
        return stress,tangent,psi_plus,g*psi_plus+psi_minus

    def solve(self, damage, eigenstrain=0., *, macro_strain=None, force=None,
              initial_displacement=None, tolerance=1e-9, max_iterations=40):
        g=self.damage_model.mean_degradation(damage);n=len(self.mesh.cells)
        eps=np.array(np.broadcast_to(eigenstrain,(n,6)),float,copy=True)
        if not np.isfinite(eps).all():raise ValueError('Finite eigenstrain required')
        if macro_strain is not None and self.base.constraint.boundary=='neumann':
            raise ValueError('Neumann problem uses forces rather than macro strain')
        lift=np.zeros(self.D.shape[1]) if macro_strain is None else self.base.affine@np.asarray(macro_strain,float)
        external=np.zeros_like(lift) if force is None else np.asarray(force,float).reshape(-1)
        if external.shape!=lift.shape or not np.isfinite(external).all() or not np.isfinite(lift).all():raise ValueError('Invalid load')
        # P columns identify independent DOFs; projecting a warm start requires
        # the diagonal P.T P normalization for periodic equivalence classes.
        if initial_displacement is None:u=lift.copy()
        else:
            initial=np.asarray(initial_displacement,float).reshape(-1)
            if initial.shape!=lift.shape or not np.isfinite(initial).all():raise ValueError('Invalid warm start')
            norm=np.asarray(self.P.power(2).sum(axis=0)).ravel()
            u=lift+self.P@(np.asarray(self.P.T@(initial-lift))/norm)
        volumes=self.mesh.volumes
        for iteration in range(max_iterations+1):
            strain=np.asarray(self.D@u).reshape(n,6)
            stress,C,plus,density=self.constitutive(strain-eps,g)
            internal=np.asarray(self.D.T@(volumes[:,None]*stress).ravel())
            r=internal-external;reduced=np.asarray(self.P.T@r)
            elastic=float(volumes@density);potential=elastic-float(external@u)
            scale=max(np.linalg.norm(internal),np.linalg.norm(external),
                      np.linalg.norm(self.base.eigenstrain_load(eps)),self.base.scale*np.linalg.norm(u)*np.finfo(float).eps,np.finfo(float).tiny)
            residual=float(np.linalg.norm(reduced)/scale)
            physical=float(np.linalg.norm(self.base.constraint.equilibrium_projection.T@r)/scale)
            if residual<=tolerance:
                if physical>tolerance:raise MechanicalSolveError('Unbalanced force on rigid gauges')
                return SplitState(u.reshape(-1,3),strain,stress,plus,elastic,potential,r.reshape(-1,3),physical,iteration)
            weighted=(volumes[:,None,None]*C).ravel()
            block=coo_matrix((weighted,(self.rows,self.cols)),shape=(6*n,6*n)).tocsr()
            tangent=(self.P.T@self.D.T@block@self.D@self.P).tocsc()
            delta=spsolve(tangent,-reduced)
            if not np.isfinite(delta).all():raise MechanicalSolveError('Singular split tangent')
            direction=np.asarray(self.P@delta);slope=float(reduced@delta)
            if slope>=0:raise MechanicalSolveError('Split Newton direction is not descending')
            step=1.
            for _ in range(30):
                trial=u+step*direction
                _,_,_,trial_density=self.constitutive(np.asarray(self.D@trial).reshape(n,6)-eps,g)
                energy=float(volumes@trial_density-external@trial)
                if energy<=potential+1e-4*step*slope+1e-14*max(abs(potential),np.finfo(float).tiny):
                    u=trial;break
                step*=.5
            else:raise MechanicalSolveError('Split Newton line search failed')
        raise MechanicalSolveError('Split mechanical equilibrium did not converge')


@dataclass(frozen=True)
class CoupledStep:
    damage: np.ndarray
    mechanics: SplitState
    fatigue: FatigueHistory
    stagger_iterations: int
    damage_kkt: float
    fixed_history_energy_trace_J: np.ndarray
    fatigue_fracture_energy_change_J: float


def staggered_step(mechanics, accepted_damage, accepted_history, eigenstrain=0., *,
                   macro_strain=None, force=None, fatigue=True, threshold_J_m3=None,
                   tolerance=1e-8, max_iterations=100):
    """One accepted load endpoint, immutable rollback, lagged fatigue update.

    Hold accepted fatigue toughness fixed during alternating minimization.
    Commit positive alpha variation exactly once on convergence. This explicit
    fatigue-history discretization needs path/time refinement; it is not an
    unconditional coupled fatigue energy theorem or a cycle-jump algorithm.
    """
    model=mechanics.damage_model
    if len(accepted_history.accumulated_J_m3)!=len(model.mesh.cells):raise ValueError('History/mesh mismatch')
    threshold=model.Gc/(12*model.ell) if threshold_J_m3 is None else threshold_J_m3
    factor=accepted_history.toughness_factor(threshold) if fatigue else np.ones(len(model.mesh.cells))
    lower=model._damage(accepted_damage).copy();d=lower.copy();u=None;trace=[]
    for iteration in range(1,max_iterations+1):
        state=mechanics.solve(d,eigenstrain,macro_strain=macro_strain,force=force,initial_displacement=u)
        energy=state.potential_energy_J+model.energies(d,state.tensile_energy_J_m3,factor)[1]
        if trace and energy>trace[-1]+1e-9*max(abs(trace[-1]),np.finfo(float).tiny):
            raise RuntimeError('Alternating minimization increased fixed-history energy')
        trace.append(energy)
        solved=model.solve(state.tensile_energy_J_m3,lower,factor,tolerance=min(tolerance*.01,1e-10))
        change=float(np.max(np.abs(solved.damage-d)))
        if change<tolerance:
            # Return the mechanically equilibrated current d, checking its own
            # damage residual, rather than returning mismatched (u_old,d_new).
            A,b=model.assemble(state.tensile_energy_J_m3,factor)
            kkt=float(np.max(np.abs(d-np.clip(d-(A@d-b)/A.diagonal(),lower,1))))
            if kkt<tolerance:
                history=accepted_history.trial(model.mean_degradation(d)*state.tensile_energy_J_m3) if fatigue else accepted_history
                next_factor=history.toughness_factor(threshold) if fatigue else factor
                fracture_now=model.energies(d,state.tensile_energy_J_m3,factor)[1]
                fracture_next=model.energies(d,state.tensile_energy_J_m3,next_factor)[1]
                d.flags.writeable=False
                return CoupledStep(d,state,history,iteration,kkt,np.array(trace),fracture_next-fracture_now)
        d=solved.damage.copy();u=state.displacement_m
    raise RuntimeError('Staggered solve failed; accepted state is unchanged')
