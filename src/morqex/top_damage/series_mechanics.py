"""Energy-stationary equal-area plane-strain stack, fixed total thickness.

The common axial FIRST PIOLA stress follows from constrained minimization.
Local deformation gradients differ. Lateral macroscopic stretches stay one;
this is a one-dimensional laminate, not general three-dimensional cell balance.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import LinearOperator,minres,spsolve
from morqex.mechanics.reference import MechanicalSolveError


@dataclass(frozen=True)
class AxialResponse:
    energy_density_J_m3: float
    nominal_stress_Pa: float
    tangent_Pa: float
    state: object


class MicrostructureLayer:
    def __init__(self,mechanics,chemical_F,degradation,*,axis=0,initial_displacement=None):
        if mechanics.constraint.boundary!='kubc' or axis not in (0,1,2):
            raise ValueError('KUBC with a declared axial direction required')
        self.mechanics=mechanics;self.chemical_F=chemical_F;self.degradation=degradation
        self.axis=axis;self.warm=initial_displacement
        self.modulus_scale=float(np.max(mechanics.bulk+4*mechanics.shear/3))

    def evaluate(self,stretch):
        m=self.mechanics;axis=self.axis;F=np.eye(3);F[axis,axis]=stretch
        state=m.solve(self.chemical_F,self.degradation,macro_F=F,initial_displacement=self.warm,tolerance=1e-10)
        self.warm=state.displacement_m
        response=m.assemble(state.displacement_m,self.chemical_F,self.degradation)
        A=response[2];del response
        lift=np.zeros_like(m.mesh.points);lift[:,axis]=m.mesh.points[:,axis]-m.mesh.box[:,axis].mean();lift=lift.ravel()
        rhs=-np.asarray(m.P.T@(A@lift))
        if len(rhs):
            if m.matrix_free:
                reduced=LinearOperator((len(rhs),)*2,matvec=lambda v:m.P.T@(A@(m.P@v)),dtype=float)
                w,info=minres(reduced,rhs,M=m.preconditioner,rtol=1e-12,maxiter=4000)
                if info!=0:raise MechanicalSolveError('Axial condensed tangent solve failed')
            else:
                reduced=(m.P.T@A@m.P).tocsc();w=spsolve(reduced,rhs)
            defect=np.linalg.norm(reduced@w-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny)
            if defect>1e-7:raise MechanicalSolveError('Axial condensed tangent residual failed')
            direction=lift+m.P@w
        else:direction=lift
        tangent=float(lift@(A@direction)/m.mesh.rve_volume)
        stress=float(m.mesh.volumes@state.first_piola_Pa[:,axis,axis]/m.mesh.rve_volume)
        # Independent boundary virtual-work check includes void-volume normalization.
        boundary=float(state.reaction_N.ravel()@lift/m.mesh.rve_volume)
        if abs(stress-boundary)>1e-7*max(abs(stress),self.modulus_scale*1e-8):
            raise MechanicalSolveError('Axial Hill-Mandel boundary work mismatch')
        return AxialResponse(state.elastic_energy_J/m.mesh.rve_volume,stress,tangent,state)


class HomogeneousLayer:
    """Same compiled Neo-Hookean energy for separator/metal numerical layers.

    Explicit constants are required; using this elastic law does not calibrate
    plastic lithium or viscoelastic/porous separators.
    """
    def __init__(self,backend,young_Pa,poisson,*,axis=0,chemical_F=None):
        if not np.isfinite([young_Pa,poisson]).all() or young_Pa<=0 or not -1<poisson<.5 or axis not in (0,1,2):raise ValueError('Valid passive layer constants required')
        self.backend=backend;self.mu=young_Pa/(2*(1+poisson));self.K=young_Pa/(3*(1-2*poisson));self.axis=axis
        self.chemical_F=np.eye(3) if chemical_F is None else np.array(chemical_F,float)
        self.modulus_scale=self.K+4*self.mu/3

    def evaluate(self,stretch):
        F=np.eye(3)[None];F[0,self.axis,self.axis]=stretch
        w,_,p,C=self.backend.evaluate(F,self.chemical_F,self.mu,self.K)
        a=3*self.axis+self.axis
        return AxialResponse(float(w[0]),float(p[0,self.axis,self.axis]),float(C[0,a,a]),None)


@dataclass(frozen=True)
class SeriesState:
    stretches: np.ndarray
    responses: tuple
    common_nominal_stress_Pa: float
    force_scale_Pa: float
    normalized_force_defect: float
    mean_stretch_defect: float
    iterations: int


def solve_fixed_thickness(layers,thicknesses_m,*,mean_stretch=1.,initial_stretches=None,tolerance=1e-8,max_iterations=25):
    lengths=np.asarray(thicknesses_m,float);n=len(layers)
    if n<2 or lengths.shape!=(n,) or not np.isfinite(lengths).all() or np.any(lengths<=0):raise ValueError('Positive thickness per layer required')
    if not np.isfinite(mean_stretch) or mean_stretch<=0 or not 0<tolerance<1e-3:raise ValueError('Positive mean stretch and strict tolerance required')
    weights=lengths/lengths.sum();x=np.full(n,mean_stretch) if initial_stretches is None else np.array(initial_stretches,float)
    if x.shape!=(n,) or not np.isfinite(x).all():raise ValueError('One finite stretch per layer required')
    x+=mean_stretch-weights@x
    if np.any(x<=0):raise ValueError('Initial stack has inverted layer')
    values=tuple(layer.evaluate(lam) for layer,lam in zip(layers,x))
    pressure_scale=max(max(abs(r.nominal_stress_Pa) for r in values),max(layer.modulus_scale for layer in layers)*1e-8)
    noise=128*np.finfo(float).eps*sum(w*layer.modulus_scale for w,layer in zip(weights,layers))
    for iteration in range(max_iterations+1):
        stress=np.array([r.nominal_stress_Pa for r in values]);C=np.array([r.tangent_Pa for r in values])
        if not np.isfinite(C).all() or np.any(C<=0):raise MechanicalSolveError('Unstable fixed-damage axial tangent; refine load path')
        defect=float(np.max(abs(stress-weights@stress))/pressure_scale)
        thickness=float(abs(weights@x-mean_stretch))
        if defect<tolerance and thickness<1e-12:
            x.flags.writeable=False
            return SeriesState(x,values,float(weights@stress),pressure_scale,defect,thickness,iteration)
        if iteration==max_iterations:break
        common=(mean_stretch-weights@x+weights@(stress/C))/(weights@(1/C))
        direction=(common-stress)/C
        slope=float(weights@(stress*direction));energy=sum(w*r.energy_density_J_m3 for w,r in zip(weights,values))
        step=1.
        for _ in range(25):
            trial=x+step*direction
            if np.any(trial<=0):step*=.5;continue
            candidate=tuple(layer.evaluate(lam) for layer,lam in zip(layers,trial))
            potential=sum(w*r.energy_density_J_m3 for w,r in zip(weights,candidate))
            stresses=np.array([r.nominal_stress_Pa for r in candidate])
            residual=float(np.max(abs(stresses-weights@stresses))/pressure_scale)
            if potential<=energy+1e-4*step*slope or (potential<=energy+noise and residual<defect):
                x=trial;values=candidate;break
            step*=.5
        else:raise MechanicalSolveError('Stack constrained energy line search failed')
    raise MechanicalSolveError('Stack force balance failed; no state accepted')


def finite_series_step(mechanics,damage_models,accepted_damage,accepted_history,chemical_F,
                       thicknesses_m,*,passive_layers=(),passive_thicknesses_m=(),
                       initial_stretches=None,initial_displacements=None,tolerance=1e-7,max_iterations=100):
    """Joint macro/micro/AT2 minimization with frozen accepted fatigue history.

    Fatigue is committed once at the converged load endpoint, not on macro or
    staggered trial iterations. All SVEs retain their actual individual states.
    """
    from .hyperelastic import FiniteCoupledStep
    n=len(mechanics)
    if not all(len(a)==n for a in (damage_models,accepted_damage,accepted_history,chemical_F,thicknesses_m)):
        raise ValueError('Matching microstructure-layer states required')
    if len(passive_layers)!=len(passive_thicknesses_m):raise ValueError('Thickness required for every passive layer')
    lower=[model._damage(d).copy() for model,d in zip(damage_models,accepted_damage)]
    damage=[d.copy() for d in lower]
    factors=[h.toughness_factor(model.Gc/(12*model.ell)) for h,model in zip(accepted_history,damage_models)]
    warm=[None]*n if initial_displacements is None else list(initial_displacements)
    lengths=np.r_[thicknesses_m,passive_thicknesses_m];weights=lengths/lengths.sum()
    stretches=initial_stretches;trace=[]
    for iteration in range(1,max_iterations+1):
        layers=[MicrostructureLayer(m,fc,model.mean_degradation(d),initial_displacement=u)
                for m,model,d,fc,u in zip(mechanics,damage_models,damage,chemical_F,warm)]
        series=solve_fixed_thickness(layers+list(passive_layers),lengths,initial_stretches=stretches)
        states=[r.state for r in series.responses[:n]]
        energy=sum(w*r.energy_density_J_m3 for w,r in zip(weights,series.responses))
        energy+=sum(weights[i]*model.energies(d,states[i].tensile_energy_J_m3,factor)[1]/model.mesh.rve_volume
                    for i,(model,d,factor) in enumerate(zip(damage_models,damage,factors)))
        if trace and energy>trace[-1]+1e-8*max(abs(trace[-1]),np.finfo(float).tiny):
            raise MechanicalSolveError('Joint stack/AT2 frozen-history energy increased')
        trace.append(energy);trials=[];kkts=[]
        for model,d,lo,factor,state in zip(damage_models,damage,lower,factors,states):
            psi=np.maximum(state.tensile_energy_J_m3,0)
            solved=model.solve(psi,lo,factor,tolerance=min(tolerance*.01,1e-10));trials.append(solved.damage)
            A,b=model.assemble(psi,factor)
            kkts.append(float(np.max(abs(d-np.clip(d-(A@d-b)/A.diagonal(),lo,1)))))
        change=max(np.max(abs(a-b)) for a,b in zip(damage,trials))
        if change<tolerance and max(kkts)<tolerance:
            results=[]
            for i,(model,d,h,factor,state) in enumerate(zip(damage_models,damage,accepted_history,factors,states)):
                psi=np.maximum(state.tensile_energy_J_m3,0)
                history=h.trial(model.mean_degradation(d)*psi);changed=history.toughness_factor(model.Gc/(12*model.ell))
                delta=model.energies(d,psi,changed)[1]-model.energies(d,psi,factor)[1]
                d.flags.writeable=False
                # Each local energy need not decrease while load redistributes.
                # The joint trace is returned separately with its actual units.
                results.append(FiniteCoupledStep(d,state,history,iteration,kkts[i],np.empty(0),delta))
            return tuple(results),series,np.asarray(trace)
        damage=[a.copy() for a in trials];warm=[s.displacement_m for s in states];stretches=series.stretches
    raise MechanicalSolveError('Joint stack/AT2 did not converge; accepted history unchanged')
