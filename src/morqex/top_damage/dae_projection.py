"""Algebraic reconstruction of dense-output DAE samples.

Differential states, step endpoints and cutoff events are retained bit for bit.
This restores the algebraic constraint at requested interior output times; it
is not a correction of integration error, a mass projection, or a lower-accuracy
replacement for a time-convergence study. Uses the vendored PyBaMM/CasADi Newton
rootfinder with an independent residual check.
"""
from dataclasses import dataclass
import time
import numpy as np
import pybamm as pb


@dataclass(frozen=True)
class ProjectionAudit:
    sample_count: int
    maximum_algebraic_residual_before: float
    maximum_algebraic_residual_after: float
    maximum_state_change: float
    seconds: float


class AlgebraicOutputProjection:
    def __init__(self, *, tolerance=1e-12):
        if not np.isfinite(tolerance) or not 0<tolerance<=1e-8:
            raise ValueError('Positive algebraic projection tolerance <= 1e-8 required')
        self.tolerance=float(tolerance);self._model=None;self._root=None;self._names=None

    def __call__(self, solution):
        start=time.perf_counter()
        if len(solution.all_models)!=1 or len(solution.all_inputs)!=1 or len(solution.all_ts)!=1:
            raise ValueError('One full-state, constant-input DAE segment required')
        if solution.sensitivities or solution.variables_returned:
            raise ValueError('Projection of sensitivities or output-only states is not implemented')
        model=solution.all_models[0];inputs=solution.all_inputs[0]
        y=np.array(solution.y,dtype=float,copy=True);nr=model.len_rhs
        if y.ndim!=2 or y.shape[0]!=model.len_rhs_and_alg or not 0<nr<y.shape[0] or not np.isfinite(y).all():
            raise ValueError('Finite differential and algebraic full-state vector required')
        names=tuple(inputs)
        if self._model is not model or self._names!=names:
            old=getattr(model,'algebraic_root_solver',None)
            try:
                root=pb.CasadiAlgebraicSolver(tol=self.tolerance,step_tol=1e-14)
                root.set_up_root_solver(model,inputs,solution.t)
                self._root=model.algebraic_root_solver
            finally:
                model.algebraic_root_solver=old
            self._model=model;self._names=names
        values=np.concatenate([np.asarray(v).reshape(-1) for v in inputs.values()]) if inputs else np.empty(0)
        old=y.copy();before=[];after=[]
        for k,t in enumerate(solution.t):
            r=np.asarray(model.casadi_algebraic(t,y[:,k],values)).ravel()
            before.append(float(np.max(abs(r))))
            if 0<k<len(solution.t)-1:
                y[nr:,k]=np.asarray(self._root(y[nr:,k],np.r_[t,y[:nr,k],values])).ravel()
                checked=np.asarray(model.casadi_algebraic(t,y[:,k],values)).ravel()
                if not self._root.stats()['success'] or not np.isfinite(y[:,k]).all() or not np.isfinite(checked).all() or np.max(abs(checked))>10*self.tolerance:
                    raise RuntimeError('Algebraic output projection failed; original state unmodified')
                after.append(float(np.max(abs(checked))))
            else:after.append(before[-1])
        if not np.array_equal(y[:nr],old[:nr]) or not np.array_equal(y[:,[0,-1]],old[:,[0,-1]]):
            raise RuntimeError('Algebraic projection modified a differential state or step endpoint')
        # Do not copy interpolation derivatives: they describe the unprojected
        # dense interpolant. Returned values are the corrected output samples.
        result=pb.Solution(solution.t,y,model,inputs,t_event=solution.t_event,
            y_event=solution.y_event,termination=solution.termination,check_solution=False)
        for name in ('closest_event_idx','set_up_time','solve_time','integration_time'):
            setattr(result,name,getattr(solution,name,None))
        return result,ProjectionAudit(max(0,len(solution.t)-2),max(before),max(after),
            float(np.max(abs(y-old))),time.perf_counter()-start)
