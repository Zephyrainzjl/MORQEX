"""Transactional event updates for conservative nonlinear reaction graphs.

This controls quasi-static voltage AND face-current errors at supplied chemical
states. It is not a cycle-jump algorithm or a bound on accumulated Li inventory.
Only passive interior elimination is reused; every accepted query re-solves BV.
"""
from dataclasses import dataclass
import numpy as np
from .contracts import ConductingNetwork
from .reaction_network import ReactionFaces,ReactiveDualNetwork,WorkingNetworkState,ReactionPortErrorBound


@dataclass(frozen=True)
class ReactionUpdate:
    state: WorkingNetworkState
    certificate: ReactionPortErrorBound
    rebuilt: bool
    reason: str
    relative_face_current_bound: float
    rebuild_count: int
    reuse_count: int


class ReactionNetworkCache:
    """Compare to the last factorization, with mandatory topology rebuilding.

    Current error is the sum of certified absolute face-current errors divided
    by max(|I|, sum |j|). This remains meaningful for an internally redistributing
    floating particle and avoids dividing by an individual near-zero reaction.
    The target and cached passive operators have the same retained BV vertices.
    On failure no accepted operator, state, or counters are modified.
    """
    def __init__(self,*,voltage_tolerance_V=1e-3,relative_current_tolerance=1e-3,
                 solver_tolerance=1e-10,extended_precision=False,max_ports=512):
        controls=[voltage_tolerance_V,relative_current_tolerance,solver_tolerance]
        if (not np.isfinite(controls).all() or min(controls)<=0
                or relative_current_tolerance>=1 or solver_tolerance>=1e-3):
            raise ValueError('Positive voltage/current and nonlinear solver tolerances required')
        if type(extended_precision) is not bool or type(max_ports) is not int or max_ports<2:
            raise ValueError('Explicit precision and positive port budget required')
        self.voltage_tolerance_V=float(voltage_tolerance_V)
        self.relative_current_tolerance=float(relative_current_tolerance)
        self.solver_tolerance=float(solver_tolerance)
        self.extended_precision=extended_precision;self.max_ports=max_ports
        self.operator=None;self.accepted=None;self.rebuilds=0;self.reuses=0

    def _rebuild_reason(self,electronic,ionic,faces):
        old=self.operator
        if old is None:return 'initial factorization'
        for a,b in ((old.electronic,electronic),(old.ionic,ionic)):
            if (a.vertex_count!=b.vertex_count or not np.array_equal(a.edges,b.edges)
                    or not np.array_equal(a.anchors,b.anchors)):
                return 'graph incidence or reservoir changed'
            mask=a.conductance_S>0
            if not np.array_equal(mask,b.conductance_S>0):return 'positive edge support changed'
            if np.max(abs(b.conductance_S[mask]/a.conductance_S[mask]-1),initial=0)>=1:
                return 'relative conductance bound outside coercive range'
        if (not np.array_equal(old.faces.electronic_vertices,faces.electronic_vertices)
                or not np.array_equal(old.faces.ionic_vertices,faces.ionic_vertices)):
            return 'reaction-face incidence changed'
        if not np.array_equal(old.amplitude>0,faces.exchange_A_m2>0):
            return 'positive reaction support changed'
        return ''

    def _solve(self,op,current):
        previous=None if self.accepted is None else self.accepted.state
        initial=None
        if (previous is not None and len(previous.electronic_potential_V)==op.electronic.vertex_count
                and len(previous.ionic_potential_V)==op.ionic.vertex_count):
            initial=op.initial_from(previous)
        return op.solve(current,initial=initial,tolerance=self.solver_tolerance,
                        extended_precision=self.extended_precision)

    def _certificate(self,op,state,electronic,ionic):
        bound=op.changed_problem_bound(state,electronic,ionic,include_reaction_bounds=True)
        scale=max(abs(state.discharge_current_A),float(np.sum(abs(state.reaction_current_A))),1e-30)
        relative=float(np.sum(bound.reaction_current_absolute_bounds_A)/scale)
        passed=(bound.voltage_absolute_bound_V<=self.voltage_tolerance_V
                and relative<=self.relative_current_tolerance)
        return bound,relative,passed

    def evaluate(self,electronic,ionic,faces,*,temperature_K,discharge_current_A):
        if (not isinstance(electronic,ConductingNetwork) or electronic.channel!='electronic'
                or not isinstance(ionic,ConductingNetwork) or ionic.channel!='ionic'
                or len(electronic.anchors)!=1 or len(ionic.anchors)!=1
                or not isinstance(faces,ReactionFaces)):
            raise ValueError('Physical dual networks, one reservoir each, and reaction faces required')
        if not np.isfinite([temperature_K,discharge_current_A]).all() or temperature_K<=0:
            raise ValueError('Finite current and positive temperature required')
        reason=self._rebuild_reason(electronic,ionic,faces)
        if not reason:
            candidate=self.operator.with_reaction_state(faces,temperature_K=temperature_K)
            try:
                state=self._solve(candidate,discharge_current_A)
                certificate,relative,passed=self._certificate(candidate,state,electronic,ionic)
            except (RuntimeError,ArithmeticError,np.linalg.LinAlgError):
                # Recover only by solving the target problem; never relax gates.
                reason='cached nonlinear solve or certificate failed'
            else:
                if not passed:reason='voltage or face-current bound exceeded'
        rebuilt=bool(reason)
        if rebuilt:
            candidate=ReactiveDualNetwork(electronic,ionic,faces,
                electronic_collector=int(electronic.anchors[0]),ionic_reservoir=int(ionic.anchors[0]),
                temperature_K=temperature_K,reduce=True,max_ports=self.max_ports)
            state=self._solve(candidate,discharge_current_A)
            certificate,relative,passed=self._certificate(candidate,state,electronic,ionic)
            if not passed:raise RuntimeError('Fresh graph solve fails voltage/face-current acceptance bounds')
        update=ReactionUpdate(state,certificate,rebuilt,reason or 'certified passive reuse; new BV currents solved',
                              relative,self.rebuilds+int(rebuilt),self.reuses+int(not rebuilt))
        self.operator=candidate;self.accepted=update
        self.rebuilds=update.rebuild_count;self.reuses=update.reuse_count
        return update
