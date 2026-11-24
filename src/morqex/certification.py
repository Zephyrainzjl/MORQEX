"""Discrete fixed-Fick error bounds, explicitly separate from mechanics residuals.

The algebraic bounds are exact-arithmetic statements. FP64 evaluations include
roundoff padding and are tested against full solves, not interval arithmetic.
They do not certify space/time discretization, nonlinear kinetics, or DFN.
"""
from dataclasses import dataclass
import numpy as np
from .concentration import ReducedDiffusion, ControlledDiffusion, component_constants
from .energy import _digest
from .transport import ConcentrationInfeasible


def tree_poincare_lower_bound(diffusion):
    """Positive lower bound on nonzero generalized graph eigenvalues.

    For a spanning tree and total component weight W, Cauchy along every path
    gives ||v||_M^2 <= [sum_edges W_left W_right/g_edge]/W * v.T K v
    on zero-mean vectors. Non-tree edges only increase v.T K v.
    """
    k = diffusion.stiffness.tocsr()
    weights = diffusion.weights
    bounds = []
    for component in range(diffusion.n_components):
        ids = np.flatnonzero(diffusion.components == component)
        if len(ids) == 1: continue
        root = int(ids[0]); order = [root]; parent = {root: (-1, 0.)}
        for node in order:
            for j, value in zip(k.indices[k.indptr[node]:k.indptr[node+1]], k.data[k.indptr[node]:k.indptr[node+1]]):
                if value >= 0 or int(j) in parent: continue
                parent[int(j)] = (node, -float(value)); order.append(int(j))
        if len(order) != len(ids): raise ValueError("Disconnected graph inconsistent with diffusion components")
        total = float(weights[ids].sum())
        subtree = {i: float(weights[i]) for i in order}
        denominator = 0.
        for node in reversed(order[1:]):
            previous, conductance = parent[node]
            denominator += subtree[node]*(total-subtree[node])/conductance
            subtree[previous] += subtree[node]
        bounds.append(total/denominator)
    return float(min(bounds))*(1-1e-12) if bounds else np.inf


@dataclass(frozen=True)
class DiffusionErrorBound:
    zero_mean_bound: float
    component_mean_error: np.ndarray
    total_bound: float
    residual_dual_norm: float = 0.

    def __post_init__(self):
        mean = np.array(self.component_mean_error, dtype=float, copy=True)
        mean.setflags(write=False)
        object.__setattr__(self, "component_mean_error", mean)


class DiffusionCertificate:
    def __init__(self, diffusion, basis, flux_patterns):
        reduced = ReducedDiffusion(diffusion, basis) # verifies all particle mass modes
        patterns = np.asarray(flux_patterns, dtype=float)
        if patterns.ndim != 2 or patterns.shape[0] != len(diffusion.face_areas) or not np.isfinite(patterns).all():
            raise ValueError("Fixed finite face-flux pattern library required")
        self.basis, self.weights = basis, diffusion.weights.copy()
        self.modes, self.channels = basis.values.shape[1], patterns.shape[1]
        self.gap_lower = tree_poincare_lower_bound(diffusion)
        constants = component_constants(diffusion)
        orthogonal_constants = np.sqrt(self.weights)[:, None]*constants
        columns = np.column_stack([self.weights[:, None]*basis.values,
            diffusion.stiffness @ basis.values, -np.asarray(diffusion.flux_operator @ patterns)])
        scaled = columns/np.sqrt(self.weights)[:, None]
        self.mean_operator = orthogonal_constants.T @ scaled
        zero = scaled-orthogonal_constants @ self.mean_operator
        # Square-root form avoids cancellation of a residual Gram quadratic.
        self.residual_root = np.linalg.qr(zero, mode="r")
        self.roundoff_operator_norm = float(np.linalg.norm(scaled, ord="fro"))
        self.mass_constants = constants
        self.mass = reduced.mass
        self.stiffness = reduced.stiffness
        self.source = reduced.flux_operator @ patterns

    def initialize(self, initial_field, coefficients):
        c, a = np.asarray(initial_field, dtype=float), np.asarray(coefficients, dtype=float)
        error = c-self.basis.decode(a)
        if error.shape != self.weights.shape or not np.isfinite(error).all(): raise ValueError("Invalid initial error")
        mean = self.mass_constants.T @ (self.weights*error)
        zero = error-self.mass_constants @ mean
        norm = float(np.sqrt(self.weights @ zero**2))
        return DiffusionErrorBound(norm, mean, float(np.hypot(norm, np.linalg.norm(mean))))

    def step(self, previous, a_old, a_new, dt, channel_flux):
        a_old, a_new = np.asarray(a_old), np.asarray(a_new)
        flux = np.asarray(channel_flux, dtype=float)
        if a_old.shape != (self.modes,) or a_new.shape != (self.modes,) or flux.shape != (self.channels,) or dt <= 0 or not np.isfinite(dt):
            raise ValueError("Invalid certificate step")
        z = np.r_[a_new-a_old, dt*a_new, dt*flux]
        if not np.isfinite(z).all(): raise ValueError("Nonfinite residual coordinates")
        mean_residual = self.mean_operator @ z
        zero_residual = float(np.linalg.norm(self.residual_root @ z))
        # Numerical padding is not a directed-rounding proof.
        padding = 128*np.finfo(float).eps*self.roundoff_operator_norm*np.linalg.norm(z)
        zero = (previous.zero_mean_bound+zero_residual+padding)/(1+dt*self.gap_lower)
        mean = previous.component_mean_error-mean_residual
        total = float(np.hypot(zero, np.linalg.norm(mean)))+padding
        return DiffusionErrorBound(float(zero), mean, float(total), float(np.hypot(zero_residual, np.linalg.norm(mean_residual))))


@dataclass
class StressErrorTransfer:
    l2_constant: float
    pointwise_constant: float

    @classmethod
    def build(cls, full_response, concentration_weights):
        weights = np.asarray(concentration_weights, dtype=float)
        if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights <= 0) or full_response.dimension-6 != len(weights):
            raise ValueError("Positive full-concentration coordinate weights required")
        operator = full_response.stress_operator[:, :, 6:]/np.sqrt(weights)[None, None, :]
        if operator.shape[-1] != len(weights) or np.any(weights <= 0): raise ValueError("Full concentration coordinates required")
        gram = np.einsum("eim,ein,e->mn", operator, operator, full_response.volumes/full_response.rve_volume, optimize=True)
        # Largest eigenvalue <= max absolute row sum: a computable upper bound,
        # not an unverified coercivity constant from numerical eigensampling.
        l2 = np.sqrt(np.max(np.sum(np.abs(gram), axis=1)))*(1+1e-12)
        pointwise = float(np.max(np.linalg.norm(operator, axis=(1, 2))))*(1+1e-12)
        return cls(float(l2), pointwise)

    def stress_l2_bound(self, concentration_bound): return self.l2_constant*concentration_bound

    def quantile_bound(self, concentration_bound):
        """Discrete fixed-mesh principal-stress quantile difference; not a continuum peak bound."""
        return self.pointwise_constant*concentration_bound


class ConcentrationAccuracyError(RuntimeError):
    def __init__(self, actual_bound, tolerance):
        self.error_bound, self.tolerance = float(actual_bound), float(tolerance)
        super().__init__(f"Discrete concentration error bound {actual_bound:.6g} exceeds {tolerance:.6g}; step rejected, enrich or use a full solve")


@dataclass(frozen=True)
class CertifiedDiffusionState:
    coefficients: np.ndarray
    error: DiffusionErrorBound
    time: float
    model_fingerprint: str

    def __post_init__(self):
        a = np.array(self.coefficients, dtype=float, copy=True)
        a.setflags(write=False)
        object.__setattr__(self, "coefficients", a)


class CertifiedControlledDiffusion:
    """Transactional Fick stepping: no over-tolerance state is returned.

    Caller decides enrichment or full-solve recovery. This does not silently
    reset an accumulated bound, transfer a coupled history, or fit truth data.
    """
    def __init__(self, diffusion, basis, patterns, *, reference_concentration, maximum_concentration):
        self.basis = basis
        self.model = ControlledDiffusion(diffusion, basis, patterns,
            reference_concentration=reference_concentration, maximum_concentration=maximum_concentration)
        self.certificate = DiffusionCertificate(diffusion, basis, patterns)
        self.minimum_sqrt_weight = float(np.sqrt(np.min(diffusion.weights)))
        self.fingerprint = _digest(diffusion.voxel_ids, basis.values, basis.weights, self.model.mass,
            self.model.stiffness, self.model.source, np.array([self.model.reference, self.model.maximum]))

    @staticmethod
    def _check(bound, tolerance):
        if not np.isfinite(tolerance) or tolerance <= 0: raise ValueError("Positive concentration error tolerance required")
        if bound.total_bound > tolerance: raise ConcentrationAccuracyError(bound.total_bound, tolerance)

    def _physical_check(self, coefficients, bound):
        # A physical reconstructed state alone does not certify the full state.
        # For this fixed voxel mass norm, max_i |e_i| <= delta/sqrt(min_i w_i).
        excursion = self.model.suprema @ np.abs(coefficients)+bound.total_bound/self.minimum_sqrt_weight
        if excursion > self.model.margin:
            raise ConcentrationInfeasible("Concentration error enclosure reaches physical bounds; step rejected")

    def initialize(self, initial_deviation, *, error_tolerance):
        c = self.model.reference+np.asarray(initial_deviation)
        if np.any(c < 0) or np.any(c > self.model.maximum): raise ValueError("Initial concentration outside physical bounds")
        a = self.basis.encode(initial_deviation)
        bound = self.certificate.initialize(initial_deviation, a)
        self._check(bound, error_tolerance)
        self._physical_check(a, bound)
        return CertifiedDiffusionState(a, bound, 0., self.fingerprint)

    def step(self, state, dt, channel_flux, *, error_tolerance):
        if state.model_fingerprint != self.fingerprint: raise ValueError("Certified state belongs to a different domain, basis or model")
        a = self.model.step(state.coefficients, dt, channel_flux)
        bound = self.certificate.step(state.error, state.coefficients, a, dt, channel_flux)
        self._check(bound, error_tolerance)
        self._physical_check(a, bound)
        return CertifiedDiffusionState(a, bound, state.time+dt, self.fingerprint)
