"""Variational AT2 damage subproblem and Carrara-type fatigue history.

Reference: Carrara et al., CMAME 361 (2020) 112731,
doi:10.1016/j.cma.2019.112731. Ai et al., JPS 544 (2022) 231805
uses a related fatigue law in battery particles; this module alone is neither
a reproduction of their hybrid mechanical model nor a battery lifetime model.

SI units: psi and alpha [J/m^3], Gc [J/m^2], ell [m], d dimensionless.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import spsolve


def _field(value, size, name, positive=False):
    a = np.array(np.broadcast_to(value, (size,)), dtype=float, copy=True)
    if not np.isfinite(a).all() or np.any(a <= 0 if positive else a < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return a


@dataclass(frozen=True)
class FatigueHistory:
    """Accepted quadrature/cell history; trials never mutate accepted arrays."""
    accumulated_J_m3: np.ndarray
    previous_alpha_J_m3: np.ndarray

    def __post_init__(self):
        a = np.array(self.accumulated_J_m3, dtype=float, copy=True)
        p = np.array(self.previous_alpha_J_m3, dtype=float, copy=True)
        if a.ndim != 1 or p.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(p).all() or min(a.min(initial=0), p.min(initial=0)) < 0:
            raise ValueError('Nonnegative finite same-size fatigue fields required')
        a.flags.writeable = False; p.flags.writeable = False
        object.__setattr__(self, 'accumulated_J_m3', a)
        object.__setattr__(self, 'previous_alpha_J_m3', p)

    @classmethod
    def zero(cls, cells):
        return cls(np.zeros(cells), np.zeros(cells))

    def trial(self, alpha_J_m3):
        """Positive variation between resolved path endpoints.

        Unresolved turning points are not captured. Calling this repeatedly on
        one accepted state is idempotent. Only replace the accepted state after
        convergence; this is not a cycle-jump extrapolation.
        """
        alpha = _field(alpha_J_m3, len(self.previous_alpha_J_m3), 'alpha')
        return FatigueHistory(self.accumulated_J_m3 + np.maximum(alpha-self.previous_alpha_J_m3, 0), alpha)

    def toughness_factor(self, threshold_J_m3):
        threshold = _field(threshold_J_m3, len(self.previous_alpha_J_m3), 'fatigue threshold', True)
        ratio = np.maximum(self.accumulated_J_m3 / threshold, 1.)
        return (2. / (1. + ratio))**2


@dataclass(frozen=True)
class DamageSolution:
    damage: np.ndarray
    projected_kkt: float
    iterations: int
    elastic_driving_energy_J: float
    fracture_energy_J: float
    objective_change_J: float


def solve_box_quadratic(A, b, lower, *, tolerance=1e-10, max_iterations=100):
    """Primal-dual active set for a strictly convex FE damage quadratic.

    Stopping uses the diagonally scaled projected KKT residual; a failure is
    explicit. There is no unconstrained solve followed by an accepted clipping.
    """
    lower = np.asarray(lower, float)
    if lower.shape != b.shape or not np.isfinite(lower).all() or np.any((lower < 0) | (lower > 1)):
        raise ValueError('Damage lower bound must lie in [0, 1]')
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('Positive subunit KKT tolerance required')
    diagonal = A.diagonal()
    if not np.isfinite(diagonal).all() or np.min(diagonal) <= 0:
        raise ValueError('Strictly positive damage Hessian diagonal required')
    x = lower.copy()
    for iteration in range(1, max_iterations+1):
        grad = A@x-b
        predictor = x-grad/diagonal
        low = predictor <= lower
        high = (predictor >= 1) & ~low
        free = ~(low | high)
        x[low] = lower[low]; x[high] = 1.
        if np.any(free):
            x[free] = spsolve(A[free][:, free].tocsc(), b[free]-A[free][:, ~free]@x[~free])
        residual = x-np.clip(x-(A@x-b)/diagonal, lower, 1.)
        norm = float(np.max(np.abs(residual)))
        if np.isfinite(norm) and norm <= tolerance:
            # Exact bounds remove roundoff only, after the KKT gate.
            x = np.clip(x, lower, 1.)
            checked = float(np.max(np.abs(x-np.clip(x-(A@x-b)/diagonal, lower, 1.))))
            if checked <= tolerance:
                return x, checked, iteration
    raise RuntimeError(f'Damage active set failed KKT gate after {max_iterations} iterations')


class AT2Damage:
    """Fixed-driving-energy, irreversible 3D AT2 solve using MFEM assembly.

    Cellwise psi+ and Gc_eff; continuous P1 damage. Variable Gc_eff is inside
    the gradient integral, retaining its weak-form spatial variation. Natural
    no-flux damage BC; prescribed precracks are lower_bound=1 nodes.
    """
    def __init__(self, mesh, backend, fracture_toughness_J_m2, length_m, residual_stiffness=1e-5):
        self.mesh, self.backend = mesh, backend
        self.Gc = _field(fracture_toughness_J_m2, len(mesh.cells), 'Gc', True)
        self.ell = _field(length_m, len(mesh.cells), 'ell', True)
        if not np.isfinite(residual_stiffness) or not 0 < residual_stiffness < 1:
            raise ValueError('Residual stiffness must lie in (0,1)')
        self.k = float(residual_stiffness)

    def mean_degradation(self, damage):
        d = self._damage(damage)[self.mesh.cells]
        mean_square = (np.sum(d, axis=1)**2+np.sum(d*d, axis=1))/20
        return self.k+(1-self.k)*(1-2*np.mean(d, axis=1)+mean_square)

    def _damage(self, damage):
        d = np.asarray(damage, float)
        if d.shape != (len(self.mesh.points),) or not np.isfinite(d).all() or np.any((d < 0) | (d > 1)):
            raise ValueError('One nodal damage in [0,1] required')
        return d

    def _effective_toughness(self, factor):
        factor = _field(factor, len(self.mesh.cells), 'toughness factor', True)
        if np.any(factor > 1):
            raise ValueError('Fatigue factor cannot exceed 1')
        return self.Gc*factor

    def assemble(self, tensile_energy_J_m3, toughness_factor=1.):
        psi = _field(tensile_energy_J_m3, len(self.mesh.cells), 'tensile energy')
        gc = self._effective_toughness(toughness_factor)
        source = 2*(1-self.k)*psi
        A, b = self.backend.assemble(self.mesh, source+gc/self.ell,
                                     (gc*self.ell)[:, None, None]*np.eye(3), source)
        return A, b

    def energies(self, damage, tensile_energy_J_m3, toughness_factor=1.):
        d = self._damage(damage)[self.mesh.cells]
        psi = _field(tensile_energy_J_m3, len(self.mesh.cells), 'tensile energy')
        gc = self._effective_toughness(toughness_factor)
        d2 = (np.sum(d, axis=1)**2+np.sum(d*d, axis=1))/20
        grad = np.einsum('ei,eij->ej', d, self.mesh.gradients)
        fracture = float(self.mesh.volumes @ (gc/2*(d2/self.ell+self.ell*np.sum(grad*grad, axis=1))))
        elastic = float(self.mesh.volumes @ (self.mean_degradation(damage)*psi))
        return elastic, fracture

    def solve(self, tensile_energy_J_m3, lower_bound, toughness_factor=1., *, tolerance=1e-10):
        lower_bound = self._damage(lower_bound)
        A, b = self.assemble(tensile_energy_J_m3, toughness_factor)
        d, residual, iterations = solve_box_quadratic(A, b, lower_bound, tolerance=tolerance)
        elastic, fracture = self.energies(d, tensile_energy_J_m3, toughness_factor)
        old_energy = sum(self.energies(lower_bound, tensile_energy_J_m3, toughness_factor))
        change = elastic+fracture-old_energy
        if change > 1e-9*max(old_energy, np.finfo(float).tiny):
            raise RuntimeError('Damage minimization increased fixed-history energy')
        d.flags.writeable = False
        return DamageSolution(d, residual, iterations, elastic, fracture, change)
