"""Mass-preserving geometric, diffusion-spectral, POD and joint-metric spaces."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import distance_transform_edt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from .transport import ConcentrationInfeasible


def _orthogonal(candidates, weights, *, initial=None, count=None):
    columns = [] if initial is None else [x.copy() for x in initial.T]
    if count is not None and len(columns) >= count: return np.column_stack(columns[:count])
    for column in np.asarray(candidates).T:
        norm0 = np.sqrt(np.dot(weights, column*column))
        vector = column.copy()
        for _ in range(2):
            for previous in columns: vector -= previous*np.dot(weights*previous, vector)
        norm = np.sqrt(np.dot(weights, vector*vector))
        if norm > 1e-9*norm0 and norm > np.finfo(float).tiny:
            columns.append(vector/norm)
        if count is not None and len(columns) >= count: break
    if count is not None and len(columns) < count: raise ValueError("Insufficient independent concentration directions")
    return np.column_stack(columns)


def component_constants(diffusion):
    labels, weights = diffusion.components, diffusion.weights
    result = np.zeros((len(labels), diffusion.n_components))
    for k in range(diffusion.n_components):
        selected = labels == k; result[selected, k] = 1/np.sqrt(weights[selected].sum())
    return result


@dataclass
class ConcentrationBasis:
    values: np.ndarray
    weights: np.ndarray
    method: str
    component_count: int

    @property
    def mass(self): return self.values.T @ (self.weights[:, None]*self.values)

    def encode(self, concentration):
        c = np.asarray(concentration, dtype=float)
        if c.shape[0] != len(self.weights) or not np.isfinite(c).all(): raise ValueError("Invalid concentration snapshots")
        return np.linalg.solve(self.mass, self.values.T @ (self.weights[(slice(None),)+(None,)*(c.ndim-1)]*c))

    def decode(self, coefficients): return self.values @ np.asarray(coefficients, dtype=float)

    @classmethod
    def spectral(cls, diffusion, count):
        n = len(diffusion.weights)
        if not diffusion.n_components <= count <= n: raise ValueError("Must retain every diffusion-component mass mode")
        constants = component_constants(diffusion)
        if count == diffusion.n_components: values = constants
        else:
            inverse = 1/np.sqrt(diffusion.weights)
            symmetric = diags(inverse) @ diffusion.stiffness @ diags(inverse)
            if n <= 256 or count == n:
                if n > 1024: raise ValueError("Full dense spectrum exceeds the reference budget")
                _, vectors = eigh(symmetric.toarray())
            else:
                k = min(n-1, count+diffusion.n_components+4)
                _, vectors = eigsh(symmetric, k=k, which="SM", tol=1e-10, maxiter=20000,
                                  v0=np.random.default_rng(20261124).normal(size=n))
            candidates = inverse[:, None]*vectors
            values = _orthogonal(candidates, diffusion.weights, initial=constants, count=count)
        return cls(values, diffusion.weights.copy(), "diffusion-spectral", diffusion.n_components)

    @classmethod
    def geometric(cls, diffusion, count):
        if count < diffusion.n_components: raise ValueError("Must retain every component mass mode")
        constants = component_constants(diffusion)
        if count == diffusion.n_components: return cls(constants, diffusion.weights.copy(), "geometric", diffusion.n_components)
        spacing = np.asarray(diffusion.volume.spacing)
        distance = distance_transform_edt(np.pad(diffusion.mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1][diffusion.mask]
        distance /= max(distance.max(), np.finfo(float).tiny)
        xyz = (diffusion.centers-np.asarray(diffusion.volume.origin)[::-1])/(np.asarray(diffusion.mask.shape)*spacing)[::-1]
        x, y, z = xyz.T
        candidates = np.column_stack([distance, x, y, z, distance**2, x*x, y*y, z*z, x*y, x*z, y*z, distance**3])
        values = _orthogonal(candidates, diffusion.weights, initial=constants, count=count)
        return cls(values, diffusion.weights.copy(), "geometric-distance-and-direction", diffusion.n_components)

    @classmethod
    def pod(cls, diffusion, snapshots, count, *, concentration_scale=1., target_action=None):
        if count < diffusion.n_components or concentration_scale <= 0: raise ValueError("Invalid count or concentration scale")
        constants = component_constants(diffusion)
        x = np.asarray(snapshots, dtype=float)
        if x.ndim != 2 or x.shape[0] != len(diffusion.weights) or not np.isfinite(x).all(): raise ValueError("Snapshots must be [active voxels, samples]")
        x = x-constants @ (constants.T @ (diffusion.weights[:, None]*x))
        normalized = x/concentration_scale
        gram = normalized.T @ (diffusion.weights[:, None]*normalized)
        if target_action is not None:
            y = np.asarray(target_action(normalized), dtype=float)
            if y.ndim != 2 or y.shape[1] != x.shape[1] or not np.isfinite(y).all(): raise ValueError("Mechanical target features have incorrect shape")
            gram += y.T @ y
        _, vectors = eigh(gram)
        candidates = x @ vectors[:, ::-1]
        values = _orthogonal(candidates, diffusion.weights, initial=constants, count=count)
        return cls(values, diffusion.weights.copy(), "joint-metric-POD" if target_action is not None else "mass-POD", diffusion.n_components)


class ReducedDiffusion:
    def __init__(self, diffusion, basis):
        if basis.values.shape[0] != len(diffusion.weights) or not np.allclose(basis.weights, diffusion.weights):
            raise ValueError("Basis is incompatible with diffusion domain")
        self.full, self.basis = diffusion, basis
        self.mass = basis.mass
        if not np.isfinite(self.mass).all() or np.linalg.eigvalsh(self.mass).min() <= 0:
            raise ValueError("Concentration basis must be finite and linearly independent")
        constants = component_constants(diffusion)
        projected = basis.values @ np.linalg.solve(self.mass, basis.values.T @ (diffusion.weights[:, None]*constants))
        if np.linalg.norm(np.sqrt(diffusion.weights)[:, None]*(projected-constants)) > 1e-9:
            raise ValueError("Basis omits a diffusion-component mass mode; local inventories would not be conserved")
        self.stiffness = basis.values.T @ (diffusion.stiffness @ basis.values)
        self.flux_operator = np.asarray(basis.values.T @ diffusion.flux_operator)

    def step(self, coefficients, dt, *, flux=None, baseline=0., maximum=None):
        if not np.isfinite(dt) or dt <= 0: raise ValueError("Positive time step required")
        a = np.asarray(coefficients, dtype=float)
        source = np.zeros(len(a)) if flux is None else self.flux_operator @ np.broadcast_to(flux, self.full.face_areas.shape)
        updated = np.linalg.solve(self.mass+dt*self.stiffness, self.mass @ a+dt*source)
        c = baseline+self.basis.decode(updated)
        if np.any(c < -1e-10) or (maximum is not None and np.any(c > maximum)):
            raise ConcentrationInfeasible("Reduced diffusion is not automatically positive; enrich/reject rather than clip")
        return updated
