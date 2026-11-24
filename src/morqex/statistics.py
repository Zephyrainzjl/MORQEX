"""Exact tensor moments and discrete, volume-weighted tail enclosures.

Bounds apply to the stored discrete linear response and a stated q-domain.
They do not certify unresolved concentration, material physics, or continuum peaks.
"""
from dataclasses import dataclass
import numpy as np
from .mechanics.tensors import from_mandel


def principal_tension(stress):
    return np.linalg.eigvalsh(from_mandel(stress))[..., -1]


def von_mises(stress):
    a = np.asarray(stress)
    return np.sqrt(np.maximum(1.5*(np.sum(a*a, axis=-1)-np.sum(a[..., :3], axis=-1)**2/3), 0))


def weighted_quantile(values, weights, probability):
    x, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if x.ndim != 1 or x.shape != w.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Finite values and nonnegative positive-total volume weights required")
    if not 0 <= probability <= 1: raise ValueError("Probability must lie in [0,1]")
    keep = w > 0
    x, w = x[keep], w[keep]
    order = np.argsort(x, kind="stable"); x, w = x[order], w[order]
    return float(x[min(np.searchsorted(np.cumsum(w), probability*w.sum(), side="left"), len(x)-1)])


def weighted_cvar(values, weights, probability=.99):
    if not 0 <= probability < 1: raise ValueError("CVaR probability must lie in [0,1)")
    weighted_quantile(values, weights, probability)  # validate shared convention
    x, w = np.asarray(values), np.asarray(weights, dtype=float)
    order = np.argsort(x)[::-1]; x, w = x[order], w[order]/w.sum()
    used = np.minimum(w, np.maximum((1-probability)-np.r_[0., np.cumsum(w)[:-1]], 0))
    return float(used @ x/(1-probability))


@dataclass
class StressMoments:
    mean_operator: np.ndarray
    covariance_gram: np.ndarray

    @classmethod
    def build(cls, response, selection=None):
        ids = np.arange(len(response.volumes)) if selection is None else np.arange(len(response.volumes))[selection]
        weights = response.volumes[ids]
        if not len(ids) or weights.sum() <= 0: raise ValueError("Empty statistical region")
        weights = weights/weights.sum()
        field = np.concatenate([response.stress_operator[ids], response.stress_offset[ids, :, None]], axis=-1)
        mean = np.einsum("n,nid->id", weights, field)
        centered = field-mean
        gram = np.einsum("n,nid,nje->ijde", weights, centered, centered, optimize=True)
        return cls(mean, gram)

    def evaluate(self, q):
        z = np.r_[np.asarray(q, dtype=float), 1.]
        mean = self.mean_operator @ z
        cov = np.einsum("d,ijde,e->ij", z, self.covariance_gram, z)
        return mean, .5*(cov+cov.T)


@dataclass
class StressEnvelope:
    representative_operator: np.ndarray
    representative_offset: np.ndarray
    weights: np.ndarray
    delta: np.ndarray
    scales: np.ndarray
    radius: float
    element_ids: np.ndarray

    @classmethod
    def build(cls, response, count, scales, *, radius=1., selection=None):
        scales = np.asarray(scales, dtype=float)
        if scales.shape != (response.dimension,) or not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(radius) or radius <= 0:
            raise ValueError("Positive input scales and domain radius required")
        ids = np.arange(len(response.volumes)) if selection is None else np.arange(len(response.volumes))[selection]
        if not len(ids) or count < 1: raise ValueError("Nonempty region and positive sample count required")
        op, offset = response.stress_operator[ids], response.stress_offset[ids]
        scaled = op*scales[None, None, :]
        features = np.concatenate([radius*scaled.reshape(len(ids), -1), offset], axis=1)
        representatives, distance, assignment = [], np.full(len(ids), np.inf), np.zeros(len(ids), dtype=int)
        index = 0
        for j in range(min(count, len(ids))):
            representatives.append(index)
            dist = np.sum((features-features[index])**2, axis=1)
            improve = dist < distance
            distance[improve], assignment[improve] = dist[improve], j
            if np.max(distance) == 0: break
            index = int(np.argmax(distance))
        weights, delta = [], []
        for j, representative in enumerate(representatives):
            group = assignment == j
            # Frobenius operator norm upper bounds induced 2-norm; Weyl bounds
            # principal-eigenvalue error by tensor Frobenius stress error.
            e0 = np.linalg.norm(offset[group]-offset[representative], axis=1)
            ed = np.linalg.norm((scaled[group]-scaled[representative]).reshape(np.sum(group), -1), axis=1)
            delta.append(float(np.max(e0+radius*ed)))
            weights.append(float(response.volumes[ids[group]].sum()))
        representatives = np.array(representatives)
        weights = np.array(weights); weights /= weights.sum()
        return cls(op[representatives].copy(), offset[representatives].copy(), weights, np.array(delta),
                   scales.copy(), float(radius), ids[representatives])

    def evaluate(self, q, *, probability=.99, threshold=0.):
        q = np.asarray(q, dtype=float)
        if q.shape != self.scales.shape or not np.isfinite(q).all(): raise ValueError("Invalid q")
        if np.linalg.norm(q/self.scales) > self.radius*(1+1e-12):
            raise ValueError("Input outside tail-envelope domain; rebuild or recover the full field")
        stress = np.einsum("nid,d->ni", self.representative_operator, q)+self.representative_offset
        value = principal_tension(stress)
        low, high = value-self.delta, value+self.delta
        return {"estimate": weighted_quantile(value, self.weights, probability),
                "quantile_lower": weighted_quantile(low, self.weights, probability),
                "quantile_upper": weighted_quantile(high, self.weights, probability),
                "cvar_lower": weighted_cvar(low, self.weights, probability),
                "cvar_upper": weighted_cvar(high, self.weights, probability),
                "exceedance_lower": float(self.weights[low > threshold].sum()),
                "exceedance_upper": float(self.weights[high > threshold].sum()),
                "samples": len(value), "bound_scope": "stored discrete linear stress response only"}
