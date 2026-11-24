"""Query-adaptive interval subdivision for a discrete stress quantile.

Uses established interval arithmetic, Weyl's inequality and branch subdivision.
No novelty claim. Bounds concern the stored linear response at its fixed mesh;
unresolved concentration and continuum singularities require separate errors.
"""
from dataclasses import dataclass
import numpy as np
from .statistics import principal_tension, weighted_quantile, weighted_cvar


@dataclass
class QuantileResult:
    lower: float
    upper: float
    estimate: float
    cvar_lower: float
    cvar_upper: float
    exceedance_lower: float
    exceedance_upper: float
    converged: bool
    evaluated_nodes: int
    evaluated_leaves: int
    frontier_nodes: int
    total_elements: int
    reason: str


class QuantileTree:
    def __init__(self, response, *, scales=None, selection=None):
        ids = np.arange(len(response.volumes)) if selection is None else np.atleast_1d(np.arange(len(response.volumes))[selection])
        if not len(ids): raise ValueError("Nonempty statistical region required")
        scale = np.ones(response.dimension) if scales is None else np.asarray(scales, dtype=float)
        if scale.shape != (response.dimension,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Positive response scales required for offline tree partition")
        operator = np.concatenate([response.stress_operator[ids], response.stress_offset[ids, :, None]], axis=-1)
        feature_scale = np.r_[scale, 1.]
        features = (operator*feature_scale[None, None, :]).reshape(len(ids), -1)
        self.nodes = []
        def build(local_ids):
            index = len(self.nodes)
            values = operator[local_ids]
            node = {"minimum": values.min(axis=0), "maximum": values.max(axis=0),
                "weight": float(response.volumes[ids[local_ids]].sum()), "left": -1, "right": -1}
            self.nodes.append(node)
            if len(local_ids) > 1:
                subset = features[local_ids]
                axis = int(np.argmax(np.ptp(subset, axis=0)))
                order = local_ids[np.argsort(subset[:, axis], kind="stable")]
                middle = len(order)//2
                node["left"], node["right"] = build(order[:middle]), build(order[middle:])
            return index
        build(np.arange(len(ids)))
        self.minimum = np.array([x["minimum"] for x in self.nodes])
        self.maximum = np.array([x["maximum"] for x in self.nodes])
        self.weights = np.array([x["weight"] for x in self.nodes])/response.volumes[ids].sum()
        self.children = np.array([[x["left"], x["right"]] for x in self.nodes], dtype=int)
        self.dimension, self.total_elements = response.dimension, len(ids)
        del self.nodes

    def query(self, q, *, probability=.99, relative_tolerance=.05, absolute_tolerance=1.,
              scale_floor=1., threshold=0., maximum_nodes=None):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dimension,) or not np.isfinite(q).all(): raise ValueError("Finite response input required")
        if not 0 <= probability < 1 or min(relative_tolerance, absolute_tolerance) < 0 or scale_floor <= 0:
            raise ValueError("Invalid probability or tolerance")
        budget = len(self.weights) if maximum_nodes is None else int(maximum_nodes)
        if budget < 1: raise ValueError("Positive query node budget required")
        z = np.r_[q, 1.]
        evaluations, leaves = 0, 0
        def evaluate(indices):
            nonlocal evaluations, leaves
            indices = np.asarray(indices, dtype=int)
            low = np.sum(np.minimum(self.minimum[indices]*z, self.maximum[indices]*z), axis=-1)
            high = np.sum(np.maximum(self.minimum[indices]*z, self.maximum[indices]*z), axis=-1)
            center, radius = .5*(low+high), .5*(high-low)
            value = principal_tension(center)
            delta = np.linalg.norm(radius, axis=1)
            # Supplement Weyl lower enclosure by the Rayleigh quotient on axes.
            lower = np.maximum(value-delta, low[:, :3].max(axis=1))
            upper = value+delta
            padding = 64*np.finfo(float).eps*np.maximum(np.max(np.abs(center), axis=1)+delta, 1.)
            evaluations += len(indices)
            leaves += int(np.sum(self.children[indices, 0] < 0))
            return lower-padding, upper+padding
        frontier = [0]
        lo, hi = evaluate(frontier)
        lower, upper = list(lo), list(hi)
        while True:
            ids = np.array(frontier); low, high = np.array(lower), np.array(upper)
            weights = self.weights[ids]
            qlow = weighted_quantile(low, weights, probability)
            qhigh = weighted_quantile(high, weights, probability)
            # This condition guarantees width <= abs + rel*|true quantile|,
            # using a valid interval lower bound on the unknown magnitude.
            minimum_magnitude = 0. if qlow <= 0 <= qhigh else min(abs(qlow), abs(qhigh))
            allowed = absolute_tolerance+relative_tolerance*max(scale_floor, minimum_magnitude)
            converged = qhigh-qlow <= allowed
            if converged: reason = "requested discrete quantile interval width reached"; break
            candidates = np.flatnonzero((self.children[ids, 0] >= 0) & (high >= qlow) & (low <= qhigh))
            if not len(candidates): reason = "no unresolved quantile-overlap node"; break
            if evaluations+2 > budget: reason = "node budget exhausted; interval retained, full/ROI fallback required"; break
            # Refine the widest relevant node; all regions remain represented
            # with their exact physical volume weight at every query stage.
            split = int(candidates[np.argmax(high[candidates]-low[candidates])])
            children = self.children[frontier[split]]
            child_low, child_high = evaluate(children)
            frontier[split:split+1] = children.tolist()
            lower[split:split+1] = child_low.tolist(); upper[split:split+1] = child_high.tolist()
        return QuantileResult(float(qlow), float(qhigh), float((qlow+qhigh)/2),
            weighted_cvar(low, weights, probability), weighted_cvar(high, weights, probability),
            float(weights[low > threshold].sum()), float(weights[high > threshold].sum()), bool(converged),
            evaluations, leaves, len(frontier), self.total_elements, reason)
