"""Explicit physical networks and cache identities for topology transitions."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from morqex.extensions import Registry


def _owned(array, dtype=None):
    result = np.array(array, dtype=dtype, copy=True, order='C')
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ConductingNetwork:
    """Undirected, explicit contact graph for ONE transport mechanism.

    Electronic conductance uses siemens. Ionic is an electrical ionic network,
    not solid diffusion. A zero edge is a physical disconnection. No spatial
    proximity or damage-threshold rule is inferred by this class. Floating
    components remain present, including their inventory in the caller.
    """
    vertex_count: int
    edges: np.ndarray
    conductance_S: np.ndarray
    anchors: np.ndarray
    channel: str
    provenance: str

    def __post_init__(self):
        n = self.vertex_count
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError('Positive integer vertex_count required')
        if self.channel not in {'electronic', 'ionic'} or not self.provenance.strip():
            raise ValueError('Explicit channel and provenance required')
        e, g, a = np.asarray(self.edges), np.asarray(self.conductance_S, dtype=float), np.asarray(self.anchors)
        if e.ndim != 2 or e.shape[1] != 2 or e.dtype.kind not in 'iu':
            raise ValueError('Integer edges with shape (E,2) required')
        if np.any(e < 0) or np.any(e >= n) or np.any(e[:, 0] == e[:, 1]):
            raise ValueError('Invalid graph edge')
        e = np.sort(e, axis=1)
        order = np.lexsort((e[:, 1], e[:, 0]))
        if len(np.unique(e, axis=0)) != len(e):
            raise ValueError('Aggregate parallel physical contacts explicitly before construction')
        if g.shape != (len(e),) or not np.isfinite(g).all() or np.any(g < 0):
            raise ValueError('Finite nonnegative conductance per edge required')
        if a.ndim != 1 or a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a >= n):
            raise ValueError('Integer reservoir-anchor vertices required')
        object.__setattr__(self, 'edges', _owned(e[order], np.int64))
        object.__setattr__(self, 'conductance_S', _owned(g[order], np.float64))
        object.__setattr__(self, 'anchors', _owned(np.unique(a), np.int64))

    def components(self):
        edges = self.edges[self.conductance_S > 0]
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(self.vertex_count, self.vertex_count)).tocsr()
        return connected_components(graph, directed=False)

    def accessible(self):
        """Reservoir reachability only; not an electrochemical activity fraction."""
        _, labels = self.components()
        return np.isin(labels, labels[self.anchors])

    @property
    def fingerprint(self):
        h = hashlib.sha256(json.dumps([int(self.vertex_count), self.channel, self.provenance]).encode())
        for a in (self.edges, self.conductance_S, self.anchors):
            h.update(a.tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class OperatorStamp:
    """Invalidate physical operators even when phase-label arrays are unchanged.

    Each input is a SHA256 of canonical data, not a user-readable version label.
    Material includes orientation/state-dependent laws; topology includes each
    physical contact channel, not only a thresholded crack image.
    """
    geometry: str
    topology: str
    materials: str
    boundary: str
    discretization: str
    backend: str

    def __post_init__(self):
        for value in vars(self).values():
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Every operator dependency requires a lowercase SHA256')

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(vars(self), sort_keys=True).encode()).hexdigest()

    def require_same(self, current):
        if not isinstance(current, OperatorStamp) or self != current:
            raise ValueError('Stale operator: rehomogenize/rebuild before advancing')


class TopDamageRegistry(Registry):
    """Same explicit ComponentSpec/factory API as MORQEX; no auto-discovery."""
    kinds = (*Registry.kinds, 'fatigue', 'topology-update', 'homogenizer',
             'cycle-stepper', 'state-transfer')
