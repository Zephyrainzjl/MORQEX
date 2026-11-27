"""Physical resistor graphs, source-aware Kron reduction and update certificates.

Independent implementation of Dörfler & Bullo, IEEE TCAS-I (2013),
doi:10.1109/TCSI.2012.2215780. This is NOT a new reduction method.
Only frozen, linear, passive transport is condensed. Diffusion inventories,
nonlinear reactions and fatigue history must remain explicit states.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .contracts import ConductingNetwork, _owned


def laplacian(network):
    e, g, n = network.edges, network.conductance_S, network.vertex_count
    i, j = e.T
    return sparse.coo_matrix((np.r_[g, g, -g, -g],
        (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()


@dataclass(frozen=True)
class GraphSolution:
    potential_V: np.ndarray
    terminal_current_A: np.ndarray
    dissipation_W: float
    relative_residual: float
    power_defect_W: float


class KronOperator:
    """Exact static Schur complement with explicit floating components.

    Terminal currents are outgoing into the network minus prescribed nodal
    sources. A source-free floating island has arbitrary constant potential;
    zero is reported as a gauge, never as an electrical connection. Sources
    on such islands are rejected (even balanced sources require a separate
    local solve). No dense N-by-N array or full inverse is constructed.
    """
    def __init__(self, network, terminals, *, max_terminals=256):
        b = np.asarray(terminals)
        if b.ndim != 1 or b.dtype.kind not in 'iu' or not len(b) or len(np.unique(b)) != len(b):
            raise ValueError('Distinct integer terminal vertices required')
        if np.any(b < 0) or np.any(b >= network.vertex_count) or len(b) > max_terminals:
            raise ValueError('Terminal range or dense Schur storage budget exceeded')
        self.network = network; self.terminals = _owned(b, np.int64)
        self.L = laplacian(network)
        _, labels = network.components()
        reachable = np.isin(labels, labels[b])
        free = reachable.copy(); free[b] = False
        self.interior = np.flatnonzero(free)
        self.floating = np.flatnonzero(~reachable)
        self.Lib = self.L[self.interior][:, b].tocsc()
        self.factor = None
        self.scale = 1.
        S = self.L[b][:, b].toarray()
        if len(self.interior):
            A = self.L[self.interior][:, self.interior].tocsc()
            self.scale = float(A.diagonal().max())
            self.factor = splu(A / self.scale)
            # Solve columns in small batches: bounded N*batch temporary memory.
            for start in range(0, len(b), 8):
                end = min(start + 8, len(b))
                X = self.factor.solve(self.Lib[:, start:end].toarray() / self.scale)
                S[:, start:end] -= self.Lib.T @ X
        asym = np.max(abs(S - S.T)) / max(np.max(abs(S)), np.finfo(float).tiny)
        if asym > 1e-9: raise ArithmeticError('Schur reciprocity failed')
        # A component with only one terminal has identically zero source-free
        # terminal conductance; avoid cancellation noise posing as leakage.
        terminal_labels = labels[b]
        for k, label in enumerate(terminal_labels):
            if np.count_nonzero(terminal_labels == label) == 1:
                S[k, :] = 0.; S[:, k] = 0.
        self.matrix_S = _owned((S + S.T) / 2)

    def condensed_source(self, source_A=None):
        # The overwhelmingly common source-free online query must not allocate
        # an N-vector or solve a zero N-dimensional right-hand side.
        if source_A is None:return np.zeros(len(self.terminals))
        f = np.asarray(source_A, float)
        if f.shape != (self.network.vertex_count,) or not np.isfinite(f).all():
            raise ValueError('Finite source at every vertex required')
        if np.any(f[self.floating] != 0):
            raise ValueError('Floating island sources require a separate conservative local solve')
        result = f[self.terminals].copy()
        if self.factor is not None:
            result -= self.Lib.T @ self.factor.solve(f[self.interior] / self.scale)
        return result

    def currents(self, terminal_potential_V, source_A=None):
        v = np.asarray(terminal_potential_V, float)
        if v.shape != (len(self.terminals),) or not np.isfinite(v).all():
            raise ValueError('Finite terminal potentials required')
        return self.matrix_S @ v - self.condensed_source(source_A)

    def recover(self, terminal_potential_V, source_A=None):
        v = np.asarray(terminal_potential_V, float)
        currents = self.currents(v, source_A)
        f = np.zeros(self.network.vertex_count) if source_A is None else np.asarray(source_A, float)
        u = np.zeros(self.network.vertex_count); u[self.terminals] = v
        if self.factor is not None:
            u[self.interior] = self.factor.solve((f[self.interior] - self.Lib @ v) / self.scale)
        residual = self.L @ u - f
        scale = max(np.linalg.norm(f), np.linalg.norm(self.Lib @ v), np.linalg.norm(self.L @ u), np.finfo(float).tiny)
        relative = float(np.linalg.norm(residual[self.interior]) / scale)
        e = self.network.edges
        power = float(self.network.conductance_S @ (u[e[:, 0]] - u[e[:, 1]])**2)
        defect = float(power - v @ currents - u @ f)
        roundoff = 128*np.finfo(float).eps*float(self.network.conductance_S.sum())*float(np.max(abs(u),initial=0))**2
        if relative > 1e-8 or abs(defect) > max(roundoff, 1e-8 * max(abs(power), abs(v @ currents), abs(u @ f), 1e-30)):
            raise ArithmeticError('Graph residual/power identity failed')
        return GraphSolution(_owned(u), _owned(currents), power, relative, defect)


@dataclass(frozen=True)
class UpdateCertificate:
    rebuild: bool
    relative_edge_change: float
    response_lower_factor: float
    response_upper_factor: float
    reason: str


def certify_update(reference, current, *, tolerance=.01):
    """Loewner bound for frozen passive networks, NOT a voltage/capacity bound.

    If |g_new/g_ref-1| <= eta < 1 on the same positive edges, then
    (1-eta) L_ref <= L_new <= (1+eta) L_ref. Minimizing interior
    potentials preserves this quadratic-form bound for the Schur response.
    Compare against the LAST REBUILT operator, not the preceding time step.
    Any edge creation/deletion triggers an unconditional rebuild.
    """
    if not np.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError('Subunit nonnegative update tolerance required')
    if (reference.vertex_count != current.vertex_count or reference.channel != current.channel
            or not np.array_equal(reference.edges, current.edges)
            or not np.array_equal(reference.anchors, current.anchors)):
        return UpdateCertificate(True, float('inf'), 0., float('inf'), 'graph layout or boundary changed')
    old, new = reference.conductance_S, current.conductance_S
    if np.any((old > 0) != (new > 0)):
        return UpdateCertificate(True, float('inf'), 0., float('inf'), 'edge creation or deletion')
    mask = old > 0
    eta = float(np.max(abs(new[mask] / old[mask] - 1), initial=0))
    return UpdateCertificate(eta > tolerance, eta, max(0., 1-eta), 1+eta,
        'coefficient tolerance exceeded' if eta > tolerance else 'passive quadratic-response bound')


@dataclass(frozen=True)
class VoxelNetwork:
    network: ConductingNetwork
    voxel_ids: np.ndarray
    voxel_edges: np.ndarray
    edge_axes_zyx: np.ndarray
    boundary_terminals: np.ndarray
    length_m: float
    area_m2: float
    shape: tuple

    def reservoir_accessible(self):
        # The opposite homogenization electrode is a TEST boundary only.
        # Keeping it here would short disconnected material through that face.
        e = self.network.edges
        probe = np.setdiff1d(self.boundary_terminals, self.network.anchors)
        keep = ~np.isin(e, probe).any(axis=1)
        physical = ConductingNetwork(self.network.vertex_count, e[keep],
            self.network.conductance_S[keep], self.network.anchors,
            self.network.channel, 'single physical reservoir; opposite test terminal removed')
        field = np.zeros(int(np.prod(self.shape)), bool)
        field[self.voxel_ids] = physical.accessible()[:len(self.voxel_ids)]
        return field.reshape(self.shape)

    def homogenize(self):
        op = KronOperator(self.network, self.boundary_terminals)
        result = op.recover([0., 1.])
        field = np.zeros(int(np.prod(self.shape)))
        field[self.voxel_ids] = result.potential_V[:len(self.voxel_ids)]
        return {'effective_S_m': result.dissipation_W * self.length_m / self.area_m2,
                'operator': op, 'solution': result, 'potential': field.reshape(self.shape)}


def voxel_network(conductivity_S_m, spacing_zyx_m, *, axis=2,
                  channel='electronic', reservoir_side=1, face_factors=None):
    """Orthogonal FVM graph with exact half-cell boundary resistances.

    face_factors[axis] has full voxel shape; its last plane is unused. It
    multiplies the conductance of the positive-axis face, e.g. supplied by
    an independently validated contact law. Zero explicitly severs a face.
    No rule converting phase-field damage into pore space is imposed here.
    """
    k = np.asarray(conductivity_S_m, float); h = np.asarray(spacing_zyx_m, float)
    if k.ndim != 3 or min(k.shape) < 2 or not np.isfinite(k).all() or np.any(k < 0):
        raise ValueError('Finite nonnegative 3D conductivity required')
    if h.shape != (3,) or not np.isfinite(h).all() or np.any(h <= 0) or axis not in (0,1,2) or reservoir_side not in (0,1):
        raise ValueError('Positive spacing and valid boundary axis/side required')
    factors = np.ones((3,) + k.shape) if face_factors is None else np.asarray(face_factors, float)
    if factors.shape != (3,) + k.shape or not np.isfinite(factors).all() or np.any(factors < 0):
        raise ValueError('Finite nonnegative physical face factors required')
    ids = np.flatnonzero(k > 0); n = len(ids)
    index = np.full(k.shape, -1, np.int64); index.ravel()[ids] = np.arange(n)
    all_ids = np.arange(k.size).reshape(k.shape)
    edges = []; weights = []; voxel_edges = []; axes = []
    for a in range(3):
        lo = [slice(None)]*3; hi = lo.copy(); lo[a] = slice(None,-1); hi[a] = slice(1,None)
        lo, hi = tuple(lo), tuple(hi)
        mask = (index[lo] >= 0) & (index[hi] >= 0)
        kl, kr = k[lo][mask], k[hi][mask]
        small, large = np.minimum(kl,kr), np.maximum(kl,kr)
        g = 2*small/(1+small/large)*np.prod(h)/h[a]**2*factors[a][lo][mask]
        edges.extend(zip(index[lo][mask], index[hi][mask])); weights.extend(g)
        voxel_edges.extend(zip(all_ids[lo][mask], all_ids[hi][mask])); axes.extend([a]*len(g))
    coords = np.asarray(np.unravel_index(ids,k.shape)).T
    for side in (0,1):
        face = np.flatnonzero(coords[:,axis] == (0 if side == 0 else k.shape[axis]-1))
        edges.extend((int(i),n+side) for i in face)
        weights.extend(2*k.ravel()[ids[face]]*np.prod(h)/h[axis]**2)
    # ConductingNetwork canonicalizes edges; auxiliary voxel-edge order is
    # deliberately separate and is never indexed by its sorted conductances.
    graph = ConductingNetwork(n+2,np.asarray(edges,dtype=np.int64).reshape(-1,2),np.asarray(weights),
        np.array([n+reservoir_side]),channel,'orthogonal voxel FVM; supplied constitutive conductivity and face factors')
    length = float(k.shape[axis]*h[axis]); area = float(np.prod(np.array(k.shape)*h)/length)
    return VoxelNetwork(graph,_owned(ids),_owned(np.asarray(voxel_edges,np.int64).reshape(-1,2)),
        _owned(axes,np.int8),_owned([n,n+1],np.int64),length,area,k.shape)
