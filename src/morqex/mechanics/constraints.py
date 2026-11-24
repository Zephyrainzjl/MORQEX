"""Kinematic/periodic identification and explicitly removed compatible rigid modes."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space, qr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class ConstraintMap:
    projection: object
    equilibrium_projection: object
    boundary: str
    rigid_modes_removed: int


def _rigid(points, scale):
    x = (points-points.mean(axis=0))/scale
    modes = np.zeros((len(points), 3, 6))
    modes[:, :, :3] = np.eye(3)
    for k in range(3):
        modes[:, :, 3+k] = np.cross(np.eye(3)[k], x)
    return modes


def constraints(mesh, boundary):
    if boundary not in {"kubc", "periodic", "neumann"}:
        raise ValueError("Boundary must be kubc, periodic or neumann")
    n = len(mesh.points)
    fixed = mesh.outer_nodes() if boundary == "kubc" else np.array([], dtype=int)
    pairs = []
    if boundary == "periodic":
        scaled = (mesh.points-mesh.box[0])/(mesh.box[1]-mesh.box[0])
        integer_keys = np.round(scaled*1e10).astype(np.int64)
        lookup = {tuple(row): i for i, row in enumerate(integer_keys)}
        for axis in range(3):
            for i in np.flatnonzero((integer_keys[:, axis] == 0) | (integer_keys[:, axis] == 10**10)):
                other = integer_keys[i].copy(); other[axis] = 10**10 - other[axis]
                if tuple(other) not in lookup:
                    raise ValueError("Periodic faces require matching solid nodes; geometry is not periodic")
                j = lookup[tuple(other)]
                pairs.append((i, j))
        keys = integer_keys.copy(); keys[keys == 10**10] = 0
        _, groups = np.unique(keys, axis=0, return_inverse=True)
    else:
        groups = np.arange(n)
    rows = np.arange(3*n)
    columns = (3*groups[:, None]+np.arange(3)).reshape(-1)
    available = np.ones(columns.max()+1, dtype=bool)
    available[(3*groups[fixed, None]+np.arange(3)).reshape(-1)] = False
    original_columns = np.flatnonzero(available)
    full = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(3*n, len(available))).tocsr()
    before_gauge = full[:, original_columns]
    gauge_columns = []
    scale = float(np.max(mesh.box[1]-mesh.box[0]))
    # An island cut by a periodic face has separate Euclidean components but one
    # periodic body. Its rigid rotation needs different translations on each cut
    # piece. Compute that compatible nullspace instead of suppressing rotations.
    if pairs:
        edges = np.array([[mesh.components[i], mesh.components[j]] for i, j in pairs])
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(mesh.n_components,)*2)
        cluster_count, cluster = connected_components(graph, directed=False)
    else:
        cluster_count, cluster = mesh.n_components, np.arange(mesh.n_components)
    for body in range(cluster_count):
        components = np.flatnonzero(cluster == body)
        nodes = np.flatnonzero(np.isin(mesh.components, components))
        if len(nodes)*3*6*len(components)*8 > 128*2**20:
            raise ValueError("Periodic rigid-nullspace analysis exceeds the small-reference budget; partition geometry")
        modes = np.zeros((len(nodes), 3, 6*len(components)))
        for block, component in enumerate(components):
            inside = mesh.components[nodes] == component
            modes[inside, :, 6*block:6*(block+1)] = _rigid(mesh.points[nodes[inside]], scale)
        local_index = {int(node): i for i, node in enumerate(nodes)}
        restrictions = [modes[local_index[int(i)]] for i in fixed if cluster[mesh.components[i]] == body]
        restrictions += [modes[local_index[i]]-modes[local_index[j]] for i, j in pairs if cluster[mesh.components[i]] == body]
        allowed = null_space(np.concatenate(restrictions, axis=0), rcond=1e-10) if restrictions else np.eye(modes.shape[-1])
        if allowed.shape[1] == 0:
            continue
        values = (modes @ allowed).reshape(-1, allowed.shape[1])
        global_values = np.zeros((3*n, allowed.shape[1]))
        global_values[(3*nodes[:, None]+np.arange(3)).reshape(-1)] = values
        multiplicity = np.asarray(before_gauge.sum(axis=0)).ravel()
        reduced = np.asarray(before_gauge.T @ global_values) / multiplicity[:, None]
        _, triangular, pivots = qr(reduced.T, pivoting=True, mode="economic")
        rank = np.sum(np.abs(np.diag(triangular)) > 1e-10)
        if rank != allowed.shape[1]:
            raise ValueError("Rigid modes incompatible with constraint map")
        gauge_columns.extend(pivots[:rank].tolist())
    keep = np.ones(before_gauge.shape[1], dtype=bool); keep[gauge_columns] = False
    return ConstraintMap(before_gauge[:, keep], before_gauge, boundary, len(gauge_columns))
