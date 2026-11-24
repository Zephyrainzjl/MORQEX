"""Conservative cell-centered diffusion on actual active voxels.

Diagonal XYZ diffusion only. Off-diagonal anisotropy needs an MPFA/FEM scheme
and is explicitly rejected rather than silently discarded.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


class ConcentrationInfeasible(ValueError):
    pass


@dataclass
class DiffusionState:
    concentration: np.ndarray
    inventory_mol: float
    inventory_error_mol: float
    relative_residual: float


class VoxelDiffusion:
    def __init__(self, volume, *, active_phases, diffusivity, particle_ids=None, reactive_phases=(), exterior_reactive=False):
        self.volume = volume
        a = np.asarray(volume.data)
        self.mask = np.isin(a, active_phases)
        self.voxel_ids = np.flatnonzero(self.mask)
        if not len(self.voxel_ids): raise ValueError("No active diffusion cells")
        index = np.full(a.shape, -1, dtype=int); index[self.mask] = np.arange(len(self.voxel_ids))
        self.coordinates_zyx = np.argwhere(self.mask)
        spacing = np.asarray(volume.spacing, dtype=float)
        self.centers = ((self.coordinates_zyx+.5)*spacing+volume.origin)[:, ::-1]
        self.rve_volume = float(np.prod(np.asarray(a.shape)*spacing))
        self.volumes = np.full(len(self.voxel_ids), np.prod(spacing))
        self.weights = self.volumes/self.rve_volume
        self.mass = diags(self.weights).tocsr()
        d = np.asarray(diffusivity, dtype=float)
        if d.shape == (3, 3) or (d.ndim >= 2 and d.shape[-2:] == (3, 3)):
            raise ValueError("Use diagonal XYZ diffusivity entries; full tensor diffusion is not implemented")
        d = np.broadcast_to(d, (len(self.voxel_ids), 3)).copy()
        if not np.isfinite(d).all() or np.any(d <= 0): raise ValueError("Positive finite diagonal diffusivities required")
        if particle_ids is not None:
            particle_ids = np.asarray(particle_ids)
            if particle_ids.shape != a.shape or particle_ids.dtype.kind not in "iu" or np.any(particle_ids[self.mask] <= 0):
                raise ValueError("Positive integer particle IDs required on all active voxels")
        self.particle_ids = particle_ids
        edge_i, edge_j, conductance = [], [], []
        for axis in range(3):
            low, high = [slice(None)]*3, [slice(None)]*3
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            low, high = tuple(low), tuple(high)
            valid = (index[low] >= 0) & (index[high] >= 0)
            if particle_ids is not None: valid &= particle_ids[low] == particle_ids[high]
            i, j = index[low][valid], index[high][valid]
            di, dj = d[i, 2-axis], d[j, 2-axis]
            edge_i.extend(i); edge_j.extend(j)
            conductance.extend((2*di*dj/(di+dj))*np.prod(spacing)/spacing[axis]**2/self.rve_volume)
        i, j, g = np.array(edge_i, dtype=int), np.array(edge_j, dtype=int), np.array(conductance)
        n = len(self.voxel_ids)
        self.stiffness = coo_matrix((np.r_[g, g, -g, -g], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()
        adjacency = coo_matrix((np.ones(2*len(i)), (np.r_[i, j], np.r_[j, i])), shape=(n, n))
        self.n_components, self.components = connected_components(adjacency)
        cells, areas, normals, centers, neighbors = [], [], [], [], []
        for cell, coordinate in enumerate(self.coordinates_zyx):
            for axis in range(3):
                for sign in (-1, 1):
                    other = coordinate.copy(); other[axis] += sign
                    outside = np.any(other < 0) or np.any(other >= a.shape)
                    neighbor = -1 if outside else int(a[tuple(other)])
                    continuous = not outside and self.mask[tuple(other)]
                    if continuous and particle_ids is not None:
                        continuous = particle_ids[tuple(coordinate)] == particle_ids[tuple(other)]
                    if continuous: continue
                    normal = np.zeros(3); normal[2-axis] = sign
                    cells.append(cell); areas.append(np.prod(spacing)/spacing[axis])
                    normals.append(normal); centers.append(self.centers[cell]+.5*spacing[axis]*normal)
                    neighbors.append(neighbor)
        self.face_cells, self.face_areas = np.array(cells, dtype=int), np.array(areas)
        self.face_normals, self.face_centers = np.array(normals), np.array(centers)
        self.face_neighbor_phase = np.array(neighbors)
        self.reactive = np.isin(self.face_neighbor_phase, reactive_phases)
        if exterior_reactive: self.reactive |= self.face_neighbor_phase == -1
        self.flux_operator = coo_matrix((self.face_areas/self.rve_volume, (self.face_cells, np.arange(len(cells)))), shape=(n, len(cells))).tocsr()
        self._factor = None
        self._dt = None

    def source(self, flux):
        if flux is None: return np.zeros(len(self.voxel_ids))
        j = np.broadcast_to(np.asarray(flux, dtype=float), self.face_areas.shape)
        if not np.isfinite(j).all(): raise ValueError("Finite inward molar face flux required")
        return np.asarray(self.flux_operator @ j)

    def step(self, concentration, dt, *, flux=None, maximum=None):
        if not np.isfinite(dt) or dt <= 0: raise ValueError("Positive time step required")
        c = np.asarray(concentration, dtype=float)
        if c.shape != self.weights.shape or not np.isfinite(c).all(): raise ValueError("Invalid concentration vector")
        if np.any(c < 0) or (maximum is not None and (not np.isfinite(maximum) or maximum <= 0 or np.any(c > maximum))):
            raise ConcentrationInfeasible("Initial concentration is outside physical bounds")
        if self._dt != dt:
            self._factor = splu((self.mass+dt*self.stiffness).tocsc()); self._dt = dt
        source = self.source(flux)
        rhs = self.weights*c+dt*source
        updated = self._factor.solve(rhs)
        if np.min(updated) < -1e-12*max(1., np.max(np.abs(c))) or (maximum is not None and np.any(updated > maximum)):
            raise ConcentrationInfeasible("Diffusion/flux step violates concentration bounds; reject the step rather than clip")
        inventory = self.volumes @ updated
        balance = inventory-self.volumes @ c-dt*self.rve_volume*source.sum()
        residual = (self.mass+dt*self.stiffness) @ updated-rhs
        return DiffusionState(updated, float(inventory), float(balance), float(np.linalg.norm(residual)/max(np.linalg.norm(rhs), np.finfo(float).tiny)))

    def expand(self, values, *, inactive=0.):
        a = np.full(self.volume.data.shape, inactive, dtype=float)
        a.ravel()[self.voxel_ids] = values
        return a


def conservative_box_projection(values, weights, target, *, lower=0., upper=1.):
    """Weighted L2 projection with exact weighted-mass equality and box bounds.

    This is a state transfer/feasibility tool, not an energy-preserving time step.
    """
    y, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    lo, hi = np.broadcast_to(lower, y.shape), np.broadcast_to(upper, y.shape)
    if y.ndim != 1 or y.shape != w.shape or not np.isfinite([y, w, lo, hi]).all() or np.any(w <= 0) or np.any(lo > hi) or not np.isfinite(target):
        raise ValueError("Invalid bounded projection")
    min_mass, max_mass = float(w @ lo), float(w @ hi)
    tolerance = 32*np.finfo(float).eps*max(abs(min_mass), abs(max_mass), abs(target), np.finfo(float).tiny)
    if target < min_mass-tolerance or target > max_mass+tolerance:
        raise ConcentrationInfeasible("Requested mass outside feasible interval")
    if target <= min_mass: return lo.copy()
    if target >= max_mass: return hi.copy()
    left, right = float(np.min(lo-y)), float(np.max(hi-y))
    for _ in range(100):
        middle = (left+right)/2
        candidate = np.clip(y+middle, lo, hi)
        if w @ candidate < target: left = middle
        else: right = middle
    result = np.clip(y+(left+right)/2, lo, hi)
    if abs(w @ result-target) > 4*tolerance: raise ConcentrationInfeasible("Bounded mass projection failed")
    return result
