"""Validated solid-only Tet4 geometry; RVE volume never shrinks with porosity."""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class TetMesh:
    points: np.ndarray
    cells: np.ndarray
    phases: np.ndarray
    box: np.ndarray
    voxel_ids: np.ndarray | None = None

    def __post_init__(self):
        self.points = np.array(self.points, dtype=float, copy=True)
        self.cells = np.array(self.cells, dtype=np.int64, copy=True)
        self.phases = np.array(self.phases, copy=True)
        self.box = np.array(self.box, dtype=float, copy=True)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.isfinite(self.points).all():
            raise ValueError("Finite XYZ points required")
        if self.cells.ndim != 2 or self.cells.shape[1] != 4 or len(self.cells) == 0:
            raise ValueError("Nonempty tetrahedral connectivity required")
        if self.cells.min() < 0 or self.cells.max() >= len(self.points):
            raise ValueError("Invalid node index")
        if len(np.unique(self.cells)) != len(self.points):
            raise ValueError("Remove unused nodes before constructing solid mesh")
        if self.phases.shape != (len(self.cells),) or self.phases.dtype.kind not in "iu":
            raise ValueError("One integer phase ID per tetrahedron required")
        if self.box.shape != (2, 3) or not np.isfinite(self.box).all() or np.any(self.box[1] <= self.box[0]):
            raise ValueError("Explicit RVE box [[xmin,ymin,zmin],[xmax,ymax,zmax]] required")
        tol = np.max(self.box[1] - self.box[0]) * 1e-10
        if np.any(self.points < self.box[0] - tol) or np.any(self.points > self.box[1] + tol):
            raise ValueError("Solid nodes outside RVE box")
        coordinates = self.points[self.cells]
        jac = np.swapaxes(coordinates[:, 1:] - coordinates[:, :1], 1, 2)
        determinant = np.linalg.det(jac)
        scale = np.linalg.norm(jac, axis=(1, 2))**3
        if np.any(determinant <= np.finfo(float).eps * scale * 10):
            raise ValueError("Tetrahedra must have positive, nondegenerate orientation")
        self.volumes = determinant / 6
        self.gradients = np.einsum("ij,ejk->eik", np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.linalg.inv(jac))
        self.centers = coordinates.mean(axis=1)
        self.rve_volume = float(np.prod(self.box[1] - self.box[0]))
        if self.voxel_ids is not None:
            self.voxel_ids = np.asarray(self.voxel_ids, dtype=np.int64).copy()
            if self.voxel_ids.shape != (len(self.cells),) or np.any(self.voxel_ids < 0):
                raise ValueError("One nonnegative parent voxel ID per tetrahedron required")
            self.voxel_ids.flags.writeable = False
        if self.volumes.sum() > self.rve_volume * (1 + 1e-10):
            raise ValueError("Element volume exceeds RVE volume (overlap or invalid box)")
        self.dofs = (3 * self.cells[..., None] + np.arange(3)).reshape(-1, 12)
        edges = self.cells[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)
        graph = coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(self.points),)*2)
        self.n_components, self.components = connected_components(graph)
        for array in (self.points, self.cells, self.phases, self.box, self.volumes, self.gradients, self.dofs):
            array.flags.writeable = False

    @classmethod
    def from_volume(cls, volume, solid_phases=None):
        from morqex.microstructure.mesh.voxel import voxel_to_tetra
        mesh = voxel_to_tetra(volume, phases=solid_phases)
        box = np.array([volume.origin, np.asarray(volume.origin) + np.asarray(volume.spacing)*volume.data.shape])[:, ::-1]
        selected = np.arange(volume.data.size) if solid_phases is None else np.flatnonzero(np.isin(volume.data, solid_phases))
        return cls(mesh.points, mesh.cells, mesh.phases, box, np.repeat(selected, 6))

    @property
    def fingerprint(self):
        h = hashlib.sha256()
        for a in (self.points, self.cells, self.phases, self.box):
            h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
        if self.voxel_ids is not None: h.update(self.voxel_ids.tobytes())
        return h.hexdigest()

    def boundary(self):
        faces = self.cells[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
        _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
        faces = faces[first[counts == 1]].copy()
        p = self.points[faces]
        cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        return faces, lengths/2, cross/lengths[:, None]

    def outer_nodes(self):
        tol = np.max(self.box[1]-self.box[0]) * 1e-10
        return np.flatnonzero(np.any((np.abs(self.points-self.box[0]) < tol) | (np.abs(self.points-self.box[1]) < tol), axis=1))

    def traction_load(self, traction, face_mask=None):
        faces, area, _ = self.boundary()
        if face_mask is not None:
            faces, area = faces[face_mask], area[face_mask]
        force = np.broadcast_to(np.asarray(traction, dtype=float), (len(faces), 3))
        if not np.isfinite(force).all():
            raise ValueError("Finite traction required")
        result = np.zeros_like(self.points)
        for j in range(3):
            np.add.at(result, faces[:, j], force * area[:, None] / 3)
        return result.reshape(-1)

    def pressure_load(self, pressure, face_mask=None):
        _, _, normals = self.boundary()
        if face_mask is not None:
            normals = normals[face_mask]
        p = np.asarray(pressure, dtype=float)
        return self.traction_load(-p[..., None] * normals, face_mask)

    def body_load(self, body_force):
        values = np.broadcast_to(np.asarray(body_force, dtype=float), (len(self.cells), 3))
        result = np.zeros_like(self.points)
        for j in range(4):
            np.add.at(result, self.cells[:, j], self.volumes[:, None]*values/4)
        return result.reshape(-1)
