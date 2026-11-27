from dataclasses import dataclass
from pathlib import Path
import numpy as np
from morqex.microstructure.volume import Volume

@dataclass
class FEMesh:
    points: np.ndarray  # xyz in metres
    cells: np.ndarray
    phases: np.ndarray
    cell_type: str = 'tetra'

    def signed_volumes(self):
        if self.cell_type != 'tetra':
            raise ValueError('Requires tetrahedra')
        p = self.points[self.cells]
        return np.linalg.det(p[:, 1:] - p[:, :1]) / 6

    def boundary_faces(self):
        if self.cell_type != 'tetra':
            raise ValueError('Requires tetrahedra')
        faces = self.cells[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
        _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
        return faces[first[counts == 1]]

    def save(self, path):
        import meshio
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meshio.write(path, meshio.Mesh(self.points, [(self.cell_type, self.cells)],
                     cell_data={'phase': [self.phases]}))
        return path

    def save_comsol(self,path,**kwargs):
        """COMSOL Mesh-class4 writer, explicit region/domain mapping, no optional GUI dependencies."""
        from morqex.modeling.comsol import export_comsol
        return export_comsol(self,path,**kwargs)


def voxel_to_hex(volume, phases=None):
    if not isinstance(volume, Volume):
        volume = Volume(volume)
    data = volume.data
    if data.dtype.kind not in 'biu':
        raise ValueError('Mesh input must contain integer phase labels; segment grayscale CT first')
    if phases is None:
        try:
            from ._voxel_native import full_hexa
        except ImportError:
            pass
        else:
            points,cells,labels = full_hexa(np.ascontiguousarray(data,dtype=np.int64),volume.spacing,volume.origin)
            return FEMesh(points,cells,labels.astype(data.dtype,copy=False),'hexahedron')
    mask = np.ones(data.shape, bool) if phases is None else np.isin(data, phases)
    zyx = np.argwhere(mask)
    if len(zyx) == 0:
        raise ValueError('No selected voxels')
    # VTK hexahedron ordering; no duplicate vertices, shared phase interfaces.
    offsets = np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0],
                        [1,0,0],[1,0,1],[1,1,1],[1,1,0]])
    grid_shape = np.asarray(data.shape)+1
    strides = np.array([grid_shape[1]*grid_shape[2],grid_shape[2],1],dtype=np.int64)
    corner_ids = (zyx@strides)[:,None] + (offsets@strides)[None]
    if phases is None:
        ids = np.arange(np.prod(grid_shape))
        inverse = corner_ids.ravel()
    else:
        ids,inverse = np.unique(corner_ids.ravel(),return_inverse=True)
    points = np.column_stack(np.unravel_index(ids,tuple(grid_shape)))
    points = (points * np.asarray(volume.spacing) + volume.origin)[:, ::-1].copy()
    return FEMesh(points, inverse.reshape(-1, 8), data[tuple(zyx.T)], 'hexahedron')

def voxel_to_tetra(volume, phases=None):
    h = voxel_to_hex(volume, phases)
    # Globally consistent Freudenthal split along the low->high body diagonal.
    pattern = np.array([[0,1,2,6], [0,2,3,6], [0,3,7,6],
                        [0,7,4,6], [0,4,5,6], [0,5,1,6]])
    t = FEMesh(h.points, h.cells[:, pattern].reshape(-1, 4), np.repeat(h.phases, 6))
    neg = t.signed_volumes() < 0
    t.cells[neg] = t.cells[neg][:, [1,0,2,3]]
    return t

def surface(volume, phase=1):
    from skimage.measure import marching_cubes
    if not isinstance(volume, Volume):
        volume = Volume(volume)
    a = np.pad((volume.data == phase).astype(np.float32), 1)
    if not a.any():
        raise ValueError('Phase absent')
    p, f, _, _ = marching_cubes(a, .5, spacing=volume.spacing)
    p = (p - volume.spacing + np.asarray(volume.origin))[:, ::-1].copy()
    return FEMesh(p, f[:, ::-1].copy(), np.full(len(f), phase), 'triangle')
