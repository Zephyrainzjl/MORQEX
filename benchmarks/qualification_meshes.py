"""Independent convex-ball Delaunay mesh for analytic qualification only."""
import numpy as np
from scipy.spatial import Delaunay
from morqex.mechanics import TetMesh


def fitted_ball(n, radius=.4, seed=20261124):
    h = 1/n
    axis = np.arange(-radius+h/2, radius, h)
    inside = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = inside[np.linalg.norm(inside, axis=1) < radius-.45*h]
    inside += np.random.default_rng(seed).uniform(-.03*h, .03*h, size=inside.shape)
    count = int(np.ceil(4*np.pi*radius**2/h**2))
    k = np.arange(count)
    z = 1-2*(k+.5)/count
    phi = np.pi*(3-np.sqrt(5))*k
    sphere = radius*np.column_stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z])
    points = np.vstack([inside, sphere])+.5
    cells = Delaunay(points).simplices.copy()
    x = points[cells]
    det = np.linalg.det(np.swapaxes(x[:, 1:]-x[:, :1], 1, 2))
    negative = det < 0
    cells[negative] = cells[negative][:, [1, 0, 2, 3]]
    return TetMesh(points, cells, np.ones(len(cells), dtype=np.uint8), np.array([[0., 0., 0.], [1., 1., 1.]]))
