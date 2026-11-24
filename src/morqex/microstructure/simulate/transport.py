"""NGSolve C++ FEM: piecewise isotropic steady conduction/diffusion on GEM meshes."""
from dataclasses import dataclass
import numpy as np

def to_ngsolve(mesh,periodic=False):
    import ngsolve as ng
    from netgen.meshing import Mesh, MeshPoint, Pnt, Element3D, Element2D, FaceDescriptor
    if mesh.cell_type != 'tetra' or np.any(mesh.signed_volumes() <= 0):
        raise ValueError('Expected positive tetrahedra')
    m = Mesh(dim=3)
    pids = [m.Add(MeshPoint(Pnt(*p))) for p in mesh.points]
    labels = sorted(np.unique(mesh.phases).tolist())
    regions = {label: i+1 for i, label in enumerate(labels)}
    for label, i in regions.items():
        m.SetMaterial(i, f'phase_{label}')
    for cell, phase in zip(mesh.cells, mesh.phases):
        m.Add(Element3D(regions[phase], [pids[int(i)] for i in cell]))
    lo, hi = mesh.points.min(0), mesh.points.max(0)
    faces = mesh.cells[:, [[1,2,3], [0,3,2], [0,1,3], [0,2,1]]].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    descriptors = {}
    names = ['xmin','xmax','ymin','ymax','zmin','zmax','insulated']
    tol = max(np.max(hi-lo)*1e-9, np.finfo(float).eps*10)
    for index in first[counts == 1]:
        face = faces[index]
        coords = mesh.points[face]
        tag = 6
        for axis in range(3):
            if np.all(abs(coords[:, axis]-lo[axis]) < tol): tag = 2*axis
            if np.all(abs(coords[:, axis]-hi[axis]) < tol): tag = 2*axis+1
        domain = regions[mesh.phases[index // 4]]
        key = (tag, domain)
        if key not in descriptors:
            descriptors[key] = m.Add(FaceDescriptor(surfnr=tag+1, domin=domain, domout=0, bc=tag+1))
            m.SetBCName(descriptors[key]-1, names[tag])
        m.Add(Element2D(descriptors[key], [pids[int(i)] for i in face]))
    if periodic:
        extent=hi-lo
        for axis in range(3):
            others=[i for i in range(3) if i != axis]
            low=np.flatnonzero(abs(mesh.points[:,axis]-lo[axis])<tol)
            high=np.flatnonzero(abs(mesh.points[:,axis]-hi[axis])<tol)
            def key(index):return tuple(np.round((mesh.points[index,others]-lo[others])/extent[others],12))
            lookup={key(i):i for i in low}
            if len(low)!=len(high) or any(key(i) not in lookup for i in high):
                raise ValueError('Periodic faces require matching nodes')
            for i in high:m.AddPointIdentification(pids[lookup[key(i)]],pids[i],axis+1,2)
    return ng.Mesh(m)

@dataclass
class ConductionResult:
    mesh: object
    field: object
    effective_conductivity: float
    relative_residual: float
    energy: float
    ndof: int

def conduction(mesh, coefficients, axis='x', order=2):
    """Unit potential difference; natural zero flux on lateral and pore boundaries.

    Every connected selected domain must touch a Dirichlet face. For isolated
    inclusions use the full multiphase mesh with finite phase conductivities.
    """
    import ngsolve as ng
    if axis not in 'xyz' or len(axis) != 1 or order < 1:
        raise ValueError('axis=x/y/z; order>=1')
    labels = np.unique(mesh.phases)
    if any(p not in coefficients or not np.isfinite(coefficients[p]) or coefficients[p] <= 0 for p in labels):
        raise ValueError('Each phase needs positive finite conductivity')
    m = to_ngsolve(mesh)
    fes = ng.H1(m, order=order, dirichlet=f'{axis}min|{axis}max')
    u, v = fes.TnT()
    k = m.MaterialCF({f'phase_{p}': float(coefficients[p]) for p in labels})
    a = ng.BilinearForm(fes, symmetric=True)
    a += k * ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx
    g = ng.GridFunction(fes)
    i = 'xyz'.index(axis)
    low, high = mesh.points[:, i].min(), mesh.points[:, i].max()
    length = high-low
    coord = (ng.x, ng.y, ng.z)[i]
    g.Set((coord-low)/length, definedon=m.Boundaries(f'{axis}min|{axis}max'))
    with ng.TaskManager():
        a.Assemble()
        rhs = g.vec.CreateVector()
        rhs.data = -a.mat*g.vec
        inv = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky')
        g.vec.data += inv*rhs
    residual_vec = g.vec.CreateVector()
    residual_vec.data = a.mat*g.vec
    residual = residual_vec.FV().NumPy().copy()
    free = np.array([fes.FreeDofs()[j] for j in range(fes.ndof)], bool)
    scale = np.linalg.norm(residual)
    rel = float(np.linalg.norm(residual[free])/max(scale, 1e-30))
    energy = float(ng.Integrate(k*ng.InnerProduct(ng.grad(g), ng.grad(g)), m))
    extent = np.ptp(mesh.points, axis=0)
    area = float(np.prod(np.delete(extent, i)))
    return ConductionResult(m, g, energy*length/area, rel, energy, fes.ndof)

def save_field(result, path):
    import ngsolve as ng
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ng.VTKOutput(ma=result.mesh, coefs=[result.field], names=['potential'],
                 filename=str(path.with_suffix('')), subdivision=0).Do()
