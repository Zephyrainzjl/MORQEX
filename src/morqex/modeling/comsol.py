"""Streaming COMSOL Mesh-class v4 ASCII export with explicit material mapping.

Specification: COMSOL6.3 Programming Reference, Mesh file format. This module
writes mesh geometry, not an MPH physics model or a COMSOL solver invocation.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np


@contextmanager
def _arrays(mesh):
    from .meshing import MeshArtifact
    if isinstance(mesh,MeshArtifact):
        import h5py
        with h5py.File(mesh.hdf5) as f:
            if not f.attrs.get('complete',False):raise ValueError('Incomplete mesh artifact')
            yield f['mesh/points_m'],f['mesh/tetrahedra'],f['mesh/phase'],'tetra'
    else:
        from morqex.microstructure.mesh.voxel import FEMesh
        if not isinstance(mesh,FEMesh):raise TypeError('FEMesh or MeshArtifact required')
        yield mesh.points,mesh.cells,mesh.phases,mesh.cell_type


def export_comsol(mesh,path,*,region_names=None,chunk_size=50000,include_boundaries=True):
    """Write .mphtxt and .regions.json without a complete ASCII string in RAM.

    Original region0 is a valid material. Domains start1; boundary entities
    start0. TetGen's marked material interfaces are retained. Other meshes
    export domains only, letting COMSOL reconstruct boundary topology.
    """
    from .voxel import external_path
    path=external_path(path)
    if path.suffix!='.mphtxt':raise ValueError('Use .mphtxt for native COMSOL ASCII mesh')
    if type(chunk_size) is not int or chunk_size<1:raise ValueError('Positive chunk size required')
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pending')
    with _arrays(mesh) as (points,cells,phases,kind):
        if kind not in ('tetra','hexahedron'):raise ValueError('Only linear Tet4 and Hex8 are supported')
        width=4 if kind=='tetra' else 8;ordering=np.arange(4) if kind=='tetra' else np.array([0,1,3,2,4,5,7,6])
        if len(cells)==0 or cells.shape!=(len(phases),width):raise ValueError('Invalid element/phase array shapes')
        labels=set()
        for i in range(0,len(phases),chunk_size):
            block=np.asarray(phases[i:i+chunk_size])
            if block.dtype.kind not in 'biu' or np.any(block<0):raise ValueError('Nonnegative integer material labels required')
            labels.update(map(int,np.unique(block)))
        labels=sorted(labels);mapping={k:i+1 for i,k in enumerate(labels)}
        boundaries=getattr(mesh,'facets',None) if include_boundaries else None
        markers=getattr(mesh,'facet_markers',None) if include_boundaries else None
        # HDF5 TetGen output retains its marked faces; avoid loading its volume mesh.
        if boundaries is None and include_boundaries and hasattr(mesh,'hdf5'):
            import h5py
            with h5py.File(mesh.hdf5) as h:
                if 'mesh/facets' in h:boundaries=h['mesh/facets'][:];markers=h['mesh/facet_markers'][:]
        if boundaries is not None:
            boundaries=np.asarray(boundaries);markers=np.asarray(markers)
            if boundaries.shape!=(len(markers),3):raise ValueError('Boundary triangles/markers mismatch')
            boundary_map={int(m):i for i,m in enumerate(np.unique(markers))}
        else:boundary_map={}
        with temporary.open('w') as out:
            out.write('''# MORQEX; coordinates in metres; material map in .regions.json
0 1
1 # number of tags
5 mesh1
1 # number of types
3 obj
0 0 1
4 Mesh
4 # Mesh class version
3 # space dimension
''')
            out.write(f'{len(points)} # number of vertices\n0 # lowest vertex index\n')
            for i in range(0,len(points),chunk_size):
                block=np.asarray(points[i:i+chunk_size])
                if block.shape[1]!=3 or not np.isfinite(block).all():raise ValueError('Invalid mesh coordinates')
                np.savetxt(out,block,fmt='%.17g')
            out.write(f'{1+bool(boundary_map)} # number of element types\n')
            def header(name,nvertices,count):out.write(f'{len(name)} {name}\n{nvertices}\n{count}\n')
            header('tet' if kind=='tetra' else 'hex',width,len(cells))
            for i in range(0,len(cells),chunk_size):
                block=np.asarray(cells[i:i+chunk_size])
                if block.dtype.kind not in 'iu' or block.min()<0 or block.max()>=len(points):raise ValueError('Invalid mesh connectivity')
                np.savetxt(out,block[:,ordering],fmt='%d')
            out.write(f'{len(cells)} # geometric entity indices\n')
            keys=np.array(labels)
            for i in range(0,len(phases),chunk_size):
                np.savetxt(out,np.searchsorted(keys,phases[i:i+chunk_size])+1,fmt='%d')
            if boundary_map:
                header('tri',3,len(boundaries))
                if boundaries.min()<0 or boundaries.max()>=len(points):raise ValueError('Invalid boundary node IDs')
                for i in range(0,len(boundaries),chunk_size):np.savetxt(out,boundaries[i:i+chunk_size],fmt='%d')
                out.write(f'{len(boundaries)}\n')
                np.savetxt(out,np.searchsorted(sorted(boundary_map),markers),fmt='%d')
    report={'format':'COMSOL Mesh class4 ASCII','coordinates':'XYZ metres; set COMSOL import length unit to m',
        'region_to_domain':mapping,'region_names':region_names or {},'facet_marker_to_boundary':boundary_map,
        'boundary_policy':'marked PLC interfaces' if boundary_map else 'COMSOL generates missing boundary/edge/point elements',
        'verification':'format and numerical read-back; COMSOL application import requires a local COMSOL installation'}
    temporary.replace(path)
    path.with_suffix('.regions.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return path


def read_comsol(path):
    """Independent strict reader for this Mesh-class4 subset, for round-trip checks.

    Not a general COMSOL model parser; selections/higher-order elements are
    intentionally unsupported. Use meshio/COMSOL for unrelated file variants.
    """
    with Path(path).open() as src:
        lines=(line.split('#',1)[0].strip() for line in src)
        lines=(line for line in lines if line)
        def line():
            try:return next(lines)
            except StopIteration:raise ValueError('Truncated COMSOL mesh') from None
        def number():return int(line())
        def string():
            n,s=line().split(' ',1)
            if int(n)!=len(s):raise ValueError('Invalid COMSOL string length')
            return s
        if line()!='0 1' or number()!=1:raise ValueError('Unsupported COMSOL file header')
        string()
        if number()!=1 or string()!='obj' or line()!='0 0 1' or string()!='Mesh':raise ValueError('Unsupported object header')
        if number()!=4 or number()!=3:raise ValueError('Requires Mesh-class4 /3D')
        n=number();offset=number();points=np.array([[float(x) for x in line().split()] for _ in range(n)])
        if points.shape!=(n,3) or not np.isfinite(points).all():raise ValueError('Invalid COMSOL vertices')
        elements={}
        for _ in range(number()):
            name=string();width=number();count=number()
            if name not in ('tet','hex','tri') or width!={'tet':4,'hex':8,'tri':3}[name]:raise ValueError('Unsupported element type')
            cells=np.array([[int(x) for x in line().split()] for _ in range(count)],dtype=np.int64)-offset
            if cells.shape!=(count,width) or cells.min()<0 or cells.max()>=n:raise ValueError('Invalid COMSOL element indices')
            if number()!=count:raise ValueError('Entity count does not match elements')
            ids=np.array([number() for _ in range(count)])
            if ids.min()<(1 if name in ('tet','hex') else 0):raise ValueError('Invalid geometric entity numbering')
            elements[name]={'cells':cells,'entities':ids}
        if next(lines,None) is not None:raise ValueError('Unsupported trailing objects')
    return {'points':points,'elements':elements}
