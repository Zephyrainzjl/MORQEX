"""Streaming COMSOL Mesh-class v4 ASCII export with explicit material mapping.

Specification: COMSOL6.3 Programming Reference, Mesh file format. This module
writes mesh geometry, not an MPH physics model or a COMSOL solver invocation.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np

LENGTH_UNITS={'m':1.,'cm':1e-2,'mm':1e-3,'um':1e-6,'µm':1e-6,'nm':1e-9}


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


def export_comsol(mesh,path,*,region_names=None,chunk_size=50000,include_boundaries=True,
                  include_region_selections=True,selections=None,mesh_tag='mesh1',preserve_domain_ids=False,length_unit='m'):
    """Write .mphtxt and .regions.json without a complete ASCII string in RAM.

    Original region0 is a valid material. Domains start1; boundary entities
    start0. TetGen's marked material interfaces are retained. Other meshes
    export domains only, letting COMSOL reconstruct boundary topology.
    """
    from .voxel import external_path
    if length_unit not in LENGTH_UNITS:raise ValueError('Supported length units: m, cm, mm, um, nm')
    unit_scale=LENGTH_UNITS[length_unit]
    path=external_path(path)
    if path.suffix!='.mphtxt':raise ValueError('Use .mphtxt for native COMSOL ASCII mesh')
    if type(chunk_size) is not int or chunk_size<1:raise ValueError('Positive chunk size required')
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    if temporary.exists():raise FileExistsError('Archive interrupted COMSOL output before retrying: '+str(temporary))
    with _arrays(mesh) as (points,cells,phases,kind):
        if kind not in ('tetra','hexahedron'):raise ValueError('Only linear Tet4 and Hex8 are supported')
        width=4 if kind=='tetra' else 8;ordering=np.arange(4) if kind=='tetra' else np.array([0,1,3,2,4,5,7,6])
        if len(cells)==0 or cells.shape!=(len(phases),width):raise ValueError('Invalid element/phase array shapes')
        labels=set()
        for i in range(0,len(phases),chunk_size):
            block=np.asarray(phases[i:i+chunk_size])
            if block.dtype.kind not in 'biu' or np.any(block<0):raise ValueError('Nonnegative integer material labels required')
            labels.update(map(int,np.unique(block)))
        labels=sorted(labels)
        if preserve_domain_ids and min(labels)<1:raise ValueError('Preserved COMSOL domain IDs must be positive')
        mapping={k:k if preserve_domain_ids else i+1 for i,k in enumerate(labels)}
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
        import re
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',mesh_tag):raise ValueError('ASCII COMSOL mesh tag required')
        named={}
        if include_region_selections:
            for k in labels:
                label=str((region_names or {}).get(k,f'region_{k}'))
                # Stable ASCII labels are portable across COMSOL versions;
                # the sidecar retains arbitrary original Unicode names.
                if not label.isascii():label=f'region_{k}'
                named[f'region_{k}']={'label':label,'dimension':3,'entities':[mapping[k]]}
        for tag,selection in (selections or {}).items():
            if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',tag) or tag==mesh_tag:raise ValueError('Unique ASCII selection tag required')
            if tag in named:raise ValueError('Selection tag collides with an automatic domain selection; use region_names or disable automatic selections')
            item=dict(selection);dim=item.get('dimension',3);label=item.get('label',tag)
            if not isinstance(label,str) or not label.isascii() or '\n' in label or '\r' in label:raise ValueError('Use a one-line ASCII selection label; Unicode region names remain in the sidecar')
            if 'regions' in item:
                if dim!=3 or 'entities' in item:raise ValueError('regions specifies domain selection only')
                entities=[mapping[int(k)] for k in item['regions']]
            else:entities=list(item.get('entities',[]))
            valid=set(mapping.values()) if dim==3 else set(boundary_map.values()) if dim==2 else set()
            if dim not in (0,1,2,3) or not set(entities)<=valid:raise ValueError('Selection refers to undefined mesh entities')
            named[tag]={'label':label,'dimension':dim,'entities':sorted(set(map(int,entities)))}
        with temporary.open('w') as out:
            out.write(f'# MORQEX length_unit={length_unit}; source geometry in SI metres; material map in .regions.json\n0 1\n')
            out.write(f'{1+len(named)} # number of tags\n{len(mesh_tag)} {mesh_tag}\n')
            for tag in named:out.write(f'{len(tag)} {tag}\n')
            out.write(f'{1+len(named)} # number of types\n'+'3 obj\n'*(1+len(named)))
            out.write('0 0 1\n4 Mesh\n4 # Mesh class version\n3 # space dimension\n')
            out.write(f'{len(points)} # number of vertices\n0 # lowest vertex index\n')
            for i in range(0,len(points),chunk_size):
                block=np.asarray(points[i:i+chunk_size])
                if block.shape[1]!=3 or not np.isfinite(block).all():raise ValueError('Invalid mesh coordinates')
                block=block/unit_scale
                if not np.isfinite(block).all():raise ValueError('Export unit produces nonfinite mesh coordinates')
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
                np.savetxt(out,np.array([mapping[k] for k in labels])[np.searchsorted(keys,phases[i:i+chunk_size])],fmt='%d')
            if boundary_map:
                header('tri',3,len(boundaries))
                if boundaries.min()<0 or boundaries.max()>=len(points):raise ValueError('Invalid boundary node IDs')
                for i in range(0,len(boundaries),chunk_size):np.savetxt(out,boundaries[i:i+chunk_size],fmt='%d')
                out.write(f'{len(boundaries)}\n')
                np.savetxt(out,np.searchsorted(sorted(boundary_map),markers),fmt='%d')
            for item in named.values():
                label=item['label'];entities=item['entities']
                out.write(f"0 0 1\n9 Selection\n0\n{len(label)} {label}\n{len(mesh_tag)} {mesh_tag}\n{item['dimension']}\n{len(entities)}\n")
                np.savetxt(out,entities,fmt='%d')
    report={'format':'COMSOL Mesh class4 ASCII','coordinates':f'XYZ {length_unit}; set COMSOL import length unit to {length_unit}',
        'length_unit':length_unit,'unit_scale_m':unit_scale,
        'region_to_domain':mapping,'region_names':region_names or {},'facet_marker_to_boundary':boundary_map,
        'boundary_policy':'marked PLC interfaces' if boundary_map else 'COMSOL generates missing boundary/edge/point elements',
        'mesh_tag':mesh_tag,'selections':named,
        'verification':'format and numerical read-back; COMSOL application import requires a local COMSOL installation'}
    temporary.replace(path)
    path.with_suffix('.regions.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return path


def read_comsol(path):
    """Read one3D Mesh-class4 plus Selection-class0 objects, without relabelling.

    Coordinates retain file units. They are NOT inferred from their magnitudes.
    New MORQEX files declare units in a comment; unrelated files return None.
    Only linear element types below are supported, not arbitrary MPH models.
    """
    widths={'tet':4,'hex':8,'tri':3,'quad':4,'edg':2,'vtx':1,'prism':6,'pyr':5}
    dimensions={'tet':3,'hex':3,'prism':3,'pyr':3,'tri':2,'quad':2,'edg':1,'vtx':0}
    coordinate_unit=None
    with Path(path).open(encoding='utf-8') as src:
        def line():
            nonlocal coordinate_unit
            for raw in src:
                if raw.startswith('# MORQEX length_unit='):coordinate_unit=raw.split('=',1)[1].split(';',1)[0]
                text=raw.split('#',1)[0].strip()
                if text:return text
            raise ValueError('Truncated COMSOL mesh')
        def number():return int(line())
        def string():
            value=line();parts=value.split(maxsplit=1)
            if len(parts)!=2:raise ValueError('Invalid COMSOL string')
            n,s=parts
            if int(n) not in (len(s),len(s.encode('utf-8'))):raise ValueError('Invalid COMSOL string length')
            return s
        def matrix(count,width,dtype):
            if count<0 or count>100000000:raise ValueError('COMSOL allocation budget exceeded')
            out=np.empty((count,width),dtype=dtype)
            for start in range(0,count,50000):
                end=min(start+50000,count)
                data=np.fromstring(' '.join(line() for _ in range(end-start)),sep=' ',dtype=dtype)
                if data.size!=(end-start)*width:raise ValueError('Invalid COMSOL matrix row width')
                out[start:end]=data.reshape(-1,width)
            return out
        if line().split()!=['0','1']:raise ValueError('Unsupported COMSOL file version')
        count=number()
        if not 1<=count<=100000:raise ValueError('Invalid object count')
        tags=[string() for _ in range(count)]
        if len(set(tags))!=count or number()!=count:raise ValueError('Inconsistent COMSOL tags/types')
        if any(string()!='obj' for _ in range(count)):raise ValueError('Unsupported COMSOL object type')
        selections={};elements={};points=None;mesh_tag=None
        for tag in tags:
            if line().split()!=['0','0','1']:raise ValueError('Unsupported serialization header')
            kind=string()
            if kind=='Mesh':
                if points is not None or number()!=4 or number()!=3:raise ValueError('Requires exactly one Mesh-class4 /3D')
                mesh_tag=tag;n=number();offset=number();points=matrix(n,3,np.float64)
                if not n or not np.isfinite(points).all():raise ValueError('Invalid COMSOL coordinates')
                for _ in range(number()):
                    name=string();width=number();size=number()
                    if name not in widths or width!=widths[name] or name in elements:raise ValueError('Unsupported/repeated linear element type')
                    cells=matrix(size,width,np.int64)-offset
                    if cells.size and (cells.min()<0 or cells.max()>=n):raise ValueError('Invalid COMSOL element indices')
                    if number()!=size:raise ValueError('Entity count does not match elements')
                    entities=matrix(size,1,np.int64).ravel()
                    if entities.size and entities.min()<(1 if dimensions[name]==3 else 0):raise ValueError('Invalid geometric entity numbering')
                    elements[name]={'cells':cells,'entities':entities}
            elif kind=='Selection':
                if number()!=0:raise ValueError('Unsupported Selection version')
                label=string();target=string();dimension=number();size=number()
                selections[tag]={'label':label,'mesh_tag':target,'dimension':dimension,'entities':matrix(size,1,np.int64).ravel().tolist()}
            else:raise ValueError(f'Unsupported COMSOL class {kind}')
        if points is None:raise ValueError('Mesh object missing')
        valid={d:set() for d in range(4)}
        for name,block in elements.items():valid[dimensions[name]].update(map(int,np.unique(block['entities'])))
        for selection in selections.values():
            d=selection['dimension']
            if selection['mesh_tag']!=mesh_tag or d not in valid or not set(selection['entities'])<=valid[d]:
                raise ValueError('Selection refers to undefined mesh entities')
        for raw in src:
            if raw.split('#',1)[0].strip():raise ValueError('Unsupported trailing objects')
    return {'points':points,'elements':elements,'selections':selections,'mesh_tag':mesh_tag,'tags':tags,'length_unit':coordinate_unit}
