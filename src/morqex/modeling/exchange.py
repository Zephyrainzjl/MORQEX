"""Region-preserving exchange bundles with independent numerical read-backs."""
from pathlib import Path
import json,re
import numpy as np
from .voxel import external_path
from .comsol import export_comsol,read_comsol,LENGTH_UNITS


def _face_hash(face):
    """Fast candidate index only; every match is verified by node triples."""
    f=np.asarray(face,dtype=np.uint64)
    return (f[:,0]*np.uint64(0x9E3779B185EBCA87))^(f[:,1]*np.uint64(0xC2B2AE3D27D4EB4F))^(f[:,2]*np.uint64(0x165667B19E3779F9))


def _facet_owners(cells,facets,point_count):
    base=point_count;hashed=base**3>=np.iinfo(np.int64).max;structured=False
    def key(f):
        f=np.sort(f,axis=1).astype(np.int64)
        if structured:return np.ascontiguousarray(f).view(np.dtype([('a','i8'),('b','i8'),('c','i8')])).reshape(-1)
        return _face_hash(f) if hashed else (f[:,0]*base+f[:,1])*base+f[:,2]
    target=key(facets);order=np.argsort(target);sorted_keys=target[order]
    if len(target)>1 and np.any(sorted_keys[1:]==sorted_keys[:-1]):
        if hashed:
            # A rare hash collision must not merge two physical faces.
            structured=True;target=key(facets);order=np.argsort(target);sorted_keys=target[order]
        if np.any(sorted_keys[1:]==sorted_keys[:-1]):raise ValueError('Duplicate marked facet')
    owners=np.full((len(facets),2),-1,np.int64);sides=np.zeros((len(facets),2),np.int8)
    local=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    for start in range(0,len(cells),50000):
        block=cells[start:start+50000];faces=block[:,local].reshape(-1,3);keys=key(faces)
        loc=np.searchsorted(sorted_keys,keys);valid=loc<len(target)
        selected=np.flatnonzero(valid);selected=selected[sorted_keys[loc[selected]]==keys[selected]]
        if hashed and not structured:
            candidates=order[loc[selected]]
            same=np.all(np.sort(faces[selected],axis=1)==np.sort(facets[candidates],axis=1),axis=1)
            selected=selected[same]
        ids=order[loc[selected]];cell_id=start+selected//4;side=np.array([3,4,2,1])[selected%4]
        # Sort only matching marked faces in this bounded block. Repeated
        # interfaces have at most two owners, including across block boundaries.
        rank_order=np.argsort(ids,kind='stable');ids=ids[rank_order]
        cell_id=cell_id[rank_order];side=side[rank_order]
        if len(ids):
            first=np.r_[0,np.flatnonzero(np.diff(ids))+1]
            rank=np.arange(len(ids))-np.repeat(first,np.diff(np.r_[first,len(ids)]))
            column=(owners[ids,0]>=0).astype(np.int64)+rank
            if np.any(column>1) or np.any(owners[ids,column]>=0):raise ValueError('Nonmanifold facet owner count')
            owners[ids,column]=cell_id;sides[ids,column]=side
    if np.any(owners[:,0]<0):raise ValueError('Marked surface is not an actual volume-cell face')
    return owners,sides


def _abaqus(path,points,cells,domains,facets,markers,names):
    with path.open('w') as f:
        f.write('*HEADING\nMORQEX region and interface export; see bundle.json for units\n*NODE\n')
        for start in range(0,len(points),50000):
            p=points[start:start+50000];np.savetxt(f,np.column_stack((np.arange(start+1,start+len(p)+1),p)),delimiter=',',fmt=['%d','%.17g','%.17g','%.17g'])
        f.write('*ELEMENT, TYPE=C3D4\n')
        for start in range(0,len(cells),50000):
            c=cells[start:start+50000];np.savetxt(f,np.column_stack((np.arange(start+1,start+len(c)+1),c+1)),delimiter=',',fmt='%d')
        for k in np.unique(domains):
            name=names[int(k)];f.write(f'*ELSET, ELSET={name}\n')
            ids=np.flatnonzero(domains==k)+1
            for start in range(0,len(ids),16):f.write(', '.join(map(str,ids[start:start+16]))+'\n')
        if len(facets):
            owners,sides=_facet_owners(cells,facets,len(points))
            for marker in np.unique(markers):
                rows=np.flatnonzero(markers==marker)
                incident=owners[rows];present=incident>=0
                physical_domains=np.unique(domains[incident[present]])
                for domain in physical_domains:
                    selection=np.zeros(incident.shape,bool)
                    selection[present]=domains[incident[present]]==domain
                    rr,ss=np.nonzero(selection);chosen=rows[rr]
                    if not len(chosen):continue
                    f.write(f'*SURFACE, TYPE=ELEMENT, NAME=boundary_{int(marker)}_domain_{int(domain)}\n')
                    for row,side in zip(chosen,ss):f.write(f'{owners[row,side]+1}, S{int(sides[row,side])}\n')


def _verify_abaqus(path,points,cells,domains,names):
    """Independent bounded parser; meshio's large ASCII lists exceed RAM caps.

    Parse every node, Tet4 and material-set entry without retaining duplicate
    Python lists for tens of millions of elements. Named surface connectivity
    is independently checked by small API tests and actual facet ownership.
    """
    counts={'NODE':0,'ELEMENT':0};sets={v:k for k,v in names.items()}
    seen=np.zeros(len(cells),np.uint8);section='';domain=None
    with path.open() as stream:
        pending=None
        while True:
            line=pending if pending is not None else stream.readline();pending=None
            if not line:break
            if line.startswith('*'):
                parts=line.strip().split(',');section=parts[0][1:]
                if section=='ELSET':domain=sets[parts[1].split('=',1)[1].strip()]
                continue
            if section not in ('NODE','ELEMENT','ELSET'):continue
            rows=[line]
            for _ in range(9999):
                next_=stream.readline()
                if not next_:break
                if next_.startswith('*'):pending=next_;break
                rows.append(next_)
            numbers=np.fromstring(' '.join(rows).replace(',',' '),sep=' ',dtype=np.float64 if section=='NODE' else np.int64)
            if section in ('NODE','ELEMENT'):
                width=4 if section=='NODE' else 5
                values=numbers.reshape(-1,width);start=counts[section];stop=start+len(values)
                np.testing.assert_array_equal(values[:,0],np.arange(start+1,stop+1))
                if section=='NODE':np.testing.assert_allclose(values[:,1:],points[start:stop],rtol=1e-12,atol=1e-20)
                else:np.testing.assert_array_equal(values[:,1:]-1,cells[start:stop])
                counts[section]=stop
            else:
                indices=numbers-1
                if np.any((indices<0)|(indices>=len(cells))):raise ArithmeticError('Invalid Abaqus element set index')
                np.testing.assert_array_equal(domains[indices],np.full(len(indices),domain))
                if len(np.unique(indices))!=len(indices) or np.any(seen[indices]):raise ArithmeticError('Repeated material membership')
                seen[indices]=1
    if counts!={'NODE':len(points),'ELEMENT':len(cells)} or np.any(seen!=1):raise ArithmeticError('Incomplete Abaqus readback')


def export_bundle(mesh,directory,*,region_names=None,selections=None,length_unit='m',
                  formats=('comsol','vtu','gmsh','medit','abaqus')):
    """Export selected formats; verify nodes/cells/domain tags by reading them.

    Canonical HDF5/XDMF remains SI. Other formats use length_unit and a unit
    sidecar because their mesh file formats do not consistently encode units.
    Shared-node interfaces are bonded; naming surfaces does not add contact.
    """
    import meshio
    from .comsol import _arrays
    if length_unit not in LENGTH_UNITS:raise ValueError('Unsupported export length unit')
    if not set(formats)<=set(('comsol','vtu','gmsh','medit','abaqus')):raise ValueError('Unsupported exchange format')
    out=external_path(directory);out.mkdir(parents=True,exist_ok=True)
    with _arrays(mesh) as (p,c,phase,kind):
        if kind!='tetra':raise ValueError('Bundle requires linear Tet4')
        original=np.asarray(p);cells=np.asarray(c);phase=np.asarray(phase)
    points=original/LENGTH_UNITS[length_unit]
    if not np.all(np.isfinite(points)):raise ValueError('Nonfinite export coordinates after unit conversion')
    labels=sorted(map(int,np.unique(phase)))
    mapping={k:i+1 for i,k in enumerate(labels)};domains=np.searchsorted(labels,phase).astype(np.int32)+1
    names={mapping[k]:re.sub('[^A-Za-z0-9_]','_',str((region_names or {}).get(k,f'region_{k}'))) for k in labels}
    names={k:('domain_'+v if not v or not v[0].isalpha() else v)+f'_id{k}' for k,v in names.items()}
    facets=np.asarray(getattr(mesh,'facets',np.empty((0,3),np.int32)))
    markers=np.asarray(getattr(mesh,'facet_markers',np.empty(0,np.int32)))
    if facets.ndim!=2:facets=np.empty((0,3),np.int32);markers=np.empty(0,np.int32)
    if not len(facets) and hasattr(mesh,'hdf5'):
        import h5py
        with h5py.File(mesh.hdf5) as f:
            if 'mesh/facets' in f:facets=f['mesh/facets'][:];markers=f['mesh/facet_markers'][:]
    blocks=[('tetra',cells)]+([('triangle',facets)] if len(facets) else [])
    tags=[domains]+([markers] if len(facets) else [])
    checks=[]
    for fmt in formats:
        if fmt=='medit':
            from .medit_stream import write_and_verify_medit
            path=out/'mesh.mesh'
            result=write_and_verify_medit(path,points,cells,domains,facets,markers)
            checks.append({'format':fmt,'path':str(path),'nodes':len(points),'tetrahedra':len(cells),**result})
            continue
        if fmt=='vtu':
            from .vtu_stream import write_and_verify_vtu
            path=out/'mesh.vtu'
            result=write_and_verify_vtu(path,points,cells,domains,facets,markers)
            checks.append({'format':fmt,'path':str(path),'nodes':len(points),'tetrahedra':len(cells),**result})
            continue
        if fmt=='comsol':
            path=export_comsol(mesh,out/'mesh.mphtxt',region_names=region_names,selections=selections,length_unit=length_unit)
            back=read_comsol(path);bpoints=back['points'];bcells=back['elements']['tet']['cells'];btags=back['elements']['tet']['entities']
            if len(facets):
                np.testing.assert_array_equal(back['elements']['tri']['cells'],facets)
                np.testing.assert_array_equal(back['elements']['tri']['entities'],np.searchsorted(np.unique(markers),markers))
        elif fmt=='abaqus':
            path=out/'mesh.inp'
            if path.exists():raise FileExistsError(path)
            _abaqus(path,points,cells,domains,facets,markers,names)
            _verify_abaqus(path,points,cells,domains,names)
            checks.append({'format':fmt,'path':str(path),'nodes':len(points),'tetrahedra':len(cells),'domain_readback_passed':True,'readback':'independent bounded ASCII parser; all nodes/cells/material memberships'})
            continue
        else:
            suffix={'vtu':'.vtu','gmsh':'.msh','medit':'.mesh'}[fmt];path=out/('mesh'+suffix)
            if path.exists():raise FileExistsError(path)
            if fmt=='vtu':data={'domain_id':[domains]+([np.zeros(len(facets),np.int32)] if len(facets) else []),
                'boundary_id':[np.zeros(len(cells),np.int32)]+([markers] if len(facets) else [])}
            elif fmt=='gmsh':data={'gmsh:physical':tags,'gmsh:geometrical':tags}
            else:data={'medit:ref':tags}
            fields={name:np.array([k,3]) for k,name in names.items()}
            fields.update({f'boundary_{int(k)}':np.array([k,2]) for k in np.unique(markers)})
            obj=meshio.Mesh(points,blocks,cell_data=data,field_data=fields)
            meshio.write(path,obj,file_format={'vtu':'vtu','gmsh':'gmsh22','medit':'medit'}[fmt],**({'binary':True} if fmt=='gmsh' else {}))
            back=meshio.read(path);bpoints=back.points;bcells=back.get_cells_type('tetra')
            tagname={'vtu':'domain_id','gmsh':'gmsh:physical','medit':'medit:ref'}[fmt];btags=back.cell_data_dict[tagname]['tetra']
            if len(facets):
                np.testing.assert_array_equal(back.get_cells_type('triangle'),facets)
                boundary_tag='boundary_id' if fmt=='vtu' else tagname
                np.testing.assert_array_equal(back.cell_data_dict[boundary_tag]['triangle'],markers)
        np.testing.assert_allclose(bpoints,points,rtol=1e-12,atol=max(float(np.ptp(points,axis=0).max()),1e-20)*1e-14)
        np.testing.assert_array_equal(bcells,cells);np.testing.assert_array_equal(btags,domains)
        checks.append({'format':fmt,'path':str(path),'nodes':len(points),'tetrahedra':len(cells),'domain_readback_passed':True})
    report={'length_unit':length_unit,'scale_to_m':LENGTH_UNITS[length_unit],'region_to_domain':mapping,
        'domain_names':names,'selections':selections or {},'checks':checks,'passed':True,
        'comsol_application_tested':False,'abaqus_solver_tested':False,
        'interface_policy':'shared nodes plus per-material named element surfaces boundary_<marker>_domain_<domain>; no cohesive/contact law implied'}
    (out/'bundle.json').write_text(json.dumps(report,indent=2));return report
