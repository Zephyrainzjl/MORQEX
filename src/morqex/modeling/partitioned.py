"""Conforming slab meshing of a full shared-interface PLC, out of core.

Each common cut uses ONE constrained2D triangulation. TetGen-Y preserves that
cap on both sides. Assembly must verify identical interface faces, not merely
coincident node positions. This module does not coarsen the source labels.
Triangle is an optional runtime dependency with a separate upstream license.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def _edge_keys(edges,n):
    e=np.sort(edges,axis=1);return e[:,0].astype(np.int64)*n+e[:,1]


def cut_cap(points,faces,markers,pairs,plane,*,axis=2,quantization=1e9,minimum_angle=0.,maximum_steiner=200000):
    """PSLG cap labelled from oriented true interfaces, not image centroids."""
    import triangle
    if not 0<=minimum_angle<=30 or type(maximum_steiner) is not int or maximum_steiner<0:raise ValueError('Bounded planar quality/refinement budgets required')
    other=[k for k in range(3) if k!=axis];segments=[];normal=[];source_markers=[]
    for start in range(0,len(faces),50000):
        f=faces[start:start+50000];p=points[f];d=p[:,:,axis]-plane
        chosen=(d.min(1)<0)&(d.max(1)>0)
        for tri,sign,marker in zip(p[chosen],d[chosen],markers[start:start+50000][chosen]):
            cuts=[]
            for a,b in ((0,1),(1,2),(2,0)):
                if sign[a]*sign[b]<0:cuts.append(tri[a]+sign[a]/(sign[a]-sign[b])*(tri[b]-tri[a]))
            if len(cuts)!=2:raise ArithmeticError('Cut plane intersects an original vertex; choose a nondegenerate plane')
            segments.append(np.asarray(cuts)[:,other]);normal.append(np.cross(tri[1]-tri[0],tri[2]-tri[0])[other]);source_markers.append(int(marker))
    if not segments:raise ValueError('Cut does not intersect PLC')
    coordinates,inverse=np.unique(np.rint(np.asarray(segments).reshape(-1,2)*quantization).astype(np.int64),axis=0,return_inverse=True)
    coordinates=coordinates/quantization;edges=inverse.reshape(-1,2).astype(np.int32)
    if np.any(edges[:,0]==edges[:,1]):raise ArithmeticError('Quantization would erase a real cap segment')
    if len(np.unique(_edge_keys(edges,len(coordinates))))!=len(edges):raise ArithmeticError('Overlapping cap segments')
    cap=triangle.triangulate({'vertices':coordinates,'segments':edges,
        'segment_markers':np.arange(1,len(edges)+1,dtype=np.int32)[:,None]},
        'pYQ'+(f'q{minimum_angle:.8g}S{maximum_steiner}' if minimum_angle else ''))
    vertices=np.asarray(cap['vertices']);triangles=np.asarray(cap['triangles'],np.int32)
    constrained=np.asarray(cap['segments']);segment_ids=np.asarray(cap['segment_markers']).ravel()-1
    if len(vertices)<len(coordinates) or not np.array_equal(vertices[:len(coordinates)],coordinates) or np.any(constrained>=len(coordinates)):
        raise ArithmeticError('Triangle inserted/split a boundary point; the3D cut is not conforming')
    n=len(vertices);all_edges=triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    keys=_edge_keys(all_edges,n);sort=np.argsort(keys);values,first,counts=np.unique(keys[sort],return_index=True,return_counts=True)
    constraints=_edge_keys(constrained,n)
    internal=(counts==2)&~np.isin(values,constraints)
    a=sort[first[internal]]//3;b=sort[first[internal]+1]//3
    graph=sparse.coo_matrix((np.ones(len(a),np.int8),(a,b)),shape=(len(triangles),)*2).tocsr()
    count,component=connected_components(graph,directed=False)
    labels=np.full(count,-999,np.int64);centre=vertices[triangles].mean(1)
    normal=np.asarray(normal);source_markers=np.asarray(source_markers)
    loc=np.searchsorted(values,constraints)
    if np.any(loc>=len(values)) or np.any(values[loc]!=constraints):raise ArithmeticError('Cap constraint is not a triangle edge')
    for edge,segment_id,index in zip(constrained,segment_ids,loc):
        middle=vertices[edge].mean(0);pair=pairs[str(int(source_markers[segment_id]))]
        for owner in sort[first[index]:first[index]+counts[index]]//3:
            direction=float((centre[owner]-middle)@normal[segment_id])
            if abs(direction)<1e-24:raise ArithmeticError('Ambiguous side of a cap interface')
            label=int(pair[1] if direction>0 else pair[0]);group=component[owner]
            if label<0:raise ArithmeticError('Cap triangulation includes exterior space')
            if labels[group] not in (-999,label):raise ArithmeticError('Contradictory oriented material labels on one cap region')
            labels[group]=label
    if np.any(labels<0):raise ArithmeticError('Unlabelled cap compartment')
    t=vertices[triangles];u=t[:,1]-t[:,0];v=t[:,2]-t[:,0];area=u[:,0]*v[:,1]-u[:,1]*v[:,0]
    negative=area<0;triangles[negative]=triangles[negative][:,[0,2,1]];area=abs(area)/2
    seeds=[]
    for group in range(count):
        ids=np.flatnonzero(component==group);row=ids[np.argmax(area[ids])]
        seeds.append([*vertices[triangles[row]].mean(0),int(labels[group])])
    return {'points':vertices,'triangles':triangles,'phases':labels[component],
        'seeds':np.asarray(seeds),'plane':float(plane),'axis':axis,'other':other,'segments':edges,
        'area':float(area.sum()),'compartments':count,'boundary_points_unchanged':True,
        'interior_quality_points':len(vertices)-len(coordinates),'minimum_angle_requested':float(minimum_angle)}


def _clip_polygon(triangle,low,high,axis):
    result=list(triangle)
    for plane,side in ((low,1),(high,-1)):
        output=[]
        for a,b in zip(result,result[1:]+result[:1]):
            inside_a=side*(a[axis]-plane)>=0;inside_b=side*(b[axis]-plane)>=0
            if inside_a:output.append(a)
            if inside_a!=inside_b:
                p=a+(plane-a[axis])/(b[axis]-a[axis])*(b-a);p[axis]=plane;output.append(p)
        result=output
        if not result:break
    return result


def slab_plc(source,low,high,*,lower_cap=None,upper_cap=None,axis=2,length_scale=1.,origin=(0.,0.,0.),quantization=1e9):
    """Clip real source triangles, add the common caps, retain material pairs."""
    from .tetgen import SurfacePLC
    from .adaptive import _region_volumes
    points=(source.points-np.asarray(origin))/length_scale;faces=np.asarray(source.facets)
    # Physical end walls can differ from an analytically computed slab end by
    # a few ulps after anisotropic SI normalization. Do not clip away an entire
    # coplanar wall because one representation is infinitesimally larger.
    for value in (low,high):
        near=np.abs(points[:,axis]-value)<=2/quantization
        points[near,axis]=value
    triangles=[];marks=[];pairs=dict(source.metadata['interface_pairs'])
    for start in range(0,len(faces),50000):
        p=points[faces[start:start+50000]];lo=p[:,:,axis].min(1);hi=p[:,:,axis].max(1)
        selected=(hi>=low)&(lo<=high);inside=selected&(lo>=low)&(hi<=high)
        if inside.any():triangles.append(p[inside]);marks.append(source.markers[start:start+50000][inside])
        for triangle,marker in zip(p[selected&~inside],source.markers[start:start+50000][selected&~inside]):
            poly=_clip_polygon(triangle,low,high,axis)
            if len(poly)<3:continue
            patch=np.array([[poly[0],poly[k],poly[k+1]] for k in range(1,len(poly)-1)])
            triangles.append(patch);marks.append(np.full(len(patch),marker,np.int32))
    seeds={k:[p.tolist() for p in (np.asarray(v)-origin)/length_scale if low+1e-6<p[axis]<high-1e-6] for k,v in source.seeds.items()}
    for cap,upper in ((lower_cap,False),(upper_cap,True)):
        if cap is None:continue
        xyz=np.empty((len(cap['points']),3));xyz[:,axis]=cap['plane'];xyz[:,cap['other']]=cap['points']
        t=cap['triangles'].copy()
        # CCW in the other axes gives +axis except when axis==1. The pair
        # [-1,phase] points INTO the slab, hence upper caps point downward.
        if upper!=(axis==1):t=t[:,::-1]
        marker_array=np.empty(len(t),np.int32)
        for label in np.unique(cap['phases']):
            pair=[-1,int(label)];matches=[int(k) for k,v in pairs.items() if v==pair]
            marker=matches[0] if matches else max(map(int,pairs))+1;pairs[str(marker)]=pair
            marker_array[cap['phases']==label]=marker
        triangles.append(xyz[t]);marks.append(marker_array)
        for seed in cap['seeds']:
            p=np.empty(3);p[cap['other']]=seed[:2];p[axis]=cap['plane']+(-1e-5 if upper else 1e-5)
            seeds.setdefault(int(seed[2]),[]).append(p.tolist())
    coordinates=np.concatenate(triangles).reshape(-1,3);markers=np.concatenate(marks)
    q,inverse=np.unique(np.rint(coordinates*quantization).astype(np.int64),axis=0,return_inverse=True)
    p=q/quantization;f=inverse.reshape(-1,3).astype(np.int32)
    # Preserve exact outer dimensions and exact slab planes after integer-key
    # welding; only values within the declared grid quantization are snapped.
    for direction in range(3):
        for value in (points[:,direction].min(),points[:,direction].max()):
            p[np.abs(p[:,direction]-value)<=2/quantization,direction]=value
    for value in (low,high):p[np.abs(p[:,axis]-value)<=2/quantization,axis]=value
    seeds={k:np.asarray(v)*length_scale+origin for k,v in seeds.items() if len(v)}
    volumes=_region_volumes(p,f,markers,pairs,seeds)
    # The front end explicitly preserves a rectangular outer box. A missing
    # end wall otherwise produces a positive but open divergence integral.
    if source.metadata.get('box_planes_preserved'):
        transverse=[k for k in range(3) if k!=axis]
        expected=(high-low)*np.prod(np.ptp(points,axis=0)[transverse])
        if abs(sum(volumes.values())/expected-1)>1e-8:raise ArithmeticError('Clipped slab does not close the complete rectangular domain')
    return SurfacePLC(p*length_scale+origin,f,seeds,markers,region_volumes={k:v*length_scale**3 for k,v in volumes.items()},
        metadata={'interface_pairs':pairs,'surface_method':'exact clipped shared surface with common constrained cap',
            'region_names':source.metadata.get('region_names',{}),'partition_axis_xyz':axis,'slab_bounds_grid_units':[low,high],
            'length_scale_m':length_scale,'quantization_in_grid_units':1/quantization,
            'origin_xyz_m':list(origin),
            'cap_contract':'one PSLG triangulation, material sides from oriented interfaces; no centroid-image region assignment'})


def assemble_slabs(paths,path,*,length_scale=1.,quantization=1e9):
    """Stream conforming TetGen artifacts into ONE mesh, checking every cap.

    Input slabs must be ordered and adjacent. Only the preceding cap has a
    Python coordinate map. Interior partition faces are omitted from physical
    boundary entities. No material is reassigned by position or interpolation.
    """
    import h5py,json
    from pathlib import Path
    from .voxel import external_path
    from .meshing import MeshArtifact
    paths=list(paths)
    if not paths:raise ValueError('At least one complete slab required')
    if not np.isfinite(length_scale) or length_scale<=0 or not np.isfinite(quantization) or quantization<=0:
        raise ValueError('Positive finite length scale and quantization required')
    path=external_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    pending=path.with_suffix('.pending')
    if pending.exists():raise FileExistsError('Remove or archive interrupted assembly explicitly: '+str(pending))
    previous=None;previous_faces=None;previous_high=None;phase_volumes={};minimum=np.inf;pairs={};pair_ids={};checks=[];quality_reports=[];quality_status=[]
    def append(dataset,a):
        n=len(dataset);dataset.resize(n+len(a),axis=0);dataset[n:]=a
    def canonical(a):
        a=np.asarray(a,dtype=np.int64)
        return a[np.lexsort(tuple(a[:,i] for i in range(a.shape[1]-1,-1,-1)))]
    with h5py.File(pending,'x') as out:
        def dataset(name,width,dtype):
            shape=(0,width) if width else (0,);maximum=(None,width) if width else (None,)
            return out.create_dataset('mesh/'+name,shape=shape,maxshape=maximum,dtype=dtype,
                chunks=(50000,width) if width else (50000,),compression='gzip',compression_opts=1,shuffle=True)
        nodes=dataset('points_m',3,'f8');cells=dataset('tetrahedra',4,'u4');phases=dataset('phase',0,'i4')
        facets=dataset('facets',3,'u4');markers=dataset('facet_markers',0,'i4')
        names={}
        for index,part in enumerate(paths):
            with h5py.File(part) as src:
                if not src.attrs.get('complete',False):raise ValueError('Incomplete slab')
                report=json.loads(src.attrs['report']);geo=report['geometry'];axis=geo['partition_axis_xyz']
                if report.get('quality'):quality_reports.append(report['quality'])
                quality_status.append(report.get('quality_limits_satisfied'))
                low,high=geo['slab_bounds_grid_units'];p=src['mesh/points_m'][:];f=src['mesh/facets'][:];m=src['mesh/facet_markers'][:]
                if not np.all(np.isfinite(p)):raise ArithmeticError('Nonfinite slab coordinates')
                origin=np.asarray(geo.get('origin_xyz_m',[0.,0.,0.]))
                grid=(p-origin)/length_scale;other=[k for k in range(3) if k!=axis]
                on_bottom=np.isclose(grid[:,axis],low,rtol=0,atol=2/quantization)
                on_top=np.isclose(grid[:,axis],high,rtol=0,atol=2/quantization)
                exterior=np.array([geo['interface_pairs'][str(int(k))][0]==-1 for k in m])
                bottom_face=on_bottom[f].all(1)&exterior;top_face=on_top[f].all(1)&exterior
                bottom=np.zeros(len(p),bool);bottom[f[bottom_face].ravel()]=True
                top=np.zeros(len(p),bool);top[f[top_face].ravel()]=True
                ids=np.empty(len(p),np.uint32);fresh=~bottom if previous is not None else np.ones(len(p),bool)
                if len(nodes)+fresh.sum()>=2**32:raise ValueError('Mesh exceeds UInt32 global node budget')
                ids[fresh]=np.arange(len(nodes),len(nodes)+fresh.sum(),dtype=np.uint32)
                if previous is not None:
                    if abs(low-previous_high)>1/quantization:raise ArithmeticError('Slabs are not adjacent')
                    keys=np.rint(grid[bottom][:,other]*quantization).astype(np.int64)
                    try:shared=np.array([previous[tuple(k)] for k in keys],np.uint32)
                    except KeyError as e:raise ArithmeticError('Adjacent cap node mismatch') from e
                    if len(shared)!=len(previous) or len(np.unique(shared))!=len(shared):raise ArithmeticError('Cap node sets differ')
                    ids[bottom]=shared
                    # Include physical phase, not only face-node coincidence.
                    local_pairs=geo['interface_pairs']
                    labels=np.array([max(local_pairs[str(int(k))]) for k in m[bottom_face]])
                    actual=canonical(np.column_stack((np.sort(ids[f[bottom_face]],axis=1),labels)))
                    if not np.array_equal(actual,previous_faces):raise ArithmeticError('Adjacent cap triangle/material mismatch')
                    checks.append({'left_part':index-1,'right_part':index,'shared_nodes':len(shared),'identical_material_faces':len(actual)})
                append(nodes,p[fresh])
                local_pairs=geo['interface_pairs'];names.update(geo.get('region_names',{}))
                physical=np.ones(len(f),bool)
                if index:physical&=~bottom_face
                if index<len(paths)-1:physical&=~top_face
                mm=np.empty(int(physical.sum()),np.int32)
                for label in np.unique(m[physical]):
                    pair=tuple(local_pairs[str(int(label))])
                    if pair not in pair_ids:
                        k=len(pair_ids)+1;pair_ids[pair]=k;pairs[str(k)]=list(pair)
                    mm[m[physical]==label]=pair_ids[pair]
                append(facets,ids[f[physical]]);append(markers,mm)
                for start in range(0,len(src['mesh/tetrahedra']),50000):
                    t=src['mesh/tetrahedra'][start:start+50000];label=src['mesh/phase'][start:start+50000]
                    if label.dtype.kind not in 'iu' or np.any(label<0) or int(label.max())>np.iinfo(np.int32).max:
                        raise ValueError('Slab region IDs exceed the signed32-bit phase-label contract')
                    v=np.linalg.det(p[t[:,1:]]-p[t[:,:1]])/6
                    if np.any(~np.isfinite(v)) or np.any(v<=0):raise ArithmeticError('Nonpositive or nonfinite assembled tetrahedron')
                    minimum=min(minimum,float(v.min()))
                    for k in np.unique(label):phase_volumes[int(k)]=phase_volumes.get(int(k),0.)+float(v[label==k].sum())
                    append(cells,ids[t]);append(phases,label)
                keys=np.rint(grid[top][:,other]*quantization).astype(np.int64)
                previous={tuple(k):int(i) for k,i in zip(keys,ids[top])};previous_high=high
                labels=np.array([max(local_pairs[str(int(k))]) for k in m[top_face]])
                previous_faces=canonical(np.column_stack((np.sort(ids[f[top_face]],axis=1),labels)))
                out.attrs['parts_completed']=index+1;out.flush()
                print(json.dumps({'assembled_parts':index+1,'nodes':len(nodes),'tetrahedra':len(cells)}),flush=True)
        report={'mesher':'conforming partitioned source-integrated TetGen1.6.0',
            'geometry':{'interface_pairs':pairs,'region_names':names,'partition_contract':'identical material-labelled cap triangles and shared node IDs'},
            'nodes':len(nodes),'tetrahedra':len(cells),'phase_volumes_m3':phase_volumes,
            'minimum_signed_volume_m3':minimum,'cap_checks':checks,
            'quality_limits_satisfied':False if False in quality_status else True if all(v is True for v in quality_status) else None,
            'scope':'geometric conformity and positive-volume checks; no FE convergence claim'}
        if quality_reports:
            report['quality']={'radius_edge_max':max(r['radius_edge_max'] for r in quality_reports),
                'minimum_dihedral_degrees':min(r['minimum_dihedral_degrees'] for r in quality_reports),
                'minimum_volume_m3':minimum,'total_volume_m3':sum(phase_volumes.values()),
                'scope':'exact max/min aggregation over parts; pooled quantiles not computed'}
        out.attrs.update(complete=True,report=json.dumps(report),coordinate_units='m',coordinate_order='XYZ',
            selected_regions=sorted(phase_volumes),node_count=len(nodes),tetrahedra=len(cells))
    pending.rename(path)
    artifact=MeshArtifact.load(path);fn=path.name
    artifact.xdmf.write_text(f'''<Xdmf Version="3.0"><Domain><Grid Name="Complete conforming TetGen mesh">
<Topology TopologyType="Tetrahedron" NumberOfElements="{artifact.tetrahedra}"><DataItem Dimensions="{artifact.tetrahedra} 4" NumberType="UInt" Precision="4" Format="HDF">{fn}:/mesh/tetrahedra</DataItem></Topology>
<Geometry GeometryType="XYZ"><DataItem Dimensions="{artifact.nodes} 3" NumberType="Float" Precision="8" Format="HDF">{fn}:/mesh/points_m</DataItem></Geometry>
<Attribute Name="region" Center="Cell"><DataItem Dimensions="{artifact.tetrahedra}" NumberType="Int" Precision="4" Format="HDF">{fn}:/mesh/phase</DataItem></Attribute></Grid></Domain></Xdmf>''')
    return artifact


def artifact_cap(path,plane,*,length_scale=1.,axis=2,quantization=1e9):
    """Extract actual cap nodes/faces/materials, including recovery Steiner nodes."""
    import h5py,json
    with h5py.File(path) as f:
        geometry=json.loads(f.attrs['report'])['geometry']
        p=(f['mesh/points_m'][:]-geometry.get('origin_xyz_m',[0.,0.,0.]))/length_scale
        facets=f['mesh/facets'][:];markers=f['mesh/facet_markers'][:];pairs=geometry['interface_pairs']
    chosen=np.isclose(p[:,axis],plane,rtol=0,atol=2/quantization)[facets].all(1)
    facets=facets[chosen];markers=markers[chosen];used,inverse=np.unique(facets,return_inverse=True)
    other=[k for k in range(3) if k!=axis];labels=[]
    for marker in markers:
        pair=pairs[str(int(marker))]
        if pair[0]!=-1:raise ArithmeticError('Partition cap coincides with a real material interface')
        labels.append(pair[1])
    return {'points':p[used][:,other],'triangles':inverse.reshape(-1,3),'phases':np.array(labels,np.int64)}


def reconcile_cap(original,left,right,*,quantization=1e9):
    """Common PSLG cap containing recovery nodes from BOTH native meshes.

    Physical constraints and one interior seed per original2D compartment
    determine attributes. This is constrained planar material triangulation,
    not voxel-centroid classification. Returns None when already conforming.
    """
    import triangle
    def canonical(a):return a[np.lexsort(tuple(a[:,i] for i in range(a.shape[1]-1,-1,-1)))]
    q,inv=np.unique(np.rint(np.concatenate((left['points'],right['points']))*quantization).astype(np.int64),axis=0,return_inverse=True)
    n=len(left['points']);a=inv[:n][left['triangles']];b=inv[n:][right['triangles']]
    a=canonical(np.column_stack((np.sort(a,axis=1),left['phases'])))
    b=canonical(np.column_stack((np.sort(b,axis=1),right['phases'])))
    if np.array_equal(a,b):return None
    coords,inv=np.unique(np.rint(np.concatenate((original['points'],q/quantization))*quantization).astype(np.int64),axis=0,return_inverse=True)
    edges=inv[:len(original['points'])][original['segments']]
    seeds=np.asarray(original['seeds']);regions=np.column_stack((seeds[:,:2],seeds[:,2]+1,np.full(len(seeds),-1.)))
    result=triangle.triangulate({'vertices':coords/quantization,'segments':edges,'regions':regions},'pYQA')
    points=np.asarray(result['vertices']);tri=np.asarray(result['triangles'],np.int32)
    values=np.asarray(result['triangle_attributes']).ravel();labels=np.rint(values).astype(np.int64)-1
    if np.any(abs(values-labels-1)>1e-10) or not set(labels)<=set(map(int,original['phases'])) or np.any(labels<0):raise ArithmeticError('Unseeded/invalid reconciled cap material')
    def areas(p,t,phase):
        x=p[t];u=x[:,1]-x[:,0];v=x[:,2]-x[:,0];signed=u[:,0]*v[:,1]-u[:,1]*v[:,0]
        return {int(k):float(abs(signed[phase==k]).sum()/2) for k in np.unique(phase)},signed
    reference,_=areas(original['points'],original['triangles'],original['phases']);measured,signed=areas(points,tri,labels)
    if set(reference)!=set(measured) or max(abs(measured[k]/v-1) for k,v in reference.items())>1e-8:raise ArithmeticError('Reconciled cap changed physical material areas')
    tri[signed<0]=tri[signed<0][:,[0,2,1]]
    return dict(original)|{'points':points,'triangles':tri,'phases':labels,'segments':np.asarray(result['segments'],np.int32),
        'reconciled_from_both_native_caps':True,'native_left_points':len(left['points']),'native_right_points':len(right['points'])}
