"""Label native geometric compartments from oriented shared interfaces.

Unlike a nearest-voxel centroid assignment, this uses TetGen's own connected
region IDs and the material on each side of every constrained mesh face.
Contradictory or missing evidence rejects the entire mesh.
"""
import time
import numpy as np


def interface_tetgen(plc,options,workspace=None):
    from .tetgen_backend import execute,identity
    from .tetgen import TetGenMesh,tetra_quality
    from .exchange import _facet_owners
    if np.asarray(plc.facets).ndim!=2 or np.asarray(plc.facets).shape[1]!=3:raise ValueError('Oriented triangle interfaces required')
    if not options.preserve_surface or options.merge_coplanar:
        raise ValueError('Interface labelling currently requires preserve_surface=True and merge_coplanar=False')
    if options.maximum_volume is not None or options.region_maximum_volumes:raise ValueError('Interface labelling currently uses native q refinement without volume constraints')
    pairs=plc.metadata.get('interface_pairs')
    if not pairs:raise ValueError('Explicit interface material pairs required')
    start=time.perf_counter();low=plc.points.min(0);scale=float(np.ptp(plc.points,axis=0).max())
    source=(plc.points-low)/scale
    # TetGen's default PLC output already contains ALL constrained subfaces,
    # including internal material interfaces. -f additionally allocates every
    # interior volume face, which this algorithm would immediately discard.
    flags=f'p{options.facet_separation_angle:.8f}q{options.radius_edge_ratio:.8g}/{options.minimum_dihedral:.8g}AzQYMS{options.maximum_steiner}T{options.coplanarity_tolerance:.8g}'
    if options.consistency_check:flags+='C'
    p,t,a,faces,markers=execute(source,plc.facets,np.arange(1,len(plc.facets)+1,dtype=np.int32),
        np.empty((0,5)),(plc.holes-low)/scale,flags,workspace=workspace,maximum_cells=options.maximum_cells)
    if a.shape[1]!=1 or not np.isfinite(a).all():raise ArithmeticError('Native automatic region attributes missing')
    geometric=np.rint(a[:,0]).astype(np.int64)
    if np.any(abs(geometric-a[:,0])>1e-8) or np.any(geometric<1):raise ArithmeticError('Invalid native geometric region IDs')
    np.testing.assert_allclose(p[:len(source)],source,rtol=0,atol=1e-13)
    marked=markers>0;boundary=faces[marked];source_ids=markers[marked]-1
    if np.any(source_ids>=len(plc.facets)):raise ArithmeticError('Unknown native constrained-face marker')
    expected=np.sort(plc.facets[source_ids],axis=1)
    mismatch=np.flatnonzero(np.any(np.sort(boundary,axis=1)!=expected,axis=1))
    # TetGen1.6 may retain mandatory facet-recovery Steiner points even under
    # -Y. Accept only exact planar SUBDIVISION of the original triangle, never
    # an arbitrary replacement. The slab assembler independently requires
    # identical complete cap triangulations on adjacent sides.
    recovered=np.bincount(source_ids,minlength=len(plc.facets))
    if np.any(recovered==0):raise ArithmeticError('Native recovery lost an original constrained facet')
    if len(mismatch):
        groups=np.unique(source_ids[mismatch]);rows=np.flatnonzero(np.isin(source_ids,groups))
        src=source[plc.facets[source_ids[rows]]];actual=p[boundary[rows]]
        u=src[:,1]-src[:,0];v=src[:,2]-src[:,0];n=np.cross(u,v);norm=np.linalg.norm(n,axis=1)
        w=actual-src[:,:1];edge=np.maximum(np.linalg.norm(u,axis=1),np.linalg.norm(v,axis=1))
        plane=np.abs(np.einsum('nki,ni->nk',w,n)).max(1)/norm
        if np.any(plane>edge*1e-10):raise ArithmeticError('Facet recovery moved a triangle out of its original plane')
        den=norm**2
        b=np.einsum('nki,ni->nk',np.cross(w,v[:,None,:]),n)/den[:,None]
        c=np.einsum('nki,ni->nk',np.cross(u[:,None,:],w),n)/den[:,None]
        if np.any(b< -1e-8) or np.any(c< -1e-8) or np.any(b+c>1+1e-8):raise ArithmeticError('Native subfacet leaves original triangle')
        area=np.linalg.norm(np.cross(actual[:,1]-actual[:,0],actual[:,2]-actual[:,0]),axis=1)
        accumulated=np.bincount(source_ids[rows],weights=area,minlength=len(plc.facets))
        original=source[plc.facets[groups]];target=np.linalg.norm(np.cross(original[:,1]-original[:,0],original[:,2]-original[:,0]),axis=1)
        if np.any(abs(accumulated[groups]/target-1)>1e-8):raise ArithmeticError('Recovered subfacets do not cover their original physical interface')
    owners,_=_facet_owners(t,boundary,len(p))
    material=np.full(int(geometric.max())+1,-999,np.int64)
    table=np.array([pairs[str(int(k))] for k in plc.markers],np.int64)
    for begin in range(0,len(boundary),50000):
        src=source_ids[begin:begin+50000];tri=source[plc.facets[src]].astype(np.longdouble)
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        actual_ids=boundary[begin:begin+50000];actual=p[actual_ids].astype(np.longdouble)
        actual_normal=np.cross(actual[:,1]-actual[:,0],actual[:,2]-actual[:,0])
        alignment=np.sign(np.sum(actual_normal*normal,axis=1))
        if np.any(alignment==0):raise ArithmeticError('Recovered face orientation is ambiguous')
        pair=table[src]
        for side in range(2):
            cell=owners[begin:begin+50000,side];valid=cell>=0
            c=cell[valid];vertices=t[c]
            opposite=~np.any(vertices[:,:,None]==actual_ids[valid][:,None,:],axis=2)
            if np.any(opposite.sum(1)!=1):raise ArithmeticError('Constrained face owner has no unique opposite node')
            point=p[vertices[opposite]].astype(np.longdouble)
            # Opposite-vertex orientation in extended precision avoids losing
            # the sign when a very thin Tet4 centroid rounds onto its face.
            dot=np.sum((point-actual[valid,0])*actual_normal[valid],axis=1)*alignment[valid]
            if np.any(dot==0):raise ArithmeticError('Ambiguous side of a constrained tetrahedral face')
            label=pair[valid][np.arange(len(c)),(dot>0).astype(int)]
            if np.any(label<0):raise ArithmeticError('Native volume occupies the exterior side of a PLC')
            region=geometric[c]
            # Every geometric compartment must receive ONE material. This is
            # independently constrained by all its incident physical faces.
            vote=np.unique(np.column_stack((region,label)),axis=0)
            if len(np.unique(vote[:,0]))!=len(vote):raise ArithmeticError('Conflicting materials in one native connected region')
            old=material[vote[:,0]]
            if np.any((old!=-999)&(old!=vote[:,1])):
                conflict=vote[(old!=-999)&(old!=vote[:,1])]
                raise ArithmeticError('Conflicting oriented interface evidence across blocks: '+str(conflict[:5].tolist()))
            material[vote[:,0]]=vote[:,1]
    phases=material[geometric]
    if np.any(phases<0) or set(map(int,np.unique(phases)))!=set(plc.seeds):raise ArithmeticError('Unlabelled or missing physical compartment')
    # Keep orientation checks bounded for whole-electrode meshes. A single
    # (N_tet,3,3) FP64 determinant buffer otherwise needs gigabytes by itself.
    for begin in range(0,len(t),50000):
        block=t[begin:begin+50000]
        reverse=np.linalg.det(p[block[:,1:]]-p[block[:,:1]])<0
        block[reverse]=block[reverse][:,[1,0,2,3]]
    p=p*scale+low;quality,volumes=tetra_quality(p,t)
    errors={str(k):abs(float(volumes[phases==k].sum())/v-1) for k,v in plc.region_volumes.items()}
    if errors and max(errors.values())>1e-8:raise ArithmeticError(f'Exact interface-labelled PLC volume error: {errors}')
    report={'mesher':'source-integrated TetGen1.6.0','source_identity':identity(),'switches':flags,
        'input_points':len(source),'input_facets':len(plc.facets),'nodes':len(p),'tetrahedra':len(t),
        'region_volume_relative_errors':errors,'quality':quality,'geometry':plc.metadata,'elapsed_seconds':time.perf_counter()-start,
        'region_volumes_m3':{str(int(k)):float(volumes[phases==k].sum()) for k in np.unique(phases)},
        'region_labelling':'native connected compartments + all oriented constrained-face sides; no nearest-voxel classification',
        'native_geometric_compartments':len(np.unique(geometric)),
        'verified_planar_subdivision_faces':int(len(mismatch)),
        'quality_limits_satisfied':bool(quality['radius_edge_max']<=options.radius_edge_ratio*(1+1e-6) and quality['minimum_dihedral_degrees']+1e-6>=options.minimum_dihedral)}
    if options.require_quality and not report['quality_limits_satisfied']:raise ArithmeticError('Interface mesh misses requested shape quality')
    return TetGenMesh(p,t,phases,facets=boundary,facet_markers=plc.markers[source_ids],report=report)
