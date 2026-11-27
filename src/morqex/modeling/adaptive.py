"""GEMsim-style residual-weighted ACVD surface remeshing, then TetGen.

This is geometric adaptation, not a PDE-residual error estimator. Multiphase
patches pin common junctions; independent phase contours are never overlapped.
"""
from dataclasses import dataclass,replace
import json,time,os
import numpy as np
from scipy import ndimage,sparse
from scipy.sparse.csgraph import connected_components,breadth_first_order
from scipy.spatial import cKDTree
from .tetgen import SurfacePLC,TetGenOptions,tetgen_mesh,voxel_plc,marching_cubes_plc,_component_seeds


@dataclass(frozen=True)
class AdaptiveMeshOptions:
    vertex_fraction: float = .12
    levels: int = 4
    sigma: float = 1.
    weight_power: float = 2.
    weight_floor: float = .2
    weight_strength: float = 3.
    maximum_weight_ratio: float = 16.
    minimum_component_vertices: int = 128
    iterations: int = 30
    topology_retries: int = 3
    maximum_facets: int = 4000000
    maximum_phase_volume_error: float = .03
    surface_method: str = 'auto'
    project_surface: bool = False
    retry_without_facet_merging: bool = True
    phase_order: tuple = ()
    # Optional user multiplier sampled on source surface vertices.
    # It changes target density, not physical material parameters.
    density: object = None

    def __post_init__(self):
        if not 0<self.vertex_fraction<=1:raise ValueError('vertex_fraction must be in(0,1]')
        if type(self.levels) is not int or not 1<=self.levels<=6:raise ValueError('levels must be1..6')
        if min(self.sigma,self.weight_power,self.weight_floor)<=0 or not np.isfinite([self.sigma,self.weight_power,self.weight_floor,self.weight_strength]).all() or self.weight_strength<0:
            raise ValueError('Positive finite residual parameters and nonnegative strength required')
        if not np.isfinite(self.maximum_weight_ratio) or self.maximum_weight_ratio<1:raise ValueError('maximum_weight_ratio must be finite and>=1')
        if type(self.minimum_component_vertices) is not int or self.minimum_component_vertices<4:raise ValueError('minimum_component_vertices must be an integer>=4')
        if type(self.maximum_facets) is not int or self.maximum_facets<4:raise ValueError('maximum_facets must be an integer>=4')
        if type(self.project_surface) is not bool:raise ValueError('project_surface must be boolean')
        if type(self.retry_without_facet_merging) is not bool:raise ValueError('retry_without_facet_merging must be boolean')
        if not 0<self.maximum_phase_volume_error<1:raise ValueError('Positive relative phase-volume tolerance required')
        if self.surface_method not in ('auto','marching_cubes','shared_interfaces','nested_contours'):raise ValueError('Unknown adaptive surface route')
        if type(self.iterations) is not int or self.iterations<1 or type(self.topology_retries) is not int or self.topology_retries<0:raise ValueError('Integer iteration budgets required')
        if self.density is not None and not callable(self.density):raise ValueError('density must be a callback(points_xyz_m)')

    @classmethod
    def precision(cls,name,**overrides):
        choices={'coarse':dict(vertex_fraction=.06,maximum_phase_volume_error=.06),
                 'balanced':dict(vertex_fraction=.12,maximum_phase_volume_error=.03),
                 'fine':dict(vertex_fraction=.25,maximum_phase_volume_error=.015)}
        if name not in choices:raise ValueError('Precision is coarse, balanced or fine')
        return cls(**(choices[name]|overrides))


def multiresolution_weights(model,points,options):
    """Port of TriMesh.volres_weight: phase median residuals across scales,
    power/scale weighting, followed by a Gaussian multiresolution sum.

    Stream levels instead of retaining many256³ float64 volumes. Unlike the
    legacy modulo lookup, sample in physical ZYX coordinates with explicit
    nearest exterior treatment. Positive floor avoids zero-area clusters.
    """
    from skimage.transform import resize
    labels=model.regions;shape=np.array(labels.shape);aggregate=np.zeros(labels.shape,np.float32)
    for level in range(options.levels):
        small=np.maximum(2,np.rint(shape/(2**level)).astype(int));residual=np.zeros(tuple(small),np.float32)
        for k in np.unique(labels):
            mask=resize(labels==k,tuple(small),order=0,preserve_range=True,anti_aliasing=False)
            residual+=np.not_equal(mask,ndimage.median_filter(mask,size=3,mode='nearest')).astype(np.float32)*(2 if k==0 else 1)
        expanded=resize(residual,tuple(shape),order=1,preserve_range=True,anti_aliasing=False)
        aggregate+=(expanded/(2**level))**options.weight_power
    weight=np.zeros(labels.shape,np.float32)
    for level in range(options.levels):
        small=np.maximum(2,np.rint(shape/(2**level)).astype(int))
        reduced=resize(aggregate,tuple(small),order=1,preserve_range=True,anti_aliasing=level>0)
        smoothed=ndimage.gaussian_filter(reduced,options.sigma,mode='nearest')
        weight+=resize(smoothed,tuple(shape),order=1,preserve_range=True,anti_aliasing=False)
    weight/=sum((1/2**i)**options.weight_power for i in range(options.levels))
    coords=((points[:,::-1]-model.origin)/model.spacing-.5).T
    values=ndimage.map_coordinates(weight,coords,order=1,mode='nearest')
    reference=float(np.quantile(values,.95))
    values=options.weight_floor+options.weight_strength*values/max(reference,1e-12)
    if options.density is not None:
        multiplier=np.asarray(options.density(points),dtype=float)
        if multiplier.shape!=(len(points),) or not np.isfinite(multiplier).all() or np.any(multiplier<=0):raise ValueError('density must return one finite positive multiplier per source vertex')
        values*=multiplier
    # Bound *numerical mesh density*, not any physical state variable. This
    # avoids near-zero surface edges at isolated residual outliers.
    values=np.minimum(values,float(values.min())*options.maximum_weight_ratio)
    return np.ascontiguousarray(values,dtype=np.float64)


def _topology(points,faces):
    edges=np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    base=len(points);keys,counts=np.unique(edges[:,0].astype(np.int64)*base+edges[:,1],return_counts=True)
    unique=np.column_stack((keys//base,keys%base))
    used=np.unique(faces);remap=np.full(len(points),-1);remap[used]=np.arange(len(used))
    graph=sparse.coo_matrix((np.ones(len(unique)),(remap[unique[:,0]],remap[unique[:,1]])),shape=(len(used),)*2).tocsr()
    components=connected_components(graph,directed=False,return_labels=False)
    return {'vertices':len(used),'faces':len(faces),'edges':len(unique),'components':int(components),
        'euler':int(len(used)-len(unique)+len(faces)),'boundary_edges':int(np.count_nonzero(counts==1)),
        'nonmanifold_edges':int(np.count_nonzero(counts>2))}


def _same_topology(a,b):return all(a[k]==b[k] for k in ('components','euler','boundary_edges','nonmanifold_edges'))


def _project_to_triangles(query,points,faces):
    """Project to actual source triangles, not to a nearest vertex cloud.

    A centroid KD tree proposes16 triangles; every returned point lies on one
    real source triangle. This is not a certified global Hausdorff estimator.
    """
    triangles=points[faces];tree=cKDTree(triangles.mean(axis=1));output=np.empty_like(query)
    for start in range(0,len(query),2048):
        q=query[start:start+2048];idx=tree.query(q,k=min(16,len(faces)))[1]
        idx=np.asarray(idx).reshape(len(q),-1);p=triangles[idx]
        a,b,c=p[:,:,0],p[:,:,1],p[:,:,2];v=b-a;w=c-a;u=q[:,None,:]-a
        vv=np.sum(v*v,axis=2);ww=np.sum(w*w,axis=2);vw=np.sum(v*w,axis=2)
        uv=np.sum(u*v,axis=2);uw=np.sum(u*w,axis=2);den=vv*ww-vw**2
        s=(uv*ww-uw*vw)/np.maximum(den,1e-300);t=(uw*vv-uv*vw)/np.maximum(den,1e-300)
        projected=a+s[:,:,None]*v+t[:,:,None]*w
        valid=(s>=0)&(t>=0)&(s+t<=1)
        candidates=[projected];distance=[np.where(valid,np.sum((projected-q[:,None,:])**2,axis=2),np.inf)]
        for x,y in ((a,b),(b,c),(c,a)):
            edge=y-x;amount=np.clip(np.sum((q[:,None,:]-x)*edge,axis=2)/np.maximum(np.sum(edge*edge,axis=2),1e-300),0,1)
            point=x+amount[:,:,None]*edge
            candidates.append(point);distance.append(np.sum((point-q[:,None,:])**2,axis=2))
        d=np.stack(distance,axis=2);best=np.argmin(d.reshape(len(q),-1),axis=1)
        candidate=np.stack(candidates,axis=2).reshape(len(q),-1,3)
        output[start:start+len(q)]=candidate[np.arange(len(q)),best]
    return output


def _orient_surface(points,faces,expected):
    """Orient a remeshed complex consistently, preserving source normal sense.

    Undirected closure/Euler checks alone do not ensure the directed surface
    chain is closed. A reversed tiny quotient face corrupts enclosed volumes.
    """
    edges=faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2);base=len(points)
    keys=np.minimum(edges[:,0],edges[:,1]).astype(np.int64)*base+np.maximum(edges[:,0],edges[:,1])
    sort=np.argsort(keys);sorted_keys=keys[sort]
    _,first,counts=np.unique(sorted_keys,return_index=True,return_counts=True)
    paired=counts==2;a=sort[first[paired]];b=sort[first[paired]+1]
    direction=np.where(edges[:,0]<edges[:,1],1,-1)
    if np.all(direction[a]+direction[b]==0):return faces
    fa=a//3;fb=b//3;relation=(-direction[a]*direction[b]).astype(np.int8)
    n=len(faces)
    graph=sparse.coo_matrix((np.r_[relation,relation],(np.r_[fa,fb],np.r_[fb,fa])),shape=(n,n)).tocsr()
    _,components=connected_components(graph,directed=False)
    _,roots=np.unique(components,return_index=True)
    coo=graph.tocoo();row=np.r_[coo.row,roots,np.full(len(roots),n)];col=np.r_[coo.col,np.full(len(roots),n),roots]
    data=np.r_[coo.data,np.ones(2*len(roots),np.int8)]
    augmented=sparse.coo_matrix((data,(row,col)),shape=(n+1,n+1)).tocsr()
    order,parent=breadth_first_order(augmented,n,directed=False,return_predecessors=True)
    children=order[1:];parents=parent[children]
    rel=np.asarray(augmented[parents,children]).ravel()
    sign=np.ones(n+1,np.int8)
    for child,ancestor,value in zip(children,parents,rel):sign[child]=sign[ancestor]*value
    if np.any(sign[fa]*relation!=sign[fb]):raise ArithmeticError('Nonorientable ACVD quotient; original resolution must be retained locally')
    normal=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    score=np.bincount(components,weights=np.einsum('ij,ij->i',normal,expected)*sign[:n])
    sign[:n]*=np.where(score[components]<0,-1,1)
    faces=faces.copy();flip=sign[:n]<0;faces[flip]=faces[flip][:,[0,2,1]]
    return faces


def _clean_flat_stars(points,faces,pinned,tolerance):
    """GEMsim-style flat-cusp cleanup by a conservative3→1 surface move.

    Only an unpinned valence-three interior star whose central point lies on
    its supporting triangle within the geometric tolerance is simplified.
    Its boundary cycle is replaced, never left as a hole. Topology is checked
    before accepting the entire operation. No volume elements are deleted.
    """
    original_points=points;original_faces=faces;signature=_topology(points,faces)
    removed=0
    for iteration in range(3):
        incidence=sparse.coo_matrix((np.ones(faces.size,np.int8),(faces.ravel(),np.repeat(np.arange(len(faces)),3))),shape=(len(points),len(faces))).tocsr()
        candidates=np.flatnonzero((np.diff(incidence.indptr)==3)&~pinned)
        touched=np.zeros(len(faces),bool);delete=np.zeros(len(faces),bool);add=[]
        base=len(points);sf=np.sort(faces,axis=1)
        existing=set(map(int,(sf[:,0].astype(np.int64)*base+sf[:,1])*base+sf[:,2]))
        for vertex in candidates:
            attached=incidence.indices[incidence.indptr[vertex]:incidence.indptr[vertex+1]]
            if touched[attached].any():continue
            neighbors=np.unique(faces[attached]);neighbors=neighbors[neighbors!=vertex]
            if len(neighbors)!=3:continue
            key=(int(neighbors[0])*base+int(neighbors[1]))*base+int(neighbors[2])
            if key in existing:continue  # A genuine isolated tetrahedral shell.
            a,b,c=points[neighbors];normal=np.cross(b-a,c-a);norm=np.linalg.norm(normal)
            if norm<1e-20:continue
            if abs((points[vertex]-a)@normal)/norm>tolerance:continue
            basis=np.column_stack((b-a,c-a));uv=np.linalg.lstsq(basis,points[vertex]-a,rcond=None)[0]
            if uv.min()<-1e-7 or uv.sum()>1+1e-7:continue
            old=faces[attached[0]];boundary=[(old[i],old[(i+1)%3]) for i in range(3) if old[i]!=vertex and old[(i+1)%3]!=vertex][0]
            other=neighbors[~np.isin(neighbors,boundary)][0]
            add.append([boundary[0],boundary[1],other]);delete[attached]=True;touched[attached]=True;removed+=1
        if not add:break
        faces=np.concatenate((faces[~delete],np.asarray(add,np.int32)))
    used,inv=np.unique(faces,return_inverse=True);newfaces=inv.reshape(-1,3).astype(np.int32)
    if not _same_topology(signature,_topology(points[used],newfaces)):
        return original_points,original_faces,np.arange(len(original_points)),0
    mapping=np.full(len(points),-1,np.int64);mapping[used]=np.arange(len(used))
    return points[used],newfaces,mapping,removed


def _orient_against_mask(model,mask,points,faces,desired_outward=1):
    """Fix each closed component's material sense, including inner cavities.

    A closed chain can have globally wrong winding on one cavity and still
    pass Euler/edge tests. Vote using two-sided samples of the original union
    field, with increasing physical offsets and largest-area source faces.
    """
    edges=faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    graph=sparse.coo_matrix((np.ones(len(edges),np.int8),(edges[:,0],edges[:,1])),shape=(len(points),)*2).tocsr()
    _,components=connected_components(graph,directed=False)
    fc=components[faces[:,0]];p=points[faces]
    normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(normal,axis=1)
    normal/=np.maximum(area[:,None],1e-300);centre=p.mean(axis=1)
    field=mask.astype(np.float32);faces=faces.copy();flipped=0
    for component in np.unique(fc):
        group=np.flatnonzero(fc==component);selected=group[np.argsort(area[group])[-min(64,len(group)):]]
        votes=np.zeros(len(selected));h=min(model.spacing)
        for scale in (.5,1.,2.,3.,4.,6.,8.):
            unresolved=np.flatnonzero(votes==0)
            if not len(unresolved):break
            rows=selected[unresolved]
            plus=centre[rows]+normal[rows]*h*scale;minus=centre[rows]-normal[rows]*h*scale
            def sample(q):return ndimage.map_coordinates(field,((q[:,::-1]-model.origin)/model.spacing-.5).T,order=1,mode='constant',cval=0)
            plus=sample(plus);minus=sample(minus)
            votes[unresolved[(plus<.4)&(minus>.6)]]=1
            votes[unresolved[(plus>.6)&(minus<.4)]]=-1
        score=float(np.sum(area[selected]*votes));total=float(np.sum(area[selected]*abs(votes)))
        if total==0 or abs(score)<.1*total:raise ArithmeticError('Cannot determine a reconstructed cavity material orientation; increase surface resolution')
        if score*desired_outward<0:
            faces[group]=faces[group][:,[0,2,1]];flipped+=1
    return faces,flipped


def _cluster_patch(points,faces,weights,pinned,fraction,iterations,workspace,project=True,minimum_component_vertices=128):
    from .acvd_backend import clustering_class
    if fraction>=.999 or pinned.all():return points.copy(),faces.copy(),np.arange(len(points)),0
    original_edges=faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    component_graph=sparse.coo_matrix((np.ones(len(original_edges),np.int8),(original_edges[:,0],original_edges[:,1])),shape=(len(points),)*2).tocsr()
    _,component_ids=connected_components(component_graph,directed=False)
    component_sizes=np.bincount(component_ids)
    pinned=pinned.copy();pinned|=component_sizes[component_ids]<minimum_component_vertices
    clustering=clustering_class(workspace)(np.ascontiguousarray(points,dtype=float),np.ascontiguousarray(faces,dtype=np.int32),weights)
    clustering.cluster(max(4,min(len(points)-1,int(len(points)*fraction))),maxiter=iterations)
    ids=np.asarray(clustering.clusters).copy()
    if np.any(ids<0):raise ArithmeticError('ACVD left unassigned surface vertices')
    # Never merge a material junction or patch boundary with another vertex.
    pins=np.flatnonzero(pinned);ids[pins]=ids.max()+1+np.arange(len(pins))
    _,ids=np.unique(ids,return_inverse=True)
    # Isolating fixed junctions can disconnect an otherwise connected ACVD
    # cluster or leave an annular cluster surrounding a pinned vertex. Such a
    # cluster is not a contractible surface disk. Split disconnected pieces,
    # and retain original vertices for non-disk clusters before contraction.
    edges=np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    base=len(points);key=np.unique(edges[:,0].astype(np.int64)*base+edges[:,1])
    edges=np.column_stack((key//base,key%base))
    same=ids[edges[:,0]]==ids[edges[:,1]];e=edges[same]
    graph=sparse.coo_matrix((np.ones(len(e),np.int8),(e[:,0],e[:,1])),shape=(len(points),)*2).tocsr()
    _,ids=connected_components(graph,directed=False)
    vertices_per_cluster=np.bincount(ids)
    same=ids[edges[:,0]]==ids[edges[:,1]]
    inner_edges=np.bincount(ids[edges[same,0]],minlength=len(vertices_per_cluster))
    face_ids=ids[faces];inner=np.all(face_ids==face_ids[:,:1],axis=1)
    inner_faces=np.bincount(face_ids[inner,0],minlength=len(vertices_per_cluster))
    invalid=vertices_per_cluster-inner_edges+inner_faces!=1
    split=np.flatnonzero(invalid[ids]);ids[split]=ids.max()+1+np.arange(len(split))
    _,ids=np.unique(ids,return_inverse=True)
    signature=_topology(points,faces)
    # Enforce the quotient link condition locally: pinching a handle or a
    # junction produces duplicate faces or new open/nonmanifold edges. Split
    # only the implicated clusters, retaining original resolution there.
    for repair in range(8):
        candidate=ids[faces];candidate=candidate[np.all(np.diff(np.sort(candidate,axis=1),axis=1)>0,axis=1)]
        sorted_candidate=np.sort(candidate,axis=1);base=int(ids.max())+1
        keys=(sorted_candidate[:,0].astype(np.int64)*base+sorted_candidate[:,1])*base+sorted_candidate[:,2]
        _,first,counts=np.unique(keys,return_index=True,return_counts=True)
        unique_faces=candidate[first]
        result=_topology(np.empty((base,3)),unique_faces)
        if _same_topology(signature,result):break
        edge=np.sort(unique_faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
        k,n=np.unique(edge[:,0].astype(np.int64)*base+edge[:,1],return_counts=True)
        bad=k[n!=2];bad=np.unique(np.concatenate((bad//base,bad%base)))
        sizes=np.bincount(ids,minlength=base)
        bad=bad[sizes[bad]>1]
        if not len(bad):bad=np.unique(candidate[first[counts>1]])
        bad=bad[sizes[bad]>1]
        if not len(bad):break
        split=np.flatnonzero(np.isin(ids,bad));ids[split]=base+np.arange(len(split))
        _,ids=np.unique(ids,return_inverse=True)
    area=clustering._area
    total=np.bincount(ids,weights=area)
    centres=np.column_stack([np.bincount(ids,weights=area*points[:,i])/np.maximum(total,1e-300) for i in range(3)])
    if project:
        centres=.98*_project_to_triangles(centres,points,faces)+.02*centres
    centres[ids[pins]]=points[pins]
    newfaces=ids[faces];valid=np.all(np.diff(np.sort(newfaces,axis=1),axis=1)>0,axis=1)
    newfaces=newfaces[valid]
    sorted_faces=np.sort(newfaces,axis=1);base=len(centres)
    if base**3<np.iinfo(np.int64).max:
        _,first=np.unique((sorted_faces[:,0].astype(np.int64)*base+sorted_faces[:,1])*base+sorted_faces[:,2],return_index=True)
    else:_,first=np.unique(sorted_faces,axis=0,return_index=True)
    newfaces=newfaces[np.sort(first)]
    used,inv=np.unique(newfaces,return_inverse=True)
    # Map pinned source vertices to local output indices for stitching.
    pin_map=np.full(len(points),-1,np.int64)
    pin_map[pins]=np.searchsorted(used,ids[pins])
    if len(pins) and not np.array_equal(used[pin_map[pins]],ids[pins]):raise ArithmeticError('A pinned junction vanished')
    output_faces=inv.reshape(-1,3).astype(np.int32)
    source_normal=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    vertex_normal=np.zeros_like(points)
    np.add.at(vertex_normal,faces.ravel(),np.repeat(source_normal,3,axis=0))
    cluster_normal=np.column_stack([np.bincount(ids,weights=vertex_normal[:,i]) for i in range(3)])[used]
    expected=cluster_normal[output_faces].sum(axis=1)
    output_faces=_orient_surface(centres[used],output_faces,expected)
    spacing=np.median(np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1))
    output_pinned=np.zeros(len(used),bool);output_pinned[pin_map[pins]]=True
    cleaned,output_faces,mapping,removed=_clean_flat_stars(centres[used],output_faces,output_pinned,.05*spacing)
    pin_map[pins]=mapping[pin_map[pins]]
    return cleaned,output_faces,pin_map,removed


def _orient_quads(model,plc):
    quads=np.asarray(plc.facets);p=plc.points[quads];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    sample=p.mean(axis=1)-normal*min(model.spacing)*.25
    ijk=np.floor((sample[:,::-1]-model.origin)/model.spacing).astype(int)
    inside=np.all((ijk>=0)&(ijk<model.regions.shape),axis=1)
    left=np.full(len(ijk),-1,np.int64);left[inside]=model.regions[tuple(ijk[inside].T)]
    pair_map=plc.metadata['interface_pairs'];first=np.array([pair_map[str(int(i))][0] for i in plc.markers])
    flip=left!=first;quads[flip]=quads[flip][:,::-1]
    return np.concatenate((quads[:,[0,1,2]],quads[:,[0,2,3]])),np.tile(plc.markers,2)


def _region_volumes(points,faces,markers,pairs,regions):
    reference=points.mean(axis=0);p=points[faces]-reference
    volume=np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2]))/6
    totals={}
    for k in regions:
        total=0.
        for marker,pair in pairs.items():
            if k==pair[0]:total+=float(volume[markers==int(marker)].sum())
            elif k==pair[1]:total-=float(volume[markers==int(marker)].sum())
        if total<=0:raise ArithmeticError(f'Nonpositive enclosed surface volume for region{k}')
        totals[k]=total
    return totals


def _multiphase_surface(model,regions,options,workspace):
    raw=voxel_plc(model,regions=regions,maximum_facets=options.maximum_facets)
    faces,markers=_orient_quads(model,raw);points=raw.points
    weights=multiresolution_weights(model,points,options)
    # Separate outer box planes, so flat walls cannot be rounded across edges.
    pairs=raw.metadata['interface_pairs'];patch=markers.astype(np.int64)*7
    lo=np.asarray(model.origin)[::-1];hi=(np.asarray(model.origin)+np.asarray(model.spacing)*model.regions.shape)[::-1]
    for axis in range(3):
        for side,value in enumerate((lo[axis],hi[axis])):
            mask=np.all(np.isclose(points[faces,axis],value,atol=min(model.spacing)*1e-5,rtol=0),axis=1)
            patch[mask]+=axis*2+side+1
    minmark=np.full(len(points),np.iinfo(np.int64).max);maxmark=np.zeros(len(points),np.int64)
    np.minimum.at(minmark,faces.ravel(),np.repeat(patch,3));np.maximum.at(maxmark,faces.ravel(),np.repeat(patch,3))
    pinned=minmark!=maxmark
    # Also pin nonmanifold local edges; ordinary union-find cannot repair deletion.
    for marker in np.unique(patch):
        f=faces[patch==marker];edges=np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
        base=len(points);key,count=np.unique(edges[:,0].astype(np.int64)*base+edges[:,1],return_counts=True)
        e=np.column_stack((key//base,key%base));pinned[np.unique(e[count!=2])]=True
    global_pins=np.flatnonzero(pinned);pin_ids=np.full(len(points),-1,np.int64);pin_ids[global_pins]=np.arange(len(global_pins))
    output=[points[global_pins]];outfaces=[];outmarkers=[];cursor=len(global_pins);logs=[]
    length=max(hi-lo)
    for marker in np.unique(patch):
        source=faces[patch==marker];ids,local=np.unique(source,return_inverse=True);local=local.reshape(-1,3).astype(np.int32)
        before=_topology(points[ids],local);success=False
        fractions=sorted(set([min(1.,options.vertex_fraction*2**a) for a in range(options.topology_retries+1)]+[1.]))
        for fraction in fractions:
            p,f,pin_map,cleanup=_cluster_patch((points[ids]-lo)/length,local,weights[ids],pinned[ids],fraction,options.iterations,workspace,options.project_surface,options.minimum_component_vertices)
            after=_topology(p,f)
            if not _same_topology(before,after):continue
            p=p*length+lo
            newmap=np.full(len(p),-1,np.int64)
            lp=np.flatnonzero(pinned[ids]);newmap[pin_map[lp]]=pin_ids[ids[lp]]
            free=np.flatnonzero(newmap<0);newmap[free]=cursor+np.arange(len(free));cursor+=len(free)
            output.append(p[free]);outfaces.append(newmap[f]);outmarkers.append(np.full(len(f),marker//7,np.int32))
            logs.append({'patch':int(marker),'fraction':fraction,'before':before,'after':after,'pinned':len(lp),'flat_stars_replaced':cleanup});success=True;break
        if not success:raise ArithmeticError(f'ACVD patch{marker} did not preserve topology; increase vertex_fraction')
    p=np.concatenate(output);f=np.concatenate(outfaces);m=np.concatenate(outmarkers)
    volumes=_region_volumes(p,f,m,pairs,raw.seeds)
    return raw,p,f,m,volumes,logs,weights


def _nested_surface(model,regions,options,workspace):
    """Nonintersecting material contours for an explicitly declared phase chain.

    E.g. pore→CBD→active. No presumed numeric meaning of labels. Requires an
    outer phase on all box faces and resolved separation of nonadjacent phases.
    Arbitrary triple-junction segmentations use shared_interfaces instead.
    """
    from .voxel import from_regions
    order=tuple(map(int,options.phase_order or model.metadata.get('phase_order',())))
    if len(order)!=len(set(order)) or set(order)!=set(map(int,np.unique(model.regions))) or not set(regions)<=set(order):
        raise ValueError('phase_order must name every source phase exactly once; regions selects existing phases')
    a=model.regions;h=np.asarray(model.spacing);origin=np.asarray(model.origin)
    for axis in range(3):
        view=np.moveaxis(a,axis,0)
        if np.any(view[0]!=order[0]) or np.any(view[-1]!=order[0]):raise ValueError('Nested contours require the declared outer phase on all box faces')
    for i,k in enumerate(order[:-2]):
        distance=ndimage.distance_transform_edt(a!=k,sampling=h)
        for inside in order[i+2:]:
            if np.min(distance[a==inside])<2*min(h):raise ValueError('Nonadjacent nested phases have insufficient resolved separation; use shared_interfaces')
    low=origin[::-1];high=(origin+h*a.shape)[::-1]
    corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    box=low+corners*(high-low)
    q=np.array([[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]])
    f=np.concatenate((q[:,[0,1,2]],q[:,[0,2,3]]))[:,::-1]
    include_box=order[0] in regions
    points=[box] if include_box else [];faces=[f] if include_box else [];markers=[np.ones(len(f),np.int32)] if include_box else []
    masks={};normal_sense={}
    pairs={'1':[-1,order[0]]} if include_box else {};cursor=8 if include_box else 0
    for i in range(len(order)-1):
        if order[i] not in regions and order[i+1] not in regions:continue
        mask=np.isin(a,order[i+1:]).astype(np.uint8)
        contour=marching_cubes_plc(from_regions(mask,spacing=model.spacing,origin=model.origin),region=1,maximum_facets=options.maximum_facets)
        p=contour.points;f=np.asarray(contour.facets).copy();reference=p.mean(axis=0)
        signed=float(np.einsum('ij,ij->i',p[f[:,0]]-reference,np.cross(p[f[:,1]]-reference,p[f[:,2]]-reference)).sum()/6)
        pair=sorted([order[i],order[i+1]]);desired=1 if pair[0]==order[i+1] else -1
        if signed*desired<0:f=f[:,::-1]
        points.append(p);faces.append(f+cursor);cursor+=len(p);marker=i+2
        markers.append(np.full(len(f),marker,np.int32));pairs[str(marker)]=pair
        masks[marker]=mask;normal_sense[marker]=desired
    points=np.concatenate(points);faces=np.concatenate(faces);markers=np.concatenate(markers)
    if len(faces)>options.maximum_facets:raise ValueError('Nested contour input facet budget exceeded')
    weights=multiresolution_weights(model,points,options);output=[];outfaces=[];outmarkers=[];cursor=0;logs=[]
    length=max(high-low)
    for marker in np.unique(markers):
        source=faces[markers==marker];ids,local=np.unique(source,return_inverse=True);local=local.reshape(-1,3).astype(np.int32)
        before=_topology(points[ids],local)
        pinned=np.full(len(ids),marker==1 and include_box,bool)
        for attempt in range(options.topology_retries+1):
            fraction=min(1.,options.vertex_fraction*2**attempt)
            p,f,_,cleanup=_cluster_patch((points[ids]-low)/length,local,weights[ids],pinned,fraction,options.iterations,workspace,options.project_surface,options.minimum_component_vertices)
            after=_topology(p,f)
            if _same_topology(before,after):break
        else:raise ArithmeticError(f'Nested ACVD contour{marker} did not preserve topology')
        p=p*length+low;flips=0
        if marker in masks:f,flips=_orient_against_mask(model,masks[marker],p,f,normal_sense[marker])
        output.append(p);outfaces.append(f+cursor);cursor+=len(p);outmarkers.append(np.full(len(f),marker,np.int32))
        logs.append({'patch':int(marker),'fraction':fraction,'before':before,'after':after,'pinned':int(pinned.sum()),'flat_stars_replaced':cleanup,'cavity_orientation_components_flipped':flips})
    p=np.concatenate(output);f=np.concatenate(outfaces);m=np.concatenate(outmarkers)
    seeds={k:_component_seeds(a==k,h,origin) for k in regions}
    holes=_component_seeds(~np.isin(a,regions),h,origin,bounded=True)
    raw=SurfacePLC(points,faces,seeds,markers,holes=holes,region_volumes={},metadata={'interface_pairs':pairs})
    volumes=_region_volumes(p,f,m,pairs,regions)
    return raw,p,f,m,volumes,logs,weights


def adaptive_surface(model,*,regions=None,options=None,workspace=None):
    """Construct a weighted-remeshed, labelled PLC; retain full input geometry.

    Multiple labels use shared-interface patches with fixed common junctions.
    Single-label mode uses full-resolution marching cubes as in GEMsim.
    """
    start=time.perf_counter();options=options or AdaptiveMeshOptions()
    def progress(stage,**data):
        if os.environ.get('MORQEX_MESH_PROGRESS')=='1':print(json.dumps({'stage':stage,**data}),flush=True)
    if not isinstance(options,AdaptiveMeshOptions):raise TypeError('AdaptiveMeshOptions required')
    chosen=tuple(map(int,np.unique(model.regions) if regions is None else regions))
    if not chosen or len(chosen)!=len(set(chosen)) or not set(chosen)<=set(map(int,np.unique(model.regions))):
        raise ValueError('regions must be nonempty, unique labels present in the source')
    single=options.surface_method=='marching_cubes' or (options.surface_method=='auto' and len(chosen)==1)
    if single:
        if len(chosen)!=1:raise ValueError('Marching-cubes adaptation requires a single explicit region')
        raw=marching_cubes_plc(model,region=chosen[0],step_size=1,maximum_facets=options.maximum_facets)
        progress('full_resolution_surface',vertices=len(raw.points),facets=len(raw.facets))
        points=raw.points;faces=np.asarray(raw.facets);weights=multiresolution_weights(model,points,options)
        progress('residual_weights',minimum=float(weights.min()),maximum=float(weights.max()))
        before=_topology(points,faces);lo=points.min(axis=0);length=np.ptp(points,axis=0).max()
        for attempt in range(options.topology_retries+1):
            fraction=min(1.,options.vertex_fraction*2**attempt)
            p,f,_,cleanup=_cluster_patch((points-lo)/length,faces,weights,np.zeros(len(points),bool),fraction,options.iterations,workspace,options.project_surface,options.minimum_component_vertices)
            after=_topology(p,f)
            progress('weighted_acvd',fraction=fraction,vertices=len(p),topology_passed=_same_topology(before,after))
            if _same_topology(before,after):break
        else:raise ArithmeticError('ACVD did not preserve closed-surface topology; increase vertex_fraction')
        p=p*length+lo;m=np.ones(len(f),np.int32)
        f,flips=_orient_against_mask(model,model.regions==chosen[0],p,f,1)
        # Orient each closed connected contour consistently. Lorensen/Lewiner
        # input faces may be globally reversed by the ZYX→XYZ permutation.
        volume=float(np.einsum('ij,ij->i',p[f[:,0]]-lo,np.cross(p[f[:,1]]-lo,p[f[:,2]]-lo)).sum()/6)
        if volume<0:f=f[:,::-1];volume=-volume
        volumes={chosen[0]:volume}
        logs=[{'fraction':fraction,'before':before,'after':after,'pinned':0,'flat_stars_replaced':cleanup,'cavity_orientation_components_flipped':flips}]
    else:
        nested=options.surface_method=='nested_contours' or (options.surface_method=='auto' and (options.phase_order or model.metadata.get('phase_order')))
        raw,p,f,m,volumes,logs,weights=(_nested_surface if nested else _multiphase_surface)(model,chosen,options,workspace)
    voxel={k:float(np.count_nonzero(model.regions==k)*np.prod(model.spacing)) for k in chosen}
    errors={str(k):abs(volumes[k]/voxel[k]-1) for k in chosen}
    if max(errors.values())>options.maximum_phase_volume_error:
        raise ArithmeticError(f'Adaptive surface phase-volume error{errors} exceeds {options.maximum_phase_volume_error}; increase vertex_fraction')
    metadata={'surface_method':'gemsim_adaptive','source_shape_zyx':model.regions.shape,
        'source_voxels':int(model.regions.size),'full_resolution':True,
        'minimum_component_vertices':options.minimum_component_vertices,
        'selected_regions':list(chosen),
        'algorithm':'full-resolution interfaces → GEMsim multiresolution median residual weights → original weighted ACVD → topology/phase-volume checks → TetGen',
        'input_surface_vertices':len(raw.points),'input_surface_facets':len(raw.facets),
        'output_surface_vertices':len(p),'output_surface_facets':len(f),
        'source_voxel_volume_m3':voxel,'phase_voxel_volume_relative_errors':errors,
        'patches':logs,'weight_range':[float(weights.min()),float(weights.max())],
        'region_names':model.region_names,'interface_pairs':raw.metadata.get('interface_pairs',{}),
        'phase_order':list(options.phase_order or model.metadata.get('phase_order',())),
        'elapsed_seconds':time.perf_counter()-start,
        'adaptivity':'geometric residual-weighted surface density; not a certified PDE error estimator',
        'upstream':'https://github.com/Zephyrainzjl/GEMsim/blob/main/src/gem/mesh/elements.py',
        'changes':'positive bounded weights, SI coordinates, shared-junction and tiny-component pinning, disk-cluster recovery, topology-preserving flat-star replacement, material-oriented cavities; no skipped-face or low-quality-cell deletion'}
    return SurfacePLC(p,np.asarray(f,np.int32),raw.seeds,np.asarray(m,np.int32),raw.holes,volumes,metadata)


def adaptive_mesh(model,*,regions=None,options=None,tetgen_options=None,workspace=None):
    options=options or AdaptiveMeshOptions();start=time.perf_counter();failures=[]
    native=tetgen_options or TetGenOptions.for_adaptive()
    fractions=sorted(set(min(1.,options.vertex_fraction*2**attempt) for attempt in range(options.topology_retries+1)))
    for fraction in fractions:
        trial=replace(options,vertex_fraction=fraction)
        try:
            plc=adaptive_surface(model,regions=regions,options=trial,workspace=workspace)
            if os.environ.get('MORQEX_MESH_PROGRESS')=='1':print(json.dumps({'stage':'tetgen_start','vertices':len(plc.points),'facets':len(plc.facets)}),flush=True)
            try:
                result=tetgen_mesh(plc,options=native,workspace=workspace,maximum_facets=options.maximum_facets)
            except (ArithmeticError,RuntimeError) as error:
                # Native merging can round almost-planar curved facets even
                # with a strict p-angle. Keep the SAME geometric PLC and the
                # SAME1e-8 volume/quality gates; retain its facets and retry.
                # This can constrain shape quality, which remains measured.
                if not (options.retry_without_facet_merging and native.merge_coplanar and
                        'Exact PLC region ' in str(error) and ' volume error ' in str(error)):raise
                failures.append({'fraction':trial.vertex_fraction,'stage':'native_facet_merge',
                    'reason':str(error),'retry':'same PLC with merge_coplanar=False; acceptance tolerances unchanged'})
                if os.environ.get('MORQEX_MESH_PROGRESS')=='1':print(json.dumps({'stage':'native_interface_retry',**failures[-1]}),flush=True)
                result=tetgen_mesh(plc,options=replace(native,merge_coplanar=False),workspace=workspace,
                    maximum_facets=options.maximum_facets)
            result.report.update(pipeline='GEMsim residual-weighted ACVD + TetGen',
                adaptive_elapsed_seconds=time.perf_counter()-start,adaptive_retries=failures)
            return result
        except (ArithmeticError,RuntimeError) as error:
            failures.append({'fraction':trial.vertex_fraction,'reason':str(error)})
            if os.environ.get('MORQEX_MESH_PROGRESS')=='1':print(json.dumps({'stage':'validation_retry',**failures[-1]}),flush=True)
    raise ArithmeticError('Adaptive mesh did not pass validation; all original faces retained in input. '+json.dumps(failures))
