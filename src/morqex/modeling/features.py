"""Physical feature descriptors for mesh advice, not certified error bounds."""
import hashlib,time
import numpy as np
from scipy import ndimage
from .progress import progress_stage


def voxel_fingerprint(model):
    import json
    d=hashlib.sha256(np.ascontiguousarray(model.regions).view(np.uint8))
    semantics={'shape':model.regions.shape,'dtype':str(model.regions.dtype),'spacing':model.spacing,
        'origin':model.origin,'names':model.region_names}
    if model.metadata.get('selection_groups'):semantics['selection_groups']=model.metadata['selection_groups']
    d.update(json.dumps(semantics,sort_keys=True).encode())
    return d.hexdigest()


def contact_areas(model):
    """Actual face contacts in SI; mere proximity is not a contact."""
    a=model.regions;base=int(a.max())+1;result={}
    for axis in range(3):
        lo=[slice(None)]*3;hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        left=a[tuple(lo)];right=a[tuple(hi)];different=left!=right
        x=left[different].astype(np.int64);y=right[different].astype(np.int64)
        keys,counts=np.unique(np.minimum(x,y)*base+np.maximum(x,y),return_counts=True)
        area=float(np.prod(np.delete(model.spacing,axis)))
        for key,n in zip(keys,counts):
            pair=(int(key//base),int(key%base));name=f'{pair[0]}:{pair[1]}'
            result[name]=result.get(name,0.)+int(n)*area
    return result


def analyze_features(model,*,progress=None):
    """EDT medial-maximum diameter proxies with explicit voxel uncertainty.

    A radius measured between voxel centres is not an exact physical throat
    radius. The corrected proxy below is only a mesh-sizing heuristic. All
    nonempty phases are included; no gray value is interpreted as pore/CBD.
    """
    from .voxel import VoxelModel
    if not isinstance(model,VoxelModel):raise TypeError('VoxelModel required')
    start=time.perf_counter();phases=np.unique(model.regions);regions={};h=np.array(model.spacing)
    with progress_stage('feature analysis / 最小特征分析',progress,estimated_seconds=max(.1,model.regions.size*len(phases)/8e6)) as stage:
        for j,k in enumerate(phases):
            mask=model.regions==k;components,n=ndimage.label(mask);counts=np.bincount(components.ravel())[1:]
            # Padding gives a defined exterior even for a completely solid box.
            distance=ndimage.distance_transform_edt(np.pad(mask,1),sampling=h)[1:-1,1:-1,1:-1]
            maxima=mask&(distance>=ndimage.maximum_filter(distance,size=3,mode='constant'))
            radii=distance[maxima];sample=radii[::max(1,len(radii)//200000)]
            widths=np.maximum(float(h.min()),2*sample-float(h.max()))
            q=np.quantile(widths,[0,.05,.1,.5,.9,1])
            through=[]
            # Connectivity is computed from actual face-connected voxels.
            connected=components
            for axis in range(3):
                a=np.unique(np.take(connected,0,axis));b=np.unique(np.take(connected,-1,axis))
                through.append(bool(np.intersect1d(a[a>0],b[b>0]).size))
            del connected,components
            regions[str(int(k))]={'name':model.region_names[int(k)],'voxels':int(mask.sum()),'fraction':float(mask.mean()),
                'components_6':int(n),'largest_component_voxels':int(counts.max()),'spans_zyx':through,
                'thickness_proxy_m':dict(zip(('minimum','p05','p10','median','p90','maximum'),map(float,q))),
                'medial_samples':int(len(radii)),'single_voxel_components':int(np.count_nonzero(counts==1))}
            stage.update(j+1,len(phases),f'region {int(k)}')
    contact=contact_areas(model);extent=np.array(model.regions.shape)*h
    thin=min(r['thickness_proxy_m']['p10'] for r in regions.values())
    finest=float(h.min()/2);coarsest=float(max(finest*4,extent.min()/16))
    return {'source_fingerprint':voxel_fingerprint(model),'shape_zyx':list(model.regions.shape),
        'spacing_zyx_m':list(map(float,h)),'origin_zyx_m':list(map(float,model.origin)),
        'extent_zyx_m':extent.tolist(),'regions':regions,'contact_areas_m2':contact,
        'recommended_sizes':{'bulk_size_range_m':[finest,coarsest],
            'balanced_bulk_size_m':float(extent.min()/32),
            'feature_surface_size_m':float(max(h.min(),min(thin/2,extent.min()/48))),
            'thickness_proxy_p10_min_m':float(thin),'voxel_geometry_uncertainty_scale_m':float(h.max()),
            'scope':'advisory sizing requests, not strict edge bounds or PDE certification; coarse bulk sizes require smaller cells near thin interfaces'},
        'elapsed_seconds':time.perf_counter()-start,
        'thickness_definition':'max(min_spacing, 2*EDT_at_3x3x3_local_maxima-max_spacing); approximate centre-grid proxy, not a certified minimum throat width',
        'resolution_scope':'no smoothing or resampling; complete original array retained'}
