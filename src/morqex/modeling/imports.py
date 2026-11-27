"""Traceable TIFF/COMSOL import; material roles and length scales are explicit."""
from pathlib import Path
import hashlib
import numpy as np
from .voxel import from_regions


def tiff_voxels(path,*,spacing,origin=(0.,0.,0.),region_names=None,axes_to_zyx=None):
    import tifffile
    path=Path(path).resolve()
    with tifffile.TiffFile(path) as f:
        series=f.series[0]
        if len(series.shape)!=3 or np.prod(series.shape)*np.dtype(series.dtype).itemsize>128*2**20:
            raise ValueError('Requires a segmented3D TIFF below128MiB decoded')
        if axes_to_zyx is None:
            if series.axes!='ZYX':raise ValueError('Declare axes_to_zyx explicitly for a TIFF without ZYX axes')
            axes_to_zyx=(0,1,2)
        if sorted(axes_to_zyx)!=[0,1,2]:raise ValueError('axes_to_zyx must be a permutation')
        data=f.asarray(maxworkers=1).transpose(axes_to_zyx)
        axes=series.axes
    names=region_names or {int(k):f'gray_{k}' for k in np.unique(data)}
    with path.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    return from_regions(data,spacing=spacing,origin=origin,region_names=names,metadata={
        'source':str(path),'source_sha256':checksum,'source_axes':axes,'axes_to_zyx':axes_to_zyx,
        'source_shape':list(data.shape),'segmentation_changed':False,
        'spacing_source':'explicit caller choice; not inferred from intensity or array size',
        'phase_roles':'caller supplied names or unassigned original gray labels'})


def material_interfaces(cells,phases):
    """Unique actual Tet4 faces separating domains, plus exterior boundaries.

    Integer sorting detects shared faces; no centroid or proximity welding.
    Input orientation should already follow the positive Tet4 convention.
    """
    cells=np.asarray(cells);phases=np.asarray(phases)
    faces=cells[:,[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1,3)
    sorted_faces=np.sort(faces,axis=1);base=int(cells.max())+1
    if base**3<np.iinfo(np.int64).max:
        keys=(sorted_faces[:,0].astype(np.int64)*base+sorted_faces[:,1])*base+sorted_faces[:,2]
    else:keys=np.ascontiguousarray(sorted_faces,dtype=np.int64).view(np.dtype([('a','i8'),('b','i8'),('c','i8')])).reshape(-1)
    order=np.argsort(keys);unique,first,counts=np.unique(keys[order],return_index=True,return_counts=True)
    if np.any(counts>2):raise ValueError('Nonmanifold Tet4 face shared by more than two cells')
    a=order[first];left=phases[a//4];right=np.full(len(a),-1,np.int64)
    inside=counts==2;right[inside]=phases[order[first[inside]+1]//4]
    chosen=(counts==1)|(left!=right)
    pairs,markers=np.unique(np.sort(np.column_stack((left[chosen],right[chosen])),axis=1),axis=0,return_inverse=True)
    selected=faces[a[chosen]].copy();reverse=left[chosen]>right[chosen]
    selected[reverse]=selected[reverse][:,::-1]
    return selected,markers.astype(np.int32)+1,{str(i+1):list(map(int,pair)) for i,pair in enumerate(pairs)}


def import_comsol(path,*,length_unit=None,region_names=None):
    """Convert a linear Tet4 COMSOL mesh to SI, retaining domain IDs/metadata.

    A MORQEX unit comment is used unless the caller supplies a consistent unit.
    Reference files without unit metadata require the caller's import unit.
    Reversed local node ordering is canonicalized and counted, never deleted.
    Interface markers are derived from actual volume-cell ownership; original
    boundary-selection metadata is recorded, not silently reattached to those
    new markers. The supplied reference has three domain selections.
    """
    from .comsol import read_comsol,LENGTH_UNITS
    from .tetgen import TetGenMesh,tetra_quality
    source=read_comsol(path)
    declared=source['length_unit'];unit_source='explicit caller choice'
    if length_unit is None:
        if declared is None:raise ValueError('Explicit length_unit required for COMSOL mesh without unit metadata')
        length_unit=declared;unit_source='MORQEX length_unit file comment'
    if length_unit not in LENGTH_UNITS:raise ValueError('Unsupported COMSOL input unit')
    if declared is not None and (declared not in LENGTH_UNITS or LENGTH_UNITS[declared]!=LENGTH_UNITS[length_unit]):
        raise ValueError('COMSOL input length_unit conflicts with declared file unit')
    if 'tet' not in source['elements']:raise ValueError('Tet4 COMSOL mesh required')
    points=source['points']*LENGTH_UNITS[length_unit]
    cells=source['elements']['tet']['cells'].copy();phases=source['elements']['tet']['entities'].copy()
    flips=0
    for start in range(0,len(cells),50000):
        c=cells[start:start+50000];p=points[c]
        negative=np.linalg.det(p[:,1:]-p[:,:1])<0
        c[negative]=c[negative][:,[1,0,2,3]];flips+=int(negative.sum())
    names={int(k):f'domain_{k}' for k in np.unique(phases)}
    for item in source['selections'].values():
        if item['dimension']==3 and len(item['entities'])==1:names[item['entities'][0]]=item['label']
    if region_names is not None:names={int(k):str(v) for k,v in region_names.items()}
    if set(names)!=set(map(int,np.unique(phases))):raise ValueError('Name all imported domains')
    facets,markers,pairs=material_interfaces(cells,phases);quality,volumes=tetra_quality(points,cells)
    with Path(path).open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    report={'mesher':'imported COMSOL reference, not regenerated','nodes':len(points),'tetrahedra':len(cells),
        'quality':quality,'quality_limits_satisfied':None,'quality_request':'none for imported reference',
        'orientation_permutations':flips,'source_selections':source['selections'],'source_mesh_tag':source['mesh_tag'],
        'region_volume_m3':{str(k):float(volumes[phases==k].sum()) for k in names},
        'geometry':{'surface_method':'actual imported Tet4 material faces','region_names':names,'interface_pairs':pairs,
            'source':str(Path(path).resolve()),'source_sha256':checksum,'import_length_unit':length_unit,
            'length_unit_source':unit_source,
            'length_scale_to_m':LENGTH_UNITS[length_unit],'matching_tiff':'none; independent user reference'}}
    return TetGenMesh(points,cells,phases,facets=facets,facet_markers=markers,report=report)
