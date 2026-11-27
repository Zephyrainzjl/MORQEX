"""Read every full256³ mesh cell back and verify geometry/region association."""
import json,os,time
from pathlib import Path
import numpy as np,h5py

root=Path(os.environ['MORQEX_WORKSPACE'])/'atlas256';records=[]
for file in sorted((root/'structures').glob('*/structure.h5')):
    tick=time.perf_counter();record=json.loads(file.with_name('metadata.json').read_text())
    with h5py.File(file) as f:
        if not f.attrs['complete']:raise ValueError('Incomplete mesh')
        labels=f['phase_labels'][:];spacing=np.asarray(f.attrs['spacing_zyx_m']);origin=np.asarray(f.attrs['origin_zyx_m'])
        if labels.shape!=(256,256,256):raise ValueError('Expected actual256³ source geometry')
        selected=np.flatnonzero(np.isin(labels,f.attrs['selected_regions']))
        points=f['mesh/points_m'][:];used=np.zeros(len(points),bool)
        indices=np.rint((points[:,::-1]-origin)/spacing).astype(np.int64)
        np.testing.assert_allclose(points,(indices*spacing+origin)[:,::-1],rtol=1e-14,atol=1e-20)
        global_ids=np.ravel_multi_index(indices.T,(257,)*3)
        if np.any(np.diff(global_ids)<=0):raise ValueError('Duplicate or noncanonical global node IDs')
        del indices,global_ids
        total=len(f['mesh/tetrahedra']);volume=0.;minimum=np.inf
        if total!=6*len(selected) or total!=record['tetrahedra']:raise ValueError('Element count mismatch')
        for start in range(0,total,131072):
            stop=min(total,start+131072);cells=f['mesh/tetrahedra'][start:stop]
            if cells.max()>=len(points):raise ValueError('Node outside range')
            used[cells]=True;parents=f['mesh/parent_voxel'][start:stop]
            np.testing.assert_array_equal(parents,selected[np.arange(start,stop)//6])
            np.testing.assert_array_equal(f['mesh/phase'][start:stop],labels.ravel()[parents])
            xyz=points[cells];jac=xyz[:,1:]-xyz[:,:1];vol=np.linalg.det(jac)/6
            if np.any(vol<=0):raise ValueError('Nonpositive tetrahedron')
            np.testing.assert_allclose(vol,np.prod(spacing)/6,rtol=1e-11,atol=0)
            volume+=float(vol.sum());minimum=min(minimum,float(vol.min()))
        if not used.all():raise ValueError('Unused global nodes')
        if not np.isclose(volume,len(selected)*np.prod(spacing),rtol=1e-11):raise ValueError('Total volume mismatch')
    result={'structure':record['key'],'all_cells_checked':total,'all_nodes_checked':len(points),
        'actual_shape':[256]*3,'positive_volumes':True,'exact_region_parent_association':True,
        'shared_unique_global_nodes':True,'volume_relative_error':float(abs(volume/(len(selected)*np.prod(spacing))-1)),
        'seconds':time.perf_counter()-tick}
    records.append(result);print(json.dumps(result),flush=True)
if len(records)!=10:raise ValueError('All ten structures must be checked')
(root/'validation.json').write_text(json.dumps({'passed':True,'structures':records,
    'total_tetrahedra_checked':sum(r['all_cells_checked'] for r in records),
    'scope':'Full geometry and mesh consistency, not a physical solve or mesh-convergence claim'},indent=2))
