"""Prove every displayed local Tet4 is present in the full exported mesh."""
import base64,gzip,json,os
from pathlib import Path
import h5py,numpy as np

root=Path(os.environ['MORQEX_WORKSPACE'])/'atlas256';records=[]
def unpack(value):return np.frombuffer(gzip.decompress(base64.b64decode(value['gzip'])),dtype=value['dtype']).reshape(value['shape'])
for i in range(10):
    text=(root/'data'/f'model_{i:02d}.js').read_text();payload=json.loads(text[text.index(',')+1:-2])
    record=payload['record'];probe=payload['probe'];xyz=unpack(probe['points']).astype(np.float64)*4
    attributes=unpack(probe['attributes']);voxel_xyz=np.floor(xyz.mean(axis=1)).astype(int)
    parent=np.ravel_multi_index(voxel_xyz[:,::-1].T,(256,)*3)
    with h5py.File(root/record['files']['hdf5']) as f:
        labels=f['phase_labels'][:];selected=np.flatnonzero(labels.ravel()>0)
        rank=np.searchsorted(selected,parent);np.testing.assert_array_equal(selected[rank],parent)
        rows=rank*6+np.arange(len(rank))%6
        cells=f['mesh/tetrahedra'][rows];nodes,inverse=np.unique(cells,return_inverse=True)
        actual=f['mesh/points_m'][nodes][inverse].reshape(-1,4,3)
        np.testing.assert_allclose(actual,xyz*record['voxel_size_m'],rtol=1e-13,atol=1e-20)
        np.testing.assert_array_equal(f['mesh/parent_voxel'][rows],parent)
        np.testing.assert_array_equal(f['mesh/phase'][rows],attributes[:,0])
    records.append({'structure':record['key'],'displayed_tetrahedra_verified':len(cells),'exact_full_mesh_association':True})
(root/'probe_mapping_validation.json').write_text(json.dumps({'passed':True,'structures':records},indent=2))
print(json.dumps({'passed':True,'structures':10,'displayed_cells':sum(r['displayed_tetrahedra_verified'] for r in records)}))
