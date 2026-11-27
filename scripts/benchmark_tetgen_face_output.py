"""Measure all-face vs constrained-face output on the same physical PLC.

Input is an external MORQEX Tet4 HDF5 artifact. Run each mode separately under
run_local.py. This is an output-memory benchmark, not a PDE accuracy benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--mode',choices=['all_faces','constrained'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-input-cells',type=int,default=5000000)
    parser.add_argument('--maximum-cells',type=int,default=8000000)
    args=parser.parse_args()
    from morqex.modeling import SurfacePLC,TetGenOptions,tetgen_mesh,material_interfaces
    from morqex.modeling.voxel import external_path
    import morqex.modeling.tetgen_backend as backend
    output=external_path(args.output)
    if output.exists():raise FileExistsError(output)
    start=time.perf_counter()
    with h5py.File(args.input) as h:
        if not h.attrs.get('complete') or len(h['mesh/tetrahedra'])>args.maximum_input_cells:
            raise ValueError('Complete input within an explicit cell budget required')
        points=h['mesh/points_m'][:];cells=h['mesh/tetrahedra'][:];phases=h['mesh/phase'][:]
    facets,markers,pairs=material_interfaces(cells,phases)
    seeds={};volumes={}
    for start_ in range(0,len(cells),50000):
        c=cells[start_:start_+50000];p=points[c];v=np.linalg.det(p[:,1:]-p[:,:1])/6
        if not np.all(np.isfinite(v)) or np.any(v<=0):raise ArithmeticError('Invalid input volume')
        local=phases[start_:start_+50000]
        for k in np.unique(local):
            key=int(k);selected=local==k
            volumes[key]=volumes.get(key,0.)+float(v[selected].sum())
            if key not in seeds:seeds[key]=p[np.flatnonzero(selected)[0]].mean(0,keepdims=True)
    source_cells=len(cells);del cells,phases,p,c,v
    used,inverse=np.unique(facets,return_inverse=True)
    plc=SurfacePLC(points[used],inverse.reshape(-1,3),seeds,markers,
        region_volumes=volumes,metadata={'interface_pairs':pairs,
            'source':'actual exterior and material interfaces of supplied Tet4 mesh'})
    del points,facets,used,inverse
    native=backend.execute;counts={}
    def execute(points,facets,markers,regions,holes,switches,**kwargs):
        flags=switches+('f' if args.mode=='all_faces' else '')
        result=native(points,facets,markers,regions,holes,flags,**kwargs)
        counts.update(raw_returned_faces=len(result[3]),raw_marked_faces=int(np.count_nonzero(result[4]>0)),switches=flags)
        return result
    backend.execute=execute
    begin=time.perf_counter()
    try:
        mesh=tetgen_mesh(plc,maximum_facets=16000000,options=TetGenOptions(
            preserve_surface=True,merge_coplanar=False,region_labelling='interfaces',
            maximum_cells=args.maximum_cells,maximum_steiner=800000,coplanarity_tolerance=1e-15))
    finally:backend.execute=native
    elapsed=time.perf_counter()-begin
    mesh.report['base_call_switches']=mesh.report['switches']
    mesh.report['switches']=counts['switches']
    digest=lambda a:hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
    canonical=np.column_stack((np.sort(mesh.facets,axis=1),mesh.facet_markers))
    canonical=canonical[np.lexsort(tuple(canonical[:,i] for i in (3,2,1,0)))]
    with args.input.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    result={'passed_geometry':True,'input':str(args.input.resolve()),'input_sha256':checksum,
        'input_cells':source_cells,'mode':args.mode,'nodes':len(mesh.points),'cells':len(mesh.cells),
        'native_and_validation_seconds':elapsed,'wall_seconds':time.perf_counter()-start,
        'hashes':{'points':digest(mesh.points),'cells':digest(mesh.cells),'phases':digest(mesh.phases),'physical_faces':digest(canonical)},
        'report':mesh.report,**counts,
        'scope':'single measured execution; numerical and shape validation are separate; no statistical speedup claim'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='report'}),flush=True)


if __name__=='__main__':main()
