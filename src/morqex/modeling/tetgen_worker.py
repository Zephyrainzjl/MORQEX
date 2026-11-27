"""Isolated native TetGen execution; all IPC files remain external."""
from dataclasses import asdict,replace
from pathlib import Path
import json,os,subprocess,sys,tempfile,time
import numpy as np


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(f'Nonserializable mesh metadata: {type(value).__name__}')


def isolated_mesh(plc,options,workspace,maximum_facets):
    import h5py
    from .voxel import external_path
    from .tetgen import TetGenMesh
    from .tetgen_backend import library_path
    root=external_path(workspace or os.environ.get('MORQEX_WORKSPACE',str(Path(tempfile.gettempdir())/'morqex')))
    folder=root/'cache/native_mesh_calls';folder.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(dir=folder,prefix='tetgen_') as temporary:
        directory=Path(temporary);source=directory/'input.h5';target=directory/'output.h5'
        with h5py.File(source,'w') as f:
            f.create_dataset('points',data=plc.points)
            if isinstance(plc.facets,np.ndarray):
                indices=plc.facets.ravel();offsets=np.arange(len(plc.facets)+1,dtype=np.int64)*plc.facets.shape[1]
            else:
                indices=np.concatenate(plc.facets);offsets=np.r_[0,np.cumsum([len(p) for p in plc.facets])]
            f.create_dataset('facet_indices',data=indices)
            f.create_dataset('facet_offsets',data=offsets)
            f.create_dataset('markers',data=plc.markers);f.create_dataset('holes',data=plc.holes)
            f.attrs.update(seeds=json.dumps(plc.seeds,default=_json),volumes=json.dumps(plc.region_volumes,default=_json),
                metadata=json.dumps(plc.metadata,default=_json),options=json.dumps(asdict(replace(options,isolated=False))),maximum_facets=maximum_facets)
        env=os.environ.copy();env['MORQEX_TETGEN_LIBRARY']=str(library_path(workspace))
        # Source checkout and installed wheel both resolve through the active
        # package root, including notebooks with a modified sys.path.
        package_root=str(Path(__file__).resolve().parents[2]);env['PYTHONPATH']=package_root+os.pathsep+env.get('PYTHONPATH','')
        try:
            result=subprocess.run([sys.executable,'-m','morqex.modeling.tetgen_worker',str(source),str(target)],
                env=env,capture_output=True,text=True,timeout=options.native_timeout_seconds)
        except subprocess.TimeoutExpired as error:raise RuntimeError('Isolated TetGen exceeded its native time budget') from error
        if result.returncode or not target.exists():
            reason=(result.stderr or result.stdout)[-1600:]
            raise RuntimeError(f'Isolated TetGen failed (exit {result.returncode}); no mesh accepted. {reason}')
        with h5py.File(target,'r') as f:
            mesh=TetGenMesh(f['mesh/points_m'][:],f['mesh/tetrahedra'][:],f['mesh/phase'][:],
                facets=f['mesh/facets'][:],facet_markers=f['mesh/facet_markers'][:],report=json.loads(f.attrs['report']))
        mesh.report.update(native_process_isolated=True,native_process_wall_seconds=time.perf_counter()-start)
        return mesh


def main(source,target):
    import faulthandler,h5py
    faulthandler.enable()
    from .tetgen import SurfacePLC,TetGenOptions,tetgen_mesh
    with h5py.File(source,'r') as f:
        indices=f['facet_indices'][:];offsets=f['facet_offsets'][:]
        sizes=np.diff(offsets)
        facets=indices.reshape(-1,int(sizes[0])) if np.all(sizes==sizes[0]) else [indices[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
        options=json.loads(f.attrs['options'])
        options['region_maximum_volumes']={int(k):v for k,v in options['region_maximum_volumes'].items()}
        plc=SurfacePLC(f['points'][:],facets,{int(k):v for k,v in json.loads(f.attrs['seeds']).items()},
            f['markers'][:],f['holes'][:],{int(k):v for k,v in json.loads(f.attrs['volumes']).items()},json.loads(f.attrs['metadata']))
        maximum=int(f.attrs['maximum_facets'])
    tetgen_mesh(plc,options=TetGenOptions(**options),maximum_facets=maximum).save_hdf5(target)


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
