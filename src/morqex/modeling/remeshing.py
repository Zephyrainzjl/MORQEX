"""Optional source-integrated MMG remeshing with measured geometric gates.

MMG adapts an existing conforming multi-material Tet4 mesh. It does not infer
geometry from a voxel segmentation, and cannot remove thin physical features
while simultaneously promising exact geometry. No solver-error claim is made.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import ctypes as ct
import json,os,subprocess,sys,tempfile,time
import numpy as np
from .tetgen import TetGenMesh,tetra_quality


@dataclass(frozen=True)
class RemeshOptions:
    minimum_edge_m:float|None=None
    maximum_edge_m:float|None=None
    uniform_edge_m:float|None=None
    hausdorff_m:float|None=None
    gradation:float=1.3
    feature_angle_degrees:float=45.
    preserve_surface:bool=False
    optimize_only:bool=False
    open_boundaries:bool=False
    maximum_phase_volume_error:float=.02
    maximum_cells:int=5000000
    maximum_input_cells:int|None=None
    maximum_nodes:int=3000000
    memory_mib:int=2500
    timeout_seconds:float=240.
    isolated:bool=True
    check_surface_topology:bool=True

    def __post_init__(self):
        for value in (self.minimum_edge_m,self.maximum_edge_m,self.uniform_edge_m,self.hausdorff_m):
            if value is not None and (not np.isfinite(value) or value<=0):raise ValueError('Positive finite SI sizes required')
        if self.minimum_edge_m is not None and self.maximum_edge_m is not None and self.minimum_edge_m>self.maximum_edge_m:raise ValueError('minimum_edge_m exceeds maximum_edge_m')
        if self.uniform_edge_m is not None and self.optimize_only:raise ValueError('uniform_edge_m and optimize_only are mutually exclusive')
        if not np.isfinite(self.gradation) or self.gradation<=1:raise ValueError('gradation must exceed1')
        if not 0<self.feature_angle_degrees<=180:raise ValueError('feature angle must be in(0,180]')
        if not 0<self.maximum_phase_volume_error<1:raise ValueError('Positive phase-volume tolerance required')
        for value in (self.maximum_cells,self.maximum_nodes,self.memory_mib):
            if type(value) is not int or value<1:raise ValueError('Positive integer resource budgets required')
        if self.maximum_input_cells is not None and (type(self.maximum_input_cells) is not int or self.maximum_input_cells<1):raise ValueError('Positive integer input-cell budget required')
        if not np.isfinite(self.timeout_seconds) or self.timeout_seconds<=0:raise ValueError('Positive finite time budget required')


def _surface_signature(cells,regions,facets,owners):
    """Boundary component count and Euler characteristic, not an isotopy proof."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    base=int(cells.max())+1;signature={}
    for region in np.unique(regions):
        selected=(regions[owners[:,0]]==region)|((owners[:,1]>=0)&(regions[np.maximum(owners[:,1],0)]==region))
        face=facets[selected];vertices=np.unique(face)
        edge=np.sort(face[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1).astype(np.int64)
        key=edge[:,0]*base+edge[:,1];key,counts=np.unique(key,return_counts=True)
        a=np.searchsorted(vertices,key//base);b=np.searchsorted(vertices,key%base)
        graph=coo_matrix((np.ones(len(a),bool),(a,b)),shape=(len(vertices),len(vertices))).tocsr()
        components=connected_components(graph,directed=False,return_labels=False)
        signature[str(int(region))]={'surface_components':int(components),'euler_characteristic':int(len(vertices)-len(key)+len(face)),
            'has_nonmanifold_edges':bool(np.any(counts>2)),'has_open_edges':bool(np.any(counts<2))}
    return signature


def _tag_pairs(regions,markers,owners):
    left=regions[owners[:,0]].astype(np.int64);right=np.full(len(left),-1,np.int64)
    inside=owners[:,1]>=0;right[inside]=regions[owners[inside,1]]
    return np.unique(np.column_stack((markers,np.minimum(left,right),np.maximum(left,right))),axis=0)


def _native(points,cells,regions,facets,markers,options,workspace):
    from .mmg_backend import library_path
    lib=ct.CDLL(str(library_path(workspace)));ip=ct.POINTER(ct.c_int);dp=ct.POINTER(ct.c_double)
    lib.mq_mmg_abi.restype=ct.c_int
    if lib.mq_mmg_abi()!=1:raise RuntimeError('Unsupported MMG bridge ABI')
    lib.mq_mmg_run.argtypes=[ct.c_int,dp,ct.c_int,ip,ip,ct.c_int,ip,ip,dp,ip,ct.c_char_p,ct.c_int]
    lib.mq_mmg_run.restype=ct.c_void_p
    lib.mq_mmg_counts.argtypes=[ct.c_void_p,ip];lib.mq_mmg_counts.restype=ct.c_int
    lib.mq_mmg_copy.argtypes=[ct.c_void_p,dp,ip,ip,ip,ip];lib.mq_mmg_copy.restype=ct.c_int
    lib.mq_mmg_free.argtypes=[ct.c_void_p]
    origin=np.min(points,axis=0);scale=float(np.ptp(points,axis=0).max())
    if not np.isfinite(scale) or scale<=0:raise ValueError('Nondegenerate finite mesh required')
    p=np.ascontiguousarray((points-origin)/scale,dtype=np.float64)
    c=np.ascontiguousarray(cells+1,dtype=np.int32);faces=np.ascontiguousarray(facets+1,dtype=np.int32)
    labels=np.unique(regions);r=np.ascontiguousarray(np.searchsorted(labels,regions)+1,dtype=np.int32)
    m=np.ascontiguousarray(markers,dtype=np.int32)
    sizing=np.array([(v or 0.)/scale for v in (options.minimum_edge_m,options.maximum_edge_m,options.uniform_edge_m,options.hausdorff_m)]+[options.gradation,options.feature_angle_degrees],np.float64)
    # MMG's native -m parameter is decimal MB; our public budget is binary MiB.
    native_memory_mb=options.memory_mib*1048576//1000000
    if native_memory_mb>np.iinfo(np.int32).max:raise ValueError('Memory budget exceeds MMG bridge integer range')
    flags=np.array([native_memory_mb,options.preserve_surface,options.optimize_only,options.open_boundaries],np.int32)
    err=ct.create_string_buffer(2048)
    def d(a):return a.ctypes.data_as(dp)
    def i(a):return a.ctypes.data_as(ip)
    handle=lib.mq_mmg_run(len(p),d(p),len(c),i(c),i(r),len(faces),i(faces),i(m),d(sizing),i(flags),err,len(err))
    if not handle:raise RuntimeError(err.value.decode(errors='replace'))
    try:
        counts=np.zeros(3,np.int32)
        if not lib.mq_mmg_counts(handle,i(counts)):raise RuntimeError('MMG returned unsupported volume cells')
        np_,nt,nf=map(int,counts)
        if not 0<np_<=options.maximum_nodes or not 0<nt<=options.maximum_cells or nf<0 or nf>4*nt:raise RuntimeError('MMG output exceeds hard node/cell budget')
        p=np.empty((np_,3),np.float64);c=np.empty((nt,4),np.int32);r=np.empty(nt,np.int32)
        faces=np.empty((nf,3),np.int32);m=np.empty(nf,np.int32)
        if not lib.mq_mmg_copy(handle,d(p),i(c),i(r),i(faces),i(m)):raise RuntimeError('MMG output copy failed')
        if np.any((r<1)|(r>len(labels))):raise ArithmeticError('MMG changed material references')
        return p*scale+origin,c-1,labels[r-1],faces-1,m
    finally:lib.mq_mmg_free(handle)


def remesh(mesh,*,options=None,workspace=None):
    """Adapt one existing Tet4 mesh; source arrays remain unchanged.

    phase volume gates are measured against this input mesh, not against an
    unknown underlying continuous surface. Surface tags are retained only if
    MMG returns every original tag; arbitrary BC interpretation stays explicit.
    """
    from .voxel import external_path
    from .mmg_backend import identity,library_path
    o=options or RemeshOptions()
    if not isinstance(o,RemeshOptions):raise TypeError('RemeshOptions required')
    if not isinstance(mesh,TetGenMesh):raise TypeError('Load a TetGenMesh with actual facets and markers')
    p=np.asarray(mesh.points);c=np.asarray(mesh.cells);r=np.asarray(mesh.phases)
    f=np.asarray(mesh.facets);m=np.asarray(mesh.facet_markers)
    if any(a.dtype.kind not in 'iu' for a in (c,r,f,m)):raise ValueError('Integer connectivity, material references and markers required')
    if len(c)>(o.maximum_input_cells or o.maximum_cells) or len(p)>o.maximum_nodes:raise ValueError('Input exceeds declared MMG resource budget')
    if p.ndim!=2 or p.shape[1]!=3 or c.ndim!=2 or c.shape[1]!=4 or f.ndim!=2 or f.shape[1]!=3:raise ValueError('Tet4 and triangular surfaces required')
    if len(r)!=len(c) or len(m)!=len(f) or np.any(m<1):raise ValueError('One material per cell and positive surface tags required')
    if not np.all(np.isfinite(p)) or np.min(c)<0 or np.max(c)>=len(p) or np.min(f)<0 or np.max(f)>=len(p):raise ValueError('Invalid coordinates/connectivity')
    if len(p)>=np.iinfo(np.int32).max or len(c)>=np.iinfo(np.int32).max:raise ValueError('MMG bridge index range exceeded')
    if o.isolated:
        import h5py
        from dataclasses import replace
        root=external_path(workspace or os.environ.get('MORQEX_WORKSPACE',str(Path(tempfile.gettempdir())/'morqex')))
        cache=root/'cache/mmg_calls';cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache,prefix='remesh_') as tmp:
            source=Path(tmp)/'input.h5';target=Path(tmp)/'output.h5'
            with h5py.File(source,'w') as h:
                for key,a in [('p',p),('c',c),('r',r),('f',f),('m',m)]:h.create_dataset(key,data=a)
                h.attrs.update(options=json.dumps(asdict(replace(o,isolated=False))),report=json.dumps(mesh.report))
            env=os.environ.copy();env['MORQEX_MMG_LIBRARY']=str(library_path(workspace))
            env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
            try:result=subprocess.run([sys.executable,'-m','morqex.modeling.native_geometry_worker','mmg',str(source),str(target)],env=env,capture_output=True,text=True,timeout=o.timeout_seconds)
            except subprocess.TimeoutExpired as e:raise RuntimeError('MMG remeshing exceeded time budget; no output accepted') from e
            if result.returncode or not target.exists():raise RuntimeError('Isolated MMG failed; no output accepted: '+(result.stderr or result.stdout)[-2000:])
            with h5py.File(target) as h:
                out=TetGenMesh(h['mesh/points_m'][:],h['mesh/tetrahedra'][:],h['mesh/phase'][:],facets=h['mesh/facets'][:],facet_markers=h['mesh/facet_markers'][:],report=json.loads(h.attrs['report']))
            out.report['native_process_isolated']=True
            return out
    start=time.perf_counter();before,volumes=tetra_quality(p,c)
    input_volumes={str(k):float(volumes[r==k].sum()) for k in np.unique(r)}
    del volumes  # do not retain one FP64 entry per original cell during MMG
    points,cells,regions,facets,markers=_native(p,c,r,f,m,o,workspace)
    # Canonical local numbering changes no physical tetrahedron.
    for start_ in range(0,len(cells),50000):
        block=cells[start_:start_+50000];v=points[block]
        reverse=np.linalg.det(v[:,1:]-v[:,:1])<0;block[reverse]=block[reverse][:,[1,0,2,3]]
    quality,v=tetra_quality(points,cells)
    if set(map(int,np.unique(regions)))!=set(map(int,np.unique(r))):raise ArithmeticError('MMG deleted an input material')
    actual={str(k):float(v[regions==k].sum()) for k in np.unique(regions)}
    errors={k:abs(actual[k]/value-1) for k,value in input_volumes.items()}
    if max(errors.values())>o.maximum_phase_volume_error:raise ArithmeticError('MMG phase volume tolerance exceeded: '+str(errors))
    if set(map(int,np.unique(markers)))!=set(map(int,np.unique(m))):raise ArithmeticError('MMG changed or omitted an interface/boundary tag')
    from .exchange import _facet_owners
    owners,_=_facet_owners(cells,facets,len(points))
    input_owners,_=_facet_owners(c,f,len(p))
    if not np.array_equal(_tag_pairs(r,m,input_owners),_tag_pairs(regions,markers,owners)):
        raise ArithmeticError('MMG reassigned a surface tag to a different physical material pair')
    if np.any((owners[:,1]>=0)&(regions[owners[:,0]]==regions[np.maximum(owners[:,1],0)])) and not o.open_boundaries:
        raise ArithmeticError('Unexpected same-material internal marked surface')
    # Count all unlike-material/exterior faces independently of MMG's surface
    # list. This detects omitted interfaces as well as nonmanifold faces.
    from .imports import material_interfaces
    complete,_,_=material_interfaces(cells,regions)
    if len(complete)!=len(facets) and not o.open_boundaries:raise ArithmeticError('MMG interface surface list is incomplete')
    topology_before=topology_after=None
    if o.check_surface_topology:
        topology_before=_surface_signature(c,r,f,input_owners);topology_after=_surface_signature(cells,regions,facets,owners)
        if topology_before!=topology_after:raise ArithmeticError('MMG changed checked surface topology invariants: '+str((topology_before,topology_after)))
    input_geometry=mesh.report.get('geometry',{})
    # Names and material-pair meanings survive adaptation. The input mesher's
    # slab/cap or voxel-fidelity contracts do not automatically apply to MMG's
    # new cells; retain them explicitly as input provenance only.
    geometry={key:input_geometry[key] for key in ('region_names','interface_pairs','selection_groups') if key in input_geometry}
    geometry['surface_method']='MMG-adapted material interfaces'
    report={'mesher':'MMG5.8.0, source-integrated optional remeshing','backend':identity(),
        'nodes':len(points),'tetrahedra':len(cells),'quality':quality,'quality_limits_satisfied':None,
        'quality_request':'measured MMG adaptation; no TetGen q-bound implied',
        'input_quality':before,'input_nodes':len(p),'input_tetrahedra':len(c),
        'region_volumes_m3':actual,'phase_volume_errors_relative_to_input_mesh':errors,
        'options':asdict(o),'seconds':time.perf_counter()-start,'geometry':geometry,'input_geometry':input_geometry,
        'surface_topology_before':topology_before,'surface_topology_after':topology_after,
        'checks':{'positive_cells':True,'all_materials_retained':True,'all_surface_tags_retained':True,'actual_face_ownership':True,'complete_material_interfaces':not o.open_boundaries},
        'scope':'geometric mesh validation only; not a PDE convergence or source voxel fidelity certificate'}
    return TetGenMesh(points,cells,regions,facets=facets,facet_markers=markers,report=report)


def _worker_main(source,target):
    import faulthandler,h5py
    faulthandler.enable()
    with h5py.File(source) as h:
        mesh=TetGenMesh(h['p'][:],h['c'][:],h['r'][:],facets=h['f'][:],facet_markers=h['m'][:],report=json.loads(h.attrs['report']))
        o=RemeshOptions(**json.loads(h.attrs['options']))
    remesh(mesh,options=o).save_hdf5(target)


def remesh_artifact(artifact,path,*,options=None,workspace=None):
    """Isolate once and load the full input only inside the native worker.

    Unlike remesh(TetGenMesh), the parent never holds duplicate node/cell arrays.
    maximum_input_cells and maximum_cells separately bound input/output work.
    """
    from dataclasses import replace
    from .meshing import MeshArtifact
    from .voxel import external_path
    from .mmg_backend import library_path
    import h5py
    if not isinstance(artifact,MeshArtifact):raise TypeError('MeshArtifact required')
    o=options or RemeshOptions()
    if not isinstance(o,RemeshOptions):raise TypeError('RemeshOptions required')
    if not o.isolated:raise ValueError('remesh_artifact always isolates once; use remesh(TetGenMesh) for in-process execution')
    artifact=MeshArtifact.load(artifact.hdf5)
    if artifact.tetrahedra>(o.maximum_input_cells or o.maximum_cells) or artifact.nodes>o.maximum_nodes:raise ValueError('Input exceeds declared MMG resource budget')
    target=external_path(path);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.with_suffix('.xdmf').exists():raise FileExistsError(target)
    pending=target.with_name(target.stem+'.pending.h5')
    if pending.exists():raise FileExistsError('Interrupted MMG artifact exists: '+str(pending))
    root=external_path(workspace or os.environ.get('MORQEX_WORKSPACE',str(Path(tempfile.gettempdir())/'morqex')))
    cache=root/'cache/mmg_calls';cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache,prefix='artifact_') as tmp:
        config=Path(tmp)/'options.json';config.write_text(json.dumps(asdict(replace(o,isolated=False))))
        env=os.environ.copy();env['MORQEX_MMG_LIBRARY']=str(library_path(workspace))
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
        try:result=subprocess.run([sys.executable,'-m','morqex.modeling.native_geometry_worker','mmg-artifact',str(artifact.hdf5),str(pending),str(config)],
            env=env,capture_output=True,text=True,timeout=o.timeout_seconds)
        except subprocess.TimeoutExpired as e:raise RuntimeError('MMG artifact worker exceeded time budget; no mesh accepted') from e
        if result.returncode or not pending.exists():raise RuntimeError('MMG artifact worker failed; no mesh accepted: '+(result.stderr or result.stdout)[-2000:])
    with h5py.File(pending,'r+') as h:
        report=json.loads(h.attrs['report']);report.update(native_process_isolated=True,
            input_loading='HDF5 loaded only inside isolated worker; no parent node/cell copy',input_artifact=str(artifact.hdf5),options=asdict(o))
        h.attrs['report']=json.dumps(report)
    from xml.sax.saxutils import escape
    text=pending.with_suffix('.xdmf').read_text().replace(escape(pending.name)+':',escape(target.name)+':')
    pending.rename(target);target.with_suffix('.xdmf').write_text(text);pending.with_suffix('.xdmf').unlink()
    return MeshArtifact.load(target)


def _artifact_worker(source,target,config):
    import h5py,faulthandler
    faulthandler.enable()
    with h5py.File(source) as h:
        mesh=TetGenMesh(h['mesh/points_m'][:],h['mesh/tetrahedra'][:],h['mesh/phase'][:],
            facets=h['mesh/facets'][:],facet_markers=h['mesh/facet_markers'][:],report=json.loads(h.attrs['report']))
    o=RemeshOptions(**json.loads(Path(config).read_text()))
    remesh(mesh,options=o).save_hdf5(target)


if __name__=='__main__':_worker_main(sys.argv[1],sys.argv[2])
