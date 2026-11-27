"""Source-built TetGen ABI; no Cython or PyVista requirement for this adapter."""
import ctypes as ct
import hashlib,importlib.util,importlib.metadata,json,os,platform,subprocess,sys,threading,shutil
from pathlib import Path
import numpy as np

NATIVE=Path(__file__).with_name('native_tetgen')
CORE=Path(__file__).parents[1]/'microstructure/mesh/tetgen/cython/tetgen'
_LOCK=threading.RLock()


def identity():
    h=hashlib.sha256(platform.platform().encode())
    h.update(Path(__file__).read_bytes())
    for p in [NATIVE/'bridge.cpp',NATIVE/'CMakeLists.txt',*[CORE/n for n in ('tetgen.cxx','tetgen.h','predicates.cxx')]]:
        h.update(p.name.encode()+p.read_bytes())
    return h.hexdigest()


def build_tetgen(workspace):
    from .voxel import external_path
    workspace=external_path(workspace);folder=workspace/'build/tetgen'/identity()[:20]
    folder.mkdir(parents=True,exist_ok=True)
    # A content-identical checkout/wheel can move. Keep CMake's source path
    # stable rather than reconfiguring a cache against another installation.
    adapter=folder/'source/modeling/native_tetgen';core=folder/'source/microstructure/mesh/tetgen/cython/tetgen'
    if not adapter.exists():shutil.copytree(NATIVE,adapter)
    if not core.exists():shutil.copytree(CORE,core)
    subprocess.run(['cmake','-S',str(adapter),'-B',str(folder),'-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(folder),'--parallel','1'],check=True)
    candidates=list(folder.glob('*morqex_tetgen*'))
    libs=[p for p in candidates if p.suffix in ('.so','.dylib','.dll')]
    if len(libs)!=1:raise RuntimeError('Expected one built TetGen library')
    (folder/'build.json').write_text(json.dumps({'identity':identity(),'library':str(libs[0]),'cpu_threads':1,'version':'1.6.0'}))
    return libs[0]


def library_path(workspace=None):
    override=os.environ.get('MORQEX_TETGEN_LIBRARY')
    if override:
        p=Path(override).resolve()
        if not p.is_file():raise RuntimeError('Configured TetGen library does not exist')
        return p
    spec=importlib.util.find_spec('morqex._tetgen_native')
    if spec and spec.origin:return Path(spec.origin)
    # A source checkout may shadow an installed package. Resolve only the
    # declared installed distribution, never an unrelated system library.
    try:
        dist=importlib.metadata.distribution('morqex-research')
        for f in dist.files or ():
            if f.name.startswith('_tetgen_native') and f.suffix in ('.so','.pyd','.dylib'):
                return Path(dist.locate_file(f))
    except importlib.metadata.PackageNotFoundError:pass
    if workspace is None:workspace=os.environ.get('MORQEX_WORKSPACE')
    if workspace:
        folder=Path(workspace)/'build/tetgen'/identity()[:20]
        files=[p for p in folder.glob('*morqex_tetgen*') if p.suffix in ('.so','.dylib','.dll')]
        if len(files)==1:return files[0]
    raise RuntimeError('TetGen is not compiled. Install with MORQEX_BUILD_TETGEN=1 pip install . or call build_tetgen(workspace) explicitly.')


def execute(points,facets,markers,regions,holes,switches,*,workspace=None,maximum_cells=2000000):
    points=np.ascontiguousarray(points,dtype=np.float64);markers=np.ascontiguousarray(markers,dtype=np.int32)
    sizes=np.full(len(facets),facets.shape[1],np.int32) if isinstance(facets,np.ndarray) and facets.ndim==2 else np.ascontiguousarray([len(f) for f in facets],dtype=np.int32)
    faces=np.ascontiguousarray(facets.reshape(-1) if isinstance(facets,np.ndarray) else np.concatenate(facets),dtype=np.int32)
    regions=np.ascontiguousarray(regions,dtype=np.float64).reshape(-1,5)
    holes=np.ascontiguousarray(holes,dtype=np.float64).reshape(-1,3)
    lib=ct.CDLL(str(library_path(workspace)));dp=ct.POINTER(ct.c_double);ip=ct.POINTER(ct.c_int)
    lib.mq_tetgen_abi.restype=ct.c_int
    if lib.mq_tetgen_abi()!=1:raise RuntimeError('Unsupported TetGen ABI')
    lib.mq_tetgen_run.restype=ct.c_void_p
    lib.mq_tetgen_run.argtypes=[ct.c_int,dp,ct.c_int,ip,ip,ip,ct.c_int,dp,ct.c_int,dp,ct.c_char_p,ct.c_char_p,ct.c_int]
    lib.mq_tetgen_counts.argtypes=[ct.c_void_p,ip]
    lib.mq_tetgen_copy.argtypes=[ct.c_void_p,dp,ip,dp,ip,ip]
    lib.mq_tetgen_free.argtypes=[ct.c_void_p]
    def ptr(a):return a.ctypes.data_as(dp if a.dtype==np.float64 else ip)
    error=ct.create_string_buffer(1024)
    with _LOCK:
        handle=lib.mq_tetgen_run(len(points),ptr(points),len(facets),ptr(faces),ptr(sizes),ptr(markers),
            len(regions),ptr(regions),len(holes),ptr(holes),switches.encode(),error,len(error))
        if not handle:raise RuntimeError(error.value.decode())
        try:
            counts=np.zeros(4,np.int32);lib.mq_tetgen_counts(handle,ptr(counts));n,e,f,k=map(int,counts)
            if e<=0 or e>maximum_cells:raise RuntimeError(f'TetGen returned {e} cells outside explicit output budget {maximum_cells}')
            p=np.empty((n,3));t=np.empty((e,4),np.int32);a=np.empty((e,k));bf=np.empty((f,3),np.int32);bm=np.empty(f,np.int32)
            lib.mq_tetgen_copy(handle,ptr(p),ptr(t),ptr(a),ptr(bf),ptr(bm))
            return p,t,a,bf,bm
        finally:lib.mq_tetgen_free(handle)
