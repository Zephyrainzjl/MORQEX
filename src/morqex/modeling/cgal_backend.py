"""Explicit optional CGAL Mesh_3 source build; no numerical imports here."""
from pathlib import Path
import hashlib,importlib.util,json,os,shutil,subprocess,sys,sysconfig,time
LOCAL=Path(__file__).resolve().parent
VENDOR=LOCAL.parent/'vendor/cgal'


def identity():
    digest=hashlib.sha256(Path(__file__).read_bytes())
    for root in (VENDOR,LOCAL/'native_cgal'):
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix!='.pyc':digest.update(str(p.relative_to(root)).encode()+b'\0'+p.read_bytes())
    return {'version':'6.0.1','source_sha256':digest.hexdigest(),'abi':3,
        'upstream':'https://github.com/CGAL/cgal/tree/v6.0.1/Mesh_3',
        'license':'Mesh_3 GPL-3.0-or-later; per-header dependency notices retained'}


def build_cgal(workspace,*,dependency_prefix=None,python_extension=False):
    """One compiler job. Requires Boost/Eigen3/GMP/MPFR development headers.

    A prefix is an existing external development installation, not permission
    to install system packages. No dependency is downloaded by this function.
    """
    workspace=Path(workspace).expanduser().resolve();container=LOCAL.parents[1]
    source=container.parent if container.name=='src' and (container.parent/'pyproject.toml').is_file() else container
    if workspace==source or source in workspace.parents:raise ValueError('Build outside MORQEX source')
    prefix=dependency_prefix or os.environ.get('MORQEX_CGAL_DEPENDENCY_PREFIX')
    prefix=None if prefix is None else Path(prefix).expanduser().resolve()
    dep={}
    if prefix:
        for name in ('boost/version.hpp','eigen3/Eigen/src/Core/util/Macros.h','mpfr.h','x86_64-linux-gnu/gmp.h','gmp.h'):
            p=prefix/'include'/name
            if p.is_file():dep[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    depkey=hashlib.sha256(json.dumps(dep,sort_keys=True).encode()).hexdigest()[:12]
    info=identity();key=info['source_sha256'][:20]+'_'+depkey+('_py_'+str(sysconfig.get_config_var('SOABI')) if python_extension else '')
    out=workspace/'build/cgal'/key;out.mkdir(parents=True,exist_ok=True)
    name='morqex_cgal.dll' if os.name=='nt' else 'libmorqex_cgal.dylib' if sys.platform=='darwin' else 'libmorqex_cgal.so'
    library=out/name
    if not library.is_file():
        mirror=out/'source';vendor=mirror/'vendor/cgal';adapter=mirror/'modeling/native_cgal'
        if not vendor.exists():shutil.copytree(VENDOR,vendor)
        if not adapter.exists():shutil.copytree(LOCAL/'native_cgal',adapter)
        command=['cmake','-S',str(adapter),'-B',str(out),'-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG','-DMORQEX_PYTHON_EXTENSION='+('ON' if python_extension else 'OFF'),
            '-DMORQEX_SOURCE_ID='+info['source_sha256'],
            '-DPython_EXECUTABLE='+sys.executable]
        if prefix:command.append('-DMORQEX_CGAL_DEPENDENCY_PREFIX='+str(prefix))
        subprocess.run(command,check=True)
        subprocess.run(['cmake','--build',str(out),'--target','morqex_cgal','--parallel','1'],check=True)
    record=info|{'library':str(library),'dependency_prefix':None if prefix is None else str(prefix),
        'explicit_dependency_header_hashes':dep,'python_extension':python_extension}
    (out/'identity.json').write_text(json.dumps(record,indent=2))
    registry=workspace/'build/cgal'/('registry_python.json' if python_extension else 'registry.json')
    temp=registry.with_name(registry.name+f'.{time.time_ns()}.pending');temp.write_text(json.dumps(record,indent=2));temp.replace(registry)
    return library


def library_path(workspace=None):
    override=os.environ.get('MORQEX_CGAL_LIBRARY')
    if override:
        path=Path(override).resolve()
        if not path.is_file():raise RuntimeError('MORQEX_CGAL_LIBRARY does not exist')
        return path
    installed=importlib.util.find_spec('morqex._cgal_native')
    if installed is not None:return Path(installed.origin)
    work=Path(workspace or os.environ.get('MORQEX_WORKSPACE',str(LOCAL.parents[3]/'morph_workspace')))
    registry=work/'build/cgal/registry.json'
    if registry.is_file():
        record=json.loads(registry.read_text());path=Path(record['library'])
        if record['source_sha256']==identity()['source_sha256'] and path.is_file():return path
    raise RuntimeError('CGAL is not compiled for these sources; call build_cgal(workspace, dependency_prefix=...) explicitly')
