"""Source-integrated optional MMG5.8.0 build and ABI, without numeric imports."""
import hashlib,json,os,subprocess,sys,sysconfig,importlib.util,shutil
from pathlib import Path
LOCAL=Path(__file__).resolve().parent
VENDOR=LOCAL.parent/'vendor/mmg'


def identity():
    signature=hashlib.sha256()
    signature.update(Path(__file__).read_bytes())
    for root in (VENDOR,LOCAL/'native_mmg'):
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix!='.pyc':signature.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes())
    return {'version':'5.8.0','upstream_commit':'4d8232c8aebfed877935d75d4d4a67e850962422',
        'source_sha256':signature.hexdigest(),'license':'LGPL-3.0-or-later','abi':1}


def build_mmg(workspace,*,python_extension=False):
    workspace=Path(workspace).expanduser().resolve();container=LOCAL.parents[1]
    source=(container.parent if container.name=='src' and (container.parent/'pyproject.toml').is_file() else container)
    if workspace==source or source in workspace.parents:raise ValueError('Build outside MORQEX source')
    key=identity()['source_sha256'][:20]+('_py_'+str(sysconfig.get_config_var('SOABI')) if python_extension else '')
    out=workspace/'build/mmg'/key;out.mkdir(parents=True,exist_ok=True)
    library=out/('morqex_mmg.dll' if os.name=='nt' else 'libmorqex_mmg.dylib' if sys.platform=='darwin' else 'libmorqex_mmg.so')
    if not library.exists():
        # CMake embeds source paths in its cache. A wheel/sdist or a frozen
        # snapshot can move, although its content identity is unchanged.
        # Configure against a stable, content-addressed external source copy.
        mirror=out/'source';vendor=mirror/'vendor/mmg';adapter=mirror/'modeling/native_mmg'
        if not vendor.exists():shutil.copytree(VENDOR,vendor)
        if not adapter.exists():shutil.copytree(LOCAL/'native_mmg',adapter)
        # Wheel/snapshot archives need not preserve executable mode bits.
        # Only the external copy is adjusted; upstream file contents stay exact.
        for script in (vendor/'scripts').glob('*.sh'):script.chmod(script.stat().st_mode|0o111)
        cmd=['cmake','-S',str(adapter),'-B',str(out),'-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_POLICY_VERSION_MINIMUM=3.5','-DCMAKE_C_FLAGS_RELEASE=-O2 -DNDEBUG',
            '-DMORQEX_PYTHON_EXTENSION='+('ON' if python_extension else 'OFF'),'-DPython_EXECUTABLE='+sys.executable]
        subprocess.run(cmd,check=True)
        subprocess.run(['cmake','--build',str(out),'--target','morqex_mmg','--parallel','1'],check=True)
    (out/'identity.json').write_text(json.dumps(identity(),indent=2))
    return library


def library_path(workspace=None):
    override=os.environ.get('MORQEX_MMG_LIBRARY')
    if override:
        path=Path(override).resolve()
        if not path.is_file():raise RuntimeError('MORQEX_MMG_LIBRARY does not exist')
        return path
    installed=importlib.util.find_spec('morqex._mmg_native')
    if installed is not None:return Path(installed.origin)
    workspace=Path(workspace or os.environ.get('MORQEX_WORKSPACE',str(LOCAL.parents[3]/'morph_workspace')))
    path=workspace/'build/mmg'/identity()['source_sha256'][:20]/'libmorqex_mmg.so'
    if not path.exists():raise RuntimeError('MMG is not compiled; call build_mmg(workspace) explicitly')
    return path
