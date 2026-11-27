"""Load/build the original source-integrated GEMsim weighted ACVD kernel.

All generated C++/objects/libraries remain in an external, ABI-keyed cache.
"""
import hashlib,importlib,importlib.util,importlib.metadata,os,sys,sysconfig
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).parents[1]/'microstructure/mesh/pyacvd/cython/_clustering.pyx'


def identity():
    h=hashlib.sha256(SOURCE.read_bytes())
    h.update(str((sysconfig.get_platform(),sysconfig.get_config_var('SOABI'),np.__version__)).encode())
    return h.hexdigest()


def build_acvd(workspace):
    from .voxel import external_path
    from setuptools import Distribution,Extension
    from Cython.Build import cythonize
    folder=external_path(workspace)/'build/acvd'/identity()[:20]
    folder.mkdir(parents=True,exist_ok=True)
    extension=Extension('morqex._acvd_native',[str(SOURCE)],include_dirs=[np.get_include()],
        language='c++',extra_compile_args=['/O2'] if os.name=='nt' else ['-O2'])
    modules=cythonize([extension],build_dir=str(folder/'generated'),compiler_directives={'language_level':3})
    distribution=Distribution({'name':'morqex-acvd','ext_modules':modules})
    command=distribution.get_command_obj('build_ext');command.build_lib=str(folder/'lib')
    command.build_temp=str(folder/'objects');command.inplace=False;command.parallel=1
    distribution.run_command('build_ext')
    return Path(command.get_ext_fullpath('morqex._acvd_native')).resolve()


def load_acvd(workspace=None):
    name='morqex._acvd_native'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.find_spec(name)
    if spec is None:
        files=[]
        try:
            d=importlib.metadata.distribution('morqex-research')
            files=[Path(d.locate_file(f)) for f in d.files or [] if f.name.startswith('_acvd_native') and f.suffix in ('.so','.pyd')]
        except importlib.metadata.PackageNotFoundError:pass
        workspace=workspace or os.environ.get('MORQEX_WORKSPACE')
        if not files and workspace:
            files=list((Path(workspace)/'build/acvd'/identity()[:20]/'lib/morqex').glob('_acvd_native*'))
        if len(files)!=1:raise RuntimeError('Weighted ACVD is not compiled. Enable MORQEX_BUILD_ACVD=1 at pip install, or call build_acvd(workspace).')
        spec=importlib.util.spec_from_file_location(name,files[0])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def clustering_class(workspace=None):
    # The original Python driver imports this relative kernel. Register the
    # ABI-matching module under that name, retaining the original driver code.
    kernel=load_acvd(workspace)
    sys.modules['morqex.microstructure.mesh.pyacvd._clustering']=kernel
    return importlib.import_module('morqex.microstructure.mesh.pyacvd.clustering').Clustering
