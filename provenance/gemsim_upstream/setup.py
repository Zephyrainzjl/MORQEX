"""Compile original Cython/C++ kernels under the gem namespace on Windows/Linux."""
import os
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
import subprocess
import sys

class CMakeExtension(Extension):
    def __init__(self):
        super().__init__('_gem_echemamr',sources=[])

class NativeBuild(build_ext):
    def build_extension(self,ext):
        if not isinstance(ext,CMakeExtension):
            return super().build_extension(ext)
        import pybind11
        output=Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build=Path(self.build_temp).resolve()/'electrochemistry'
        build.mkdir(parents=True,exist_ok=True)
        source=Path('src/gem/simulate/native').resolve()
        args=['-DCMAKE_BUILD_TYPE=Release',f'-Dpybind11_DIR={pybind11.get_cmake_dir()}',
              f'-DPYTHON_EXECUTABLE={sys.executable}',
              f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}',f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={output}',
              f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={output}']
        if os.environ.get('GEM_CUDA')=='1':args+=['-DAMReX_GPU_BACKEND=CUDA']
        if os.environ.get('GEM_AMREX_SOURCE'):args += ['-DFETCHCONTENT_SOURCE_DIR_AMREX='+os.environ['GEM_AMREX_SOURCE']]
        subprocess.run(['cmake','-S',str(source),'-B',str(build),*args],check=True)
        subprocess.run(['cmake','--build',str(build),'--config','Release','--parallel',os.environ.get('GEM_NATIVE_JOBS','2')],check=True)

base = Path('src/gem')
ext = []
ext.append(Extension('gem.mesh._voxel_native',
    [str(base/'mesh/_voxel_native.pyx')],include_dirs=[np.get_include()],language='c++',
    extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3']))
for name in ('cy_discrete', 'cy_morphology', 'cy_geometry'):
    ext.append(Extension(f'gem.core.cyfunc.{name}',
        [str(base / 'core/cyfunc' / (name + '.pyx'))],
        include_dirs=[np.get_include()], language='c++',
        extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3']))
tet = base / 'mesh/tetgen/cython/tetgen'
ext.append(Extension('gem.mesh.tetgen._tetgen',
    [str(tet / f) for f in ('_tetgen.pyx', 'tetgen.cxx', 'predicates.cxx', 'tetgen_wrap.cxx')],
    include_dirs=[np.get_include(), str(tet)], define_macros=[('TETLIBRARY', None)],
    language='c++', extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3']))
ext.append(Extension('gem.mesh.pyacvd._clustering',
    [str(base / 'mesh/pyacvd/cython/_clustering.pyx')],
    include_dirs=[np.get_include()], language='c++'))
compiled=cythonize(ext,compiler_directives={'language_level':3},build_dir=os.environ.get('GEM_BUILD_DIR','build/cython'))
if os.environ.get('GEM_BUILD_ECHEM')=='1':compiled.append(CMakeExtension())
setup(ext_modules=compiled,cmdclass={'build_ext':NativeBuild})
