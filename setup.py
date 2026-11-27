"""Optional integrated Cython geometry build; all generated files stay external."""
import os
import hashlib
import sysconfig
import importlib.metadata
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

root = Path(__file__).resolve().parent
workspace = Path(os.environ.get("MORQEX_WORKSPACE", root.parent / "morph_workspace")).resolve()
if workspace == root or root in workspace.parents:
    raise RuntimeError("Build workspace must be outside source")
# Source snapshots deliberately normalize mtimes. A shared timestamp-based
# build/lib can silently reuse old files, so isolate by actual package content.
signature=hashlib.sha256()
signature.update(repr((sysconfig.get_platform(), sysconfig.get_config_var('SOABI'),
    os.environ.get('MORQEX_BUILD_GEOMETRY'), os.environ.get('MORQEX_BUILD_TETGEN'),os.environ.get('MORQEX_BUILD_ACVD'))).encode())
if any(os.environ.get(k)=='1' for k in ('MORQEX_BUILD_ACVD','MORQEX_BUILD_GEOMETRY')):
    for name in ('numpy','Cython'):
        signature.update((name+':'+importlib.metadata.version(name)).encode())
for path in [root/'setup.py',root/'pyproject.toml',root/'build_backend.py',root/'MANIFEST.in',root/'THIRD_PARTY_NOTICES.txt',*sorted((root/'src').rglob('*'))]:
    if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
    signature.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes()+b'\0')
build = workspace / "build/packages" / signature.hexdigest()[:24]
metadata = build / "metadata"
metadata.mkdir(parents=True, exist_ok=True)
extensions = []
if os.environ.get('MORQEX_BUILD_ACVD')=='1':
    import numpy as np
    from Cython.Build import cythonize
    extension=Extension('morqex._acvd_native',
        ['src/morqex/microstructure/mesh/pyacvd/cython/_clustering.pyx'],
        include_dirs=[np.get_include()],language='c++',extra_compile_args=['/O2'] if os.name=='nt' else ['-O2'])
    # Keep original relative.pyx paths during metadata/sdist discovery. Delay
    # Cython translation to build_ext so generated absolute.cpp paths do not
    # leak into the source manifest or require in-source generated files.
    extensions.append(extension)
if os.environ.get("MORQEX_BUILD_TETGEN") == "1":
    tet = Path("src/morqex/microstructure/mesh/tetgen/cython/tetgen")
    extensions.append(Extension("morqex._tetgen_native",
        ["src/morqex/modeling/native_tetgen/bridge.cpp", str(tet / "tetgen.cxx"), str(tet / "predicates.cxx")],
        include_dirs=[str(tet)], define_macros=[("TETLIBRARY", None), ("MORQEX_PYTHON_EXTENSION", "1")],
        language="c++", extra_compile_args=["/O2", "/std:c++17"] if os.name == "nt" else ["-O2", "-std=c++17"]))
if os.environ.get("MORQEX_BUILD_GEOMETRY") == "1":
    import numpy as np
    from Cython.Build import cythonize
    base = Path("src/morqex/microstructure")
    compile_args = ["/O2"] if os.name == "nt" else ["-O3"]
    def extension(name, sources, include=(), macros=()):
        return Extension("morqex.microstructure." + name, list(map(str, sources)),
                         include_dirs=[np.get_include(), *map(str, include)],
                         define_macros=list(macros), language="c++", extra_compile_args=compile_args)
    extensions.append(extension("mesh._voxel_native", [base / "mesh/_voxel_native.pyx"]))
    for name in ("cy_discrete", "cy_morphology", "cy_geometry"):
        extensions.append(extension("core.cyfunc." + name, [base / "core/cyfunc" / (name + ".pyx")]))
    tet = base / "mesh/tetgen/cython/tetgen"
    extensions.append(extension("mesh.tetgen._tetgen",
        [tet / name for name in ("_tetgen.pyx", "tetgen.cxx", "predicates.cxx", "tetgen_wrap.cxx")],
        include=[tet], macros=[("TETLIBRARY", None)]))
    extensions.append(extension("mesh.pyacvd._clustering", [base / "mesh/pyacvd/cython/_clustering.pyx"]))
    extensions = cythonize(extensions, compiler_directives={"language_level": 3}, build_dir=str(build / "cython"))

class ExternalBuildExt(build_ext):
    def run(self):
        if self.inplace and self.extensions:
            raise RuntimeError('Native editable builds would put binaries in src. Use a non-editable pip install with MORQEX_WORKSPACE outside source, or build_tetgen(workspace).')
        if any(str(source).endswith('.pyx') for extension in self.extensions for source in extension.sources):
            from Cython.Build import cythonize
            translated={e.name:e for e in cythonize(self.extensions,build_dir=str(build/'cython_late'),compiler_directives={'language_level':3})}
            # Cython clones Extension instances. Preserve setuptools' already
            # finalized objects (_needs_stub/ext_map/etc.) and replace only
            # their translated sources, keeping metadata's original.pyx paths.
            for extension in self.extensions:
                extension.sources=translated[extension.name].sources
        super().run()

setup(ext_modules=extensions, cmdclass={"build_ext": ExternalBuildExt}, options={
    "build": {"build_base": str(build / "python")},
    "egg_info": {"egg_base": str(metadata)},
    "bdist_wheel": {"bdist_dir": str(build / "wheel")},
})

