"""Optional integrated Cython geometry build; all generated files stay external."""
import os
import hashlib
from pathlib import Path
from setuptools import Extension, setup

root = Path(__file__).resolve().parent
workspace = Path(os.environ.get("MORQEX_WORKSPACE", root.parent / "morph_workspace")).resolve()
if workspace == root or root in workspace.parents:
    raise RuntimeError("Build workspace must be outside source")
# Source snapshots deliberately normalize mtimes. A shared timestamp-based
# build/lib can silently reuse old files, so isolate by actual package content.
signature=hashlib.sha256()
for path in [root/'setup.py',root/'pyproject.toml',*sorted((root/'src').rglob('*'))]:
    if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
    signature.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes()+b'\0')
build = workspace / "build/packages" / signature.hexdigest()[:24]
metadata = build / "metadata"
metadata.mkdir(parents=True, exist_ok=True)
extensions = []
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

setup(ext_modules=extensions, options={
    "build": {"build_base": str(build / "python")},
    "egg_info": {"egg_base": str(metadata)},
    "bdist_wheel": {"bdist_dir": str(build / "wheel")},
})

