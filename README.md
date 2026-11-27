# MORQEX

**Microstructure modelling, finite-element meshing and multiscale battery research.**

MORQEX brings voxel geometry, region-labelled finite-element meshes and battery
research solvers into one Python source tree. Its geometry API accepts both
parameterized shapes and user-supplied three-dimensional region arrays. The
source-integrated GEM and MicroStructPy modules provide reusable geometry and
mesh capabilities; optional backends retain their original licenses and runtime
requirements.

[Tutorials](tutorials/README.md) · [API reference](tutorials/API.md) ·
[Voxel modelling](tutorials/01_geometry_and_voxels.md) ·
[Finite-element meshing](tutorials/02_finite_element_meshes.md) ·
[Integrated backends](tutorials/03_integrated_geometry_backends.md) ·
[TetGen + COMSOL / 原生编译与导出](tutorials/05_tetgen_and_comsol.md)

中文教程：[几何与体素](tutorials/01_geometry_and_voxels.md#zh) ·
[有限元网格](tutorials/02_finite_element_meshes.md#zh) ·
[集成框架与扩展](tutorials/03_integrated_geometry_backends.md#zh)。

[Notebook实时二维/三维可视化](tutorials/04_notebook_visualization.md) ·
[Executable notebook / 可执行示例](tutorials/notebooks/geometry_workflow.ipynb)。

## Optional native TetGen / 可选原生网格

The integrated TetGen C++ core can be compiled during a regular pip install:
已集成的 TetGen C++ 核心可以在 pip 安装时选择编译：

```bash
export MORQEX_WORKSPACE=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
MORQEX_BUILD_TETGEN=1 python -m pip install '.[modeling,geometry]'
```

Omit the environment flag for a Python-only build. Use an external environment;
native editable (`-e`) builds are rejected to keep binaries out of `src`.
不设该开关则不编译；环境及构建产物均放在源码外，原生可编辑安装会被拒绝。

```python
from morqex.modeling import mesh,export_comsol
# Given a VoxelModel / 已有统一体素对象
# result = mesh(model,method='tetgen',regions=[1,2],surface_method='voxel_plc')
# result.save_hdf5('/external/tetgen.h5')
# export_comsol(result,'/external/tetgen.mphtxt',region_names=model.region_names)
```

Exact shared multiphase PLC, approximate single-phase contours and custom
closed surfaces are distinct choices. Quality and per-material volume are
measured. Native COMSOL mesh export preserves explicit domain/boundary maps.
The [tutorial](tutorials/05_tetgen_and_comsol.md) covers installation, licenses,
TOML, all sizing options, notebook visuals and COMSOL import steps. COMSOL
application import has not been run without an installed COMSOL runtime.
精确共享多相 PLC、单相近似轮廓和自定义面均有独立入口及误差说明；教程提供完整步骤。

## Geometry workflow

```text
Physical shapes / statistical recipes / segmented region arrays
                              │
                       VoxelModel (ZYX)
                              │
           explicit material regions + SI spacing/origin
                              │
             GEM voxel-to-Tet4 / streamed global meshing
                              │
              HDF5 + XDMF + parent-voxel association
                              │
          interactive geometry inspection / research solvers
```

- **One modelling API:** `morqex.modeling` for arrays, shapes, presets, TOML,
  generator extensions and mesh generation.
- **Custom regions:** nonnegative integer IDs with explicit names; active
  material, CBD, pores and additional phases can be represented separately.
- **Irregular presets:** granular, elongated rods, platelets, dendritic
  scaffolds, hollow shells, agglomerates, fibres, bicontinuous networks,
  gradient lamellae and curved channels.
- **Large voxel grids:** directly evaluated256×256×256 geometry; complete
  conforming Tet4 meshes are streamed to disk with compact global shared nodes.
- **Traceable geometry:** physical units, region labels, source metadata and
  element-to-original-voxel mapping remain available in the exported files.
- **Interactive inspection:** offline HTML with whole-domain display LOD,
  original-resolution slices and exact local tetrahedral mesh views.

## Quick start

Python3.11 or newer is required. A minimal scientific environment needs NumPy,
SciPy, h5py, Matplotlib, Pillow, pyquaternion and lsq-ellipse. The optional
geometry, statistics and report dependencies are declared in `pyproject.toml`.
Keep the environment and generated files outside this source directory.

```bash
cd /home/zephyrain/lw/MORQEX
python -m venv --system-site-packages ../morph_workspace/microelectrode_gallery/runtime
../morph_workspace/microelectrode_gallery/runtime/bin/python -m pip install -e '.[modeling,notebook,report]'
```

Create an irregular electrode and generate its full mesh:

```python
from morqex.modeling import electrode

model, source_regions = electrode(
    'rods', size=256, spacing=0.5e-6, seed=20261127,
    cbd_fraction=0.12,
)
mesh = model.mesh_to(
    '/home/zephyrain/lw/morph_workspace/microelectrode_gallery/rods.h5',
    regions=[1, 2], slab_depth=2,
)
print(mesh.xdmf, mesh.nodes, mesh.tetrahedra)
```

Or supply your own region array:

```python
from morqex.modeling import from_regions

model = from_regions(
    my_labels_zyx, spacing=(0.5e-6,)*3,
    region_names={0:'pore', 1:'active material', 2:'CBD'},
)
mesh = model.mesh_to('/external/workspace/custom.h5', regions=[1,2])
```

Open the generated `.xdmf` in ParaView while keeping its `.h5` sibling beside
it. Full256³ meshes can contain tens of millions of elements; use HDF5 slices
or a local mesh view for inspection. Geometry generation and a feasible
full-field mechanical solve are separate resource requirements.

## Resource-controlled execution

Numerical commands should run through the local resource launcher:

```bash
WORK=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
PY=$WORK/runtime/bin/python
python scripts/run_local.py --workspace "$WORK" --label geometry_run -- \
  "$PY" my_geometry_recipe.py
```

Default limits are one CPU thread, low process priority,3GiB process-tree RSS
and an8GiB host memory reserve. The explicit CPU reference profile permits
6GiB RSS while retaining the host reserve. GPU use is opt-in; the geometry
gallery does not require a GPU. All builds, data, meshes and run logs remain
outside MORQEX.

## Source organization

```text
src/morqex/
  modeling/          unified geometry and finite-element API
  generation/        statistical media, particles and geometry adapters
  microstructure/    integrated GEM geometry, voxelization and mesh modules
  mechanics/         reference elasticity and matrix-free solvers
  electrochemistry/  DFN and microstructure coupling research modules
  top_damage/        fatigue, topology and degradation research modules
  visualization/     geometry and computed-field HTML exporters
  vendor/            pinned third-party source with notices and licenses
src/pybamm/          source-integrated electrochemical solver
tutorials/           English module tutorials and API reference
configs/             TOML examples and resource settings
scripts/             generation, checking and reproducibility entry points
tests/               targeted mathematical and software checks
```

The complete public geometry namespaces of the integrated backends are
accessible through `morqex.modeling.framework(name)`. See the
[backend tutorial](tutorials/03_integrated_geometry_backends.md) for available
portals and native prerequisites. Micro_Reconstruction's MATLAB/R source is
included; executing it requires its original runtimes. Python statistical
generation is explicitly identified as a different method.

## Research modules and validation

MORQEX also contains electrochemical, mechanical, thermal and degradation
research interfaces. The [API reference](tutorials/API.md) records those
interfaces and their model assumptions. Source availability and successful
geometry checks do not establish calibrated material predictions, spatial
convergence or long-term battery-life validation.

The workspace was cleared on request before creating the new geometry atlas.
Earlier run outputs and numerical environments are not included in the source
repository and must be regenerated when needed. Current geometry outputs live
under `../morph_workspace/microelectrode_gallery/atlas256/`.

Third-party source provenance, copyright and licenses are retained in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and the corresponding source
directories. MORQEX does not rename established scientific methods as new
methods or remove upstream attribution.
