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
[TetGen + COMSOL / 原生编译与导出](tutorials/05_tetgen_and_comsol.md) ·
[GEMsim adaptive / 自适应重网格](tutorials/06_gemsim_adaptive_meshing.md)

中文教程：[几何与体素](tutorials/01_geometry_and_voxels.md#zh) ·
[有限元网格](tutorials/02_finite_element_meshes.md#zh) ·
[集成框架与扩展](tutorials/03_integrated_geometry_backends.md#zh)。

[Notebook实时二维/三维可视化](tutorials/04_notebook_visualization.md) ·
[Executable notebook / 可执行示例](tutorials/notebooks/geometry_workflow.ipynb)。

## Adaptive geometry workflow / 自适应几何流程

```text
Shapes / statistical recipes / user region arrays
                   ↓
          Complete VoxelModel (ZYX, SI)
                   ↓
  Multiresolution geometric residual → weighted ACVD
                   ↓
    Checked common or nested material interfaces
                   ↓
      TetGen nonuniform unstructured volume mesh
                   ↓
       HDF5/XDMF · COMSOL · actual mesh HTML
```

The `gemsim_adaptive` API reuses the original source-integrated weighted ACVD
kernel and reconstructs a nonuniform surface before calling TetGen. The old
voxel Hex8/Tet4, streamed voxel and raw PLC routes remain independent choices.
新入口先重构疏密变化的曲面，再生成贴体体网格。旧规则体素及原始PLC路线保留为独立接口。
This is geometric adaptation, not a certified PDE-error estimator.
这里的自适应是几何驱动，不等于严格的物理解误差估计。

## Install optional native kernels / 可选编译

Python3.11+ and an external environment are required. To compile the integrated
TetGen and weighted ACVD sources during a normal pip installation:
使用源码外环境，并可在pip安装时显式编译两个内核：

```bash
cd /home/zephyrain/lw/MORQEX
export MORQEX_WORKSPACE=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
python -m venv --system-site-packages "$MORQEX_WORKSPACE/runtime"
PY="$MORQEX_WORKSPACE/runtime/bin/python"
MORQEX_BUILD_TETGEN=1 MORQEX_BUILD_ACVD=1 \
  "$PY" -m pip install '.[adaptive-meshing,notebook]'
```

Omit both flags for a Python-only installation. Cython/NumPy build headers are
requested only for the opt-in ACVD/legacy geometry builds. No PyVista/VTK is
required. Native editable installs are rejected to keep binaries out of `src`.
不设置开关则不编译；构建依赖按需安装，二进制、构建目录及大型输出均留在源码外。

## Quick start / 快速开始

```python
from pathlib import Path
from morqex.modeling import (
    resolved_electrode, adaptive_mesh, AdaptiveMeshOptions, export_comsol,
)
work = Path('/home/zephyrain/lw/morph_workspace/microelectrode_gallery/my_model')
work.mkdir(parents=True,exist_ok=True)
model = resolved_electrode('rods',size=256,spacing=0.5e-6,seed=20261127,
                           cbd_thickness_voxels=4,minimum_feature_voxels=4)
model.save(work/'geometry.h5')
result = adaptive_mesh(model,regions=[0,1,2],
                       options=AdaptiveMeshOptions.precision('balanced'))
artifact = result.save_hdf5(work/'adaptive.h5')
export_comsol(result,work/'adaptive.mphtxt',region_names=model.region_names)
result.adaptive_viewer().save_html(work/'adaptive.html')
print(result.report)
```

Coarse/balanced/fine presets control geometric reconstruction independently of
the original voxel size. New resolved-feature recipes cover granular, rods,
platelets, dendritic, hollow, agglomerates, fibres, bicontinuous, lamellar and
channels. All three material domains can be meshed. These are synthetic
geometries with explicit coating/bridge assumptions, not calibrated CT data.
新配方直接生成十类三维结构，粗/中/细重构与体素分辨率独立；结构是具有明确包覆/桥接假设
的合成几何，不冒充真实CT或已经校准的电极。

User arrays remain supported via `from_regions`. General triple-junction data
uses shared-interface constraints; explicitly nested material domains can use
separated contours. See the full [bilingual tutorial](tutorials/06_gemsim_adaptive_meshing.md)
for custom density functions, TOML, units, all options, quality checks, exact
volume sections and the ten256³ gallery commands.
用户数组、一般三相交界、显式包覆结构、自定义密度及全部参数见同页双语教程。

For an exact voxel-reference discretization, retain the original API:
需要精确保留体素台阶和父体素映射时，使用原参考路线：

```python
# model.tetrahedralize(regions=[1,2])                 # small in memory
# model.mesh_to('/external/voxel.h5',regions=[1,2])  # complete streamed volume
# model.tetgen(regions=[1,2],surface_method='voxel_plc')  # raw shared PLC
```

Adaptive tetrahedra have no unique parent voxel; conservative field transfer
requires intersections/integration, not a centroid-label guess. Geometry and
COMSOL read-back checks are distinct from mechanical/electrochemical validation.
COMSOL application import is untested without an installed COMSOL runtime.
自适应单元没有唯一父体素，守恒场传递需要交叠积分；几何导出验收不等于力学/电化学验收，
本机没有COMSOL，因此未宣称实际软件导入通过。

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
tutorials/           bilingual module tutorials and API reference
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
under `../morph_workspace/microelectrode_gallery/adaptive_gallery256_gemsim/`.
The earlier voxel-reference atlas remains in `atlas256/`; it is a different
mesh method, not the adaptive gallery.

Third-party source provenance, copyright and licenses are retained in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and the corresponding source
directories. MORQEX does not rename established scientific methods as new
methods or remove upstream attribution.
