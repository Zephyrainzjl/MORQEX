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

[Complex CT, dense256³ electrodes, cell layers and exchange units /
复杂CT、256³电极、分层与单位](tutorials/07_complex_ct_layers_and_exchange.md)

中文教程：[几何与体素](tutorials/01_geometry_and_voxels.md#zh) ·
[有限元网格](tutorials/02_finite_element_meshes.md#zh) ·
[集成框架与扩展](tutorials/03_integrated_geometry_backends.md#zh)。

[Notebook实时二维/三维可视化](tutorials/04_notebook_visualization.md) ·
[Executable notebook / 可执行示例](tutorials/notebooks/geometry_workflow.ipynb)。

## Adaptive geometry workflow / 自适应几何流程

[Fixed-grid smoothing, explicit resolution reduction and implicit adaptive FEM /
原尺寸平滑、独立降低分辨率与隐式自适应网格](tutorials/08_fixed_grid_smoothing_and_implicit_meshes.md)
introduces separate preparation and mesh-density controls, feature-size advice,
phase-volume/IoU/contact checks, live progress/ETA and checkpointed TOML runs.
平滑保持原数组大小、间距和物理范围；降低分辨率另用 `lower_resolution`。
The optional source-integrated CGAL6.0.1 Mesh_3 backend is compiled explicitly;
its per-component licenses and measured validation limits remain visible.


`morqex.modeling` imports original TIFF segmentation, generates irregular
granular/rod/platelet/agglomerate/bicontinuous electrodes, and stacks independent
electrode/electrolyte/separator/collector regions. COMSOL export accepts
`length_unit='m'|'cm'|'mm'|'um'|'nm'` and writes actual named domain selection
objects. Internal geometry and HDF5 always use SI metres; software imports must
use the selected export unit. VTK shared-interface extraction and conforming
slab assembly are optional alternatives for large multi-label structures.
The two supplied TIFF and COMSOL reference files are independent geometries.
See the linked bilingual tutorial for contracts, licenses and validation limits.

For the **same unchanged voxel model**, `MeshResolution` and
`controlled_tetgen` independently control surface approximation and maximum
Tet4 volume. Choose coarse/balanced/fine/very_fine or explicit SI size limits.
The actual element count and source checksum are recorded; a requested density
is not an exact count or a certified PDE-error bound.

An existing conforming Tet4 mesh can also be adapted through the optional,
source-integrated **MMG5.8.0** backend (`remesh`, `RemeshOptions`). SI size and
Hausdorff requests are independent of display resolution. Material references,
boundary tags, phase volumes and surface topology invariants are checked;
these checks do not certify arbitrary topology equivalence or PDE accuracy.
See the [same bilingual tutorial](tutorials/07_complex_ct_layers_and_exchange.md).

```text
Shapes / statistical recipes / TIFF / user ZYX region arrays
                            ↓
                Complete VoxelModel (ZYX, SI)
   ├─ Same-grid smoothing → PreparedGeometry → implicit CGAL FEM
   ├─ Geometric residual → weighted ACVD → adaptive TetGen FEM
   ├─ Exact voxel PLC / conforming partitions → TetGen FEM
   └─ Explicit lower_resolution → fewer voxels, same physical box
                            ↓
          HDF5/XDMF · software exchange · actual mesh HTML
```

The separate `implicit_adaptive` route starts from a complete `VoxelModel`,
optionally regularizes interfaces **on the same grid**, and builds a continuous
multiphase CGAL volume mesh. Mesh density is independent of the saved prepared
geometry. `lower_resolution` explicitly changes sampling and remains a separate
choice. All quality, phase-volume and topology-change reports are visible.
另一路线先在原尺寸体素上平滑，再生成连续多相体网格；同一准备几何可以调网格粗细。
降低分辨率独立控制，默认不进行。几何驱动自适应不等于物理解误差认证。

The `gemsim_adaptive` API reuses the original source-integrated weighted ACVD
kernel and reconstructs a nonuniform surface before calling TetGen. The old
voxel Hex8/Tet4, streamed voxel and raw PLC routes remain independent choices.
新入口先重构疏密变化的曲面，再生成贴体体网格。旧规则体素及原始PLC路线保留为独立接口。
This is geometric adaptation, not a certified PDE-error estimator.
这里的自适应是几何驱动，不等于严格的物理解误差估计。

For a large **unchanged voxel segmentation**, the additional
`partitioned_tetgen(..., surface_method='voxel_aligned')` route uses canonical
integer-lattice cuts, retains material jumps across partitions and checks every
output cell centroid against the original labels. It avoids cut-induced short
edges without smoothing the source geometry. This exact stair-step route and
the smooth adaptive routes remain separate, with independent shape checks.
大体素可选精确体素边界分区：原数组不改动，计算切面上的真实材料界面保留，逐单元中心
核对材料。精确阶梯与光滑自适应分别保留；体积守恒不能代替形状验收或物理收敛。
See the [full bilingual guide](tutorials/07_complex_ct_layers_and_exchange.md).

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

For optional MMG compilation add `MORQEX_BUILD_MMG=1` (CMake and C/C++ compiler
required), or run `python -m morqex.modeling build-mmg --workspace /external`.
MMG is LGPL3+ and remains a separate shared library. VTK is needed only for
the optional Surface Nets route, not for the original TetGen/ACVD workflow.

## Quick start / 快速开始

```python
from pathlib import Path
from morqex.modeling import (
    coated_electrode, partitioned_tetgen, export_comsol,
)
work = Path('/home/zephyrain/lw/morph_workspace/microelectrode_gallery/my_model')
work.mkdir(parents=True,exist_ok=True)
model = coated_electrode('rods',size=256,spacing=0.5e-6,seed=20261127,
    active_fraction=0.5,cbd_thickness_voxels=3,pore_buffer_voxels=7)
model.save(work/'geometry.h5')
artifact = partitioned_tetgen(model,work/'parts',content_addressed=True,
    surface_method='voxel_aligned',slab_depth=32)
export_comsol(artifact,work/'whole.mphtxt',length_unit='um',
    region_names=model.region_names)
model.viewer(maximum_3d_size=64).save_html(work/'voxels.html')
artifact.adaptive_viewer(maximum_embedded_faces=450000,
    node_cache_bytes=256*2**20).save_html(work/'actual_mesh.html')
print(artifact.hdf5)  # complete volume; no display downsampling of the solve
```

Run this whole256³ example under the external reference-memory profile below.
It uses the exact voxel-boundary route exercised by the complete-gallery
qualification. Smooth GEM/ACVD reconstruction remains available through
`adaptive_mesh`, with the separate examples and controls in the tutorials.
Inspect `validation.json` in the returned mesh directory: a completed mesh
does not imply that its requested radius-edge or dihedral bounds passed.
这个完整256³例子应在下方资源限制器的`--reference-memory`配置下运行；其精确边界路线
已用于完整图库。光滑GEM/ACVD仍是独立接口。检查实际网格目录中的`validation.json`，
不能将成功输出等同于形状请求达标。显示LOD不改变完整计算数据。

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
python scripts/run_local.py --workspace "$WORK" --label geometry_run --reference-memory -- \
  "$PY" /absolute/path/my_geometry_recipe.py
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
under `../morph_workspace/microelectrode_gallery/nmc_complex_gallery/`;
open its `index.html` for the independent reference, full NMC TIFF, five dense
256³ native structures, layered cell, exact mesh sections and software exports.
The earlier ten-family galleries remain in `atlas256/` and
`adaptive_gallery256_gemsim/`; they exercise different mesh workflows.

The fixed-grid conditioning and continuous-domain FEM results are in
`../morph_workspace/microelectrode_gallery/implicit_geometry/index.html`.
Open `conditioning_study/rods/index.html` under that workspace to compare four
smoothing strengths on unchanged256³ arrays against the separate128³
resolution-reduction operation. Complete mesh checkpoints and measured
quality/phase-volume reports accompany the HTML. The synthetic256³ recipe is
[geometry_implicit.toml](configs/geometry_implicit.toml); the supplied full TIFF
recipe is [geometry_nmc_implicit.toml](configs/geometry_nmc_implicit.toml).
Both are documented in the [bilingual preparation and FEM tutorial](tutorials/08_fixed_grid_smoothing_and_implicit_meshes.md).

原尺寸平滑与自适应有限元的新图库位于上述 `implicit_geometry/index.html`。
`conditioning_study/rods/index.html` 可对比四档同尺寸256³平滑与独立128³降分辨率。
默认采用适中精度；原始数组、完整网格和实际几何/单元质量变化分别保存。

Third-party source provenance, copyright and licenses are retained in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and the corresponding source
directories. MORQEX does not rename established scientific methods as new
methods or remove upstream attribution.
