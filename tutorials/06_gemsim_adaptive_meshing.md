# GEMsim adaptive geometry and meshing / GEMsim 自适应几何与网格

[教程目录 / Index](README.md) · [API](API.md) · [网格路线比较 / Methods](02_finite_element_meshes.md) · [TetGen与COMSOL](05_tetgen_and_comsol.md)

## 1. What changed / 本次修改的重点

The new `gemsim_adaptive` route reconstructs and redistributes the **surface**
before tetrahedralizing its enclosed domains. It follows GEMsim's residual-weighted
ACVD workflow. It is distinct from a six-tetrahedra-per-voxel split and from
passing every original marching-cubes triangle directly to TetGen.
新路线先做曲面重建和加权重网格，再划分内部体单元。它与“每个体素剖成6个四面体”及
“直接把原始 MC 三角面交给 TetGen”是不同的方法，旧路线仍保留为独立 API。

```text
Complete256³ labelled geometry / 完整三维标签
  → full-resolution shared interfaces or nonintersecting nested contours
  → multiresolution phase residuals / 多层分相几何残差
  → positive, bounded surface density / 有正下限的数值网格密度
  → source-integrated weighted ACVD / 已集成源码的加权 ACVD
  → local topology retention and shared junction constraints
  → region seeds + enclosed-volume checks / 种子及包围体积核对
  → TetGen nonuniform unstructured volume mesh
  → HDF5/XDMF + COMSOL + actual surface/section HTML
```

TetGen's volume elements are still tetrahedra. What changes is their geometry,
connectivity and size distribution: vertices are relocated independently of the
voxel lattice, and a tetrahedron can span multiple original voxels.
TetGen 的体单元仍然是四面体；改变的是节点位置、连接关系及疏密分布。节点不再受体素格点
约束，一个体单元可以跨多个原始体素。表面三角网格与内部四面体网格也不是同一概念。

This is **geometric** adaptation, driven by microstructure complexity and user
mesh-density functions. It does not use stress, diffusion or PDE residuals and
must not be called certified solution-error adaptivity.
这里的“自适应”是几何驱动的网格密度调整，不是应力/扩散方程求解后的残差自适应，
不能宣称已获得严格的物理解误差界。

## 2. Source audit and upstream reuse / 原始流程审计与复用

The actual private GEMsim repository was read using the connected GitHub account.
Verified commit: `6fd789c78d58d49ef645c539414567fc2a2bc171`.
The relevant `src/gem/mesh/elements.py` Git blob is
`1ef3e7ffb75ee50d4311dec8c94c226dc66f869e`.
已通过已连接的 GitHub 账户实际读取原始私有仓库，核实了上述提交及文件版本。

The original chain is `TriMesh.volres_weight` → `sur_optimize` →
`pyacvd.Clustering` → `TetMesh.from_sur`/TetGen. MORQEX directly compiles the
already integrated weighted ACVD Cython/C++ kernel and calls the original
Python `Clustering` driver. The multiresolution median-residual calculation is
ported into bounded-memory NumPy/SciPy/skimage code with physical coordinates.
原始 ACVD 内核和 Python 驱动直接复用；多层中值残差计算移植为分层处理，避免同时保留
多份256³双精度数组，并明确了坐标和单位。

The new safe route additionally checks directed surface orientation and topology, preserves common material
junctions, keeps a positive weight floor and limits extreme numerical density
ratios. It never invokes the legacy fallback that removes skipped surface
faces or deletes low-quality tetrahedra. Those operations can invalidate
material volume or create holes.
新入口增加闭合性/拓扑核对、材料交线约束、权重正下限和密度比限制，不调用旧版失败后
删面、删低质量体单元的分支。这些历史兼容代码保留，但不是新路线的验收依据。

## 3. Optional pip compilation / pip 可选源码编译

```bash
cd /home/zephyrain/lw/MORQEX
export MORQEX_WORKSPACE=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
PY="$MORQEX_WORKSPACE/runtime/bin/python"

# Install API dependencies first / 先安装API依赖
"$PY" -m pip install '.[adaptive-meshing,notebook]'

# Compile only the requested native kernels / 按需编译原生组件
MORQEX_BUILD_TETGEN=1 MORQEX_BUILD_ACVD=1 \
  "$PY" -m pip install --no-cache-dir --no-deps .
```

Base installation does not compile these kernels. The PEP517 backend requests
NumPy headers and Cython only when `MORQEX_BUILD_ACVD=1` or the older
`MORQEX_BUILD_GEOMETRY=1` is enabled. The independent TetGen-only build still
requires neither Cython nor PyVista. No VTK/PyVista is required for this workflow.
默认不编译；只有开启 ACVD/旧 GEM 扩展时才动态申请 Cython 和 NumPy 构建依赖。
单独 TetGen 编译仍不需要 Cython，整个新流程不依赖 PyVista/VTK。

Use an external environment and a regular installation. Native editable
installs are rejected because they would copy binaries into `src`.
环境和构建产物均应放在源码外；原生可编辑安装会向 `src` 放二进制，因此被明确拒绝。
All large meshes, HTML, wheels and generated C++ files stay in the external workspace.
大型网格、HTML、wheel 及生成的 C++ 文件均留在外部工作区。

External explicit build / 已安装后显式外部构建：

```bash
"$PY" -m pip install 'Cython>=3.0'   # only for an explicit ACVD source build
"$PY" -m morqex.modeling build-acvd --workspace "$MORQEX_WORKSPACE"
"$PY" -m morqex.modeling build-tetgen --workspace "$MORQEX_WORKSPACE"
```

```python
import os
from morqex.modeling import build_acvd,build_tetgen,capabilities
work = '/home/zephyrain/lw/morph_workspace/microelectrode_gallery'
os.environ['MORQEX_WORKSPACE'] = work
# Explicit one-job compilation; no automatic compile on import.
# 显式单核编译，导入模块不会偷偷编译。
print(build_acvd(work))
print(build_tetgen(work))
print(capabilities()['weighted_acvd'])
```

## 4. Three precision presets / 三种几何精度预设

```python
from morqex.modeling import AdaptiveMeshOptions
coarse = AdaptiveMeshOptions.precision('coarse')
balanced = AdaptiveMeshOptions.precision('balanced')
fine = AdaptiveMeshOptions.precision('fine')
```

|Preset / 预设|Target surface vertex fraction / 目标表面点比例|Maximum per-phase voxel-volume difference / 分相体积差门槛|
|---|---:|---:|
|coarse / 粗|0.06|6%|
|balanced / 平衡|0.12|3%|
|fine / 细|0.25|1.5%|

These are explicit geometry tolerances and starting fractions. They are not
fixed output counts or guarantees of a given stress error. Topology and volume
checks may locally retain original vertices or retry at a higher fraction.
Actual fractions and input/output counts are recorded in `result.report`.
这些是几何门槛及初始比例，不是固定输出数量或应力误差保证。为了保留拓扑或体积，程序
可能局部保留原始节点，或提高比例重试；实际值记录在报告中。

The original256³ label array remains unchanged during meshing. Increasing
precision changes the reconstructed computational surface and volume mesh,
not the number of source voxels. Coarsening the source grid is a separate,
explicit operation, not an implicit meshing shortcut.
调整精度不修改原始256³标签；源分辨率和有限元网格分辨率是两个独立参数。

## 5. Generate the new ten morphology families / 生成十种新结构

```python
from pathlib import Path
from morqex.modeling import resolved_electrode,adaptive_mesh,AdaptiveMeshOptions
work = Path('/home/zephyrain/lw/morph_workspace/microelectrode_gallery/my_adaptive')
work.mkdir(parents=True,exist_ok=True)
model = resolved_electrode(
    'rods', size=256, spacing=0.5e-6, seed=20261127,
    cbd_thickness_voxels=4,minimum_feature_voxels=4,cbd_bridges=True,
)
model.save(work/'geometry.h5')
result = adaptive_mesh(model,regions=[0,1,2],
                       options=AdaptiveMeshOptions.precision('balanced'))
artifact = result.save_hdf5(work/'mesh.h5')
```

`resolved_electrode` is a **new** geometry generator, separate from the earlier
`electrode` dense/random recipes. It supports96..256 cells per axis; custom
smaller examples use `from_shapes`/`from_regions`. The old generator and all
previous files remain available.
新生成器独立于原来的高密度随机配方，支持96..256；更小的自定义几何使用原语/数组接口。
旧配方和旧文件均保留，不会被偷偷替换为另一种结构。

|Name / 名称|Geometry / 几何|
|---|---|
|`granular`|Jittered irregular ellipsoidal particles with CBD bridges / 不规则颗粒及CBD桥|
|`rods`|Long irregular capsule particles / 长条棒状颗粒|
|`platelets`|Thin tilted irregular platelets / 倾斜不规则片层|
|`dendritic`|Connected branched solid scaffolds / 分枝固体骨架|
|`hollow`|Irregular hollow shells, inner/outer coatings / 空心壳层及内外包覆|
|`agglomerates`|Overlapping primary-shape agglomerates / 多级团聚结构|
|`fibres`|Curved interweaving3D fibres / 三维曲折纤维|
|`bicontinuous`|Irregular gyroid-like bicontinuous field / 双连续隐式结构|
|`lamellar`|Wavy three-dimensional layers / 三维波纹层片|
|`channels`|Solid with curved directional pore channels / 曲折孔道结构|

All are evaluated directly on the requested3D grid, without copying a small
SVE into a larger cube. The Shape/MicroStructPy geometry implementation and
explicit implicit-field recipes are used; no neural network or trained model
participates.
全部在指定三维格点上直接生成，不复制小 SVE。使用集成的 Shape/MicroStructPy 几何与
显式隐式场配方，不调用神经网络或预训练模型。

The recipe applies an explicitly recorded opening/closing operation to impose
resolved feature sizes **before saving the source geometry**. It also creates
CBD coating/optional capsule bridges. This is part of synthetic geometry
generation, not silent smoothing of imported CT. Measured volume fractions,
connectivity and physical dimensions must still be assessed for the intended
material. These are not calibrated CT electrodes or electrochemical predictions.
最小特征的开闭运算发生在合成几何生成阶段并记录参数，不会用于偷偷修改用户 CT。
CBD 包覆和桥接也是明确的几何假设；相比例、连通性及物理适用性仍需结合实际材料评估。

Disconnected enclosed pore components smaller than `minimum_feature_voxels**3`
are filled with CBD during **synthetic generation**, with the exact count stored
as `unresolved_enclosed_pore_voxels_filled_with_cbd`. This is an explicit
resolution policy, not a claim that unresolved pores do not exist physically.
The hollow recipe uses eight large shells so that their inner pores remain
resolved. None of these operations is applied to arrays supplied by `from_regions`.
合成阶段将小于上述分辨率体积的封闭孔隙填为CBD，并记录改变的体素数量；这不是否认实际
材料可能存在微孔。空心配方采用8个大壳层以分辨内孔。用户数组完全不执行这些生成操作。

The declared phase order is `[0,2,1]`: pore→CBD→active. This permits mutually
separated material contours. CBD is an effective material domain; this label
alone does not declare it electronically conducting, ionically impermeable,
or fully electrochemically accessible.
分相顺序表示几何包覆关系，不自动赋予导电、隔离或电化学可达性等物理属性。

## 6. User arrays and two multiphase surface routes / 用户数组及两种多相路径

```python
import numpy as np
from morqex.modeling import from_regions,mesh

# Your own nonnegative integer ZYX segmentation / 用户三维整数分区
labels = np.load('/external/my_regions.npy',allow_pickle=False)
model = from_regions(labels,spacing=(0.4e-6,0.5e-6,0.6e-6),
                     origin=(0.,0.,0.),region_names={0:'pore',1:'active',2:'CBD'})
result = mesh(model,method='gemsim_adaptive',regions=[0,1,2],
    options=AdaptiveMeshOptions(surface_method='shared_interfaces'))
```

### Shared interfaces / 共享界面

`surface_method='shared_interfaces'` starts from one common interface complex.
Faces between the same pair of materials are clustered together; junction
vertices shared by distinct patches are fixed and stitched by common IDs.
Boundary topology, connected components and Euler characteristic are checked.
Disconnected/annular clusters and newly pinched local regions retain original
vertices rather than creating a hole or bridging two materials. Near-flat valence-three stars can be replaced by one supporting triangle only when their boundary cycle and topology remain unchanged. This implements the flat-cusp repair role of GEMsim without deleting volume cells.
共享界面路径为每个材料对构建共同界面，固定并共享交线节点；局部不满足拓扑条件的簇
保留原始分辨率，不通过留下孔洞或添加虚假连接来修复。近共面的三价顶点可用保边界循环的3→1曲面操作修复，同时检查拓扑；不会删掉坏体单元来伪造验收。

It supports general triple junctions, including custom IDs. A heavily
under-resolved segmentation may leave many protected vertices, so this route
can be expensive. It is not a guarantee that arbitrary pathological geometry
fits the workstation budget or defines a well-posed contact model.
支持一般三相交界及自定义编号。但欠分辨率的分割可能保留很多节点，成本较高；不能保证
任意病态几何都可在工作站预算内生成，也不自动解决接触物理。

### Nested contours / 分离的包覆界面

```python
options = AdaptiveMeshOptions(
    surface_method='nested_contours',phase_order=(0,2,1),vertex_fraction=.12,
)
# Region IDs can be different, e.g.(5,11,23). Labels carry no implicit meaning.
# 编号可以不同；数值本身不决定材料物理意义。
```

The outer phase must occupy all box faces; nonadjacent phases require resolved
separation. A full-resolution contour is extracted for every nested union,
then weighted ACVD is performed on each real interface. This avoids overlapping
independent phase contours at triple junctions. Input violating those conditions
is rejected; it must use shared interfaces or be re-segmented with a justified
physical model. `auto` uses nesting only if explicitly declared in the model
metadata/options. Otherwise it uses shared interfaces for multiple labels.
只有明确声明包覆关系、满足外层边界与非相邻相间距条件时，才走独立嵌套曲面；否则拒绝，
不会把任意三相分割伪装成包覆结构。`auto` 只读取显式声明，不猜测标签含义。

Selecting `[1,2]` keeps active and CBD solid cells and treats bounded excluded
regions as holes. Selecting `[0,1,2]` also meshes the pore domain. Selecting one
label uses a single-region contour by default. Region0 remains a valid material
ID; it is not removed unless the caller excludes it.
可以只保留固体，也可以连同孔隙一起生成体单元；编号0不会被自动删除。

## 7. Complete adaptive options / 完整参数说明

|Parameter / 参数|Meaning / 含义|
|---|---|
|`vertex_fraction`|Starting target fraction; actual output is checked and reported / 初始曲面点比例|
|`levels`|1..6 residual scales / 几何残差层数|
|`sigma`|Gaussian scale in each residual level / 各层高斯尺度|
|`weight_power`|Power/scale decay in GEMsim residual weighting / 残差幂次及尺度衰减|
|`weight_floor`|Positive numerical density floor / 正的网格密度下限|
|`weight_strength`|Geometric residual influence;0 gives uniform-density ACVD / 几何权重强度|
|`maximum_weight_ratio`|Limits numerical mesh-density contrast; does not clip physical fields / 密度比上限|
|`minimum_component_vertices`|Closed components below128 source vertices are fully protected by default / 默认保护少于128点的小连通曲面|
|`iterations`|ACVD optimization iteration budget / ACVD迭代预算|
|`topology_retries`|Surface/whole-pipeline refinement attempts / 重试预算|
|`maximum_facets`|Default4000000; explicit surface input/output budget, not a hard RAM cap / 默认400万面，仍需进程内存限制|
|`maximum_phase_volume_error`|Per-phase volume difference from original voxel union / 分相体积差门槛|
|`surface_method`|`auto`, `marching_cubes`, `shared_interfaces`, `nested_contours`|
|`phase_order`|Explicit outer→inner material chain / 显式包覆关系|
|`project_surface`|Optional projection toward actual source triangles; defaultFalse follows ACVD centroids / 可选投影，默认保留ACVD质心|
|`retry_without_facet_merging`|DefaultTrue; on an exact PLC volume failure, retry the same geometry with M; quality and volume gates unchanged / 体积失败时保面重试，可关闭|
|`density`|Python callback receiving XYZ-metre surface vertices / 自定义数值密度函数|

Custom spatial density / 自定义空间密度：

```python
options = AdaptiveMeshOptions(
    vertex_fraction=.15,
    density=lambda xyz:np.where(xyz[:,0]>50e-6,6.,1.),
    maximum_weight_ratio=32,
)
# More target points in the prescribed region / 指定区域布置更多目标节点
result = model.adaptive_mesh(options=options)
```

A standalone surface can be obtained and inspected before volume meshing:
可以先检查重构曲面，再划分体单元：

```python
surface = model.adaptive_surface(regions=[0,1,2],options=options)
surface.show(title='Actual ACVD surface / 真实重构曲面')
surface.viewer().save_html(work/'surface.html')
```

The returned `SurfacePLC` also works with `tetgen_mesh(surface,options=...)`.
`TetGenOptions` controls volume refinement separately. Adaptive meshing defaults to `TetGenOptions.for_adaptive()`: it permits merging only practically coplanar faces (`facet_separation_angle=179.99999` degrees). This preserves curved geometry without forcing all redundant planar edges to remain. Use that constructor or set the fields explicitly when supplying `tetgen_options`. A global maximum volume
is distinct from the ACVD surface density and may make the interior unnecessarily
uniform. Use sizing and quality options deliberately.
曲面密度与体网格的最大体积、二面角等参数独立。新路线默认使用for_adaptive策略，将共面合并角收紧到179.99999度，保留曲面几何并允许去除冗余平面边；显式传入选项时应使用该构造器或设置相应字段。全局最大体积会约束整个内部区域，
不应把它与几何局部加密混为一谈。

`TetGenOptions.for_adaptive()` also defaults to `isolated=True`. The native
solver runs in a Python worker process with `native_timeout_seconds=180`.
Native errors, process aborts and timeouts become exceptions in the caller;
they do not count as accepted meshes. Temporary HDF5 IPC files stay under
`$MORQEX_WORKSPACE/cache/native_mesh_calls/` and are removed after the call.
This requires h5py, included in `adaptive-meshing`. An explicit options object
constructed with `TetGenOptions(...)` retains the low-level default
`isolated=False`; use the adaptive constructor or set the field yourself.
自适应入口默认隔离原生进程，避免TetGen崩溃直接终止Notebook；默认原生时间预算180秒，
错误、超时均返回异常。HDF5临时交换文件位于源码外并在调用后清除。隔离不等于硬内存限制，
仍应通过资源启动器限制整个进程树，不能同时无预算启动多个网格任务。

## 8. TOML and CLI / TOML 与命令行

```python
from morqex.modeling import from_toml,mesh_from_toml,export_comsol
config = '/home/zephyrain/lw/MORQEX/configs/geometry_adaptive.toml'
model = from_toml(config)
result = mesh_from_toml(model,config)
result.save_hdf5(work/'adaptive.h5')
export_comsol(result,work/'adaptive.mphtxt',region_names=model.region_names)
```

[Complete TOML recipe / 完整配置](../configs/geometry_adaptive.toml) separates
`[geometry]`, `[meshing.options]` and `[meshing.tetgen_options]`. Density callbacks
are Python extensions rather than executable strings in TOML.
自定义密度函数用 Python 回调，不在 TOML 中执行任意代码字符串。

Without `--method`, `generate ... --mesh ...` follows `[meshing]` in its TOML.
An explicit `--regions` overrides that recipe's selection. Otherwise the recipe
is respected; standalone `mesh` defaults to all present labels. Pass `--regions
1,2` explicitly to exclude pore0. `--mesh-preview` also works with the retained
raw-TetGen and streamed-voxel routes, using their own mesh display APIs.
`generate`未指定方法时采用TOML网格配方；显式`--regions`可覆盖配方。单独`mesh`默认所有
存在的标签，需要排除孔隙时明确选1,2。旧路线同样支持预览，但不会冒充自适应网格。

```bash
"$PY" -m morqex.modeling mesh "$MORQEX_WORKSPACE/geometry.h5" \
  --method gemsim_adaptive --precision balanced --regions all \
  --output "$MORQEX_WORKSPACE/adaptive.h5" \
  --mesh-preview "$MORQEX_WORKSPACE/adaptive.html" \
  --comsol "$MORQEX_WORKSPACE/adaptive.mphtxt"
```


## 9. Full surface and true volume-section visualization / 完整曲面与真实体切面

```python
from IPython.display import display
model.show(maximum_3d_size=48,title='Source voxels / 原始体素')
display(result.adaptive_viewer(title='Nonuniform mesh / 疏密变化网格'))
result.adaptive_viewer().save_html(work/'adaptive.html')
```

The viewer embeds the complete marked computational surface, with selectable
material interfaces, camera rotation, wireframe and an explicit display-face
budget. “All” renders all stored boundary triangles; a display budget never
alters mesh data.
查看器包含完整计算曲面，可选择材料界面、旋转、线框和显示面预算。显示预算不改变数据；
选择“All”显示全部保存的边界三角形。

The YZ/XZ/XY views intersect the **actual Tet4 elements** with a physical plane.
Their triangles/quadrilaterals are real element intersections, not copied
voxels or a substitute mesh. They show interior grading much more clearly than
scattered tetrahedral samples. The source viewer's2D slices retain the complete
original label array; its3D voxel preview uses an explicit display LOD.
体切面显示真实单元与物理平面的交多边形，更容易看清内部疏密；不是体素切图冒充网格。
原始体素查看器的二维切片保留全分辨率，三维体素预览才使用明确标注的显示LOD。

For HDF5 sections, connectivity is streamed in50000-cell chunks. Node tables
up to2000000 vertices use a48MB-bounded cache; larger tables use contiguous
32768-node reads instead of expensive HDF5 random selections. The3D renderer
uses origin-relative FP32 display coordinates; the original mesh remains FP64
in SI metres. The three default sections are midplanes. Other planes can be
computed with `section_polygons(result,axis=0,fraction=0.35)`; a display cut of
the outer surface is distinct from a newly computed volume section.
截面逐块读取体单元；节点小缓存上限48MB，超限时采用连续分块读取。三维显示采用相对原点的
单精度坐标，原始网格仍是米制双精度。默认三个中平面；自定义位置使用上述API重新计算，
曲面剖切滑块与体单元切面计算是不同操作。

## 10. Generate all ten256³ examples / 批量生成十个完整256³样例

```bash
# Run controller directly. It launches only one bounded child at a time.
# 直接启动控制器；它逐个调用资源限制器，不要再把控制器套进run_local。
"$PY" scripts/adaptive_geometry_gallery.py \
  --workspace "$MORQEX_WORKSPACE" \
  --output "$MORQEX_WORKSPACE/adaptive_gallery256_gemsim" \
  --size 256 --precisions balanced

# Additional precision comparison, retaining earlier outputs / 保留旧结果并增加精度对比
"$PY" scripts/adaptive_geometry_gallery.py \
  --workspace "$MORQEX_WORKSPACE" \
  --output "$MORQEX_WORKSPACE/adaptive_gallery256_gemsim" \
  --cases granular rods --precisions coarse fine
```

A source snapshot is frozen outside the repository before computations.
Every case saves source geometry, native mesh, XDMF, COMSOL file, domain map,
actual mesh HTML and a numerical validation report. The top-level HTML updates
as cases finish. Completed cases are retained for restart; after changing a
numerical recipe, use a new output directory to avoid mixing old/new results.
每次计算有外部冻结源码；每个样例保存几何、完整网格、COMSOL、域映射、HTML及验收报告。
顶层页面随完成进度更新。修改数值配方后应换输出目录，避免混合版本。
Incomplete case directories are moved to `rejected_cases/` before a retry;
accepted cases are retained. No failed mesh is promoted by deleting bad cells.
未完成的单例在重跑前移动到`rejected_cases/`保留，已验收结果不覆盖。

## 11. Acceptance and limitations / 验收与适用范围

Checks include unchanged source-grid size, patch components/Euler/open-edge
statistics, positive element volumes, region seed containment, enclosed PLC
volume, per-phase voxel-volume difference, actual relocated boundary vertices,
and exact COMSOL coordinate/connectivity read-back.
检查源网格大小、界面拓扑、正体积、种子所在分区、曲面包围体积、分相体积偏差、真实节点
重定位和 COMSOL 数值读回。

Per-phase geometric error is not a concentration, stress or electrochemical
error. Finite-element convergence and conditioning require their own studies.
`TetGenOptions(require_quality=True)` rejects unmet measured radius/dihedral
requests; requesting `q2` is not proof that every constrained element meets it.
Reports and HTML separate geometry/export acceptance from shape-quality targets.
几何体积差不等于浓度或应力误差；请求 q2 不代表最差单元已达标，报告中独立列出。

Imported pathological contacts or unresolved features can still exceed the
memory budget or fail geometric checks. This is reported, never repaired by
silently changing material labels, deleting physical interfaces or dropping
bad tetrahedra. The resource runner bounds CPU/RAM, and no neural network is used.
病态接触或欠分辨率特征仍可能失败或超预算；不会通过偷改标签、删界面或删体单元掩盖。

COMSOL native-format/read-back checks are performed; actual COMSOL application
import is not tested without an installed/licensed application.
本机缺少 COMSOL，因此没有宣称已完成实际软件导入。

### TetGen numerical interface policy / TetGen界面数值策略

`coplanarity_tolerance` controls TetGen's normalized T predicate tolerance,
not the accepted material-volume error. Adaptive workflows use a strict facet merge angle to retain
reconstructed geometry; M (`merge_coplanar=False`) can retain every input facet, but often overconstrains the surface; Y (`preserve_surface=True`) is available to prohibit
boundary subdivision, matching the original GEMsim choice, but can impede
quality improvement. M permits boundary refinement while keeping the geometric
interface constraints. Output material-volume checks remain mandatory.
T控制内部共面判据，与分相体积验收门槛独立。默认收紧p的共面合并角；M完全保留每个面片但可能过约束；Y禁止边界细分，也可能限制质量改善。
这两个开关不能当成质量达标证明。

If native merging violates the mandatory `1e-8` PLC-volume gate, the adaptive
API can retry that **same** reconstructed surface with `merge_coplanar=False`.
This is controlled by `retry_without_facet_merging`, recorded in
`report['adaptive_retries']`, and never relaxes volume or requested-quality
acceptance. Retaining every facet can yield very poor constrained elements;
inspect measured metrics and use `require_quality=True` for a strict FE project.
On the fine rod diagnostic, further tightening T or the merge angle did not
remove the volume defect, whereas retaining the facets did. Geometry/export
acceptance of that mesh is explicitly not a passed FE-convergence claim.
若原生合并造成超差，则在同一重构曲面上保留全部面片重试；记录在报告中，可显式关闭，
且不降低任何门槛。长条细网格实测中收紧T或合并角无效，保面则通过体积检查；代价可能是
很差的局部单元形状。严格有限元用途应打开`require_quality=True`并开展收敛验证。

See [TetGen's official switch documentation](https://www.wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html).

The branch-case diagnostic compared ordinary facet merging, M and the strict
merge-angle policy without changing the volume acceptance gate. The strict
angle retained the enclosed volume to roundoff while substantially reducing
the worst radius/edge ratio relative to M on that case. This is a numerical
configuration study, not a universal quality guarantee. Original/failed outputs
remain outside source for diagnosis.
分枝案例对比了默认合并、M和严格合并角：体积门槛没有降低；严格角策略保留包围体积，
并较M显著改善该例最差半径边长比。此为实测配置结论，不是对所有几何的保证。
