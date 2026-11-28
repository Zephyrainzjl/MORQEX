# Finite-element meshes from voxel regions

[中文版本](02_finite_element_meshes.md#zh) · [Tutorial index](README.md)

## Recommended geometric-adaptive route / 几何自适应新路线

Use `model.adaptive_mesh(...)` or `mesh(model,method='gemsim_adaptive',...)`
for the GEMsim-style multiresolution residual → weighted ACVD → TetGen
workflow. It reconstructs a nonuniform unstructured surface before volume
meshing. The older routes below remain available for their own purposes.
新路线先重建疏密变化的非规则曲面，再用 TetGen 划分体域；下面的体素精确剖分和原始 PLC
等路线保留，但不应混称为 GEMsim 自适应重网格。

Read the complete bilingual [adaptive tutorial](06_gemsim_adaptive_meshing.md)
for precision presets, arbitrary region arrays, shared/nested material
interfaces, user density callbacks, whole-surface visualization, exact volume
sections and the ten complete256³ cases.
完整参数、精度、材料界面、自定义密度及十种256³样例见自适应教程。

## Available methods and their differences / 全部现有网格路线

This inventory covers the mesh-generation routes currently present in MORQEX;
optional inherited APIs require their own native dependencies. It does not
claim that every backend or every topology has passed numerical qualification.
下表覆盖当前源码中的网格路线；继承接口不代表已通过所有组合的验证。

|Route / 路线|Public API / 入口|Result / 输出|Requirements and scope / 依赖及范围|
|---|---|---|---|
|Structured Hex8 / 规则六面体|`framework('finite_elements').voxel_to_hex(model.volume(),phases=...)`|`FEMesh`,8-node cells / 八节点体单元|NumPy; optional native accelerator / 可选原生加速|
|Voxel Tet4 / 体素四面体|`model.tetrahedralize(regions=...)` or `mesh(method='tetrahedral')`|6Tet4 per selected voxel / 每体素6单元|Exact voxel labels/volume; small in-memory mesh / 精确标签及体积|
|Streamed voxel Tet4 / 分块体素四面体|`model.mesh_to(path,...)` or `mesh(method='streamed_tetrahedral',path=...)`|Complete shared-node HDF5/XDMF / 全量共享节点|Bounded slab cells; global node map / 分层流式|
|Exact shared PLC / 精确共享多相面|`voxel_plc(model,regions=...)`|Planar quads+seeds+holes / 四边形界面及种子|Surface construction, not volume mesh / 只是界面构造|
|Single-phase contour / 单相轮廓|`marching_cubes_plc(model,region=...,step_size=...)`|Closed triangle PLC / 闭合三角面|scikit-image; approximate geometry / 几何近似|
|GEMsim adaptive / 自适应重网格|`model.adaptive_mesh`, `adaptive_surface`, `mesh(method='gemsim_adaptive')`|Weighted ACVD surface + graded TetGen volume / 加权曲面与非均匀体网格|Optional original ACVD + TetGen native kernels / 集成原生内核|
|TetGen volume / TetGen 体网格|`model.tetgen(...)`, `tetgen_mesh(...)`, `mesh(method='tetgen')`|Region-labelled Tet4 with actual quality / 带分区和实测质量|Optional C++ compile; shared PLC / 可选源码编译|
|Custom surface PLC / 自定义面模型|`SurfacePLC(points,facets,seeds,markers,holes)` → `tetgen_mesh`|User geometry→Tet4 / 自定义贴体网格|Explicit interfaces and valid closed surfaces / 明确材料界面|
|Legacy GEM contour and ACVD / 旧 GEM 曲面及重网格|`framework('finite_elements').TriMesh.from_vol(...)`, `.TetMesh.from_vol(...)`, `.TetMesh.from_sur(...)`|Legacy mesh classes / 旧网格对象|Original Cython+ACVD+TetGen extensions; not the new tested native route / 依赖原扩展|
|Legacy GEM structured / 旧 GEM 规则单元|`LineMesh.from_vol`, `TriMesh.from_vol2d`, `QuadMesh.from_vol_2d`, `QuadMesh.from_vol`, `HexMesh.from_vol`, `HsHexMesh.from_vol`|1D/2D/3D specialized meshes / 专用单元|Inherited PhaseVol contract and optional kernels / 继承分区对象与扩展依赖|
|Statistical tessellation / 统计镶嵌|`framework('tessellation').PolyMesh.from_seeds(...)`|Power/Voronoi polygon/polyhedron cells / 多边形或多面体|Optional pyvoro; not automatically a Tet4 solve mesh / 不是自动生成四面体|
|Statistical raster / 统计规则网格|`framework('tessellation').RasterMesh.from_polymesh(...)`|Quad/Hex raster / 规则单元|Inherited MicroStructPy dependencies / 原依赖|
|Statistical Triangle/TetGen / 统计非结构网格|`framework('tessellation').TriMesh.from_polymesh(...,mesher='Triangle/Tetgen')`|Triangle in2D, tetrahedron in3D / 二维三角、三维四面体|MeshPy native engine; distinct from MORQEX native ABI / MeshPy 引擎|
|Statistical Gmsh / 统计 Gmsh 网格|same `TriMesh.from_polymesh(...,mesher='Gmsh')`|Gmsh mesh / Gmsh 网格|pygmsh/Gmsh and system libraries; optional / 原生系统依赖|
|Custom volume mesher / 自定义体网格|`register_mesher(name,callback)` then `mesh(method=name)`|`FEMesh`/`MeshArtifact` / 统一返回契约|Explicit validation required / 必须自行验证|

The new TetGen wrapper uses the existing source-integrated1.6.0 C++ core and
accepts real material-region and cavity seeds. It does not call the legacy
`sur_tetgen` failure path that retries after removing faces. The legacy route
is retained for compatibility, not endorsed for unattended reference meshes.
新 TetGen 直接使用已集成的 C++ 核心，并避开旧实现失败后删面的路径。
旧路线保留兼容，不作为无人值守参考网格的推荐入口。

Read the complete bilingual [TetGen build, sizing, quality, TOML, notebook and
COMSOL tutorial](05_tetgen_and_comsol.md) for executable examples. All current
linear Tet4/Hex8 volume routes can use `export_comsol`. A surface-only STL is
not a COMSOL volume mesh.
完整编译、尺寸、质量、配置、可视化及 COMSOL 示例见该教程。仅表面 STL 不等于体网格。

## One unified mesh entry

```python
from morqex.modeling import from_regions
model = from_regions(labels,spacing=(0.5e-6,)*3)

# Small arrays: returns the integrated FEMesh object.
mesh = model.tetrahedralize(regions=[1,2])
print(mesh.points.shape,mesh.cells.shape,mesh.phases.shape)

# Full256³ arrays: bounded-memory generation into an external HDF5 file.
artifact = model.mesh_to(
 '/home/zephyrain/lw/morph_workspace/microelectrode_gallery/my_mesh.h5',
 regions=[1,2],slab_depth=2,compression=1,
)
print(artifact.hdf5,artifact.xdmf,artifact.nodes,artifact.tetrahedra)
```

Region0 is omitted **because the caller selected[1,2]**, not because the API
assumes0 always means pore. To mesh all regions, use `regions=None`. Calling
the in-memory method on more than262144 voxels requires an explicit larger
budget; the normal256³ route is `mesh_to`.

## Discretization and interfaces

The existing GEM `voxel_to_tetra` module splits each selected hexahedral voxel
into six Tet4 elements with a globally consistent Freudenthal diagonal. The
streamed API calls that same module on slabs and maps every slab vertex to one
compact **global** node index. Slab interfaces therefore share actual nodes;
they are not disconnected mesh pieces. Region IDs and the original flattened
ZYX parent-voxel ID accompany every element.

The default topology bonds solid voxels that share nodes, including point/edge
contacts. This is an explicit modelling assumption. For small mechanical meshes,
the mechanics-only `morqex.mechanics.mesh.TetMesh.from_volume(...,separate_point_contacts=True)` implements
a different policy. That policy is not silently substituted into the streamed
mesh. Crack insertion, cohesive interface nodes, periodic node identification
and contact mechanics require separate explicit topology operations.

Voxel staircases are preserved. The generated tetrahedra are valid geometric
elements; a staircase boundary is not equivalent to a smoothed body-fitted
surface. This route has no automatic adaptive error estimator, element-order
upgrade or convergence guarantee.

## Data layout

|HDF5 dataset|Meaning|
|---|---|
|`phase_labels`|Original complete ZYX region array|
|`mesh/points_m`|FP64 `(N,3)` global XYZ vertices, SI metres|
|`mesh/tetrahedra`|UInt32 `(E,4)` global node connectivity|
|`mesh/phase`|One original region label per Tet4|
|`mesh/parent_voxel`|Flattened original ZYX voxel index per Tet4|

The generated `.xdmf` file references those datasets. Open it in ParaView while
keeping the `.h5` file beside it. `MeshArtifact` is a path/count contract, not
an in-memory array containing tens of millions of elements. Use HDF5 slicing:

```python
import h5py
with h5py.File(artifact.hdf5) as f:
    cells = f['mesh/tetrahedra'][0:100000]
    phase = f['mesh/phase'][0:100000]
```

Do not load the complete dataset just to preview the model. A half-solid256³
domain contains approximately50 million tetrahedra; four UInt32 node IDs alone
occupy about800MB before compression. Additional stresses, gradients and global
matrices are substantially larger. Geometric meshing does not certify that a
full mechanical solve fits the workstation budget.

## Checks and limits

Generation checks positive signed element volumes, exact region/parent
association, the six-elements-per-selected-voxel relation, total solid volume
and node index consistency. The output becomes final only after checks pass;
an incomplete `.pending` file is not an accepted artifact. Existing final mesh
files are never overwritten automatically. A failed large job may require
rerunning meshing; the current streamer does not resume a partly written slab.

`slab_depth` can be1..16; use1 or2 on a workstation. Compression level1 is a
useful starting point for geometry. The global compact-node map scales with
the grid, while connectivity and validation temporaries scale with a slab.

The new stream is checked against the original in-memory module on anisotropic
grids with nonzero origins, custom labels and pores. Identical global cells,
vertices, phases and parent indices are required. The256³ gallery receives
additional read-back checks on the exported full mesh.

## Existing alternative mesh interfaces

```python
from morqex.modeling import framework
fe = framework('finite_elements')
hex_mesh = fe.voxel_to_hex(model.volume(),phases=[1,2])
surface_mesh = fe.surface(model.volume(),phase=1)
```

`surface` uses marching cubes and requires scikit-image. It is a visualization /
surface extraction, not a solid Tet4 mesh. TetGen/MeshPy/Gmsh require their
compiled dependencies, valid closed surfaces and independent quality checks.
The namespace portal exposes installed backend APIs but does not install their
native engines or assert that every possible backend combination is qualified.

## Large output budget and custom meshers

Use `scripts/run_local.py --mesh-output` for a complete256³ HDF5 mesh that may
exceed the default per-file cap. This permits a1GiB external mesh file while
retaining the default CPU, RSS and host reserve. It does not increase the
mechanical solver's capacity.

```python
from morqex.modeling import register_mesher,mesh

def my_mesher(model,regions):
    return model.tetrahedralize(regions=regions)

register_mesher('my_mesher',my_mesher)
result = mesh(model,method='my_mesher',regions=[1,2])
```

Custom mesh callbacks must return `FEMesh` or `MeshArtifact`. Duplicate names
are rejected. Start with a small array and compare volumes, coordinates, region
labels and source associations before increasing scale.

In ParaView, `Surface With Edges` displays the actual boundary triangles;
`Clip` exposes internal regions. `Shrink` can visually separate local elements,
but is a display filter and does not alter the computational geometry. Avoid
loading every element merely to inspect a small feature; use the HTML local
mesh or an explicit HDF5 subset.


---

<a id="zh"></a>

# 体素到有限元网格教程

[English version](02_finite_element_meshes.md) · [体素建模](01_geometry_and_voxels.md#zh) · [教程目录](README.md)

## 1. 什么输入会得到什么网格

输入为三维整数区域数组。`regions=[1,2]` 表示仅对活性材料与CBD生成体单元；
孔隙不赋予固体刚度。`regions=None` 表示为全部标签生成网格，包括区域0。
标签含义由用户规定，程序不会自动假设0一定是孔隙。

输出为四节点四面体Tet4：每个选中体素严格划分为6个四面体。划分使用集成GEM的
`voxel_to_tetra`，相邻体素采用一致的面内对角线。

```python
from morqex.modeling import from_regions
model = from_regions(my_labels_zyx,spacing=(0.5e-6,)*3)
```

## 2. 小网格：直接返回节点与单元

```python
mesh = model.tetrahedralize(regions=[1,2])
points = mesh.points  # (N,3), XYZ, 米
cells = mesh.cells    # (E,4), 节点索引，从0开始
phases = mesh.phases  # (E,), 每单元所属区域ID
volumes = mesh.signed_volumes()
assert (volumes > 0).all()
```

默认内存入口限制总输入体素数不超过262144，即64³，防止误将256³的全部单元、
梯度和中间数组一次装入内存。确有充分资源时可显式调整 `maximum_voxels`，但常规
大网格应使用下面的磁盘分块方法。

## 3. 256³完整网格：分块生成，共享全局节点

```python
artifact = model.mesh_to(
    '/home/zephyrain/lw/morph_workspace/microelectrode_gallery/custom_mesh.h5',
    regions=[1,2],
    slab_depth=2,
    compression=1,
)
print(artifact.nodes)
print(artifact.tetrahedra)
print(artifact.hdf5)
print(artifact.xdmf)
```

生成过程：

1. 统计被选中区域的体素，确定总单元数。
2. 标记这些体素实际使用的格点，建立紧凑全局节点编号。
3. 按Z方向逐薄层调用同一GEM四面体网格模块。
4. 将薄层中的局部节点映射回全局节点，写入HDF5。
5. 写入每个单元的区域ID与原始体素索引。
6. 检查单元正体积、标签对应、数量和总体积，通过后形成最终文件。

不同薄层不是相互脱开的网格。它们的界面使用相同的全局节点编号。相界面两侧
默认共享节点，即连续位移假设；这不等于界面已经具有脱粘、接触或粘聚单元。

## 4. 数量、内存与文件尺寸

256³共有16777216个体素。若一半为固体，约产生50331648个Tet4。仅四个UInt32
节点索引就需要约0.8GB未压缩存储，还不包括节点、标签、应力、单元梯度和矩阵。

`mesh_to` 不同时创建完整全局连接数组；连接与临时验证数组按薄层处理。
节点映射仍随原始网格规模增长，所以它不是完全与模型规模无关的内存算法。

|参数|建议与影响|
|---|---|
|`slab_depth`|1~16；工作站建议1或2，值越大临时内存越大|
|`compression`|HDF5 gzip等级，建议1；高等级通常更慢|
|`regions`|明确选择要生成体单元的标签|
|`spacing`|从VoxelModel继承，不在网格步骤偷偷改变尺寸|
|`origin`|从VoxelModel继承，支持非零和负坐标原点|

大文件经受控运行器运行时，使用显式的文件输出预算：

```bash
WORK=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
PY=$WORK/runtime/bin/python
python scripts/run_local.py --workspace "$WORK" --mesh-output \
  --label my_large_mesh -- "$PY" my_mesh_recipe.py
```

`--mesh-output` 允许单个外部输出文件达到1GiB；默认单核、RSS和主机内存余量限制
仍保持。它不会令力学求解器自动具备处理整个大网格的能力。

## 5. 文件结构与读取

|数据集|类型与含义|
|---|---|
|`phase_labels`|原始ZYX三维分区数组|
|`mesh/points_m`|FP64 `(N,3)`，XYZ米|
|`mesh/tetrahedra`|UInt32 `(E,4)`，共享全局节点索引|
|`mesh/phase`|原始区域ID，每单元一个|
|`mesh/parent_voxel`|原始三维数组按C顺序展平后的体素索引|

```python
import h5py,numpy as np
with h5py.File(artifact.hdf5,'r') as f:
    cells = f['mesh/tetrahedra'][:100000]
    parent = f['mesh/parent_voxel'][:100000]
    region = f['mesh/phase'][:100000]
    z,y,x = np.unravel_index(parent,f['phase_labels'].shape)
```

一个原始固体体素对应6个连续Tet4。`parent_voxel` 把单元映射到原始区域和体素场；
以后传递浓度、材料参数或后处理数据时，应保留这个映射，而不是靠最近邻猜测。

## 6. ParaView查看

将 `.xdmf` 与 `.h5` 保持在同一文件夹，在ParaView中打开 `.xdmf`。可选：

- `Surface`：观察固体外表面与孔隙边界。
- `Surface With Edges`：观察实际四面体的边界三角面。
- `Clip`：去掉一部分实体以显示内部结构。
- `Extract Cells By Region` 或区域阈值：只保留某种材料。
- `Shrink`：适当缩小单元显示，观察局部四面体连接；这是显示操作，不改原始网格。

不建议仅为看图就载入数千万单元。HTML的局部有限元视图和HDF5分段读取更节省资源。
全域HTML显示LOD与完整计算网格之间的区别已在页面明确标注。

## 7. 网格的正确性与物理限制

本路线检查：全部单元正体积、每体素6单元、原始区域完全对应、节点索引有效、
共享全局节点唯一、总体积与选中体素体积一致。256³图谱还逐单元读回了完整文件。

这些检查证明几何和数据的一致性，不能替代：

- 力学、扩散或电化学方程的物理验证；
- 网格收敛和尖角应力的客观性检查；
- 任意复杂几何的RVE代表性；
- 接触、裂纹、界面分离和周期边界的单独建模。

默认共享格点策略会连接只在边或角接触的体素。小网格另有
`morqex.mechanics.mesh.TetMesh.from_volume(...,separate_point_contacts=True)`；分块大网格目前没有
静默使用此策略。需要这一物理区别时，必须显式选择并验证对应的拓扑处理。

生成时先写 `.pending` 文件，全部检查通过后再形成最终HDF5；已有最终文件不会
被自动覆盖。当前流式生成器尚不支持从中途薄层恢复，失败后应重新生成或更换输出路径。

## 8. 六面体、表面网格与插件

```python
from morqex.modeling import framework,register_mesher,mesh
fe = framework('finite_elements')
hexa = fe.voxel_to_hex(model.volume(),phases=[1,2])
surface = fe.surface(model.volume(),phase=1)
```

`surface` 使用marching cubes，需要scikit-image；它是表面三角网格，不是体四面体网格。
TetGen、MeshPy、Gmsh等体贴合方法需要安装各自的本地后端，且对输入封闭性和质量
另有要求，不能用表面提取冒充完整体网格。

```python
def my_mesher(model,regions):
    return model.tetrahedralize(regions=regions)
register_mesher('my_mesher',my_mesher)
result = mesh(model,method='my_mesher',regions=[1,2])
```

自定义网格回调必须返回统一的 `FEMesh` 或 `MeshArtifact`，同名回调不能覆盖已有
方法。推荐先用小网格与内置方法核对几何、相标签、体积和节点对应，再扩大规模。


## Inherited mesh APIs and conversion / 继承网格 API 与转换

```python
# Pure NumPy Hex8, all specified materials / 规则 Hex8，保留明确指定的材料
from morqex.modeling import framework,export_comsol
fe = framework('finite_elements')
hex_mesh = fe.voxel_to_hex(model.volume(),phases=[1,2])
export_comsol(hex_mesh,work/'hex.mphtxt')
```

The old `fe.surface` is a legacy visualization extractor with its original
coordinate convention. For the unified voxel-centre convention and a contour
passed to TetGen, use `marching_cubes_plc`; do not mix the two without checking
the half-cell location. `TriMesh.from_vol` accepts legacy `PhaseVol`, not the
new `VoxelModel` object. Inspect a backend signature/docstring in Python before
using an inherited method:
旧 `surface` 保持其原有坐标约定；统一体素中心坐标及 TetGen 输入使用
`marching_cubes_plc`。旧 `TriMesh.from_vol` 接受 `PhaseVol`，不接受 `VoxelModel`。

```python
import inspect
# Imports may require additional original native kernels / 导入可能需要原扩展
# print(inspect.signature(fe.TriMesh.from_vol))
# help(fe.TetMesh.from_sur)
# stats_mesh = framework('tessellation')
# help(stats_mesh.PolyMesh.from_seeds)
# help(stats_mesh.TriMesh.from_polymesh)
```

MicroStructPy's `PolyMesh` contains `points`, `facets`, `regions`,
`seed_numbers` and `phase_numbers`; its `TriMesh` contains `points`, `elements`,
`element_attributes`, `facets`, `facet_attributes`. A statistical seed ID is
not automatically the same as a material phase. Convert labels explicitly,
then use the shared output API:
统计种子编号不自动等于材料编号，应明确映射后再转成统一数据结构：

```python
import numpy as np
from morqex.microstructure.mesh.voxel import FEMesh
# Given a completed inherited3D TriMesh / 已生成的继承三维 TriMesh
# original_phase = np.array([seed_to_phase[int(s)] for s in inherited.element_attributes])
# result = FEMesh(np.asarray(inherited.points),np.asarray(inherited.elements),original_phase)
# negative = result.signed_volumes()<0
# result.cells[negative] = result.cells[negative][:,[1,0,2,3]]
# assert np.all(result.signed_volumes()>0)
# export_comsol(result,work/'statistical.mphtxt')
```

Import existing Tet4/Hex8 files through meshio and explicitly choose the phase
attribute and coordinate scaling. Do not rerun a mesher if the input is already
a valid volume mesh. Surface files need a valid closed PLC and interior seeds.
已有体网格直接导入，不需要重复划分；外部文件的单位和材料属性必须显式指定：

```python
import meshio
# imported = meshio.read('/external/electrode.msh')
# cells = imported.get_cells_type('tetra')
# phases = imported.get_cell_data('gmsh:physical','tetra')
# result = FEMesh(imported.points*1e-6,cells,phases) # only if source was in µm
# export_comsol(result,work/'imported.mphtxt')
```

Periodic pairing, high-order elevation, mechanical cracks/cohesive nodes and
mesh adaptivity are not implied by a successful volume mesh generation.
周期配对、高阶提升、裂纹黏聚节点与自适应不因体网格生成成功而自动具备。

## Fixed-grid smoothing and independent mesh density / 原尺寸平滑与独立网格密度

[The full new tutorial / 完整新教程](08_fixed_grid_smoothing_and_implicit_meshes.md)
explains same-size smoothing, separate `lower_resolution`, continuous multiphase
CGAL meshes, feature-size advice and all progress/TOML/notebook/export controls.
平滑保留原数组及物理范围；降低分辨率不属于平滑，必须明确调用。
Existing TetGen, ACVD, exact voxel and MMG entry points remain available.
原有各种网格与重网格 API 保留。
