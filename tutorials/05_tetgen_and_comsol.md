# TetGen compilation, volume meshing and COMSOL / TetGen 编译、体网格与 COMSOL

[All mesh methods / 全部网格路线](02_finite_element_meshes.md) · [API](API.md) · [Tutorial index / 教程目录](README.md)

This file contains both languages. All generated files, builds, environments,
HTML and wheels belong outside the MORQEX source tree.
本文件中英文合并。网格、构建目录、环境、HTML、wheel 均写入源码外工作区。
Coordinates of mesh vertices and PLC seeds are **XYZ metres**; voxel arrays,
`spacing`, `origin` and shape parameters use **ZYX metres**.
网格顶点及 PLC 种子用 **XYZ 米**；体素数组、间距、原点及形状参数用 **ZYX 米**。

For the newly requested GEMsim-style adaptive surface reconstruction, see
[weighted ACVD → TetGen](06_gemsim_adaptive_meshing.md). This file documents the
lower-level TetGen PLC/voxel routes, which remain available independently.
新要求的 GEMsim 自适应曲面重建见该教程；本文件描述保留的底层 TetGen/PLC/体素路线。

## 1. Installation choices / 安装时选择是否编译

A normal installation stays Python-only. TetGen's original C++ source is
already included; the native adapter is compiled only when explicitly enabled.
默认安装不编译原生组件。TetGen 原始 C++ 源码已集成，只有显式开启才编译适配库。

```bash
cd /home/zephyrain/lw/MORQEX
export MORQEX_WORKSPACE=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
python -m venv --system-site-packages "$MORQEX_WORKSPACE/runtime"
PY="$MORQEX_WORKSPACE/runtime/bin/python"

# Python-only; no C++ compiler needed / 纯 Python，不需要 C++ 编译器
"$PY" -m pip install '.[modeling,geometry,notebook]'

# Source-build TetGen / 可选地从本项目源码编译 TetGen
# Linux requires a C++17 compiler and matching Python development headers.
# Linux 需要 C++17 编译器和对应 Python 头文件；不要在源码中建立环境。
MORQEX_BUILD_TETGEN=1 "$PY" -m pip install --no-cache-dir --force-reinstall --no-deps .
```

Install dependencies first with the Python-only command, then compile with
`--no-deps` to avoid reinstalling the scientific environment. Without a prior
dependency installation, use `MORQEX_BUILD_TETGEN=1 "$PY" -m pip install '.[modeling,geometry]'`.
先用纯 Python 命令安装依赖，再用 `--no-deps` 编译，避免重装 NumPy/SciPy 等。
如果尚未安装依赖，可直接使用上述包含 extras 的编译命令。
`geometry` supplies scikit-image for marching cubes and meshio for VTK; exact
voxel PLC and native TetGen do not require either of those packages.
`geometry` 提供 marching cubes 的 scikit-image 和 VTK 导出的 meshio；精确体素 PLC
和 TetGen 核心本身不需要这两个包。

An extra such as `.[geometry]` installs dependencies; pip extras do **not**
change native build flags. `MORQEX_BUILD_TETGEN=1` is the compile switch.
extras 只决定依赖，不会自动改变构建开关；必须显式设置环境变量。
`MORQEX_BUILD_GEOMETRY=1` is a different, older opt-in for the complete GEM
Cython kernels, ACVD and legacy TetGen wrapper; it additionally needs Cython,
NumPy build headers and the dependencies of those extensions. The new TetGen
path needs neither Cython nor PyVista and does not invoke the legacy wrapper.
旧开关会编译一组不同的 GEM 扩展，需要额外构建依赖；新路线不依赖 Cython/PyVista。

The C++ objects, generated metadata and wheel intermediates are written to
`$MORQEX_WORKSPACE/build/packages/<source+ABI+build-flags hash>/`. The installed
shared library goes inside the chosen external Python environment. Source
content, Python ABI and build options separate caches, preventing a previously
compiled library from leaking into a later pure-Python wheel.
构建缓存按源码、Python ABI 和编译选项隔离；由此避免先编译后纯安装时错误夹带二进制。

**Do not use native editable installation (`-e`)**: setuptools would normally
copy compiled libraries into `src`. MORQEX rejects this operation. Keep an
editable Python checkout and use the external CMake build below, or install a
regular compiled wheel. Pure-Python editable installations remain possible.
原生可编辑安装会把二进制复制进源码，因此本项目明确拒绝；可以使用普通 wheel，
或者纯 Python 可编辑安装配合下面的外部 CMake 编译。

For a resource-limited build, run the same pip command through the local runner:
通过本地资源限制器单核编译，避免占满机器：

```bash
export MORQEX_BUILD_TETGEN=1
python scripts/run_local.py --workspace "$MORQEX_WORKSPACE" \
  --label pip_tetgen --dependency-install -- \
  "$PY" -m pip install --no-cache-dir --no-deps /home/zephyrain/lw/MORQEX
unset MORQEX_BUILD_TETGEN
```

A plain `pip install morqex-research` from an index only installs what that
index publishes. These commands target **this checkout** and do not claim an
upstream MORQEX release exists. Windows source builds require MSVC C++17 and
Python headers; the recorded qualification is Linux, not a Windows test.
以上命令针对本地源码，不声称 PyPI 存在已发布版本。Windows 需 MSVC；本次实测是 Linux。

## 2. Explicit external build / 已安装后显式外部编译

```python
import os
from morqex.modeling import build_tetgen, capabilities
work = '/home/zephyrain/lw/morph_workspace/microelectrode_gallery'
os.environ['MORQEX_WORKSPACE'] = work
library = build_tetgen(work)  # cmake --build ... --parallel1
print(library)
print(capabilities()['tetgen'])
```

This alternative requires CMake>=3.16 and a C++17 compiler but no Python
extension headers. Build is explicit, never a side effect of importing a
module or generating a mesh. Compilation uses one job. The same source feeds
pip and CMake, through the same C ABI and geometry code.
此路线需要 CMake 和 C++17，不需要编译 Python 扩展头文件。导入模块、生成网格不会偷偷
触发编译；pip 和 CMake 使用同一份 TetGen 核心与 C ABI。

CLI / 命令行：

```bash
"$PY" -m morqex.modeling build-tetgen --workspace "$MORQEX_WORKSPACE"
"$PY" -m morqex.modeling capabilities
```

Resolution order: `MORQEX_TETGEN_LIBRARY` explicit path, installed optional
`morqex._tetgen_native`, installed distribution metadata if a source checkout
shadows it, then matching CMake cache in `MORQEX_WORKSPACE`. A missing library
produces a precise installation error; there is no silent voxel-mesh fallback.
库查找顺序为显式路径、已安装扩展、安装包元数据、外部匹配缓存；找不到时明确报错，不冒充 TetGen。
The C ABI serializes native calls in one process; use bounded worker processes
for independent meshes, with total CPU/RAM budgets. TetGen itself is not CUDA.
同一进程的原生调用串行化；多个网格可用有总资源预算的独立进程。TetGen 本身不是 CUDA 求解器。

## 3. Exact multiphase voxel PLC / 精确保留分区的多相路线

```python
import numpy as np
from pathlib import Path
from morqex.modeling import from_regions, mesh, TetGenOptions, export_comsol
work = Path('/home/zephyrain/lw/morph_workspace/microelectrode_gallery/my_tetgen')
work.mkdir(parents=True, exist_ok=True)
labels = np.ones((24,24,24), dtype=np.uint8)
labels[:,:,12:] = 7
labels[8:16,8:16,8:16] = 2
model = from_regions(labels, spacing=(0.4e-6,0.5e-6,0.6e-6),
    region_names={1:'active',2:'CBD',7:'electrolyte'})
result = mesh(model, method='tetgen', regions=[1,2,7],
    surface_method='voxel_plc', maximum_facets=500000,
    options=TetGenOptions(radius_edge_ratio=2.0,
        minimum_dihedral=0.0, maximum_steiner=300000,
        maximum_cells=2000000))
print(result.report)
artifact = result.save_hdf5(work/'multiphase.h5')
result.save(work/'multiphase.vtu')        # optional meshio
export_comsol(result,work/'multiphase.mphtxt',region_names=model.region_names)
result.viewer(cell_count=1500).save_html(work/'tetgen.html')
```

The original integer array determines each phase. Interfaces are extracted
once, sharing the same vertices/facets between adjacent regions. Unit quads
are retained without independently smoothing each phase. Region attributes
come from seeds, not from assigning a tetrahedron's centroid a material after
meshing. A seed is placed inside **every**6-connected selected component;
bounded excluded components receive hole seeds. Original IDs, including0,
are encoded to positive TetGen attributes and decoded on output.
每个界面只提取一次并共享节点。每个六连通区域都放置内部种子；内部未选区域设置孔洞种子。
区域0也是普通合法材料编号，不自动等于孔隙。禁止网格生成后仅按质心标签冒充贴体多相网格。

Every region's integrated tetrahedral volume must match the original voxel
volume within1e-8 relative tolerance. SI geometry is normalized for meshing,
then transformed back. This path preserves staircase geometry; TetGen does
not magically convert a segmented staircase into the unknown true smooth
particle surface. Exact geometry preservation and smooth contour approximation
are two different choices.
每相体积相对误差必须≤1e-8。内部归一化后恢复 SI 坐标。精确保留台阶边界与光滑近似是不同选择。

Point/edge-only contacts, intersecting facets and touching cavity surfaces may
not define a valid manifold PLC. The wrapper never drops offending triangles
or invents a conductive bridge on failure. Inspect segmentation and define a
physically justified contact/void policy before meshing. To represent a crack,
separate surfaces/cohesive/contact conditions need explicit construction;
a conforming material interface alone is a bonded interface.
点接触、边接触、自交及相切空腔可能不是有效 PLC。失败时不会删面或凭空加导电桥。
裂纹/脱粘需要明确的独立面、黏聚或接触拓扑；普通共享节点界面表示粘结。

## 4. Smooth contour approximation / 单相轮廓近似

```python
# labels1 must form valid closed contours / 仅对一个明确标签生成闭合轮廓
curved = model.tetgen(regions=[1], surface_method='marching_cubes',
                     step_size=1, maximum_facets=500000)
print(curved.report['voxel_volume_relative_difference'])
```

`marching_cubes` needs scikit-image. The0.5 contour is located using voxel
**centres**, accounting for padding and the half-cell coordinate offset.
`step_size=1` uses the full input resolution;2,4,... explicitly change geometric
resolution, potentially erasing small particles or channels. Contour edge
manifold checks, seeds and returned material checks reject several invalid
cases; they are not a proof of geometric equivalence. Inspect volume change,
component counts and a resolution-convergence study. The original256³ array
is retained by `VoxelModel`; the contour's resolution is separately reported.
MC 使用体素中心和正确的半格偏移。步长增大改变几何分辨率，可能抹掉小颗粒或通道；
需要检查体积、连通性和分辨率收敛，不能将粗轮廓称为原始256³精确网格。

Multiple independently generated phase contours generally do not form matching
interfaces at triple junctions. This public contour route requires **one**
explicit region. Use shared voxel PLC or a user-constructed common interface
PLC for multiphase material-conforming meshing.
三相交线处逐相独立 MC 通常不匹配；本接口限制为单相。多相应使用共享 PLC。

## 5. Your own closed surface / 自定义三角面、区域与孔洞

```python
from morqex.modeling import SurfacePLC, tetgen_mesh
points = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                   [0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)*10e-6
faces = [[0,3,2,1],[4,5,6,7],[0,1,5,4],
         [1,2,6,5],[2,3,7,6],[3,0,4,7]]
plc = SurfacePLC(points,faces,seeds={42:[[5e-6,5e-6,5e-6]]},
                 markers=np.arange(1,7),region_volumes={42:1e-15})
solid = tetgen_mesh(plc)
```

Facets may be planar polygons. `seeds` maps original integer region IDs to
one or more XYZ interior coordinates; `holes` is `(H,3)` XYZ coordinates.
Every material compartment needs a seed and a separating interface. A seed
alone cannot create a new material boundary. `region_volumes`, if supplied,
are acceptance references, not a way to resize geometry. User-provided surfaces
must be nonintersecting and valid; TetGen performs its PLC checks in addition
to shape/index/planarity checks in Python.
支持平面多边形面。种子不能凭空建立材料边界；每个分区必须已有真实界面。
参考体积用于检查，不用于改变几何。用户面必须满足有效 PLC 条件。

Imported surfaces can be read with meshio, then passed as triangle facets:
外部面网格可以用 meshio 读取，然后显式设置种子和孔洞：

```python
import meshio
surface = meshio.read('/external/closed_surface.stl')
# STL often repeats vertices: merge exactly identical vertices first.
# STL 通常重复存储顶点；这里只合并完全相同点，不用任意容差焊接裂纹。
points,inverse = np.unique(surface.points,axis=0,return_inverse=True)
facets = inverse[surface.get_cells_type('triangle')]
plc = SurfacePLC(points,facets,seeds={1:[[5e-6,5e-6,5e-6]]})
# STL contains no guaranteed length unit; convert coordinates to metres first.
# STL 无可靠长度单位；构建 PLC 前必须自行转换到米。
```

## 6. Options, quality and failure handling / 参数、质量及失败处理

|`TetGenOptions` field / 参数|Default / 默认|Meaning / 含义|
|---|---|---|
|`radius_edge_ratio`|2|Requested circumradius / shortest edge / 请求外接球半径边长比|
|`minimum_dihedral`|0°|Requested minimum angle,0..20 / 请求最小二面角|
|`maximum_volume`|None|Maximum Tet4 volume in **m³**, not µm³ / 最大单元体积，米³|
|`region_maximum_volumes`|{}|Original region ID → max volume / 分相体积上限|
|`maximum_steiner`|300000|Native added-point budget / 新增点预算|
|`maximum_cells`|2000000|Output acceptance cap, checked after native solve / 输出单元上限|
|`merge_coplanar`|True|False adds M to retain every facet / 是否合并近共面面片|
|`preserve_surface`|False|`Y`; forbids boundary splitting and may impede quality / 不细分表面|
|`facet_separation_angle`|179.9°|p-angle for facet merging; for_adaptive uses179.99999° / 面片合并角|
|`coplanarity_tolerance`|1e-12|Normalized TetGen T tolerance, independent of output volume acceptance / 内部共面容差|
|`consistency_check`|True|TetGen `C` consistency checks / 内部一致性检查|
|`require_quality`|False|Reject measured quality missing requested ratios/angles / 实测不达标则拒绝|
|`isolated`|False; True for `for_adaptive()`|Native worker process; preserve caller on native abort / 原生进程隔离|
|`native_timeout_seconds`|180|Wall-time budget of an isolated call / 隔离调用的时间预算|

Requested `q` bounds can be unattainable at small PLC angles or when a Steiner
budget is exhausted. A returned valid mesh is not automatically a high-quality
mesh: inspect `report['quality_limits_satisfied']`, minimum dihedral and radius
ratio. For strict projects enable `require_quality=True`. Radius ratio alone
does not detect all slivers. The report measures both ratio and angle.
小输入角或点预算可能阻止质量目标。合法网格不等于质量已达标；严格项目必须打开质量门槛。
半径边长比不足以识别所有薄片单元，因此同时测量二面角。

Maximum-volume constraints are **checked**. Small residual violations trigger
up to three progressively finer native calls; the user's acceptance threshold
is unchanged. Unresolved violations raise an error. Exact-PLC region-volume,
positive-volume, seed/region and element-count checks always apply.
最大体积不是仅传给后端就算完成；实测超限时收紧输入尺寸重算，最多三次，仍失败则报错。

`maximum_cells` is a post-solve output cap, not a native memory allocator cap.
For a hard workstation limit use the local runner; the native mesher can need
more intermediate memory than the final arrays. Input facet limits are checked
before native meshing. Use one job, bounded surfaces and inspection of resource
manifests before scaling up.
单元上限是求解后门槛，不是硬内存上限；硬内存与时间限制应使用资源启动器。

For a notebook or service, prefer `TetGenOptions.for_adaptive()` or explicitly
set `isolated=True`. Native failures then raise a Python exception instead of
terminating the notebook kernel. This path uses temporary HDF5 files in the
external workspace and needs h5py. It does not relax any geometric checks and
does not impose an independent memory cap; use the process-tree resource
launcher as well. Low-level in-process execution remains available.
Notebook或服务建议使用隔离选项；不放宽验收，也不代替整个进程树的内存限制。

## 7. TOML and CLI / TOML 与命令行

See the executable recipe [`configs/geometry_tetgen.toml`](../configs/geometry_tetgen.toml).
参见可执行配置文件。几何与网格配方共用一个 TOML，输出目录由调用者指定。

```python
from morqex.modeling import from_toml,mesh_from_toml
model = from_toml('/home/zephyrain/lw/MORQEX/configs/geometry_tetgen.toml')
solid = mesh_from_toml(model,'/home/zephyrain/lw/MORQEX/configs/geometry_tetgen.toml')
solid.save_hdf5(work/'from_toml.h5')
```

```bash
"$PY" -m morqex.modeling mesh "$MORQEX_WORKSPACE/my_geometry.h5" \
  --method tetgen --regions 1,2 --surface-method voxel_plc \
  --output "$MORQEX_WORKSPACE/tetgen_mesh.h5" \
  --comsol "$MORQEX_WORKSPACE/tetgen_mesh.mphtxt"
```

Use `--regions all` to mesh every present label. CLI exposes the common options;
Python/TOML exposes the full typed option object. No COMSOL runtime is invoked.
所有标签用 `--regions all`；完整质量控制用 Python/TOML，CLI 不会启动 COMSOL。

## 8. COMSOL import and material/boundary mapping / COMSOL 适配

```python
from morqex.modeling import export_comsol,read_comsol
path = export_comsol(solid,work/'electrode.mphtxt',
                     region_names=model.region_names,chunk_size=50000)
back = read_comsol(path)
print(back['points'].shape,back['elements']['tet']['cells'].shape)
# Identical API for voxel Tet4, Hex8 and a streamed HDF5 MeshArtifact:
# 体素 Tet4、Hex8、流式 HDF5 使用同一个导出入口。
# artifact.export_comsol(work/'streamed.mphtxt')
# voxel_mesh.save_comsol(work/'voxel.mphtxt')
```

Output pairs / 输出文件：

- `electrode.mphtxt`: COMSOL Mesh-class4 ASCII, vertices and Tet4/Hex8, material
  domain IDs. TetGen results additionally retain marked triangular outer and
  internal material interfaces. 单元与域编号，并保留 TetGen 的外边界及内部材料界面三角面。
- `electrode.regions.json`: original region → COMSOL domain, facet marker →
  boundary entity, names and unit statement. 原始分区、域、边界与单位的明确对照。

COMSOL requires domain entity numbers starting from**1** and boundary entities
starting from**0**. Original region0 is remapped explicitly, never erased.
The internal coordinate values remain SI metres. `length_unit='m'|'cm'|'mm'|'um'|'nm'`
sets exported numerical coordinates, with17 significant digits. For example a
1m edge is1 in m,1000 in mm, or1000000 in um. This changes representation, not
physical geometry. The file comment and `.regions.json` record the required
software import unit; a mesh text file does not set a COMSOL physics-model unit.
内部统一用米；导出单位可配置，只换算坐标数值，不改变实际尺寸。导入软件时选择同一单位。
Hex8 vertex
ordering is converted from VTK to COMSOL. Large ASCII output is written in
chunks; it can still be much larger on disk than HDF5. Check disk capacity
before exporting a complete hundred-million-element mesh.
COMSOL 的域从1开始，边界从0开始；原始标签0显式映射而不删除。Hex8 节点顺序转换，
ASCII 分块写入避免大字符串，但磁盘体积仍可能非常大。

In COMSOL / 在 COMSOL 中：

1. Create a3D component and a Mesh node; add **Import**.
   新建三维组件，在网格下添加“导入”。
2. Choose `electrode.mphtxt`, set the length unit to the selected `length_unit`
   (default **m**), and build. 选择与导出相同的单位，默认米。
3. Use **Boundary partitioning → Minimal** to preserve the supplied partition;
   inspect the entity IDs against the JSON map before assigning physics.
   使用最小边界划分，并根据 JSON 核对材料域及边界编号后设置物理场。
4. A domain-only voxel export lets COMSOL create missing boundaries, edges and
   points. Their IDs are generated by COMSOL and are not promised to equal
   TetGen facet markers. By default the writer now emits actual **Selection0**
   objects for every domain, plus optional aggregate `selections` such as
   cathode, anode, separator and electrolyte. These are present inside the
   `.mphtxt`, not only the JSON. `include_region_selections=False` disables
   automatic selections. 现在文件内含真正的域命名选择对象，也支持多个域组成的聚合选择；
   仅体单元路线由 COMSOL 重建低维实体，低维编号不能预设。
5. The result is a mesh, not a configured battery/structural MPH model.
   导入的是几何网格，不是已经设置好电池或结构物理场的 MPH 模型。

The automated checks independently read back coordinates, connectivity,
material IDs and boundary IDs. **A COMSOL application is not installed on this
machine; actual GUI import/solve has not been executed.** The strict reader
supports Mesh4 plus Selection0 and3D linear element blocks, including the
user-supplied independent934288-Tet4 reference; not arbitrary COMSOL physics
models or higher-order elements. This limits the validation claim explicitly.
自动验证包括格式及数值读回；本机没有 COMSOL，因此没有宣称已通过实际 GUI 导入与求解。
独立读回器只支持本导出器的三维线性子集，不是所有 COMSOL 格式的通用解析器。

## 9. Notebook2D/3D preview / Notebook 可视化

```python
from IPython.display import display
# Complete original voxel array retained; interactive3D uses labelled display LOD.
# 原始分区完整保留；三维体素预览使用明确标注的显示降采样。
voxels = model.show(maximum_3d_size=48,title='Source labels / 原始分区')
display(solid.viewer(cell_count=2000,title='Actual TetGen elements / 真实 TetGen 单元'))
voxels.save_html(work/'source.html')
solid.viewer(cell_count=2000).save_html(work/'actual_tetgen.html')
```

The TetGen viewer shows actual stored elements. Its cell budget is a display
subset, not a claim that the whole volume was solved or reduced to that many
cells. Keep full HDF5/XDMF for ParaView and numerical work.
预览显示真实单元子集，显示预算不改变完整网格；全部体网格保存在 HDF5/XDMF。

## 10. Provenance and references / 来源与许可

Bundled TetGen1.6.0 remains under its original license. The official project
identifies that version as AGPLv3 and offers a separate commercial licensing
route. The license text/provenance is retained beside the source; see
[`THIRD_PARTY_NOTICES.txt`](../THIRD_PARTY_NOTICES.txt). A compiler flag does not
change third-party licensing obligations.
原始 TetGen 源码及许可保留；编译开关不改变第三方许可义务。

- [TetGen official project, downloads/licenses and technical paper](https://tetgen.github.io/)
- [COMSOL Mesh-class4 format and entity numbering](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.35.html)
- [COMSOL native mesh import, domain and boundary transfer](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_mesh.24.32.html)

## 11. Reproduce installation and geometry checks / 复现实测

```bash
# Packaging tests create pure→native→pure wheels and isolated target installs.
# 打包测试创建三类 wheel，分别安装到外部独立目录，不修改现有环境。
python scripts/run_local.py --workspace "$MORQEX_WORKSPACE" \
  --mesh-output --label tetgen_install_check -- \
  "$PY" /home/zephyrain/lw/MORQEX/scripts/validate_tetgen_installation.py

# Use a new output folder to retain existing validated meshes.
# 新目录保留旧结果，禁止覆盖已验收的网格。
export MORQEX_TETGEN_OUTPUT="$MORQEX_WORKSPACE/tetgen_repeat_001"
python scripts/run_local.py --workspace "$MORQEX_WORKSPACE" \
  --mesh-output --label tetgen_geometry_check -- \
  "$PY" /home/zephyrain/lw/MORQEX/scripts/validate_tetgen_geometry.py
unset MORQEX_TETGEN_OUTPUT
```

The geometry protocol meshes all three phases of10 complete32³ morphology
families, one complete256³ single-phase irregular structure with contour
step1, and the24³ TOML case. It saves actual meshes, COMSOL read-backs, per-case
quality reports and offline HTML. This is not a claim that ten256³ TetGen
meshes were computed: the earlier ten256³ atlas uses voxel-split Tet4.
验证包含十类完整32³多相结构、一个完整256³单相全分辨率轮廓、一个24³配置示例。
原有十类256³图库仍是体素剖分路线，不能混称为十个256³ TetGen 结果。
This is geometry qualification, not validation of battery mechanics or fatigue.
这属于几何及格式验收，不是电池力学或疲劳模型验收。

Reopen a stored mesh without reading all connectivity / 不读取全部单元重新打开网格：

```python
from morqex.modeling import MeshArtifact
stored = MeshArtifact.load(work/'multiphase.h5')  # or any complete mesh HDF5
stored.viewer(cell_count=2000,sampling='uniform').display()
# Original-volume/TetGen element IDs are retained in the display payload.
# 显示载荷保留原始单元编号，抽样只用于显示。
```


## 12. Recorded Linux qualification / 本次 Linux 实测范围

The external `tetgen_current/validation.json` and resource manifests record the
actual runs; these numbers describe this workstation and geometry protocol.
以下是本次实际记录，不能推广为其他网格或求解任务的性能保证。

|Check / 检查|Observed / 实测|
|---|---|
|Related automated regression / 相关自动回归|42 passed / 42项通过|
|Installation / 安装|Pure→compiled→pure wheels built and independently installed; native API called; editable native build correctly rejected / 三种wheel实际构建安装并调用|
|Multiphase / 多相|10 complete32³ geometries; every selected phase volume matched within1.6e-15 in these runs / 十类完整结构逐相体积核对|
|Full-resolution contour / 全分辨率轮廓|256³ original array, step1,454095 Tet4; contour volume differs from voxel union by−0.04924% / 完整256³原始场，非裁剪区域|
|Full256 meshing API time / 完整256网格接口耗时|About17.4s; includes contour construction and validation, excludes report/export / 约17.4秒|
|All12 geometry/export cases / 全部12例生成与导出|About43.3s, sampled peak process-group RSS about1.12GB, one CPU / 单核、约1.12GB峰值内存|
|Notebook / 笔记本|8 code cells executed,0 errors / 8个代码单元无错误|
|Browser / 浏览器|6 actual-mesh views checked using software rendering / 六种实际网格视图通过|

Geometry, region conservation and native-format read-back passed. **Ten of the
12 complex cases did not meet the requested radius/dihedral quality setting**;
this is recorded as `quality_limits_satisfied=false`. They are valid labelled
geometry/export examples, not certified reference-quality FE discretizations.
A mechanics study must use its own strict quality, conditioning and convergence
gates; `require_quality=True` rejects unmet shape targets. No failed shape
criterion is converted into a pass by reporting only average element quality.
几何、体积及文件读回通过，但12例中10例未达到请求的单元形状质量，已经逐项明确标记。
这些是合法的分区/导出示例，不是已通过参考力学离散验收的网格；实际求解需严格质量、
条件数及收敛门槛，不能仅凭平均质量掩盖最差单元。
COMSOL GUI import is untested because the application is absent.
本机没有 COMSOL，实际 GUI 导入未测试。
