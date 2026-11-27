# Complex CT, dense electrodes and software exchange / 复杂CT、致密电极与软件交换

This module keeps source labels, physical size, surface approximation, volume
meshing and display resolution separate. 本模块分别管理原始分区、真实尺寸、曲面近似、
体网格和显示精度；显示抽样不改变源体素或有限元数据。

## 1. Units and independent references / 单位与独立参考

Arrays, spacing and origin use **ZYX**. Mesh points use **XYZ, metres**.
数组、体素边长及原点均为ZYX；有限元节点为XYZ，内部单位统一为米。

Portable region IDs are integers in `[0,2147483647]`; zero is a valid material.
Larger IDs are rejected before meshing instead of wrapping into negative phases.
To use an external identifier space, explicitly compact labels and retain its map:
区域编号采用非负有符号32位范围，0也是有效材料。较大外部编号应显式映射，不能直接
强制转换为int32而导致溢出；保存映射以便恢复原始标识：

```python
import numpy as np
from morqex.modeling import from_regions
# Replace this example with your own 3D partition array / 替换为自己的三维分区数组。
user_region_array = np.full((16,16,16),2**40,dtype=np.uint64)
user_region_array[4:12,4:12,4:12] = 2**40+7
original_ids, inverse = np.unique(user_region_array,return_inverse=True)
compact = inverse.reshape(user_region_array.shape).astype(np.int32)
model = from_regions(compact,spacing=(0.5e-6,)*3,
    metadata={'external_region_id_map':{int(i):int(v) for i,v in enumerate(original_ids)}})
```

The supplied `NMC.tif` and `TetMesh-zjl_mesh2.mphtxt` are independent geometries.
The COMSOL model has approximately1m extent in all directions, as confirmed by
the owner. It has158385 nodes,934288 Tet4 and three named domain selections.
Do not use it as a pointwise FEM reference for the TIFF. GEMsim's original
`TriMesh.from_vol` normalizes mesh coordinates by `vxl_len`; it does not infer
micrometres. 这两个文件不是配对数据，不可计算二者的逐点重建误差；旧GEMsim的归一化
坐标不能自动解释成微米。以下TIFF示例0.5µm仅为可修改的用户建模输入，不是扫描标定。

```python
from pathlib import Path
from morqex.modeling import tiff_voxels, import_comsol
work = Path('/home/zephyrain/lw/morph_workspace/geometry_examples')
work.mkdir(parents=True, exist_ok=True)
ct = tiff_voxels('/home/zephyrain/lw/NMC.tif', spacing=(0.5e-6,)*3)
# Original gray0/128/255 remain distinct; do not invent their chemical roles.
# 原始灰度保持不变；若有可靠分割说明，再明确指定region_names。
reference = import_comsol('/home/zephyrain/lw/TetMesh-zjl_mesh2.mphtxt', length_unit='m')
reference.adaptive_viewer(title='Actual independent COMSOL reference').save_html(work/'reference.html')
ct.save(work/'ct.h5')
ct.viewer(maximum_3d_size=64).display()  # notebook: slices + rotatable3D / 切片与三维
# Explicitly isolate a label in3D;2D still shows all original labels.
# 三维单独显示指定相；二维仍显示完整原始切片。
ct.viewer(maximum_3d_size=64, show_region=128).display()
```

TIFF import never rescales labels, fills cavities or smooths the source array.
It requires explicit physical spacing and either ZYX metadata or a supplied
`axes_to_zyx` permutation. The original source SHA256 is recorded.
TIFF导入不重标记、不填孔、不平滑；源文件哈希、轴和尺寸假设均可追踪。

`import_comsol` converts a MORQEX export's declared file unit to SI by default.
An unrelated file without that comment requires `length_unit` explicitly; a
caller unit conflicting with a declared file unit is rejected. Unit aliases
`um` and `µm` denote the same scale. `read_comsol` intentionally returns file
numbers without converting them. 导入自有导出文件时可读取明确的单位注释；外部无单位
文件必须传入单位，不能根据数值大小猜测。读取原始数值与转换为SI是两个不同入口。

## 2. Five native256³ morphology families / 五种原生256³结构

```python
from morqex.modeling import coated_electrode, dense_electrode
families = ['granular','rods','platelets','agglomerates','bicontinuous']
views = {}
for family in families:
    model = coated_electrode(family, size=256, seed=20261127,
        active_fraction=0.50, cbd_thickness_voxels=3, pore_buffer_voxels=7)
    model.save(work/f'{family}.h5')
    view = model.viewer(maximum_3d_size=64, title=family)
    view.save_html(work/f'{family}.html')
    views[family] = view
views['rods'].display()
```

`coated_electrode` creates a resolved CBD coating and pore boundary buffer;
the resulting CBD fraction is measured, not forced. `dense_electrode` instead
sets exact active/CBD fractions with correlated patchy CBD and may create
difficult triple junctions. They are different explicit recipes, not repairs
silently applied to CT. Both rasterize the full256³ volume, without tiling a
smaller SVE. Granular/rod/platelet families use irregular rotated primitives;
agglomerates use clustered subparticles; bicontinuous uses a correlated3D field.
涂层路线具有明确CBD厚度，CBD体积分数以实际生成值为准；斑块CBD路线可以设定体积分数，
但更难网格化。二者都是明确可选的生成配方，不会用来静默修改导入CT。

These are synthetic morphology examples, not validated reconstructions of
specific NMC chemistry or CT statistics. Effective porous CBD does not imply
that its whole domain blocks ions. Transport coefficients require a separate
physical model. 这些为合成形貌示例，不能据此宣称重建了特定材料或获得真实输运物性。

### Generator parameters and size changes / 生成参数与尺寸变化

|Parameter / 参数|Meaning and constraints / 含义与约束|
|---|---|
|`morphology`|`granular`, `rods`, `platelets`, `agglomerates`, `bicontinuous`; Boolean unions can form connected bodies, so source-primitive count is not particle count / 形状布尔并可连通，生成源数量不是独立颗粒数量|
|`size=256`|Dense generators accept an integer48..256; user arrays and `from_shapes` support independent ZYX dimensions / 致密预设为立方体，自定义数组可用独立三向尺寸|
|`spacing=5e-7`|Positive isotropic voxel edge in metres; import anisotropic arrays with `from_regions(...,spacing=(hz,hy,hx))` / 此处为米制标量，非等距数据通过数组入口传入|
|`seed=20261127`|Local NumPy generator seed; coating perturbations use a deterministic offset / 固定局部随机种子|
|`active_fraction`|Default0.55 for `dense_electrode`,0.50 for `coated_electrode`; active voxel count must fit inside the pore buffer / 活性体素数量必须能容纳于缓冲区内|
|`cbd_fraction=0.11`|`dense_electrode` only; positive and smaller than `1-active_fraction`. Coated CBD fraction is measured instead / 仅斑块CBD配方可直接设定|
|`particle_scale_voxels=17`|Reference scale at256³. Internally `max(3,particle_scale_voxels*size/256)` is used; reference range3..51.2. It is a family scale, not an exact particle radius / 以256³为参考随尺寸缩放，不是每个颗粒的精确半径|
|`irregularity=0.22`|`dense_electrode` keyword, range `[0,0.35)`; coating currently inherits0.22. Use `Shape` for independently controlled primitives / 斑块配方可设不规则度，涂层配方当前继承默认值|
|`cbd_thickness_voxels=3`|Coating recipe; at least2 / 涂层至少两层体素|
|`pore_buffer_voxels=7`|Coating recipe; `cbd_thickness_voxels < pore_buffer_voxels < size/8` / 缓冲层必须比涂层厚|

The coating also requires
`round(active_fraction*size**3) <= (size-2*pore_buffer_voxels)**3`.
Defaults0.5/7 require `size>=68`. A smaller valid example is below; it is a
separate example, not a substitute for full256³ validation.
涂层配方还有体积分数约束，不能只修改尺寸而保留不可能容纳的缓冲区；以下64³示例合法：

```python
small = coated_electrode('rods', size=64, seed=19,
    active_fraction=0.5,cbd_thickness_voxels=2,pore_buffer_voxels=5)
small.viewer(maximum_3d_size=64).display()
```

The synthetic coating algorithm replaces pore components smaller than27 voxels
with CBD and records the count in
`synthetic_unresolved_pore_voxels_filled_with_CBD`. This threshold is currently
an implementation constant, not an exposed keyword. It is never applied by
TIFF import. For a different construction law, supply your own region array
or register a generator. 当前涂层生成器包含明确记录的小孔处理；不能将它误写成原始
CT的默认修复，也不能把生成源数量当作分割得到的真实独立颗粒数量。

## 3. Stack independently named regions / 电池分层与独立材料域

```python
from morqex.modeling import uniform_layer, porous_layer, stack_layers
h = (0.5e-6,)*3
cathode = coated_electrode('granular', size=256, seed=41)
anode = uniform_layer((256,256,48), spacing=h, name='lithium_metal')
separator = uniform_layer((256,256,24), spacing=h, name='separator_effective_porous')
electrolyte = uniform_layer((256,256,4), spacing=h, name='negative_electrolyte')
cell = stack_layers([('anode',anode),('negative_electrolyte',electrolyte),
    ('separator',separator),('cathode',cathode)], axis=2)
print(cell.metadata['layer_region_ids'])
print(cell.metadata['layer_slices'])
cell.viewer(maximum_3d_size=64).display()
```

`axis` is a ZYX array axis. Spacing and transverse dimensions must match;
implicit resampling is forbidden. Every layer×phase obtains a distinct global
domain ID, even if two layers use the same local ID. Aggregate selections
identify each layer. `porous_layer` optionally resolves a correlated pore
network; `uniform_layer` is an explicitly effective-medium layer. Current
collectors are additional uniform layers. 层×相独立编号；相同原始标签不意味着跨层合并。
均匀隔膜是等效介质，不能当作已解析的孔隙结构；可选porous_layer显式生成孔隙。

## 4. Surface and FEM routes / 曲面与有限元路线

| Route / 路线 | Contract / 适用范围 |
|---|---|
| `stream_tetrahedralize` | Exact voxel cells, regular Freudenthal subdivision; not a smooth adaptive mesh / 精确体素单元，规则剖分 |
| `voxel_plc` + TetGen | Shared planar voxel interfaces, unstructured volume / 共享阶梯界面，非结构体网格 |
| `marching_cubes_plc` | One closed phase, explicit contour approximation / 单相闭合近似曲面 |
| `adaptive_mesh` + ACVD | GEMsim residual-weighted surface reduction + TetGen, retries and measured geometry errors / 残差加权曲面重网格 |
| `surface_nets_plc` | Shared multi-label VTK interface, optional bounded volume correction / 共享多相界面 |
| `cut_cap`, `slab_plc`, `assemble_slabs` | Same shared surface, common constrained cuts, conforming whole-volume assembly / 共享切面、严格匹配的全体积拼接 |
| `remesh` + MMG5.8.0 | Adapt an existing labelled Tet4 mesh with explicit geometry/size requests / 对已有分区四面体网格重新调节疏密 |

No route may delete invalid faces/tetrahedra to manufacture a passing result.
An ACVD topology check alone does not rule out self-intersections. TetGen
rejection is a failed candidate, preserved in logs. A more resolved route can
cost substantially more memory and output. 不能删坏面或坏单元制造通过；保持拓扑并不自动
保证无自交。相交、丢域、负体积或切面不匹配均必须拒绝。

```python
from morqex.modeling import surface_nets_plc, AdaptiveMeshOptions, adaptive_mesh
front = surface_nets_plc(ct, conserve_volumes=True)
# This is a surface preview, not an accepted FEM volume.
# 这是曲面预览，不代表已生成合格体网格。
front.viewer(maximum_embedded_faces=250000).display()
```

The optional phase-volume correction constrains global geometric phase
volumes, bounded vertex displacement and face orientation; it does not prove
pointwise geometry, contact topology or PDE solution accuracy. 相体积校正仅约束全局几何量，
不能据此声称逐点几何或有限元解误差达到同一数值。

For very large inputs use `scripts/partitioned_geometry_gallery.py` as the
serial resource-limited example. Every adjacent slab must have identical
material-labelled cap triangles and shared global node IDs; matching only
coordinates is insufficient. Artificial cut faces are removed from physical
boundary groups after assembly. 大模型分块不是互相独立的近似子模型：必须共用节点和切面
三角形，最终统一为一个连续网格；人为切面不能作为物理材料边界输出。

The packaged API accepts any supplied complete region array, rather than
requiring a gallery case name. 统一API不依赖图库名称，也不需要为新数组改粒子代码：

```python
from morqex.modeling import partitioned_tetgen
# Run through run_local's reference-memory profile for large arrays.
# 大数组在外部资源限制器下运行；全部相（包括电解液）均参与完整域剖分。
artifact = partitioned_tetgen(ct, work/'complete_ct',
    surface_method='voxel_aligned', slab_depth=32, axis=0,content_addressed=True)
# surface_nets is the separate smoothed shared-interface choice.
# axis uses ZYX; anisotropic spacing and nonzero origins are retained.
# 可选surface_nets光滑共享曲面；保留各向异性体素尺寸和非零原点。
from morqex.visualization.adaptive_mesh import AdaptiveMeshView
AdaptiveMeshView(artifact, maximum_embedded_faces=450000,
    node_cache_bytes=256*2**20).save_html(work/'complete_ct.html')
```

```bash
python -m morqex.modeling partition /external/geometry.h5 \
  --directory /external/complete_mesh --surface-method voxel_aligned \
  --slab-depth 32 --axis 0 --content-addressed --preview /external/complete_mesh.html
```

With the default `content_addressed=False`, `directory` must initially be empty
and outside source. Configuration, exact
source-array checksum, native logs, actual cap reconciliations and accepted
parts are retained. Repeating the same call resumes unchanged accepted work;
a changed array or recipe requires a new directory. Workers are serial and
have explicit native timeout/cell budgets. Failure retains the checkpoint and
does not return an accepted mesh. A low resource budget may require smaller
`slab_depth`; this changes the numerical partition plan, not the source geometry.
首次目录须为空；哈希防止错用旧网格。失败保留断点而非返回假结果；通用接口不等于
任何分割都必然成功，遇到病态拓扑仍会给出明确的失败日志。
Core algorithm/backend fingerprints also prevent reuse after a mesher source
change. Internal cuts automatically avoid original vertices within a bounded,
deterministic search; moving a computational cut does not move a material
interface. This plane adjustment applies to `voxel_plc` and `surface_nets`;
`voxel_aligned` instead uses exact integer voxel boundaries as described below.
核心源码改变会使旧断点拒绝复用；前两种路线自动微调的是计算分块平面，而非物理界面。

### Exact lattice-aligned partitioning / 精确体素边界分区

`surface_method="voxel_aligned"` partitions the **original array** at integer
voxel faces. Each subblock uses exact shared material quads and a canonical
diagonal, so both sides construct the same triangular cap independently.
If adjacent voxels across a cut have different material IDs, assembly retains
that triangle **once as a real material interface**. Equal-material cap faces
are removed as computational boundaries. No material is filled, resampled or
assigned by a nearest-centroid guess.

该模式在原始数组的整数体素边界分区，避免半体素切面切穿三角形后产生额外短边。
每个四边形都使用字典序最小顶点对应的对角线；两侧实际节点和三角面必须一致。
切面两侧若材料不同，该面保留一次作为真实材料界面；相同则只删除计算分区面。
原始数组、物理尺寸、各向异性间距及非零原点均保留，不会为剖分成功擅自填孔。

Accepted output additionally checks positive cell volumes, exact source phase
volumes (relative tolerance `1e-8`), and the original voxel label at **every**
cell centroid. The centroid check is a validation cross-check, **not** the
material-assignment algorithm or a rigorous pointwise surface-error proof.
All physical faces can be independently derived from cell ownership in the
small regression benchmarks, including material jumps on computational cuts.
相标签仍来自TetGen的区域属性；逐单元中心检查不能替代PLC共形证明，也不等于有限元
离散收敛。体素阶梯几何中的点/边接触仍可能不可剖分；实际切面若出现不一致的强制恢复点，
接口会拒绝返回合格结果。该模式不承诺任意分割都成功，也不保证请求的单元形状上限。

Use `region_labelling="seeds"` for the conservative exact-voxel route. All
present labels, including electrolyte/pore regions, are meshed. Per-region
maximum-volume keys are checked globally and filtered only for absent labels
in a particular subblock. Choose a **new** or content-addressed cache when
changing methods; the original `voxel_plc` clipped-cap route remains available.
保守的精确体素模式默认使用连通域内部种子，支持各相不同的体积控制。不存在的全局相ID
会报错，不会静默忽略。新模式与旧的裁切PLC、光滑Surface Nets路线分别保留。

```toml
[meshing]
method = "partitioned_tetgen"
directory = "/external/aligned_exact_mesh"
content_addressed = true
surface_method = "voxel_aligned"
slab_depth = 32
axis = 0
[meshing.tetgen_options]
preserve_surface = true
merge_coplanar = false
region_labelling = "seeds"
coplanarity_tolerance = 1e-15
maximum_cells = 8000000
maximum_steiner = 800000
require_quality = false  # Report shape failures; true rejects them / true则拒绝形状未达标结果
```

`require_quality=false` does **not** mean the quality request passed. Inspect
`quality_limits_satisfied`, the measured radius-edge maximum and minimum
dihedral; use `true` whenever downstream analysis requires those bounds.
不能把正体积或体积守恒当作形状达标，更不能仅据此宣称物理求解收敛。

The independent layered-conduction regression solves P1 finite elements on the
generated unstructured mesh and compares nodal potential, total current and
Dirichlet quadratic functional (half the dissipation, not stored energy) with the series-layer analytic solution. It tests
all three partition directions and a material jump exactly on a cut, with
`1e-9` potential/current/energy tolerances. This is a transmission/patch test,
not convergence evidence for an arbitrary electrode PDE.
分层导电回归独立装配P1有限元，检查电势、两端电流、自由节点残差和二次能量；材料界面
恰在计算分区面上，三个方向均验证。导电二次泛函是耗散的一半，不是存储自由能；解析比较不能替代真实电极的网格收敛实验。

To reproduce all exact-route gallery stages under the serial local resource
guard, use the controller below. It freezes the source, retains logs for each
case/stage, verifies every selected whole-model software export, and generates
`nmc_complex_gallery/voxel_aligned.html`. Large inputs must already exist in
the external complex gallery; initial generation is described above.
下面的控制器串行执行，默认一个CPU、低优先级、6GiB进程组内存上限、8GiB主机余量，
不使用GPU；每个导出格式单独验收，任一阶段失败会保留断点并停止接受结果。

```bash
python scripts/run_aligned_geometry.py \
  --workspace /home/zephyrain/lw/morph_workspace/microelectrode_gallery \
  --cases nmc granular rods platelets agglomerates bicontinuous cell_stack \
  --stages mesh preview exchange vtk \
  --formats comsol vtu gmsh medit abaqus --length-unit um --slab-depth 32
```

For generation only use `--stages mesh`; for existing meshes use
`--stages preview exchange`. The fixed gallery checkpoint directory belongs
to its recorded source/algorithm; archive it explicitly or choose a fresh
gallery workspace after changing the core algorithm. For general API use,
`content_addressed=True` automatically separates new structures and parameter
sets without any manual geometric repair.
只剖分、只可视化或只导出均可选择。图库断点绑定源码版本；通用API的内容寻址模式则
自动管理新结构和参数，不需要每次手动修几何。更改算法后不能把旧验收当作新验收。

The optional final `vtk` stage uses the **compiled VTK XML reader** and compares
every coordinate, cell connection, offset, element type, domain and boundary
tag against the complete HDF5 mesh. It requires the `vtu` export and VTK.
This is an independent application-library readback; it is not a COMSOL or
Abaqus solver execution. 可选`vtk`阶段使用真实VTK C++读取库，逐项核对完整数据，
保留独立JSON证据；不将其冒充COMSOL/Abaqus实际求解验证。

### Independent numerical verification / 独立数值验证

`scripts/benchmark_geometry_transmission.py` solves a classical spherical
conductivity transmission problem on three voxel resolutions. The analytic
potential is prescribed on the unit-cube boundary; potential and normal flux
are continuous on the exact sphere. The P1 solve uses Jacobi-preconditioned
CG (`rtol=1e-11`), with an independently checked free-node residual.
该脚本在三个体素分辨率上求解经典球形夹杂导电问题：解析电势作为外边界条件，真实球面上
电势与法向通量连续。输出完整节点电势VTU、误差、二次Dirichlet泛函及独立HTML。

```bash
WORK=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
PY="$WORK/runtime/bin/python"
python scripts/run_local.py --workspace "$WORK" --label geometry_scalar_transmission \
  --reference-memory -- "$PY" "$PWD/scripts/benchmark_geometry_transmission.py" \
  --sizes 16 32 64
```

The measured errors include **both voxel geometry and P1 approximation**;
four-point quadrature gives estimates, not rigorous norm bounds. The Dirichlet
functional equals half the dissipation in this conductivity problem and is not
stored thermodynamic energy. This benchmark does not qualify every full
electrode mesh and does not simulate NMC charge/discharge.
这里不能将体素几何误差当作纯求解器误差，也不能将数学验证冒充真实电极的收敛认证或
NMC充放电结果。严格单元形状请求、界面/体积检查和PDE收敛仍分别报告。

With `content_addressed=True`, `directory` is a cache parent. Each distinct
source array, parameter set and algorithm/native-backend fingerprint gets a
separate hash subdirectory. An unchanged call resumes that same subdirectory;
other validated states remain untouched. This storage choice does not alter
the physical problem. 启用后更换结构、精度或升级源码会自动进入新目录，不会要求清空
旧计算，也不会将不兼容的旧网格冒充新结果；CLI发布文件仍使用显式指定的文件名。

```toml
[meshing]
method = "partitioned_tetgen"
directory = "/external/whole_mesh"
content_addressed = true
surface_method = "surface_nets"
slab_depth = 32
axis = 0
[meshing.tetgen_options]
preserve_surface = true
merge_coplanar = false
region_labelling = "interfaces"
coplanarity_tolerance = 1e-15
maximum_cells = 8000000
maximum_steiner = 800000
```

The partition gallery processes complete labelled domains, including pore/
electrolyte labels; it does not silently fill holes in a selected solid-only
PLC. TetGen can retain mandatory recovery points despite `-Y`; the controller
therefore compares the **actual** caps, reconciles their physical PSLG and
recomputes only affected parts until node/triangle/material equality holds.
An unresolved mismatch rejects assembly. Tiny trailing slabs are merged to
avoid unnecessary thin-domain quality refinement. 分块示例要求完整分区（包括孔隙），
不为仅选固相的空洞PLC擅自填充材料。原生恢复新增节点时必须协调实际切面，不能只凭
同一输入切面就宣称共形；不匹配则拒绝拼接。

## 5. Configurable export units and real selections / 可配置单位与真实选择集

### Same voxels, different TetGen precision / 同一体素的网格精细度控制

Surface approximation and interior refinement are independent controls. A
smaller cell volume cannot recover a surface feature already removed during
coarsening. Conversely, a highly resolved surface can force many tetrahedra
even when the requested interior is coarse. **Coarse meshing may fail rather
than erase a thin CBD bridge or tiny isolated phase.** 曲面保真与内部尺寸分别控制；
加密体单元不会找回已被粗化删去的曲面细节。细界面也可能强制产生大量单元，因此不能
保证任意细微结构都能压成很小的网格；失败时应提高分辨率，而不是静默删相。

```python
from morqex.modeling import MeshResolution, controlled_tetgen, TetGenOptions

# Keep the same model object. Do not downsample its source array.
# 保持同一个完整VoxelModel，不对源分区数组降采样。
def mesh_at_precision(model, precision, output):
    result = controlled_tetgen(model,
        resolution=MeshResolution(precision),  # coarse / balanced / fine / very_fine
        surface_method='gemsim_adaptive',
        tetgen_options=TetGenOptions.for_adaptive(
            maximum_cells=5000000, maximum_steiner=800000))
    result.save_hdf5(output)
    print(result.report['resolution_controls'])
    print(result.report['quality_limits_satisfied'])
    return result

# Explicit high-fidelity requests; run under the external resource guard.
# 自定义保真与内部尺寸；应在外部工作区、资源限制下执行。
custom = MeshResolution('fine', vertex_fraction=0.65,
    maximum_phase_volume_error=0.01, maximum_volume_m3=5e-18)
# A different request using a nominal equilateral-equivalent edge:
# 通过等边四面体等效边长定义体积门槛；并不是所有单元的最长边保证。
custom_edge = MeshResolution('fine', nominal_edge_m=0.8e-6)

# For exact stair-step voxel boundaries use surface_method='voxel_plc'.
# To keep a chosen smooth front end unchanged, build SurfacePLC once and call
# tetgen_mesh(front, options=TetGenOptions(...maximum_volume=...)) repeatedly.
# 精确保留体素阶梯边界用voxel_plc；固定光滑曲面则预先构建一次SurfacePLC，
# 再仅改变TetGenOptions.maximum_volume，避免曲面与体单元同时变化。
```

| Preset / 预设 | Retained surface fraction / 曲面顶点比例 | Allowed phase-volume approximation / 相体积近似门槛 | Default density target / 默认体单元密度目标 |
|---|---:|---:|---:|
| coarse / 粗 | .06 | .06 | 4000 |
| balanced / 均衡 | .12 | .03 | 16000 |
| fine / 精细 | .25 | .015 | 64000 |
| very_fine / 更精细 | .50 | .0075 | 256000 |

These are requests; topology-preserving retries may retain more surface
vertices. The target sets `maximum_volume=selected_source_volume/target_cells`,
so it is a **minimum density request, not an exact element count**. Actual
tetrahedra, measured geometry errors, source SHA256, source shape/spacing and
requested controls are recorded. `maximum_volume_m3` overrides the target;
`nominal_edge_m` sets (V_{max}=h^3/(6\sqrt2)). It is not a strict edge-length
or PDE-error bound. `require_quality=True` separately makes measured shape
limits mandatory. 所有预设均是请求；需要报告最终实际值与回退情况。精确度、质量、
最大单元数是独立约束，不会为了达到目标数量而放宽物理分区或验收条件。

```toml
[meshing]
method = "controlled_tetgen"
surface_method = "gemsim_adaptive"
[meshing.resolution]
precision = "fine"
vertex_fraction = 0.6
maximum_phase_volume_error = 0.01
nominal_edge_m = 8e-7
[meshing.tetgen_options]
maximum_cells = 5000000
maximum_input_cells = 30000000 # optional independent input limit / 可选的独立输入预算
maximum_steiner = 800000
isolated = true
```

```bash
python -m morqex.modeling mesh /external/small_geometry.h5 --method tetgen \
  --surface-method voxel_plc --precision fine --maximum-cell-volume 1e-20 \
  --maximum-cells 5000000 --output /external/fine.h5 \
  --comsol /external/fine.mphtxt --length-unit um
```

```python
from morqex.modeling import export_comsol, read_comsol, export_bundle
# reference.points are SI metres. / 内部坐标为米。
path = export_comsol(reference, work/'reference_mm.mphtxt', length_unit='mm',
    preserve_domain_ids=True, region_names={1:'domain_A',2:'domain_B',3:'domain_C'},
    selections={'electrode': {'regions':[1,2,3], 'label':'electrode'}})
back = read_comsol(path)
assert back['length_unit'] == 'mm'
import numpy as np
np.testing.assert_allclose(back['points']*1e-3, reference.points)
bundle = export_bundle(reference, work/'software', length_unit='um',
    region_names={1:'domain_A',2:'domain_B',3:'domain_C'})
```

`m`, `cm`, `mm`, `um` (`µm`) and `nm` are supported. COMSOL imports must use
that same selected unit; comments/JSON are not a replacement for the software
import setting. Mesh4 plus actual Selection0 objects preserve domain names and
aggregate sets. HDF5/XDMF always stays SI. VTU, Gmsh2.2, Medit and Abaqus have
no universal enforced length unit, so `bundle.json` is authoritative. Abaqus
uses C3D4, ELSET and true S1–S4 element faces; Gmsh writes physical groups.
全部软件共享一个长度单位声明；不能只改文件后缀。HDF5保持米，其他交换格式的单位由
manifest记录；COMSOL导入时必须匹配。域选择不是仅保存在侧面JSON中的字符串。

VTU uses standard appended-raw binary arrays with UInt64 block headers, and
Gmsh uses binary2.2. Medit and Abaqus use bounded ASCII writing/readback.
Every point, connectivity entry, material and interface reference is checked;
the large gallery additionally reads complete VTU files through the compiled
VTK reader in a fresh process. These routes avoid constructing a second giant
ASCII/XML mesh in memory. VTU is uncompressed; HDF5 remains the compact primary
archive. VTU与Medit采用分块核对，避免数千万单元导出时整块内存复制；VTK原生回读与
写入使用不同进程，防止两份完整网格同时占用内存。

The notebook saves an independent source-voxel snapshot beside each newly
generated volume mesh. Re-running with new geometry parameters never silently
reuses an old mesh. Notebook每次网格化都新建输出目录并保存对应源分区，避免参数已变、
体网格仍旧的错误。The HTML catalogue loads only the currently selected structure;
its surface-face display sample and complete Tet4 sections are explicitly
distinguished. 图库一次只加载当前结构；实际面片抽样与完整体单元切面分别注明。

CLI `generate` also reads `[exports].length_unit` and `formats`;
`--length-unit` overrides TOML. `[exports].directory` is resolved relative to
the TOML file unless `--export-dir` is supplied. `from_toml()` itself remains a
geometry-only API; Python callers explicitly call the mesh/export functions.
命令行读取导出配置，显式命令行参数优先；Python分阶段API保持各自职责。

The local COMSOL application is unavailable: format round trips are tested,
not GUI physics-model import or solving. Shape quality, geometric fidelity and
FE convergence are separate checks. 本机未安装COMSOL，不宣称已完成其GUI求解；形状质量、
几何误差和有限元收敛必须分别报告。

## 6. Installation, resources and provenance / 安装、资源与来源

Native compilation can also be selected explicitly by our own TOML/API. This
section never compiles as a side effect of `from_toml()` geometry generation.
也可以用统一TOML指定编译后端；生成几何不会偷偷执行编译阶段。

```toml
[native_build]
workspace = "/external/morqex_work"
backends = ["tetgen", "acvd", "mmg"]
```

```python
from morqex.modeling import build_native_from_toml
libraries = build_native_from_toml('/external/native.toml')
print(libraries)
```

```bash
python -m morqex.modeling build-backends configs/native_backends.toml \
  --workspace /home/zephyrain/lw/morph_workspace/microelectrode_gallery
```

Only named backends are built, serially, with one compiler job. The same file
may include geometry/mesh sections, but build and solve calls remain explicit.
仅编译指定内核，逐个、单编译任务执行；可与几何配置写在同一TOML，仍须分阶段显式调用。

```bash
# Optional native mesher. Build/cache outside source, one compiler job.
MORQEX_BUILD_TETGEN=1 MORQEX_WORKSPACE=/home/zephyrain/lw/morph_workspace/build \
  pip install '.[multimaterial-meshing]'
# Optional conforming slab caps: review Triangle's separate core terms first.
pip install '.[partitioned-meshing]'
```

See [TetGen build details](05_tetgen_and_comsol.md) for exact environment
variables supported by the build backend. VTK is BSD-3; Triangle's wrapper is
LGPL3 but its core restricts commercial inclusion. TetGen is AGPL3 or separately
licensed. Read `THIRD_PARTY_NOTICES.txt`; installing an extra does not remove
license obligations. 大文件、编译产物、临时文件、图形和测试缓存全部置于源码外。

The example runner retains one CPU, low scheduling priority, an8GiB host RAM
reserve and a6GiB process-group RSS cap for reference meshing. GPU is not needed
for serial TetGen; GPU utilization is not claimed as algorithmic acceleration.
The large-output flag changes only the disk file cap. 单核低优先级运行，保留系统内存；
大文件模式不放宽CPU或物理内存限制。

Primary references / 原始文档：[COMSOL Mesh format](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.35.html),
[COMSOL import selections](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_mesh.24.32.html),
[VTK Surface Nets](https://vtk.org/doc/nightly/html/classvtkSurfaceNets3D.html),
[Triangle core terms](https://www.cs.cmu.edu/~quake/triangle.html).
VTU specification / VTU格式：
[VTK XML file format](https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html).

## 7. Optional compiled MMG adaptation / 可选编译MMG重网格

The unmodified MMG5.8.0 scientific source is integrated under
`src/morqex/vendor/mmg`, pinned to commit
`4d8232c8aebfed877935d75d4d4a67e850962422`. It is **LGPL3+**, dynamically
linked through our own C++17 bridge. The default build disables VTK, Scotch
and Elas dependencies. Both build paths compile with one job outside source.
源码已集成并保留逐文件哈希与许可证；不是仅下载一个外部可执行程序。可替换共享库
随可选wheel安装，CMake构建和大数据均在外部工作区。

```bash
export MORQEX_WORKSPACE=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
# Option1: compile with a non-editable pip installation / pip显式选择编译
MORQEX_BUILD_MMG=1 python -m pip install .
# Option2: external cached library for source-checkout use / 源码调用外部缓存库
python -m morqex.modeling build-mmg --workspace "$MORQEX_WORKSPACE"
```

```python
from morqex.modeling import RemeshOptions, remesh, build_mmg
# Explicit compilation is required; calling remesh does not silently compile.
# 显式编译；remesh不会在执行时偷偷启动编译器。
build_mmg(work)
adapted = remesh(reference, workspace=work, options=RemeshOptions(
    maximum_edge_m=0.02,       # metres / 米，适合此处1m参考几何的示例请求
    hausdorff_m=0.0005,        # MMG geometric request, not a certified bound
    maximum_phase_volume_error=0.02,
    memory_mib=2500, timeout_seconds=240,
    maximum_nodes=3000000, maximum_cells=5000000))
adapted.save_hdf5(work/'adapted.h5')
adapted.adaptive_viewer().save_html(work/'adapted.html')
```

`minimum_edge_m`, `maximum_edge_m`, `uniform_edge_m`, `hausdorff_m`,
`gradation` and `feature_angle_degrees` control adaptation. They are MMG
requests, not guarantees for every individual edge. `optimize_only=True`
requests existing-size optimization and cannot be combined with uniform size.
`preserve_surface=True` freezes surface modification; this can strongly limit
coarsening of finely triangulated material interfaces. `open_boundaries=True`
permits explicitly marked same-material internal surfaces. 模型很小（微米）时
仍以米输入；内部归一化后传入求解器，返回时恢复原始SI坐标。

MMG is run in an isolated worker by default. Only its full SUCCESS return is
accepted; LOWFAILURE, missing tags/materials, nonpositive cells, changed
tag-to-material incidence, exceeded phase-volume tolerance and changed checked
surface invariants reject the candidate. The invariants are surface connected
component counts, Euler characteristics and open/nonmanifold status. They are
useful checks, **not a proof of geometric isotopy or electrical connectivity**.
默认原生计算隔离执行，超时、内存不足和不完整结果不能充当已验收网格；相体积以输入
网格为参考，不是对未知真实CT曲面的误差保证。相同不变量也不能证明全部拓扑等价。

TOML applies to an existing Tet4 mesh, separately from voxel generation:
TOML重网格作用于已有体网格，不能将体素生成和重网格混为同一过程。

```toml
[remeshing]
uniform_edge_m = 2.5e-6
hausdorff_m = 1e-7
gradation = 1.3
maximum_phase_volume_error = 0.02
maximum_cells = 5000000
maximum_nodes = 3000000
memory_mib = 2500
timeout_seconds = 240
check_surface_topology = true
isolated = true
```

```bash
python -m morqex.modeling remesh /external/input.h5 \
  --config /external/remesh.toml --output /external/output.h5 \
  --preview /external/output.html --workspace "$MORQEX_WORKSPACE"
```

Python equivalent / Python等价入口:
`remesh_from_toml(existing_mesh, config, workspace=work)`.
For a large saved mesh, pass a `MeshArtifact` and an explicit external output;
only the isolated native worker loads the full arrays. Calling `remesh` on an
already loaded `TetGenMesh` additionally retains the parent's input arrays.
大文件应使用下列文件接口，避免父进程和原生计算进程同时持有完整输入数组：

```python
from morqex.modeling import MeshArtifact, remesh_artifact, remesh_from_toml
source = MeshArtifact.load(work/'input.h5')  # reads metadata only / 只读元数据
adapted = remesh_artifact(source,work/'coarse.h5',workspace=work,
    options=RemeshOptions(uniform_edge_m=2.5e-6,hausdorff_m=1e-7,
        maximum_input_cells=30000000,maximum_cells=5000000,
        maximum_nodes=6000000,memory_mib=2500,timeout_seconds=240))
# Equivalent TOML entry / 对应TOML入口：
# adapted = remesh_from_toml(source,config,output=work/'coarse.h5',workspace=work)
```

`maximum_input_cells` defaults to `maximum_cells` when omitted; the output limit
does not forbid a deliberately larger input when this separate budget is set.
Both limits and the node limit are hard acceptance gates. `memory_mib` is binary
MiB, converted to MMG's decimal MB; the external process RSS guard still includes
Python arrays and validation buffers, so it must exceed the native-only budget.
文件接口强制隔离；输入/输出预算、原生内存请求和外部整个进程组内存上限含义不同。
超时或资源不足时不发布候选文件，原网格保持不变。

The regular two-material size-control benchmark is validated independently of
the large galleries. Full tens-of-millions-cell MMG coarsening has not completed
within the bounded single-core time budget; do not infer its performance from
the small benchmark. 大规模复杂界面的粗化受拓扑细节和资源预算限制，不能将小例子的
疏密控制能力冒充全部大模型已完成快速粗化。
The native integer references are compacted only inside the worker; original
region IDs including zero are restored. No source array or original mesh is
overwritten. Tiny physical bridges and complicated junctions can prevent
aggressive coarsening. A request rejected by any gate must be refined or
reconsidered; disabling a check is not a passing validation.
原始标签（含0）不会被永久重编号；粗化不能以删除物理细节、放宽已声明门槛作为通过。

Original algorithm/API documentation / 原算法与接口：
[MMG5.8.0 source](https://github.com/MmgTools/mmg/tree/v5.8.0),
[MMG3D API](https://mmgtools.github.io/libmmg3d_8h.html).

## 8. Reproduce installation checks / 重现安装包检查

The following commands freeze source into the external workspace, build a
wheel, install it into a fresh external target and call the installed API in
a separate interpreter. The native run checks actual libraries from that
installation, not cached libraries from the source checkout. The source
distribution also includes configurations, tutorials and verification scripts.
以下命令分别验收纯Python安装与显式编译安装；日志记录源码哈希，所有构建、安装和
输出目录均在工作区，不会在源码创建 `build`、大型数据或二进制。

```bash
PROJECT=/home/zephyrain/lw/MORQEX
PY="$MORQEX_WORKSPACE/runtime/bin/python"
python "$PROJECT/scripts/run_local.py" --workspace "$MORQEX_WORKSPACE" \
  --label package_pure --reference-memory -- \
  "$PY" "$PROJECT/scripts/qualify_geometry_package.py" --sdist
python "$PROJECT/scripts/run_local.py" --workspace "$MORQEX_WORKSPACE" \
  --label package_native --reference-memory -- \
  "$PY" "$PROJECT/scripts/qualify_geometry_package.py" --backends tetgen acvd mmg
```

The guarded worker runs in its own output directory; these commands therefore
use absolute interpreter/script paths. 原生编译需要C/C++编译器、CMake及Cython/NumPy头文件；安装包检查
不代表COMSOL或Abaqus实际求解器验证，也不代替网格收敛验证。

## 9. Inspect measured results / 查看实测结果

The generated gallery is external to the source tree:
`/home/zephyrain/lw/morph_workspace/microelectrode_gallery/nmc_complex_gallery/index.html`.
Its structure selector loads one pair of viewers at a time. The left viewer
shows source voxels; the right viewer shows an actual Tet4 section. Use the
separate surface link to rotate the mesh interfaces. Phase zero is selectable;
slice controls use original array indices. 网页只加载当前结构，避免同时打开全部大模型。
体素三维显示可抽样，二维切片保留原始像素；网格切面来自完整体网格，显示限制在页面中注明。

`acceptance.html`, `acceptance.json` and `mesh_catalogue.csv` separate geometry,
shape, format readback, browser and regression gates. A process exit code of
zero does not mean that every shape request passed. Reports retain failures.
验收报告保留未达形状请求的模型，不能将格式回读或正体积单元等同于物理计算收敛。

After generating the gallery, reproduce the interactive browser check with
the optional `browser-test` dependency and a locally installed Chromium:
生成网页后，可使用可选依赖和本机Chromium验收切换、切片及相选择：

```bash
# python -m pip install 'morqex[browser-test]'
PROJECT=/home/zephyrain/lw/MORQEX
PY="$MORQEX_WORKSPACE/runtime/bin/python"
python "$PROJECT/scripts/run_local.py" --workspace "$MORQEX_WORKSPACE" \
  --label geometry_dashboard_browser --render -- \
  "$PY" "$PROJECT/scripts/check_geometry_dashboard.py" --browser /absolute/path/to/chromium
python "$PROJECT/scripts/run_local.py" --workspace "$MORQEX_WORKSPACE" \
  --label geometry_qualification_report -- \
  "$PY" "$PROJECT/scripts/report_complex_geometry.py"
```

The browser check uses one browser and two attached frames, disables GPU, and
saves screenshots and JSON externally. It does not require a network service.
浏览器验收不使用GPU，所有截图、日志和大文件仍保存在外部工作区。

## 10. Reproduce the native output benchmark / 重现原生输出开销对照

`scripts/benchmark_tetgen_face_output.py` reconstructs one physical PLC from
an existing Tet4 artifact, then changes only whether TetGen exports ordinary
interior faces. Every material interface is retained. It saves input and output
hashes, actual native flags and separate numerical/shape reports. The default
input budget is five million cells; all runs remain subject to the resource
guard. 这个对照不改物理几何、求解容限或硬件，不删除坏单元；比较前先核对同一输入及
节点、单元、材料和规范化带标签界面的哈希。几何通过与形状通过仍是不同门槛。

```bash
PROJECT=/home/zephyrain/lw/MORQEX
PY="$MORQEX_WORKSPACE/runtime/bin/python"
INPUT="$MORQEX_WORKSPACE/nmc_complex_gallery/native/granular/partitioned/part_00.h5"
BENCH="$MORQEX_WORKSPACE/benchmarks/face_output_example"
python "$PROJECT/scripts/run_local.py" --workspace "$MORQEX_WORKSPACE" \
  --label faces_baseline --reference-memory -- \
  "$PY" "$PROJECT/scripts/benchmark_tetgen_face_output.py" "$INPUT" \
  --mode all_faces --output "$BENCH/all_faces.json"
python "$PROJECT/scripts/run_local.py" --workspace "$MORQEX_WORKSPACE" \
  --label faces_constrained --reference-memory -- \
  "$PY" "$PROJECT/scripts/benchmark_tetgen_face_output.py" "$INPUT" \
  --mode constrained --output "$BENCH/constrained.json"
```

Use a new output directory for another experiment; results are not overwritten.
In Python, load both JSON files, check `input_sha256` and `hashes` equality,
then compare `raw_returned_faces` and `native_and_validation_seconds`. Peak
process-group RSS is recorded in each launcher's manifest, rather than inferred
from array sizes. A single timing pair is a measured observation, not a
statistical speedup estimate. 再次实验选择新目录；实际峰值内存读取运行清单，不以数组
大小估算冒充。基准可换成其他已有体网格，但不能将不同输入的耗时直接相除。

### Full-model overview plus unsampled FEM detail / 全模型概览与不抽样的有限元细节

```bash
python scripts/run_aligned_geometry.py \
  --workspace /home/zephyrain/lw/morph_workspace/microelectrode_gallery \
  --cases nmc granular rods platelets agglomerates bicontinuous cell_stack \
  --stages detail --roi-voxels 48
```

The gallery links each `roi_mesh.html`. It contains the complete original
triangles selected inside the central48-voxel-wide physical display box,
without a new SVE or cropped numerical solve. `roi_display.json` records its
physical XYZ bounds, full source-cell count, selected/embedded face counts
and unchanged source-file checks. The whole-model source and exact volume
sections remain available alongside it. See the
[complete ROI API and code](04_notebook_visualization.md).
每个局部详情都绑定完整大模型，不能当作另一个小SVE计算结果；完整源数据、全域体网格
切面和全局质量限制均保留。局部未生成新网格，没有补面，没有隐藏计算规模。
