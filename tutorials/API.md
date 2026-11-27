# MORQEX API reference

## Complex geometry and units / 复杂几何与单位

See [the complete bilingual workflow](07_complex_ct_layers_and_exchange.md).

| API / 接口 | Contract / 契约 |
|---|---|
| `tiff_voxels(path, spacing=..., axes_to_zyx=None, region_names=None)` | Full original labels + SHA256, explicit ZYX spacing in m / 保留全量灰度与来源 |
| `import_comsol(path,length_unit=None,region_names=None)` | Read Mesh4 + Selection0; use a declared file unit or require the caller's explicit unit; reject conflicts / 读文件单位或显式指定，拒绝单位冲突 |
| `dense_electrode(morphology,size=256,spacing=5e-7,seed=...,active_fraction=.55,cbd_fraction=.11,...)` | Exact phase counts, correlated patchy CBD / 原生分相体素 |
| `coated_electrode(morphology,size=256,spacing=5e-7,seed=...,active_fraction=.5,cbd_thickness_voxels=3,pore_buffer_voxels=7,...)` | Resolved CBD coating; measured CBD fraction / 可解析涂层路线 |
| `uniform_layer(shape,spacing,name,region=1)` | Named effective layer / 命名均匀层 |
| `porous_layer(shape,spacing,porosity=.45,correlation_voxels=3,seed=0,name=...)` | Correlated resolved pores and solid / 显式孔隙层 |
| `stack_layers(layers,axis=0,origin=...)` | Distinct layer×phase IDs and selections; matching transverse grid / 复合分层 |
| `surface_nets_plc(model,iterations=8,constraint=.2,conserve_volumes=False,...)` | Optional VTK shared front end; smoothing geometry assumptions explicit / 共享界面 |
| `cut_cap(points,faces,markers,pairs,plane,axis=2,...)` | One Triangle PSLG cap, labels from oriented physical interfaces / 统一切面 |
| `slab_plc(source,low,high,lower_cap,upper_cap,axis=2,length_scale=...,origin=...)` | Clip original triangles + shared end caps; no geometry coarsening / 分块PLC |
| `assemble_slabs(paths,path,length_scale=...,preserve_material_jumps=False)` | Stream one full mesh; assert common nodes and material faces / 严格拼接 |
| `export_comsol(...,length_unit='m',selections=None,include_region_selections=True,preserve_domain_ids=False)` | Actual Mesh4 + Selection0; SI-to-export coordinate conversion / 真实命名选择与单位换算 |
| `export_bundle(mesh,directory,region_names,selections,length_unit='m',formats=...)` | COMSOL/VTU/Gmsh/Medit/Abaqus, region and unit manifest + readback / 多软件交换 |
| `export_from_toml(mesh,model,path,directory=None)` | Apply explicit `[exports]`; output remains outside source / TOML导出 |
| `VoxelModel.viewer(...,slice_origin='upper')` | Physical voxel aspect; `lower` matches Cartesian sections, selectable region IDs and SI centre-plane coordinates / 物理比例、笛卡尔方向与体素中心坐标 |
| `AdaptiveMeshView(...,maximum_embedded_faces=None,node_cache_bytes=...,section_fractions=(.5,.5,.5),surface_bounds_m=None)` | Optional actual-face display sample; three requested XYZ planes; full HDF5 untouched / 显示抽样、指定物理切面，不改网格 |
| `MeshResolution(precision='balanced',target_cells=None,maximum_volume_m3=None,nominal_edge_m=None,vertex_fraction=None,maximum_phase_volume_error=None)` | Independent surface/volume fidelity, SI sizing; coarse/balanced/fine/very_fine / 独立曲面与内部尺寸 |
| `controlled_tetgen(model,resolution,regions,surface_method,adaptive_options,tetgen_options,...)` | Same voxel source SHA, actual mesh counts and measured checks recorded / 同一体素多精度网格 |
| `partitioned_tetgen(model,directory,surface_method='surface_nets',slab_depth=32,axis=0,...)` | Generic resumable whole-domain pipeline, common actual caps, explicit budgets / 通用全域分块、断点恢复和严格共形 |
| `RemeshOptions(...)`, `remesh(existing_mesh,options,workspace)` | Source-integrated MMG5.8.0; measured size/geometry/topology gates / 已有网格疏密优化 |
| `remesh_artifact(artifact,output,options,workspace)` | Large HDF5 input loaded only inside an isolated worker; separate input/output cell budgets / 大网格文件隔离重网格，避免父子进程重复载入 |
| `build_mmg(workspace)` | Explicit one-job external CMake build; optional pip flag `MORQEX_BUILD_MMG=1` / 外部显式编译 |
| `build_native_from_toml(path,workspace=None)` | Explicit `[native_build]` list of TetGen/ACVD/MMG; serial external builds / TOML统一编译 |
| `remesh_from_toml(mesh,path,workspace=None)` | Apply explicit `[remeshing]` to existing Tet4 / TOML体网格重构 |

`partitioned_tetgen` supports `surface_method='voxel_aligned'` (exact integer
voxel partitions), `'voxel_plc'` (clipped original stair-step PLC) and
`'surface_nets'` (explicitly approximated smooth shared surface). The aligned
route preserves physical material jumps on cuts and verifies original labels
at every cell centroid. Shape requests are reported separately; see the
[complete bilingual contract](07_complex_ct_layers_and_exchange.md).
三条路线分别保留；精确体素分区不代表自动获得优质单元，不得忽略实测形状失败。

`from_toml` accepts `[geometry]`, optional `[[layers]]`, `[meshing]` and
`[exports]`. It generates geometry only; call `mesh_from_toml` and
`export_from_toml` explicitly for the subsequent stages. 数组为ZYX，网格为XYZ米。
COMSOL导入单位必须与导出一致；有效参数和物理模型不是网格导出的一部分。
VTU uses bounded appended-raw binary IO; Medit/Abaqus use bounded ASCII IO.
VTU采用分块二进制，Medit与Abaqus采用分块文本核对；完整数据不受HTML抽样影响。

The unified geometry API is `morqex.modeling`. Begin with:

- [Geometry and voxel modelling](01_geometry_and_voxels.md)
- [Finite-element meshes](02_finite_element_meshes.md)
- [Integrated geometry frameworks](03_integrated_geometry_backends.md)

中文：[几何与体素](01_geometry_and_voxels.md#zh) ·
[有限元网格](02_finite_element_meshes.md#zh) ·
[集成框架与扩展](03_integrated_geometry_backends.md#zh)。

```python
from morqex.modeling import (
    VoxelModel, Shape, from_regions, from_shapes, electrode,
    tetrahedralize, stream_tetrahedralize, framework, capabilities,
    register_generator, generate, from_toml, register_mesher, mesh,
)
```

Native meshing and COMSOL / 原生网格与COMSOL：

```python
from morqex.modeling import (
    SurfacePLC, TetGenOptions, TetGenMesh, voxel_plc, marching_cubes_plc,
    tetgen_mesh, build_tetgen, tetra_quality, export_comsol, read_comsol,
    mesh_from_toml,
)
```

|API / 接口|Contract / 契约|
|---|---|
|`build_tetgen(workspace)`|Explicit single-job CMake build outside source; optional pip build uses `MORQEX_BUILD_TETGEN=1` / 单核外部编译|
|`voxel_plc(model,regions,maximum_facets)`|Exact common multiphase interfaces and interior/hole seeds / 精确共享多相界面及种子|
|`marching_cubes_plc(model,region,step_size,maximum_facets)`|Approximate closed single-phase contour; correct voxel-centre coordinates / 单相近似闭合轮廓|
|`SurfacePLC(points,facets,seeds,markers,holes,region_volumes,metadata)`|XYZ SI points/seeds, original integer region IDs / XYZ米及原始分区标签|
|`tetgen_mesh(geometry,regions,surface_method,step_size,maximum_facets,options,workspace)`|`VoxelModel` or `SurfacePLC` to `TetGenMesh`; no face deletion / 贴体体网格，不删面|
|`VoxelModel.tetgen(**kwargs)`|Same typed entry / 同一原生入口|
|`TetGenOptions.for_adaptive(**overrides)`|Strict facet-merge-angle policy, native-process isolation, full volume checks / 严格共面合并角、原生隔离及体积检查|
|`TetGenOptions(...)`|Sizing/quality/Steiner/output caps; see complete table in tutorial / 尺寸、质量与资源门槛|
|`TetGenMesh.report`|Actual quality, time, region-volume errors and geometry convention / 实测质量及几何范围|
|`MeshArtifact.load(path)`|Reopen complete voxel/TetGen HDF5 without loading all cells / 不加载全部单元重新打开|
|`TetGenMesh.save_hdf5(path)`|Complete Tet4+marked facets in HDF5/XDMF / 全量网格和标记面|
|`TetGenMesh.viewer/show`|Actual Tet4 display subset, offline notebook/HTML / 真实单元可视化|
|`tetra_quality(points,cells)`|Measured radius ratio, dihedral and positive volumes / 实测形状指标|
|`export_comsol(mesh,path,region_names,chunk_size,include_boundaries)`|Tet4/Hex8 to `.mphtxt` + explicit `.regions.json`; MeshArtifact accepted / COMSOL原生文本|
|`FEMesh.save_comsol`, `MeshArtifact.export_comsol`|Same exporter / 同一导出器|
|`read_comsol(path)`|Strict numerical read-back of emitted3D linear subset / 导出子集的严格读回|
|`mesh_from_toml(model,path)`|Apply `[meshing]` and `[meshing.options]` recipe / TOML网格配方|

Detailed bilingual instructions, pip options and validation limits are in
[TetGen and COMSOL](05_tetgen_and_comsol.md). A successful format round-trip
is not a claim that a local COMSOL application has imported/solved the file.
详细双语说明见教程；格式读回不等于已在本机 COMSOL 中导入求解。

GEMsim residual-weighted adaptive geometry / GEMsim几何自适应：

```python
from morqex.modeling import (
    AdaptiveMeshOptions,adaptive_surface,adaptive_mesh,multiresolution_weights,
    build_acvd,resolved_electrode,
)
```

|API / 接口|Contract / 契约|
|---|---|
|`build_acvd(workspace)`|Explicit external single-job compilation of the original weighted kernel / 原始加权内核外部编译|
|`resolved_electrode(preset,size,spacing,seed,cbd_thickness_voxels,minimum_feature_voxels,cbd_bridges)`|New ten-family resolved-feature synthetic geometry;96..256³ / 十类新配方|
|`AdaptiveMeshOptions.precision(name,**overrides)`|coarse/balanced/fine geometry gates and target densities / 三种几何精度|
|`adaptive_surface(model,regions,options,workspace)`|Full-resolution interfaces→weighted ACVD→validated SurfacePLC / 曲面重构|
|`adaptive_mesh(model,regions,options,tetgen_options,workspace)`|Full adaptive pipeline→TetGenMesh / 完整网格流程|
|`VoxelModel.adaptive_surface`, `.adaptive_mesh`|Same public implementations / 同一公开实现|
|`mesh(model,method='gemsim_adaptive',...)`|Shared registry entry / 统一注册入口|
|`multiresolution_weights(model,points,options)`|Numerical surface density at XYZ SI points / 数值曲面密度|
|`SurfacePLC.viewer/show`|Actual whole surface viewer / 真实完整曲面|
|`TetGenMesh.adaptive_viewer(sections=True,height,title)`|Complete marked surface plus exact Tet4-plane sections / 完整曲面及真实体切面|
|`morqex.visualization.adaptive_mesh.section_polygons(mesh,axis,fraction)`|Actual volume-cell intersections;XYZ axis0..2 / 真实体单元截面|

All adaptive option fields, typed TOML, compilation details and the ten256³
protocol are in the complete bilingual [adaptive tutorial](06_gemsim_adaptive_meshing.md).
An adaptive tetrahedron has no unique parent voxel; centroid lookup is not a
conservative field-transfer operator.
完整参数及复现流程见教程；自适应单元跨多个体素，不能把质心查询当成守恒场映射。

Public geometry fields use ZYX arrays and SI spacing/origin. FEM vertices use
XYZ in SI metres. `VoxelModel.mesh_to` supports full256³ region arrays through
bounded slab processing. See the tutorials for the input/output contracts,
explicit region selection, topology policy and optional backend requirements.

|Public entry / 公开入口|Purpose / 用途|
|---|---|
|`VoxelModel(regions,spacing,origin,region_names,metadata)`|Integer-region geometry contract / 整数分区几何|
|`from_regions(...)`|User-provided ZYX region arrays / 用户数组|
|`Shape(kind,region,centre,axes,rotation,irregularity)`|SI primitive definition / 物理尺寸原语|
|`from_shapes(shapes,shape,spacing,origin,background,region_names)`|Direct voxel-centre rasterization / 直接体素化|
|`electrode(preset,size,spacing,seed,cbd_fraction)`|Ten irregular morphology families; returns model and source IDs / 十种预设|
|`PRESETS`|Preset identifiers and names / 预设列表|
|`VoxelModel.save`, `VoxelModel.load`|HDF5 geometry I/O / 几何存储|
|`VoxelModel.volume()`|Existing GEM Volume adapter / GEM数据适配|
|`tetrahedralize`, `VoxelModel.tetrahedralize`|Small in-memory Tet4 / 小规模内存网格|
|`stream_tetrahedralize`, `VoxelModel.mesh_to`|Full shared-node HDF5/XDMF mesh / 大规模共享节点网格|
|`MeshArtifact`|HDF5/XDMF paths, node/element counts and selected regions / 文件与数量契约|
|`framework(name)`|Complete integrated public module namespace / 集成模块公开API入口|
|`capabilities()`|Portal names and optional native prerequisites / 能力与依赖状态|
|`register_generator`, `generate`|Custom VoxelModel callbacks / 自定义几何回调|
|`register_mesher`, `mesh`|Custom FEMesh/MeshArtifact callbacks / 自定义网格回调|
|`from_toml(path)`|Presets, shape lists or relative `.npy` region files / 配置入口|
|`VoxelModel.show`, `VoxelModel.viewer`|Live notebook2D/3D display handle / 实时预览|
|`VoxelView.update`, `VoxelView.save_html`|Update the same output or save standalone HTML / 同位置刷新和导出|
|`MeshArtifact.show`, `MeshArtifact.viewer`, `MeshView`|Exact bounded subset of stored or in-memory Tet4 / 真实四面体子集显示|

`from_shapes(...,progress=view.update,progress_every=5)` shows actual partial
geometry during construction. See the bilingual
[notebook visualization tutorial](04_notebook_visualization.md) and
[executable notebook](notebooks/geometry_workflow.ipynb).

The remaining sections document existing MORQEX research modules. Paths to
previous run outputs are historical examples; the workspace has been cleared
at the user's request and those files must be regenerated before use.

---

# MORQEX API 与操作说明

源码、配置和测试在 MORQEX；数据、环境、编译、算子、断点、结果、HTML 全部在相邻的 `morph_workspace`。新全体积接口与旧版 SVE 接口分别验证，不共享通过标签。本文末尾为从源码 AST 生成的公开定义索引；研究内部入口仍需遵守各模块 docstring。

## 动态网络与逐节点三维疲劳循环（数值候选）

统一入口 `network-cycle`：每个 DFN 厚度单元保留独立的三维有限变形位移、不可逆 AT2 相场、Carrara 型疲劳历史及浓度状态；微结构反应面积、扩散图、电子导率和离子输运效率作为空间端口反馈。相邻步保留原材料库存；电子失联不删除浓度状态。当前周期驱动器采用弥散损伤电子/固相导率和固定孔隙/CBD 离子几何，尚未自动生成经验证的开放裂纹、浸润、新生反应面积或宏观力平衡。**协议完成、数值检查、物理模型资格和论文验收是不同状态。**

```bash
PY=/home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python
ROOT=/home/zephyrain/lw/MORQEX
# 先将配置复制到外部工作区并设置唯一 output；下列入口每次最多100接受步。
$PY "$ROOT/scripts/run_local.py" --label network_cycle -- $PY -m morqex.top_damage network-cycle /home/zephyrain/lw/morph_workspace/experiments/my_network.toml --steps 100
# 重复相同命令，自动验证源代码/配置/几何身份后从最后完整HDF5恢复。
$PY "$ROOT/scripts/run_local.py" --label network_report -- $PY -m morqex.top_damage network-report /home/zephyrain/lw/morph_workspace/results/top_damage/network_cycles/MY_OUTPUT --output MY_OUTPUT.html --export
# 各倍率/结构/材料、浓度空间、长循环研究使用同一求解入口：
$PY "$ROOT/scripts/run_local.py" --label network_study -- $PY "$ROOT/benchmarks/study_fatigue_network.py" --cases full8_time24 --steps 100
$PY "$ROOT/scripts/run_local.py" --label network_evidence -- $PY "$ROOT/benchmarks/report_network_study.py"
```

TOML 模板：`configs/top_damage_network_cycle.toml`。`[cycle]` 控制输出名称、材料 `host`、浓度区间 `stoichiometry_window`、放电/充电倍率 `C_rate/charge_C_rate`、实际完整循环数 `cycles`、厚度单元 `nodes>=3`、每半循环试步数、等温温度、弹性/韧度/正则长度、晶体化学膨胀张量对角分量、AM/CBD 导率、CBD 孔隙率/输运效率、梯度能、图更新门槛及开关。`dfn_rtol/dfn_atol` 默认 `1e-10/1e-11`，独立检查全部输出时刻的电荷平衡，不能只检查步端点。`mechanics_enabled=false` 是无力学对照；`topology_feedback=false` 冻结传输图反馈但仍保留力学化学势，并非“去掉全部力学”。

`concentration_regions=0` 使用初始晶粒区域，正整数使用连通聚合，`-1` 为每个活性体素一个状态。为了保持连通性，实际区域数可稍大于请求值。`local_mobility=true` 采用对称边中点迁移率。区域状态替代原正极径向状态，没有重复锂质量。当前参数 D 显式用作正迁移率参考，不应直接宣称等同于原生 DFN 的化学扩散系数；非凸 OCP 需要正梯度能及独立分辨率验证。

`[microstructure]` 可用原 `PolycrystalSpec` 生成规则三维颗粒，也可使用以下完整外部身份数据：

```toml
[microstructure]
source_hdf5 = "data/my_polycrystal.h5" # 相对外部工作区；morqex-polycrystal-1
sha256 = "这里填写文件的64位SHA256"
solver_seed = 104
```

导入必须同时具备相、颗粒、连通晶粒和 SO(3) 取向，且相标签为 0=孔隙、1=活性材料、2=CBD。不能把只有 CT 分相的数据自动冒充已知晶粒和材料参数。生成器采用零原点；network-cycle 的非零原点导入已通过机械网格及 VTI/VTU 往返核查。其他后端仍按各自的坐标限制处理。

`DualNetworkClosure.evaluate(electronic_S_m, ionic_efficiency, ...)` 支持独立的电子面传输、离子面传输、固相扩散面因子和原始界面有效比例。真实零导边参与连通域更新；部分区域内部割裂时报告必须富集的区域。`NetworkEpochDFN.advance_network(...)` 接收 `(nodes,regions)` 力学化学势/界面比例、`(nodes,edges)` 扩散因子和节点导率；断开区域保留原浓度和质量。新生裂纹面积不能通过把原面积比例设为大于1实现，须扩展有几何来源的表面端口。

`ConductingNetwork` 的 Kron/Schur 端口消元是经典方法，支持浮动分量与恢复。`NetworkCache` 在固定正边支撑下给出相对于最后重建图的二次型相对界；边支撑、拓扑、锚点变化立即重建。该界只约束被动输运，不是电压、容量或长期疲劳的误差界。瞬态浓度状态没有被静态凝聚。

每个输出目录保存 `initial.h5/mesh.h5/step_*.h5/manifest.json/electrochemical_curve.csv/validation.json`。状态包含每个厚度节点的全三维位移、相场、疲劳、浓度、应力与下一步端口；接受后原子替换写盘，失败试步不覆盖已接受状态。导出 `paraview/node_NNN.pvd` 查看体素字段，`node_NNN_mechanics.pvd` 查看精确 Tet4 节点相场和位移。各节点 SVE 分开呈现，不能拼成声称真实的整电极空间。单位加载网络电势不是充放电工作电势。

HTML 保留每节点、25个实际时刻的交互三维/剖切视图，全历史保存在 HDF5。总览为外部 `reports/network_study/index.html`。CPU 单核、RSS 3 GiB，单次 CPU 上限600秒；大区域数建议一次50–100步。此前预算停止会留下完整断点，可以续算；数值门槛失败必须先查明原因，不得改标签为通过。

## 当前算例：6 节点 / 32³ CBD 正极 / 平面 Li

### TOP-DAMAGE 较密颗粒、完整电压窗口与场导出

该路线复用原 MORQEX 和已编译 MFEM 后端。先安装 `top-damage` 与 `dfn` 依赖；所有数值命令均由资源运行器启动。

```bash
PY=/home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python
$PY scripts/run_local.py --label dense_geometry -- $PY -m morqex.top_damage geometry /home/zephyrain/lw/MORQEX/configs/top_damage_dense_particles.toml
# 单次最多两个接受步；同配置/工况/源码身份自动从最后完整断点续算。
$PY scripts/run_local.py --label dense_cycle -- $PY -m morqex.top_damage dense-cycle /home/zephyrain/lw/MORQEX/configs/top_damage_voltage_cycle.toml --environment gentle --steps 2 --output gentle_voltage_v3
# 三种工况依次推进，绝不同时运行；任一失败立即停止。
$PY scripts/run_top_damage_dense_study.py --config /home/zephyrain/lw/MORQEX/configs/top_damage_voltage_cycle.toml --suffix voltage_v3 --steps 2
# 核对全部历史、导出完整场并生成 HTML。
$PY scripts/run_local.py --label dense_report -- $PY -m morqex.top_damage dense-report --export
```

`--environment gentle/fast/released` 分别是 0.5C 全外边界零位移、2C 全外边界零位移、0.5C 零牵引并去除刚体模态。这里的约束不是原六节点电芯固定厚度条件；两条路线不能混称。`[cycle] protocol="voltage-window"` 执行预充至上截止、放电至下截止、回充至上截止；`cycles` 不包括预充，没有 CV 阶段。`increments_per_half_cycle` 决定试步 `dt=3600/C/n`，实际截止由 DFN 事件定位。`requested_steps` 是该模式的安全步数上限，不是已经接受步数。`partial-depth` 是固定部分容量 CC/CC，不能用于声称全窗口容量衰减。

`EpochDFN` 位于 `morqex.top_damage.epoch_dfn`。`advance(dt,current_A,DFNFeedback(...))` 保留微分状态并重初始化代数状态；`restore_state(time_s,state,inputs)` 在同模型/离散布局下恢复。端口量为有效电子电导 S/m、电解液几何输运效率、反应比例及平均力学化学势 J/mol。Li 金属端使用精确 FVM 物种通量 `i/F`，保留内部面通量；锂库存采用 `M(t)-M(0)-Q(t)3600/F`。不能把这个端口直接用作几何变化后的锂状态重映射。

`effective_conductivity(field,spacing_zyx_m,axis=2)` 位于 `voxel_conductivity`，求解全体积标量 FVM、真实零导相和浮动分量；返回等效导率、单位加载电势、集流体可达掩码及残差/能量检查。返回电势是均匀化单位载荷场，不是运行电池的局部电势。当前还不是完整非对角输运张量。

`SplitElasticity(...,linear_solver="amg-cg")` 复用完整刚度 AMG 预条件器，每次重新装配实际损伤切线并验证真实残差。`AT2Damage(...,linear_solver="jacobi-cg")` 保留不可逆活跃集/KKT；默认直接法仍保留。加速路线与相同离散直接解独立对照；没有给出未经公平测量的整电池加速倍数。

结果在 `results/top_damage/dense_cycles/<output>`：每步 HDF5 含完整节点位移/d、单元疲劳历史、全部体素字段、DFN 状态和 I/V/Q/t；`mesh.h5` 与 `initial_geometry.h5` 共享几何。`dense-report --cases NAME ... --output NAME.html --export` 生成完整 FP64 VTI/PVD、引用 HDF5 的 XDMF 时间序列及历史验证 JSON。ParaView 打开 `voxel_history.pvd` 看体素场，打开 `mechanics_history.xdmf` 看 FEM 节点位移/相场；Warp By Vector 的物理位移放大系数为 1。网格连接没有被伪造修改，高 d 阈值只是诊断，不是已验证裂缝孔隙。

`verify_dense_history(directory)` 检查实际电流积分、反馈端口连续衔接、损伤/疲劳不可逆、固定材料体积、残差和状态连续性；另报告按相的总应变最大值/P99及超过 2% 筛查线的体积分数。2% 是保守工程筛查，不能称为严格误差界。当前三个完整窗口算例的小应变筛查均未通过，主要来自 CBD，故数值协议完成不等于物理参考合格。

HTML 为低内存对照首页与单工况页，默认一次加载一个工况。场显示副本为 FP32，数值计算/HDF5/VTI 保留 FP64。色标在同一工况的全部时刻固定；不同工况可能不同，跨工况定量比较请使用数值表。窗口完成后显示的容量差异包含倍率极化/截止影响，不能当作长期容量衰减。当前仅一轮完整窗口与此前两次部分深度循环，未验证千循环、真实裂缝侵液、颗粒失联、一般微结构扩散、热损伤或 NMC/LFP 多材料疲劳标定。

### TOP-DAMAGE 子框架：源码编译与三维参考

初始大颗粒几何从同一入口生成：

```bash
PY=/home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python
$PY scripts/run_local.py --label large_particles -- $PY -m morqex.top_damage geometry /home/zephyrain/lw/MORQEX/configs/top_damage_few_particles.toml
$PY scripts/run_local.py --label particle_seeds -- $PY /home/zephyrain/lw/MORQEX/scripts/study_top_damage_polycrystals.py
```

`[microstructure]` 参数为 `name/shape_zyx/spacing_m/particle_count/radius_voxels/radius_cv/minimum_gap_voxels/cbd_fraction/grains_per_particle/seed/grain_seed`。长度为米；`radius_voxels` 和间隙为体素数。默认块长 16 μm、名义颗粒直径 6.5 μm，4 个颗粒是稀疏形貌检查案例而非材料标定。`PolycrystalSpec.from_toml(...).generate()` 返回 `PolycrystalData`；`create_polycrystal(config, Workspace(...))` 保存初始结果。同名目录若已有不同配置或几何会拒绝覆盖。

`PolycrystalData` 保留完整 `structure/particle_ids/grain_ids/grain_rotations/grain_particles`，在每个颗粒内用固定种子的三维 Voronoi 分区；不连通碎片分开编号并报告实际晶粒数。`grain_rotations` 是逐晶粒、晶体到实验室 XYZ 坐标的 SO(3) 旋转；`rotated_tensor_table(A)` 返回 `R A Rᵀ`，不替用户补造材料参数。`grain_boundary_faces()` 返回同颗粒内不同晶粒的全体素平坦索引对、XYZ 法向轴、米平方参考面积；没有把晶界赋予任意厚度。

`save_hdf5/load_hdf5` 校验材料标签、身份、取向和派生晶界面；`initial.vti` 是 X-fastest 的完整 VTK CellData，含相/颗粒/晶粒三个字段。`index.html` 可拖动旋转、选择配色、控制三维 Z 剖切和独立 Z 切片、点击查编号。默认仅在三维视图隐藏 CBD，原始数据和二维切片仍完整保留 CBD。页面片段 `#cut=24&slice=23&mode=grain` 可固定晶粒剖切视图。只有真实接受的后续物理状态才允许输出为退化时刻；此命令只生成 `t=0` 初始态。

入口统一在 MORQEX 内；旧固定结构模块不变。已实现 MFEM 源码后端和三维力学—AT2 参考；当前没有合格的长期电池疲劳 `run`，不能把机械试件的循环测试当成充放电寿命预测。

```bash
PY=/home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python
# 一条命令自动续编；每个编译片段仍由run_local施加单核/内存/时间限制。
$PY scripts/compile_top_damage.py configs/top_damage_native.toml
# 每次最多55秒编译；返回pending时重复同一命令，保留已完成的目标文件。
$PY scripts/run_local.py --label top_damage_build -- $PY -m morqex.top_damage build /home/zephyrain/lw/MORQEX/configs/top_damage_native.toml
# 仅在build返回complete后调用；默认配置为三维缺口试件，不是DFN。
$PY scripts/run_local.py --label top_damage_reference -- $PY -m morqex.top_damage reference /home/zephyrain/lw/MORQEX/configs/top_damage_notched_reference.toml --steps 8
# 每次接受步保存checkpoint，重复续算到TOML指定的加载协议完成。
$PY scripts/run_local.py --label top_damage_reference_resume -- $PY -m morqex.top_damage reference /home/zephyrain/lw/MORQEX/configs/top_damage_notched_reference.toml --steps 8 --resume
```

`[native]` 指定 `provider="vendored"`（本地源码）或 `"pymfem-wheel"`（可选独立预编译库对照，需要 `mfem==4.10.0.1`）；当前只资格验证 `Release/jobs=1/float64/cpu`。两种库必须在不同进程交叉比较，避免同名共享库被加载器隐式复用。MFEM 库源码、示例程序和上游测试源文件位于 `src/morqex/vendor/mfem`；数据和原始完整发行归档在外部工作区。`provenance.json` 记录纳入及省略文件、逐文件哈希和归档哈希，保留 BSD-3 的 LICENSE/NOTICE。

`[reference]` 提供 ZYX 网格、米制间距、E/nu、Gc、ell、残余刚度、缺口长度、yy应变幅值、循环数、每循环增量数和迭代门槛。全部外边界为仿射 KUBC；横向应变为零，不是侧面自由拉伸试验。默认参数是数值试件；没有经过高镍正极材料标定。`ell` 是相场长度，不是物理裂缝开度。

```python
# 数值API须从 scripts/run_local.py 启动的研究脚本内调用。
from morqex.top_damage import NativeConfig, build_backend, MFEMScalarBackend
from morqex.top_damage import AT2Damage, FatigueHistory, SplitElasticity, staggered_step
from morqex import Workspace
ws = Workspace('/home/zephyrain/lw/morph_workspace')
cfg = NativeConfig.from_toml('/home/zephyrain/lw/MORQEX/configs/top_damage_native.toml')
build_result = build_backend(cfg, ws)  # pending/complete；不会隐式升级计算资源
backend = MFEMScalarBackend(cfg, ws)   # 未编译、陈旧或修改过的库会被拒绝
# mesh 为既有 morqex.mechanics.TetMesh；完整旋转3x3输运张量可用于标量弱式。
# A,b = backend.assemble(mesh, reaction_per_cell, diffusion_tensor_per_cell, source_per_cell)
# damage = AT2Damage(mesh, backend, fracture_toughness_J_m2=10., length_m=2e-6)
# mechanical = SplitElasticity(damage, young_Pa=93e9, poisson=.3, boundary='kubc')
# result = staggered_step(mechanical, accepted_d, accepted_history, eigenstrain=cell_mandel_strain,
#                        macro_strain=macro_mandel_strain, fatigue=True)
```

`AT2Damage` 用约束活跃集求解 `d_old<=d<=1`，报告投影 KKT 残差；不同于事后裁剪。`SplitElasticity` 采用各向同性体积/偏分裂，允许非均匀各向异性本征应变，但尚不支持一般各向异性断裂刚度。输出应力为单元体积平均。`FatigueHistory.trial(alpha)` 不修改旧状态；`staggered_step` 冻结已接受疲劳韧度，收敛后仅提交一次历史，需独立的加载路径细化验证。

`[reference] cache_forms=true` 可选择 `CachedP1ScalarBackend`：只缓存固定网格积分几何及稀疏装配映射，每次使用新的驱动能和疲劳韧度重装配。启用时先与编译MFEM独立交叉检查，缓存不代表冻结已损伤算子，也不是降阶。比较脚本 `scripts/validate_top_damage_reference.py` 和 `scripts/study_top_damage_path.py` 分别记录装配开销、缓存成本、同协议场误差及求解时间；单独报告输出I/O，不能把装配速度当作整电池加速倍数。

每个接受步在外部 `results/top_damage/<run>/` 写 `step_XXXXX.h5`、`step_XXXXX.vtu` 和实时更新的 `index.html`。HTML 可选步、Z切片和损伤/应力/疲劳字段；VTU保留完整三维节点和单元场。`--resume` 校验配置、网格、算法代码和编译后端身份，不允许静默混用旧状态。`--output` 可指定工作区结果子目录名；原路径存在时必须明确续算。

```bash
$PY scripts/run_local.py --label top_damage_audit -- $PY /home/zephyrain/lw/MORQEX/scripts/audit_top_damage.py
$PY -m morqex.top_damage status
$PY scripts/run_local.py --label top_damage_report -- $PY -m morqex.top_damage report --workspace /home/zephyrain/lw/morph_workspace
# 更新 editable 安装入口后：morqex top-damage status / report
```

```python
import numpy as np
from morqex.top_damage import ConductingNetwork, PartitionTransfer, OperatorStamp, TopDamageRegistry
from morqex.extensions import ComponentSpec

network = ConductingNetwork(4, np.array([[0,1],[1,2],[2,3]]),
    conductance_S=[1.,0.,1.], anchors=np.array([0]),
    channel="electronic", provenance="显式物理接触示例，非材料标定")
reachable = network.accessible()  # [True,True,False,False]；所有四个顶点仍保留
# ionic 网络需单独定义；可达性不是实际传输电阻，也不是完整反应活性判据。
transfer = PartitionTransfer(np.array([0,0,1,1]), np.array([0,1,2,3]),
    material_cell_volumes_m3=[1e-18,2e-18,3e-18,4e-18])
result = transfer.apply([10.,30.], maximum_mol_m3=50.)
# 分裂后每个区域持有自己的库存；电子失联不删除锂。固定材料参考网格专用。
registry = TopDamageRegistry().register("state-transfer",
    ComponentSpec("volume-overlap", PartitionTransfer))
```

`OperatorStamp(geometry,topology,materials,boundary,discretization,backend)` 的六个输入必须为规范内容的 SHA256，`require_same(current)` 拒绝陈旧算子。拓扑键须包含各物理通道接触与电导，材料键包含晶粒取向和历史相关物性；旧缓存尚未自动迁移到该契约。

新注册类型为 `fatigue/topology-update/homogenizer/cycle-stepper/state-transfer`，使用原有 `ComponentSpec`，无插件自动导入。注册本身不代表算法通过验证。完整字段、候选公式、许可证和验收条件在外部 11 份报告；文献能力矩阵为 `literature/top_damage_matrix.json`。多晶电化学疲劳、物理开度/拓扑、动态 DFN 和周期跳跃仍须逐项验收。

```bash
# 在下文资源运行器设置的外部工作目录中执行；默认生成真实 56 颗粒几何
$PY scripts/run_local.py --label sphere_geometry -- $PY /home/zephyrain/lw/MORQEX/scripts/prepare_sphere32.py
$PY scripts/run_local.py --label sphere_operator -- $PY /home/zephyrain/lw/MORQEX/scripts/build_sphere32_li.py --modes 64
$PY scripts/run_local.py --label li_short -- $PY /home/zephyrain/lw/MORQEX/scripts/run_sphere32_li.py --short --solver casadi
$PY scripts/run_local.py --label li_html -- $PY /home/zephyrain/lw/MORQEX/scripts/report_sphere32_li.py
$PY scripts/run_local.py --label li_cuda --gpu -- $PY /home/zephyrain/lw/MORQEX/benchmarks/probe_sphere32_cuda.py
```

`reports/MORQEX_sphere32_Li_geometry.html` 是几何页，`reports/MORQEX_sphere32_Li_short.html` 是已完成的 120 秒物理验证。32³ 两轮完整循环在当前 CPU 预算内尚未完成；`run_sphere32_li.py --cycles 2` 是研究入口，不是已通过保证。失败结果标记为未完成，报告脚本拒绝把它发布为完成协议。

```python
from morqex.generation import SpherePackSpec, sphere_pack
from morqex.nodewise import BindingSpec, bind_microstructures
from morqex.nodewise import HalfCellStack, lithium_halfcell_parameters
from morqex.nodewise.cache import load_node_kernel, NodeLocalization
from morqex.nodewise.localization import BatchLocalization

structure, particle_ids = sphere_pack(SpherePackSpec(
    shape=(32, 32, 32), particle_count=56, radius_voxels=3.3,
    minimum_gap_voxels=1.05, cbd_fraction=.14, spacing_m=(.8e-6,)*3,
    seed=104))
bindings = bind_microstructures(BindingSpec(mode="shared", shape=(32,)*3),
    6, "positive", workspace, template=structure)
# mode="slab" 在显式 reference 的各厚度区原生裁取；不缩放原始 CT。
# mode="statistical" 从全局 StatisticsCard 生成一个共享模板，节点状态独立。

parameters, provenance = lithium_halfcell_parameters(
    host="NMC532", active_fraction=float((structure.labels == 1).mean()))
kernel = load_node_kernel(external_operator_path)
stack = HalfCellStack([kernel]*6, parameters,
    constraint="fixed-thickness", metal_sei="none")
stack.update_parameters(parameters)  # 验证材料/浓度/温度参考与算子一致
var_pts = stack.var_pts(model)       # x_p=6；无负极颗粒状态
```

体素数和浓度状态数独立：32³ 是全部 32,768 个几何位置；本轮 64 个守恒浓度区域覆盖全部 56 个颗粒，并替代 DFN 球形径向状态。112 状态的三维力学算子亦已构建，但其在线编译未通过内存预算。尚未完成浓度模式、颗粒表面网格和应力尾部的收敛研究；守恒至机器精度不证明局部场误差小。每颗粒约 6–7 个体素直径，不能当作网格无关的球面解。

共享算例对应 6×32,768 个体素位置，共用一份不可变响应算子；并非 6 份独立存储，也不意味着这六块在真实电极内几何拼接。DFN 厚度控制体、代表微结构物理尺寸和尺度分离必须分别检查。

`HalfCellStack` 的节点数由正极 kernel 列表长度决定，可绑定不同合格结构。锂金属采用原生界面 BV 和有限锂库存，不构造负极粒子。当前验收使用固定参考几何、均温热耦合和 `NoDamage()`；平面 Li 空间热耦合有门禁，金属 SEI 尚未验收，不将颗粒副反应律移植为平面金属模型。32³ 完整界面算子拒绝启用脱粘；演化 cohesive 路线单独验证。NMC532/NMC811/LFP/LCO 和自定义 OCP/材料入口保留，变更机械参数需重建算子。

`BatchLocalization([NodeLocalization(path)], backend="cuda", parallel_nodes=6)` 调用源码 CUDA 核，FP64 输出全部体素应力、应变和位移；`states` 每项为 `(geometry_index, c, E_x, delta_T, damage)`。支持不同几何的索引并行；一个批次要求相同输出网格。DFN DAE 仍在 CPU。GPU 的编译/上传成本、传输和节流均分别计时，不与纯核时混淆。

MicroStructPy 和 Micro_Reconstruction 完整文本源码分别位于 `src/morqex/vendor`，二进制/数据归档在外部工作区。`morqex.generation.upstream.backend_status()` 报告原生运行时状态。后者需要 MATLAB/R，目前未安装和验收；Python 随机场不是其原生算法复现。父项目缺少顶层许可证，当前集成为本机私有使用，嵌套包许可不覆盖整体。

## 1. 环境与资源

Python ≥3.11，核心 NumPy/SciPy；`dfn`、`ct`、`gpu`、`test` 是可选依赖。项目提供集成的 `pybamm` 命名空间，建议独立环境，避免被另一个 PyBaMM 发行包覆盖。CPU 核需 g++；CUDA 核由 CuPy NVRTC 编译，不向源码写 `.so`。本地所有数值命令使用：

```bash
cd /home/zephyrain/lw/MORQEX
PY=/home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python
$PY scripts/run_local.py --label install --dependency-install -- $PY -m pip install -e '/home/zephyrain/lw/MORQEX[dfn,ct,gpu,test]'
$PY scripts/run_local.py --label tests -- $PY -m pytest /home/zephyrain/lw/MORQEX/tests -q --junitxml=test_results.xml
```

运行器创建源码外工作目录，设置 CPU affinity、单线程、nice +15、外部缓存与日志，监测进程组 RSS。默认 CPU 120 秒/墙钟 180 秒、RSS 1536 MiB、地址空间 3072 MiB、禁用 GPU。`--gpu` 才开放 GPU：显存预算 3072 MiB、空闲保留 4096 MiB、协作占空比 .2，超限中止。CUDA 不使用低虚拟地址上限，仍限制物理内存。占空比不是硬件配额；不改全局 GPU 设置或其他进程。

数值作业串行。长问题以外部断点续算；未收敛不视为通过。`--dependency-install` 只扩大安装文件预算，不放宽数值资源限制。入口 `Workspace(Path(...)).path(category,...)` 管理 `data/build/operators/runs/results/reports/checkpoints`。

## 2. 单位、相与数据格式

所有输入默认 SI：米、秒、Pa、mol/m³、J/m³、J/mol、K。体素数组 ZYX，网格点/位移分量 XYZ；Kelvin 张量顺序 `xx,yy,zz,yz,xz,xy`，剪切分量含 √2。浓度为活性材料内在浓度，库存以相体积加权；电极体积平均与固相条件平均不能混用。

`MicrostructureData(labels,spacing_zyx_m,origin_zyx_m,phase_kinds,provenance)` 为统一结构。相必须明确指定 pore/active/cbd/inert 等语义；不凭数字猜物理。真实零刚度孔隙、数值弱相、流体压力是不同模型。

## 3. GEMsim 与 CT

```python
from pathlib import Path
from morqex import Workspace, MicrostructureData
from morqex.gemsim import GEMsim
from morqex.microstructure import Volume, Microstructure

ws=Workspace(Path('/home/zephyrain/lw/morph_workspace'))
g=GEMsim(ws)
s=g.random_field(shape=(8,8,8),fractions=(.3,.55,.15),
    length_scale=(1.,)*3,seed=104,spacing=(1e-6,)*3,
    phase_kinds={0:'pore',1:'active',2:'cbd'})
volume=g.to_volume(s)
same=g.from_volume(volume,phase_kinds=s.phase_kinds)
mesh=g.mesh(s,cell_type='tetra')
```

`ct.SegmentedCT` 读取有来源记录的分割影像，`ArrayGeometry` 接收用户数组。GEMsim 全部 274 个源码文件的迁移记录见 `provenance/gemsim_migration.json`；无外部 checkout 运行依赖。AI、NGSolve、AMReX 等可选上游后端不等于已通过全部独立验证。

全体积入口 `microstructure.full_volume.prepare_full_volume(source,destination)` 要求 source 含 `path,sha256,axes_to_zyx,voxel_size_m`。现成 NLR 适配器使用 0 孔隙、1 活性、2 CBD，保存完整体积、周期六面连通区域、距离特征及 SHA manifest。非 NLR 相标号须先显式转换。

```bash
$PY scripts/run_local.py --label fetch -- $PY /home/zephyrain/lw/MORQEX/scripts/fetch_full_electrodes.py
$PY scripts/prepare_full_suite.py
$PY scripts/run_full_transport_suite.py
```

五个不同数据为 `nmc1uncal,nmc1cal,nmc2uncal,nmc2cal,nmc3uncal`。均是 NMC532 CT 活性相和来源作者重构 CBD；完整提供体积不等于整个制造电极片，不自动保证 RVE 代表性。

## 4. CBD、分相与掺杂

`materials.Phase/PhaseCatalog` 指定弹性 6×6 张量、化学应变/浓度导数、热膨胀、XYZ 扩散倍率、电子效率、离子效率、电解液体积分数。`Phase.isotropic` 是便利构造。CBD 可以同时承载、导电及通过微孔导离子；`topology.PorousCBD` 指定相应通道。微孔占比与有效输运效率是独立参数。全体积示例 CBD 离子效率 .2、液相占比 .47 是明确未标定的研究参数。

旧掺杂入口 `doping.DopantLaw,RegionalChemistry,composition_materials,composition_partition,attach_doping`，支持保留不同掺杂区域及其物性律。OCP/扩散/动力学/退化/弹性/膨胀变化必须来自用户输入或可追溯参数；合成灵敏度不是已知真实元素规律。

开放 POU 若挂接旧 `regional_chemistry` 而缺少空间迁移率投影会主动拒绝。接口可扩展不代表任意物理组合自动正确。

## 5. 开放参数与 OCP

```python
import numpy as np
from morqex.electrochemistry.open_materials import OpenMaterialCard,TabulatedOCP

p,audit=OpenMaterialCard.host_card('NMC811').build()
curve=TabulatedOCP(
    [.001,.25,.5,.75,.999],[4.3,4.,3.9,3.75,3.3],
    source='用户实验数据及批次编号',
    entropy_V_K=[0.,1e-5,2e-5,1e-5,0.],
    reference_temperature_K=298.15,extrapolate=False)
p,audit=OpenMaterialCard('custom',parameter_source='Chen2020',
    overrides={'Positive electrode diffusivity [m2.s-1]':1e-14},
    ocps={'positive':curve}).build()
```

宿主入口：NMC532→Mohtat2020，NMC811→Chen2020，LFP→Prada2013，LCO→Ai2020。`parameter_source` 可为包名或完整字典；`overrides` 任意覆盖，`supplements` 只补缺项且须提供 `source_note`。补用其他宿主的力学/热参数不能称为材料标定。在 NMC532 几何上使用其他宿主是虚拟材料替换，不是该宿主的真实影像。

`TabulatedOCP.from_csv(path,source=...,entropy_column=...)`：首行为标题，前两列 sto、V。PCHIP 保留节点和非单调性，不做凸化。Python/CasADi 使用同一分段多项式和导数。`primitive(sto,T)`、`entropy_primitive(sto)` 提供一致积分；`slope_bounds(T)` 求分段二次导数的精确极值。

`tabulate_parameter_ocps(p,knots=513,domain=(.001,.999),sampled_tolerance_V=1e-4)` 对原生函数采样并独立报告中点误差。抽查不是严格全区间界。任意有限表格可表示，不意味着任意 OCP 都对应良定的连续输运模型。浓度/温度有效域必须与表格一致。

## 6. 几何容量与倍率

```python
from morqex.electrochemistry.cell_design import CellDesign
design=CellDesign((.05,.9),(.1,.9),policy='balance-negative-thickness',
    negative_to_positive_ratio=1.1)
p2,audit=design.apply(p,active_fractions={'positive':.5})
```

SOC 窗口由用户或明确的电极 SOH 计算给定。`preserve` 保留铭牌容量，报告与几何库存之间的倍率差异；`balance-negative-thickness` 显式调整负极厚度、名义容量，记录前后值。不会擅自修改初始 SOC。库存容量不是高倍率实际可放容量；对照必须共享电流、配平和初始状态，不能只共享“0.1C”标签。

## 7. 力学参考与编译后端

稀疏 FEM：`mechanics.TetMesh,ElasticitySolver,isotropic_stiffness`。`energy.ResponseBuilder(solver,eigenstrain_modes).exact()` 给出固定几何、固定 C 的响应及统一能量 Hessian。宏观应变、本征应变、边界约束、体积归一化都进入同一能量。

```python
from morqex.acceleration.voxel import NativeStencil
from morqex.mechanics import isotropic_stiffness
C=np.stack([np.zeros((6,6)),isotropic_stiffness(100e9,.25),
            isotropic_stiffness(1e9,.3)])
beta=np.zeros((3,6));beta[1,:3]=1.25e-6/3
model=NativeStencil(labels,C,beta,spacing=(3.98e-7,)*3,backend='cpu')
# CUDA 创建必须在 run_local.py --gpu 子进程中进行。
# model=NativeStencil(labels,C,beta,backend='cuda',preconditioner='coarse')
c=np.where(labels==1,5000.,0.)
u,metrics=model.solve(c,rtol=1e-7,maxiter=1500)
stress=model.stress(u,c)                  # (6,Z,Y,X), Pa
mean=model.mean_stress(u,c)              # Kelvin 6-vector, Pa
```

兼容差分采用前向梯度及其精确伴随，由同一离散能量定义；它不是 Tet4/Hex8，性能比较须分开。`apply(u,out=None)`、`load(c,macro,T)`、`residual(u,c,macro,T)` 均为矩阵自由操作，输入形状/布局与输出缓冲区有检查。

体素求解器的内部位移未知量为 `u/h_min`，`solve` 及其断点中的 `u` 使用这个缩放量；`model.displacement_m(u)` 转成米。应变和应力接口已包含尺度变换。`visualization.voxel_fields.export_reference_fields` 分块输出完整的米制位移、应变、应力、能量密度和 von Mises 场，并记录单位、规范条件及 SHA；不得把内部向量直接标为米。

`preconditioner='fft'|'jacobi'|'coarse'|'phase-coarse'|'geometric'` 是可比较候选。`phase-coarse` 按材料区分 P0 聚合；`geometric` 是对称周期三线性传递与粗层重新离散的实验候选，粗层正则化仅用于预条件，完整 CT 实测未显示优势，不设为默认。coarse 通过 P0 Galerkin 粗网格、对称 Chebyshev 平滑和 CPU AMG V-cycle 预条件；物理 matvec/CG FP64，预条件可 FP32。不给物理孔隙偷偷增加刚度。

`StressControlledStencil(...,stress_Pa=(...))` 实现六分量平均应力 Schur 约束，`macro_strain(u,c,T)` 返回应变。零平均应力与仅轴向自由、横向固定是不同边界。

`solve(...,checkpoint=Path(...))` 未收敛可返回 `converged=False` 并保存外部断点；无断点则未收敛报错。大 CUDA 路线校验物理/算法指纹与真实残差后继续 Krylov 迭代。断点不是合格解；零模态、条件数和边界问题必须单独处理。

## 8. 全体积输运与正浓度基

`acceleration.transport.PeriodicTransport(phase,phase_conductivity,spacing)` 解周期谐均面通量单元问题；`solve(axis_xyz=0,rtol=...,maxiter=...,preconditioner='coarse')` 返回校正势、有效系数列和真实残差。其周期边界与旧 `phase_transport` 的 Dirichlet/绝缘边界不同，不能直接混作同一参考。

`ComponentBasis.allocate(components,features,target=...)` 为每个连通活性区域保留至少一个独立库存节点，再分配形状节点；`evaluate` 为非负分区统一基。节点可行域能包络重构浓度，演化过程仍需有界算法/事件检查。

`concentration_modes.full_volume.project_volume(directory,extra_nodes=8,chunk=32768)` 分块投影完整 CT 质量、表面积、周期 FV 刚度，不常驻 N×m 矩阵。刚度为同一细网格 Galerkin 投影，守恒/对称/半正定有测试。非线性化学能采用声明的 lumped 节点积分，须独立做模式收敛。

`concentration_modes.spatial_ports.build_spatial_ports(directory,bins=5)` 对整个 CT 积分空间质量 W[k,p] 和反应面积 A[k,p]，保存 `spatial_ports_5.npz/json`。返回真实体素分区体积，奇数网格不能假设每段恰为 1/5。当前稠密在线端口原型最多 512 个库存状态；更大连通区域集合须接稀疏/分块后端，不能静默合并库存。

```bash
$PY scripts/run_local.py --label ports -- $PY /home/zephyrain/lw/MORQEX/scripts/build_full_spatial_ports.py --dataset nmc1cal --bins 5
```

`electrochemistry.conservative_ports.ConservativePorts(M,T,L,area_weights,macro_volumes,constant_mode,provenance)` 支持一般有符号模式。M 对称正定，Tz=1，L1=1，物理面积与体积权重为正。`site_chemical_potential(g)` 先作 M⁻¹g 的变分投影；`modal_reaction_rate(j)` 与 `macro_current(j)` 使用互为伴随的同一反应端口。`audit(g,delta_phi,j)` 报告质量/功率恒等式和反应耗散；正耗散还需要匹配的被动动力学。

```python
from morqex.electrochemistry.conservative_ports import nodal_ports
ports=nodal_ports(weights,area_by_mode_macro,macro_volumes,
    provenance='用户 CT 批次与分割记录')
mu_site=ports.site_chemical_potential(energy_gradient)
a_dot=ports.modal_reaction_rate(site_current)
source_by_X=ports.macro_current(site_current)
check=ports.audit(energy_gradient,delta_phi_by_X,site_current)
```

这是一套**全局共享**微结构库存与空间反应接口。每 X 使用独立代表 RVE 并按 DFN 体积权重积分，也可以守恒；它是另一种局部均匀化闭合，不能仅凭数组复制判断重复计量。将它接入 DFN 必须更换全局状态子模型并使用一致的宏观积分权重；不能把现有每 X 独立的区域模型直接称为该全局闭环。上述构造器本身不完成这一步。

小 FEM 对照 `build_pou_operator(full,target=...,transport_law=...,relax_transverse=True)` 构造正 POU 和同能量力学。横向五个宏观应变可静态消去以满足零平均横向应力；微观波动仍为原 KUBC。此基并非 `RP=I`，旧分区反应证书不得复用。

## 9. DFN、热与双向力学

`MicrostructureElectrode(structure,provider,p,domain,topology=...,materials=...)` 从同一几何建立输运、可达表面和 FEM 能量；`partition/reduce` 构造区域，`detach/save` 分离在线核。微结构浓度**替换** DFN 径向状态，不能重复锂库存。

`MicrostructureStack(negative,positive,p,...)` 绑定电极；`update_parameters(p)` 写入几何分数并检查参考态。`build_microstructure_dfn` 是旧固定几何路线；`build_aging_dfn` 加入退化端口。

```python
from morqex.electrochemistry.open_particle import TransportLaw,build_open_dfn
from morqex.electrochemistry.aging_dfn import EnergyKernel
from morqex.electrochemistry.microstructure_dfn import MicrostructureStack
from morqex.degradation import DegradationOptions,NoDamage
stack=MicrostructureStack(negative,positive,p,allow_linear_ocp_entropy=True)
stack.update_parameters(p)
kernels={d:EnergyKernel(op) for d,op in stack.electrodes.items()}
model=build_open_dfn(stack,kernels,DegradationOptions(damage=NoDamage()),parameters=p)
```

`TransportLaw` 选择 `mobility='lattice'|'constant'`、非负梯度系数 `gradient_coefficient_J_m5_mol2`、正迁移率倍率及浓度有效域。扩散用正 Onsager 迁移率和化学势梯度；D 在此是 tracer/mobility 参考，不自动等于原生 DFN 的 Fick 化学扩散系数。非凸 OCP 要求显式正梯度能及界面分辨率验证。

`open_validation.chemical_primitives/open_observables` 独立审计 OCP 自由能/熵导数、梯度能、弹性能、含副反应库存的总锂、内部能、外部电功、冷却和耗散。当前热为集总模型；电解液审计限定理想热力学因子、常 t+，热容量为常数。非理想自由能、一般温度物性需要对应扩展，不可静默覆盖后冒充原材料模型。

## 10. 退化与插件契约

`DegradationOptions(negative=(...),positive=(...),damage=...,porosity_change=True,initial_damage=...,minimum_electrolyte_mol_m3=...)` 组合 `DiffusionLimitedSEI,PlatingStrippingDeadLi,EnergyDrivenDebonding,NoDamage`。默认参数为数值研究值，不是跨材料寿命标定。

自定义副反应实现 `reservoirs()->tuple[Reservoir,...]`、`resistance(states,area)`、`evaluate(ReactionContext)->SideContribution`。每个 Reservoir 明确库存、参考化学势、体积及初值；电流以氧化为正，返回各库存速率和平衡电位。不得无库存地删除不可逆锂。

`cohesive.build_cohesive`、`cohesive_reduction.compress_cohesive`、`aging_cache.save_aging_operator/load_aging_kernel` 管理真实界面刚度演化的离线/在线算子。CBD bulk 软化与界面脱粘不同；固定 C 静态算子不能代替黏弹性/塑性/损伤的完整历史模型。

`build_aging_dfn(...,factories=...)` 可替换 `states,mechanics,kinetics,particle,ocp,porosity,transport,thermal`。`extensions.Registry/ComponentSpec` 注册几何、材料、基、算子、反应、损伤和求解器，`standard_registry()` 给出内置目录。扩展须声明适用假设并数值验证。

## 11. 协议、恢复、统计与可视化

固定几何运行器 `MicrostructureDFNRuntime`；退化运行器 `runtime.aging.AgingRuntime`。使用 `advance/checkpoint/restore/recover`，恢复检查参数/算子指纹。富集需守恒转移、保持能量和状态连续；降阶有损投影不能称无损恢复。

`protocol.shared_control/solve_protocol` 复用同一符号模型完成 CC/CV/休息；原生 `pybamm.Experiment` 是独立对照。循环逐步检查截止、休息时长及 CV 停止电流；浓度或电解液提前事件是失败/适用域拒绝，不能算完整循环。高倍率物理截止不能为了画满曲线而关闭。

`OnlineMicrostructureOperator` 只保留小矩阵，`LocalizationMap/AgingLocalization` 按需载入局部映射。`moments(q)` 对线性应力分量提供二次矩；P99、主应力极值不是固定二次型。极端风险需说明采样误差、体积权重和网格奇异性，宏观/统计/场恢复分别计时。

从 DFN 恢复微结构依赖演化模式、宏观边界及温度，不是 SOC 的唯一逆映射。在不同 X 显示相同几何代表条件响应单元，不表示整片电极真实颗粒的唯一位置。LOD 仅影响绘图，不得替换全体积计算。

报告 `morph_workspace/reports/MORQEX_full_volume_report.html` 可离线打开。默认“完整 CT 分区”将同一 CT 的五个空间段对应到 DFN 位置，每个 LOD 体素只绘制一次；“代表体积重复”显示五份关联单元。两种视图都明确采用独立原生 DFN 均值着色，不能当作已验证的全 CT 非均匀浓度/应力演化。交互含倍率、循环、时间、CBD、旋转缩放及网格误差表。

`--render` 为隔离的 headless 软件 WebGL 测试取消虚拟地址上限，物理 RSS/CPU 限制不变且 GPU 禁用；不操作用户浏览器。外部 Noto Sans CJK SC Sans2.004 字体由 `fetch_report_font.py` 获取，生成已改名的 OFL 子集内嵌报告。字体原件和缓存均不入源码。

## 12. 复现与验收

```bash
$PY scripts/run_local.py --label native --gpu -- $PY /home/zephyrain/lw/MORQEX/benchmarks/probe_voxel_backend.py
$PY scripts/run_local.py --label transport --gpu -- $PY /home/zephyrain/lw/MORQEX/benchmarks/probe_voxel_transport.py
$PY scripts/run_local.py --label integrity -- $PY /home/zephyrain/lw/MORQEX/scripts/check_integrity.py
$PY scripts/run_local.py --label ct_dfn -- $PY /home/zephyrain/lw/MORQEX/benchmarks/study_full_ct_dfn.py --dataset nmc1cal --host NMC532 --rate 0.1
```

`study_full_ct_dfn.py` 先要求对应全体积输运已收敛，是“完整 CT 等效参数→原生球形 DFN”基线，不能作为完整局部力学双向闭环。`--nx/--nr` 控制网格，`--cycles` 请求循环，记录容量配平、温度、电流及终止原因。

旧版完整复现入口 `scripts/run_crossscale_release.py`，报告全部外置。公平计时须匹配边界、物理精度、容限、硬件、输出，并计入离线和恢复成本。比较应包含复用分解的高效 FEM；残差、经验误差和严格界分别报告。

完整 CT 固定载荷参考场收敛后，用 `benchmarks/plot_full_voxel_fields.py` 生成真实三个正交切片。色标采用明确的 symlog，覆盖全部最大值；界面尖角单点峰值未通过网格收敛，不能作为失效判据。完整 FP64 数组及 SHA 保存在 `results/full_voxel/<dataset>/weak_0e+00/fields`。

`scripts/build_open_wheel.py` 从外部内容寻址源码快照构建 wheel，不向源码创建 build/egg-info。`scripts/probe_open_installation.py` 在独立安装环境检查包含的 CUDA/HTML/JS 资源、C++ 编译、所有宿主入口、端口恒等式及原生 DFN。安装验证与材料级大规模双向耦合验收分开。

`acceleration.residual_space.minimum_residual_weights` 对兼容线性迭代解做成熟的仿射最小残差组合；必须重新计算物理 FP64 残差接受，不能只看 Gram 预测。`benchmarks/smooth_voxel_reference.py` 校验载荷/几何/源指纹后应用，并清除失效的 Krylov 递推状态。此项是数值候选，不作为方法原创性。

## 13. 模块索引

<!-- GENERATED_API_INDEX -->
### [morqex.acceleration.budget](../src/morqex/acceleration/budget.py)

- `GpuBudget(self)`
- `GpuBudget.batch(self)`
- `GpuBudget.metadata(self)`
- `GpuBudget.dot(self, a, b)`

### [morqex.acceleration.coarse_elasticity](../src/morqex/acceleration/coarse_elasticity.py)

- `coarse_matrix(phase, C, spacing, block=8)`
- `CoarseElasticity(self, model, *, block=None, smoothing_steps=4)`
- `CoarseElasticity.apply(self, r, out)`

### [morqex.acceleration.coarse_transport](../src/morqex/acceleration/coarse_transport.py)

- `coarse_matrix(phase, coefficient, spacing, block=8)`
- `CoarseTransport(self, model, scaling, *, block=None, smoothing_steps=4)`
- `CoarseTransport.matvec(self, x)`
- `CoarseTransport.apply(self, r, out)`

### [morqex.acceleration.diagonal](../src/morqex/acceleration/diagonal.py)

- `Diagonal(self, model)`
- `Diagonal.apply(self, r, out)`
- `Diagonal.direction(self, p, z, beta)`

### [morqex.acceleration.geometric](../src/morqex/acceleration/geometric.py)

- `coarsen(a)`
- `GeometricMultigrid(self, model)`
- `GeometricMultigrid.applied(self, k, x)`
- `GeometricMultigrid.cycle(self, k, rhs, out)`
- `GeometricMultigrid.apply(self, r, out)`

### [morqex.acceleration.macro_control](../src/morqex/acceleration/macro_control.py)

- `StressControlledStencil(self, *args, stress_Pa=(0.0,) * 6, **kwargs)`
- `StressControlledStencil.macro_strain(self, u, c, T=0.0)`
- `StressControlledStencil.apply(self, u, out=None)`
- `StressControlledStencil.load(self, concentration, macro=(0.0,) * 6, temperature=0.0, out=None)`
- `StressControlledStencil.residual(self, u, concentration, macro=(0.0,) * 6, temperature=0.0, out=None)`
- `StressControlledStencil.stress(self, u, concentration, macro=(0.0,) * 6, temperature=0.0, out=None)`

### [morqex.acceleration.pcg](../src/morqex/acceleration/pcg.py)

- `solve_pcg(model, c, macro, T, *, rtol, maxiter, checkpoint)`

### [morqex.acceleration.phase_coarse](../src/morqex/acceleration/phase_coarse.py)

- `assemble(phase, C, spacing, block, counts=None)`
- `PhaseCoarse(self, model, *, block=None)`

### [morqex.acceleration.poisson](../src/morqex/acceleration/poisson.py)

- `FastPoisson(self, model)`
- `FastPoisson.transfer(self, array, target, transpose=False)`
- `FastPoisson.apply(self, r, out)`
- `FastPoisson.direction(self, p, z, beta)`

### [morqex.acceleration.refinement](../src/morqex/acceleration/refinement.py)

- `solve_refined(model, c, macro, T, *, rtol, maxiter, checkpoint)`

### [morqex.acceleration.residual_space](../src/morqex/acceleration/residual_space.py)

- `minimum_residual_weights(residuals, *, chunk=131072)`
- `archive_iterate(checkpoint, destination)` — Hard-link immutable u plus its physical fingerprint, without a data copy.

### [morqex.acceleration.transport](../src/morqex/acceleration/transport.py)

- `ScalarPoisson(self, model, scaling)`
- `ScalarPoisson.apply(self, r, out)`
- `PeriodicTransport(self, phase, conductivity, spacing=(1.0, 1.0, 1.0), *, budget=None)`
- `PeriodicTransport.call(self, name, args)`
- `PeriodicTransport.apply(self, u, out)`
- `PeriodicTransport.solve(self, axis_xyz=0, *, rtol=1e-08, maxiter=2000, preconditioner='coarse')`

### [morqex.acceleration.verification](../src/morqex/acceleration/verification.py)

- `manufactured_periodic(n)`

### [morqex.acceleration.voxel](../src/morqex/acceleration/voxel.py)

- `NativeStencil(self, phase, C, beta=None, alpha=None, spacing=(1.0, 1.0, 1.0), *, backend='cpu', budget=None, preconditioner='fft')`
- `NativeStencil.dims(self)`
- `NativeStencil.displacement_m(self, u)` — Convert u/h_min to the physical periodic displacement fluctuation (m).
- `NativeStencil.apply(self, u, out=None)`
- `NativeStencil.load(self, concentration, macro=(0.0,) * 6, temperature=0.0, out=None)`
- `NativeStencil.stress(self, u, concentration, macro=(0.0,) * 6, temperature=0.0, out=None)`
- `NativeStencil.residual(self, u, concentration, macro=(0.0,) * 6, temperature=0.0, out=None)`
- `NativeStencil.mean_stress(self, u=None, concentration=None, macro=(0.0,) * 6, temperature=0.0)`
- `NativeStencil.solve(self, concentration, macro=(0.0,) * 6, temperature=0.0, *, rtol=1e-07, maxiter=1500, mixed_preconditioner=True, checkpoint=None, low_memory=None)`

### [morqex.adaptive_transport](../src/morqex/adaptive_transport.py)

- `AdaptiveBudgetError()`
- `AdaptiveGradientDiagnostics()`
- `AdaptiveGradientTransport(self, system, initial_basis, *, maximum_modes, maximum_enrichments_per_step=3, allow_full_fallback=False, enrichment_strategy='snapshot', resolvent_shifts=(0.025, 0.1, 0.5, 2.0), block_rank=2, channel_scales=None, macro_strain_scale=0.001, restart_dimension=None, retain_forcing_directions=False, certificate_method='graph', error_modes=16)`
- `AdaptiveGradientTransport.initialize(self, initial_deviation, *, error_tolerance, macro_strain=None)`
- `AdaptiveGradientTransport.step(self, previous, dt, channel_flux, *, error_tolerance, macro_strain=None)`
- `AdaptiveGradientTransport.save_checkpoint(self, state, workspace, name)`
- `AdaptiveGradientTransport.load_checkpoint(cls, system, path)`

### [morqex.adaptivity](../src/morqex/adaptivity.py)

- `AdaptiveResponseResult()`
- `AdaptiveResponse(self, builder, basis)`
- `AdaptiveResponse.evaluate(self, q, *, absolute_energy_tolerance)`

### [morqex.bases](../src/morqex/bases.py)

- `GeometricBasis()`
- `GeometricBasis.build(self, electrode, target)`
- `ExactBasis()`
- `ExactBasis.build(self, electrode, target)`
- `VariationalOperator()`
- `VariationalOperator.build(self, electrode, partition)`
- `PortBasis()` — Connected greedy partition from mechanical chemical-potential rows.
- `PortBasis.build(self, electrode, target)`

### [morqex.capabilities](../src/morqex/capabilities.py)

- `BackendCapabilities()`
- `BackendCapabilities.require(self, features)`

### [morqex.certification](../src/morqex/certification.py)

- `tree_poincare_lower_bound(diffusion)` — Positive lower bound on nonzero generalized graph eigenvalues.
- `DiffusionErrorBound()`
- `DiffusionCertificate(self, diffusion, basis, flux_patterns)`
- `DiffusionCertificate.initialize(self, initial_field, coefficients)`
- `DiffusionCertificate.step(self, previous, a_old, a_new, dt, channel_flux)`
- `StressErrorTransfer()`
- `StressErrorTransfer.build(cls, full_response, concentration_weights)`
- `StressErrorTransfer.stress_l2_bound(self, concentration_bound)`
- `StressErrorTransfer.quantile_bound(self, concentration_bound)` — Discrete fixed-mesh principal-stress quantile difference; not a continuum peak bound.
- `ConcentrationAccuracyError(self, actual_bound, tolerance)`
- `CertifiedDiffusionState()`
- `CertifiedControlledDiffusion(self, diffusion, basis, patterns, *, reference_concentration, maximum_concentration)` — Transactional Fick stepping: no over-tolerance state is returned.
- `CertifiedControlledDiffusion.initialize(self, initial_deviation, *, error_tolerance)`
- `CertifiedControlledDiffusion.step(self, state, dt, channel_flux, *, error_tolerance)`

### [morqex.cohesive](../src/morqex/cohesive.py)

- `CohesiveResponse()`
- `CohesiveResponse.retention(self, d)`
- `CohesiveResponse.hessian(self, d)`
- `CohesiveResponse.derivative(self, d)`
- `CohesiveResponse.recover(self, q, d)`
- `build_cohesive(electrode, P, *, normal_stiffness=1000000000000000.0, tangent_stiffness=500000000000000.0, residual=0.001)`

### [morqex.cohesive_reduction](../src/morqex/cohesive_reduction.py)

- `compress_cohesive(response, *, tolerance=1e-06, maximum_damage=0.85, maximum_modes=80, training_points=17)` — Return a variational response, plus an independent residual certificate.
- `cohesive_certificate(cohesive, arrays, q, damage)` — Residual bounds in exact arithmetic, with a declared roundoff allowance.

### [morqex.concentration](../src/morqex/concentration.py)

- `component_constants(diffusion)`
- `ConcentrationBasis()`
- `ConcentrationBasis.mass(self)`
- `ConcentrationBasis.encode(self, concentration)`
- `ConcentrationBasis.decode(self, coefficients)`
- `ConcentrationBasis.spectral(cls, diffusion, count)`
- `ConcentrationBasis.geometric(cls, diffusion, count)`
- `ConcentrationBasis.pod(cls, diffusion, snapshots, count, *, concentration_scale=1.0, target_action=None)`
- `ConcentrationBasis.boundary_resolvent(cls, diffusion, count, flux_patterns, *, shifts=(0.01, 0.1, 1.0, 10.0), moments=2)` — Established rational-resolvent response space, with exact mass modes.
- `ReducedDiffusion(self, diffusion, basis)`
- `ReducedDiffusion.step(self, coefficients, dt, *, flux=None, baseline=0.0, maximum=None)`
- `ControlledDiffusion(self, diffusion, basis, flux_patterns, *, reference_concentration, maximum_concentration)` — Low-dimensional stepping for a declared face-flux channel library.
- `ControlledDiffusion.step(self, coefficients, dt, channel_flux)`

### [morqex.concentration_modes.full_volume](../src/morqex/concentration_modes/full_volume.py)

- `project_volume(directory, *, extra_nodes=8, chunk=32768)`

### [morqex.concentration_modes.positive](../src/morqex/concentration_modes/positive.py)

- `ComponentBasis()`
- `ComponentBasis.dimension(self)`
- `ComponentBasis.allocate(cls, components, features, *, target)`
- `ComponentBasis.evaluate(self, components, features, *, xp=np)`
- `ComponentBasis.metadata(self)`
- `ComponentBasis.from_metadata(cls, data)`
- `full_features(arrays)`

### [morqex.concentration_modes.pou_operator](../src/morqex/concentration_modes/pou_operator.py)

- `POUOperator()`
- `POUOperator.metadata(self)`
- `POUOperator.detach(self)`
- `build_pou_operator(full, *, target, transport_law, relax_transverse=False)`

### [morqex.concentration_modes.spatial_ports](../src/morqex/concentration_modes/spatial_ports.py)

- `build_spatial_ports(directory, *, bins=5, chunk=32768)`

### [morqex.concentration_modes.tail](../src/morqex/concentration_modes/tail.py)

- `TailBasis()` — Geometry-only offline loads select stress-sensitive observables.
- `TailBasis.build(self, electrode, target)`

### [morqex.coupling](../src/morqex/coupling.py)

- `bind_chemical_modes(solver, diffusion, basis, expansion)`
- `CoupledState()`
- `CoupledDiagnostics()`
- `CoupledTransport(self, response, reduced_diffusion, *, chemical_curvature, reference_concentration, maximum_concentration, temperature=298.15, bounds_policy='sufficient')`
- `CoupledTransport.energy(self, a, macro_strain)`
- `CoupledTransport.initialize(self, coefficients, *, macro_strain=None)`
- `CoupledTransport.step(self, state, dt, *, macro_strain=None, flux=None, exchange_flux=None, reservoir_potential=None, tolerance=1e-11, maximum_iterations=40)`
- `CoupledTransport.concentration(self, state)` — Explicit bulk recovery: O(Nm); not called by sufficient-bound time stepping.
- `CoupledTransport.save_state(self, state, workspace, name)`
- `CoupledTransport.load_state(self, path)`

### [morqex.ct](../src/morqex/ct.py)

- `SegmentedCT()`
- `SegmentedCT.build(self, workspace=None)`
- `ArrayGeometry()`
- `ArrayGeometry.build(self, workspace=None)`

### [morqex.degradation](../src/morqex/degradation.py)

- `Reservoir()`
- `SideContribution()`
- `ReactionContext()`
- `SideLaw()`
- `SideLaw.reservoirs(self)`
- `SideLaw.resistance(self, states, area)`
- `SideLaw.evaluate(self, context: ReactionContext)`
- `DiffusionLimitedSEI()` — Cathodic solvent-diffusion limit, with passive-potential cutoff.
- `DiffusionLimitedSEI.reservoirs(self)`
- `DiffusionLimitedSEI.thickness(self, states, area)`
- `DiffusionLimitedSEI.resistance(self, states, area)`
- `DiffusionLimitedSEI.evaluate(self, c)`
- `PlatingStrippingDeadLi()` — Passive linear plating/stripping branches with finite stripping stock.
- `PlatingStrippingDeadLi.reservoirs(self)`
- `PlatingStrippingDeadLi.resistance(self, states, area)`
- `PlatingStrippingDeadLi.evaluate(self, c)`
- `EnergyDrivenDebonding()`
- `EnergyDrivenDebonding.evaluate(self, damage, release_rate)`
- `NoDamage()`
- `NoDamage.evaluate(self, damage, release_rate)`
- `DegradationOptions()`
- `standard_registry()`

### [morqex.doping](../src/morqex/doping.py)

- `DopantLaw()`
- `RegionalChemistry()`
- `RegionalChemistry.ocp_shift(self, k, sto)`
- `RegionalChemistry.primitive_shift(self, k, c, cmax)`
- `RegionalChemistry.metadata(self)`
- `RegionalChemistry.fingerprint(self)`
- `composition_partition(electrode, base_partition, fractions)` — Split regions at composition boundaries without losing inventories.
- `DopedOCP()`
- `DopedOCP.get_coupled_variables(self, v)`
- `attach_doping(stack, profiles)`
- `CompositionBasis()`
- `CompositionBasis.build(self, electrode, target)`
- `DopedOperator()`
- `DopedOperator.build(self, electrode, partition)`
- `composition_materials(structure, active_fractions, law, base_catalog)` — Refine active phase labels and their fixed mechanical properties.

### [morqex.electrochemistry.aging_cache](../src/morqex/electrochemistry/aging_cache.py)

- `CohesiveOnline()`
- `CohesiveOnline.hessian(self, d)`
- `save_aging_operator(path, op, cohesive)`
- `load_aging_kernel(path)`
- `AgingLocalization(self, path)`
- `AgingLocalization.recover(self, c, E, theta, d)`
- `load_cohesive_certificate(path)` — Load small residual coefficients without loading localization fields.

### [morqex.electrochemistry.aging_dfn](../src/morqex/electrochemistry/aging_dfn.py)

- `age(d, name, k=None)`
- `EnergyKernel()`
- `EnergyKernel.parts(self)`
- `EnergyKernel.hessian(self, d)`
- `AgingStates(self, param, options, stack, config)`
- `AgingStates.get_fundamental_variables(self)`
- `AgingStates.set_rhs(self, v)`
- `AgingStates.set_algebraic(self, v)`
- `AgingStates.set_initial_conditions(self, v)`
- `AgingStates.add_events_from(self, v)`
- `AgingMechanics(self, param, options, stack, kernels, config, feedback)`
- `AgingMechanics.get_fundamental_variables(self)`
- `AgingMechanics.set_algebraic(self, v)`
- `AgingMechanics.set_initial_conditions(self, v)`
- `AgingMechanics.get_coupled_variables(self, v)`
- `AgingNetwork(self, *args, config, **kwargs)`
- `AgingNetwork.get_coupled_variables(self, v)`
- `AgingParticle()`
- `AgingParticle.get_coupled_variables(self, v)`
- `AgingTotalCurrent()`
- `AgingTotalCurrent.get_coupled_variables(self, v)`
- `AgingPorosity(self, param, options, stack, config)`
- `AgingPorosity.get_coupled_variables(self, v)`
- `AgingPorosity.add_events_from(self, v)`
- `AgingPorosityRates()`
- `AgingPorosityRates.get_coupled_variables(self, v)`
- `AgingPorosityRates.add_events_from(self, v)`
- `AgingTransport()`
- `AgingTransport.get_coupled_variables(self, v)`
- `AgingThermal()`
- `AgingThermal.get_coupled_variables(self, v)`
- `AgingThermal.set_rhs(self, v)`
- `build_aging_dfn(stack, kernels, config, *, feedback=1.0, factories=None, control=None)`

### [morqex.electrochemistry.cathodes](../src/morqex/electrochemistry/cathodes.py)

- `cathode_parameters(family)`

### [morqex.electrochemistry.cell_design](../src/morqex/electrochemistry/cell_design.py)

- `CellDesign()`
- `CellDesign.apply(self, parameters, *, active_fractions=None)`

### [morqex.electrochemistry.conservative_ports](../src/morqex/electrochemistry/conservative_ports.py)

- `ConservativePorts()`
- `ConservativePorts.site_chemical_potential(self, energy_gradient)` — g=dW/da is not a pointwise chemical potential; first solve M mu=g.
- `ConservativePorts.site_macro_potential(self, macro_potential)`
- `ConservativePorts.modal_reaction_rate(self, site_current, faraday=96485.33212)`
- `ConservativePorts.macro_current(self, site_current)`
- `ConservativePorts.lithium_inventory(self, coefficients)`
- `ConservativePorts.audit(self, energy_gradient, macro_potential, site_current, faraday=96485.33212)`
- `nodal_ports(weights, area_by_mode_macro, macro_volumes, *, provenance)`

### [morqex.electrochemistry.dfn](../src/morqex/electrochemistry/dfn.py)

- `RVEParticleMechanics(self, param, options, stack, feedback=1.0)`
- `RVEParticleMechanics.get_coupled_variables(self, variables)`
- `RVEParticleMechanics.add_events_from(self, variables)`
- `EnergeticDiffusion(self, *args, curvature, feedback=1.0, **kwargs)`
- `EnergeticOCP(self, *args, feedback=1.0, **kwargs)`
- `EnergeticOCP.get_coupled_variables(self, variables)`
- `EnergeticThermal(self, *args, stack, feedback=1.0, **kwargs)`
- `EnergeticThermal.get_fundamental_variables(self)`
- `EnergeticThermal.set_rhs(self, variables)`
- `EnergeticThermal.set_initial_conditions(self, variables)`
- `EnergeticThermal.add_events_from(self, variables)`
- `build_model(stack, *, feedback=1.0, thermal=True)`

### [morqex.electrochemistry.linear_port](../src/morqex/electrochemistry/linear_port.py)

- `LinearPort(self, weights, *children)`
- `LinearPort.create_copy(self, new_children=None, perform_simplifications=True)`
- `linear_combination(weights, values)`

### [morqex.electrochemistry.mechanical_closure](../src/morqex/electrochemistry/mechanical_closure.py)

- `ElectrodeClosure()`
- `ElectrodeClosure.dfn_H(self)`
- `ElectrodeClosure.sphere_curvature(self)`
- `ElectrodeClosure.from_gemsim(cls, workspace, parameters, domain, *, shape=(5, 5, 5), seed=104, thermal_expansion=1e-05, cbd_young=1000000000.0)`
- `ElectrodeClosure.save(self, workspace, name)`
- `AxialStackClosure(self, negative, positive, parameters, *, constraint='pressure', pressure_Pa=0.0, thickness_change_m=0.0, separator_modulus_Pa=100000000.0)` — Plane-strain electrode/separator stack, common axial traction.
- `AxialStackClosure.traction(self, mean_deviations, temperature_deviation)`

### [morqex.electrochemistry.microstructure_cache](../src/morqex/electrochemistry/microstructure_cache.py)

- `OnlineMicrostructureOperator()` — No mesh, fine diffusion graph, basis or mechanical solver retained.
- `OnlineMicrostructureOperator.dimension(self)`
- `OnlineMicrostructureOperator.active_fraction(self)`
- `OnlineMicrostructureOperator.area_density(self)`
- `OnlineMicrostructureOperator.reference_concentration(self)`
- `OnlineMicrostructureOperator.transport(self)`
- `OnlineMicrostructureOperator.porosity(self)`
- `OnlineMicrostructureOperator.metadata(self)`
- `OnlineMicrostructureOperator.moments(self, q)`
- `OnlineMicrostructureOperator.reaction_port_bound(self, q, exchange_current, overpotential, temperature, *, feedback=1.0)` — MVT/Cauchy upper bound for fine-vs-coarse BV current at P a.
- `OnlineMicrostructureOperator.save(self, path)`
- `OnlineMicrostructureOperator.load(cls, path)`
- `save_operator(operator, workspace, name)`
- `LocalizationMap(self, path)`
- `LocalizationMap.recover(self, concentration, axial_strain, temperature_deviation, *, selection=None)`

### [morqex.electrochemistry.microstructure_dfn](../src/morqex/electrochemistry/microstructure_dfn.py)

- `key(domain, region, quantity)`
- `dot(a, b)`
- `MicrostructureStack(self, negative, positive, parameters, *, constraint='pressure', separator_modulus_Pa=100000000.0, allow_linear_ocp_entropy=False)`
- `MicrostructureStack.update_parameters(self, parameters)`
- `GeometryActiveMaterial(self, *args, operator, **kwargs)`
- `GeometryActiveMaterial.get_fundamental_variables(self)`
- `GeometryTransport(self, *args, stack, **kwargs)`
- `GeometryTransport.get_coupled_variables(self, variables)`
- `MicrostructureMechanics(self, param, options, stack)`
- `MicrostructureMechanics.get_coupled_variables(self, variables)`
- `MicrostructureParticle(self, *args, operator, feedback=1.0, **kwargs)`
- `MicrostructureParticle.get_fundamental_variables(self)`
- `MicrostructureParticle.get_coupled_variables(self, variables)`
- `MicrostructureParticle.set_rhs(self, variables)`
- `MicrostructureParticle.set_initial_conditions(self, variables)`
- `MicrostructureParticle.add_events_from(self, variables)`
- `MicrostructureOCP(self, *args, operator, feedback=1.0, **kwargs)`
- `MicrostructureOCP.get_coupled_variables(self, variables)`
- `MicrostructureKinetics(self, *args, operator, **kwargs)`
- `MicrostructureKinetics.get_coupled_variables(self, variables)`
- `PortThermal(self, *args, stack, feedback=1.0, **kwargs)`
- `PortThermal.get_fundamental_variables(self)`
- `PortThermal.get_coupled_variables(self, variables)`
- `PortThermal.set_rhs(self, variables)`
- `PortThermal.set_initial_conditions(self, variables)`
- `PortThermal.add_events_from(self, variables)`
- `build_microstructure_dfn(stack, *, feedback=1.0, thermal=True)`

### [morqex.electrochemistry.microstructure_operator](../src/morqex/electrochemistry/microstructure_operator.py)

- `phase_transport(volume, phases, axis_xyz=0, phase_factors=None)` — Unit-conductivity TPFA cell problem, Dirichlet x ends/insulated sides.
- `MicrostructureOperator()`
- `MicrostructureOperator.dimension(self)`
- `MicrostructureOperator.active_fraction(self)`
- `MicrostructureOperator.area_density(self)`
- `MicrostructureOperator.reference_concentration(self)`
- `MicrostructureOperator.transport(self)`
- `MicrostructureOperator.porosity(self)`
- `MicrostructureOperator.restrict(self, values)`
- `MicrostructureOperator.lift(self, values)`
- `MicrostructureOperator.chemical_potential(self, concentration, axial_strain, temperature_deviation)`
- `MicrostructureOperator.recover(self, concentration, axial_strain, temperature_deviation)`
- `MicrostructureOperator.metadata(self)`
- `MicrostructureOperator.transfer(self, new_operator, values)`
- `MicrostructureOperator.detach(self)`
- `MicrostructureOperator.save(self, workspace, name)`
- `MicrostructureElectrode(self, structure, provider, parameters, domain, *, thermal_expansion=1e-05, cbd_young=1000000000.0, topology=None, materials=None)` — Full active-voxel diffusion and fixed-C FEM energy on the SAME geometry.
- `MicrostructureElectrode.response(self)`
- `MicrostructureElectrode.H(self)`
- `MicrostructureElectrode.partition(self, target, strategy='geometry')`
- `MicrostructureElectrode.reduce(self, partition=None, strategy='full')`

### [morqex.electrochemistry.microstructure_parameters](../src/morqex/electrochemistry/microstructure_parameters.py)

- `microstructure_parameters(*, value_tolerance_V=2e-05, relative_slope_tolerance=0.05, source_parameter_set='OKane2022', allow_idealization=False)`

### [morqex.electrochemistry.microstructure_runtime](../src/morqex/electrochemistry/microstructure_runtime.py)

- `MicrostructureDFNRuntime(self, operator_paths, parameters, *, nx=4, feedback=1.0, constraint='pressure')` — Warm online stepping retains compact operators, never a fine FEM solver.
- `MicrostructureDFNRuntime.solution(self)`
- `MicrostructureDFNRuntime.advance(self, dt, current_A)`
- `MicrostructureDFNRuntime.recover(self, domain, *, x_index=0, selection=None)`
- `MicrostructureDFNRuntime.checkpoint(self, path)`
- `MicrostructureDFNRuntime.estimate_defect(self, domain, *, x_index=0, lookahead_s=10.0)` — Fine FV residual at the represented state, without a new FEM solve.
- `MicrostructureDFNRuntime.reaction_bound(self, domain, *, x_index=0)`
- `MicrostructureDFNRuntime.enrich_if_needed(self, new_paths, *, relative_port_tolerance=0.01, concentration_defect_tolerance=None, lookahead_s=30.0)`
- `MicrostructureDFNRuntime.restore(self, path)`
- `MicrostructureDFNRuntime.enrich(self, new_paths)`

### [morqex.electrochemistry.microstructure_validation](../src/morqex/electrochemistry/microstructure_validation.py)

- `ocp_primitive(parameters, domain, concentration)` — Primitive of pinned phase.U, including its endpoint asymptote terms.
- `macro_weights(processed_variable)`
- `thermodynamic_observables(solution, stack, parameters, *, feedback=1.0)`

### [morqex.electrochemistry.open_materials](../src/morqex/electrochemistry/open_materials.py)

- `PchipFunction(self, curve, child, *, entropy=False, order=0)`
- `PchipFunction.create_copy(self, new_children=None, perform_simplifications=True)`
- `TabulatedOCP()`
- `TabulatedOCP.fingerprint(self)`
- `TabulatedOCP.entropy(self, sto)`
- `TabulatedOCP.primitive(self, sto, temperature_K=None)`
- `TabulatedOCP.entropy_primitive(self, sto)`
- `TabulatedOCP.slope_bounds(self, temperature_K=None)`
- `TabulatedOCP.metadata(self)`
- `TabulatedOCP.from_csv(cls, path, *, source, entropy_column=None, **kwargs)`
- `OpenMaterialCard()`
- `OpenMaterialCard.build(self)`
- `OpenMaterialCard.host_card(cls, host, **kwargs)`
- `tabulate_parameter_ocps(parameters, *, knots=513, domain=(0.001, 0.999), sampled_tolerance_V=0.0001)` — Sample native/callable curves with a separate midpoint error audit.

### [morqex.electrochemistry.open_particle](../src/morqex/electrochemistry/open_particle.py)

- `TransportLaw()`
- `law_for(op)`
- `gradient_potential(op, c)`
- `OpenOCP()`
- `OpenOCP.get_coupled_variables(self, v)`
- `OpenParticle(self, *args, **kwargs)`
- `OpenParticle.add_events_from(self, v)`
- `OpenParticle.get_coupled_variables(self, v)`
- `build_open_dfn(stack, kernels, config, *, parameters, feedback=1.0, factories=None, control=None, temperature_interval_K=(273.15, 333.15))` — Compose raw OCP/positive mobility particles with the degradation ports.

### [morqex.electrochemistry.open_validation](../src/morqex/electrochemistry/open_validation.py)

- `chemical_primitives(parameters, domain, concentration, temperature_K)`
- `open_observables(solution, stack, parameters, config, *, feedback=1.0)`

### [morqex.electrochemistry.parameters](../src/morqex/electrochemistry/parameters.py)

- `convex_parameters(name='OKane2022', *, slope_floor_V=0.01, points=4001, allow_idealization=False)`

### [morqex.electrochemistry.protocol](../src/morqex/electrochemistry/protocol.py)

- `shared_control(v)`
- `add_protocol_event(model)`
- `solve_protocol(simulation, *, kind, cycles=1, current_A=1.0, lower_V=3.5, upper_V=4.0, duration_s=120.0, period_s=30.0, rest_s=300.0, cv_stop_A=0.25, pulse_discharge_A=0.5)`

### [morqex.electrochemistry.validation](../src/morqex/electrochemistry/validation.py)

- `check_protocol(solution, *, cycles, steps_per_cycle, lower_V, upper_V, current_A, rest_s=600.0)`

### [morqex.energy](../src/morqex/energy.py)

- `EnergyResponse()`
- `EnergyResponse.dimension(self)`
- `EnergyResponse.energy_density(self, q)`
- `EnergyResponse.gradient(self, q)`
- `EnergyResponse.macro_stress(self, q)`
- `EnergyResponse.generalized_chemical_force(self, q)`
- `EnergyResponse.projected_chemical_potential(self, q, concentration_mass)`
- `EnergyResponse.macro_strain_at_stress(self, coefficients, target_stress)`
- `EnergyResponse.stress(self, q, selection=None)`
- `EnergyResponse.strain(self, q, selection=None)`
- `EnergyResponse.displacement(self, q, selection=None)`
- `EnergyResponse.save(self, workspace, name)`
- `EnergyResponse.load(cls, path, *, expected_fingerprint=None)`
- `ResponseBuilder(self, solver, eigenstrain_modes, *, reference_eigenstrain=0.0, reference_displacement=None)`
- `ResponseBuilder.exact(self)`
- `ResponseBuilder.galerkin(self, basis)`
- `ResponseBuilder.displacement_basis(self, rank, *, energy_weighted=False)`
- `ResidualCertificate()`
- `ResidualCertificate.energy_norm_squared_density(self, q)`
- `ResidualCertificate.potential_error_density(self, q)`

### [morqex.error_dynamics](../src/morqex/error_dynamics.py)

- `SpectralErrorSpace(self, system, retained_modes)`
- `SpectralErrorSpace.configure_bounds(self, system)`
- `SpectralErrorSpace.decompose(self, error, null_basis)`
- `SpectralErrorSpace.advance(self, previous, slow_residual, tail_residual_norm, dt, padding)`

### [morqex.error_estimation.aging_port](../src/morqex/error_estimation/aging_port.py)

- `compile_reaction_certificate(kernel_path, destination=None)` — Offline only: integrate conjugate stress and QR regional area norms.
- `AgingReactionCertificate(self, path, kernel_path)`
- `AgingReactionCertificate.bound(self, q, damage, j0, eta, temperature, *, feedback=1.0, mechanical_residual=None, film_residual=None, film_resistance=None)`

### [morqex.extensions](../src/morqex/extensions.py)

- `GeometryProvider()`
- `GeometryProvider.build(self, workspace)`
- `BasisBuilder()`
- `BasisBuilder.build(self, electrode, target: int)`
- `OperatorBuilder()`
- `OperatorBuilder.build(self, electrode, partition: np.ndarray)`
- `Mechanism()` — A symbolic law called during assembly; must not mutate its context.
- `Mechanism.evaluate(self, context)`
- `ComponentSpec()`
- `ComponentSpec.fingerprint(self)`
- `Registry(self)` — Separate registries can be passed by the caller; no mutable global singleton.
- `Registry.register(self, kind, spec)`
- `Registry.create(self, kind, name, **configuration)`
- `Registry.manifest(self)`
- `validate_partition(electrode, labels)`
- `build_operator(electrode, basis: BasisBuilder, operator: OperatorBuilder, target)`

### [morqex.gemsim](../src/morqex/gemsim.py)

- `GEMsim(self, workspace: Workspace)`
- `GEMsim.from_volume(self, volume, *, phase_kinds: dict[int, str])`
- `GEMsim.to_volume(self, structure: MicrostructureData)`
- `GEMsim.load(self, path: Path, *, phase_kinds: dict[int, str])`
- `GEMsim.random_field(self, *, phase_kinds, **kwargs)` — GEMsim's CPU random-field baseline, not a trained CT reconstruction.
- `GEMsim.mesh(self, structure: MicrostructureData, *, cell_type='tetra', phases=None)`

### [morqex.geometry](../src/morqex/geometry.py)

- `MicrostructureData()`
- `MicrostructureData.fingerprint(self)`

### [morqex.gradient_transport](../src/morqex/gradient_transport.py)

- `immutable(value)`
- `EnergyAccuracyError(self, error_bound, tolerance)`
- `EnergyErrorBound()`
- `GradientState()`
- `GradientDiagnostics()`
- `QuadraticGradientSystem(self, response, diffusion, coordinate_basis, patterns, *, chemical_curvature, reference_concentration, maximum_concentration)`
- `QuadraticGradientSystem.error_space(self, count)`
- `QuadraticGradientSystem.full_step(self, concentration_deviation, dt, channel_flux, *, macro_strain=None)`
- `QuadraticGradientSystem.energy(self, c, macro_strain)`
- `QuadraticGradientSystem.norm(self, c)`
- `QuadraticGradientSystem.conservative_basis(self, basis)`
- `PortGradientModel(self, system, basis, *, projection='energy', certificate_method='graph', error_modes=16)`
- `PortGradientModel.dimension(self)`
- `PortGradientModel.energy(self, a, e)`
- `PortGradientModel.initialize(self, initial_deviation, *, macro_strain=None, error_tolerance)`
- `PortGradientModel.accept(self, state, tolerance)`
- `PortGradientModel.trial(self, previous, dt, channel_flux, *, macro_strain=None)` — Unaccepted candidate for analysis/adaptivity; use step for a qualified state.
- `PortGradientModel.error_after(self, previous_error, old, a, dt, u, e)` — Full-generator residual bound; also applicable to other approximations in this V.
- `PortGradientModel.step(self, previous, dt, channel_flux, *, error_tolerance, macro_strain=None)`
- `PortGradientModel.transfer_to(self, previous, new_model)`
- `PortGradientModel.transfer_preserving_state(self, previous, new_model)` — Non-nested restart only when its space contains the current field.

### [morqex.macroscale](../src/morqex/macroscale.py)

- `StackState()`
- `LayerStack(self, responses, thicknesses)`
- `LayerStack.solve(self, coefficients, *, lateral_strain=(0.0, 0.0, 0.0), pressure=None, thickness_change=None)`

### [morqex.materials](../src/morqex/materials.py)

- `Phase()`
- `Phase.isotropic(cls, name, kind, young, poisson, **kwargs)`
- `Phase.metadata(self)`
- `PhaseCatalog()`
- `PhaseCatalog.validate(self, structure)`
- `PhaseCatalog.ionic_phases(self, structure)`
- `PhaseCatalog.ionic_efficiencies(self, structure)`
- `PhaseCatalog.electronic_phases(self, structure)`
- `PhaseCatalog.electronic_efficiencies(self, structure)`
- `PhaseCatalog.porosity(self, structure)`
- `PhaseCatalog.metadata(self)`

### [morqex.mechanics.constraints](../src/morqex/mechanics/constraints.py)

- `ConstraintMap()`
- `constraints(mesh, boundary)`

### [morqex.mechanics.matrix_free](../src/morqex/mechanics/matrix_free.py)

- `MatrixFreeElasticitySolver(self, mesh, stiffness, *, boundary='kubc', relative_tolerance=1e-11, maximum_iterations=3000)`
- `MatrixFreeElasticitySolver.solve_reduced(self, rhs)`

### [morqex.mechanics.mesh](../src/morqex/mechanics/mesh.py)

- `TetMesh()`
- `TetMesh.from_volume(cls, volume, solid_phases=None)`
- `TetMesh.fingerprint(self)`
- `TetMesh.boundary(self)`
- `TetMesh.outer_nodes(self)`
- `TetMesh.traction_load(self, traction, face_mask=None)`
- `TetMesh.pressure_load(self, pressure, face_mask=None)`
- `TetMesh.body_load(self, body_force)`

### [morqex.mechanics.reference](../src/morqex/mechanics/reference.py)

- `MechanicalSolveError()`
- `ElasticState()`
- `ElasticitySolver(self, mesh, stiffness, *, boundary='kubc', pivot_tolerance=1e-13)`
- `ElasticitySolver.solve_reduced(self, rhs)`
- `ElasticitySolver.eigenstrain_load(self, eigenstrain)`
- `ElasticitySolver.solve(self, eigenstrain=0.0, *, macro_strain=None, prescribed_displacement=None, force=None, tolerance=1e-09)`

### [morqex.mechanics.tensors](../src/morqex/mechanics/tensors.py)

- `to_mandel(tensor)`
- `from_mandel(vector)`
- `isotropic_stiffness(young, poisson)`
- `validate_stiffness(stiffness)`
- `rotate_stiffness(stiffness, rotation)`
- `elastic_energy_and_mu(strain, concentration, stiffness, expansion, *, stiffness_derivative=None, eigenstrain_reference=None, reference_concentration=0.0)` — Local fixed-strain energy and mechanical mu, c in mol/m^3, beta in m^3/mol.

### [morqex.microstructure.analysis](../src/morqex/microstructure/analysis.py)

- `describe(volume)`
- `two_point(volume, phase=1, axis=0)` — Nonperiodic directional S2 with unbiased overlap normalization.
- `compare(reference, generated)`

### [morqex.microstructure.cli](../src/morqex/microstructure/cli.py)

- `main()`

### [morqex.microstructure.core.distributions](../src/morqex/microstructure/core/distributions.py)

- `DiscreteDnsFunc()` — 离散分布
- `smooth(self, p=[])` — 数据平滑分布
- `smooth.fit(cls, x_tab)` — 类方法,从距离统计拟合分布, 返回值为自身初始化参数
- `smooth.from_itv(cls, x1=-np.inf, x2=np.inf)` — 从指定区间内的均匀分布初始化
- `smooth.from_dist(cls, dist)` — 从scipy连续分布初始化
- `smooth.from_data(cls, data)` — 从采样数据初始化
- `smooth.x(cls)` — 预定义采样点
- `smooth.pad_x(cls)`
- `smooth.args(self)` — 初始化参数
- `smooth.dist(self)` — 分布类型
- `smooth.rvs(self, n: int=1)` — 采样,输入样本数目
- `smooth.pdf(self, kx)` — 概率密度函数 通过数组查找实现连续采样 无上边界
- `smooth.cdf(self, kx)` — 概率分布函数
- `same()` — 和数据保持完全相同的分布

### [morqex.microstructure.core.graph_codec](../src/morqex/microstructure/core/graph_codec.py)

- `dfs(node, graph, visited, stack)`
- `transpose(graph)`
- `get_sccs(graph)`

### [morqex.microstructure.core.particle_graph](../src/morqex/microstructure/core/particle_graph.py)

- `auto_check(func)` — 函数修饰器 在调用函数前检查依赖是否改变
- `auto_memo(func)` — 自动缓存函数调用结果, 根据key函数返回的值标识状态
- `hex_id(obj)`
- `OrderSet(self, *args, **kwargs)`
- `OrderSet.check(self)`
- `OrderSet.add(self, item)`
- `OrderSet.update(self, item_set)`
- `OrderSet.intersection(self, item_set)`
- `OrderSet.difference(self, item_set)`
- `OrderSet.union(self, item_set)`
- `Node(self, *in_args, **in_kwargs)`
- `Node.dec_key(self)` — 节点标识数据 自身数据域 依赖的数据域
- `Node.run_key(self)`
- `Node.modified(self)`
- `Node.update(self)` — 强制更新自身状态 source子类不重载
- `Node.check_update(self)` — 检查依赖状态 判断自身是否发生参数更新
- `Node.load_instance(self, instance)` — 从对象加载状态 保持自身地址
- `Node.cls_id_str(self)`
- `Node.cls_str(self)`
- `Node.id_str(self)`
- `Link(self, *args, **kwargs)` — 链接节点自身无参数 其状态由依赖节点的状态决定 其依赖的对象由key描述 其依赖对象的状态由dep_key描述
- `Link.dec_key(self)` — 标识自身当前依赖对象(自身参数) 返回值必须为tuple
- `Link.run_key(self)` — 在声明节点未改变前提下 查询自身运行状态 自身参数+依赖配置参数
- `Link.update(self)` — 更新自身状态 维持自身与输入节点的关系 Link子类重载
- `Link.modified(self)`
- `Link.check_update(self)` — 检查依赖状态 判断自身是否发生参数更新
- `ProcessBar(self)` — 进度条
- `ProcessBar.print_process(self, end='')`
- `ProcessBar.start(self, n_bat: int)`
- `ProcessBar.inc(self)`
- `ProcessBar.end(self)`
- `ProcessBar.print(self, *args, **kwargs)`
- `FeatureFunc()` — 从物体点计算最近邻特征场, 包括距离变换,取向变换,相对坐标变换
- `FeatureFunc.ftr_shape(self, obj_vol: ObjVol)`
- `FeatureFunc.transform(self, obj_pnt: ObjectPnt)` — 约定: 最内部维度包含不同特征, 默认返回四维数组
- `FeatureFunc.tgt_transform(self, obj_pnt: SeedsPnt=None)` — 计算采样点
- `FeatureFunc.sur_dist(self, obj_pnt: SeedsPnt=None)` — 目标表面距离(体素)
- `FeatureFunc.copy(self)`
- `NNFeatureFunc(self, measure=cdt)` — 最近邻特征提取
- `NNFeatureFunc.dec_key(self)`
- `NNFeatureFunc.copy(self)`
- `NNDst()` — 最近邻距离变换
- `NNDst.transform(self, obj_pnt: ObjectPnt)` — 约定: 最内部维度包含不同特征, 默认返回四维数组
- `NNDst.tgt_transform(self, obj_pnt: SeedsPnt)` — 计算采样点 取种子中心点
- `NNLoc()` — 最近邻相对位置变换
- `NNLoc.transform(self, obj_pnt: ObjVol)` — 约定: 最内部维度包含不同特征, 默认返回四维数组
- `SymLoc()` — 对称相对位置, 小于0部分取绝对值
- `SymLoc.transform(self, obj_pnt: ObjVol)`
- `Std2PntCor(self)` — 标准两点相关函数距离变换(同一相随机点距离,不归一化)
- `Std2PntCor.transform(self, obj_pnt1: ObjectPnt, obj_pnt2: ObjectPnt=None)` — 空间中所有点到随机物体点的距离
- `Std2PntCor.tgt_transform(self, obj_pnt: SeedsPnt=None)` — 计算采样点 取obj_obt占据体素
- `Std2PntCor.sur_dist(self, obj_pnt: SeedsPnt=None)` — 目标表面距离 和特征变换类型有关
- `DistFunc(self, *params)` — 分布函数 其特征相对域定义
- `DistFunc.reset_data(self)`
- `DistFunc.dec_key(self)`
- `DistFunc.run_key(self)`
- `DistFunc.update(self)`
- `DistFunc.update_params(self, params)`
- `DistFunc.invld_data_mask(cls, data, tol=0.0)`
- `DistFunc.from_pcdt(cls, dists: DistVol, ndim=1)`
- `DistFunc.from_pcpnt(cls, pnt1: ObjectPnt, pnt2: ObjectPnt=None, ftr_func=stdpcf)`
- `DistFunc.from_dist(cls, dist)`
- `DistFunc.from_data(cls, data)`
- `DistFunc.fit(self, data: np.ndarray)` — 仅保存数据,data的id随之改变,在update内更新
- `DistFunc.sample(self, n_sample=1)` — 采样 返回特征值 不缓存结果
- `DistFunc.density(self, x: np.ndarray)` — 返回某个特征出现的概率密度
- `DistFunc.cumdensity(self, x: np.ndarray)` — 返回某个特征出现的分部函数
- `DistFunc.nndensity(self, x: np.ndarray)` — 根据变量最小值运算得到最近邻距离
- `DistFunc.glbdensity(self, x: np.ndarray)` — 迭代计算全局距离分布
- `DistFunc.density_map_r(self, R_range=(0.0, 1.0))` — Wicksell's problem 修正2维概率密度
- `DistFunc.density_map_d(self)`
- `DistFunc.copy(self)`
- `dist_error(dist1: DistFunc, dist2: DistFunc)` — 两个distfunc间的误差
- `data_error(data1: np.ndarray, data2: np.ndarray)`
- `MulNormDist(self, mean=np.zeros(3), cov=np.eye(3))`
- `MulNormDist.dec_key(self)`
- `MulNormDist.run_key(self)`
- `NormDist(self, loc=0.0, scale=1.0)`
- `NormDist.dec_key(self)`
- `NormDist.run_key(self)`
- `UniformDist(self, loc=0.0, scale=1.0)`
- `UniformDist.dec_key(self)`
- `UniformDist.run_key(self)`
- `IndependDist(self, loc=-1.0, scale=2.0)` — 独立分布
- `ExponDist(self, loc=0.0, scale=1.0)`
- `ExponDist.dec_key(self)`
- `ExponDist.run_key(self)`
- `BetaDist(self, a=2.0, b=8.0, loc=0.0, scale=1.0)`
- `BetaDist.dec_key(self)`
- `BetaDist.run_key(self)`
- `DiscreteDist()`
- `DiscreteDist.density_map_r(self, R_range=(0.0, 1.0))` — Wicksell's problem 修正2维概率密度
- `DiscreteDist.density_map_d(self)`
- `DiscreteDist.rescale(self, scale)` — 分布按照尺度缩放
- `SmoothDist()`
- `CompDist(self, dfs=[UniformDist(-1.0, 2.0)] * 3)` — 三向复合分布, 支持假设沿x,y,z正方向的各向异性
- `CompDist.update(self)`
- `CompDist.dec_key(self)`
- `CompDist.run_key(self)`
- `CompDist.params(self)`
- `CompDist.params(self, value)`
- `CompDist.update_params(self, params)`
- `CompDist.from_dist(cls, dist)`
- `CompDist.from_data(cls, data)`
- `CompDist.from_pcdt(cls, dists: DistVol, ndim=1)`
- `CompDist.from_pcpnt(cls, pnt1: ObjectPnt, pnt2: ObjectPnt=None, ftr_func=stdpcf)`
- `CompDist.fit(self, data: np.ndarray)` — 仅保存数据,data的id随之改变,在update内更新
- `CompDist.sample(self, n_sample=1)` — 采样 返回特征值 不缓存结果
- `CompDist.density(self, x: np.ndarray)` — 返回某个特征出现的概率密度
- `CompDist.cumdensity(self, x: np.ndarray)` — 返回某个特征出现的分部函数
- `CompDist.xmin(self)`
- `CompDist.xmax(self)`
- `CompDist.xmin(self, value)`
- `CompDist.xmax(self, value)`
- `Region(self, res: float=1.0, size: list[float]=[80.0, 80.0, 80.0], periods: list[bool]=[False, False, False])` — 材料域 允许周期性边界
- `Region.dec_key(self)`
- `Region.update(self)`
- `Region.ndim(self)`
- `Region.len(self)`
- `Region.volume(self)`
- `Region.vxl_len(self)`
- `Region.vxl_unit(self)` — 体素单位长度
- `Region.shape(self)`
- `Region.limits(self)`
- `Region.rescale(self, fac: float)`
- `Region.adapt(self, rad)` — 自适应域分辨率 同一个SCC内共用reg 需要所有源可随reg变化
- `Region.copy(self)`
- `MSPRegion()`
- `MSPRegion.ndim(self)`
- `Region2D(self, res: float=1.0, size_2d: list[float]=[80.0, 80.0], periods_2d: list[bool]=[False, False])`
- `Region2D.ndim(self)`
- `Region2D.size(self)`
- `Region2D.size(self, value)`
- `Region2D.periods(self)`
- `Region2D.periods(self, value)`
- `Region2D.copy(self)`
- `Splitor(self, *args)`
- `rths_mod(get_rths)`
- `RadSplitor(self, *args)`
- `RadSplitor.get_rths(self, seeds, n_batch, ndim=3)` — 均匀切分半径阈值,得到n_batch个批量,给出n_batch-1个阈值
- `MulOtsuRadSplitor(self, n_rad_bin=200)` — MultiOTSU半径批量分配器
- `MulOtsuRadSplitor.dec_key(self)`
- `MulOtsuRadSplitor.hist_split(self, hist: np.ndarray, n_batch: int)` — 按照直方图给出划分 时间复杂度随批量数量增加很快 当批量数目大于4时将会在子区间上递归调用该算法进行划分
- `MulOtsuRadSplitor.get_rths(self, seeds: Seeds, nbat: int, reg=Region(1, [120, 120, 120], [False, False, False]))` — 在半径的体积加权后,按照多阈值OTSU算法给出当前种子半径阈值
- `UniRadSplitor(self)` — 均匀分割种子数目,并且按照半径排序. 最后一个批量将补齐.
- `UniRadSplitor.get_rths(self, seeds: Seeds, n_batch: int, ndim=3)` — 均匀分割种子数目
- `RndSplitor(self, prop0=0.2, propn=0.04, prop_ini=0.0)`
- `RndSplitor.dec_key(self)`
- `RndSplitor.get_prop(self, seeds: Seeds, n_it: int, cur_it: int)` — 获取分割后的种子
- `ExpSplitor(self, prop0=0.32, propn=0.04, prop_ini=0.0, fac=2.0)`
- `ExpSplitor.get_prop(self, seeds: Seeds, n_it: int, cur_it: int)` — 获取分割后的种子
- `PropSplitor(self, prop0=0.2, prop_ini=0.0)`
- `PropSplitor.get_prop(self, seeds: Seeds, n_it: int, cur_it: int)` — 获取分割后的种子
- `Object(self, mark=1)` — 物体采样点, 点相关图顶点基类
- `Object.dec_key(self)`
- `Object.discrete(self, reg: Region)`
- `Object.ndim(self)`
- `Object.ndim(self, value)`
- `Object.r(self)`
- `Object.r(self, value)`
- `Object.copy(self)`
- `Object.subs(self)` — 基本物体 保持顺序
- `Object.cnt(self)`
- `Seeds(self, bdata: np.ndarray=np.array([[]]), idx: np.ndarray=np.array([]))` — 种子集合模型,提供半径采样函数
- `Seeds.from_info(cls, loc: np.ndarray, r=0.0, mark=1, ort=np.zeros(3), b=0.0, a=0.0)`
- `Seeds.from_msp_seeds(cls, msp_seeds)`
- `Seeds.from_seeds(cls, value: Seeds)`
- `Seeds.update(self)`
- `Seeds.mod_time(self)`
- `Seeds.dec_key(self)`
- `Seeds.run_key(self)`
- `Seeds.copy(self)` — 返回单纯的种子复制 不包括引用关系
- `Seeds.mark_map(self, map)` — 属性映射
- `Seeds.void(self)`
- `Seeds.void_seeds(self)`
- `Seeds.data(self)`
- `Seeds.loc(self)`
- `Seeds.x(self)`
- `Seeds.y(self)`
- `Seeds.z(self)`
- `Seeds.r(self)` — 返回球形种子的半径, 椭球种子较短的轴
- `Seeds.r_median(self)`
- `Seeds.mark(self)`
- `Seeds.ort(self)`
- `Seeds.vx(self)`
- `Seeds.vy(self)`
- `Seeds.vz(self)`
- `Seeds.b(self)`
- `Seeds.a(self)`
- `Seeds.aabb(self)` — 边界盒子坐标:(种子,维度,最大最小坐标)
- `Seeds.cnt(self)`
- `Seeds.mask(self)`
- `Seeds.loc(self, value: np.ndarray)`
- `Seeds.x(self, value: np.ndarray)`
- `Seeds.y(self, value: np.ndarray)`
- `Seeds.z(self, value: np.ndarray)`
- `Seeds.r(self, value)`
- `Seeds.mark(self, value)`
- `Seeds.data(self, value)` — 注意seeds.data[:] = value将不会进行赋值
- `Seeds.ort(self, value: np.ndarray)`
- `Seeds.vx(self, value: np.ndarray)`
- `Seeds.vy(self, value: np.ndarray)`
- `Seeds.vz(self, value: np.ndarray)`
- `Seeds.b(self, value: np.ndarray)`
- `Seeds.a(self, value: np.ndarray)`
- `Seeds.move(self, objpnt: ObjectPnt, delta=1.2, natp=1000)` — 位置移动 返回不重叠的种子坐标
- `Seeds.mask_ref(self, mask: np.ndarray)` — 在当前索引基础上 创建索引引用种子(子集)
- `Seeds.idx_ref(self, idx: np.ndarray)`
- `Seeds.cut_ref(self, axis: int, rpos: float)`
- `Seeds.mask_hide(self, mask)` — 隐藏种子 坐标更新前不再作为源渲染
- `Seeds.remove_hide(self)` — 移除隐藏种子,返回种子引用
- `Seeds.data_mask_ref(self, data_mask: np.ndarray)` — 创建引用 重新设置底层数据掩膜
- `Seeds.data_idx_ref(self, data_idx: np.ndarray)` — 创建索引引用种子(子集) 重新设置索引
- `Seeds.rad_ref(self, r_th: float)`
- `Seeds.get_rths(self, n_rad: int, rad_splitor: RadSplitor)`
- `Seeds.itv_ref(self, r_itv: list[float])`
- `Seeds.rad_bat_ref(self, n_rad: int, cur_rad: int, rad_splitor: RadSplitor)` — ref_seeds未改变时 在多个半径引用间切换实现(切换前检查ref_seeds)
- `Seeds.cum_bat_ref(self, n_rad: int, cur_rad: int, rad_splitor: RadSplitor)` — ref_seeds未改变时 在多个半径引用间切换实现(切换前检查ref_seeds)
- `Seeds.mark_ref(self, mark: int)`
- `Seeds.phase_ref(self, phase: Phase)`
- `Seeds.rand_ref(self, prop: float=0)` — 随机批量引用
- `Seeds.rnd_bat_ref(self, n_it: int, cur_it: int, rnd_splitor: RndSplitor)`
- `Seeds.queue_ref(self, prop: float=0, mode='solid')` — 循环队列引用 不重复引用,直到所有种子完成一次引用
- `Seeds.sort(self)` — 属性从大到小 半径从大到小排序 对idx操作
- `Seeds.que_bat_ref(self, n_it: int, cur_it: int, rnd_splitor: PropSplitor)`
- `Seeds.discrete(self, reg: Region, rpos=1.0)`
- `Seeds.discrete_counts(self, reg: Region, rpos=1.0)` — 计算域内部种子体素数目 域内部设置周期性边界
- `Seeds.discrete_dmn(self, reg: Region, ashp: AlphaShape)`
- `Seeds.discrete_dmn_hrc(self, reg: Region, ashp: AlphaShape, n_hrc=4)` — 计算种子占据域 允许层级alphashape修正
- `Seeds.search_seeds_vf(self, reg: Region, vol_frac: float)` — 二分查找确定种子数目 优先保留靠前的种子
- `Seeds.mark_map(self, map)`
- `closest_axes(orientations)` — 计算多个方向向量与 x, y, z 轴方向最接近的坐标轴索引。
- `ellipsoids_to_circles(centers, orientations, radiis, heights, marks, num_circles, limits)` — 将多个圆柱体种子分解为多个二维圆形种子，使用向量化操作。
- `oblate_ellipsoids_to_ellipsoids(obla, num_elli=4)` — 扁椭球转化为椭球集合
- `Phase(self, r_dist=DistFunc(), v_frac: float=1.0, mark: int=1, name='')` — 相: 同一相种子的宏观描述
- `Phase.dec_key(self)`
- `Phase.run_key(self)`
- `Phase.copy(self)`
- `Phase.subs(self)`
- `Phase.reset_data(self)` — 清空缓存的拟合种子
- `Phase.sample(self, reg: Region)` — 采样,获得种子集合实例. 相对region的len定义半径
- `Phase.fit(self, seeds: Seeds, reg: Region)` — 从种子集合计算distfunc参数
- `Phase.update(self)`
- `Phase.sub_marks(self)`
- `Phase.sub_marks(self, value: int)`
- `Phase.sub_marks_str(self)`
- `AnisoPhase(self, r_dist=DistFunc(), ab_dist=UniformDist(1.0, 0.0), ort_dist=CompDist(), v_frac: float=1.0, mark: int=1, name='')` — 各向异性相
- `AnisoPhase.dec_key(self)`
- `AnisoPhase.run_key(self)`
- `AnisoPhase.copy(self)`
- `AnisoPhase.sample(self, reg: Region)` — 采样,获得种子集合实例. 相对region的len定义半径
- `AnisoPhase.update(self)`
- `ConstPhase()`
- `ConstPhase.fit(self, seeds: Seeds, reg: Region)`
- `ConstAnisoPhase()`
- `ConstAnisoPhase.fit(self, seeds: Seeds, reg: Region)`
- `CompPhase(self, phss: list[Phase]=[], name='')`
- `CompPhase.dec_key(self)`
- `CompPhase.run_key(self)`
- `CompPhase.copy(self)`
- `CompPhase.subs(self)` — 依赖的子相
- `CompPhase.sample(self, reg: Region)`
- `CompPhase.fit(self, seeds: Seeds, reg: Region)`
- `CompPhase.get_dict(self)`
- `PCNodeOperation(self, phs: Phase, reg: Region, *args)` — 顶点操作
- `PCNodeOperation.dec_key(self)`
- `PCNodeOperation.run_key(self)`
- `PCNodeSample(self, phs: Phase, reg: Region)`
- `PCNodeSample.update(self)`
- `PCNodeFit(self, phs: Phase, seeds: Seeds, reg: Region)`
- `PCNodeFit.dec_key(self)`
- `PCNodeFit.run_key(self)`
- `PCNodeFit.update(self)`
- `PhaseSeeds(self, phs: Phase, reg: Region)`
- `PhaseSeeds.run_key(self)`
- `PhaseSeeds.update(self)`
- `RefSeeds(self, ref_seeds: RefSeeds, func: Callable, *args)` — 按照一定规则自动同步更改的种子集合 避免外部管理分配
- `RefSeeds.mod_time(self)` — 底层数据更新 反向传播的时间更新
- `RefSeeds.dec_key(self)`
- `RefSeeds.run_key(self)`
- `RefSeeds.update(self)`
- `RefSeeds.copy(self)` — 返回单纯的种子复制 不包括引用关系
- `RefSeeds.base_ref(self)`
- `SelfRef(self, ref_seeds: Seeds)` — 创建种子集合的自身引用
- `SelfRef.dec_key(self)`
- `SelfRef.run_key(self)` — 依赖自身引用参数和源种子数据
- `SelfRef.update(self)`
- `PhaseRef(self, ref_seeds: RefSeeds, phs: Phase)` — 特定相的引用 支持复合相
- `PhaseRef.dec_key(self)`
- `PhaseRef.run_key(self)`
- `PhaseRef.update(self)`
- `RadItvRef(self, ref_seeds: RefSeeds, r1: float, r2: float)` — 半径区间引用
- `RadItvRef.dec_key(self)`
- `RadItvRef.run_key(self)`
- `RadItvRef.update(self)`
- `RadThRef(self, ref_seeds: RefSeeds, rth: float)` — 半径阈值引用
- `RadThRef.dec_key(self)`
- `RadThRef.run_key(self)`
- `RadThRef.update(self)`
- `RandRef(self, ref_seeds: RefSeeds, prop: float)` — 随机批量引用
- `RandRef.dec_key(self)`
- `RandRef.run_key(self)`
- `RandRef.update(self)`
- `BatRef(self, ref_seeds: RefSeeds, func: Callable, n_bat: int=1, cur_bat: int=0, splitor=Splitor())` — 批量引用基类 自动创建bat和curbat参数 在第一次使用时调用set_bat 更新状态
- `BatRef.dec_key(self)`
- `BatRef.run_key(self)`
- `BatRef.update(self)`
- `RadBatRef(self, ref_seeds: RefSeeds, n_bat: int=1, cur_bat: int=0, splitor=MulOtsuRadSplitor())` — 半径批量引用关系
- `RadBatRef.mod_time(self)` — 底层数据更新 不进行引用更新传播 而是利用cur_bat在外部管理更新(半径批量间独立)
- `MulRadBatRef(self, ref_seeds: RefSeeds, n_bat: int=1, cur_bat: int=0, splitor=MulOtsuRadSplitor())` — 多半径批量引用 跨批量重建
- `RndBatRef(self, ref_seeds: RefSeeds, n_bat: int=1, cur_bat: int=0, splitor=RndSplitor())` — 随机批量引用关系
- `QueBatRef(self, ref_seeds: RefSeeds, n_bat: int=1, cur_bat: int=0, splitor=PropSplitor())`
- `RefOperator(self, seeds1: RefSeeds, seeds2: RefSeeds, func: Callable)` — 引用操作符
- `RefOperator.dec_key(self)`
- `RefOperator.run_key(self)`
- `RefOperator.update(self)`
- `RefOperator.mod_time(self)` — 反向传播的时间更新
- `AndRef(self, seeds1: RefSeeds, seeds2: RefSeeds)` — 交集引用
- `AndRef.mod_time(self)` — 反向传播的时间更新
- `OrRef(self, seeds1: RefSeeds, seeds2: RefSeeds)` — 并集引用
- `SubRef(self, seeds1: RefSeeds, seeds2: RefSeeds)`
- `SubRef.mod_time(self)` — 反向传播的时间更新
- `Bound(self, axis: int=0, rpos=0.0, mark=1)` — 边界物体,一般对应方向不设置周期性
- `Bound.dec_key(self)`
- `Bound.discrete(self, reg: Region)`
- `FillBound(self, axis: int=0, rpos: tuple[float]=(0.0, 0.5), mark=1)` — 填充边界场
- `FillBound.dec_key(self)`
- `FillBound.discrete(self, reg: Region)`
- `FillBound.r(self)`
- `FillBound.r(self, value)`
- `FillBound.copy(self)`
- `Cylinder(self, axis: int=0, rpos: tuple[float]=(0.5, 0.5), r: float=0.0, h: tuple[float]=(0.0, 1.0), mark=1)`
- `Cylinder.dec_key(self)`
- `Cylinder.discrete(self, reg: Region)`
- `Cylinder.copy(self)`
- `Cube(self, axis: int=0, rpos: tuple[float]=(0.5, 0.5, 0.5), h: tuple[float]=(0.1, 0.1, 0.1), mark=1)`
- `Cube.dec_key(self)`
- `Cube.discrete(self, reg: Region)`
- `Cube.copy(self)`
- `Cube.r(self)`
- `Cube.r(self, value)`
- `ImgObject(self, img=PhaseVol())` — 从三维数组建立的图像物体对象 用图像的voronoi覆盖剖分网格
- `ImgObject.dec_key(self)`
- `ImgObject.run_key(self)`
- `ImgObject.discrete(self, reg: Region)`
- `ImgObject.vol(self)`
- `ImgObject.vol(self, value)`
- `ImgObject.inv(self)`
- `ImgObject.size(self)`
- `ImgObject.r(self)`
- `ImgObject.mark(self)`
- `ImgObject.mark(self, value)`
- `FileImg(self, fname='data/img.png', ths=[], n_smth=0, cuts=(slice(None), slice(None), slice(None)), process_func=lambda img: img)` — 从文件加载的图像对象
- `FileImg.dec_key(self)`
- `FileImg.run_key(self)`
- `FileImg.img(self)`
- `CompObject(self, objs: list[Object]=[])`
- `CompObject.dec_key(self)`
- `CompObject.run_key(self)`
- `CompObject.discrete(self, reg: Region)`
- `CompObject.ndim(self)`
- `CompObject.r(self)`
- `CompObject.r(self, value)`
- `CompObject.mark(self)`
- `CompObject.mark(self, value)` — 设置mark, obj可以是复合物体
- `CompObject.subs(self)`
- `CompObject.copy(self)`
- `CompImg(self, vol: PhaseVol, res=0.0)`
- `ObjectPnt(self, obj: Object, reg: Region, *args)`
- `ObjectPnt.dec_key(self)`
- `ObjectPnt.run_key(self)`
- `ObjectPnt.ndim(self)`
- `ObjectPnt.pnt_r(self)`
- `ObjectPnt.nvxl(self)`
- `ObjectPnt.ftr_trans(self, ftr_func: FeatureFunc)` — 缓存不同ftr的变换结果 srs变化时清空
- `ObjectPnt.sur_ftr_vol(self, tgt: SeedsPnt=None, ftr_func: FeatureFunc=surdt)` — 到目标的表面距离场 tgt变化时需要重新计算sur_dist
- `ObjectPnt.cen_ftr_vol(self, srs: SeedsPnt=None, ftr_func: FeatureFunc=surdt)` — 起点到目标中心距离场
- `ObjectPnt.vol(self)`
- `ObjectPnt.seeds(self)`
- `ObjectPnt.obj_cnt(self)`
- `AlphaShape(self, rpos=1.0, rfix=0.6)` — 种子占据域的形状 alpha-shape计算修正
- `AlphaShape.dec_key(self)`
- `AlphaShape.a_shape(self, vol, r, reg)`
- `InvAShape()`
- `InvAShape.a_shape(self, vol, r, reg)`
- `PolyShape(self, rpos=1.0, rfix=0.1)` — 晶体间隙形状修正
- `PolyShape.a_shape(self, vol, r, reg)`
- `VoroShape(self, phs=Phase(UniformDist(), 0.0, 10000))` — 晶体形态修正
- `VoroShape.dec_key(self)`
- `VoroShape.a_shape(self, vol, r, reg)`
- `SeedsShape(self, rpos=1.0)` — 种子外围形状 不进行修正
- `SeedsShape.dec_key(self)`
- `SeedsShape.a_shape(self, vol, r, reg)`
- `AnisoShape()` — 各向异性形状,有边界限制
- `SeedsPnt(self, seeds: Seeds, ashp: AlphaShape, reg: Region)`
- `SeedsPnt.dec_key(self)`
- `SeedsPnt.run_key(self)`
- `SeedsPnt.mark(self)`
- `SeedsPnt.sub_marks(self)`
- `SeedsPnt.pnt_r(self)`
- `SeedsPnt.vol(self)`
- `SeedsPnt.from_phs_obj(cls, phs_obj: PhaseObject, seeds: Seeds, reg: Region)`
- `SeedsPnt.obj_cnt(self)`
- `SeedsPnt.seeds(self)`
- `ConstSeedsPnt()` — 固定的种子点
- `CumSeedsPnt(self, seeds: RefSeeds, rad_bat: RadBatRef, ashp: AlphaShape, reg: Region)` — 累积种子采样点 缓存层级渲染结果
- `CumSeedsPnt.dec_key(self)`
- `CumSeedsPnt.run_key(self)`
- `CumSeedsPnt.get_bat_vol(self, cur_bat)` — 调用此函数时 对应seeds的cur_bat之上累积批量已不再改变
- `CumSeedsPnt.get_bat_seeds(self, cur_bat)`
- `CumSeedsPnt.get_cum_vol(self, cur_bat)` — 按照输入cur_bat缓存/读取数据 本身不起作用
- `CumSeedsPnt.vol(self)` — 根据缓存的种子渲染结果 实时计算当前cur_bat时的数组 因为pnt本身不随cur_bat改变,此为vol的行为
- `CumSeedsPnt.seeds(self)`
- `HrcSeedsPnt(self, seeds: RefSeeds, ashp: AlphaShape, reg: Region, n_hrc: int=1)` — 固定的层级渲染种子
- `HrcSeedsPnt.dec_key(self)`
- `HrcSeedsPnt.run_key(self)`
- `HrcSeedsPnt.vol(self)`
- `SeedsPntOperator(self, pnt1: ObjectPnt, pnt2: ObjectPnt, func: Callable)` — 采样点操作 两个采样点为相同类型的Pnt
- `SeedsPntOperator.dec_key(self)`
- `SeedsPntOperator.run_key(self)`
- `SeedsPntOperator.obj(self)`
- `SeedsPntOperator.vol(self)`
- `SeedsPntOperator.ndim(self)` — 返回两个采样点更大的维度
- `SeedsPntOperator.pnt_r(self)`
- `SeedsPntOperator.seeds(self)`
- `SeedsPntOperator.obj_cnt(self)`
- `AddSeedsPnt(self, pnt1: ObjectPnt, pnt2: ObjectPnt)`
- `AddSeedsPnt.vol(self)`
- `AddSeedsPnt.pnt_r(self)`
- `AddSeedsPnt.obj_cnt(self)`
- `PhaseObject(self, phs=Phase(), ashp=AlphaShape())` — 相的采样点 为相种子添加alphashape描述 用于点相关图定义声明顶点
- `PhaseObject.dec_key(self)`
- `PhaseObject.run_key(self)`
- `PhaseObject.discrete(self, reg: Region)` — 非实体,仅用于可视化
- `ConstPhaseObject()` — 固定的相物体 不在迭代中更新坐标
- `ObjectRelation(self, srs=Object(), tgt=PhaseObject(), ftr_func=FeatureFunc(), dns_func=DistFunc())` — 相的采样点 为相种子添加alphashape描述 用于点相关图定义声明顶点(只与连通关系有关)
- `ObjectRelation.dec_key(self)`
- `ObjectRelation.run_key(self)`
- `ObjectRelation.tgt_itsc(self, phss: set[Phase])` — 目标相的交集
- `ObjectRelation.srs_itsc(self, phss: set[Phase])` — 源相的交集
- `ObjectRelation.copy(self)`
- `PhaseRelation(self, srs=Object(), tgt=PhaseObject(), ftr_func=FeatureFunc(), dns_func=DistFunc())` — 相的采样点 为相种子添加alphashape描述 用于点相关图定义声明顶点(只与连通关系有关)
- `PhaseRelation.inv(self)`
- `SeedsPntCor(self, srs: ObjectPnt, tgt: SeedsPnt, ftr_func: FeatureFunc, dns_func: DistFunc)` — 种子间的点相关关系 输入seedspnt
- `SeedsPntCor.dec_key(self)`
- `PCEdgeOperation(self, pcfs: set[SeedsPntCor], *args)` — 单次点相关操作
- `PCEdgeOperation.dec_key(self)`
- `PCEdgeOperation.run_key(self)`
- `PCEdgeSample(self, pcfs: set[SeedsPntCor])`
- `PCEdgeSample.update(self)`
- `PCEdgeSampleInit(self, pcfs: set[SeedsPntCor])`
- `PCEdgeSampleInit.update(self)`
- `PCEdgeFit(self, pcfs: set[SeedsPntCor])` — 边的拟合 采样的对称过程
- `PCEdgeFit.update(self)`
- `PCEdgeCheck(self, pcfs: set[SeedsPntCor])` — 边的检查 移除不合要求的种子
- `PCEdgeCheck.update(self)`
- `PCFuncFit(self, pcfs: set[SeedsPntCor])`
- `PCFuncFit.update(self)`
- `PCEdgeMove(self, pcfs: set[SeedsPntCor], T: float=1.0, cur_error=None)` — 点相关图边的移动和接受检验
- `PCEdgeMove.update(self)`
- `PCGAssumption(self, n_hrc, n_it, rnd_splitor=ExpSplitor(), rad_splitor=MulOtsuRadSplitor())` — 点相关采样重建假设
- `PCGAssumption.dec_key(self)`
- `PCGAssumption.get_n_hrc(self, seeds)`
- `PCGAssumption.get_n_it(self, seeds)`
- `AnnealAssumption(self, n_hrc, r_it, rnd_splitor=ExpSplitor(), rad_splitor=MulOtsuRadSplitor(), T=ExpSplitor())` — 模拟退火重建假设
- `AnnealAssumption.dec_key(self)`
- `AnnealAssumption.get_n_it(self, seeds)`
- `PhsRltSCC(self, phss: list[Phase]=[], rlts: list[ObjectRelation]=[])` — 强连通分量重建 内部种子部分顺序
- `PhsRltSCC.dec_key(self)`
- `PhsRltSCC.run_key(self)`
- `PhsRltSCC.update(self)`
- `PhsRltSCC.parse_phs_rlts(self, seeds: RefSeeds, reg: Region, assp: PCGAssumption, tgt_sub=True)` — 将相和物体间关系转化为 种子和物体采样点之间的关系
- `PhsRltSCC.sample(self, seeds: Seeds, reg: Region, assp=sample_assp, rnd=None, on=True)`
- `PhsRltSCC.fit(self, seeds: Seeds, reg: Region, assp=fit_assp, rnd=None, on=True)`
- `PhsRltSCC.ann_fit(self, seeds: Seeds, reg: Region, assp=nonit_assp, rnd=None, on=True)`
- `PhsRltSCC.fill(self, seeds: Seeds, reg: Region, assp=fill_assp, rnd=None, on=True)`
- `PhsRltSCC.check(self, seeds: Seeds, reg: Region, assp=check_assp, rnd=None, on=True)`
- `PhsRltSCC.anneal(self, seeds: Seeds, reg: Region, assp=anneal_assp, rnd=None, on=True)`
- `PhsRltSCC.sub_rlts(self)` — 按照目标相分解关系, 获得重建相间关系
- `PhsRltSCC.is_cycle(self)`
- `PhsRltSCC.pcg(self)` — 相之间构成的点相关图 参数化物体不参与 所有节点为待重建节点 复合相转化为基本相
- `PhsRltSCC.sccs_dict(self)`
- `PhsRltSCC.sccs_nodes(self)`
- `PhsRltSCC.copy(self)`
- `PhsRltSCC.get_rlt(self, *mark)`
- `PhsRltSCC.get_phs(self, mark)`
- `PhsRltGraph(self, phss: list[Phase]=[], rlts: list[ObjectRelation]=[])` — 相间关系图 解析连通分量 转化相关系定义
- `PhsRltGraph.dec_key(self)`
- `PhsRltGraph.run_key(self)`
- `PhsRltGraph.update(self)`
- `PhsRltGraph.sample(self, reg: Region, assp=sample_assp, seeds=None, rnd=None, on=True)` — 采样重建 所有种子将被保留
- `PhsRltGraph.fit(self, seeds: Seeds, reg: Region, assp=fit_assp, rnd=None, on=True)` — 声明连通分量 输入先验相间关系 拟合
- `PhsRltGraph.ann_fit(self, seeds: Seeds, reg: Region, assp=nonit_assp, rnd=None, on=True)`
- `PhsRltGraph.fill(self, reg: Region, assp=fill_assp, seeds=None, rnd=None, on=True)` — 声明连通分量 输入先验相间关系 拟合
- `PhsRltGraph.check(self, seeds: Seeds, reg: Region, assp=check_assp, rnd=None, on=True)` — 声明连通分量 输入先验相间关系 拟合
- `PhsRltGraph.anneal(self, reg: Region, assp: AnnealAssumption=anneal_assp, seeds=None, rnd=None, on=True)` — 声明连通分量 模拟退火重建
- `tgt_itsc(phss: set[Phase], rlts: set[ObjectRelation])`
- `idp_decomp(rlts: set[ObjectRelation])`
- `srs_itsc(phss: set[Phase], rlts: set[ObjectRelation])`
- `sub_phss(phss: set[Phase])` — 分解复合相
- `sub_comp_rlts(rlts: set[ObjectRelation], sub_srs=False)` — 分解复合物体参与的关系, 分解复合相作为目标参与的关系  当计算依赖关系时启用sub_srs, 当计算重建关系时关闭.
- `get_pcg(rlts: set[ObjectRelation])` — 获取点相关图 在基本相间关系定义基础上
- `get_sccs(pcg, sub_rlts)` — 强连通分量字典 返回字典: 分量相元组->相互作用 此时包含物体关系
- `get_sccs_nodes(pcg)`
- `Voronoi(self, seeds=Seeds(), id=-1)`
- `Voronoi.dec_key(self)`
- `Voronoi.discrete(self, reg: Region)` — 离散到域
- `Voronoi.attr_map(self, map)`
- `Voronoi.r(self)`
- `Voronoi.r(self, value)`
- `Voronoi.mark(self)`
- `Voronoi.mark(self, value)`
- `LaguerreVoro()`
- `LaguerreVoro.from_vol(cls, vol, ndim=None, min_vxl=3)` — 求Voronoi覆盖 转化为种子表示 对属性不为0的部分生效
- `LaguerreVoro.resample(self, reg)` — 种子Voro离散后重新计算Voronoi覆盖
- `LaguerreVoro.voro_reg(self, reg: Region)`
- `filter_region(vol, itv_th, ndim=None, nvxl=3)` — 按照局部域大小过滤体素网格
- `PolyVoro()`
- `PolyVoro.discrete(self, reg: Region, rnd_mark=True)` — 离散到域
- `Generator(self, objs)` — 物体生成器

### [morqex.microstructure.core.rasterization](../src/morqex/microstructure/core/rasterization.py)

- `shperef_mask(d)` — 获取一定直径球表面的体素掩膜
- `shperef_rcoord(d)` — 获取一定直径球表面的相对中心坐标
- `circlef_mask(d)` — 获取一定直径圆表面的体素掩膜
- `circlef_rcoord(d)` — 获取一定直径圆表面的相对中心坐标
- `period_clip(bound_size, coords)`
- `bound_clip(bound_size, coords)`
- `Voxelizer(self, max_r=20, ndim=3)` — 体素化器,为体素坐标种子进行体素化
- `Voxelizer.set_ndim(self, ndim)`
- `Voxelizer.max_r(self)`
- `Voxelizer.set_rcod_tab(self, max_r, ndim=3)` — 设置当前半径相对坐标表
- `Voxelizer.check_rads(self, rads, cnt)` — 检查并处理即将渲染的半径
- `Voxelizer.check_centers(self, centers)`
- `Voxelizer.check_marks(self, marks, cnt)`
- `Voxelizer.discrete_check(self, centers, rads, marks)` — 统一体素化函数,按照self.bound_funcs进行周期性边界处理
- `Voxelizer.discrete_surface(self, vol, centers, rads, marks, periods)` — 球表面离散,输入均为体素坐标
- `Voxelizer.discrete_volume(self, vol, centers, rads, marks, periods)` — 球体积离散,输入均为体素坐标
- `Voxelizer.discrete_check_axis(self, centers, rads, axis, marks)` — 统一体素化函数,按照self.bound_funcs进行周期性边界处理
- `Voxelizer.discrete_volume_axis(self, vol, centers, rads, marks, axis, periods)`

### [morqex.microstructure.core.voxel_fields](../src/morqex/microstructure/core/voxel_fields.py)

- `dist_trans(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True, measure_func=distance_transform_edt)`
- `fedt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True, parallel=n_para)`
- `invfedt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True, parallel=n_para)`
- `edt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True)`
- `cdt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True)`
- `ndt(obj_vol, *args, **kwargs)`
- `invdt(obj_vol, *args, **kwargs)`
- `hex_id(obj)`
- `auto_vol(func)`
- `DataVol(self, vol: np.ndarray=np.array([]))` — 数据数组基类
- `DataVol.ones(cls, shape)`
- `DataVol.zeros(cls, shape)`
- `DataVol.copy(self)`
- `DataVol.equals(self, value)`
- `DataVol.nequals(self, value)`
- `DataVol.ndim(self)`
- `DataVol.query(self, loc: np.ndarray)` — 相对坐标->体素坐标 超出索引范围,值置为-1
- `DataVol.itp_query(self, loc: np.ndarray)` — 插值查询 坐标0.5表示体素中心
- `DataVol.count_vol(self, delta=1.0)`
- `DataVol.concatenate(self, vol: DataVol, axis: int)`
- `DataVol.max(self)`
- `DataVol.min(self)`
- `DataVol.sum(self)`
- `DataVol.is_uni(self)`
- `DataVol.is_zero(self)`
- `DataVol.is_pos(self)`
- `DataVol.is_norm(self)`
- `DataVol.shape(self)`
- `DataVol.vxl_len(self)`
- `DataVol.vol_len(self)`
- `DataVol.cls_id_str(self)`
- `DataVol.cls_str(self)`
- `DataVol.id_str(self)`
- `DataVol.reshape(self, *args)`
- `DataVol.rescale(self, scale)` — 保持自身类型 缩放指定倍数
- `DataVol.resize(self, shape)` — 插值重新采样进行放缩
- `DataVol.trans_resize(self, shape)`
- `DataVol.swapaxes(self, axis0: int, axis1: int)`
- `DataVol.smooth(self, sigma=1.2)`
- `DataVol.median_smooth(self)` — 中值平滑 保持自身类型
- `DataVol.modal_smooth(self, depth=1)` — 众数平滑 uint8类型
- `DataVol.open_smooth(self, depth=1)` — 开运算+众数平滑 uint8类型
- `DataVol.residual(self, sigma=1.0)`
- `DataVol.scaled_residual(self, scale, sigma=1.0)` — 层级残差
- `DataVol.scaled_smooth(self, scale, sigma=1.0)` — 层级平滑
- `DataVol.scaled_median(self, scale)`
- `DataVol.n_level(self)`
- `DataVol.multi_residual(self, sigma=1.0, n_level=5)` — 多级残差 返回有相同尺度的列表 更大层级应视作均匀背景
- `DataVol.multi_smooth(self, sigma=1.0, n_level=5)` — 多级残差 返回有相同尺度的列表 更大层级应视作均匀背景
- `DataVol.from_img(cls, fname, cuts=(slice(None), slice(None), slice(None)), process_func=lambda img: img)`
- `DataVol.threshold(self, ths)` — 阈值化与处理
- `ObjVol()` — 单相数组 0视作没有物体
- `ObjVol.a_shape(self, n_voxel=1.0, measure=fedt, pad=True)` — 获取rolling_ball算法计算的外边界
- `ObjVol.inv_a_shape(self, n_voxel=1.0, measure=fedt, pad=True)`
- `ObjVol.inv(self)` — 属性取反 不维持之前的属性
- `ObjVol.attrs(self)` — 属性表 0表示空位
- `ObjVol.attr_map(self, map_arr: np.ndarray)`
- `ObjVol.is_uni(self)`
- `ObjVol.is_zero(self)`
- `ObjVol.open(self, depth=1.0, measure=cdt)` — 开运算 先腐蚀后膨胀 移除桥
- `ObjVol.close(self, depth=1.0, measure=cdt)` — 闭运算 先膨胀后腐蚀 移除孔
- `ObjVol.erosion(self, depth=1.0, measure=cdt, connect=1)`
- `ObjVol.dilation(self, depth=1.0, measure=cdt, connect=1)`
- `ObjVol.rescale(self, scale: float)` — 均匀插值放缩
- `ObjVol.resize(self, shape)` — 插值重新采样进行放缩
- `ObjVol.vol_frac(self)`
- `ObjVol.surface(self)` — 外表面提取
- `ObjVol.out_surface(self)` — 外表面提取
- `ObjVol.spec_surf_area(self, voxel_len=1.0)` — 比表面积 voxel_len给出网格边长
- `ObjVol.surf_area(self, voxel_len=1.0)` — 比表面积 voxel_len给出网格边长
- `ObjVol.fractal_ndim(self)` — 分型维数计算
- `ObjVol.tortuosity(self, start_vol: ObjVol, weight_vol=DataVol(np.array([])), mode='avg', max_prg=1000)` — 计算带有局部距离权重的空间曲折度分布,输入物体数组,指定传播轴和起始位置.
- `ObjVol.label_vol(self, connect=1)` — 按照4邻接标记域 对角相连不视作相连. color_vol则默认按照8邻接计算邻接关系图, 避免对角相连相视作不相连.
- `ObjVol.segment(self, min_dist=3)` — 有属性部分从1开始的分割
- `ObjVol.seg_center(self, min_dist=3)`
- `ObjVol.seg_erosion(self, depth=1.0, measure=fedt)`
- `ObjVol.seeds_shape(self, depth=1.0, measure=fedt)`
- `PhaseVol()` — 多相数组 加法不叠加(覆盖) 0视作没有相
- `PhaseVol.open(self, depth=1.0, measure=cdt)` — 开运算 先腐蚀后膨胀 移除桥
- `PhaseVol.close(self, depth=1.0, measure=cdt)` — 闭运算 先膨胀后腐蚀 移除孔 多属性运算累积
- `PhaseVol.surface(self)`
- `PhaseVol.surf_area(self)`
- `PhaseVol.spec_surf_area(self)`
- `PhaseVol.intf_area(self, a0, a1)`
- `PhaseVol.spec_intf_area(self, a0, a1)`
- `PhaseVol.segment(self, min_dist=1)` — 分水岭分割 LaggureVoro覆盖 未分割区域标记0 分割区域序号1~n
- `PhaseVol.seg_center(self, min_dist=1)` — 分水岭分割 LaggureVoro覆盖的中心点 顺序与segment对应
- `PhaseVol.seg_erosion(self, depth=1.0, measure=cdt)`
- `PhaseVol.rad_dist(self)`
- `PhaseVol.a_shape(self, nvxls=1.0, measure=fedt, pad=True)` — 为属性体素网格计算外边界 n_vxls输入除0位置之外的修正体素
- `PhaseVol.inv_a_shape(self, nvxls=1.0, measure=fedt, pad=True)`
- `PhaseVol.is_uni(self)`
- `PhaseVol.is_zero(self)`
- `PhaseVol.vol_frac(self)`
- `PhaseVol.residual(self, sigma=1.0)` — 对属性0进行同样的计算
- `PhaseVol.multi_residual(self, sigma=1.0, n_level=5)` — 不在自身的rescale上计算层级残差 而是在自身obj上计算层级残差后求和
- `PhaseVol.rescale(self, scale: float)` — 均匀插值放缩
- `PhaseVol.erosion(self, depth=1, measure=cdt, connect=1)`
- `PhaseVol.dilation(self, depth=1, measure=cdt, connect=1)` — 多相膨胀 存在冲突时 属性值大的覆盖属性值小的
- `PhaseVol.resize(self, shape)`
- `PhaseVol.watershed(self)` — 体素voronoi变换
- `PhaseVol.query_attr(self, loc, sigma=1.2, smooth=True)` — 点的属性插值,选取多相内最权重大属性作为点属性
- `PhaseVol.connect_graph(self, connect=2)` — 不同属性间的连通关系
- `PhaseVol.color_vol(self)` — 图着色
- `PhaseVol.from_hsdat(self, shape, head_rows=1)` — shape对应 HsDat(x,y,z) -> PhaseVol(z,y,x)
- `remove_dul_points(vertices)` — 移除重复点,输入整数点坐标
- `color_nodes(graph)` — 图着色贪婪算法
- `DistVol()`
- `ProbVol()` — 概率场,不允许小于0
- `ProbVol.normalize(self)` — 归一化,检查是否为0概率场,尽量少调用
- `ProbVol.sample(self, cnt: int)` — 对概率场采样,返回点在域中的坐标
- `vol_pos(list_pos, vol_shape)` — 将列表坐标转化为体素坐标
- `list_pos(vol_pos, vol_shape)` — 将体素坐标转化为列表坐标
- `move_axes(arr)` — 向量维度轮换 最外维度轮换到最内部
- `s2_point_correlation(pnt1_vol: ObjVol, pnt2_vol: ObjVol, ndim, n_sample=100000, periods=[False, False, False], nbin=400)`
- `s2_pntcor_data(pnt1_vol: ObjVol, pnt2_vol: ObjVol, n_sample=100000, periods=[False, False, False])`
- `c2_point_correlation(pnt1_vol: ObjVol, pnt2_vol: ObjVol, n_sample=100000, nbin=100)`
- `calculate_3d_correlation_lengths(voxel_grid)` — 计算三维体素网格在x、y、z三个方向上的相关长度
- `calculate_3d_autocorrelation(voxel_grid)` — 计算三维体素网格的自相关函数和相关长度
- `find_correlation_length(acf_1d, center_index)` — 从一维自相关函数中找到相关长度

### [morqex.microstructure.design](../src/morqex/microstructure/design.py)

- `random_field(shape=(64, 64, 64), fractions=(0.3, 0.7), length_scale=(3.0, 3.0, 3.0), seed=0, spacing=(1.0, 1.0, 1.0))`

### [morqex.microstructure.full_volume](../src/morqex/microstructure/full_volume.py)

- `periodic_components(mask)`
- `periodic_distance(mask, *, halo=32, slab=24)` — Exact EDT when the returned interior distances stay below halo.
- `prepare_full_volume(source, destination, *, halo=32)`
- `positive_basis(features, *, orders=(2, 2, 2), xp=np)` — Tensor Bernstein partition of unity; eight nodes for (2,2,2).

### [morqex.microstructure.generate.conditioning](../src/morqex/microstructure/generate/conditioning.py)

- `composition_potential(scores, fractions, temperature=0.1, iterations=18, max_points=65536)` — Return biased scores and auditable soft/hard composition diagnostics.

### [morqex.microstructure.generate.config](../src/morqex/microstructure/generate/config.py)

- `GANConfig()`
- `GANConfig.legacy_dict(self)`
- `train_gan(config)`
- `sample_gan(config, checkpoint=True, return_raw=False, spacing=(1.0, 1.0, 1.0))`

### [morqex.microstructure.generate.constraints](../src/morqex/microstructure/generate/constraints.py)

- `surface_coating(volume, substrate, coating, matrix, coating_fraction, seed=0, roughness=0.15)` — Project a binder phase onto the exterior of an active-material skeleton.

### [morqex.microstructure.generate.hybrid](../src/morqex/microstructure/generate/hybrid.py)

- `Generator(self, phases, noise_channels=16, width=64, residual=False, tri_reference=False, normalize_reference=False, zero_reference=False)`
- `Generator.forward(self, z, reference, fractions)`
- `Discriminator(self, phases)`
- `Discriminator.forward(self, x, fractions, axis)`
- `HybridConfig()`
- `distill(teacher_checkpoint, paths, run_dir, config=None)`
- `sample_student(checkpoint, fractions=None, seed=42, device='cuda', size=None, exact_fractions=False, paths=None, composition_guidance=False, composition_temperature=0.05)`

### [morqex.microstructure.generate.reconstruction](../src/morqex/microstructure/generate/reconstruction.py)

- `Reconstructor()`
- `Reconstructor.load(cls, manifest, workspace=None)`
- `Reconstructor.generate(self, fractions=None, size=64, seed=42, device='cuda', spacing=(1.0, 1.0, 1.0), **overrides)`

### [morqex.microstructure.generate.selection](../src/morqex/microstructure/generate/selection.py)

- `Candidate()`
- `rve_references(ct, size=64, partition='validation', count=4, seed=42, observation_planes=None)` — Disjoint z halves for validation/test; same-size statistical comparison.
- `descriptors(volume)`
- `error(reference, generated)`
- `benchmark(candidates, ct, output, seeds=(0, 1, 2), partition='validation', size=64, baseline=None, observation_planes=None)` — Preset score weights; reports each metric and avoids hiding failures in score.

### [morqex.microstructure.generate.spatial](../src/morqex/microstructure/generate/spatial.py)

- `VoxelDiffusionConfig()`
- `Residual(self, cin, cout, padding_mode='circular')`
- `Residual.forward(self, x, embedding)`
- `Denoiser(self, cfg)`
- `Denoiser.forward(self, x, t, z, temperature=1.0)`
- `AxialAttention(self, channels)` — Global information exchange along three axes at the coarse grid only.
- `AxialAttention.forward(self, x)`
- `SliceCritic(self, cfg)`
- `SliceCritic.forward(self, current, previous, t, axis)`
- `Chain(self, steps, device)`
- `Chain.real_pair(self, clean, t)`
- `Chain.reverse(self, clean, previous, t)`
- `fit_volume(images, run_dir, config=None, resume=None)` — Train a new 3D diffusion model from one/three 2D phase-label images.
- `draw_volume(checkpoint, size=256, seed=42, device='cuda', spacing=(1.0, 1.0, 1.0))` — Native-size 3D diffusion chain; no numerical property input/postprojection.

### [morqex.microstructure.generate.v0.adversarial](../src/morqex/microstructure/generate/v0/adversarial.py)

- `Trainer(config_path)`

### [morqex.microstructure.generate.v0.evaluation_images](../src/morqex/microstructure/generate/v0/evaluation_images.py)

- `batch_crop_two_slice_folders(input_dir1: str, input_dir2: str, output_dir1: str, output_dir2: str, crop_size: Optional[Tuple[int, int]]=None, use_smaller_common_size: bool=True, max_slices: Optional[int]=None, crop_mode: str='center', allowed_exts: Optional[Tuple[str, ...]]=None, sort_key=None, keep_original_names: bool=True, output_format: Optional[str]=None, tif_compression: Optional[str]=None, verbose: bool=True)` — 对两个文件夹中的切片图像进行批量裁剪，并输出到两个新文件夹。

### [morqex.microstructure.generate.v0.fields](../src/morqex/microstructure/generate/v0/fields.py)

- `dist_trans(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True, measure_func=distance_transform_edt)`
- `fedt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True, parallel=n_para)`
- `invfedt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True, parallel=n_para)`
- `edt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True)`
- `cdt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True)`
- `ndt(obj_vol, *args, **kwargs)`
- `invdt(obj_vol, *args, **kwargs)`
- `hex_id(obj)`
- `auto_vol(func)`
- `DataVol(self, vol: np.ndarray=np.array([]))` — 数据数组基类
- `DataVol.ones(cls, shape)`
- `DataVol.zeros(cls, shape)`
- `DataVol.copy(self)`
- `DataVol.equals(self, value)`
- `DataVol.nequals(self, value)`
- `DataVol.ndim(self)`
- `DataVol.query(self, loc: np.ndarray)` — 相对坐标->体素坐标 超出索引范围,值置为-1
- `DataVol.itp_query(self, loc: np.ndarray)` — 插值查询 坐标0.5表示体素中心
- `DataVol.count_vol(self, delta=1.0)`
- `DataVol.concatenate(self, vol: DataVol, axis: int)`
- `DataVol.max(self)`
- `DataVol.min(self)`
- `DataVol.sum(self)`
- `DataVol.is_uni(self)`
- `DataVol.is_zero(self)`
- `DataVol.is_pos(self)`
- `DataVol.is_norm(self)`
- `DataVol.shape(self)`
- `DataVol.vxl_len(self)`
- `DataVol.vol_len(self)`
- `DataVol.cls_id_str(self)`
- `DataVol.cls_str(self)`
- `DataVol.id_str(self)`
- `DataVol.reshape(self, *args)`
- `DataVol.rescale(self, scale)` — 保持自身类型 缩放指定倍数
- `DataVol.resize(self, shape)` — 插值重新采样进行放缩
- `DataVol.trans_resize(self, shape)`
- `DataVol.swapaxes(self, axis0: int, axis1: int)`
- `DataVol.smooth(self, sigma=1.2)`
- `DataVol.median_smooth(self)` — 中值平滑 保持自身类型
- `DataVol.modal_smooth(self, depth=1)` — 众数平滑 uint8类型
- `DataVol.open_smooth(self, depth=1)` — 开运算+众数平滑 uint8类型
- `DataVol.residual(self, sigma=1.0)`
- `DataVol.scaled_residual(self, scale, sigma=1.0)` — 层级残差
- `DataVol.scaled_smooth(self, scale, sigma=1.0)` — 层级平滑
- `DataVol.scaled_median(self, scale)`
- `DataVol.n_level(self)`
- `DataVol.multi_residual(self, sigma=1.0, n_level=5)` — 多级残差 返回有相同尺度的列表 更大层级应视作均匀背景
- `DataVol.multi_smooth(self, sigma=1.0, n_level=5)` — 多级残差 返回有相同尺度的列表 更大层级应视作均匀背景
- `DataVol.from_img(cls, fname, cuts=(slice(None), slice(None), slice(None)), process_func=lambda img: img)`
- `DataVol.threshold(self, ths)` — 阈值化与处理
- `ObjVol()` — 单相数组 0视作没有物体
- `ObjVol.a_shape(self, n_voxel=1.0, measure=fedt, pad=True)` — 获取rolling_ball算法计算的外边界
- `ObjVol.inv_a_shape(self, n_voxel=1.0, measure=fedt, pad=True)`
- `ObjVol.inv(self)` — 属性取反 不维持之前的属性
- `ObjVol.attrs(self)` — 属性表 0表示空位
- `ObjVol.attr_map(self, map_arr: np.ndarray)`
- `ObjVol.is_uni(self)`
- `ObjVol.is_zero(self)`
- `ObjVol.open(self, depth=1.0, measure=cdt)` — 开运算 先腐蚀后膨胀 移除桥
- `ObjVol.close(self, depth=1.0, measure=cdt)` — 闭运算 先膨胀后腐蚀 移除孔
- `ObjVol.erosion(self, depth=1.0, measure=cdt, connect=1)`
- `ObjVol.dilation(self, depth=1.0, measure=cdt, connect=1)`
- `ObjVol.rescale(self, scale: float)` — 均匀插值放缩
- `ObjVol.resize(self, shape)` — 插值重新采样进行放缩
- `ObjVol.vol_frac(self)`
- `ObjVol.surface(self)` — 外表面提取
- `ObjVol.out_surface(self)` — 外表面提取
- `ObjVol.spec_surf_area(self, voxel_len=1.0)` — 比表面积 voxel_len给出网格边长
- `ObjVol.surf_area(self, voxel_len=1.0)` — 比表面积 voxel_len给出网格边长
- `ObjVol.fractal_ndim(self)` — 分型维数计算
- `ObjVol.tortuosity(self, start_vol: ObjVol, weight_vol=DataVol(np.array([])), mode='avg', max_prg=1000)` — 计算带有局部距离权重的空间曲折度分布,输入物体数组,指定传播轴和起始位置.
- `ObjVol.label_vol(self, connect=1)` — 按照4邻接标记域 对角相连不视作相连. color_vol则默认按照8邻接计算邻接关系图, 避免对角相连相视作不相连.
- `ObjVol.segment(self, min_dist=3)` — 有属性部分从1开始的分割
- `ObjVol.seg_center(self, min_dist=3)`
- `ObjVol.seg_erosion(self, depth=1.0, measure=fedt)`
- `ObjVol.seeds_shape(self, depth=1.0, measure=fedt)`
- `PhaseVol()` — 多相数组 加法不叠加(覆盖) 0视作没有相
- `PhaseVol.open(self, depth=1.0, measure=cdt)` — 开运算 先腐蚀后膨胀 移除桥
- `PhaseVol.close(self, depth=1.0, measure=cdt)` — 闭运算 先膨胀后腐蚀 移除孔 多属性运算累积
- `PhaseVol.surface(self)`
- `PhaseVol.surf_area(self)`
- `PhaseVol.spec_surf_area(self)`
- `PhaseVol.intf_area(self, a0, a1)`
- `PhaseVol.spec_intf_area(self, a0, a1)`
- `PhaseVol.segment(self, min_dist=1)` — 分水岭分割 LaggureVoro覆盖 未分割区域标记0 分割区域序号1~n
- `PhaseVol.seg_center(self, min_dist=1)` — 分水岭分割 LaggureVoro覆盖的中心点 顺序与segment对应
- `PhaseVol.seg_erosion(self, depth=1.0, measure=cdt)`
- `PhaseVol.rad_dist(self)`
- `PhaseVol.a_shape(self, nvxls=1.0, measure=fedt, pad=True)` — 为属性体素网格计算外边界 n_vxls输入除0位置之外的修正体素
- `PhaseVol.inv_a_shape(self, nvxls=1.0, measure=fedt, pad=True)`
- `PhaseVol.is_uni(self)`
- `PhaseVol.is_zero(self)`
- `PhaseVol.vol_frac(self)`
- `PhaseVol.residual(self, sigma=1.0)` — 对属性0进行同样的计算
- `PhaseVol.multi_residual(self, sigma=1.0, n_level=5)` — 不在自身的rescale上计算层级残差 而是在自身obj上计算层级残差后求和
- `PhaseVol.rescale(self, scale: float)` — 均匀插值放缩
- `PhaseVol.erosion(self, depth=1, measure=cdt, connect=1)`
- `PhaseVol.dilation(self, depth=1, measure=cdt, connect=1)` — 多相膨胀 存在冲突时 属性值大的覆盖属性值小的
- `PhaseVol.resize(self, shape)`
- `PhaseVol.watershed(self)` — 体素voronoi变换
- `PhaseVol.query_attr(self, loc, sigma=1.2, smooth=True)` — 点的属性插值,选取多相内最权重大属性作为点属性
- `PhaseVol.connect_graph(self, connect=2)` — 不同属性间的连通关系
- `PhaseVol.color_vol(self)` — 图着色
- `PhaseVol.from_hsdat(self, shape, head_rows=1)` — shape对应 HsDat(x,y,z) -> PhaseVol(z,y,x)
- `remove_dul_points(vertices)` — 移除重复点,输入整数点坐标
- `color_nodes(graph)` — 图着色贪婪算法
- `DistVol()`
- `ProbVol()` — 概率场,不允许小于0
- `ProbVol.normalize(self)` — 归一化,检查是否为0概率场,尽量少调用
- `ProbVol.sample(self, cnt: int)` — 对概率场采样,返回点在域中的坐标
- `vol_pos(list_pos, vol_shape)` — 将列表坐标转化为体素坐标
- `list_pos(vol_pos, vol_shape)` — 将体素坐标转化为列表坐标
- `move_axes(arr)` — 向量维度轮换 最外维度轮换到最内部
- `s2_point_correlation(pnt1_vol: ObjVol, pnt2_vol: ObjVol, ndim, n_sample=100000, periods=[False, False, False], nbin=400)`
- `s2_pntcor_data(pnt1_vol: ObjVol, pnt2_vol: ObjVol, n_sample=100000, periods=[False, False, False])`
- `c2_point_correlation(pnt1_vol: ObjVol, pnt2_vol: ObjVol, n_sample=100000, nbin=100)`
- `calculate_3d_correlation_lengths(voxel_grid)` — 计算三维体素网格在x、y、z三个方向上的相关长度
- `calculate_3d_autocorrelation(voxel_grid)` — 计算三维体素网格的自相关函数和相关长度
- `find_correlation_length(acf_1d, center_index)` — 从一维自相关函数中找到相关长度

### [morqex.microstructure.generate.v0.inference](../src/morqex/microstructure/generate/v0/inference.py)

- `generator(config_path)`
- `generator_checkpoint(config_path)`
- `save_3d_tif_to_2d_slices(input_tif_path: str, output_folder: str, slice_prefix: str='slice', axis: str='x', check_dim_order: bool=False, enforce_same_size: bool=False, size_mode: str='crop', pad_value: int=0)` — 将三维TIF文件沿指定方向分割为二维切片并保存为 int8 tif（保证直观看效果一致）
- `save_3d_tif_to_2d_slices_denoise(input_tif_path: str, output_folder: str, slice_prefix: str='slice', check_dim_order: bool=False)` — 将三维TIF文件分割为二维切片并保存到指定文件夹
- `denoise_3D_tif(input_tif_path: str, output_folder: 'output_2', slice_prefix: str='slice', check_dim_order: bool=False, save=False)` — 将三维TIF文件分割为二维切片并保存到指定文件夹
- `smooth_voxel_domains_iterative(input_data, kernel_size=3, iterations=5, save_path=None)` — 对三维体素进行迭代式区域平滑。
- `remove_small_islands(data, min_size=27)` — 强力模式：直接移除小于 min_size (例如27个像素) 的所有连通域。

### [morqex.microstructure.generate.v0.operations](../src/morqex/microstructure/generate/v0/operations.py)

- `load_config(config_path)` — 读取 YAML 配置文件并返回参数 dict，
- `mkdr(proj, proj_dir, Training)` — When training, creates a new project directory or overwrites an existing directory according to user input. When testing, returns the full project path
- `weights_init(m)` — 初始化训练权重
- `calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda, nc)` — 计算一批真实数据和生成数据的梯度惩罚
- `show_training_time(steps, time, start, i, epoch_num, num_epochs)` — 根据已用时间和训练轮数估算剩余时间
- `plot_transfer(img, img_type)` — 将一张热成像图像转换回灰度图
- `training_plotter(img, num_axis, img_type, img_pth)` — 创建一个包含 3*num_axis 子图的图，显示训练过程中沿三个轴的切片
- `dinamic_plot(data, labels, pth, name)` — 适用于各种不同图表的简单作图工具，进行训练过程的动态绘图
- `generate_img(pth, img_type, netG, nz=64, lf=4, periodic=False)` — 为已训练或正在训练的生成器保存测试卷
- `generate_img_checkpoint(pth, img_type, netG, nz=64, lf=4, periodic=False, device='cuda')` — 为已训练或正在训练的生成器保存测试卷
- `generate_img_cond(pth, img_type, netG, phi, nz=64, lf=4, periodic=False)` — 为已训练或正在训练的生成器保存测试卷（条件：孔隙率 phi）

### [morqex.microstructure.generate.v0.patches](../src/morqex/microstructure/generate/v0/patches.py)

- `CreateTensorDataset(data_path, data_type, image_size, scale)` — 从训练微观结构中随机生成张量数据集

### [morqex.microstructure.generate.v0.restoration](../src/morqex/microstructure/generate/v0/restoration.py)

- `advanced_denoising_pipeline(img)`
- `sota_detail_preserving_pipeline(img)`
- `remove_small_regions_strict(img_labels, min_size=4)` — 严格的拓扑过滤器：
- `final_advanced_pipeline(img)`
- `snap_to_closest(arr)` — 输入: 任意维度的 numpy 数组或列表

### [morqex.microstructure.generate.v0.settings](../src/morqex/microstructure/generate/v0/settings.py)

- `ConfigDict(self, *args, **kwargs)` — 一个辅助类，让字典可以通过点符号访问属性 (config.project.name)
- `ConfigProcessor(self, config_path)`
- `ConfigProcessor.reload(self)` — 重新加载配置文件（用于在运行时修改yaml后刷新）
- `ConfigProcessor.show_all(self)` — 格式化输出所有当前变量
- `ConfigProcessor.check_consistency(self)` — 根据 YAML 中的注释逻辑检查参数一致性

### [morqex.microstructure.generate.v0.stacks](../src/morqex/microstructure/generate/v0/stacks.py)

- `save_3d_tif_to_2d_slices(input_tif_path: str, output_folder: str, slice_prefix: str='slice', check_dim_order: bool=False)` — 将三维TIF文件分割为二维切片并保存到指定文件夹

### [morqex.microstructure.generate.v0.synthesis](../src/morqex/microstructure/generate/v0/synthesis.py)

- `AdaptiveUpsample3D(self, in_channels, out_channels, upscale_factor, kernel_size=3)` — 3D自适应上采样模块（基于光流/特征预测的思想）
- `AdaptiveUpsample3D.forward(self, x)`
- `PixelShuffle3D(self, upscale_factor)` — 3D 版 Pixel Shuffle (Voxel Shuffle)。
- `PixelShuffle3D.forward(self, x)`
- `SELayer3D(self, channel, reduction=16)`
- `SELayer3D.forward(self, x)`
- `upsample_net(pth, Training, imtype, d_kernel, d_stride, d_feature_channels, d_padding, g_scale, g_feature_channels)`
- `upsample_net_conv(pth, Training, imtype, d_kernel, d_stride, d_feature_channels, d_padding, g_scale, g_feature_channels)` — complex_net

### [morqex.microstructure.generate.v0.unavailable_conditioned](../src/morqex/microstructure/generate/v0/unavailable_conditioned.py)

- `generator(config_path, phi)`
- `save_3d_tif_to_2d_slices(input_tif_path: str, output_folder: str, slice_prefix: str='slice', check_dim_order: bool=False)` — 将三维TIF文件分割为二维切片并保存到指定文件夹
- `save_3d_tif_to_2d_slices_denoise(input_tif_path: str, output_folder: str, slice_prefix: str='slice', check_dim_order: bool=False)` — 将三维TIF文件分割为二维切片并保存到指定文件夹
- `denoise_3D_tif(input_tif_path: str, output_folder: 'output_2', slice_prefix: str='slice', check_dim_order: bool=False, save=False)` — 将三维TIF文件分割为二维切片并保存到指定文件夹
- `smooth_voxel_domains_iterative(input_data, kernel_size=3, iterations=5, save_path=None)` — 对三维体素进行迭代式区域平滑。
- `remove_small_islands(data, min_size=27)` — 强力模式：直接移除小于 min_size (例如27个像素) 的所有连通域。

### [morqex.microstructure.generate.v1](../src/morqex/microstructure/generate/v1.py)

- `DiffusionConfig()`
- `train(paths, run_dir, config=None, resume=None)` — Crops on device, BF16 AMP, EMA, deadline, full optimizer/RNG resumption.
- `sample(checkpoint, paths=None, fractions=None, size=None, steps=50, guidance=1.5, seed=42, device='cuda', slice_batch=16, spacing=(1.0, 1.0, 1.0), exact_fractions=False, prior=None, strength=1.0, overlap=16, composition_guidance=False, composition_temperature=0.05, composition_start=0.35)` — One/three 2D conditions -> 3D labels. Optional GAN volume as diffusion prior.
- `tune_batch_size(config, target_fraction=0.88, maximum=512)` — Measure actual forward/backward peak; leave headroom for optimizer/EMA.

### [morqex.microstructure.helper.desktop](../src/morqex/microstructure/helper/desktop.py)

- `float_repr(x)`
- `custom_display_hook(value)`
- `custom_format(obj)`
- `TKUI()`
- `RedirectText(self, text_widget)`
- `RedirectText.write(self, output)`
- `RedirectText.flush(self)`
- `RedirectBox(self, master, title='massage')`
- `RedirectBox.write(self, output)`
- `RedirectBox.flush(self)`
- `set_scaling(scale)` — 动态调整 UI 缩放 + 窗口大小
- `exec_in_main_thread(func, *args, **kwargs)` — 在 Tk 主线程执行（适用于 UI 更新）
- `Context(self, env=None)`
- `Context.init_state(self)`
- `Context.get_name(self, obj)`
- `Context.capture_obj(self)` — 捕获当前对话中的UI对象
- `Context.relist_obj(self, event='<Enter>')`
- `Context.sort_obj(self)`
- `Context.capture_file(self)` — 捕获当前的外部文件
- `Context.relist_file(self, event='<Enter>')`
- `Context.view_select(self, event='<Enter>')`
- `Context.del_select(self, event='<Enter>')`
- `Context.copy_select(self, event='<Enter>')`
- `Context.rename_select(self, event='<Enter>')`
- `Context.save_select(self, event='<Enter')`
- `Context.load_select(self, event='<Enter')`
- `Context.get_select_obj_names(self)`
- `Context.get_select_file_names(self)`
- `Context.view_obj(self, names)` — 显示对象控件
- `Context.exec_scr_cmd(self, event='<Enter>')` — 执行脚本框
- `Context.save_scr(self, event='<Enter>')` — 保存脚本
- `Context.load_scr(self, event='<Enter>')` — 导入脚本
- `Context.exec_cmd(self, event='<Enter>', cmd='')` — 外层接口：在新线程中执行命令，避免阻塞 UI
- `Context.new_proj(self)`
- `Context.save_proj_path(self, out_path)`
- `Context.save_as_proj(self)`
- `Context.save_proj(self)`
- `Context.load_proj(self)`
- `tkinit_dec(initfunc, vars={}, pltfuncs=[], pltsize=(3, 3), *args, **kwargs)` — TKinter面板
- `tkregister(self, master)`
- `tklayout(self, event='<Enter>', side=tk.TOP)`
- `tklayout_left(self, event='<Enter>')`
- `tkrepr_dec(reprfunc)`
- `tkunrepr(self)`
- `tkstr_dec(strfunc)` — 声明复合变量
- `tkredraw(self, event='<Enter>')`
- `tkprocess(self, end='')`
- `tkstart(self, n_bat)`
- `tkinc(self)`
- `tkend(self)`
- `plt_vol(ax, vol, cb=False, legend={})` — 绘制分布密度 支持多维分布
- `plt_legend(ax, vol, legend: dict={})`
- `plt_view_vol(obj, ax, reg)` — 绘制相 采用缓存
- `plt_dist(dist, ax, reg=None, nbin=1000, smth=1)` — 绘制分布密度 支持多维分布
- `plt_compdist(compdist, ax, reg=None, nbin=600, smth=1)`
- `TKUI()` — UI封装基类
- `TKUI.precs(self)`
- `TKUI.succs(self)`
- `TKUI.update(self)`
- `TKUI.check_update(self)`
- `TKUI.modified(self)`
- `exec_str_ret(vstr, env=None)` — 执行字符串 返回结果
- `dist_init(self, *params)`
- `dist_str(self)`
- `TKDistFunc()`
- `TKDistFunc.dec_key(self)`
- `TKDistFunc.run_key(self)`
- `TKDistFunc.stat_model(self)`
- `TKDistFunc.stat_model(self, value)`
- `TKDistFunc.model(self)`
- `init_compdist(self, dfs=[TKDistFunc(-1.0, 2.0), TKDistFunc(-1.0, 2.0), TKDistFunc(-1.0, 2.0)])`
- `TKCompDist()`
- `TKCompDist.precs(self)`
- `TKCompDist.df0(self)`
- `TKCompDist.df0(self, value)`
- `TKCompDist.df1(self)`
- `TKCompDist.df1(self, value)`
- `TKCompDist.df2(self)`
- `TKCompDist.df2(self, value)`
- `TKCompDist.dfs(self)`
- `TKCompDist.dfs(self, value)`
- `TKRegion()`
- `TKRegion2D()`
- `phs_init(self, r_dist=TKDistFunc(), v_frac: float=1.0, mark: int=1, name='')`
- `TKPhase()`
- `TKPhase.r_dist(self)`
- `TKPhase.r_dist(self, value)`
- `TKPhase.view_vol(self, reg)`
- `TKPhase.precs(self)`
- `aniphs_init(self, r_dist=TKDistFunc(), ab_dist=TKDistFunc(1.0, 0.0), ort_dist=TKCompDist([TKDistFunc(-1.0, 2.0), TKDistFunc(-1.0, 2.0), TKDistFunc(-1.0, 2.0)]), v_frac: float=1.0, mark: int=1, name='')`
- `TKAniPhase()`
- `TKAniPhase.r_dist(self)`
- `TKAniPhase.r_dist(self, value)`
- `TKAniPhase.ab_dist(self)`
- `TKAniPhase.ab_dist(self, value)`
- `TKAniPhase.ort_dist(self)`
- `TKAniPhase.ort_dist(self, value)`
- `TKAniPhase.view_vol(self, reg)`
- `TKAniPhase.precs(self)`
- `TKCompPhase()`
- `TKCompPhase.phss(self)`
- `TKCompPhase.phss(self, value)`
- `TKCompPhase.view_vol(self, reg)`
- `TKAlphaShape()`
- `TKAlphaShape.a_shape(self, *args, **kwargs)`
- `TKAlphaShape.shape_model(self)`
- `TKAlphaShape.shape_model(self, value)`
- `view_vol(self, reg)`
- `pobj_init(self, phs=None, ashp=None)`
- `TKPhasePnt()`
- `TKPhasePnt.precs(self)`
- `TKPhasePnt.phs(self)`
- `TKPhasePnt.phs(self, value)`
- `TKPhasePnt.ashp(self)`
- `TKPhasePnt.ashp(self, value)`
- `TKFillBound()`
- `TKCylinder()`
- `TKFileImg()`
- `TKFileImg.regist(self, master)`
- `TKFileImg.layout(self)`
- `TKFileImg.redraw(self, event='<Enter>')`
- `TKFileImg.fname(self)`
- `TKFileImg.fname(self, value)`
- `TKFileImg.recording(self)`
- `TKFileImg.view_vol(self, reg)`
- `imgobj_init(self, img=PhaseVol.zeros([10, 10, 10]))` — 图像物体(单相物体)初始化
- `TKImage()`
- `TKImage.regist(self, master)`
- `TKImage.layout(self)`
- `TKImage.redraw(self, event='<Enter>')`
- `TKImage.precs(self)`
- `TKImage.view_vol(self, reg)`
- `TKImage.img(self)`
- `TKImage.img(self, value)`
- `compobj_init(self, objs=[])`
- `TKCompObject()`
- `TKCompObject.regist(self, master)`
- `TKCompObject.layout(self)`
- `TKCompObject.redraw(self, event='<Enter>')`
- `TKCompObject.discreting(self)`
- `TKCompObject.dec_key(self)`
- `TKCompObject.run_key(self)`
- `TKCompObject.precs(self)`
- `TKCompObject.view_vol(self, reg)`
- `TKCompObject.objs(self)`
- `TKCompObject.objs(self, value)`
- `TKCompObject.reg(self)`
- `TKCompObject.reg(self, value)`
- `tetmesh_init(self, *args, **kwargs)`
- `TKTetMesh()` — 四面体网格
- `TKTetMesh.regist(self, master)`
- `TKTetMesh.layout(self)`
- `TKTetMesh.redraw(self, event='<Enter>')`
- `TKTetMesh.from_mesh(cls, value)`
- `TKTetMesh.writing(self)`
- `TKTetMesh.askdir(self)`
- `TKTetMesh.update(self)`
- `phsvol_init(self, *args, **kwargs)`
- `TKPhaseVol()` — 体素网格
- `TKPhaseVol.view_vol(self, reg)`
- `TKPhaseVol.from_vol(cls, value)`
- `TKPhaseVol.regist(self, master)`
- `TKPhaseVol.layout(self)`
- `TKPhaseVol.redraw(self, event='<Enter>')`
- `TKPhaseVol.meshing(self)`
- `rlt_init(self, srs=None, tgt=None, ftr_func=surdt, dns_func=None)`
- `TKRelation()`
- `TKRelation.precs(self)`
- `TKRelation.srs(self)`
- `TKRelation.srs(self, value)`
- `TKRelation.tgt(self)`
- `TKRelation.tgt(self, value)`
- `TKRelation.ftr_func(self)`
- `TKRelation.ftr_func(self, value)`
- `TKRelation.dns_func(self)`
- `TKRelation.dns_func(self, value)`
- `init_pcg(self, phss: list[TKPhase]=[], rlts: list[TKRelation]=[])`
- `TKPCG()`
- `TKPCG.dec_key(self)`
- `TKPCG.run_key(self)`
- `TKPCG.regist(self, master)`
- `TKPCG.layout(self)`
- `TKPCG.redraw(self, event='<Enter>')`
- `TKPCG.view_vol(self, reg)`
- `TKPCG.phss(self)`
- `TKPCG.phss(self, value)`
- `TKPCG.rlts(self)`
- `TKPCG.rlts(self, value)`
- `TKPCG.precs(self)`
- `TKPCG.reg(self)`
- `TKPCG.reg(self, value)`
- `TKPCG.sample_seeds(self)`
- `TKPCG.fit_seeds(self)`
- `TKPCG.fill_seeds(self)`
- `TKSeeds()`
- `TKSeeds.view_vol(self, reg)`
- `voro_init(self, seeds=None)`
- `TKVoro()`
- `TKVoro.view_vol(self, reg)`
- `TKVoro.reg(self)`
- `TKVoro.seeds(self)`
- `TKVoro.seeds(self, value)`
- `view_str(self)`
- `view_cfg_init(self, rpos=0.5, axis=2)`
- `ViewConfig()`
- `save_str(self)`
- `save_cfg_init(self, dir='proj', fmt='json')`
- `SaveConfig()`
- `SaveConfig.regist(self, master)`
- `SaveConfig.layout(self)`
- `SaveConfig.redraw(self, event='<Enter>')`
- `SaveConfig.askdir(self)`
- `view_reg_init(self, res=1.0, size=[64.0, 64.0, 64.0], periods=[False, False, False])`
- `ViewRegion()`
- `aload(cls, path_str)` — 根据id字符串加载,返回实例
- `asave(self, path_str='')` — 默认保存为类名+id字符串
- `jsonlize(obj)` — 转换为json序列对象
- `objectlize(js)` — 转换为python对象

### [morqex.microstructure.helper.persistence](../src/morqex/microstructure/helper/persistence.py)

- `Config()`
- `Config.load(self, load_cfg)`
- `SaveConfig(self)`
- `jsonlize(obj)` — 转换为json序列对象
- `jcsvlize(obj, name='-')` — 转换为json-csv对象
- `objectlize(js)` — 转换为python对象
- `auto_load(cls, path_str: str)` — 根据id字符串加载,返回实例
- `auto_save(self, path_str='')` — 默认保存为类名+id字符串
- `save_vol(self, path_str)`
- `load_vol(cls, path_str, *args)`

### [morqex.microstructure.helper.plotting](../src/morqex/microstructure/helper/plotting.py)

- `Config()`
- `Config.load(self, load_cfg)`
- `ViewConfig(self)`
- `plt_vol(vol, legend={}, cb=False, vrange=(), cmap='viridis')` — 绘制离散空间体积
- `plt_vol_sem(phases_vol, phase_scale=12, depth_scale=0.2, cmap='viridis')` — 绘制离散空间体积,渲染时重新赋予相序号
- `plt_cb()`
- `plt_legend(vol, legend: dict={}, cmap='viridis', *args, **kwargs)`
- `plt_data(data, nbin=50, density=True, weights=[], cum=False, *args, **kwargs)` — "分布数据的绘制
- `plt_dns(mats, rlt_mark, x, colors, methods, marks, styles, smth=1, loc='best', every=3)`
- `plt_bar(hs, xs, labs, clrs, loc=2, per=True, sci=False, **kwargs)` — 绘制多级条形图
- `plt_bar_stack(hs, xs, labs, clrs, loc=2, per=True)` — 绘制多级条形图
- `poly(x, y, xi, reg=1)`
- `plt_img_save(fname)`
- `adjust_pts_order(pts_2ds)` — sort rectangle points by counterclockwise
- `plt_circle2d(seeds, figure, axes, colors=None, fill=True)`
- `plt_voro2d(seeds, figure, axes, colors=None, fill=True)`
- `stereographic_projection(v)` — v: array of shape (N,3)
- `plot_pole_figure(vectors)`
- `normalize(v)`
- `cubic_ipf_sector_reduce(v)` — Map arbitrary direction to cubic IPF sector
- `plot_pole_figure_kde(vectors, grid_res=400)`
- `ipf_sector_projection(v)`
- `plot_ipf_sector(vectors, title='Cubic Inverse Pole Figure (Fan-shaped)')`
- `data_vol_plt(data_vol: DataVol)`
- `phs_vol_plt(phs_vol: ObjVol)`
- `obj_plt(obj: Object)`
- `obj_pnt_plt(obj_pnt: ObjectPnt)`
- `seeds_pnt_plt(obj_pnt: SeedsPnt)`
- `pnt_opt_plt(obj_pnt: SeedsPntOperator)`
- `dist_func_plt(df, smth=1, *args, **kwargs)` — 绘制分布密度 支持多维分布
- `seeds_plt(seeds)`
- `rnd_sp_plt(rp)`
- `hist_plt(data, nbin=40)`
- `error_plt(avg, std, alpha=0.5, x=[], label=None)`
- `curve_error(x1, y1, x2, y2)`

### [morqex.microstructure.image](../src/morqex/microstructure/image.py)

- `load_stack(directory, spacing=(1.0, 1.0, 1.0))`
- `segment(array, thresholds=None, phases=2)`
- `clean(volume, minimum_size=16)`
- `save_slices(volume, directory, axis='z', prefix='slice')`

### [morqex.microstructure.io](../src/morqex/microstructure/io.py)

- `load_legacy(path)` — Load the user's trusted PCG pickle, without installing global import aliases.
- `convert_legacy_volume(path, output, spacing=(1.0, 1.0, 1.0))`

### [morqex.microstructure.legacy_generator.network](../src/morqex/microstructure/legacy_generator/network.py)

- `ResBlock3D(self, channels)` — 3D 残差块，用于生成器
- `ResBlock3D.forward(self, x)`
- `ResBlock2D(self, channels)` — 2D 残差块，用于判别器
- `ResBlock2D.forward(self, x)`
- `Generator(self, z_dim=64, dim=64, out_channels=1)`
- `Generator.forward(self, z)`
- `Discriminator(self, dim=64, in_channels=1)`
- `Discriminator.forward(self, x)`

### [morqex.microstructure.legacy_generator.train](../src/morqex/microstructure/legacy_generator/train.py)

- `create_dummy_data_if_not_exists()` — 如果没有训练图片，生成一张模拟的多孔结构图
- `load_data()`
- `main_process()`

### [morqex.microstructure.legacy_generator.utils](../src/morqex/microstructure/legacy_generator/utils.py)

- `compute_gradient_penalty(D, real_samples, fake_samples, device)` — 计算 WGAN-GP 的梯度惩罚
- `post_process_output(generated_vol, reference_img)` — 后处理：将生成的[-1, 1]连续体积映射回参考图像的整数分布 (相分布)
- `save_checkpoint(netG, netD, epoch, path='checkpoint.pth')`

### [morqex.microstructure.mesh.elements](../src/morqex/microstructure/mesh/elements.py)

- `set_float_precision(value_arr, precision=12, zero_th=1e-32)`
- `hex_id(obj)`
- `Mesh(self, verts=np.array([]), cells=np.array([]), vert_data=np.array([]), cell_data=np.array([]))`
- `Mesh.parse_name(self, path_str='')`
- `Mesh.save_mesh(self, path_str='', on=True)`
- `Mesh.load_mesh(cls, name_str='', on=True)`
- `Mesh.read_meshio(cls, path='', name='meshio_mesh', fmt='vtk', on=True)`
- `Mesh.read_gmsh(cls, path='', name='meshio_mesh', fmt='vtk')` — 调用meshio.gmsh读取gmsh文件
- `Mesh.n_cell(self)`
- `Mesh.n_vert(self)`
- `Mesh.n_attr(self)`
- `Mesh.unique_attrs(self)`
- `Mesh.cell_centers(self)`
- `Mesh.vol_frac(self)`
- `Mesh.spec_surf_area(self)` — 比表面积
- `Mesh.cell_volume(self)`
- `Mesh.reg_volume(self)`
- `Mesh.cls_id_str(self)`
- `Mesh.cls_str(self)`
- `Mesh.id_str(self)`
- `Mesh.find(self, mark)`
- `Mesh.mask_mesh(self, cell_mask)` — 从当前单元网格掩膜构造新的网格
- `Mesh.grid_sample(self, n_intp=(50, 50, 50), mode='vert')`
- `Mesh.write_meshio(self, path='', name='gmsh_mesh', fmt='vtk', on=True)` — 调用meshio写文件
- `Mesh.write_gmsh(self, path='', name='meshio_mesh', fmt='msh', binary=False, on=True)` — 调用meshio.gmsh写.msh文件
- `Mesh.write_mph(self, path='', name='mph_mesh', fmt='mphtxt', on=True)` — 导出comsol支持的mph格式(三角形, 四边形, 四面体, 六面体)
- `Mesh.attr_map(self, map: list[int])`
- `Mesh.vol_query(self, vol, stable=False)` — 顶点插值采样 网格采样 返回datavol
- `Mesh.query(self, loc, stable=False)` — 顶点插值采样
- `Mesh.ndim(self)` — 空间维度 默认verts按顺序填充维度
- `LineMesh()`
- `LineMesh.from_vol(cls, vol: PhaseVol)` — 从体素网格(三维, (n, 1, 1))初始化的线段网格
- `TriMesh()`
- `TriMesh.surf_area(self)`
- `TriMesh.h(self)`
- `TriMesh.cell_volume(self)` — 网格表面积计算
- `TriMesh.volres_weight(cls, phs_vol: PhaseVol, smp_verts, sigma=1.0, n_level=np.inf, wfac=2.0)` — 残差权重计算  中值+平滑/高斯
- `TriMesh.mcvrt_weight(cls, phs_vol: PhaseVol, smp_verts, sigma=1.0, n_level=np.inf, wfac=2.0)` — 残差权重计算  中值+平滑/高斯
- `TriMesh.from_vol(cls, vol: PhaseVol, vsub=0.1, sigma=1.0, n_level=np.inf, wfac=2.0)` — 从体素网格初始化的四面体网格
- `TriMesh.from_vol2d(cls, vol: PhaseVol, vsub=0.1, sigma=1.0, n_level=np.inf, wfac=1.0)` — 从体素网格初始化, 平面三角网格
- `TriMesh.skewness_quality(self)` — 向量化计算三角形网格的偏度
- `remove_degenerate(tri_verts, tri_faces, quality_th=0.01)` — 移除有重叠部分的三角面 按照对角长度选择面进行修复
- `remove_entities(entities, to_remove)` — 从实体列表中移除指定的实体（忽略顶点顺序），完全向量化实现。
- `fix_tetrahedrons(verts, faces, tets)` — 将平面四面体划分转化为一致的三角面
- `TetMesh()`
- `TetMesh.from_sur(cls, sur_mesh, vol=PhaseVol.ones((3, 3, 3)))` — 从表面网格剖分四面体网格
- `TetMesh.from_vol(cls, vol, vsub=0.1, sigma=1.0, n_level=np.inf, wfac=2.0)` — 从体素网格剖分四面体网格
- `TetMesh.cell_volume(self)`
- `TetMesh.circum_radius(self)` — 计算四面体网格中每个四面体的外接圆半径
- `TetMesh.skewness_quality(self)` — 向量化计算四面体网格的偏度
- `TetMesh.h(self)` — 计算四面体网格中所有四面体的最大高度
- `TetMesh.surf_mesh(self)` — 求表面网格,界面属性重新映射,表面重新旋转
- `TetMesh.surf_area(self)` — 四面体网格表面积计算
- `attr_marching_cubes(attr_vol: PhaseVol, method='lorensen')` — 多属性marching_cubes算法
- `extract_surface_triangles(label)` — label: (Nx,Ny,Nz) 多相体素标签，0 表示空位
- `sur_optimize(verts, faces, nvrt, weights=np.array([]))` — 加权表面粗化
- `sur_tetgen(verts, faces, order=1, mindihedral=15, minratio=2.0, nobisect=True)` — 四面体化
- `remove_dul_points(vertices)` — 移除重复点,输入整数点坐标
- `remove_unuse_points(vertices, faces)`
- `remove_dul_points_xyz(vertices)`
- `face_ncell(cells)` — 建立单元-相邻单元表
- `edge_length(verts, tets)` — 计算单元的边长
- `QuadMesh()`
- `QuadMesh.from_vol_2d(cls, attr_vol: PhaseVol)`
- `QuadMesh.from_vol(cls, vol: PhaseVol)`
- `HexMesh(self, verts, cells, vert_attr, cell_attr)`
- `HexMesh.from_vol(cls, vol: PhaseVol)` — 将体素网格转化为六面体网格
- `HexMesh.cell_volume(self)`
- `HexMesh.vol_from_tet(cell_verts, cells, cell_attrs, voxel_l, stride=1)` — 将四面体网格转化为体素网格
- `HsHexMesh(self, verts, cells, vert_data, cell_data, face_mark=np.array([]), cell_mark=np.array([]))`
- `HsHexMesh.from_vol(cls, vol: PhaseVol)` — 将体素网格转化为六面体网格
- `HsHexMesh.cell_volume(self)`
- `HsHexMesh.add_axis_bound(self, axis, mark=(0, 0), cell_relative_th=0.6, face_relative_th=0.1)` — 为体积网格添加坐标轴指定的边界,mark指示上下界的标记,0表示不为边界 要求mark的值大于0. 默认单元顺序zyx.
- `HsHexMesh.read_hsdat(cls, path='', name='hsdat_mesh', fmt='hsdat', on=True)`
- `HsHexMesh.vol(self)`
- `HsHexMesh.write_hsdat(self, path='', name='hsdat_mesh', fmt='hsdat', on=True)`
- `HsHexMesh.write_bounds(self, path='', name='hsdat_bnd', fmt='hsbnd', on=True)` — 如果标记了边界则进行边界文件的写出
- `HsHexMesh.faces(self)`
- `HsHexMesh.n_face(self)`
- `HsHexMesh.face_centers(self)`
- `where_xyz(vol)` — 按xyz方向顺序的where
- `np_mode(arr)` — 针对非负整数的优化版本
- `extract_faces_tet(cells)` — 从单元提取表面,顺序对应
- `extract_tets_hex(cells)` — 从单元提取四面体
- `extract_faces_hex(cells)` — 从单元提取表面,顺序对应(有冗余,临时)
- `extract_faces_quad(cells)`
- `query_value(points, data, shape=(50, 50, 50))` — 获取网格数据
- `subdivide_mesh(vertices, faces)` — 对三角网格进行一次细分，每个三角形细分为4个小三角形
- `compute_face_normals(vertices, faces)` — 计算三角形面的法向量
- `compute_vertex_normals(vertices, faces)` — 计算三角表面网格的顶点法向量
- `vert_face_normals(vert_normals, faces)` — 从顶点计算面法向量
- `orient_tetrahedrons(tetrahedrons, vertices)` — 调整四面体顶点顺序使法向一致向外
- `calc_voro(points, limits, dispersion, weights)` — 计算voroni图 返回顶点坐标列表,多边形顶点索引列表
- `voro_extract(voro_regions)` — 从voro域提取全局顶点坐标和多边形索引表
- `file_str_list(fname, comm_start='#')`
- `read_node(fname)` — 读取节点坐标文件
- `read_ele(fname)` — 读取单元文件
- `read_face(fname)` — 读取面(包含邻接关系)文件
- `read_neigh(fname)` — 读取邻接单元文件
- `read_tetgen(fname, idx=1)` — 读取tetgen文件,返回顶点,面,单元,邻接关系
- `VoronoiMesh(self, points, reg, rads=[], attrs=[])`
- `VoronoiMesh.from_seeds(cls, seeds, reg)` — 从种子初始化
- `VoronoiMesh.n_region(self)`
- `VoronoiMesh.n_point(self)`
- `VoronoiMesh.n_vertice(self)`
- `VoronoiMesh.write_poly(voro_mesh, fname='')` — 写voronoi线性多面体网格文件(tetgen的poly)
- `VoronoiMesh.calc_tetra_tetgen(voro_mesh, min_angle=14)`

### [morqex.microstructure.mesh.pyacvd.clustering](../src/morqex/microstructure/mesh/pyacvd/clustering.py)

- `Clustering(self, points, faces, weights=[])` — Uniform point clustering based on ACVD.
- `Clustering.cluster(self, nclus, maxiter=100, debug=False, iso_try=10)` — Cluster points
- `Clustering.create_mesh(self, flipnorm=True)` — Generates mesh from clusters
- `Clustering.cluster_norm(self)` — Return cluster norms
- `Clustering.cluster_centroid(self)` — Computes an area normalized value for each cluster
- `cluster_centroid(cent, area, clusters)` — Computes an area normalized centroid for each cluster
- `create_mesh(points, faces, area, clusters, cnorm, flipnorm=True)` — Generates a new mesh given cluster data
- `unique_row_indices(a)` — Indices of unique rows
- `weighted_points(points, faces, return_weighted=True, additional_weights=None)` — Returns point weight based on area weight and weighted points.
- `neighbors_from_mesh(points, faces)` — Assemble neighbor array.  Assumes all-triangular mesh.
- `compute_face_normals(vertices, faces)` — 计算三角形面的法向量
- `compute_vertex_normals(vertices, faces)` — 计算三角表面网格的顶点法向量

### [morqex.microstructure.mesh.pyacvd.clustering_bk](../src/morqex/microstructure/mesh/pyacvd/clustering_bk.py)

- `Clustering(self, verts, faces, weights)` — Uniform point clustering based on ACVD.
- `Clustering.cluster(self, nclus, maxiter=100, debug=False, iso_try=10)` — Cluster points
- `Clustering.subdivide(self, nsub)` — Perform a linear subdivision of the mesh.
- `Clustering.create_mesh(self, flipnorm=True)` — Generates mesh from clusters
- `Clustering.cluster_norm(self)` — Return cluster norms
- `Clustering.cluster_centroid(self)` — Computes an area normalized value for each cluster
- `cluster_centroid(cent, area, clusters)` — Computes an area normalized centroid for each cluster
- `create_mesh(mesh, area, clusters, cnorm, flipnorm=True)` — Generates a new mesh given cluster data
- `unique_row_indices(a)` — Indices of unique rows
- `weighted_points(mesh, return_weighted=True, additional_weights=None)` — Returns point weight based on area weight and weighted points.
- `neighbors_from_mesh(mesh)` — Assemble neighbor array.  Assumes all-triangular mesh.

### [morqex.microstructure.mesh.tetgen.pytetgen](../src/morqex/microstructure/mesh/tetgen/pytetgen.py)

- `TetGen(self, *args)` — Input, clean, and tetrahedralize surface meshes using TetGen.
- `TetGen.tetrahedralize(self, plc=True, psc=0.0, refine=0.0, quality=True, nobisect=False, cdt=0.0, cdtrefine=7.0, coarsen=0.0, weighted=0.0, brio_hilbert=1.0, flipinsert=0.0, metric=0.0, varvolume=0.0, fixedvolume=0.0, regionattrib=0.0, insertaddpoints=0.0, diagnose=0.0, convex=0.0, nomergefacet=0.0, nomergevertex=0.0, noexact=0.0, nostaticfilter=0.0, zeroindex=0.0, facesout=0.0, edgesout=0.0, neighout=0.0, voroout=0.0, meditview=0.0, vtkview=0.0, vtksurfview=0.0, nobound=0.0, nonodewritten=0.0, noelewritten=0.0, nofacewritten=0.0, noiterationnum=0.0, nojettison=0.0, docheck=0.0, quiet=0.0, nowarning=0.0, verbose=0.0, vertexperblock=4092.0, tetrahedraperblock=8188.0, shellfaceperblock=2044.0, supsteiner_level=2.0, addsteiner_algo=1.0, coarsen_param=0.0, weighted_param=0.0, fliplinklevel=-1.0, flipstarsize=-1.0, fliplinklevelinc=1.0, opt_max_flip_level=3.0, opt_scheme=7.0, opt_iterations=3.0, smooth_cirterion=1.0, smooth_maxiter=7.0, delmaxfliplevel=1.0, order=1.0, reversetetori=0.0, steinerleft=100000.0, unflip_queue_limit=1000.0, no_sort=0.0, hilbert_order=52.0, hilbert_limit=8.0, brio_threshold=64.0, brio_ratio=0.125, epsilon=1e-08, facet_separate_ang_tol=179.9, collinear_ang_tol=179.9, facet_small_ang_tol=15.0, maxvolume=-1.0, maxvolume_length=-1.0, minratio=2.0, opt_max_asp_ratio=1000.0, opt_max_edge_ratio=100.0, mindihedral=0.0, optmaxdihedral=177.0, metric_scale=1.0, smooth_alpha=0.3, coarsen_percent=1.0, elem_growth_ratio=0.0, refine_progress_ratio=0.333, switches=None)` — Generates tetrahedrals interior to the surface mesh

### [morqex.microstructure.mesh.voxel](../src/morqex/microstructure/mesh/voxel.py)

- `FEMesh()`
- `FEMesh.signed_volumes(self)`
- `FEMesh.boundary_faces(self)`
- `FEMesh.save(self, path)`
- `FEMesh.save_comsol(self, path, **kwargs)` — Unified streaming COMSOL Mesh-class4 writer; explicit material/domain mapping, no GUI dependency.
- `voxel_to_hex(volume, phases=None)`
- `voxel_to_tetra(volume, phases=None)`
- `surface(volume, phase=1)`

### [morqex.microstructure.pipeline](../src/morqex/microstructure/pipeline.py)

- `Microstructure()`
- `Microstructure.load(cls, path, **kwargs)`
- `Microstructure.generate(cls, checkpoint, **kwargs)`
- `Microstructure.save(self, path)`
- `Microstructure.mesh(self, kind='tetra', phases=None)`
- `Microstructure.describe(self)`
- `Microstructure.show(self)`
- `Microstructure.homogenize(self, coefficients, order=2)`

### [morqex.microstructure.reporting](../src/morqex/microstructure/reporting.py)

- `structure_html(volume, path, title='GEM v1 三维微结构', phase_names=None, max_triangles=120000, reference=None)`

### [morqex.microstructure.simulate.cell](../src/morqex/microstructure/simulate/cell.py)

- `discharge(model='DFN', parameter_set='Chen2020', c_rate=1.0, duration=3600.0, points=101, updates=None)`
- `microstructure_parameters(volume, pore_phase=0, electrode='Negative', tortuosity=None, active_phase=None)` — Map measured pore and active fractions consistently to a cell model.

### [morqex.microstructure.simulate.elasticity](../src/morqex/microstructure/simulate/elasticity.py)

- `homogenize_elasticity(mesh, properties, order=1)` — Returns engineering-Voigt stiffness [xx,yy,zz,yz,xz,xy].

### [morqex.microstructure.simulate.electrochemistry](../src/morqex/microstructure/simulate/electrochemistry.py)

- `backend_info()` — Report the installed native build, independently of numerical validation.
- `assemble_cell(anode, cathode, separator_voxels, anode_map, cathode_map)` — Assemble cathode|separator|anode along x, matching upstream boundary conventions.
- `prepare_case(volume, directory, phase_map, parameters=None, template=None)` — Write geometry, complete upstream Microstructure input template, and overrides.
- `export_geometry(volume, directory, phase_map, nanoporosity=1.0, macmullin=1.0)` — Upstream x-fastest format: nx ny nz then phase labels.
- `AMRConfig()`
- `AMRConfig.write(self, path)`
- `run(inputs, workdir, timeout=None, overrides=None)` — Native Python binding runs in an isolated process (AMReX global lifecycle).
- `DischargeSettings()`
- `DischargeCurve()`
- `DischargeCurve.plot(self)`
- `read_discharge(directory)`
- `audit_solver(directory)` — Read actual convergence/mass evidence; never infer convergence from a curve.
- `discharge_3d(anode, cathode, directory, anode_map, cathode_map, separator_voxels=4, settings=None, timeout=None, parameters=None)` — Fully coupled upstream microstructure-scale discharge; SI units required.

### [morqex.microstructure.simulate.evolution](../src/morqex/microstructure/simulate/evolution.py)

- `DiffusionHistory()`
- `diffuse(mesh, coefficients, initial=1.0, dt=0.01, steps=10, order=2, dirichlet='', boundary_value=0.0, save_every=1)`

### [morqex.microstructure.simulate.homogenization](../src/morqex/microstructure/simulate/homogenization.py)

- `homogenize(volume, coefficients, order=2)`
- `tortuosity(volume, pore_phase=0, axis='x', order=2)` — Pore-only RVE; isolated pores are removed to avoid singular FEM systems.

### [morqex.microstructure.simulate.periodic](../src/morqex/microstructure/simulate/periodic.py)

- `conductivity_tensor(volume, coefficients, order=2)`

### [morqex.microstructure.simulate.transport](../src/morqex/microstructure/simulate/transport.py)

- `to_ngsolve(mesh, periodic=False)`
- `ConductionResult()`
- `conduction(mesh, coefficients, axis='x', order=2)` — Unit potential difference; natural zero flux on lateral and pore boundaries.
- `save_field(result, path)`

### [morqex.microstructure.simulate.voxel_transport](../src/morqex/microstructure/simulate/voxel_transport.py)

- `effective_diffusion(volume, phase=0, axis='z', device='cuda', tolerance=1e-05, max_iterations=50000)`

### [morqex.microstructure.view](../src/morqex/microstructure/view.py)

- `slices(volume, indices=None)`
- `mesh(femesh, **kwargs)`
- `raytrace(volume, phases=None, colors=None, start=True)` — Optional OptiX voxel rendering without hard-coded files or scene dimensions.

### [morqex.microstructure.volume](../src/morqex/microstructure/volume.py)

- `Volume()`
- `Volume.fractions(self)`
- `Volume.save(self, path)`
- `Volume.load(cls, path, **kwargs)`

### [morqex.microstructure.workspace](../src/morqex/microstructure/workspace.py)

- `Workspace()`
- `Workspace.default(cls)`
- `Workspace.path(self, category, *parts)`
- `Workspace.initialize(self)`

### [morqex.runtime.aging](../src/morqex/runtime/aging.py)

- `AgingRuntime(self, operator_paths, parameters, config, *, nx=3, feedback=1.0, configuration_manifest=None, rtol=1e-07, atol=1e-08)`
- `AgingRuntime.advance(self, dt, current_A)`
- `AgingRuntime.recover(self, domain, *, x_index=0, selection=None)`
- `AgingRuntime.reaction_bound(self, domain, *, x_index=0)`
- `AgingRuntime.estimate_defect(self, *args, **kwargs)`
- `AgingRuntime.checkpoint(self, path)`
- `AgingRuntime.restore(self, path)`
- `AgingRuntime.cohesive_error_bound(self, domain, *, x_index=0)` — Fixed-state compression bound; never a nonlinear trajectory bound.
- `AgingRuntime.enrich(self, new_paths, *, relative_energy_tolerance=1e-05)` — Nested concentration and reaction-area reservoir transfer.

### [morqex.solvers.pybamm_backend](../src/morqex/solvers/pybamm_backend.py)

- `solver(*, rtol=1e-07, atol=1e-08)`
- `provenance()`

### [morqex.sparse_coupled](../src/morqex/sparse_coupled.py)

- `sparse_bytes(a)`
- `SparseCoupledState()`
- `SparseChemoelasticSystem(self, solver, diffusion, expansion, patterns, *, chemical_curvature, reference_concentration, maximum_concentration)`
- `SparseChemoelasticSystem.solve_mechanical(self, rhs)`
- `SparseChemoelasticSystem.h_action(self, z)`
- `SparseChemoelasticSystem.solve_h(self, rhs, *, relative_tolerance=1e-12)`
- `SparseChemoelasticSystem.fields(self, c, e)`
- `SparseChemoelasticSystem.energy(self, z, v, e)`
- `SparseChemoelasticSystem.initialize(self, c, *, macro_strain=None)`
- `SparseChemoelasticSystem.prepare(self, dt, method)`
- `SparseChemoelasticSystem.step(self, previous, dt, u, *, macro_strain=None, method='Schur-GMRES', relative_tolerance=1e-10)`
- `SparseChemoelasticSystem.retained_sparse_bytes(self)`
- `SparseChemoelasticSystem.coupled_resolvent_basis(self, count, *, shifts=(0.01, 0.1, 1.0, 10.0), moments=3)` — Established rational Krylov responses of the actual coupled generator.

### [morqex.statistics](../src/morqex/statistics.py)

- `principal_tension(stress)`
- `von_mises(stress)`
- `weighted_quantile(values, weights, probability)`
- `weighted_cvar(values, weights, probability=0.99)`
- `StressMoments()`
- `StressMoments.build(cls, response, selection=None)`
- `StressMoments.evaluate(self, q)`
- `StressEnvelope()`
- `StressEnvelope.build(cls, response, count, scales, *, radius=1.0, selection=None)`
- `StressEnvelope.evaluate(self, q, *, probability=0.99, threshold=0.0)`

### [morqex.streamed_rom](../src/morqex/streamed_rom.py)

- `StreamedState()`
- `OnlineEnergyKernel()`
- `OnlineEnergyKernel.dimension(self)`
- `OnlineEnergyKernel.energy(self, a, e)`
- `OnlineEnergyKernel.accept(self, state, tolerance)`
- `OnlineEnergyKernel.trial(self, previous, dt, u, *, macro_strain=None)`
- `OnlineEnergyKernel.step(self, previous, dt, u, *, error_tolerance, macro_strain=None)`
- `OnlineEnergyKernel.online_array_bytes(self)`
- `OnlineEnergyKernel.macro_stress(self, state)`
- `OnlineEnergyKernel.batch_macro(self, coefficients, strains)`
- `OnlineEnergyKernel.save(self, workspace, name, state=None)`
- `OnlineEnergyKernel.load(cls, path)`
- `StreamedEnergyROM(self, system, basis, *, projection='energy', add_mass_riesz=True, add_macro_lifting=True)`
- `StreamedEnergyROM.initialize(self, initial, system, *, macro_strain=None, error_tolerance)`
- `StreamedEnergyROM.detach_online(self)`
- `StressLocalization(self, system, rom, selection, *, maximum_bytes=128 * 2 ** 20)` — Optional stress-only output columns; construction is streamed by mode.
- `StressLocalization.stress(self, state)`
- `StressLocalization.moments(self, state)`
- `StressLocalization.detach_statistics(self)`
- `QuadraticStressStatistics(self, mean, second, von_mises_second, fingerprint)` — Exact selected-volume linear means and quadratic moments, O(m²) storage.
- `QuadraticStressStatistics.moments(self, state)`

### [morqex.tail](../src/morqex/tail.py)

- `QuantileResult()`
- `QuantileTree(self, response, *, scales=None, selection=None)`
- `QuantileTree.query(self, q, *, probability=0.99, relative_tolerance=0.05, absolute_tolerance=1.0, scale_floor=1.0, threshold=0.0, maximum_nodes=None)`

### [morqex.topology](../src/morqex/topology.py)

- `PorousCBD()`
- `PorousCBD.ionic_phases(self, structure)`
- `PorousCBD.ionic_efficiencies(self, structure)`
- `PorousCBD.porosity(self, structure)`
- `PorousCBD.metadata(self)`

### [morqex.transport](../src/morqex/transport.py)

- `ConcentrationInfeasible()`
- `DiffusionState()`
- `VoxelDiffusion(self, volume, *, active_phases, diffusivity, particle_ids=None, reactive_phases=(), exterior_reactive=False)`
- `VoxelDiffusion.source(self, flux)`
- `VoxelDiffusion.step(self, concentration, dt, *, flux=None, maximum=None)`
- `VoxelDiffusion.expand(self, values, *, inactive=0.0)`
- `conservative_box_projection(values, weights, target, *, lower=0.0, upper=1.0)` — Weighted L2 projection with exact weighted-mass equality and box bounds.

### [morqex.visualization.full_volume](../src/morqex/visualization/full_volume.py)

- `render_report(payload, destination, *, font_path=None, font_license=None)`

### [morqex.visualization.report_font](../src/morqex/visualization/report_font.py)

- `embed_font(html, path, license_text)`

### [morqex.visualization.voxel_fields](../src/morqex/visualization/voxel_fields.py)

- `file_hash(path)`
- `export_reference_fields(model, u, c, directory, *, macro=(0.0,) * 6, temperature=0.0, chunk=131072)` — Exact discrete gradients/stress; no LOD in the saved numerical fields.

### [morqex.workflow](../src/morqex/workflow.py)

- `ElectrodeRecipe()`
- `ElectrodeRecipe.prepare(self, workspace, parameters, domain, name)`
- `CoupledRecipe()`
- `CoupledRecipe.prepare(self, workspace, parameters, name)`
- `CoupledRecipe.model(self, paths, parameters)`

### TOP-DAMAGE 2S finite-strain reference

`morqex.top_damage.hyperelastic` exposes:

- `build_hyperelastic(NativeConfig, Workspace)`: compile a small C++ adapter against the existing hash-verified vendored MFEM library; no upstream rebuild.
- `HyperelasticBackend.evaluate(F, chemical_F, shear, bulk, degradation=1, tangent=True)`: reference energy, undegraded tensile driving energy, first Piola stress and consistent 9×9 tangent.
- `Hyperelasticity(mesh, backend, young_Pa, poisson, boundary='kubc', linear_solver='direct')`: Tet4 total-Lagrangian equilibrium; `matrix-free-amg-minres` avoids global tangent assembly.
- `Hyperelasticity.solve(chemical_F, degradation, macro_F=..., force=..., initial_displacement=...)`: full 3×3 macro deformation or dead nominal loads. Rejects inverted elements and changed geometry.
- `chemical_potential(P, F, chemical_F, derivative_chemical_F)`: elastic derivative at fixed F and fixed moduli; concentration-dependent modulus derivatives are not silently supplied.
- `finite_staggered_step(...)`: fixed accepted fatigue history during alternating minimization; commits positive fatigue variation once on convergence.

The finite energy uses MFEM's isochoric Neo-Hookean form with a tensile/compressive volumetric split. It is not the logarithmic-volume DOLFINx demo energy. Elastic moduli remain isotropic; chemical deformation may be anisotropic. Contact, wetting and physical crack topology require separate models. A converged positive-J solution is not automatically a calibrated CBD prediction.

```bash
PY=/home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python
$PY scripts/run_local.py --label finite_build -- $PY -m morqex.top_damage build-hyperelastic /home/zephyrain/lw/MORQEX/configs/top_damage_native.toml
$PY scripts/run_local.py --label finite_refinement -- $PY /home/zephyrain/lw/MORQEX/scripts/run_top_damage_finite_validation.py --levels 4 8 16 --output finite_B0_refined
$PY scripts/run_local.py --label finite_cycle -- $PY -m morqex.top_damage dense-cycle /home/zephyrain/lw/MORQEX/configs/top_damage_finite_cycle.toml --environment gentle --steps 1 --output finite_gentle_v1
```

Repeat the last command to advance bounded accepted steps. Reusing an output with changed physical-source identity is rejected. Finite checkpoints additionally contain `finite/deformation_gradient`, `chemical_deformation_gradient`, `first_piola_Pa`, `cauchy_stress_Pa`, and `mechanical_chemical_potential_J_mol`; displayed strain is Green–Lagrange. This current cycle candidate retains the explicitly restricted radial concentration and diffuse conductivity closure, not physical pore creation.

### [morqex.workspace](../src/morqex/workspace.py)

- `Workspace()`
- `Workspace.path(self, *parts: str)`

### TOP-DAMAGE topology candidates and periodic transport

- `periodic_transport(field, spacing_zyx_m, tolerance=1e-10)`: three periodic finite-volume correctors with one shared AMG preconditioner. Returns an immutable **XYZ 3×3 tensor**, corrector fields, full conservation residuals and a flux/energy check. True zero phases and floating components are retained. The input is a local scalar constitutive field; this does not infer particle diffusivity, anisotropic local tensors or crack aperture. Periodic effective tensors must not be confused with the insulated-side finite-block measurement from `effective_conductivity`.
- `reconstruct_opening(points_m, displacement_m, centres_m, normals, inner_distance_m=..., outer_distance_m=..., tangential_radius_m=..., sample_damage=..., weights=...)`: two-sided affine displacement extrapolation on **explicitly supplied crack patches**. Includes finite-rotation covariance, positive aperture, signed gap, tangential slip, support conditioning and fit diagnostics. `result.require_usable()` rejects unresolved fits. Negative signed gap is preserved; this is not a solved contact constraint. The caller must supply the relevant material neighbourhood, crack surface and support convergence study. No automatic damage-to-pore conversion or wetting is performed.

The displacement-plane idea is prior art (Babaoğlu, Kaynar & Göktepe, CMAME 458, 119016, DOI [10.1016/j.cma.2026.119016](https://doi.org/10.1016/j.cma.2026.119016)). The present independent 3D local fit does not reproduce that paper's complete spline/path-detection algorithm. Its kernel tests are not physical battery aperture qualification.

The finite-strain Newton solver records `solve_diagnostics`. A near-equilibrium residual-decrease merit avoids rejecting a valid force correction solely because strain-energy differences fall below the floating-point invariant-subtraction scale. The requested equilibrium tolerance is unchanged; this arithmetic scale is not a rigorous physical error bound.

The residual denominator is fixed for each Newton solve using the chemical/macroscopic lift force, applied load, initial trial force and a documented material-scale floor. This avoids a false nonconvergence when both internal force and residual approach zero at a free boundary. The returned state includes `equilibrium_residual_N` and `equilibrium_force_scale_N`; a relative residual alone is not a discretization error estimate.

Native ABI 2 adds a serial FP64 Tet4 tangent action. `matrix-free-amg-minres` uses the compiled action; `numpy-matrix-free-amg-minres` retains the independent NumPy action for regression and fair timing. `scripts/benchmark_top_damage_tangent.py` measures identical stored tangents on the same CPU. Its speed ratio applies only to one matrix-vector action, not to full cycles or GPU execution.

### TOP-DAMAGE geometric concentration ports

`morqex.top_damage.region_port.RegionPort.from_diffusion(fv, partition, reactive_faces, reference_concentration)` reuses `VoxelDiffusion` with **unit** geometric diffusivity. It assembles conservative regional mass, actual accessible reaction area and a graph Laplacian. The caller supplies a verified reactive-face subset; a region cannot merge disconnected material. `lift`, `restrict`, `field` and `project_mechanical_potential` share the same material volumes. The potential projection preserves virtual chemical work for the represented concentration; it does not claim pointwise equality or eliminate closure error.

`morqex.top_damage.region_dfn.RegionEpochDFN(port, host='NMC811', porosity=..., nx=6, transport_law=TransportLaw(...))` replaces radial particle states with regional concentrations at each positive-electrode node. It retains the planar Li counterelectrode and conservative half-cell electrolyte. `nx >= 3` is required by the current DFN boundary extrapolation. The initial template is shared, but states and mechanical-potential inputs are independent at each node.

- `advance_regions(dt, current_A, DFNFeedback(...), mechanical_potential_J_mol)` requires a finite `(nx, regions)` potential array and rejects simultaneous scalar mechanical/reaction-fraction corrections. A failed numerical gate rolls back the solver's accepted solution as well as the public state.
- `concentrations()` returns `(nx, regions)`; `port.field(...)` restores **all** active voxels for every requested node, with inactive cells marked NaN by default.
- `restore_state(time_s, state, inputs)` supports an unchanged state layout; topology changes require inventory-preserving transfer and a rebuilt operator/model.
- Positive mobility and declared gradient energy follow the existing `TransportLaw`. Reference diffusivity is a tracer/mobility input, not automatically a measured chemical Fick diffusivity. Nonconvex OCP requires positive gradient energy and subsequent spatial-resolution validation.

Qualification: synthetic local-potential feedback, conservation, chemical port identity, rejected-step rollback and restart tests for NMC532/NMC811/LFP/LCO. This fixed-topology candidate does not yet advance the published dense-cycle trajectory, infer electrolyte wetting, solve six nonlinear microstructures simultaneously, or establish phase-interface/material/lifetime accuracy. The nonlinear mechanics input is frozen within a segment; coupling time-step/Picard convergence remains required.

The bounded research workflow is now available through the same MORQEX CLI:

```bash
$PY scripts/run_local.py --label region_probe -- $PY -m morqex.top_damage region-probe /home/zephyrain/lw/MORQEX/configs/top_damage_region_probe.toml
```

This command prepares the regional DFN state, solves each configured microstructure node serially, projects finite-mechanics chemical potentials, advances a second DFN segment, runs a matched zero-feedback comparison, and creates the voxel-to-node HTML. `--stage prepare`, `--stage node --node 0`, `--stage feedback`, and `--stage report` provide bounded restart points. Repeating a completed stage reads its result. Configuration/source/geometry/checksum changes reject resume; select a new output identifier instead. `run_probe(RegionProbeConfig(...), workspace, stage=...)` provides the identical Python API.

The TOML controls node/region counts, geometry, material host, temperature, time increment, C-rate, explicit CBD microporosity/ionic efficiency, chemical stretch, elastic/fracture constants and solver tolerance. This fixture uses pore/active/CBD labels 0/1/2 and fixed zero-macro-strain KUBC at each microdomain; it does **not** enforce a series-stack force condition. A 64-region request produced 66 connected regions on the supplied geometry; disconnected regional supports are split rather than silently merged. The supplied values are research sensitivity parameters, not a calibrated NMC fatigue card. Porous CBD here is an unresolved overlapping continuum, not CT-resolved liquid pores.

## 解析三维电化学 → 有限变形/疲劳相场

所有命令通过 `scripts/run_local.py`，单线程、低优先级；不在源码生成数据。
以下 `PY` 指外部虚拟环境 Python，`ROOT` 指 MORQEX 源目录。

```bash
PY=/home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python
ROOT=/home/zephyrain/lw/MORQEX
$PY $ROOT/scripts/run_local.py --label resolved_build -- $PY -m morqex.top_damage build-resolved $ROOT/configs/top_damage_resolved_hypre.toml
```

构建返回 `dependency_pending` 或 `pending` 时重复同一命令，直到 `complete`。
每个 Ninja 步限一线程/55秒。固定 HYPRE 2.32.0 压缩包 SHA256 和源文件逐项校验，
许可证保留；首次需网络。仅 CPU/FP64 被验证，不能把它当作 GPU 路径。

```bash
$PY $ROOT/scripts/run_local.py --label planar -- $PY $ROOT/benchmarks/validate_resolved_echem.py --output my_planar_check
$PY $ROOT/scripts/run_local.py --label particles -- $PY $ROOT/benchmarks/validate_resolved_particles.py --output resolved_particles_reference
$PY $ROOT/scripts/run_local.py --label phase -- $PY -m morqex.top_damage resolved-phase-field $ROOT/configs/top_damage_resolved_phase_field.toml
# 也可对全部已存三维场执行历史链；已运行第0步则用 --start 1
$PY $ROOT/scripts/run_local.py --label history -- $PY $ROOT/benchmarks/validate_resolved_phase_history.py results/top_damage/resolved_particles_reference --start 1
```

颗粒基准需要 `configs/top_damage_dense_particles.toml` 生成的父几何，
通过 `top-damage geometry` 入口先生成。基准明确创建一个新的初始连通 CBD 结构，
保留全部活性颗粒，记录初始闭孔/CBD 处理；该处理不是损伤演化规则。
基准检查惰性 CBD 锂浓度为零、固相电流积分库存、电解液库存、逐次非线性残差，
但还不证明微结构网格收敛或真实材料寿命。`--dt`、`--stop` 可指定数值验证窗口。
`--warm-start` 复用电势迭代初值；`--linear-budget 0.1` 将外层物理残差容差的
10% 分配给内层线性残差（单层网格的 l2/linf 换算），外层验收容差不变。
这不是点态误差界，也不能消除显式反应—隐式扩散分裂的时间步限制。
所有原生试步在进入 OCP/BV 前检查浓度可行性，越界即失败并保留前一检查点，
禁止截断浓度来取得通过。大步失稳案例及拒绝验证见
`benchmarks/validate_resolved_safeguards.py`；时间步场对照用
`benchmarks/compare_resolved_time_steps.py coarse_result fine_result --output unique.json`。

三维相场 TOML 的 `phase_field` 指定 `geometry`、`field`、`output` 和可选 `previous`。
下一时刻必须提供上一接受态；同一几何、网格、材料/源码身份及严格递增时间被检查，
已有输出禁止覆盖。`mechanics` 可指定弹性、断裂、化学膨胀、正则化长度、约束和浓度尺度。
默认机械参数是合成验证参数，不等于实验标定。
批量历史驱动每帧使用独立进程、保存 `mechanics_*.solver.log`；可用 `--steps 20`
分批，再用 `--start 20` 续算，避免长期保留求解器导致内存累积。
AMG 预条件器已消除持有力学对象的引用环；同进程内存回归用
`benchmarks/validate_resolved_memory.py --seeded-isolated --output unique_name`。
该对照固定 PyAMG 谱半径估计所用的随机序列，逐字段比较进程内外结果，
不使用手工垃圾回收或缩小网格。不同源码版本的历史身份不能手改；旧接受态应由
记录的源版本恢复。40 秒/41 帧参考对应版本 `0367d2a`。

```python
from morqex.top_damage.resolved_echem import ResolvedElectrochemistry
from morqex.top_damage.resolved_phase_field import ResolvedMechanicsConfig, advance_resolved_phase_field
backend = ResolvedElectrochemistry(workspace)
# backend.run(volume, directory, phase_map, parameters, restart=checkpoint)
# 原生正 Crate 表示充电，负值表示放电。必须显式提供时间步和停止时刻。
# export_fields 由 AMReX 原生 reader 导出完整 FP64 ZYX 场，不做插值。
record = advance_resolved_phase_field(geometry_h5, fields_h5, new_state_h5, workspace,
    config=ResolvedMechanicsConfig(), previous=previous_state_h5)
```

场接口核对两种固相掩码、原点/间距/轴序、浓度可行性；当前有限元桥要求零原点，
正极占据全横截面并位于 x 低端。独立体积积分检查体素→四面体锂库存。
相场返回力学化学势，但**尚未将其反馈到原生三维扩散/BV**。
原生三维限定单级、封闭电解液、阴极侧法向电势梯度和阳极侧基准电势边界，
拒绝浮置电势分量、CBD 双连续体、浓度截断及不适用选项。一般 MORQEX DFN API
仍独立存在，不能把两个已运行的分支称为全部耦合验收。
原生重启检查原来的晶格间距、原点和相拓扑；改变微结构后必须另行实现保守状态
映射，不能拿旧检查点覆盖新几何后继续称作动态拓扑求解。

```bash
$PY $ROOT/scripts/run_local.py --label fields --render -- $PY -m morqex.top_damage resolved-report results/top_damage/resolved_particles_reference --export
```

HTML 可切换时间、固相/液相浓度、电势、应力、损伤增量、疲劳和力学化学势，
完整 HDF5/VTI/PVD/XDMF 均在输入结果目录。损伤场没有被自动改成孔隙。

## Ai2022 独立颗粒复现接口

目标文献：Ai, Wu, Martínez-Pañeda, *Journal of Power Sources* 544 (2022),
231805，DOI `10.1016/j.jpowsour.2022.231805`。本接口为独立实现，没有复制作者
COMSOL 模型源码，不需要 COMSOL 许可。外部可选依赖见 `paper-particle` extra；
Gmsh 为独立安装的 GPL 工具，scikit-fem 为 BSD-3-Clause，pypdf 为 BSD-3-Clause。
作者模型、PDF 和研究数据只保存在外部工作区，保留各自权利和来源。

```bash
PY=/home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python
# 本机 Gmsh wheel 的 GLU 已安装在外部目录；系统有 GLU 时不需要此设置。
export LD_LIBRARY_PATH=/home/zephyrain/lw/morph_workspace/runtime/glu/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
$PY scripts/run_local.py --label paper_probe -- $PY -m morqex.top_damage paper-particle /home/zephyrain/lw/MORQEX/configs/top_damage_paper_sphere.toml --steps 20 --output paper_probe
# 每次追加有限步数；严格核对配置、求解器源码、依赖版本身份，不手改检查点身份。
$PY scripts/run_local.py --label paper_resume -- $PY -m morqex.top_damage paper-particle /home/zephyrain/lw/MORQEX/configs/top_damage_paper_sphere.toml --steps 20 --resume --output paper_probe
$PY scripts/run_local.py --label paper_reference -- $PY -m morqex.top_damage paper-reference /home/zephyrain/lw/morph_workspace/cache/ai2022_article.pdf
$PY scripts/run_local.py --label paper_html -- $PY -m morqex.top_damage paper-report /home/zephyrain/lw/morph_workspace/results/top_damage/paper_reproduction/paper_probe --reference /home/zephyrain/lw/morph_workspace/references/ai2022_figure5.json
```

`PaperParticle(mesh, ParticleMaterial(...), ...)` 接收无量纲 r/R,z/R 的 P1 三角形网格。
`advance(accepted, dt_s, outward_flux_mol_m2_s)` 返回新不可变接受态与残差、实际通量、
守恒、浓度界、裂纹域和疲劳指标。正通量是脱锂。位移按 R 缩放，浓度按 cmax 缩放；
能量密度、历史场按 E 缩放。`fields` 的应力按 E 缩放；报告转为 MPa。
HDF5 含全部 u/c/d 和积分点历史，不能把积分点历史误当体素或节点数据。
HTML 是轴对称子午面及裂纹附近放大图，不能冒充任意三维 CT 的相场解。

`configs/top_damage_paper_sphere.toml` 使用原文参数与散度形式相场，默认严格浓度检查；
失败保留 `failure.json` 和上一个接受态。`top_damage_ai2022_author_controls.toml`
则核对公开完整模型的 300 K、参考 SOC 0.9、初始宽度 2.5ell、韧度下限 0.01 和
Helmholtz 形式。此配置的 `dilute_diagnostic` 明确允许记录 c>cmax 的诊断结果，
**不能用于声称物理浓度门槛通过**。两种相场形式在空间非均匀疲劳下并不等价。
两条路线目前都使用 P1、Backward Euler、瞬时反向及可选不可逆 VI，
未实现作者高阶单元、BDF 和 Do Carmo/Galeão crosswind 稳定化的逐项等价。

`boundary_policy="surface_limits"` 对表面通量施加浓度互补条件，输出实际通量与
名义通量比值，守恒使用实际边界反力。它是独立定义的闭合，不是已证实的作者截断操作。
无论哪种模式，都不在求解后裁剪浓度。`linearization="chord"` 重用稀疏分解，
每次迭代仍重算完整非线性残差和从上个接受态出发的历史；失败刷新切线。
短十步同网格对照曾测得约 3.29 倍加速，不能外推为长期或 GPU 加速。

`benchmarks/validate_ai2022_time.py` 在同网格上执行两循环时间细化并比较完整场，
对照必须具有相同物理配置和求解器版本。输出浓度、损伤、疲劳差异和独立物理门槛，
不因浓度误差小而把疲劳标为收敛。PDF 曲线使用 SHA256 和轴标定追踪，读图容差
不等于原作者数值误差界。300 循环曲线、J 积分、原 CT 颗粒和 140 循环三维复现仍未验收。

报告可复用外部 OFL 中文字体：`scripts/fetch_report_font.py` 下载到
`morph_workspace/data/report_font`，`paper-report` 会嵌入改名后的字体子集，避免离线设备
显示方框。空间对照入口为 `benchmarks/validate_ai2022_space.py COARSE_DIR FINE_DIR --step 80`，
同样经 `run_local.py` 启动。它在共同体积积分点比较 P1 场并报告排除体积，使用网格定位器
避免全体积分点与全部单元两两搜索的内存爆炸；两个网格不能单独确立渐近收敛阶。

## 反应端口图与输出一致性补充

`ReactiveDualNetwork` / `ReactionFaces`（`top_damage.reaction_network`）在电子和离子网络上保留所有反应顶点，只对无源内部顶点做 Schur 消元。输入真实接触导率 S、界面面积 m²、OCP V、交换电流密度 A/m²、温度 K；`solve(discharge_current_A)` 联立求解两类工作电势与对称 Butler–Volmer 电流，正电流表示放电。返回电流守恒缺陷、欧姆/反应耗散及功率平衡。离子真实储库固定为0，集流体电压由求解得到；不存在的导电路径不能用任意小电导强行补上。

`voxel_reaction_problem(...)` 从同一体素网格的真实相邻界面构造这些端口，并移除均匀化所用的另一端虚拟试验电极。它的离子导率输入是实际 S/m；返回工作电势而非单位加载电势。当前验证范围是给定浓度、等温、准静态欧姆输运与非线性反应，未包含浓溶液扩散电势，也未默认替换长期 DFN 耦合。电子浮动但仍有离子接触的颗粒保留：其净反应电流必须为零，局部双极反应/内部再分布可以非零。不能将这种情况等同于删除其全部浓度状态。

`benchmarks/study_topology_counterexample.py --rates 0.5 8 --tag v1` 运行预设等面积干式接触失效的 DFN 两循环对照，保留失联锂库存；这里没有假称接触失效来自相场。`benchmarks/study_reactive_graph.py` 对比同一非线性问题的全图与反应端口消元解，保存实际三维工作电势。两者是不同验证层级。

`project_algebraic_output=true` 可在 `network-cycle` 中启用 DAE 输出时刻代数重构。浓度/容量等微分状态、接受步端点与截止事件逐位保留，使用源码内 PyBaMM/CasADi 重求内部输出时刻的代数约束。它不消除时间积分误差，必须与严格容差完整循环比较；参考默认仍是 `dfn_rtol=1e-10/dfn_atol=1e-11`。

`diffusivity_interpretation="chemical_diffusivity"` 配合 `local_mobility=true`，在核实 OCP 严格递减后，采用 `M=D_chem/(dmu_chem/dc)` 的对称边中点迁移率，包含实际 OCP 的温度和原生渐近项导数。平坦/非凸 OCP 不允许使用此换算，也不对曲率裁剪。默认 `"tracer_reference"` 保留原正迁移率候选。两种物理模型不能混作同一基线。PCHIP 在结点处仅 C1，局部二阶 Fick 极限测试不能扩展为所有结点上的二阶收敛证明。

非零导入原点现已通过机械网格与 VTI/VTU 往返测试。离子几何效率的图内计算显式使用 `ionic_reference_conductivity_S_m`（默认1 S/m），有效电导再除以同一参考值得到无量纲效率，实际宏观电解液 `kappa(c_e,T)` 不重复计入。

`ReactiveDualNetwork.changed_conductance_bound(state, electronic, ionic)` 在反应律、温度、电流及正导率边支持不变时，计算缓存解相对新图解的局部端电压误差上界。所有反应顶点必须保留。该方法检查全图边导率变化，求解保留空间的强单调性下界矩阵，不组装新的 Schur 补。返回 `voltage_absolute_bound_V`、边相对变化及数值残差项。边删除/创建或相对变化≥1时拒绝复用。它是解析上界的 FP64 计算，并非区间算术证明，也不是累计 DFN 容量或疲劳误差界。`solve(..., diagnostics=[])` 可记录 Newton/线搜索轨迹。

核查脚本 `benchmarks/study_reactive_updates.py` 将这个上界与独立更新的全图非线性解比较。测试包含电子浮动颗粒及0.5/8C；原失败涉及高导率网络能量相减造成的线搜索停滞，现使用等价的边电势差计算功率和残差，保持原1e-10求解容限。

可选的浓度依赖化学变形使用 `[cycle] chemical_stretch_table="materials/stretch.json"` 与 `chemical_stretch_sha256="..."`。文件位于外部工作区，JSON schema为 `morqex-chemical-stretch-1`，还须包含 `host`、严格递增的 `stoichiometry`、每行三个正值的 `principal_stretches`、`reference_stoichiometry=0.5` 和 `provenance`。输入为无量纲晶格长度比；工程应变应先转成 `1+strain`，不能直接传应变。归一化到初始参考态，允许各主方向非单调的晶格收缩/膨胀，以 PCHIP 插值并返回解析浓度导数。采用真实晶粒 SO(3) 取向，力学化学势使用同一个化学变形导数。禁止外推、负伸长和静默裁剪。选择此表时替代常系数指数膨胀；它是等温无历史的材料接口，不能代替相变滞回或疲劳标定。原默认数值参数不因此变成经过标定的 NMC 数据。

### Conservative nonlinear stack and voxel-contact policy

`[cycle] mechanical_constraint = "fixed_thickness_stack"` selects equal-area, fixed-total-thickness, plane-strain mechanics. The positive thickness cells, separator and lithium layer carry the same axial FIRST PIOLA stress; individual axial stretches are solved. Separator/positive thicknesses come from the same electrochemical parameters. `separator_young_Pa`, `lithium_young_Pa`, `lithium_thickness_m` and the declared Poisson ratio define explicit elastic numerical passive layers; lithium plasticity and separator viscoelasticity are not implied. `local_kubc` retains the earlier independent local clamp.

`morqex.top_damage.series_mechanics` provides `MicrostructureLayer`, `HomogeneousLayer`, `solve_fixed_thickness` and `finite_series_step`. The fixed-damage axial tangent comes from the exact discrete Schur complement. A joint frozen-history energy check and force/thickness checks precede a single fatigue-history commit. HDF5 stores `macro/axial_stretches`; record fields contain layer stresses, signed end tractions and residual normalization. The offline node HTML displays the layer mapping and current thickness.

`[cycle] separate_point_contacts = true` is the new candidate default: local voxel stars that touch only on an edge/corner no longer introduce an implicit FEM weld. True finite-area solid faces stay bonded. Material volumes are unchanged. This is an INITIAL mesh-connectivity policy, not simulated fracture. `TetMesh.from_volume(..., separate_point_contacts=True)` exposes the same choice; the old general FEM default remains false for backwards compatibility. Periodic identification of duplicate sheets requires an explicit mapping and is currently rejected. Old cycle checkpoints with another physical source identity cannot be resumed in place.

CPU reference-mesh studies may explicitly use `scripts/run_local.py --reference-memory`: RSS6GiB, virtual12GiB, one CPU, nice15 and minimum8GiB host reserve. Default runs retain3GiB RSS. The same406272-Tet4 static problem showed equivalent fields after storage optimization,56.51s vs65.14s and34.9% lower peakRSS. These are single-run static measurements, not whole-battery/lifetime acceleration. Fixed-geometry1/2/3/4/6refinement has exposed unresolved spatial errors; do not interpret the small cycling grid as a continuum-converged stress reference.

### Source-integrated conforming mesh refinement

`python -m morqex.top_damage build-refinement configs/top_damage_native.toml` compiles the small C++17 adapter against the hash-verified vendored MFEM library. Run through `scripts/run_local.py`; all build products stay external. `RefinementBackend(NativeConfig.from_toml(...), workspace).refine(mesh, marked_cell_indices)` invokes MFEM's conforming tetrahedral bisection including conformity closure, and returns `RefinementResult(mesh, parent_cells, prolongation)`. Physical material/voxel labels and material volumes are inherited by exact parent association. Distinct coincident sheets are not welded.

`result.nodal_field(old)` is nested P1 interpolation; `result.cell_field(old)` transfers piecewise constant internal variables. Positive barycentric weights preserve nodal bounds; cell-history volume integrals are preserved. These properties do not authorize coarsening, crack insertion, state extrapolation, or skipping equilibrium after refinement. `coupled_indicator(...)` uses first-Piola traction jumps and AT2 residuals to choose static refinement cells; it is explicitly an indicator, not a certified nonlinear or fatigue error bound. `benchmarks/study_adaptive_mesh.py --level N` stores each independent static solve and common-voxel **volume-weighted** fields externally. It uses zero fatigue and a fresh damage lower bound on each mesh so coarse-mesh irreversible errors are not locked into a convergence study.

`top_damage.quadratic_reference` adds a separately qualified reference candidate: `QuadraticGeometry`, `QuadraticHyperelasticity`, `QuadratureAT2`, and `quadratic_staggered_step`. Displacement has ten Tet10 nodes, damage retains four Tet4 nodes, and fatigue history resides at the declared positive MFEM quadrature points. All constitutive derivatives, damage driving terms and mechanical energy use that same quadrature. `geometry.quadrature_count` distinguishes integration states from physical elements; `cell_to_quadrature` and `quadrature_average` preserve material association. A low-order refined mesh supplies only the AMG preconditioner. Generic P1 cycle HDF5 fields must not be interpreted as quadratic fields; the static reference saves separate mechanical points and Tet10 connectivity. This is not yet the default cycling backend or a passed spatial convergence claim.


`ReactiveDualNetwork.changed_problem_bound(state,electronic,ionic,faces=...,temperature_K=...,discharge_current_A=...)` extends the local voltage check to explicitly supplied target OCP, j0, physical reaction coverage, T and I. Positive edge and reaction supports must stay unchanged. It accounts for the known kinetic/current residual and does not certify a predicted concentration or lifetime. Merely reusing old potentials would not solve new charge equations.

`with_reaction_state(faces,temperature_K=...)` returns a new problem sharing passive graph factorizations, while replacing reaction laws. `new.solve(I,initial=new.initial_from(old_state))` resolves BV currents and Kirchhoff constraints. A subsequent `new.changed_conductance_bound(...)` checks passive reuse against updated physical conductances. Support changes require reconstruction; neither API silently deletes inventory. These working-potential APIs are not automatically substituted into the transient DFN driver.

`solve(...,extended_precision=True)` retains extended-precision potentials/residuals but FP64 sparse tangents. On the tested Linux host the longdouble storage has64 significand bits, despite NumPy naming it float128; it is not IEEE binary128. Unsupported platforms reject the option. Tangent factorization, constitutive data and analytical bounds do not become interval-certified. `benchmarks/study_reactive_state_updates.py --with-rebinding` records24 material/topology/rate/state comparisons and both conservative rebound and independent full-graph timings.

`prolong_quadratic_field(source,target,parent_cells,field)` transfers nested P2 displacement guesses through exact mother-cell shape evaluation. `quadratic_indicator` adds within-cell response variation to a heuristic marking score; this is not a certified error estimator. The adaptive benchmark versions every branch and exposes its Dörfler fraction. Reducing a marking batch to respect memory does not qualify global convergence.

HTML generation: run `report_network_study.py`, `report_topology_counterexample.py`, `report_mesh_study.py`, then `report_manuscript.py` from benchmarks through the resource runner. Manuscript source stays at external `manuscript/top_damage_2s/MANUSCRIPT.md`; the export preserves its explicit unqualified-draft status. `scripts/check_study_reports.py` checks all four pages offline; `check_network_html.py REPORT.html` retains separate screenshots per report.

### Conservative event-triggered reaction graphs

`from morqex.top_damage import ReactionNetworkCache` provides a transactional
quasi-static update API. Construct with `voltage_tolerance_V`,
`relative_current_tolerance`, `solver_tolerance`, `extended_precision` and
`max_ports`; call `evaluate(electronic, ionic, faces, temperature_K=T,
discharge_current_A=I)`. Every query re-solves the BV currents. The returned
`ReactionUpdate` includes the accepted state, voltage and per-face current
bounds, event reason and rebuild/reuse counters. A failed target solve leaves
the previous accepted operator/state/counters unchanged.

Conductance changes are compared with the last factorization, not merely the
preceding query. Edge/reaction support changes force reconstruction. Reuse needs
both the voltage bound and the sum of absolute face-current bounds divided by
`max(abs(I),sum(abs(j)))` to meet their tolerances. Individual near-zero fluxes
are not used as relative-error denominators. `changed_problem_bound(...,
include_reaction_bounds=True)` exposes these local source bounds separately.
The source inventory integrator is still responsible for accumulating fluxes,
closed-system conservation and time errors: this API is not a cycle-jump or
transient DFN replacement.

The passive constant-potential null mode is enforced after checking the Schur
row-sum defect is below `1e-10` relative to the matrix scale. Residuals use the
physical incidence form so roundoff cannot create a ground leakage current.
Newton convergence checks both local KCL and global reaction-current balance.
`benchmarks/study_reaction_events.py` verifies 324 prescribed state queries
against fresh full graphs, including reaction/current and topology changes.
