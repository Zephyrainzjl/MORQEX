# Fixed-grid smoothing and adaptive FEM / 原尺寸体素平滑与自适应有限元

This workflow separates **geometry regularization**, **resolution reduction** and **finite-element density**. It does not replace the existing exact voxel, GEMsim/ACVD/TetGen, partitioned or MMG routes. All examples write outside the source directory.

本流程将**几何平滑**、**降低分辨率**与**有限元网格精细度**分开控制。保留原有精确体素、GEMsim/ACVD/TetGen、分块与 MMG 入口。全部输出必须位于源码目录之外。

The recommended default is practical: `balanced` smoothing and FEM precision,
phase-volume control and independent positive-volume/shape checks. It does not
require voxel-for-voxel identity, identical component counts, or an excessively
fine mesh. `topology_policy='report'` records connectivity changes by default;
the stricter topology option is an explicit choice. Here `acceptance='strict'`
means enforcing the user's chosen tolerances, not demanding exact geometry.

推荐默认使用中等平滑和中等网格精度，控制各相体积变化，并独立检查正体积与单元形状。
不要求逐体素相同、连通分量数完全相同或极密网格。默认 `topology_policy='report'`
记录连通性变化，严格拓扑约束需要主动选择。`acceptance='strict'`只表示执行用户设定的
容差，不表示要求几何完全一致。小相的相对体积误差与其占整个结构的绝对体积分数变化
应一起判断；这类几何误差不能冒充电化学锂质量守恒误差。

## 1. Three independent operations / 三种独立操作

|Operation / 操作|Array / 数组|Spacing / 体素间距|Physical box / 物理范围|Purpose / 用途|
|---|---|---|---|---|
|`smooth_voxels`|Unchanged ZYX shape / 原尺寸|Unchanged / 不变|Unchanged / 不变|Regularize small islands, protrusions and rough boundaries / 减少小岛、细尖突起和锯齿|
|`lower_resolution`|Explicitly fewer samples / 明确减少体素数|Increases / 增大|Unchanged / 不变|Reduce sampling cost / 降低采样规模|
|`implicit_mesh` on one `PreparedGeometry`|Input untouched / 输入不变|Input untouched / 不变|Same target box / 同一目标范围|Change FEM density independently / 独立改变有限元密度|

For example, smoothing a256³ source at a2-voxel feature scale still produces256³ labels and256³ continuous geometry scores per phase. Reducing it to128³ is a separate request, with twice the spacing. Both describe the same physical observation box. A coarse/fine mesh pair may use **exactly the same** saved prepared geometry.

例如256³体素按2体素尺度平滑后，仍是256³标签，每相的连续几何评分场也是256³。降低到128³必须另行调用，间距变为原来的两倍，物理范围不变。同一份已保存的平滑几何可以生成粗、细两套有限元网格，期间不再次平滑。

`minimum_feature_voxels=1..4` is an **operation scale**, not a proved hard lower bound on every neck. A preserved thin material may still be thinner, and digitized curved boundaries always contain voxel stairs. The report and feature analysis provide measured changes. Do not call geometric smoothing an improvement of CT measurement resolution.

`minimum_feature_voxels=1..4`表示**规则化尺度**，不保证每个颈部都有这个严格最小宽度。需要保留的薄相可能更细，曲面离散后也仍有体素台阶。请查看实际特征与变化报告；平滑不增加 CT 的测量分辨率。

## 2. Optional source compilation / 可选源码编译

The optional backend integrates unmodified **CGAL6.0.1 Mesh_3** and its required CGAL headers. It uses Delaunay refinement on a continuous multiphase labeling function, followed by optional sliver optimization. This is a mature upstream mesher with a separate MORQEX geometry/API adapter. It does not use a neural model. It is **not TetGen**; TetGen remains available through its existing APIs.

可选后端集成未修改的 **CGAL6.0.1 Mesh_3** 及所需 CGAL 头文件，在连续多相标签域上进行 Delaunay 加密和薄片单元优化。上游算法与 MORQEX 的几何/API 适配分开。无需神经网络。该入口**不是 TetGen**；原有 TetGen 接口继续保留。

Install Python dependencies and explicitly compile with one compiler job:

先安装 Python 依赖，再明确以单任务编译：

```bash
cd /home/zephyrain/lw/MORQEX
WORK=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
PY=$WORK/runtime/bin/python
MORQEX_WORKSPACE="$WORK" "$PY" -m pip install '.[implicit-meshing]'
# Required: CMake, C++17, Boost headers, Eigen3, GMP and MPFR development files.
# 必需：CMake、C++17、Boost/Eigen3/GMP/MPFR 开发头文件及运行库。
"$PY" -m morqex.modeling build-cgal --workspace "$WORK" \
  --dependency-prefix /external/development/prefix/usr
```

A prefix is optional when the development dependencies already exist in standard locations. No system packages or network downloads are silently installed. The build, mirror, library, identity and registry stay under the external workspace.

若依赖已在标准位置，省略 `--dependency-prefix`。此 API 不会静默安装系统包或下载依赖。构建目录、源码镜像、库、来源标识和注册表全部位于外部工作区。

The same explicit build can be selected in a separate TOML file:

也可以通过独立 TOML 文件选择编译；只列出实际需要的后端：

```toml
[native_build]
workspace = "/home/zephyrain/lw/morph_workspace/microelectrode_gallery"
backends = ["tetgen", "acvd", "mmg", "cgal"]
cgal_dependency_prefix = "/external/development/prefix/usr"
```

```bash
"$PY" -m morqex.modeling build-backends /external/native_backends.toml
```

`cgal_dependency_prefix` is optional with standard development installations.
The Python equivalent is `build_native_from_toml(path, workspace=WORK)`;
neither reading a geometry recipe nor importing MORQEX triggers compilation.

开发依赖已在标准位置时可以省略 `cgal_dependency_prefix`。Python 接口为
`build_native_from_toml(path, workspace=WORK)`；读取几何配置和导入 MORQEX
均不触发隐式编译。

For a non-editable wheel with the compiled Python extension:

将编译扩展装进非 editable wheel：

```bash
MORQEX_WORKSPACE="$WORK" \
MORQEX_CGAL_DEPENDENCY_PREFIX=/external/development/prefix/usr \
MORQEX_BUILD_CGAL=1 "$PY" -m pip install --no-build-isolation '.[implicit-meshing]'
```

`MORQEX_BUILD_CGAL=1` and `MORQEX_BUILD_TETGEN=1` are independent opt-ins and can be enabled together. Installing an extra alone installs Python dependencies; it does not silently compile C++. Native editable installation is rejected because it would place binaries in `src`. Use the explicit external build for development.

CGAL 与 TetGen 的编译开关独立，也可以同时启用。仅安装 extra 不会自动编译 C++。禁止原生 editable 构建向 `src` 写入二进制；开发阶段使用显式外部构建。

Mesh_3 is GPL-3.0-or-later or commercially licensed. The native adapter is GPL-3.0-or-later. All per-header upstream notices and file hashes are retained in `src/morqex/vendor/cgal`. Other integrated frameworks retain their own licenses. See `THIRD_PARTY_NOTICES.txt`; local integration does not establish permission to redistribute every combined component under one license. [CGAL licensing](https://doc.cgal.org/6.0.1/Manual/license.html).

Mesh_3 使用 GPL-3.0-or-later 或商业授权；本原生适配器使用 GPL-3.0-or-later。上游每文件许可与 hash 保留。其他框架许可不变，不能据本地集成推断整个组合可统一再发布。

## 3. Load complete geometry and smooth / 读取完整几何并平滑

```python
from pathlib import Path
import numpy as np
from morqex.modeling import (
    VoxelModel, from_regions, tiff_voxels, generate_voxels,
    SmoothingOptions, smooth_voxels, PreparedGeometry,
    ProgressBar, NotebookProgress, HTMLProgress,
)

work = Path('/home/zephyrain/lw/morph_workspace/microelectrode_gallery')
# Original HDF5 retains spacing, origin, labels and names.
# 原始 HDF5 保留间距、原点、标签和名称。
raw = VoxelModel.load(work / 'nmc_complex_gallery/nmc_tiff/geometry.h5')

# Or import your complete ZYX array (nonnegative integer labels).
# 或导入完整 ZYX 数组：标签为非负整数，0也是有效相。
# raw = from_regions(array, spacing=(0.5e-6,)*3,
#                    origin=(0.,0.,0.), region_names={0:'phase_0',128:'phase_128',255:'phase_255'})

# Notebook-only progress; use ProgressBar() in scripts.
# Notebook 中使用，脚本改用 ProgressBar()。
progress = NotebookProgress()
settings = SmoothingOptions(
    method='gaussian', iterations=2,
    minimum_feature_voxels=2.0,
    minimum_component_voxels=8,
    sigma_voxels=0.7,
    preserve_phase_volumes=True,
    volume_balance_tolerance=0.01,
    maximum_changed_fraction=0.12,
    maximum_phase_volume_error=0.12,
    minimum_phase_iou=0.65,
    topology_policy='report', acceptance='strict',
)
prepared = smooth_voxels(raw, options=settings, progress=progress)
assert prepared.model.regions.shape == raw.regions.shape
assert prepared.model.spacing == raw.spacing
assert prepared.model.origin == raw.origin
prepared.save(work / 'my_geometry/prepared.h5', progress=progress)
```

Do not assign NMC TIFF grayscale values to active material/CBD/pore without segmentation metadata. The supplied TIFF and COMSOL mesh are independent geometries. The example0.5µm spacing is an explicit modeling choice permitted by the owner, not scan calibration. `origin` and `spacing` are ZYX SI metres; output mesh points are XYZ.

没有分割元数据时，不擅自把 NMC 灰度指定为活性相/CBD/孔隙。给定 TIFF 与 COMSOL 网格是独立结构。示例0.5µm是经所有者允许的建模选择，不是扫描标定。体素原点和间距按 ZYX、SI 米；有限元点按 XYZ。

### Smoothing schemes / 平滑方案

|`method`|Meaning / 含义|Important qualification / 限制|
|---|---|---|
|`none`|Identity labels, one-hot scores / 原标签与 one-hot 评分|No component cleanup or capacity adjustment / 不清理小岛、不调体积|
|`gaussian`|Gaussian-filter each phase indicator / 各相指示函数高斯滤波|Scores compete; phase-volume shifts may alter boundaries / 体积约束也会移动边界|
|`majority`|Box-averaged phase competition / 邻域多数竞争|Thin phases can change; inspect report / 检查薄相变化|
|`signed_distance`|Smooth signed physical distance per phase / 各相物理有符号距离平滑|More memory/time; not universally superior / 更耗资源，不保证更优|
|`morphology`|Opening/closing then Gaussian competition / 开闭运算后连续竞争|Aggressive on small features; fidelity gate can reject / 对小特征更强，可能被保真门槛拒绝|

`iterations` is1–8. Each iteration performs optional small-component regularization, score construction and phase-capacity balancing against the **original** phase counts. `minimum_component_voxels=None` uses approximately `minimum_feature_voxels**3`; set0 to disable. The largest face-connected component of each phase is retained. Selected small components are reassigned to the nearest retained other material. This is a declared geometry modification, **not evidence that those components were imaging noise**.

迭代次数为1–8。每次进行可选小连通域规则化、评分场构造、针对**原始**各相体积的约束。`minimum_component_voxels=None`使用约 `minimum_feature_voxels**3`，设0关闭。保留各相最大的面连通分量，将被选小分量改派到最近的保留异相。这是明确的几何修改，不代表这些小分量已被证实为扫描噪声。

`sigma_voxels` may be scalar or a ZYX tuple. `sigma_m` specifies the Gaussian width in SI metres, also scalar or ZYX; it is mutually exclusive with `sigma_voxels`. On anisotropic voxels these are different controls. The target feature scale is advisory; physical Gaussian width is the more precise control.

`sigma_voxels`支持标量或 ZYX 三元组；`sigma_m`指定物理高斯宽度，也支持两种形式，二者互斥。各向异性体素下含义不同。目标特征尺度用于指导，物理高斯宽度是更明确的控制。

```python
physical = SmoothingOptions(
    sigma_m=(0.25e-6, 0.35e-6, 0.50e-6),
    minimum_component_voxels=0, iterations=1,
)
# Presets alter smoothing only when you explicitly choose them.
# 只有明确选择时，这些预设才改变平滑设置。
fine_smoothing = SmoothingOptions.precision('fine')
coarse_smoothing = SmoothingOptions.precision('coarse')
```

Phase-capacity balancing shifts competing scores using bounded coordinate/quantile sweeps. It approximately conserves **hard voxel counts**, within the recorded tolerance; it does not prove exact continuous-domain volume conservation. Mesh phase volumes are checked independently afterwards. Geometry scores are not chemical concentrations or calibrated phase probabilities.

分相体积约束采用有上限的坐标/分位点迭代，按记录容差近似保持**硬体素标签数量**，不保证连续几何的严格体积守恒。生成网格后还会独立检查体积。几何评分不是锂浓度，也不是标定概率。

### Fidelity and topology / 保真度与拓扑

The report records changed fraction, per-phase IoU/volume error, component counts, actual face-contact areas and added/removed contact pairs. `topology_policy='report'` allows **reported** changes; `preserve_components_and_contacts` rejects changes in these checked descriptors. Matching descriptors are not a proof that all homology or geometric topology is identical.

报告记录改变比例、各相 IoU/体积误差、连通域数量、真实面接触面积与新增/消失接触。`report`允许记录清楚的变化；`preserve_components_and_contacts`拒绝这些已检查指标的变化。指标相同不构成完整拓扑等价证明。

`acceptance='strict'` raises when explicit limits fail. `acceptance='report'` is useful for comparing candidates; it is **not acceptance**. The mesher rejects a failed preparation by default. Source arrays are never modified. Original and prepared hashes are stored; editing saved scores/labels invalidates the preparation and requires rebuilding.

`strict`在不通过时抛错。`report`适合比较候选，但不表示合格；网格器默认拒绝这种未通过的准备结果。原数组不被修改；修改保存后的标签或评分会使 hash 检查失败，必须重新准备。

## 4. Feature-aware sizes and same-geometry precision / 特征分析与同一几何调精度

```python
from morqex.modeling import recommend_mesh_sizes, implicit_mesh, ImplicitMeshOptions
advice = recommend_mesh_sizes(prepared, progress=progress)
print(advice['recommended_sizes'])
print(advice['regions'])

meshes = {}
for precision in ('coarse', 'balanced', 'fine'):
    meshes[precision] = implicit_mesh(
        prepared,  # same saved scores, no new smoothing / 同一评分场，不再平滑
        options=ImplicitMeshOptions(precision=precision),
        workspace=work, progress=progress,
    )
    assert meshes[precision].report['geometry_score_sha256'] == prepared.report['score_sha256']
```

Feature analysis uses actual face connectivity and EDT local-maxima thickness proxies with explicit voxel-scale uncertainty. The broad bulk sizing range is **advice**, not a uniform safe size that may erase thin material. A coarse bulk region can coexist with fine material interfaces. Requesting less than one voxel does not manufacture subvoxel measurement information.

特征分析采用真实面连通性与 EDT 局部极大处的厚度代理量，并报告体素尺度不确定性。推荐的较宽体网格范围是**建议**，不表示可用同样大的均匀网格抹掉薄相。体内部可以较粗，界面仍需较细。小于体素的网格不会创造亚体素测量信息。

`surface_size_m`, `bulk_size_m`, `outer_surface_size_m` and `facet_distance_m` are independent SI requests. CGAL sizing controls are not strict maximum edge bounds or exact element counts. `grading='distance'` creates a gradual size field near interfaces. `grading='surface_driven'` uses the bulk field and lets surface constraints drive refinement. A positive `size_field` array on the complete ZYX grid, or `callback(xyz_points)->SI_sizes`, overrides the volume sizing field.

四种尺寸控制分别为内部界面、体内部、外盒表面和几何近似距离，单位都是米。它们不是严格最大边长或指定单元数量。`distance`按界面距离分级；`surface_driven`由界面约束驱动局部加密。也可以提供完整 ZYX 尺寸数组，或输入 XYZ 点输出 SI 尺寸的回调。

```python
def local_size(xyz):
    # Fine near a chosen physical region, coarser elsewhere.
    # 指定物理区域附近更细。
    centre = np.array([30e-6, 30e-6, 30e-6])
    distance = np.linalg.norm(xyz-centre, axis=1)
    return np.clip(1.2e-6 + 0.2*distance, 1.2e-6, 6e-6)

controls = ImplicitMeshOptions(
    surface_size_m=1.5e-6, bulk_size_m=5e-6,
    outer_surface_size_m=5e-6, facet_distance_m=0.25e-6,
    size_field=local_size,
    radius_edge_limit=2.0, minimum_dihedral_degrees=3.0,
    optimization='joint', optimization_target_degrees=5.0,
    optimization_seconds=20.0,
    maximum_phase_volume_error=0.08,
    maximum_box_volume_error=0.03,
    maximum_nodes=1_500_000, maximum_cells=8_000_000,
    timeout_seconds=300, isolated=True,
)
mesh = implicit_mesh(prepared, options=controls, workspace=work, progress=progress)
```

The `joint` adapter uses CGAL's sliver perturber with a score that penalizes both a small minimum dihedral and an excessive circumradius/shortest-edge ratio. Independent FP64 calculations verify both after optimization. Other options are `none`, `perturb`, `exude`, `perturb_exude`, `joint_exude`; none bypasses acceptance unless explicitly setting `require_quality=False`, which produces a clearly recorded exploratory result. The production default requires quality. A bounded retry can tighten refinement and extend optimization; it cannot guarantee success for every geometric singularity.

`joint`利用 CGAL 的薄片扰动器，同时惩罚小二面角与过大外接半径/最短边比；优化后用独立 FP64 计算验收。其他优化器可选，但不能默认绕过验收。`require_quality=False`仅用于明确记录的探索结果。有限次重试会收紧加密、延长优化，不保证任意几何奇异点都可通过。

A rejected mesh is not repaired by deleting bad tetrahedra, dropping a material, or hiding failed statistics. A tight resource budget can reject an otherwise valid geometry. Revise the explicit smoothing/density choice, or increase an appropriate external budget while retaining host headroom. An unprotected sharp multiphase junction can require more refinement than expected; the current score-domain path does not claim exact preservation of every0D/1D feature.

不允许靠删除坏单元、丢相或隐藏统计“修好”网格。资源预算也可能拒绝合理几何。应调整明确的平滑/尺寸选项，或在保留主机余量的前提下修改预算。未显式保护的尖锐多相交线可能引发较多加密；本评分域路线不承诺精确保留所有零维/一维特征。

## 5. Explicit resolution reduction / 明确降低分辨率

```python
from morqex.modeling import lower_resolution
reduced, reduction_report = lower_resolution(
    raw, shape=(128,128,128), method='area',
    allow_phase_loss=False, progress=progress,
)
# Or factor=2.0, instead of shape. / 或指定 factor，与 shape 二选一。
assert reduced.regions.shape == (128,128,128)
np.testing.assert_allclose(
    np.array(raw.regions.shape)*raw.spacing,
    np.array(reduced.regions.shape)*reduced.spacing,
)
reduced.save(work/'my_geometry/reduced128.h5')
```

`area` computes exact separable cell-overlap averages before choosing labels. Integrated **soft phase fractions** are conserved up to floating point; hard labels need not preserve fractions or connectivity. `nearest` uses the nearest source cell. Phase disappearance raises by default. Neither scheme is smoothing, even when the coarser display looks smoother.

`area`先按精确可分离体素重叠比例平均，再选择硬标签；**软相分数**在浮点误差内守恒，但硬标签体积和连通性不一定守恒。`nearest`取最近原体素。默认禁止相消失。这两种都是降低分辨率，不因视觉更光滑而称为平滑。

## 6. Notebook and offline HTML / Notebook 与离线 HTML

```python
from IPython.display import display
# Viewer controls: phase visibility, exact full-grid orthogonal slices and
# labelled whole-3D LOD. / 控制相显隐、完整体素切片及有标注的三维显示LOD。
display(raw.viewer(maximum_3d_size=64, slice_origin='lower'))
display(prepared.model.viewer(maximum_3d_size=64, slice_origin='lower'))
raw.viewer(maximum_3d_size=64).save_html(work/'my_geometry/raw.html')
prepared.model.viewer(maximum_3d_size=64).save_html(work/'my_geometry/smooth.html')
artifact = mesh.save_hdf5(work/'my_geometry/mesh.h5')
display(artifact.adaptive_viewer(maximum_embedded_faces=350_000))
artifact.adaptive_viewer(maximum_embedded_faces=350_000).save_html(work/'my_geometry/mesh.html')
```

Changing `maximum_3d_size` or embedded-face limits affects visualization only; it never changes the complete computational geometry. The adaptive mesh viewer uses actual Tet4 intersections for local slices. HDF5/XDMF retains all elements. Keep the HDF5 and XDMF together for ParaView.

显示 LOD 或嵌入面数只影响显示，不改变完整计算结构。自适应网格的局部切片与真实 Tet4 相交。HDF5/XDMF保留全部单元，ParaView读取时二者需放在一起。

A live **small** progress page avoids reloading the full geometry every two seconds:

小型进度页面避免反复重新加载大三维结构：

```python
live = HTMLProgress(work/'my_geometry/progress.html')
bar = ProgressBar()
def update(event):
    live(event)
    bar(event)
try:
    p = smooth_voxels(raw, options=settings, progress=update)
    m = implicit_mesh(p, workspace=work, progress=update)
except BaseException as error:
    live.finish(message='FAILED / 失败: '+str(error))
    raise
else:
    live.finish(message='Complete / 完成；检查独立质量报告')
```

Known loops show measured work fractions. Native operations with unknown internal counts show elapsed time and an indeterminate bar; ETA uses an explicit heuristic or external rolling throughput history and is never a promise. Native callbacks report actual counts at completed stages. Failure is visible and stops page refresh. No fabricated100% progress is used while native work continues.

已知循环显示实际完成比例，未知内部工作量的原生计算显示不定进度和耗时。ETA来自明确的启发式或外部历史吞吐率，不是承诺。原生阶段结束时报告真实点/单元数。失败可见并停止刷新；不会在仍计算时伪造100%。

## 7. TOML, checkpoint and one-command workflow / TOML、断点与统一命令

Use [the example recipe](../configs/geometry_implicit.toml). `from_toml` reads/generates the **raw** model; `prepare_from_toml` explicitly applies `[preparation]`; `mesh_from_toml` accepts either a raw or prepared model. A prepared model is never automatically resmoothed.

使用[示例配置](../configs/geometry_implicit.toml)。`from_toml`生成/读取**原始**体素，`prepare_from_toml`明确应用准备设置；`mesh_from_toml`接受原始或准备后的结构，已准备结构不被自动重复平滑。

`implicit_mesh(raw_model, options=...)` also keeps the controls independent:
without an explicit `smoothing=...`, preparation is always balanced, regardless
of FEM precision. Use `smoothing=SmoothingOptions.precision('coarse')` only when
you intentionally want stronger smoothing. Saving `PreparedGeometry` is the
recommended way to reuse the exact same geometry for several mesh densities.

直接调用 `implicit_mesh(raw_model, options=...)` 时也分别控制：不提供
`smoothing=...` 就始终使用中等平滑，不随有限元粗细变化。只有希望更强平滑时才明确
设置 `smoothing=SmoothingOptions.precision('coarse')`。多种网格密度共用同一几何时，
推荐先保存 `PreparedGeometry`，再重复网格化。

```python
from morqex.modeling import run_geometry
run = run_geometry(
    '/home/zephyrain/lw/MORQEX/configs/geometry_implicit.toml',
    work/'my_geometry/runs', workspace=work,
)
print(run.directory, run.html)
```

```bash
python -m morqex.modeling run-geometry configs/geometry_implicit.toml \
  --directory "$WORK/my_geometry/runs" --workspace "$WORK"
# Separate commands / 分步命令
python -m morqex.modeling prepare "$WORK/source.h5" \
  --config configs/geometry_implicit.toml --output "$WORK/prepared.h5" \
  --progress-html "$WORK/preparation_progress.html"
python -m morqex.modeling features "$WORK/source.h5" --output "$WORK/features.json"
python -m morqex.modeling implicit-mesh "$WORK/prepared.h5" --precision balanced \
  --workspace "$WORK" --output "$WORK/mesh.h5" --preview "$WORK/mesh.html"
python -m morqex.modeling lower-resolution "$WORK/source.h5" \
  --factor 2 --output "$WORK/reduced.h5" --preview "$WORK/reduced.html"
```

`run_geometry` hashes the recipe, complete generated source and algorithm sources. Changed inputs/settings receive a new directory automatically. Valid prepared and mesh checkpoints are reused; incomplete artifacts are quarantined under that run's `rejected` directory. Export/preview interruption does not force remeshing. It creates source/prepared/mesh HTML, a small live progress page and a measured validation report. The source-generation time estimate is approximate, especially when importing a file rather than synthesizing geometry.

统一流程对配置、完整原始结构和算法源码建立 hash。改变输入/选项会自动进入新目录。有效的准备/网格断点直接复用，不完整输出移入该次运行的 `rejected`。导出/绘图中断后不必重建网格。流程生成原始/平滑/网格 HTML、独立进度 HTML和实测报告。读取文件时的初始生成 ETA尤其只是近似。

## 8. Software exchange and verification / 软件交换与验证

```python
from morqex.modeling import export_bundle
bundle = export_bundle(
    artifact, work/'my_geometry/software',
    formats=['comsol','vtu','gmsh','medit','abaqus'],
    region_names=prepared.model.region_names, length_unit='um',
)
assert bundle['passed']
```

All materials including label0 are real domains. Interface tags are recovered from actual neighboring tetrahedra, not an arbitrary surface coloring. Domain names and physical units follow the existing exchange contract. COMSOL numeric coordinates are scaled to the requested unit; choose the same unit during import. A successful readback certifies the checked topology/coordinates/labels, not correctness of a downstream software's physics setup.

包括0标签的全部材料都是实际分域。界面标记来自真实相邻四面体，不是表面随意着色。COMSOL坐标按指定单位缩放，导入时使用相同单位。回读合格只验证已检查的网格、坐标和标签，不保证下游物理场设置正确。

The preserved exact staircase route is documented in [07](07_complex_ct_layers_and_exchange.md), and original weighted ACVD/TetGen meshing in [06](06_gemsim_adaptive_meshing.md). Use the exact route when exact voxel volumes and source-cell ownership are essential; use the smooth route when explicit geometric approximation is acceptable. Independent shape/volume, conduction patch and coarse/fine tests are required before using a new mesh for a scientific conclusion. Consult [CGAL's method documentation](https://doc.cgal.org/6.0.1/Mesh_3/index.html) for the underlying mesher assumptions.

必须严格保持体素体积和源体素归属时使用保留的精确台阶路线；允许明确几何近似时使用平滑路线。新网格用于科学结论前，仍需独立形状/体积、传导补丁与粗细收敛验证。不能仅凭“能生成”宣称达到论文标准。

### Optional distance reinitialization / 可选距离场重构

`SmoothingOptions(reconstruction='signed_distance')` first performs the selected
fixed-grid smoothing and capacity balancing, then reconstructs signed physical
distance scores from the **final hard labels**. `filtered_scores` keeps the
filtered/balanced scores instead. This independent choice changes continuous
subvoxel geometry, **not** the final label array, its dimensions or spacing.
It can avoid interpreting a tiny score margin as a measured narrow geometric
feature; it is not a proof that all junctions are repaired. The mesh phase
volumes and shape criteria must still pass independently. Mesh controls also
expose `surface_minimum_angle_degrees` in[0,30):0 disables only that surface
refinement criterion; it does not disable the Tet4 shape acceptance criteria.

`reconstruction='signed_distance'`在平滑与分相体积约束后，按**最终标签**重建物理
有符号距离评分；`filtered_scores`保留滤波评分。这个独立选项只改变亚体素连续几何，
不改变最终标签、数组大小或体素间距。它不保证修好所有交汇，仍须独立检查网格相体积
与单元形状。`surface_minimum_angle_degrees=0`只关闭表面三角形角度加密条件，
不关闭体四面体质量验收。

### Independent geometric checks and bounded refinement / 独立几何检查与有限次加密

A passing voxel-count check is not a passing mesh-volume check. Thin materials
can be underestimated by a coarse polygonal FEM interface even when the
continuous source geometry has the right volume. The mesher records both shape
and phase/box volume errors for every attempt. A bounded automatic retry halves
the interface approximation distance on a geometry-volume failure and tightens
the native radius criterion on a shape failure. It keeps the same full geometry
and unchanged acceptance limits. A budget/timeout failure still rejects the
run; no material or bad element is silently removed.

体素相数量合格不等于有限元相体积合格。连续几何体积合理时，较粗的多面体界面仍可能
低估薄相体积。每次尝试都记录形状与分相/整体体积误差；有限次重试在几何体积失败时
将界面近似距离减半，在单元形状失败时收紧原生半径准则，保留完整几何和原验收门槛。
资源或超时失败仍拒绝输出，不丢相、不删坏单元。

`oracle_tolerance_voxels` controls the absolute interface-intersection oracle
accuracy in units of the minimum voxel spacing. The implementation converts it
to CGAL6.0.1's relative bounding-box convention and caps it below one tenth of
the actual facet-distance request. The default0.01 voxel does not imply0.01
voxel CT measurement accuracy; it is a numerical geometry-query tolerance.
[CGAL oracle convention](https://doc.cgal.org/6.0.1/Mesh_3/classCGAL_1_1Labeled__mesh__domain__3.html).

此容差仅控制几何查询的数值精度，不代表 CT 测量精度。其实际 SI 值与 CGAL 相对值
均写入报告；界面加密后也同步收紧查询容差。

An optional `continuous_volume_calibration=True` in `SmoothingOptions` estimates
continuous phase volumes with scrambled Sobol points and adjusts nonnegative
nodal indicator gains without changing a final hard label. A second scramble
checks the result. `calibration_sample_power`, `calibration_iterations`,
`calibration_relative_tolerance`, and `calibration_seed` are explicit controls.
This is **sampled geometric calibration, not certified conservation**. Complete
voxel fields still enter the mesher; the points only estimate an integral. True
mesh phase volumes are checked separately. It cannot fix an underresolved FEM
interface by itself and remains an explicit optional experiment.

可选连续体积标定通过 scrambled Sobol 点估计积分，并调整非负的节点相指示增益，
不改变最终硬标签。独立扰乱序列用于复核，所有采样参数可设。它是**采样几何标定，
不是严格守恒证明**；并未降低体素分辨率。网格仍读取完整场，仍须检查真实单元体积。

Final small-component cleanup is enabled by default after the smoothing loop,
because thresholding a smooth field can itself create a new tiny isolated
piece. The same declared component-volume floor applies; changed nodes receive
scores from the nearest retained destination material, and volume/IoU/contact
checks run afterwards. `final_component_cleanup=False` disables that last
operation; `minimum_component_voxels=0` disables component cleanup throughout.
The report records final cleanup separately. Numeric ties in a nodal score are
resolved consistently with the stored hard label using a minimal floating-point
increment, with the number of corrections recorded.

默认在平滑循环结束后再检查一次小分量，因为阈值化本身可能产生新的小孤岛。
仍使用显式声明的连通域体积下限；改派位置的评分取最近保留目标材料的评分，
之后重新做体积、IoU和接触检查。可以单独关闭最终清理，或将体积下限设0关闭
整个清理过程。报告分别记录最终清理与数值平局一致化次数。

The default bounded runner remains one low-priority CPU and6GiB RSS for
`--reference-memory`. A particularly fine full-domain mesh may explicitly use
`--reference-memory --reference-rss-mib 7168`, while preserving the8GiB host
reserve, one CPU, time limit and GPU-off policy. It is never enabled implicitly.
The supervisor stops the job if other applications consume the reserved memory.

默认仍为单个低优先级CPU、6GiB参考RSS预算。特别精细的整域网格可显式申请7GiB，
但8GiB主机余量、单核、时间与GPU关闭策略不变；其他应用占用余量时会停止计算。

The `joint` optimizer can now reuse its existing triangulation for bounded
local refinement and another optimization sweep. CGAL's own documentation
notes that optimization can violate refinement criteria; the local cycle
restores them and checks both metrics again. `maximum_local_repair_cycles`
(default2,0 disables) controls this mechanism. `optimization_seconds` is a
**per-sweep** limit; `timeout_seconds` and the external runner limit the whole
native call. Native initial/final quality, repair count and optimizer return
code are saved alongside independent NumPy quality measurements. This is
reuse of established CGAL refinement, not a new meshing-theory claim.

`joint`可以复用已有三角剖分，有限次局部再加密并再优化。CGAL文档已指出优化可能
违反加密准则；因此需要重新检查，而不是认为一次优化就一定完成。
`maximum_local_repair_cycles`默认2，设0关闭。优化时间是**每次扫描**的上限，
整个原生调用仍受超时与外部预算约束。原生初始/最终指标、局部更新次数和返回码
与独立NumPy验收结果同时保存，不将既有CGAL加密思想宣称为新理论。

## Local resource-bounded execution / 本机限资源执行

Run large arrays through the existing supervisor. All processes, including
native/compiler descendants, share its single low-priority CPU budget; the GPU
stays disabled for this meshing workflow. The complete resource/source manifest
is saved outside source. Notebook kernels used for qualification inherit the
same limits. To use a normal notebook interactively, start its kernel with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before importing NumPy.
CGAL itself is compiled with `Sequential_tag` and one compiler job.

大数组请通过现有监督器启动，原生/编译子进程也纳入同一单核低优先级预算。
本几何流程不占用GPU，资源及源码清单保存在外部。验证Notebook继承相同限制；
普通Notebook应在导入NumPy前设置线程环境变量。CGAL采用顺序执行类型。

```bash
python scripts/run_local.py --workspace "$WORK" --label geometry_run \
  --reference-memory --large-mesh-output -- \
  "$PY" -m morqex.modeling run-geometry /home/zephyrain/lw/MORQEX/configs/geometry_implicit.toml \
  --directory "$WORK/my_geometry/runs" --workspace "$WORK"
```

|Smoothing preset / 平滑预设|Feature scale / 特征尺度[vox]|Gaussian sigma[vox]|Iterations / 迭代|
|---|---|---|---|
|very_fine|1|0.25|1|
|fine|1.5|0.5|1|
|balanced|2|0.7|2|
|coarse|4|1.4|3|

These presets all keep the original array shape and physical size. They are
independent of FEM precision presets. Fixed collector/separator geometry should
use the unchanged-label or exact PLC route when movement of those interfaces is
not permitted; global smoothing does not know their physical role automatically.

以上平滑预设都保留原数组及物理尺寸，与有限元精度预设独立。若集流体/隔膜边界
不允许移动，应选择原标签或精确PLC路线；全局平滑不会自动知道材料的物理角色。


`joint_exude` adds CGAL sliver exudation **after** all unweighted joint/refinement
sweeps. No perturbation or refinement follows exudation. Weights are used only
to choose the tetrahedral connectivity, not as downstream FEM material data.
Exudation can improve small dihedrals while worsening radius ratios; independent
physical Tet4 checks remain mandatory and can reject the result. This mode is
optional and never silently replaces the default `joint` mode.

`joint_exude`在全部无权联合优化/再加密**之后**追加CGAL薄片剔除，之后不再扰动或
加密。权重仅用于构造四面体连接关系，不作为下游有限元物性。该过程可能改善小二面角
却恶化半径边比，所以必须重新独立验收，失败仍拒绝；不静默替换默认联合模式。

## Full supplied NMC recipe / 用户完整NMC体素配置

[geometry_nmc_implicit.toml](../configs/geometry_nmc_implicit.toml) reads the
entire supplied TIFF, preserving its252×252×253 array and original0/128/255
labels. It explicitly chooses the stored array order and0.5µm spacing; neither
phase roles nor scan calibration are inferred. This TIFF and the independent
1m COMSOL reference are not registered to one another.

该配置读取完整252×252×253体素和原始0/128/255标签，不截取小区域、不降分辨率。
数组轴顺序与0.5µm间距为明确建模选择，不冒充扫描标定；不臆定三种灰度的材料含义。
此TIFF与另外提供的1m COMSOL参考结构不是对应关系。

```bash
python scripts/run_local.py --workspace "$WORK" --label nmc_implicit \
  --reference-memory --reference-rss-mib 7168 --large-mesh-output -- \
  "$PY" -m morqex.modeling run-geometry \
  /home/zephyrain/lw/MORQEX/configs/geometry_nmc_implicit.toml \
  --directory "$WORK/nmc_reproduction" --workspace "$WORK"
```

The saved, independently checked full NMC mesh has1,717,340 nodes and
10,478,936 tetrahedra: max radius/edge1.95997, minimum dihedral3.25003° and
fifth-percentile dihedral17.1519°. Its physical-box volume error is0.0858%;
relative phase-volume differences from prepared voxels are1.187%,0.285%,7.160%
for0,128,255. These are measured geometric approximation errors, not certified
physical-solution errors. The mesh API took304.4s with the prepared input;
preparation, saving and five-format export are separate costs. This meshing
qualification used one low-priority CPU, no GPU, and sampled about4.76GiB peak
process-group RSS including its supervisor-tracked descendants.

完整NMC网格上述实测指标均保留在外部报告中。最小角之外也报告角度分布；不把所有
单元都说成具有17°二面角。三种相体积误差针对准备后的体素，原始到平滑的变化另存。
网格API时间不含平滑和导出，不能作为完整电池求解时间；采样峰值内存也不是数学上界。
默认精度以实用性为目标，不要求消除每个亚体素几何误差。

Five complete256³ synthetic morphology families also passed: granular,
rods, platelets, agglomerates and bicontinuous. Their balanced meshes contain
1.70–2.17 million tetrahedra with max radius/edge below2, minimum dihedrals
at least3.5° and maximum phase-volume errors0.55–1.01%. On the same saved
rod geometry, coarse/balanced/fine produced1,368,398 /2,101,963 /5,259,514
cells with an identical geometry-score SHA256. These synthetic fields are not
experimental CT data or calibrated battery performance predictions.

五种完整256³合成结构已经分别验证，同一长条结构的三档网格共用完全相同的准备场。
这组结果验证几何与网格能力，不能替代真实电池材料参数和多物理求解验证。

The workspace pages are `implicit_geometry/index.html` for full structures,
`conditioning_study/rods/index.html` for fixed-size smoothing versus resolution
reduction, and `implicit_geometry/acceptance.html` for consolidated measured
checks. Their JSON/HDF5 files are the numerical records. Original-resolution
2D slices remain available even when3D display uses an explicitly labelled LOD.
Offline CJK font subsets are optionally loaded from external
`data/report_font/`; the OFL notice accompanies each embedded subset. No font
binary, generated mesh or executed notebook is stored in source.

各页面路径相对于工作区；完整报告和数据在源码外。三维显示LOD不改变计算数组，
二维切片仍为原始分辨率。外部字体存在且安装了`report`可选依赖时，HTML嵌入带OFL
声明的中文字体子集；字体、网格与已执行Notebook均不进入源码。
