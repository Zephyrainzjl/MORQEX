# Integrated framework APIs and extension points

[中文版本](03_integrated_geometry_backends.md#zh) · [Tutorial index](README.md)

## One MORQEX namespace

```python
from morqex.modeling import framework,capabilities
print(capabilities())
geometry = framework('shapes')
sphere = geometry.Sphere(radius=5.,center=(0.,0.,0.))
print(sphere.within([[0.,0.,0.],[10.,0.,0.]]))
```

`framework(name)` returns the complete public namespace of the source-integrated
module; class methods and keyword arguments remain available through this
MORQEX entry. Original upstream copyrights/licenses remain attached. It is a
transparent adapter, not a claim of newly invented algorithms or a wholesale
Python translation of non-Python source.

|MORQEX portal|Public contents / execution requirements|
|---|---|
|`shapes`|MicroStructPy Sphere, Ellipsoid, Box, Circle, Ellipse, Rectangle, Square and factory|
|`seeds`|MicroStructPy Seed / SeedList and their public methods|
|`tessellation`|MicroStructPy PolyMesh / RasterMesh / TriMesh and public methods; native engines may be optional|
|`statistics`|MORQEX Distribution, StatisticsCard, characterize, connectivity and generate|
|`statistical_adapters`|Pinned upstream seeding/tessellation/raster adapter and reconstruction runtime checks|
|`gem`|Integrated GEM Microstructure / Volume and lazy generation, meshing, analysis, simulation and viewing modules|
|`reconstruction`|Existing GEM reconstruction helpers, with their own dependencies|
|`finite_elements`|Existing FEMesh, voxel_to_hex, voxel_to_tetra, surface and legacy mesh types|

Examples:

```python
seeds = framework('seeds')
SeedList = seeds.SeedList
meshing = framework('tessellation')
PolyMesh = meshing.PolyMesh
RasterMesh = meshing.RasterMesh

statistics = framework('statistics')
distribution = statistics.Distribution('lognormal',{'log_mean':1.,'log_std':0.2})
```

Full seeding/tessellation calls retain their original physical conventions and
backend arguments. The adapter
`framework('statistical_adapters').microstructpy_voxels(...)` converts the
upstream raster to the MORQEX geometry contract, using voxel-unit geometry and
explicit SI spacing. It requires a working Voro++ / pyvoro environment; source
availability alone does not supply the compiled engine.

## Micro_Reconstruction is a different runtime

The complete integrated textual source is under
`src/morqex/vendor/micro_reconstruction`. Its algorithms include MATLAB and R
code. The framework's original MATLAB/R runtime and package requirements still
apply. A Gaussian Python random field must not be labelled an execution of
Micro_Reconstruction. `capabilities()` reports MATLAB and Rscript availability.
The existing native adapter reports missing requirements explicitly; it does
not launch uncontrolled original parallel pools or substitute another method.

Large original example datasets are external assets. Clearing the workspace
also clears downloaded archives and compiled engines; a source-only catalog
does not imply those caches are present. Rebuild/download deliberately using
their respective provenance and license records.

## Custom scientific integrations

Prefer a generator callback returning `VoxelModel` for new physics-informed
geometry. It gives all generators the same region, unit, storage and mesh API:

```python
from morqex.modeling import register_generator,generate,from_regions

def segmented_ct(labels,spacing,origin=(0.,0.,0.)):
    return from_regions(labels,spacing=spacing,origin=origin,
                        metadata={'source':'my independently segmented CT'})

register_generator('segmented_ct',segmented_ct)
model = generate('segmented_ct',labels=my_array,spacing=(0.3e-6,)*3)
artifact = model.mesh_to('/external/workspace/ct_mesh.h5',regions=[1,2])
```

A custom generator can internally call any integrated upstream API, provided
it returns explicit region labels and correct SI metadata. Internal material
fields, transport coefficients and grain orientations should be stored as
separate fields; do not overload a phase ID with several unrelated meanings.

## Scope of the current tutorial release

Unified arrays, analytic shapes, ten irregular presets, TOML, extension
callbacks and streamed Tet4 generation have executable examples and targeted
checks. Not every optional upstream seed distribution, surface mesh engine,
MATLAB script, reconstruction objective or simulation backend has been tested
in this release. Native prerequisites and unsupported physical combinations
must be resolved explicitly before claiming their results.


---

<a id="zh"></a>

# 集成几何框架与完整API入口

[English version](03_integrated_geometry_backends.md) · [教程目录](README.md)

## 1. 统一入口与上游算法归属

```python
from morqex.modeling import framework,capabilities
print(capabilities())
shapes = framework('shapes')
sphere = shapes.Sphere(radius=5.,center=(0.,0.,0.))
print(sphere.within([[0.,0.,0.],[10.,0.,0.]]))
```

`framework(name)` 在MORQEX自己的命名空间中提供集成模块的完整公开命名空间。
因此不只包装几个函数：类、类方法和原始关键字仍能通过这个入口访问。
原始框架的版权、许可证、参数约定和运行时依赖继续有效。

|MORQEX入口|内容|
|---|---|
|`shapes`|MicroStructPy Sphere、Ellipsoid、Box、Circle、Ellipse、Rectangle、Square、factory|
|`seeds`|MicroStructPy Seed、SeedList及公开方法|
|`tessellation`|MicroStructPy PolyMesh、RasterMesh、TriMesh等公开方法|
|`statistics`|Distribution、StatisticsCard、generate、characterize、connectivity|
|`statistical_adapters`|种子→Laguerre剖分→体素化与原生重构适配器|
|`gem`|集成GEM Microstructure、Volume及生成／网格／分析／显示模块|
|`reconstruction`|现有GEM重构辅助模块|
|`finite_elements`|FEMesh、voxel_to_hex、voxel_to_tetra、surface及原网格类型|

## 2. 概率分布与结构统计

```python
import numpy as np
statistics = framework('statistics')
rng = np.random.default_rng(123)
distribution = statistics.Distribution('lognormal',
    {'log_mean':np.log(8.0),'log_std':0.2})
radii_voxels = distribution.sample(rng,100)
```

支持constant、normal、lognormal、uniform、gamma和empirical。分布本身不保证
半径为正或满足放置间隙，需由几何生成器检查。不应通过裁剪半径来悄悄改变目标分布。

`StatisticsCard` 明确指定相比例、相关长度、相语义和来源；`generate` 生成随机介质，
报告实际体积分数与连通性。相关随机场不是唯一CT重建，也不能只凭相比例相同就
宣称其具有相同输运性质。

## 3. MicroStructPy种子、剖分与体素

```python
SeedList = framework('seeds').SeedList
PolyMesh = framework('tessellation').PolyMesh
RasterMesh = framework('tessellation').RasterMesh
```

种子、粒径／形状分布、位置、周期选项和网格后端参数保留原始接口。原生的Laguerre
剖分需要可工作的Voro++/pyvoro；暴露源码API不等于已安装编译引擎。
`framework('statistical_adapters').microstructpy_voxels(...)` 可将原始剖分／RasterMesh
结果转换到MORQEX数据约定，几何内部采用体素单位，外部明确给出SI体素间距。

## 4. Micro_Reconstruction的MATLAB/R运行时

相关文本源码位于 `src/morqex/vendor/micro_reconstruction`。它包含MATLAB和R算法，
执行时仍需要原始运行时、工具箱和依赖包。当前Python命名空间不是所有MATLAB/R
代码的自动翻译。`capabilities()` 会检查MATLAB和Rscript是否存在。

原生重构适配器会明确报告缺失条件，不会把Python高斯随机场当作该算法的运行结果，
也不会直接启动原始脚本中不受控的并行池。清空工作区会删除外部下载数据、压缩包
和编译缓存；源码存在不能替代这些运行资产。

## 5. 面向后续扩展的接口

推荐将新几何算法注册为返回 `VoxelModel` 的回调，将新网格算法注册为返回
`FEMesh` 或 `MeshArtifact` 的回调。这样形状、统计生成和用户CT分区都可复用同一
单位、存储和网格链路。取向、材料参数、孔隙侵液状态和损伤应保存为独立字段。

当前已执行统一数组、预设、原语、TOML、回调和分块Tet4测试；不宣称每个可选上游
算法、原生网格引擎和MATLAB重构脚本都已在本机验证。先检查依赖，再对目标流程
做独立几何、统计和物理验证。


## Weighted ACVD native backend / 加权ACVD原生后端

The source-integrated GEMsim weighted Cython kernel and original `Clustering`
driver now have an independent opt-in native build: `MORQEX_BUILD_ACVD=1`.
The complete geometric pipeline is public through `morqex.modeling.adaptive_mesh`,
not the unsafe legacy face/cell-deletion fallback. See the bilingual
[adaptive tutorial](06_gemsim_adaptive_meshing.md).
原始加权ACVD内核与驱动已接入独立编译入口，整个几何流程使用统一API；不调用旧版删面/删体单元的兜底分支。
