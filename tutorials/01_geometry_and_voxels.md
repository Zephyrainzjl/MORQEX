# Geometry and voxel modelling

[中文版本](01_geometry_and_voxels.md#zh) · [Tutorial index](README.md)

## Coordinate and label contract

`regions[z,y,x]` is a three-dimensional nonnegative integer array. `spacing`
and `origin` are in **ZYX order, SI metres**. Mesh vertex coordinates use **XYZ
order, SI metres**. The array contains region IDs, not material parameters.
The same ID may occur in several disconnected components. A source-particle
ownership ID and a crystallographic grain ID are different quantities.

No label is automatically classified as a pore. When meshing, explicitly select
the regions that should receive solid elements. The examples use0=pore,
1=active material,2=CBD; custom IDs are fully supported.

## Supply your own256³ region array

```python
import numpy as np
from morqex.modeling import from_regions

labels = np.zeros((256,256,256), dtype=np.uint8)
labels[24:232,32:224,16:240] = 1
labels[80:176,100:156,:] = 2
model = from_regions(
    labels,
    spacing=(0.5e-6,0.5e-6,0.5e-6),
    origin=(0.,0.,0.),
    region_names={0:'pore',1:'active',2:'CBD'},
    metadata={'source':'my segmented volume','seed':123},
)
model.save('/home/zephyrain/lw/morph_workspace/microelectrode_gallery/custom_voxels.h5')
```

Floating-point arrays and negative IDs are rejected. Region names must cover
exactly the labels present in the array. The API retains array references;
finish constructing an array before starting a mesh job and do not mutate it
during generation. HDF5 loading uses `VoxelModel.load(path)`. NumPy input uses
`np.load(path, allow_pickle=False)` followed by `from_regions`.

## Parametric shape construction

```python
from morqex.modeling import Shape,from_shapes

objects = [
    Shape('sphere',1,(40e-6,40e-6,40e-6),(18e-6,)*3,
          irregularity=0.22),
    Shape('capsule',1,(70e-6,65e-6,65e-6),(4e-6,4e-6,42e-6),
          irregularity=0.20),
    Shape('platelet',2,(90e-6,60e-6,60e-6),(3e-6,25e-6,30e-6),
          irregularity=0.18),
]
model = from_shapes(objects,shape=(256,)*3,spacing=(0.5e-6,)*3)
```

Available primitives:

|Kind|Axes meaning, in local ZYX order|
|---|---|
|`sphere`|The first axis is the radius; use equal axes for clarity|
|`ellipsoid`|Three semiaxes|
|`box`|Three half lengths|
|`cylinder`|`(radius,radius,half_length)`; long axis is local axis2|
|`capsule`|Same as cylinder, with hemispherical ends beyond the half length|
|`platelet`|An ellipsoid with one small semiaxis|

Centres and axes are physical SI lengths. `rotation` is a proper3×3 matrix:
its columns are the local ZYX axes in the global ZYX frame. Reflection and
nonorthogonal matrices are rejected. `irregularity` lies in[0,0.35); it applies
a declared deterministic smooth radial warp before primitive classification.
This is geometric roughness, not a measured particle-surface law.

Shapes are evaluated at voxel centres. Later shapes overwrite earlier shapes;
use a later region0 shape to carve a cavity. Shapes outside the domain are
clipped at the box. Intersecting solids form bonded regions in the default
voxel mesh. A finite draw is not automatically an RVE. Resolve small gaps and
thin coatings with multiple voxels before interpreting their topology.

## Ten irregular electrode families

```python
from morqex.modeling import electrode,PRESETS
print(PRESETS)
model,source_ids = electrode('fibres',size=256,spacing=0.5e-6,
                            seed=20261127,cbd_fraction=0.12)
```

|Preset|Geometry family|
|---|---|
|`granular`|Irregular polydisperse ellipsoidal particle packing|
|`rods`|Long warped capsule particles with preferred orientation|
|`platelets`|Thin irregular, differently oriented platelets|
|`dendritic`|Branching capsule scaffold|
|`hollow`|Irregular hollow shells and internal pore cavities|
|`agglomerates`|Clusters of irregular primary particles|
|`fibres`|Long, curved, entangled conductive/active fibres|
|`bicontinuous`|Bundled correlation-based random-medium generator|
|`lamellar`|Warped lamellae with a thickness gradient|
|`channels`|Irregular channels extending along the electrode thickness|

The full requested lattice is evaluated directly. It is not an enlargement of
a smaller voxel array. CBD is surface-biased and its requested count is explicit;
physical percolation and fabrication realism must be assessed separately.
`source_ids` records geometric ownership after overlap/carving. It must not be
interpreted as a validated segmentation or independent grain field.

## TOML and custom generators

```toml
[geometry]
method = 'electrode'
preset = 'rods'
size = 256
spacing = 5e-7
seed = 20261128
cbd_fraction = 0.12
```

```python
from morqex.modeling import from_toml,register_generator,generate,from_regions
model = from_toml('my_geometry.toml')

def my_generator(array,spacing):
    return from_regions(array,spacing=spacing)

register_generator('my_geometry',my_generator)
custom = generate('my_geometry',array=labels,spacing=(0.5e-6,)*3)
```

For `method='shapes'`, put `shape`, `spacing`, `origin` and `background` under
`[geometry]` and add `[[shapes]]` tables with `kind`, `region`, `centre`, `axes`
and optional `rotation` / `irregularity`. For `method='regions'`, set `array` to
an `.npy` filename relative to the TOML file. Custom callbacks must return a
`VoxelModel`; duplicate registry names are rejected rather than overwritten.

## Sizes, grid resolution and orientation

With256³ cells and0.5µm spacing, the box is128µm wide. Increasing spacing at
fixed cell count enlarges the box; changing a shape's SI axes changes its
physical dimensions; increasing cell count while decreasing spacing can refine
the same physical box. These operations are distinct. Sphere axes specify a
radius, ellipsoid/platelet axes are semiaxes, and capsule total length is
`2*half_length+2*radius`.

```python
import numpy as np
angle = np.deg2rad(35)
R = np.array([[np.cos(angle),0,np.sin(angle)],
              [0,1,0],[-np.sin(angle),0,np.cos(angle)]])
rod = Shape('capsule',1,(64e-6,)*3,(4e-6,4e-6,40e-6),
            rotation=tuple(map(tuple,R)),irregularity=0.2)
model = from_shapes([rod],shape=(256,)*3,spacing=(0.5e-6,)*3)
```

The long local axis is ZYX index2. Use a proper rotation to align it with
another direction. Reflections and arbitrary scaling matrices are not rotation
inputs. A thin feature or gap should have several voxels across it before
interpreting connectivity. Centre classification alone does not establish
subvoxel geometric accuracy.

## Polydispersity, bimodal mixtures and cavities

```python
rng = np.random.default_rng(20261127)
objects = []
for k in range(150):
    centre = tuple(rng.uniform(8e-6,120e-6,3))
    mean = 5e-6 if k < 100 else 12e-6
    radius = float(rng.lognormal(np.log(mean),0.18))
    objects.append(Shape('ellipsoid',1,centre,
                         (radius*.8,radius,radius*1.2),irregularity=.2))
model = from_shapes(objects,shape=(256,)*3,spacing=(0.5e-6,)*3)
```

This example permits overlaps and does not enforce a packing law. Strict
spherical gaps are available through the existing `SpherePackSpec/sphere_pack`
adapter, which checks particle count, minimum gap and admissible radius draws.
It rejects an impossible packing rather than silently shrinking particles.

For a hollow body, write a larger solid shape first and a smaller region0
shape afterwards. For a coating, write the outer coating and then the inner
core with different labels. Later shapes overwrite earlier labels. A closed
internal pore is not automatically electrolyte-accessible; geometric labels
and transport accessibility require separate definitions.

The integrated `Distribution` API supports constant, normal, lognormal,
uniform, gamma and empirical laws. Its values can parameterize shape axes,
centres or placement routines, with explicit feasibility checks. Independent
source IDs are not crystallographic grain IDs. The ten current gallery presets
do not generate independent grain partitions; use a seeding/tessellation
backend and separate grain/orientation fields when those are required.

## Display and complete data

The256³ gallery stores all original voxels and the complete Tet4 mesh. Its
whole-domain interactive picture uses an explicitly labelled64³ display LOD.
The slice view reads original256³ voxels, and the local FE view contains actual
tetrahedra associated with that full-resolution geometry. This display choice
does not change the HDF5/XDMF computational files.


---

<a id="zh"></a>

# 几何与体素建模教程

[English version](01_geometry_and_voxels.md) · [教程目录](README.md) · [有限元网格](02_finite_element_meshes.md#zh)

## 1. 数据、坐标与单位

统一入口为 `morqex.modeling`。`VoxelModel` 保存三维分区数组、体素间距、原点、
区域名称与来源信息。数组索引始终为 **`regions[z,y,x]`**；`spacing`、`origin`、
几何对象的 `centre` 和 `axes` 也采用 **ZYX**。这些几何长度均使用 **米**。
导出的有限元节点则采用 **XYZ、米**，读取时不要再次交换轴。

|概念|例子|含义|
|---|---|---|
|数组大小|`(256,256,256)`|每个方向256个体素，不是256个颗粒|
|间距|`(0.5e-6,)*3`|每个体素边长0.5μm|
|计算域|`256 × 0.5μm = 128μm`|每方向的真实长度|
|区域ID|`0,1,2` 或 `17,42`|用户指定的分区编号|
|区域名称|`pore,active,CBD`|显式命名，不自动推断材料参数|
|来源分区|`source_ids`|形状重叠、裁切后的来源归属|
|晶粒编号|独立字段|与材料相、来源归属不是同一个量|

一个区域ID可以对应多个不连通区域。相同的标签不能保证导电连通，也不能自动
证明几何适合作为RVE。体素间距、颗粒尺寸和体素数量可以分别设定。

## 2. 从自己的三维数组开始

```python
import numpy as np
from morqex.modeling import from_regions

labels = np.zeros((256,256,256), dtype=np.uint8)
labels[20:230,30:220,15:240] = 1
labels[100:150,70:190,40:210] = 2

model = from_regions(
    labels,
    spacing=(0.5e-6,0.5e-6,0.5e-6),
    origin=(0.,0.,0.),
    region_names={0:'孔隙',1:'活性材料',2:'CBD'},
    metadata={'source':'用户分区数组','seed':123},
)
model.save('/home/zephyrain/lw/morph_workspace/microelectrode_gallery/custom.h5')
```

浮点标签、负标签、空数组和二维输入会被拒绝。提供 `region_names` 时，必须恰好
命名所有实际存在的ID。数组不要求标签从0连续排列。若原始CT是灰度数组，需要先
分割为整数标签；本接口不会把灰度强度自动解释为材料相。

`from_regions` 保留数组引用。完成数组构造后再开始网格生成，生成期间不要修改原数组。
再次载入使用 `VoxelModel.load(path)`。`.npy` 文件可通过
`np.load(path,allow_pickle=False)` 读取后交给 `from_regions`。

## 3. 六类可参数化形状

```python
from morqex.modeling import Shape,from_shapes

objects = [
    Shape('sphere',1,(30e-6,30e-6,30e-6),(12e-6,)*3),
    Shape('ellipsoid',1,(70e-6,35e-6,40e-6),(8e-6,14e-6,22e-6),
          irregularity=0.22),
    Shape('capsule',1,(65e-6,85e-6,65e-6),(3e-6,3e-6,35e-6),
          irregularity=0.20),
    Shape('platelet',2,(100e-6,70e-6,70e-6),(2e-6,18e-6,25e-6),
          irregularity=0.18),
]
model = from_shapes(objects,shape=(256,)*3,spacing=(0.5e-6,)*3)
```

|`kind`|`axes` 的意义|实际尺寸|
|---|---|---|
|`sphere`|建议三个值相同，使用第一个作为半径|直径为2倍半径|
|`ellipsoid`|ZYX方向三个半轴|各方向全长为2倍半轴|
|`box`|ZYX方向三个半长|各方向边长为2倍半长|
|`cylinder`|`(r,r,h)`，局部长轴是第2轴|半径r，柱体总长2h，平端|
|`capsule`|`(r,r,h)`，局部长轴是第2轴|柱体长2h，另加两端半球，总长2h+2r|
|`platelet`|一个半轴较小的椭球|通过最小半轴控制厚度|

未旋转的长棒默认沿局部第2轴，即全局X方向。`platelet` 是扁椭球原语；非平整
表面由 `irregularity` 产生，并不等同于任意壳曲面网格。

`irregularity=0` 给出理想形状；`0.1~0.3` 给出声明的径向扰动，允许范围为
`[0,0.35)`。它改变几何边界，不是经过实验标定的粗糙度模型。半径、半轴和位置
仍需结合体素大小检查，避免薄层或间隙只有一个体素。

## 4. 旋转、大小与分辨率

`rotation` 是3×3正交矩阵，行列式必须为+1。其列表示局部ZYX轴在全局ZYX坐标
中的方向。可以沿ZY平面旋转，也可以使用任意满足条件的三维旋转矩阵。

```python
import numpy as np
angle = np.deg2rad(35)
R = np.array([[np.cos(angle),0,np.sin(angle)],
              [0,1,0],
              [-np.sin(angle),0,np.cos(angle)]])
rod = Shape('capsule',1,(64e-6,)*3,(4e-6,4e-6,40e-6),
            rotation=tuple(map(tuple,R)),irregularity=0.2)
model = from_shapes([rod],shape=(256,)*3,spacing=(0.5e-6,)*3)
```

分清三种改变方式：

1. 固定体素数，增大 `spacing`：计算域变大，给定SI尺寸的颗粒占用体素变少。
2. 固定真实计算域，增加体素数：相应减小 `spacing`，同一几何得到更细离散。
3. 固定数组和间距，只改 `Shape.axes`：改变真实颗粒尺寸与相比例。

本API直接在目标网格的体素中心进行分类，不用重复小数组来伪造高分辨率。

## 5. 多粒径、随机取向与双峰分布

```python
import numpy as np
from morqex.modeling import Shape,from_shapes
rng = np.random.default_rng(20261127)
objects = []
for k in range(150):
    centre = tuple(rng.uniform(8e-6,120e-6,3))
    mean = 5e-6 if k < 100 else 12e-6
    radius = float(rng.lognormal(np.log(mean),0.18))
    objects.append(Shape('ellipsoid',1,centre,
                         (radius*0.8,radius,radius*1.2),
                         irregularity=0.2))
model = from_shapes(objects,shape=(256,)*3,spacing=(0.5e-6,)*3)
```

上述示例允许重叠：后写入对象覆盖先前标签，相交固体会构成连接体。它不是不重叠
压实算法。若要求严格颗粒间隙，可使用集成的 `SpherePackSpec/sphere_pack`，并明确
给定颗粒数量、尺寸分布、最小间隙和CBD比例；不满足放置条件时应改变参数，而非
偷偷缩小颗粒。更多统计分布见英文教程补充与后端教程。

## 6. 空心结构、包覆与自定义多相

形状按顺序覆盖，可以用区域0在固体内部挖孔：

```python
outer = Shape('ellipsoid',1,(64e-6,)*3,(25e-6,30e-6,35e-6),
              irregularity=0.18)
inner = Shape('ellipsoid',0,(64e-6,)*3,(15e-6,18e-6,20e-6),
              irregularity=0.12)
model = from_shapes([outer,inner],shape=(256,)*3,spacing=(0.5e-6,)*3)
```

包覆可按“大外形→小内核”的顺序设置不同ID。不同形状、材料相和惰性夹杂可以混合。
闭合内孔不自动被判为电解液可达孔隙；几何相、侵液状态和输运边界条件需分别定义。

## 7. 十种预设和统一扩展

```python
from morqex.modeling import PRESETS,electrode
print(PRESETS)
model,source_ids = electrode('fibres',size=256,spacing=0.5e-6,
                            seed=20261127,cbd_fraction=0.12)
```

预设分别为：`granular`不规则颗粒、`rods`长棒、`platelets`薄片、`dendritic`分枝骨架、
`hollow`多孔壳层、`agglomerates`团聚、`fibres`缠结长纤维、`bicontinuous`双连续随机、
`lamellar`梯度层片、`channels`曲折孔道。形态类别不同，不只是随机种子不同。

预设支持尺寸、间距、种子和CBD比例。若要逐颗粒修改尺寸与位置，使用 `from_shapes`；
若要控制完整分布，使用统计接口或自定义生成器，而不是假定预设覆盖所有需求。

```python
from morqex.modeling import register_generator,generate,from_regions
def my_recipe(labels,spacing):
    return from_regions(labels,spacing=spacing)
register_generator('my_recipe',my_recipe)
model = generate('my_recipe',labels=my_array,spacing=(0.5e-6,)*3)
```

自定义生成器必须返回 `VoxelModel`，重复名称会被拒绝。TOML入口为 `from_toml`：
`method='electrode'` 使用预设；`method='shapes'` 读取 `[[shapes]]` 列表；
`method='regions'` 从相对路径 `.npy` 文件读取分区。对应英文教程给出了完整示例。

## 8. 显示与完整数据的区别

图谱的计算数组和完整有限元网格均基于256³。为保持交互响应，全域三维画面使用
明确标注的64³显示LOD；二维截面读取原始256³数据；局部有限元视图使用真实256³
网格中的局部单元。原始HDF5/XDMF不会因显示LOD而降采样。

默认预设未独立生成晶粒，不能把来源分区称为晶粒。需要多晶材料时，可通过后端
种子／Voronoi模块生成独立晶粒标签，并为每个晶粒单独配置材料和取向。

## Unified command line / 统一命令行

```bash
WORK=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
PY=$WORK/runtime/bin/python
python scripts/run_local.py --workspace "$WORK" --mesh-output --label rods_recipe -- \
  "$PY" -m morqex.modeling generate /home/zephyrain/lw/MORQEX/configs/modeling_rods256.toml \
  --output "$WORK/custom_rods_voxels.h5" --mesh "$WORK/custom_rods_mesh.h5" \
  --regions 1,2 --preview "$WORK/custom_rods.html"
```

The resource runner changes its working directory; use an absolute config path
in an actual invocation, e.g. `/home/zephyrain/lw/MORQEX/configs/modeling_rods256.toml`.
The same entry also supports `mesh GEOMETRY.h5 --output MESH.h5 --regions 1,2`
and `capabilities`. Relative `.npy` input paths in TOML are resolved against the
configuration file. Generated files are rejected inside the source tree.

运行器会切换工作目录，实际运行时应将上述配置路径写成绝对路径，例如
`/home/zephyrain/lw/MORQEX/configs/modeling_rods256.toml`。同一入口还支持从已保存几何
生成网格和查询能力。TOML中相对`.npy`路径相对于配置文件定位。API拒绝把生成文件
保存到源码目录内。


## Resolved-feature geometry for adaptive meshing / 适合自适应重网格的新配方

```python
from morqex.modeling import resolved_electrode
model = resolved_electrode('rods',size=256,spacing=0.5e-6,seed=20261127,
    cbd_thickness_voxels=4,minimum_feature_voxels=4,cbd_bridges=True)
model.show(maximum_3d_size=48)
```

This is separate from `electrode`: minimum-feature operations, coating and
CBD bridges are explicit generation assumptions recorded in metadata. It does
not smooth or replace an imported user segmentation. All three phases are
saved on the complete grid. Geometry masks do not automatically determine
conductivity, porosity inside CBD or electrochemical accessibility.
新配方记录最小特征、包覆及CBD桥接等假设，不修改用户导入的分区数据；完整三相格点均保存。
形态标签不自动决定导电、CBD内部孔隙或电化学可达性。
See [GEMsim adaptive geometry](06_gemsim_adaptive_meshing.md) for all ten
families, custom phase chains, shared junctions, precision and mesh visualization.
十种形态、自定义材料顺序、共享交线及网格精度见该完整教程。
