# Notebook visualization / Notebook 实时可视化

This module's English and Chinese instructions are kept in this same file.
本模块的中英文说明合并在本文件内，代码共用。

## 1. Two-dimensional and three-dimensional views / 二维与三维

`VoxelModel.show()` displays an offline notebook viewer and returns a live
handle. The3D view can be rotated; the2D view changes axis and slice index.
Clicking a2D voxel shows its original region label and physical centre.

`VoxelModel.show()` 在notebook输出中显示可交互界面并返回更新句柄。三维可拖动
旋转；二维可切换X/Y/Z切片轴、拖动索引，并点击查询原始区域ID和物理坐标。

```python
import numpy as np
from morqex.modeling import from_regions,Shape,from_shapes

model = from_regions(np.zeros((256,256,256),np.uint8),spacing=(0.5e-6,)*3)
view = model.show(maximum_3d_size=48,height=700,
                  title='Live electrode / 实时电极')
```

The default hides region0 for display only; choose another label or “None” in
the interface. Hidden regions remain in the model and its exports.
默认显示时隐藏区域0，可在界面改为其他标签或“不隐藏”。隐藏只影响显示，不删除数据。

For a notebook's automatic rich display, use `model.viewer()` as the final
cell expression. Use `view=model.show()` when subsequent cells need a handle.
若希望依靠单元格自动富显示，可将 `model.viewer()` 放在单元格末尾；后续要刷新时，
建议显式保存 `view=model.show()` 返回的句柄。

## 2. Refresh after geometry changes / 修改几何后刷新

```python
objects = [
    Shape('ellipsoid',1,(40e-6,40e-6,40e-6),(12e-6,18e-6,23e-6),
          irregularity=0.22),
    Shape('capsule',2,(80e-6,65e-6,65e-6),(4e-6,4e-6,35e-6),
          irregularity=0.18),
]
model = from_shapes(objects,shape=(256,)*3,spacing=(0.5e-6,)*3)
view.update(model)
```

The update uses the same notebook display ID, rather than appending another
large output. Camera state is reset when the iframe is replaced. This is a
geometry viewer, not an asynchronous simulation server.

更新复用同一个notebook显示位置，不不断追加大输出。更新会重建iframe，因此视角
会回到默认值。这是几何预览接口，不是后台仿真服务器。

## 3. See shapes during construction / 生成过程中逐批看结构

`from_shapes` accepts a progress callback. Set an interval so that a256³
structure is not repeatedly serialized after every tiny change.

`from_shapes` 支持进度回调。对256³建议每5~20个形状刷新一次，避免每加入一个
小形状都压缩整个数组而拖慢生成。回调接收一个带生成进度元数据的 `VoxelModel`。

```python
rng = np.random.default_rng(20261127)
objects = []
for k in range(40):
    objects.append(Shape(
        'capsule',1,tuple(rng.uniform(15e-6,113e-6,3)),
        (3e-6,3e-6,float(rng.uniform(12e-6,32e-6))),
        irregularity=0.2,
    ))

model = from_shapes(objects,shape=(256,)*3,spacing=(0.5e-6,)*3,
                    progress=view.update,progress_every=5)
view.update(model)
```

The callback is synchronous and runs on the same generation thread. It shows
real partially constructed labels. It does not invent a time-evolving field.
回调与几何生成在同一线程顺序执行，显示的是真实已构造分区，而不是模拟的动画帧。

## 4. Display resolution / 显示分辨率

`maximum_3d_size` bounds the3D display lattice. The viewer reports both the
original array shape and the display LOD/stride.2D slices use the complete
original array. No display operation alters the original geometry or mesh.

`maximum_3d_size` 控制三维预览规模。界面会明确显示原始数组大小、显示LOD和步长。
二维切片读取完整原始数组。显示参数不会改变原始体素或有限元网格。

```python
small_view = model.show(maximum_3d_size=32,height=600)
detail_view = model.show(maximum_3d_size=64,height=800)
```

Use32~48 for repeated live updates. Larger previews contain many boundary
faces and cost more CPU time. This viewer uses browser Canvas and local
embedded data; it has no CDN, remote rendering service or neural model.

连续刷新建议32~48。更大预览可能包含很多边界面，会增加浏览器CPU开销。界面使用
Canvas与本地内嵌数据，不依赖CDN、远程渲染或神经网络。

## 5. Save a standalone HTML / 保存独立HTML

```python
view.save_html(
 '/home/zephyrain/lw/morph_workspace/microelectrode_gallery/notebook_live.html'
)
```

The saved file contains the original labels and display data. Save outside
the source tree. Standard modern browsers need gzip DecompressionStream
support; no web server is necessary. For a full FE mesh inspection, use the
gallery's actual local Tet4 view or open the complete XDMF/HDF5 in ParaView.

保存文件包含当前原始标签和预览数据，应放在源码外。需要支持gzip
DecompressionStream的现代浏览器，不需要启动服务器。有限元内部单元可通过图谱
的真实局部Tet4视图查看；完整体网格使用ParaView打开XDMF/HDF5。

## 6. Executable notebook / 可执行notebook

Open [geometry_workflow.ipynb](notebooks/geometry_workflow.ipynb). The source
notebook contains only code and instructions. Execute a copy in the external
workspace so saved HTML, mesh data and notebook outputs do not enlarge source.

打开 [geometry_workflow.ipynb](notebooks/geometry_workflow.ipynb)。源码中的notebook
只保存代码和说明；请在外部工作区执行副本，避免把大输出保存回源码。

Jupyter's interactive output updates require an IPython kernel. Importing
geometry or constructing `model.viewer()` does not require starting a kernel.
If an environment lacks IPython, `.show()` reports the missing dependency;
the standalone viewer's `.save_html()` remains usable.

交互更新需要IPython内核。普通几何生成不依赖内核；缺少IPython时，`.show()` 会
报告依赖缺失，仍可通过 `model.viewer().save_html(...)` 生成离线页面。

## 7. Actual finite-element cells in Jupyter / 在Jupyter查看真实有限元单元

The complete streamed mesh can be inspected without loading all elements.
`MeshArtifact.show` reads only the chosen cell range and the vertices it uses.
The positions and connectivity come directly from the original HDF5.

完整分块网格不需要全部装入内存才能看。`MeshArtifact.show` 只读取选定单元段及其
使用的节点，坐标与连接关系直接来自原始HDF5，不用几何代理替代有限元单元。

```python
artifact = model.mesh_to(WORK/'my_mesh.h5',regions=[1],slab_depth=2)
mesh_view = artifact.show(first_cell=0,cell_count=600,height=620)
mesh_view.save_html(WORK/'my_actual_tetrahedra.html')
```

Use a fresh filename for a new mesh. If the artifact already exists, reconstruct
its lightweight `MeshArtifact` path/count contract, as demonstrated in the
notebook. A new `mesh_to` call does not overwrite a final accepted mesh.

新网格要使用新文件名。已有网格可按notebook示例重建轻量的 `MeshArtifact` 路径与
数量对象；再次调用 `mesh_to` 不会覆盖已完成网格。

The viewer supports drag rotation, display shrink and opacity. Shrink separates
individual tetrahedra for inspection; it changes neither coordinates nor mesh
connectivity on disk. A single view is bounded to5000 elements; default1000.
Change `first_cell` to inspect another part of the full mesh.

界面支持拖动旋转、显示收缩和透明度。收缩用于看清单个四面体，不修改磁盘坐标和
连接关系。单视图最多5000单元，默认1000；用 `first_cell` 改变查看的单元区段。

```python
from morqex.visualization.notebook import MeshView
# For an existing small in-memory FEMesh / 已有小规模内存网格：
preview = MeshView(small_fem_mesh,cell_count=300)
preview.display()
```

## Whole-mesh actual-cell sampling / 全域真实单元抽样

```python
# Spread a bounded set of actual Tet4 across the entire stored mesh.
# 在整个真实网格中均匀抽取单元，适合观察多个颗粒的整体位置。
mesh_view = artifact.viewer(cell_count=2500,sampling='uniform')
mesh_view.display()
mesh_view.save_html('/home/zephyrain/lw/morph_workspace/microelectrode_gallery/whole_mesh.html')

# Default remains a contiguous local subset / 默认仍是连续局部单元
local = artifact.viewer(first_cell=10000,cell_count=1000,sampling='contiguous')
```

Uniform display sampling keeps each selected element's real global cell ID,
region and vertex coordinates, including for HDF5 artifacts. It is a display
subset, not a new volume mesh or a conservative physical reduction. It cannot
be used for integration, connected-component tests or solving the full domain.
全域抽样保留单元原编号、材料与坐标，仅用于显示；不能代替完整积分、拓扑检查或体求解。


## Real adaptive surfaces and interior sections / 自适应曲面与内部网格

```python
# Given a generated TetGenMesh / 已生成的体网格
view = result.adaptive_viewer(title='Actual adaptive mesh / 真实自适应网格')
view.display()
view.save_html('/home/zephyrain/lw/morph_workspace/microelectrode_gallery/adaptive.html')
```

This viewer retains the full marked boundary surface and exact3D-element
plane intersections. It provides material selection, rotation, wireframe,
face-display budgets and YZ/XZ/XY interior sections. These are not a display
made from voxel splitting or a replacement finite-element mesh.
该查看器保存完整计算边界及真实体单元切面，支持材料选择、旋转、线框及内部疏密查看；
不会把体素图或其他网格冒充计算结果。详细API见
[完整教程](06_gemsim_adaptive_meshing.md)。
## Isolate material regions / 单独查看材料域

```python
# 3D isolates one label; the2D view retains the complete original slice.
# 三维单独显示一个分区；二维保留原始完整分区切片。
model.viewer(maximum_3d_size=64, show_region=1).display()
```

The Show dropdown switches between individual materials and all regions.
Hide and clipping remain available. Explicit labels must exist in the model;
the view never changes geometry or physical phase assignments. 三维可选择活性材料、
CBD、电解液及各电池层，不改变模型；显式标签不存在时接口会报错。
