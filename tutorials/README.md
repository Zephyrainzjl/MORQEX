# MORQEX tutorials / MORQEX 双语教程

These bilingual English/Chinese tutorials use the public `morqex.modeling` namespace. Source code
belongs in MORQEX; generated arrays, meshes, environments and HTML belong in
`/home/zephyrain/lw/morph_workspace/microelectrode_gallery`.

|Module / 模块|English|中文|
|---|---|---|
|Geometry / 几何与体素|[Geometry and voxels](01_geometry_and_voxels.md)|[几何与体素建模](01_geometry_and_voxels.md#zh)|
|Meshing / 有限元网格|[Finite-element meshes](02_finite_element_meshes.md)|[有限元网格生成](02_finite_element_meshes.md#zh)|
|Backends / 集成与扩展|[Integrated frameworks](03_integrated_geometry_backends.md)|[集成框架与API](03_integrated_geometry_backends.md#zh)|
|Notebook / 实时可视化|[Live2D/3D viewer](04_notebook_visualization.md)|[同页中英可视化教程](04_notebook_visualization.md)|
|Reference / 接口总览|[API reference](API.md)|[接口参考](API.md)|

每个模块的中英文合并在同一MD文件中。教程覆盖单位、形状、尺寸、取向、分区数组、概率生成、TOML、
大网格分块、节点共享、数据格式和扩展接口。先阅读几何约定，再生成网格。

The gallery's ten examples are synthetic geometry families. They are not CT
datasets, independently calibrated electrode materials, or simulation results.
The full computational arrays have256×256×256 cells. Whole-domain interactive
views use a labelled display LOD; exact local FE views and complete HDF5/XDMF
files retain the original mesh. Reading an XDMF file requires its HDF5 sibling.

Use the bounded local runner for numerical work:

```bash
cd /home/zephyrain/lw/MORQEX
WORK=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
PY=$WORK/runtime/bin/python
python scripts/run_local.py --workspace "$WORK" --label my_geometry -- "$PY" my_recipe.py
```

The installed runtime can inherit the workstation's NumPy/SciPy installation.
Optional geometry packages are declared in `pyproject.toml`; actual availability
is reported by `morqex.modeling.capabilities()`. No neural model is required.
