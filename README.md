# MORQEX

**Microstructure modelling, finite-element meshing and multiscale battery research.**

MORQEX brings voxel geometry, region-labelled finite-element meshes and battery
research solvers into one Python source tree. Its geometry API accepts both
parameterized shapes and user-supplied three-dimensional region arrays. The
source-integrated GEM and MicroStructPy modules provide reusable geometry and
mesh capabilities; optional backends retain their original licenses and runtime
requirements.

[Tutorials](tutorials/README.md) · [API reference](tutorials/API.md) ·
[Voxel modelling](tutorials/01_geometry_and_voxels.md) ·
[Finite-element meshing](tutorials/02_finite_element_meshes.md) ·
[Integrated backends](tutorials/03_integrated_geometry_backends.md)

中文教程：[几何与体素](tutorials/01_geometry_and_voxels.md#zh) ·
[有限元网格](tutorials/02_finite_element_meshes.md#zh) ·
[集成框架与扩展](tutorials/03_integrated_geometry_backends.md#zh)。

[Notebook实时二维/三维可视化](tutorials/04_notebook_visualization.md) ·
[Executable notebook / 可执行示例](tutorials/notebooks/geometry_workflow.ipynb)。

## Geometry workflow

```text
Physical shapes / statistical recipes / segmented region arrays
                              │
                       VoxelModel (ZYX)
                              │
           explicit material regions + SI spacing/origin
                              │
             GEM voxel-to-Tet4 / streamed global meshing
                              │
              HDF5 + XDMF + parent-voxel association
                              │
          interactive geometry inspection / research solvers
```

- **One modelling API:** `morqex.modeling` for arrays, shapes, presets, TOML,
  generator extensions and mesh generation.
- **Custom regions:** nonnegative integer IDs with explicit names; active
  material, CBD, pores and additional phases can be represented separately.
- **Irregular presets:** granular, elongated rods, platelets, dendritic
  scaffolds, hollow shells, agglomerates, fibres, bicontinuous networks,
  gradient lamellae and curved channels.
- **Large voxel grids:** directly evaluated256×256×256 geometry; complete
  conforming Tet4 meshes are streamed to disk with compact global shared nodes.
- **Traceable geometry:** physical units, region labels, source metadata and
  element-to-original-voxel mapping remain available in the exported files.
- **Interactive inspection:** offline HTML with whole-domain display LOD,
  original-resolution slices and exact local tetrahedral mesh views.

## Quick start

Python3.11 or newer is required. A minimal scientific environment needs NumPy,
SciPy, h5py, Matplotlib, Pillow, pyquaternion and lsq-ellipse. The optional
geometry, statistics and report dependencies are declared in `pyproject.toml`.
Keep the environment and generated files outside this source directory.

```bash
cd /home/zephyrain/lw/MORQEX
python -m venv --system-site-packages ../morph_workspace/microelectrode_gallery/runtime
../morph_workspace/microelectrode_gallery/runtime/bin/python -m pip install -e '.[modeling,notebook,report]'
```

Create an irregular electrode and generate its full mesh:

```python
from morqex.modeling import electrode

model, source_regions = electrode(
    'rods', size=256, spacing=0.5e-6, seed=20261127,
    cbd_fraction=0.12,
)
mesh = model.mesh_to(
    '/home/zephyrain/lw/morph_workspace/microelectrode_gallery/rods.h5',
    regions=[1, 2], slab_depth=2,
)
print(mesh.xdmf, mesh.nodes, mesh.tetrahedra)
```

Or supply your own region array:

```python
from morqex.modeling import from_regions

model = from_regions(
    my_labels_zyx, spacing=(0.5e-6,)*3,
    region_names={0:'pore', 1:'active material', 2:'CBD'},
)
mesh = model.mesh_to('/external/workspace/custom.h5', regions=[1,2])
```

Open the generated `.xdmf` in ParaView while keeping its `.h5` sibling beside
it. Full256³ meshes can contain tens of millions of elements; use HDF5 slices
or a local mesh view for inspection. Geometry generation and a feasible
full-field mechanical solve are separate resource requirements.

## Resource-controlled execution

Numerical commands should run through the local resource launcher:

```bash
WORK=/home/zephyrain/lw/morph_workspace/microelectrode_gallery
PY=$WORK/runtime/bin/python
python scripts/run_local.py --workspace "$WORK" --label geometry_run -- \
  "$PY" my_geometry_recipe.py
```

Default limits are one CPU thread, low process priority,3GiB process-tree RSS
and an8GiB host memory reserve. The explicit CPU reference profile permits
6GiB RSS while retaining the host reserve. GPU use is opt-in; the geometry
gallery does not require a GPU. All builds, data, meshes and run logs remain
outside MORQEX.

## Source organization

```text
src/morqex/
  modeling/          unified geometry and finite-element API
  generation/        statistical media, particles and geometry adapters
  microstructure/    integrated GEM geometry, voxelization and mesh modules
  mechanics/         reference elasticity and matrix-free solvers
  electrochemistry/  DFN and microstructure coupling research modules
  top_damage/        fatigue, topology and degradation research modules
  visualization/     geometry and computed-field HTML exporters
  vendor/            pinned third-party source with notices and licenses
src/pybamm/          source-integrated electrochemical solver
tutorials/           English module tutorials and API reference
configs/             TOML examples and resource settings
scripts/             generation, checking and reproducibility entry points
tests/               targeted mathematical and software checks
```

The complete public geometry namespaces of the integrated backends are
accessible through `morqex.modeling.framework(name)`. See the
[backend tutorial](tutorials/03_integrated_geometry_backends.md) for available
portals and native prerequisites. Micro_Reconstruction's MATLAB/R source is
included; executing it requires its original runtimes. Python statistical
generation is explicitly identified as a different method.

## Research modules and validation

MORQEX also contains electrochemical, mechanical, thermal and degradation
research interfaces. The [API reference](tutorials/API.md) records those
interfaces and their model assumptions. Source availability and successful
geometry checks do not establish calibrated material predictions, spatial
convergence or long-term battery-life validation.

The workspace was cleared on request before creating the new geometry atlas.
Earlier run outputs and numerical environments are not included in the source
repository and must be regenerated when needed. Current geometry outputs live
under `../morph_workspace/microelectrode_gallery/atlas256/`.

Third-party source provenance, copyright and licenses are retained in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and the corresponding source
directories. MORQEX does not rename established scientific methods as new
methods or remove upstream attribution.
