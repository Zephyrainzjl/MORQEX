"""Unified MORQEX geometry and finite-element modelling API.

Array axes and spacing/origin: ZYX. Mesh coordinates: XYZ in SI metres.
Optional backends retain their own licenses, dependencies and model assumptions.
"""
from .voxel import VoxelModel,Shape,from_regions,from_shapes,framework,capabilities
from .meshing import tetrahedralize,stream_tetrahedralize,MeshArtifact
from .presets import electrode,PRESETS
from .interfaces import register_generator,generate,from_toml,prepare_from_toml,register_mesher,mesh,mesh_from_toml,export_from_toml,remesh_from_toml
from .tetgen import SurfacePLC,TetGenOptions,TetGenMesh,voxel_plc,marching_cubes_plc,tetgen_mesh,tetra_quality
from .tetgen_backend import build_tetgen
from .comsol import export_comsol,read_comsol
from .adaptive import AdaptiveMeshOptions,adaptive_surface,adaptive_mesh,multiresolution_weights
from .acvd_backend import build_acvd
from .resolved_presets import resolved_electrode
from .imports import tiff_voxels,import_comsol,material_interfaces
from .dense import dense_electrode,coated_electrode,porous_layer,uniform_layer,stack_layers,DENSE_MORPHOLOGIES
from .surface_nets import surface_nets_plc
from .exchange import export_bundle
from .partitioned import cut_cap,slab_plc,assemble_slabs
from .resolution import MeshResolution,controlled_tetgen
from .remeshing import RemeshOptions,remesh,remesh_artifact
from .mmg_backend import build_mmg
from .partition_workflow import partitioned_tetgen
from .native_build import build_native_from_toml
from .preparation import SmoothingOptions,PreparedGeometry,smooth_voxels,lower_resolution,generate_voxels
from .features import analyze_features,voxel_fingerprint
from .progress import ProgressEvent,ProgressBar,NotebookProgress,HTMLProgress,TimingHistory
from .implicit import ImplicitMeshOptions,implicit_mesh,recommend_mesh_sizes
from .cgal_backend import build_cgal
from .workflow import run_geometry,GeometryRun

__all__=['VoxelModel','Shape','from_regions','from_shapes','framework','capabilities',
         'tetrahedralize','stream_tetrahedralize','MeshArtifact','electrode','PRESETS',
         'register_generator','generate','from_toml','register_mesher','mesh',
         'SurfacePLC','TetGenOptions','TetGenMesh','voxel_plc','marching_cubes_plc','tetgen_mesh',
         'tetra_quality','build_tetgen','export_comsol','read_comsol','mesh_from_toml',
         'AdaptiveMeshOptions','adaptive_surface','adaptive_mesh','multiresolution_weights','build_acvd','resolved_electrode',
         'tiff_voxels','import_comsol','material_interfaces','dense_electrode','coated_electrode','porous_layer','stack_layers',
         'DENSE_MORPHOLOGIES','surface_nets_plc','export_bundle','export_from_toml','uniform_layer',
         'cut_cap','slab_plc','assemble_slabs','MeshResolution','controlled_tetgen','RemeshOptions','remesh','remesh_artifact','build_mmg','remesh_from_toml','partitioned_tetgen','build_native_from_toml',
         'SmoothingOptions','PreparedGeometry','smooth_voxels','lower_resolution','generate_voxels','analyze_features','voxel_fingerprint',
         'ProgressEvent','ProgressBar','NotebookProgress','HTMLProgress','TimingHistory',
         'ImplicitMeshOptions','implicit_mesh','recommend_mesh_sizes','build_cgal','prepare_from_toml','run_geometry','GeometryRun']
