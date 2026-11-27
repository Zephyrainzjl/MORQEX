"""Unified MORQEX geometry and finite-element modelling API.

Array axes and spacing/origin: ZYX. Mesh coordinates: XYZ in SI metres.
Optional backends retain their own licenses, dependencies and model assumptions.
"""
from .voxel import VoxelModel,Shape,from_regions,from_shapes,framework,capabilities
from .meshing import tetrahedralize,stream_tetrahedralize,MeshArtifact
from .presets import electrode,PRESETS
from .interfaces import register_generator,generate,from_toml,register_mesher,mesh,mesh_from_toml
from .tetgen import SurfacePLC,TetGenOptions,TetGenMesh,voxel_plc,marching_cubes_plc,tetgen_mesh,tetra_quality
from .tetgen_backend import build_tetgen
from .comsol import export_comsol,read_comsol
from .adaptive import AdaptiveMeshOptions,adaptive_surface,adaptive_mesh,multiresolution_weights
from .acvd_backend import build_acvd
from .resolved_presets import resolved_electrode

__all__=['VoxelModel','Shape','from_regions','from_shapes','framework','capabilities',
         'tetrahedralize','stream_tetrahedralize','MeshArtifact','electrode','PRESETS',
         'register_generator','generate','from_toml','register_mesher','mesh',
         'SurfacePLC','TetGenOptions','TetGenMesh','voxel_plc','marching_cubes_plc','tetgen_mesh',
         'tetra_quality','build_tetgen','export_comsol','read_comsol','mesh_from_toml',
         'AdaptiveMeshOptions','adaptive_surface','adaptive_mesh','multiresolution_weights','build_acvd','resolved_electrode']
