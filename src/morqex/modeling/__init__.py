"""Unified MORQEX geometry and finite-element modelling API.

Array axes and spacing/origin: ZYX. Mesh coordinates: XYZ in SI metres.
Optional backends retain their own licenses, dependencies and model assumptions.
"""
from .voxel import VoxelModel,Shape,from_regions,from_shapes,framework,capabilities
from .meshing import tetrahedralize,stream_tetrahedralize,MeshArtifact
from .presets import electrode,PRESETS
from .interfaces import register_generator,generate,from_toml,register_mesher,mesh

__all__=['VoxelModel','Shape','from_regions','from_shapes','framework','capabilities',
         'tetrahedralize','stream_tetrahedralize','MeshArtifact','electrode','PRESETS',
         'register_generator','generate','from_toml','register_mesher','mesh']
