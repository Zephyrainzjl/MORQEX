"""Small-strain reference mechanics with explicit physical tensor conventions."""
from .tensors import isotropic_stiffness, to_mandel, from_mandel, rotate_stiffness
from .mesh import TetMesh
from .reference import ElasticitySolver, ElasticState, MechanicalSolveError
from .matrix_free import MatrixFreeElasticitySolver

__all__ = ["isotropic_stiffness", "to_mandel", "from_mandel", "rotate_stiffness",
           "TetMesh", "ElasticitySolver", "MatrixFreeElasticitySolver", "ElasticState", "MechanicalSolveError"]
