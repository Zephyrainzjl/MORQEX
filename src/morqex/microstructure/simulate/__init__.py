"""Unified simulation API; all implementations live in this package."""
from .transport import conduction,ConductionResult,save_field
from .cell import discharge,microstructure_parameters
from .homogenization import homogenize,tortuosity
from .elasticity import homogenize_elasticity
from .evolution import diffuse,DiffusionHistory
from .periodic import conductivity_tensor
from .electrochemistry import (AMRConfig,assemble_cell,prepare_case,export_geometry,
    DischargeSettings,DischargeCurve,discharge_3d,read_discharge,audit_solver,backend_info,run)
