"""Geometry contract adapter for the integrated GEMsim-derived implementation."""
from importlib import import_module
from pathlib import Path
from .geometry import MicrostructureData
from .workspace import Workspace


class GEMsim:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.commit = "6fd789c78d58d49ef645c539414567fc2a2bc171"
        self.volume_type = import_module("morqex.microstructure.volume").Volume

    def from_volume(self, volume, *, phase_kinds: dict[int, str]) -> MicrostructureData:
        if not isinstance(volume, self.volume_type):
            raise TypeError("Expected GEMsim Volume from the pinned source")
        return MicrostructureData(volume.data, volume.spacing, volume.origin, phase_kinds,
                              {"provider": "GEMsim", "commit": self.commit,
                               "upstream_metadata": dict(volume.metadata)})

    def to_volume(self, structure: MicrostructureData):
        return self.volume_type(structure.labels.copy(), structure.spacing_zyx_m,
                                structure.origin_zyx_m, dict(structure.provenance))

    def load(self, path: Path, *, phase_kinds: dict[int, str]) -> MicrostructureData:
        return self.from_volume(self.volume_type.load(path), phase_kinds=phase_kinds)

    def random_field(self, *, phase_kinds, **kwargs) -> MicrostructureData:
        """GEMsim's CPU random-field baseline, not a trained CT reconstruction."""
        volume = import_module("morqex.microstructure.design").random_field(**kwargs)
        return self.from_volume(volume, phase_kinds=phase_kinds)

    def mesh(self, structure: MicrostructureData, *, cell_type="tetra", phases=None):
        mesh_module = import_module("morqex.microstructure.mesh.voxel")
        if cell_type not in {"tetra", "hex"}:
            raise ValueError("cell_type must be tetra or hex")
        function = mesh_module.voxel_to_tetra if cell_type == "tetra" else mesh_module.voxel_to_hex
        return function(self.to_volume(structure), phases=phases)

