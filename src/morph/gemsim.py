"""Direct integration of the pinned GEMsim source, without importing ZynJoule."""
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
from .microstructure import Microstructure
from .workspace import Workspace


class GEMsim:
    def __init__(self, workspace: Workspace):
        lock = json.loads((Path(__file__).resolve().parents[2] / "upstream.lock.json").read_text())
        self.lock = lock["gemsim"]
        self.source = workspace.path(self.lock["source_relative_to_workspace"])
        if not (self.source / ".git").exists():
            raise RuntimeError("Pinned GEMsim source missing; run scripts/bootstrap_gemsim.py")
        def git(*args):
            return subprocess.check_output(["git", "-C", str(self.source), *args], text=True).strip()
        self.commit = git("rev-parse", "HEAD")
        if self.commit != self.lock["commit"] or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("GEMsim commit mismatch or modified source; preserve it and audit separately")
        package_dir = (self.source / "src/gem").resolve()
        # Never mix an installed gem package with this snapshot.
        for name, module in tuple(sys.modules.items()):
            if name == "gem" or name.startswith("gem."):
                location = getattr(module, "__file__", None)
                if location and package_dir not in Path(location).resolve().parents:
                    raise RuntimeError("Another GEMsim installation is already imported")
        path = str(self.source / "src")
        if path not in sys.path:
            sys.path.insert(0, path)
        self.volume_type = import_module("gem.volume").Volume

    def from_volume(self, volume, *, phase_kinds: dict[int, str]) -> Microstructure:
        if not isinstance(volume, self.volume_type):
            raise TypeError("Expected GEMsim Volume from the pinned source")
        return Microstructure(volume.data, volume.spacing, volume.origin, phase_kinds,
                              {"provider": "GEMsim", "commit": self.commit,
                               "upstream_metadata": dict(volume.metadata)})

    def to_volume(self, structure: Microstructure):
        return self.volume_type(structure.labels.copy(), structure.spacing_zyx_m,
                                structure.origin_zyx_m, dict(structure.provenance))

    def load(self, path: Path, *, phase_kinds: dict[int, str]) -> Microstructure:
        return self.from_volume(self.volume_type.load(path), phase_kinds=phase_kinds)

    def random_field(self, *, phase_kinds, **kwargs) -> Microstructure:
        """GEMsim's CPU random-field baseline, not a trained CT reconstruction."""
        volume = import_module("gem.design").random_field(**kwargs)
        return self.from_volume(volume, phase_kinds=phase_kinds)

    def mesh(self, structure: Microstructure, *, cell_type="tetra", phases=None):
        mesh_module = import_module("gem.mesh.voxel")
        if cell_type not in {"tetra", "hex"}:
            raise ValueError("cell_type must be tetra or hex")
        function = mesh_module.voxel_to_tetra if cell_type == "tetra" else mesh_module.voxel_to_hex
        return function(self.to_volume(structure), phases=phases)

