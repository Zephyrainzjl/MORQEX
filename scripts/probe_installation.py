"""Check an installed wheel without adding source or original GEMsim to sys.path."""
from importlib.resources import files
import json
from pathlib import Path
import sys
import numpy as np
from morqex import microstructure
from morqex.gemsim import GEMsim
from morqex.workspace import Workspace
import os

source = Path(__file__).resolve().parents[1] / "src"
assert source not in Path(microstructure.__file__).resolve().parents
assert not any("upstream/GEMsim/src" in p for p in sys.path)
assert "gem" not in sys.modules
provider = GEMsim(Workspace(Path(os.environ["MORQEX_WORKSPACE"])))
data = provider.random_field(shape=(4, 5, 6), fractions=(.3, .6, .1),
    length_scale=(1.,)*3, seed=7, spacing=(1e-6,)*3,
    phase_kinds={0: "pore", 1: "active", 2: "cbd"})
mesh = provider.mesh(data)
assert np.all(mesh.signed_volumes() > 0)
volume_path = provider.to_volume(data).save(Path.cwd() / "installed.npz")
assert provider.load(volume_path, phase_kinds=data.phase_kinds).fingerprint == data.fingerprint
mesh.save(Path.cwd() / "installed.vtu")
mesh.save_comsol(Path.cwd() / "installed.mphtxt")
from morqex.microstructure.simulate.electrochemistry import prepare_case
prepare_case(provider.to_volume(data), Path.cwd() / "native_case", {0: 6, 1: 5, 2: 7})
assert files("morqex.microstructure").joinpath("resources/echem_microstructure.inputs").is_file()
print(json.dumps({"installed_module": microstructure.__file__, "no_external_gem_dependency": True,
                  "tetrahedra": len(mesh.cells), "formats": ["npz", "vtu", "mphtxt"],
                  "native_input_template": "verified; engine not executed"}, indent=2))

