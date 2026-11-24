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
from morqex.mechanics import TetMesh, ElasticitySolver, isotropic_stiffness
from morqex.transport import VoxelDiffusion
from morqex.concentration import ConcentrationBasis, ReducedDiffusion
from morqex.coupling import bind_chemical_modes, CoupledTransport
from morqex.energy import ResponseBuilder, EnergyResponse
v = provider.volume_type(np.ones((2,)*3, dtype=np.uint8), spacing=(1e-6,)*3)
diffusion = VoxelDiffusion(v, active_phases=[1], diffusivity=1e-14)
basis = ConcentrationBasis.spectral(diffusion, 4)
solver = ElasticitySolver(TetMesh.from_volume(v), isotropic_stiffness(1e9, .25))
response = ResponseBuilder(solver, bind_chemical_modes(solver, diffusion, basis, np.array([1e-6]*3+[0.]*3))).exact()
ws = Workspace(Path(os.environ["MORQEX_WORKSPACE"]))
cached = response.save(ws, str(Path.cwd().relative_to(ws.root)/"installed_operator"))
restored = EnergyResponse.load(cached, expected_fingerprint=response.metadata["fingerprint"])
model = CoupledTransport(restored, ReducedDiffusion(diffusion, basis), chemical_curvature=.1,
                         reference_concentration=20000., maximum_concentration=40000.)
state = model.initialize(np.array([10., 30., -20., 10.]))
state, diagnostic = model.step(state, .5)
assert abs(diagnostic.energy_balance_density) < 1e-7
checkpoint = model.save_state(state, ws, str(Path.cwd().relative_to(ws.root)/"installed_state.json"))
assert model.load_state(checkpoint).time == .5
from morqex.certification import CertifiedControlledDiffusion
from morqex.tail import QuantileTree
patterns = np.column_stack([diffusion.face_normals[:, 0] > .5, diffusion.face_normals[:, 1] > .5]).astype(float)
response_basis = ConcentrationBasis.boundary_resolvent(diffusion, 3, patterns)
certified = CertifiedControlledDiffusion(diffusion, response_basis, patterns,
    reference_concentration=20000., maximum_concentration=40000.)
checked_state = certified.initialize(np.zeros(len(diffusion.weights)), error_tolerance=2.)
checked_state = certified.step(checked_state, .5, [1e-6, 0.], error_tolerance=2.)
tail = QuantileTree(restored).query(np.r_[state.macro_strain, state.coefficients], absolute_tolerance=1000.)
assert tail.converged and checked_state.error.total_bound <= 2.
print(json.dumps({"installed_module": microstructure.__file__, "no_external_gem_dependency": True,
                  "tetrahedra": len(mesh.cells), "formats": ["npz", "vtu", "mphtxt"],
                  "native_input_template": "verified; engine not executed",
                  "installed_linear_core": "FV + Tet4 + modal energy + coupling + operator cache + restart",
                  "coupling_energy_balance_J_m3": diagnostic.energy_balance_density}, indent=2))
print(json.dumps({"installed_certified_Fick_error_bound": checked_state.error.total_bound,
                  "installed_quantile_tree_converged": tail.converged}, indent=2))

