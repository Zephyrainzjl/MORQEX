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


from morqex.gradient_transport import QuadraticGradientSystem
from morqex.adaptive_transport import AdaptiveGradientTransport
identity = ConcentrationBasis(np.eye(len(diffusion.weights)),diffusion.weights.copy(),"identity",diffusion.n_components)
full_response = ResponseBuilder(solver,bind_chemical_modes(solver,diffusion,identity,np.array([1e-6]*3+[0.]*3))).exact()
gradient_system = QuadraticGradientSystem(full_response,diffusion,identity,patterns,chemical_curvature=.1,
    reference_concentration=20000.,maximum_concentration=40000.)
adaptive = AdaptiveGradientTransport(gradient_system,response_basis,maximum_modes=8,certificate_method="spectral",error_modes=2)
gradient_state = adaptive.initialize(np.zeros(gradient_system.n),error_tolerance=.5)
gradient_state,gradient_diag = adaptive.step(gradient_state,.5,[1e-5,0.],error_tolerance=.5)
adaptive_path = adaptive.save_checkpoint(gradient_state,ws,str(Path.cwd().relative_to(ws.root)/"installed_adaptive"))
restarted,reloaded = AdaptiveGradientTransport.load_checkpoint(gradient_system,adaptive_path)
assert reloaded.time == gradient_state.time and reloaded.error.total_bound == gradient_state.error.total_bound
assert abs(gradient_diag.dynamics.energy_balance) < 1e-7
print(json.dumps({"installed_coupled_adaptive":"energy metric + signed error dynamics + checkpoint",
    "energy_error_bound":reloaded.error.total_bound,"additional_error_coordinates":restarted.model.error_space.count}))

from morqex.sparse_coupled import SparseChemoelasticSystem
from morqex.streamed_rom import StreamedEnergyROM,OnlineEnergyKernel,StressLocalization
sparse_system=SparseChemoelasticSystem(solver,diffusion,np.array([1e-6]*3+[0.]*3),patterns,
    chemical_curvature=.1,reference_concentration=20000.,maximum_concentration=40000.)
sparse_rom=StreamedEnergyROM(sparse_system,response_basis)
sparse_state=sparse_rom.initialize(np.zeros(sparse_system.n),sparse_system,error_tolerance=1.)
compact=sparse_rom.detach_online();sparse_state,_=compact.step(sparse_state,.5,[1e-5,0.],error_tolerance=1.)
compact_path=compact.save(ws,str(Path.cwd().relative_to(ws.root)/"installed_compact"),sparse_state)
loaded,sparse_restarted=OnlineEnergyKernel.load(compact_path)
assert not hasattr(loaded,'basis') and not hasattr(loaded,'system')
np.testing.assert_array_equal(loaded.macro_stress(sparse_restarted),compact.macro_stress(sparse_state))
print(json.dumps({"installed_sparse_coupled":"mixed/Schur energy action + detached online checkpoint",
    "compact_kernel_bytes":loaded.online_array_bytes(),"energy_error_bound":sparse_restarted.error_bound}))
# Load a genuinely large source-built checkpoint using only the installed wheel.
scaling_manifest=json.loads((ws.root/'reports/sparse_scaling_latest.json').read_text())
scaling_data=json.loads((Path(scaling_manifest['run'])/'sparse_scaling.json').read_text())
large_case=next(c for c in scaling_data['cases'] if c['size']==18 and c['seed']==306)
large_kernel,large_state=OnlineEnergyKernel.load(large_case['compact_kernel_path'])
solid_fraction=large_case['tetrahedra']/(6*large_case['size']**3)
expected_mean=np.array(large_case['selected_fields']['sample_volume_mean'])*solid_fraction
np.testing.assert_allclose(large_kernel.macro_stress(large_state),expected_mean,rtol=1e-10,atol=1e-5)
assert not hasattr(large_kernel,'basis') and not hasattr(large_kernel,'system')
print(json.dumps({'installed_large_source_checkpoint_N':large_case['N'],'modes':large_kernel.dimension,
    'source_basis_or_solver_loaded':False,'macro_mean_cross_check':'passed'}))
