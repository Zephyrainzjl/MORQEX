"""Energy-consistent diffusion/elasticity and chemical-reservoir reaction.

This is a verified microstructure building block, not a DFN/full-cell solver.
Free chemistry is convex quadratic about a stated reference concentration.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .energy import _digest
from .transport import ConcentrationInfeasible

FARADAY = 96485.33212
GAS_CONSTANT = 8.31446261815324


def bind_chemical_modes(solver, diffusion, basis, expansion):
    mesh = solver.mesh
    if mesh.voxel_ids is None: raise ValueError("FV/Tet coupling requires parent-voxel provenance")
    volume = diffusion.volume
    expected_box = np.array([volume.origin, np.asarray(volume.origin)+np.asarray(volume.spacing)*volume.data.shape])[:, ::-1]
    if not np.allclose(mesh.box, expected_box, rtol=1e-12, atol=0.): raise ValueError("Mechanical and diffusion reference geometries differ")
    if mesh.voxel_ids.max() >= volume.data.size: raise ValueError("Parent voxel IDs outside diffusion geometry")
    local = np.full(volume.data.size, -1, dtype=int); local[diffusion.voxel_ids] = np.arange(len(diffusion.weights))
    mapping = local[mesh.voxel_ids]
    active = mapping >= 0
    represented = np.bincount(mapping[active], weights=mesh.volumes[active], minlength=len(diffusion.weights))
    if not np.allclose(represented, diffusion.volumes, rtol=1e-11, atol=0.):
        raise ValueError("Mechanical quadrature and diffusion inventories do not represent the same active volumes")
    beta = np.broadcast_to(np.asarray(expansion, dtype=float), (len(mesh.cells), 6))
    modes = np.zeros((len(mesh.cells), 6, basis.values.shape[1]))
    modes[active] = beta[active, :, None]*basis.values[mapping[active], None, :]
    return modes


@dataclass
class CoupledState:
    time: float
    coefficients: np.ndarray
    macro_strain: np.ndarray
    free_energy_density: float
    inventory_mol: float


@dataclass
class CoupledDiagnostics:
    iterations: int
    residual: float
    inventory_error_mol: float
    diffusion_dissipation_density: float
    reaction_dissipation_density: float
    external_chemical_power_density: float
    mechanical_work_density: float
    backward_euler_numerical_dissipation_density: float
    energy_balance_density: float
    inward_current_A: float


class CoupledTransport:
    def __init__(self, response, reduced_diffusion, *, chemical_curvature, reference_concentration,
                 maximum_concentration, temperature=298.15, bounds_policy="sufficient"):
        if chemical_curvature <= 0 or not np.isfinite(chemical_curvature) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Positive chemical curvature and temperature required")
        if response.dimension != reduced_diffusion.mass.shape[0]+6:
            raise ValueError("Mechanical and concentration mode dimensions differ")
        if bounds_policy not in {"sufficient", "exact"}: raise ValueError("Unknown concentration feasibility policy")
        self.response, self.diffusion = response, reduced_diffusion
        self.kappa, self.temperature = chemical_curvature, temperature
        self.mass = reduced_diffusion.mass
        self.mobility = reduced_diffusion.stiffness/chemical_curvature
        self.hessian = self.kappa*self.mass+response.H[6:, 6:]
        self.mu_jacobian = np.linalg.solve(self.mass, self.hessian)
        n = len(reduced_diffusion.full.weights)
        self.reference = np.broadcast_to(reference_concentration, (n,)).copy()
        self.maximum = np.broadcast_to(maximum_concentration, (n,)).copy()
        if not np.isfinite([self.reference, self.maximum]).all() or np.any(self.reference <= 0) or np.any(self.reference >= self.maximum):
            raise ValueError("Reference concentration must lie strictly inside physical bounds")
        self.bounds_policy = bounds_policy
        self.mode_suprema = np.max(np.abs(reduced_diffusion.basis.values), axis=0)
        self.margin = float(np.min(np.minimum(self.reference, self.maximum-self.reference)))
        self.inventory_direction = reduced_diffusion.full.weights @ reduced_diffusion.basis.values
        self.baseline_inventory = float(reduced_diffusion.full.volumes @ self.reference)
        self.reactive = reduced_diffusion.full.reactive
        self.trace = reduced_diffusion.basis.values[reduced_diffusion.full.face_cells[self.reactive]]
        self.boundary = reduced_diffusion.flux_operator[:, self.reactive]
        self.area_weight = reduced_diffusion.full.face_areas[self.reactive]/reduced_diffusion.full.rve_volume
        self.fingerprint = _digest(self.mass, self.mobility, response.H, response.h, self.reference, self.maximum,
            self.trace, self.boundary, self.area_weight, reduced_diffusion.basis.values,
            np.array([self.kappa, self.temperature]), np.frombuffer(
                (response.metadata.get("fingerprint", "")+bounds_policy+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode(), dtype=np.uint8))

    def _feasible(self, a):
        if not np.isfinite(a).all(): return False
        if self.bounds_policy == "sufficient":
            return self.mode_suprema @ np.abs(a) <= self.margin*(1+1e-12)
        c = self.reference+self.diffusion.basis.decode(a)
        return bool(np.all(c >= 0) and np.all(c <= self.maximum))

    def energy(self, a, macro_strain):
        return float(self.response.energy_density(np.r_[macro_strain, a])+.5*self.kappa*a @ self.mass @ a)

    def initialize(self, coefficients, *, macro_strain=None):
        a = np.asarray(coefficients, dtype=float).copy()
        e = np.zeros(6) if macro_strain is None else np.asarray(macro_strain, dtype=float).copy()
        if a.shape != (len(self.mass),) or e.shape != (6,) or not np.isfinite(e).all() or not self._feasible(a):
            raise ConcentrationInfeasible("Initial concentration violates the chosen exact/sufficient feasibility policy")
        return CoupledState(0., a, e, self.energy(a, e),
                            self.baseline_inventory+self.diffusion.full.rve_volume*self.inventory_direction @ a)

    def step(self, state, dt, *, macro_strain=None, flux=None, exchange_flux=None, reservoir_potential=None,
             tolerance=1e-11, maximum_iterations=40):
        if not np.isfinite(dt) or dt <= 0: raise ValueError("Positive time step required")
        e = state.macro_strain if macro_strain is None else np.asarray(macro_strain, dtype=float)
        if e.shape != (6,) or not np.isfinite(e).all(): raise ValueError("Finite macro Mandel strain required")
        previous = state.coefficients
        offset = self.response.H[6:, :6] @ e+self.response.h[6:]
        fixed_source = np.zeros(len(previous)) if flux is None else self.diffusion.flux_operator @ np.broadcast_to(flux, self.diffusion.full.face_areas.shape)
        reaction = exchange_flux is not None
        if reaction:
            if not np.any(self.reactive) or reservoir_potential is None: raise ValueError("Reactive faces and reservoir electrochemical potential required")
            rate = np.broadcast_to(exchange_flux, self.area_weight.shape)
            bath = np.broadcast_to(reservoir_potential, self.area_weight.shape)
            if not np.isfinite([rate, bath]).all() or np.any(rate < 0): raise ValueError("Invalid exchange flux/reservoir potential")
        else:
            rate, bath = np.zeros(len(self.area_weight)), np.zeros(len(self.area_weight))
        rt = GAS_CONSTANT*self.temperature
        def evaluate(a, jacobian=False):
            mu = np.linalg.solve(self.mass, self.hessian @ a+offset)
            affinity = bath-self.trace @ mu
            scaled = affinity/(2*rt)
            if reaction and np.any(np.abs(scaled) > 50): raise ArithmeticError("Reaction affinity exceeds validated numerical range")
            j = 2*rate*np.sinh(scaled) if reaction else np.zeros_like(rate)
            source = fixed_source+self.boundary @ j
            residual = self.mass @ (a-previous)+dt*self.mobility @ mu-dt*source
            if jacobian:
                derivative = rate*np.cosh(scaled)/rt if reaction else np.zeros_like(rate)
                matrix = self.mass+dt*self.mobility @ self.mu_jacobian+dt*(self.boundary*derivative) @ self.trace @ self.mu_jacobian
                return residual, matrix
            return residual, mu, affinity, j, source
        a = previous.copy()
        relative = np.inf
        for iteration in range(1, maximum_iterations+1):
            residual, matrix = evaluate(a, jacobian=True)
            scale = max(1., np.linalg.norm(self.mass @ previous), np.linalg.norm(self.mass @ a))
            relative = float(np.linalg.norm(residual)/scale)
            if relative <= tolerance: break
            direction = np.linalg.solve(matrix, -residual)
            alpha = 1.
            for _ in range(30):
                candidate = a+alpha*direction
                if self._feasible(candidate):
                    try:
                        new_norm = np.linalg.norm(evaluate(candidate)[0])
                        if new_norm < (1-1e-4*alpha)*np.linalg.norm(residual): break
                    except ArithmeticError:
                        pass
                alpha *= .5
            else:
                raise ConcentrationInfeasible("Constrained coupling step rejected; reduce dt or enrich the concentration space")
            a = candidate
        else:
            raise RuntimeError("Coupled Newton failed; no unconverged state returned")
        _, mu, affinity, j, source = evaluate(a)
        delta = a-previous
        diffusion_loss = float(mu @ self.mobility @ mu)
        reaction_loss = float(np.sum(self.area_weight*affinity*j))
        chemical_power = float(mu @ fixed_source+np.sum(self.area_weight*bath*j))
        mechanical_work = self.energy(previous, e)-self.energy(previous, state.macro_strain)
        numerical = float(.5*delta @ self.hessian @ delta)
        energy = self.energy(a, e)
        balance = energy-state.free_energy_density-mechanical_work-dt*(chemical_power-diffusion_loss-reaction_loss)+numerical
        inventory = self.baseline_inventory+self.diffusion.full.rve_volume*self.inventory_direction @ a
        fixed_rate = 0. if flux is None else self.diffusion.full.face_areas @ np.broadcast_to(flux, self.diffusion.full.face_areas.shape)
        inward_rate = float(fixed_rate + np.sum(self.area_weight*j)*self.diffusion.full.rve_volume)
        mass_error = float(inventory-state.inventory_mol-dt*inward_rate)
        new_state = CoupledState(state.time+dt, a.copy(), e.copy(), energy, float(inventory))
        diagnostics = CoupledDiagnostics(iteration, relative, mass_error, diffusion_loss, reaction_loss, chemical_power,
            float(mechanical_work), numerical, float(balance), FARADAY*inward_rate)
        return new_state, diagnostics

    def concentration(self, state):
        """Explicit bulk recovery: O(Nm); not called by sufficient-bound time stepping."""
        return self.reference+self.diffusion.basis.decode(state.coefficients)

    def save_state(self, state, workspace, name):
        path = workspace.path("checkpoints", name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": "morqex-coupled-state-1", "model_fingerprint": self.fingerprint,
            "time_s": state.time, "coefficients_mol_m3": state.coefficients.tolist(),
            "macro_mandel_strain": state.macro_strain.tolist(), "free_energy_density_J_m3": state.free_energy_density,
            "inventory_mol": state.inventory_mol}, indent=2))
        return path

    def load_state(self, path):
        saved = json.loads(Path(path).read_text())
        if saved["schema"] != "morqex-coupled-state-1" or saved["model_fingerprint"] != self.fingerprint:
            raise ValueError("Checkpoint does not match this physical model, basis or implementation")
        state = self.initialize(saved["coefficients_mol_m3"], macro_strain=saved["macro_mandel_strain"])
        if not np.isfinite(saved["time_s"]) or saved["time_s"] < 0:
            raise ValueError("Invalid checkpoint time")
        if not np.isclose(state.free_energy_density, saved["free_energy_density_J_m3"], rtol=1e-12, atol=1e-12) or not np.isclose(state.inventory_mol, saved["inventory_mol"], rtol=1e-12, atol=0):
            raise ValueError("Checkpoint energy/inventory is inconsistent with its physical state")
        state.time = float(saved["time_s"])
        return state
