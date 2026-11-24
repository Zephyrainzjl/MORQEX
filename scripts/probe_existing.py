"""Bounded independent probes of current ZynJoule; record failures without modifying it."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ZynJoule/src"))


def main():
    report = {"scope": "tiny diagnostic probes only; no production solver certification", "checks": {}}
    def check(name, func):
        try:
            report["checks"][name] = {"executed": True, **func()}
        except Exception as exc:
            report["checks"][name] = {"executed": False, "error": repr(exc), "traceback": traceback.format_exc()}

    def fem():
        from zynjoule.zynsim.core import box_tetrahedral_mesh, DirichletBC
        from zynjoule.zynsim.fem import LinearElasticityProblem
        mesh = box_tetrahedral_mesh((1., 1., 1.), (2, 2, 2))
        exterior = np.unique(mesh.exterior_faces())
        eps = .001
        solution = LinearElasticityProblem(mesh, 100., .25,
            dirichlet=(DirichletBC.vector(exterior, value=mesh.nodes[exterior]*eps),),
            eigenstrain=np.array([eps, eps, eps, 0., 0., 0.]), backend="python").solve()
        return {"stress_max_abs": float(np.max(np.abs(solution.stress))),
                "displacement_error": float(np.max(np.abs(solution.displacement-mesh.nodes*eps))),
                "residual": solution.diagnostics.residual_norm}

    def p2d():
        from zynjoule.zynsim.battery.p2d import P2DModel, P2DDiscretization, reference_graphite_nmc811_parameters
        p = reference_graphite_nmc811_parameters(discretization=P2DDiscretization(2, 2, 2, 3, 3))
        model = P2DModel(p)
        state = model.initialize(soc=.5)
        next_state = model.step(state, .1, .01)
        return {"voltage_V": float(next_state.terminal_voltage_V),
                "diagnostics": asdict(next_state.diagnostics)}

    def spectral():
        from zynjoule.zynsim.battery.mechanics import SpectralCathodeDegradationSolver, CathodeSpectralConfig, CathodeMechanicalMultiscaleModel
        solver = SpectralCathodeDegradationSolver(np.ones((3, 3, 3), dtype=int),
            config=CathodeSpectralConfig(fft_workers=1, mechanical_iterations=10))
        state = solver.initialize(theta=.4)
        coupling = CathodeMechanicalMultiscaleModel(solver)
        # Construct a legitimate bounded distribution; a clipped uniform shift need not preserve its target mean.
        coupling.states[0].theta.flat[:13] = .01
        coupling.states[0].theta.flat[13:] = .99
        coupling.synchronize_theta(.95)
        synchronized = float(coupling.states[0].theta.mean())
        return {"uniform_free_stress_max_abs_Pa": float(np.max(np.abs(state.stress_Pa))),
                "sync_requested_mean": .95, "sync_actual_mean": synchronized,
                "sync_absolute_mean_error": abs(synchronized-.95)}

    def variable_stiffness_mu():
        from zynjoule.zynsim.battery.mechanics import SpectralCathodeDegradationSolver, CathodeSpectralConfig
        solver = SpectralCathodeDegradationSolver(np.ones((3, 3, 3), dtype=int),
            config=CathodeSpectralConfig(fft_workers=1, mechanical_iterations=10))
        material = solver.material
        theta = np.full((3, 3, 3), .5)
        # Hold total strain fixed; compare local free-energy derivative to implemented mechanical mu.
        total_strain = np.zeros(theta.shape+(3,3))
        def energy(x):
            elastic = total_strain-solver._eigenstrain(x, np.zeros_like(x))
            sigma = material.transversely_isotropic_stress(elastic, solver.c_axes, x, 298.15, np.ones_like(x))
            return .5*np.sum(elastic*sigma, axis=(-2,-1)), sigma
        h = 1e-6
        psi_plus, _ = energy(theta+h)
        psi_minus, _ = energy(theta-h)
        _, stress = energy(theta)
        mu_fd = (psi_plus-psi_minus)/(2*h)/material.maximum_lithium_concentration_mol_m3
        mu_implemented = solver._chemical_potential(theta, stress, 298.15)-material.chemical_potential_J_mol(theta, 298.15)
        return {"mechanical_mu_energy_derivative_J_mol": float(mu_fd.mean()),
                "mechanical_mu_implemented_J_mol": float(mu_implemented.mean()),
                "absolute_difference_J_mol": float(np.max(np.abs(mu_fd-mu_implemented))),
                "note": "fixed strain, uniform concentration; gradient term vanishes"}

    check("tet4_uniform_free_eigenstrain", fem)
    check("p2d_one_small_step", p2d)
    check("spectral_uniform_and_sync", spectral)
    check("concentration_dependent_stiffness_chemical_potential", variable_stiffness_mu)
    out = Path(os.environ["MORPH_WORKSPACE"]) / "evidence/existing_probes.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
