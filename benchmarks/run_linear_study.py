"""Reproducible small 3D study: measured successes and failures are both retained.

Reference: exact FV concentration coordinates + the same qualified Tet4 model.
This is not a DFN, FE2, continuum-converged CT or material-design validation.
Run only through scripts/run_local.py; all files go to the external run directory.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import tomllib
import numpy as np
from morqex import Workspace
from morqex.gemsim import GEMsim
from morqex.mechanics import TetMesh, ElasticitySolver, isotropic_stiffness
from morqex.transport import VoxelDiffusion
from morqex.concentration import ConcentrationBasis, ReducedDiffusion
from morqex.coupling import bind_chemical_modes, CoupledTransport
from morqex.energy import ResponseBuilder
from morqex.statistics import StressMoments, StressEnvelope, principal_tension, weighted_quantile


def relative(error, reference, weights, floor=1e-30):
    w = weights.reshape((-1,)+(1,)*(error.ndim-1))
    return float(np.sqrt(np.sum(w*error**2)/max(np.sum(w*reference**2), floor)))


def snapshots(diffusion, seed, cases, steps, dt, *, held_out=False):
    rng = np.random.default_rng(seed)
    xyz = diffusion.centers/np.max(diffusion.centers)
    columns = []
    for _ in range(cases):
        directions = rng.normal(size=(3, 5))
        frequencies = rng.uniform(.3, 2.5 if held_out else 1.5, 5)
        pattern = np.cos(np.pi*(xyz @ directions)*frequencies+rng.uniform(0, 2*np.pi, 5)) @ rng.normal(size=5)
        pattern *= 1000/max(np.max(np.abs(pattern)), 1e-30)
        c = 20000.+pattern+rng.uniform(-400., 400.)
        for _ in range(steps):
            columns.append(c-20000.)
            c = diffusion.step(c, dt).concentration
    return np.column_stack(columns)


def make_model(response, diffusion, basis, cfg, *, full=False):
    return CoupledTransport(response, ReducedDiffusion(diffusion, basis),
        chemical_curvature=cfg["chemical_curvature_J_m3_mol2"],
        reference_concentration=cfg["reference_concentration_mol_m3"],
        maximum_concentration=cfg["maximum_concentration_mol_m3"], bounds_policy="exact" if full else "sufficient")


def timed(action, repeats):
    action() # warm imports/allocators; factorization has already occurred for both paths
    values = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(repeats): action()
        values.append((time.perf_counter()-start)/repeats)
    return float(np.median(values))


def study(seed, cfg, gem):
    case_started = time.perf_counter()
    start = time.perf_counter()
    geometry = gem.random_field(shape=tuple(cfg["shape_zyx"]), fractions=tuple(cfg["fractions"]),
        length_scale=(.7, .9, 1.1), seed=seed, spacing=(cfg["spacing_m"],)*3,
        phase_kinds={0: "pore", 1: "active", 2: "cbd"})
    volume = gem.to_volume(geometry)
    geometry_seconds = time.perf_counter()-start
    diffusion = VoxelDiffusion(volume, active_phases=[1], diffusivity=cfg["diffusivity_xyz_m2_s"], reactive_phases=[0])
    mesh = TetMesh.from_volume(volume, solid_phases=[1, 2])
    material = {1: isotropic_stiffness(cfg["young_active_Pa"], .25), 2: isotropic_stiffness(cfg["young_cbd_Pa"], .3)}
    solver = ElasticitySolver(mesh, np.array([material[int(p)] for p in mesh.phases]), boundary="kubc")
    beta = np.array(cfg["expansion_m3_mol"])
    n = len(diffusion.weights)
    full_basis = ConcentrationBasis(np.eye(n), diffusion.weights.copy(), "full-FV-coordinate", diffusion.n_components)
    start = time.perf_counter()
    full_builder = ResponseBuilder(solver, bind_chemical_modes(solver, diffusion, full_basis, beta))
    full_response = full_builder.exact()
    reference_operator_seconds = time.perf_counter()-start
    training = snapshots(diffusion, seed+1000, cfg["training_cases"], cfg["training_steps"], cfg["dt_s"])
    validation = snapshots(diffusion, seed+9000, cfg["validation_cases"], cfg["validation_steps"], cfg["dt_s"], held_out=True)
    rng = np.random.default_rng(seed+8000)
    strains = rng.normal(scale=2e-4, size=(6, validation.shape[1]))
    full_q = np.vstack([strains, validation]).T
    reference_stress = full_response.stress(full_q)
    reference_energy = full_response.energy_density(full_q)
    reference_mu = full_response.projected_chemical_potential(full_q, full_basis.mass)
    # Separate FEM solve verifies the response, using the existing factorization.
    direct = solver.solve(np.einsum("eim,m->ei", full_builder.modes, validation[:, 0]), macro_strain=strains[:, 0])
    direct_error = relative(full_response.stress(full_q[0])-direct.stress, direct.stress, mesh.volumes)
    assert direct_error < 1e-10
    def target_action(x):
        stress = np.einsum("eim,ms->eis", full_response.stress_operator[:, :, 6:], x*cfg["concentration_scale_mol_m3"])
        return (stress*np.sqrt(mesh.volumes/mesh.rve_volume)[:, None, None]/cfg["stress_scale_Pa"]).reshape(-1, x.shape[1])
    rows, models = [], {}
    for additional in cfg["additional_modes"]:
        count = diffusion.n_components+additional
        for name in ("geometric", "spectral", "POD", "joint-POD"):
            start = time.perf_counter()
            if name == "geometric": basis = ConcentrationBasis.geometric(diffusion, count)
            elif name == "spectral": basis = ConcentrationBasis.spectral(diffusion, count)
            else:
                basis = ConcentrationBasis.pod(diffusion, training, count,
                    concentration_scale=cfg["concentration_scale_mol_m3"],
                    target_action=target_action if name == "joint-POD" else None)
            basis_seconds = time.perf_counter()-start
            start = time.perf_counter()
            response = ResponseBuilder(solver, bind_chemical_modes(solver, diffusion, basis, beta)).exact()
            operator_seconds = time.perf_counter()-start
            a = basis.encode(validation)
            q = np.vstack([strains, a]).T
            restored = basis.decode(a)
            stress = response.stress(q)
            mu = basis.decode(response.projected_chemical_potential(q, basis.mass).T).T
            energy = response.energy_density(q)
            concentration_error = relative(restored-validation, validation, diffusion.weights)
            stress_error = relative(np.moveaxis(stress-reference_stress, 1, 0), np.moveaxis(reference_stress, 1, 0), mesh.volumes)
            energy_error = float(np.linalg.norm(energy-reference_energy)/np.linalg.norm(reference_energy))
            mu_error = relative((mu-reference_mu).T, reference_mu.T, diffusion.weights)
            p99 = np.array([weighted_quantile(principal_tension(s), mesh.volumes, .99) for s in stress])
            truth = np.array([weighted_quantile(principal_tension(s), mesh.volumes, .99) for s in reference_stress])
            p99_error = float(np.linalg.norm(p99-truth)/max(np.linalg.norm(truth), 1e3))
            macro_error = float(np.linalg.norm(response.macro_stress(q)-full_response.macro_stress(full_q))/np.linalg.norm(full_response.macro_stress(full_q)))
            moment = StressMoments.build(response)
            mean, covariance = moment.evaluate(q[0])
            weights = mesh.volumes/mesh.volumes.sum()
            field_mean = weights @ stress[0]
            centered = stress[0]-field_mean
            assert np.allclose(mean, field_mean, atol=1e-6)
            assert np.allclose(covariance, np.einsum("e,ei,ej->ij", weights, centered, centered), atol=1e-3)
            row = {"method": name, "modes": count, "concentration_deviation_L2": concentration_error,
                "stress_L2": stress_error, "elastic_energy_L2": energy_error, "mechanical_mu_L2": mu_error,
                "P99_principal_L2": p99_error, "macro_stress_L2": macro_error,
                "basis_seconds": basis_seconds, "operator_seconds_factor_reused": operator_seconds}
            rows.append(row)
            models[(name, count)] = (basis, response)
    # Offline validation ladder. These truth-dependent tests are NOT an online
    # concentration error estimator. Keep the failed low-rank comparisons above.
    ladder = []
    for count in sorted({min(n, diffusion.n_components+k) for k in (6, 15, 31, 63, n)}):
        trial = ConcentrationBasis.spectral(diffusion, count)
        values = trial.decode(trial.encode(validation))
        trial_stress = full_response.stress(np.vstack([strains, values]).T)
        trial_error = relative(np.moveaxis(trial_stress-reference_stress, 1, 0), np.moveaxis(reference_stress, 1, 0), mesh.volumes)
        ladder.append({"modes": count, "concentration_deviation_L2": relative(values-validation, validation, diffusion.weights),
            "stress_L2": trial_error, "passes_5_percent_stress": trial_error <= .05})
    assert ladder[-1]["stress_L2"] < 1e-10
    # Default is only a study choice, not a final architectural winner.
    basis, response = models[("spectral", diffusion.n_components+cfg["additional_modes"][-1])]
    a = basis.encode(validation[:, 0]); q = np.r_[strains[:, 0], a]
    ref_eigen = np.einsum("eim,m->ei", bind_chemical_modes(solver, diffusion, basis, beta), a)
    def reference_fields():
        # Identical requested output to Level C; omit extra reaction/residual
        # recovery from timed calls. Validate against solver.solve separately.
        lift = solver.affine @ q[:6]
        rhs = solver.P.T @ (solver.eigenstrain_load(ref_eigen)-solver.K @ lift)
        u = lift+solver.P @ solver.solve_reduced(rhs)
        strain = np.asarray(solver.D @ u).reshape(-1, 6)
        stress = np.einsum("eij,ej->ei", solver.stiffness, strain-ref_eigen)
        energy = .5*np.einsum("e,ei,ei->", mesh.volumes, strain-ref_eigen, stress)/mesh.rve_volume
        return u.reshape(-1, 3), strain, stress, energy
    checked = reference_fields()
    independently_checked = solver.solve(ref_eigen, macro_strain=q[:6])
    assert relative(checked[2]-independently_checked.stress, independently_checked.stress, mesh.volumes) < 1e-10
    ref_time = timed(reference_fields, cfg["timing_repeats"])
    macro_time = timed(lambda: (response.macro_stress(q), response.energy_density(q)), cfg["timing_repeats"])
    field_time = timed(lambda: (response.stress(q), response.strain(q), response.displacement(q), response.energy_density(q)), cfg["timing_repeats"])
    offline = next(r["operator_seconds_factor_reused"] for r in rows if r["method"] == "spectral" and r["modes"] == len(a))
    timing = {"reference_reused_factor_full_fields_s": ref_time, "online_macro_stress_and_energy_s": macro_time,
        "online_all_three_fields_and_energy_s": field_time,
        "full_fields_speedup": ref_time/field_time,
        "macro_vs_full_solve_ratio_different_output": ref_time/macro_time,
        "incremental_operator_offline_s": offline,
        "incremental_full_fields_break_even_queries": offline/(ref_time-field_time) if ref_time > field_time else None,
        "scope": "small fixed Tet4 system, warmed same CPU, reused factor/FP64; both full-field paths return u/strain/stress/W; reaction/residual excluded from both timed paths and checked separately; macro ratio has different output; no DFN/FE2 speedup claim"}
    scales = np.r_[np.full(6, .002), np.full(len(a), 5000.)]
    envelope = []
    for samples in (4, 16, 64, len(mesh.cells)):
        bound = StressEnvelope.build(response, samples, scales)
        result = bound.evaluate(q, threshold=1e7)
        truth = weighted_quantile(principal_tension(response.stress(q)), mesh.volumes, .99)
        assert result["quantile_lower"]-1e-6 <= truth <= result["quantile_upper"]+1e-6
        envelope.append({**result, "exact_P99_Pa": truth, "relative_interval_width": (result["quantile_upper"]-result["quantile_lower"])/max(abs(truth), 1e3)})
    reference_model = make_model(full_response, diffusion, full_basis, cfg, full=True)
    reduced_model = make_model(response, diffusion, basis, cfg)
    ref = reference_model.initialize(validation[:, 0])
    state = reduced_model.initialize(basis.encode(validation[:, 0]))
    trajectory, diagnostics = [], []
    for _ in range(cfg["coupling_steps"]):
        ref, _ = reference_model.step(ref, cfg["dt_s"])
        state, diag = reduced_model.step(state, cfg["dt_s"])
        truth = reference_model.concentration(ref)-cfg["reference_concentration_mol_m3"]
        approx = reduced_model.concentration(state)-cfg["reference_concentration_mol_m3"]
        trajectory.append({"time_s": state.time, "concentration_deviation_L2": relative(approx-truth, truth, diffusion.weights),
            "energy_density_J_m3": state.free_energy_density, "reference_energy_density_J_m3": ref.free_energy_density,
            "inventory_mol": state.inventory_mol})
        diagnostics.append(asdict(diag))
    # Open system: positive reservoir work can increase free energy while all losses stay positive.
    state = reduced_model.initialize(np.zeros(len(a)))
    reservoir = []
    for _ in range(6):
        state, diag = reduced_model.step(state, cfg["dt_s"], exchange_flux=1e-5, reservoir_potential=300.)
        reservoir.append({"time_s": state.time, "energy_density_J_m3": state.free_energy_density, **asdict(diag)})
    np.savez_compressed(f"seed_{seed}_fields.npz", labels=volume.data, points=mesh.points, cells=mesh.cells,
        stress_reference=reference_stress[0], stress_reconstructed=response.stress(q),
        concentration_reference=validation[:, 0], concentration_reconstructed=basis.decode(a),
        training=training, validation=validation)
    result = {"seed": seed, "geometry_fingerprint": geometry.fingerprint, "mesh_fingerprint": mesh.fingerprint,
        "active_voxels": n, "diffusion_components": diffusion.n_components, "elements": len(mesh.cells),
        "nodes": len(mesh.points), "geometry_seconds": geometry_seconds, "assembly_seconds": solver.assembly_seconds,
        "factorization_seconds": solver.factorization_seconds, "reference_operator_seconds": reference_operator_seconds,
        "independent_FEM_stress_relative_error": direct_error, "candidate_comparison": rows, "timing": timing,
        "tail_enclosures": envelope, "closed_coupling": trajectory, "closed_diagnostics": diagnostics, "reservoir_reaction": reservoir,
        "validation_mode_ladder": ladder,
        "response_fingerprint": response.metadata["fingerprint"], "wall_seconds": time.perf_counter()-case_started}
    return result


def figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for method in ("geometric", "spectral", "POD", "joint-POD"):
        for ax, key in zip(axes, ("concentration_deviation_L2", "stress_L2", "P99_principal_L2")):
            counts = sorted({r["modes"] for case in results for r in case["candidate_comparison"] if r["method"] == method})
            y = [np.mean([r[key] for case in results for r in case["candidate_comparison"] if r["method"] == method and r["modes"] == m]) for m in counts]
            ax.plot(counts, y, "o-", label=method); ax.set_title(key); ax.set_xlabel("total modes including mass"); ax.set_ylabel("relative error"); ax.grid(alpha=.3)
    axes[-1].legend(fontsize=8); fig.savefig("candidate_comparison.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for case in results:
        x = case["closed_coupling"]
        axes[0].plot([r["time_s"] for r in x], [r["energy_density_J_m3"] for r in x], label=str(case["seed"]))
        axes[1].plot([r["time_s"] for r in x], [r["concentration_deviation_L2"] for r in x], label=str(case["seed"]))
    axes[0].set_ylabel("closed free energy density [J/m3]"); axes[1].set_ylabel("concentration deviation error")
    for ax in axes: ax.set_xlabel("time [s]"); ax.grid(alpha=.3); ax.legend(title="GEMsim seed")
    fig.savefig("coupling_validation.png", dpi=180); plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    workspace = Workspace(Path(os.environ["MORQEX_WORKSPACE"]))
    if workspace.root not in Path.cwd().parents: raise ValueError("Run directory must be in the external workspace")
    cfg_bytes = args.config.read_bytes(); cfg = tomllib.loads(cfg_bytes.decode())
    if cfg["schema"] != "morqex-linear-study-1": raise ValueError("Unsupported protocol")
    if np.prod(cfg["shape_zyx"]) > 216 or len(cfg["seeds"]) > 5: raise ValueError("Study exceeds the qualified small-system budget")
    source = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((source/"src/morqex").glob("*.py"))}
    hashes.update({str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((source/"src/morqex/mechanics").glob("*.py"))})
    hashes[str(Path(__file__).relative_to(source))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {"schema": "morqex-linear-study-result-1", "config": cfg,
        "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(), "source_sha256": hashes,
        "scope": "synthetic independent small GEMsim geometries; quadratic chemistry, fixed-C KUBC, no DFN, CT or established novelty",
        "cases": []}
    for seed in cfg["seeds"]:
        case = study(seed, cfg, GEMsim(workspace)); result["cases"].append(case)
        Path("linear_study.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"seed": seed, "active_voxels": case["active_voxels"], "completed": True}), flush=True)
    figures(result["cases"])
    print(json.dumps({"result": str(Path.cwd()/"linear_study.json"), "cases": len(result["cases"])}))


if __name__ == "__main__": main()
