"""Boundary-driven load domain, discrete certification, and query-wise tail study.

All material/geometry values are synthetic. New inputs are separated from the
retained arbitrary-initial-field failure domain. No DFN or originality claim.
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
from morqex.concentration import ConcentrationBasis, ControlledDiffusion
from morqex.coupling import bind_chemical_modes
from morqex.energy import ResponseBuilder
from morqex.certification import DiffusionCertificate, StressErrorTransfer
from morqex.statistics import principal_tension, weighted_quantile, StressEnvelope
from morqex.tail import QuantileTree


def patterns_for(diffusion):
    xyz = diffusion.face_centers
    extent = np.ptp(xyz, axis=0); normalized = (xyz-xyz.min(axis=0))/np.maximum(extent, 1e-30)
    patterns = np.column_stack([np.ones(len(xyz)), 2*normalized-1,
        np.exp(-20*np.sum((normalized-np.array([.15, .8, .75]))**2, axis=1))])
    patterns *= diffusion.reactive[:, None]
    if patterns[:, -1].max() <= 0: raise ValueError("No active/pore reaction surface")
    patterns[:, -1] /= patterns[:, -1].max()
    return patterns


def forcing(cfg, seed, *, validation=False, out_of_library=False):
    rng = np.random.default_rng(seed)
    result = []
    for case in range(cfg["validation_cases"] if validation else cfg["training_cases"]):
        t = (np.arange(cfg["steps"])+1)*cfg["dt_s"]
        phase = rng.uniform(0, 2*np.pi, 4)
        frequency = rng.uniform(.06, .15, 4) if not validation else rng.uniform(.2, .7, 4)
        u = np.zeros((len(t), 5))
        u[:, 0] = 1+.3*np.sin(t*frequency[0]+phase[0])
        u[:, 1:4] = .5*np.sin(t[:, None]*frequency[None, 1:]+phase[None, 1:])
        if validation:
            u *= np.where((t//4).astype(int) % 2 == 0, 1., .2)[:, None]
        if out_of_library:
            u[:, :4] *= .2
            u[:, 4] = 3*np.where(t < t[-1]/2, 1., -.25)
        result.append(cfg["flux_scale_mol_m2_s"]*u)
    return result


def full_history(diffusion, patterns, inputs, dt, reference):
    columns = []
    for profile in inputs:
        c = np.full(len(diffusion.weights), reference)
        trajectory = []
        for u in profile:
            c = diffusion.step(c, dt, flux=patterns @ u).concentration
            trajectory.append(c-reference)
        columns.append(np.array(trajectory))
    return columns


def case_study(seed, cfg, base, workspace):
    start_case = time.perf_counter()
    provider = GEMsim(workspace)
    geometry = provider.random_field(shape=tuple(base["shape_zyx"]), fractions=tuple(base["fractions"]),
        length_scale=(.7, .9, 1.1), seed=seed, spacing=(base["spacing_m"],)*3,
        phase_kinds={0: "pore", 1: "active", 2: "cbd"})
    volume = provider.to_volume(geometry)
    diffusion = VoxelDiffusion(volume, active_phases=[1], diffusivity=base["diffusivity_xyz_m2_s"], reactive_phases=[0])
    patterns = patterns_for(diffusion)
    mesh = TetMesh.from_volume(volume, solid_phases=[1, 2])
    materials = {1: isotropic_stiffness(base["young_active_Pa"], .25), 2: isotropic_stiffness(base["young_cbd_Pa"], .3)}
    solver = ElasticitySolver(mesh, np.array([materials[int(p)] for p in mesh.phases]))
    n = len(diffusion.weights)
    full_basis = ConcentrationBasis(np.eye(n), diffusion.weights.copy(), "full-FV", diffusion.n_components)
    beta = np.array(base["expansion_m3_mol"])
    start = time.perf_counter()
    full_builder = ResponseBuilder(solver, bind_chemical_modes(solver, diffusion, full_basis, beta))
    full_response = full_builder.exact()
    full_offline_seconds = time.perf_counter()-start
    transfer = StressErrorTransfer.build(full_response, diffusion.weights)
    training_inputs = forcing(cfg, seed+1000)
    validation_inputs = forcing(cfg, seed+9000, validation=True)
    ood_inputs = forcing({**cfg, "validation_cases": 1}, seed+7000, validation=True, out_of_library=True)
    dt = cfg["dt_s"]; reference = base["reference_concentration_mol_m3"]
    training = np.concatenate(full_history(diffusion, patterns, training_inputs, dt, reference)).T
    in_truth = full_history(diffusion, patterns, validation_inputs, dt, reference)
    out_truth = full_history(diffusion, patterns, ood_inputs, dt, reference)
    # Independent FEM cross-check uses a new RHS solve, not operator self-data.
    c_check = in_truth[0][-1]
    direct = solver.solve(np.einsum("eim,m->ei", full_builder.modes, c_check))
    q_check = np.r_[np.zeros(6), c_check]
    direct_error = np.linalg.norm(full_response.stress(q_check)-direct.stress)/max(np.linalg.norm(direct.stress), 1e-30)
    assert direct_error < 1e-10
    compare, tails, selected_fields = [], [], None
    for additional in cfg["additional_modes"]:
        count = diffusion.n_components+additional
        for method in ("spectral", "mass-POD", "boundary-resolvent", "resolvent-rebuilt-with-hotspot"):
            start = time.perf_counter()
            try:
                if method == "spectral": basis = ConcentrationBasis.spectral(diffusion, count)
                elif method == "mass-POD": basis = ConcentrationBasis.pod(diffusion, training, count, concentration_scale=base["concentration_scale_mol_m3"])
                else:
                    basis = ConcentrationBasis.boundary_resolvent(diffusion, count,
                        patterns if method == "resolvent-rebuilt-with-hotspot" else patterns[:, :4],
                        shifts=cfg["resolvent_shifts_per_s"], moments=cfg["resolvent_moments"])
            except ValueError as error:
                compare.append({"method": method, "modes": count, "status": "unavailable", "reason": str(error)})
                continue
            basis_seconds = time.perf_counter()-start
            start = time.perf_counter()
            response = ResponseBuilder(solver, bind_chemical_modes(solver, diffusion, basis, beta)).exact()
            operator_seconds = time.perf_counter()-start
            start = time.perf_counter()
            certificate = DiffusionCertificate(diffusion, basis, patterns)
            certificate_offline = time.perf_counter()-start
            model = ControlledDiffusion(diffusion, basis, patterns, reference_concentration=reference,
                maximum_concentration=base["maximum_concentration_mol_m3"])
            for domain, profiles, truths in (("held-out-time", validation_inputs, in_truth), ("held-out-surface-hotspot", ood_inputs, out_truth)):
                errors_c, errors_s, errors_p, bounds, actuals, mass_errors, history = [], [], [], [], [], [], []
                step_seconds, cert_seconds = [], []
                stress_actual_norm2, stress_error_norm2 = 0., 0.
                for profile, truth_trajectory in zip(profiles, truths):
                    a = np.zeros(count)
                    bound = certificate.initialize(np.zeros(n), a)
                    for step, (u, truth) in enumerate(zip(profile, truth_trajectory)):
                        start = time.perf_counter(); updated = model.step(a, dt, u); step_seconds.append(time.perf_counter()-start)
                        start = time.perf_counter(); bound = certificate.step(bound, a, updated, dt, u); cert_seconds.append(time.perf_counter()-start)
                        actual = basis.decode(updated); error = truth-actual
                        error_norm = float(np.sqrt(diffusion.weights @ error**2))
                        bounds.append(bound.total_bound); actuals.append(error_norm)
                        errors_c.append(error_norm/max(float(np.sqrt(diffusion.weights @ truth**2)), 1.))
                        q = np.r_[np.zeros(6), updated]; q_truth = np.r_[np.zeros(6), truth]
                        stress, ref_stress = response.stress(q), full_response.stress(q_truth)
                        difference = stress-ref_stress
                        abs_stress_error = float(np.sqrt(np.einsum("e,ei,ei->", mesh.volumes/mesh.rve_volume, difference, difference)))
                        stress_norm = float(np.sqrt(np.einsum("e,ei,ei->", mesh.volumes/mesh.rve_volume, ref_stress, ref_stress)))
                        assert abs_stress_error <= transfer.stress_l2_bound(bound.total_bound)+1e-4
                        assert error_norm <= bound.total_bound+1e-8
                        errors_s.append(abs_stress_error/max(stress_norm, 1e3))
                        stress_error_norm2 += abs_stress_error**2; stress_actual_norm2 += stress_norm**2
                        p99 = weighted_quantile(principal_tension(stress), mesh.volumes, .99)
                        exact_p99 = weighted_quantile(principal_tension(ref_stress), mesh.volumes, .99)
                        errors_p.append(abs(p99-exact_p99)/max(abs(exact_p99), 1e3))
                        assert abs(p99-exact_p99) <= transfer.quantile_bound(bound.total_bound)+1e-4
                        for component in range(diffusion.n_components):
                            group = diffusion.components == component
                            mass_errors.append(abs(float(diffusion.volumes[group] @ error[group])))
                        history.append({"time_s": (step+1)*dt, "concentration_error_M": error_norm,
                            "concentration_bound_M": bound.total_bound, "stress_error_relative": errors_s[-1],
                            "P99_error_relative": errors_p[-1]})
                        a = updated
                compare.append({"method": method, "modes": count, "status": "measured", "domain": domain,
                    "stress_trajectory_L2": float(np.sqrt(stress_error_norm2/max(stress_actual_norm2, 1.))),
                    "worst_stress_relative": max(errors_s), "worst_concentration_relative": max(errors_c),
                    "worst_P99_relative": max(errors_p), "worst_concentration_bound": max(bounds),
                    "worst_actual_concentration_error": max(actuals),
                    "bound_coverage": float(np.mean(np.array(actuals) <= np.array(bounds)+1e-8)),
                    "bound_median_effectivity": float(np.median(np.array(bounds)/np.maximum(actuals, 1e-12))),
                    "mass_error_mol_max": max(mass_errors), "gap_lower_per_s": certificate.gap_lower,
                    "basis_seconds": basis_seconds, "operator_seconds": operator_seconds,
                    "certificate_offline_seconds": certificate_offline,
                    "online_step_median_s": float(np.median(step_seconds)), "certificate_step_median_s": float(np.median(cert_seconds)),
                    "certified_absolute_tolerance_passed": bool(max(bounds) <= cfg["concentration_absolute_tolerance_mol_m3"]),
                    "history": history})
                if domain == "held-out-time" and additional == cfg["additional_modes"][1] and method == "boundary-resolvent":
                    # Same q, same stored S: isolate risk-sampling error from concentration truncation.
                    scales = np.r_[np.full(6, .002), np.full(count, 5000.)]
                    start = time.perf_counter(); tree = QuantileTree(response, scales=scales); tree_offline = time.perf_counter()-start
                    full_start = time.perf_counter(); exact = weighted_quantile(principal_tension(response.stress(q)), mesh.volumes, .99)
                    exact_time = time.perf_counter()-full_start
                    old = StressEnvelope.build(response, 64, scales).evaluate(q)
                    for budget in cfg["quantile_node_budgets"]:
                        start = time.perf_counter(); result = tree.query(q, maximum_nodes=budget,
                            relative_tolerance=cfg["quantile_relative_width"], absolute_tolerance=cfg["quantile_absolute_width_Pa"])
                        elapsed = time.perf_counter()-start
                        assert result.lower-1e-5 <= exact <= result.upper+1e-5
                        tails.append({**asdict(result), "node_budget": budget, "exact_P99_Pa": exact,
                            "query_seconds": elapsed, "offline_seconds": tree_offline,
                            "exact_full_P99_seconds": exact_time,
                            "interval_width_relative": (result.upper-result.lower)/max(abs(exact), 1e3),
                            "previous_64_sample_interval_width_relative": (old["quantile_upper"]-old["quantile_lower"])/max(abs(exact), 1e3)})
                    selected_fields = {"concentration": actual, "reference_concentration": truth,
                        "stress": stress, "reference_stress": ref_stress, "bound": bound.total_bound}
    np.savez_compressed(f"controlled_seed_{seed}.npz", labels=volume.data, points=mesh.points, cells=mesh.cells,
        centers=mesh.centers, active_ids=diffusion.voxel_ids, spacing_m=volume.spacing, origin_m=volume.origin, **(selected_fields or {}))
    return {"seed": seed, "geometry_fingerprint": geometry.fingerprint, "active_voxels": n,
        "diffusion_components": diffusion.n_components, "elements": len(mesh.cells),
        "independent_FEM_stress_difference": float(direct_error), "full_operator_offline_s": full_offline_seconds,
        "stress_transfer_constants": asdict(transfer), "comparison": compare, "quantile_queries": tails,
        "wall_seconds": time.perf_counter()-start_case}


def figures(cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for method in ("spectral", "mass-POD", "boundary-resolvent", "resolvent-rebuilt-with-hotspot"):
        for ax, domain in zip(axes, ("held-out-time", "held-out-surface-hotspot")):
            counts = sorted({r["modes"]-c["diffusion_components"] for c in cases for r in c["comparison"] if r["status"] == "measured" and r["method"] == method})
            means = [np.mean([r["stress_trajectory_L2"] for c in cases for r in c["comparison"] if r["status"] == "measured" and r["method"] == method and r["domain"] == domain and r["modes"]-c["diffusion_components"] == count]) for count in counts]
            ax.semilogy(counts, means, "o-", label=method); ax.set_title(domain); ax.set_xlabel("nonconstant modes"); ax.set_ylabel("mean stress trajectory L2 error"); ax.grid(alpha=.3)
    axes[-1].legend(); fig.savefig("controlled_accuracy.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for case in cases:
        row = next(r for r in case["comparison"] if r["status"] == "measured" and r["method"] == "boundary-resolvent" and r["domain"] == "held-out-time" and r["modes"] == case["diffusion_components"]+8)
        history = row["history"][:40]
        axes[0].semilogy([x["time_s"] for x in history], [max(x["concentration_error_M"], 1e-14) for x in history], label=f"actual {case['seed']}")
        axes[0].semilogy([x["time_s"] for x in history], [max(x["concentration_bound_M"], 1e-14) for x in history], "--", label=f"bound {case['seed']}")
        axes[1].plot([r["evaluated_nodes"] for r in case["quantile_queries"]], [r["interval_width_relative"] for r in case["quantile_queries"]], "o-", label=str(case["seed"]))
    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("concentration M norm [mol/m3]")
    axes[1].set_xlabel("evaluated tree nodes (includes internal nodes)"); axes[1].set_ylabel("P99 interval relative width")
    for ax in axes: ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.savefig("certificate_and_tail.png", dpi=170); plt.close(fig)
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(len(cases), 5, figsize=(14, 2.7*len(cases)), constrained_layout=True, squeeze=False)
    for row, case in enumerate(cases):
        with np.load(f"controlled_seed_{case['seed']}.npz", allow_pickle=False) as data:
            labels = data["labels"]
            shape = labels.shape; middle = shape[0]//2
            reference = np.full(labels.size, np.nan); reference[data["active_ids"]] = data["reference_concentration"]
            error = np.full(labels.size, np.nan); error[data["active_ids"]] = data["concentration"]-data["reference_concentration"]
            reference, error = reference.reshape(shape)[middle], error.reshape(shape)[middle]
            axes[row, 0].imshow(labels[middle], origin="lower", cmap=ListedColormap(["#f2f4f7", "#338eab", "#dfad62"]), vmin=0, vmax=2)
            im = axes[row, 1].imshow(reference, origin="lower", cmap="viridis"); fig.colorbar(im, ax=axes[row, 1], shrink=.75)
            limit = max(float(np.nanmax(np.abs(error))), 1e-12)
            im = axes[row, 2].imshow(error, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit); fig.colorbar(im, ax=axes[row, 2], shrink=.75)
            centers = data["centers"]*1e6
            plane = (data["origin_m"][0]+(middle+.5)*data["spacing_m"][0])*1e6
            selected = np.abs(centers[:, 2]-plane) < .5*data["spacing_m"][0]*1e6
            sigma = principal_tension(data["reference_stress"])/1e6
            delta = principal_tension(data["stress"])/1e6-sigma
            im = axes[row, 3].scatter(centers[selected, 0], centers[selected, 1], c=sigma[selected], s=16, cmap="magma", marker="s")
            fig.colorbar(im, ax=axes[row, 3], shrink=.75)
            limit = max(float(np.max(np.abs(delta[selected]))), 1e-12)
            im = axes[row, 4].scatter(centers[selected, 0], centers[selected, 1], c=delta[selected], s=16, cmap="RdBu_r", vmin=-limit, vmax=limit, marker="s")
            fig.colorbar(im, ax=axes[row, 4], shrink=.75)
        for col in range(5):
            axes[row, col].set_aspect("equal"); axes[row, col].set_xlabel("x voxel" if col < 3 else "x [um]")
        axes[row, 0].set_ylabel(f"seed {case['seed']} / y voxel")
    for ax, title in zip(axes[0], ["phases: pore / active / CBD", "reference c - cref [mol/m3]", "concentration error [mol/m3]", "reference principal stress [MPa]", "principal stress error [MPa]"]): ax.set_title(title, fontsize=9)
    fig.savefig("controlled_fields.png", dpi=170); plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(); workspace = Workspace(Path(os.environ["MORQEX_WORKSPACE"]))
    if workspace.root not in Path.cwd().parents: raise ValueError("External run directory required")
    cfg = tomllib.loads(args.config.read_text()); base_path = args.config.parent/cfg["geometry_config"]
    base = tomllib.loads(base_path.read_text())
    if cfg["schema"] != "morqex-controlled-study-1" or np.prod(base["shape_zyx"]) > 216 or cfg["steps"] > 100 or len(base["seeds"]) > 5:
        raise ValueError("Unsupported protocol or exceeded small-system budget")
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root/"src/morqex").glob("*.py")), *sorted((root/"src/morqex/mechanics").glob("*.py")), args.config, base_path, Path(__file__)]
    result = {"schema": "morqex-controlled-result-1", "config": cfg, "geometry_config": base,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "scope": "fixed Fick diffusion with prescribed reactive-face flux; mechanics reconstructed one-way; no DFN/bidirectional evolution certification",
        "domain_note": "held-out-time uses four construction spatial channels; hotspot adds a fifth omitted from the original bases but included in the residual library. resolvent-rebuilt-with-hotspot explicitly uses that newly observed fifth shape for offline rebuilding; no validation concentration/stress enters its construction, and it is not a zero-shot comparison",
        "cases": []}
    for seed in base["seeds"]:
        case = case_study(seed, cfg, base, workspace); result["cases"].append(case)
        Path("controlled_study.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"seed": seed, "completed": True}), flush=True)
    figures(result["cases"])
    print(json.dumps({"result": str(Path.cwd()/"controlled_study.json")}))


if __name__ == "__main__": main()
