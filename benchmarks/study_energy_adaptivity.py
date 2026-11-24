"""Same-space mass/energy projections and certified nested coupled enrichment.

Full reference uses fixed-geometry linear mechanics and quadratic chemistry.
All work at enrichment (including full concentration queries) is measured.
No BV/electrolyte/DFN/CT/material-design or novelty certification is implied.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import tomllib
import numpy as np
from study_controlled_transport import patterns_for
from morqex import Workspace
from morqex.gemsim import GEMsim
from morqex.mechanics import TetMesh, ElasticitySolver, isotropic_stiffness
from morqex.transport import VoxelDiffusion, ConcentrationInfeasible
from morqex.concentration import ConcentrationBasis
from morqex.coupling import bind_chemical_modes
from morqex.energy import ResponseBuilder
from morqex.gradient_transport import QuadraticGradientSystem, PortGradientModel, EnergyAccuracyError
from morqex.adaptive_transport import AdaptiveGradientTransport, AdaptiveBudgetError
from morqex.statistics import principal_tension, weighted_quantile


def controls(seed,cfg,protocol):
    rng = np.random.default_rng(seed+5678)
    t = (np.arange(cfg["steps"])+1)*cfg["dt_s"]
    phase = rng.uniform(0,2*np.pi,4)
    u = np.zeros((len(t),5))
    u[:,0] = 1.+.2*np.sin(.4*t+phase[0])
    u[:,1:4] = .3*np.sin(t[:,None]*np.array([.2,.35,.5])[None,:]+phase[None,1:])
    start = cfg["hotspot_start_step"]
    u[start:,4] = 3*(1.+.1*np.cos(.5*t[start:]))
    e = np.zeros((len(t),6))
    if protocol == "flux-hotspot-and-macro":
        e[len(t)//4:,0] = cfg["macro_strain_amplitude"]*np.sin(.13*(t[len(t)//4:]-t[len(t)//4]))
    return cfg["flux_scale_mol_m2_s"]*u,e


def setup(seed,base,workspace):
    provider = GEMsim(workspace)
    geometry = provider.random_field(shape=tuple(base["shape_zyx"]),fractions=tuple(base["fractions"]),
        length_scale=(.7,.9,1.1),seed=seed,spacing=(base["spacing_m"],)*3,
        phase_kinds={0:"pore",1:"active",2:"cbd"})
    volume = provider.to_volume(geometry)
    diffusion = VoxelDiffusion(volume,active_phases=[1],diffusivity=base["diffusivity_xyz_m2_s"],reactive_phases=[0])
    mesh = TetMesh.from_volume(volume,solid_phases=[1,2])
    tensors = {1:isotropic_stiffness(base["young_active_Pa"],.25),2:isotropic_stiffness(base["young_cbd_Pa"],.3)}
    solver = ElasticitySolver(mesh,np.array([tensors[int(p)] for p in mesh.phases]))
    n = len(diffusion.weights)
    identity = ConcentrationBasis(np.eye(n),diffusion.weights.copy(),"identity",diffusion.n_components)
    started = time.perf_counter()
    builder = ResponseBuilder(solver,bind_chemical_modes(solver,diffusion,identity,base["expansion_m3_mol"]))
    response = builder.exact()
    operator_seconds = time.perf_counter()-started
    pattern = patterns_for(diffusion)
    started = time.perf_counter()
    system = QuadraticGradientSystem(response,diffusion,identity,pattern,
        chemical_curvature=base["chemical_curvature_J_m3_mol2"],reference_concentration=base["reference_concentration_mol_m3"],
        maximum_concentration=base["maximum_concentration_mol_m3"])
    concentration_setup = time.perf_counter()-started
    return system,solver,builder,pattern,operator_seconds,concentration_setup,geometry.fingerprint


def protocol_study(system,initial_basis,seed,cfg,protocol):
    inputs,strains = controls(seed,cfg,protocol); dt = cfg["dt_s"]
    c = np.zeros(system.n); truth,reference_time = [],[]
    start = time.perf_counter(); system.full_step(c,dt,np.zeros(system.b.shape[1])); reference_setup = time.perf_counter()-start
    for u,e in zip(inputs,strains):
        started = time.perf_counter(); c = system.full_step(c,dt,u,macro_strain=e)
        energy = system.energy(c,e); inventory = system.baseline_inventory+system.inventory_rows @ c
        reference_time.append(time.perf_counter()-started)
        truth.append(c.copy())
    truth = np.array(truth)
    reference_stress = system.response.stress(np.column_stack([strains,truth]))
    weights = system.response.volumes/system.response.rve_volume
    tolerance = np.sqrt(system.kappa)*cfg["concentration_equivalent_tolerance_mol_m3"]
    result = {"protocol":protocol,"reference_setup_s":reference_setup,"reference_online_total_s":sum(reference_time),
        "reference_output_scope":"concentration step + energy + per-component inventory; full fields excluded for all methods",
        "energy_norm_tolerance":float(tolerance),"methods":[]}
    for method in ("static-mass-projection","static-energy-projection","adaptive-snapshot","adaptive-resolvent-block","adaptive-compressed-block","adaptive-state-restart","adaptive-dynamics-restart","adaptive-spectral-certificate"):
        start = time.perf_counter()
        basis = system.conservative_basis(initial_basis)
        if method.startswith("adaptive-"):
            strategy = {"adaptive-snapshot":"snapshot","adaptive-resolvent-block":"resolvent-block","adaptive-compressed-block":"compressed-resolvent-block","adaptive-state-restart":"state-preserving-restart","adaptive-dynamics-restart":"state-preserving-restart","adaptive-spectral-certificate":"snapshot"}[method]
            adaptive = AdaptiveGradientTransport(system,basis,maximum_modes=min(cfg["maximum_modes"],system.n),
                maximum_enrichments_per_step=cfg["maximum_enrichments_per_step"],
                enrichment_strategy=strategy,resolvent_shifts=cfg["resolvent_shifts_per_s"],
                block_rank=cfg["block_rank"],channel_scales=cfg["flux_channel_bounds_mol_m2_s"],macro_strain_scale=cfg["macro_strain_amplitude"],restart_dimension=cfg["restart_dimension"],
                retain_forcing_directions=method == "adaptive-dynamics-restart",
                certificate_method="spectral" if method == "adaptive-spectral-certificate" else "graph",error_modes=cfg["retained_error_modes"])
            model = adaptive.model
        else:
            adaptive = None
            model = PortGradientModel(system,basis,projection="mass" if method == "static-mass-projection" else "energy")
        model_setup = time.perf_counter()-start
        state = model.initialize(np.zeros(system.n),error_tolerance=tolerance)
        history,events = [],[]
        elapsed,full_queries,full_space_steps,setup_work,rhs_queries,factor_queries = 0.,0,0,0.,0,0
        error_energy_sq,truth_energy_sq,stress_error_sq,stress_norm_sq = 0.,0.,0.,0.
        failure = None
        for k,(u,e,reference) in enumerate(zip(inputs,strains,truth)):
            started = time.perf_counter()
            if adaptive is None:
                current,diag = model.trial(state,dt,u,macro_strain=e)
                try: model.accept(current,tolerance); qualified = True
                except (EnergyAccuracyError,ConcentrationInfeasible): qualified = False
            else:
                try:
                    current,adaptive_diag = adaptive.step(state,dt,u,macro_strain=e,error_tolerance=tolerance)
                except AdaptiveBudgetError as error:
                    elapsed += time.perf_counter()-started; failure = str(error); break
                diag = adaptive_diag.dynamics; qualified = True
                full_queries += adaptive_diag.high_fidelity_queries; full_space_steps += adaptive_diag.full_space_steps
                rhs_queries += adaptive_diag.resolvent_right_hand_sides; factor_queries += adaptive_diag.resolvent_factorizations
                setup_work += adaptive_diag.enrichment_seconds
                events.extend(adaptive_diag.events); model = adaptive.model
            elapsed += time.perf_counter()-started
            approximate = model.basis.values @ current.coefficients
            actual_error = system.norm(reference-approximate)
            assert actual_error <= current.error.total_bound+1e-8
            assert abs(diag.energy_balance) < 1e-6
            assert np.max(np.abs(diag.component_mass_error)) < 1e-24
            expected_inventory = system.baseline_inventory+system.inventory_rows @ reference
            assert np.max(np.abs(expected_inventory-current.component_inventory)) < 1e-24
            stress = system.response.stress(np.r_[e,approximate])
            difference = stress-reference_stress[k]
            stress_abs = np.einsum("e,ei,ei->",weights,difference,difference)
            stress_size = np.einsum("e,ei,ei->",weights,reference_stress[k],reference_stress[k])
            stress_error_sq += stress_abs; stress_norm_sq += stress_size
            error_energy_sq += actual_error**2; truth_energy_sq += system.norm(reference)**2
            p99 = weighted_quantile(principal_tension(stress),system.response.volumes,.99)
            p99_truth = weighted_quantile(principal_tension(reference_stress[k]),system.response.volumes,.99)
            history.append({"time_s":current.time,"modes":model.dimension,"actual_energy_norm_error":actual_error,
                "energy_norm_error_bound":current.error.total_bound,"qualified":qualified,
                "stress_L2_relative":float(np.sqrt(stress_abs/max(stress_size,1e6))),
                "P99_relative_error":abs(p99-p99_truth)/max(abs(p99_truth),1e3),
                "energy_density":current.energy_density,"energy_balance":diag.energy_balance,
                "component_mass_error_max":float(np.max(np.abs(diag.component_mass_error)))})
            state = current
        result["methods"].append({"method":method,"status":"measured" if failure is None else "budget-rejected",
            "comparison_scope":"full-horizon" if failure is None else "prefix-only; no full-horizon accuracy/cost comparison",
            "failure":failure,"initial_modes":basis.values.shape[1],"final_modes":model.dimension,
            "retained_error_modes":model.error_space.count if model.error_space is not None else 0,
            "certificate_method":model.certificate_method,
            "steps_completed":len(history),"setup_seconds":model_setup,"online_seconds_including_enrichment":elapsed,
            "enrichment_seconds_accepted_steps":setup_work,
            "high_fidelity_correction_queries":adaptive.total_full_queries if adaptive else full_queries,
            "resolvent_right_hand_sides":adaptive.total_resolvent_rhs if adaptive else rhs_queries,
            "resolvent_factorizations":adaptive.total_resolvent_factorizations if adaptive else factor_queries,
            "full_space_steps":full_space_steps,"events":events,"history":history,
            "relative_energy_norm_trajectory_error":float(np.sqrt(error_energy_sq/max(truth_energy_sq,1e-30))) if failure is None else None,
            "relative_stress_trajectory_error":float(np.sqrt(stress_error_sq/max(stress_norm_sq,1e-30))) if failure is None else None,
            "prefix_relative_stress_error":float(np.sqrt(stress_error_sq/max(stress_norm_sq,1e-30))),
            "worst_P99_relative_error":max((x["P99_relative_error"] for x in history),default=0.),
            "qualified_steps":sum(x["qualified"] for x in history),
            "bound_coverage":float(np.mean([x["actual_energy_norm_error"] <= x["energy_norm_error_bound"]+1e-8 for x in history])) if history else None})
        record = result["methods"][-1]
        record["maximum_energy_balance_J_m3"] = max((abs(x["energy_balance"]) for x in history),default=0.)
        record["maximum_component_mass_error_mol"] = max((x["component_mass_error_max"] for x in history),default=0.)
        record["maximum_transfer_energy_jump_J_m3"] = max((abs(x["energy_jump"]) for x in events),default=0.)
        record["maximum_transfer_mass_jump_mol"] = max((max(abs(v) for v in x["mass_jump"]) for x in events),default=0.)
        record["maximum_energy_norm_bound"] = max((x["energy_norm_error_bound"] for x in history),default=0.)
        record["final_online_state_coordinates"] = model.dimension+record["retained_error_modes"]+len(system.null_basis.T)+2
        if model.error_space is not None:
            record["complement_coercivity_lower_per_s"] = model.error_space.tail_lower
            record["off_block_coupling_upper_per_s"] = model.error_space.coupling
    return result,truth,reference_stress


def figures(cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    protocols = cases[0]["protocols"]
    fig,axes = plt.subplots(2,len(protocols),figsize=(11,6),constrained_layout=True,squeeze=False)
    for column,protocol in enumerate(protocols):
        for case in cases:
            result = next(p for p in case["results"] if p["protocol"] == protocol)
            adaptive = next(m for m in result["methods"] if m["method"] == "adaptive-spectral-certificate")
            history = adaptive["history"]
            times = [r["time_s"] for r in history]
            axes[0,column].semilogy(times,[max(r["actual_energy_norm_error"],1e-12) for r in history],label=f"actual {case['seed']}")
            axes[0,column].semilogy(times,[max(r["energy_norm_error_bound"],1e-12) for r in history],"--",label=f"bound {case['seed']}")
            axes[1,column].step(times,[r["modes"] for r in history],where="post",label=str(case["seed"]))
        axes[0,column].axhline(result["energy_norm_tolerance"],color="black",ls=":",label="tolerance")
        axes[0,column].set_title(protocol); axes[0,column].set_ylabel("coupled H error [sqrt(J/m3)]")
        axes[1,column].set_ylabel("concentration modes incl. mass Riesz")
    for ax in axes.ravel(): ax.set_xlabel("time [s]");ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.savefig("adaptive_bound_and_modes.png",dpi=170);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,6),constrained_layout=True,sharey=True)
    methods = ["static-mass-projection","static-energy-projection","adaptive-snapshot","adaptive-resolvent-block","adaptive-compressed-block","adaptive-state-restart","adaptive-dynamics-restart","adaptive-spectral-certificate"]
    for protocol in protocols:
        values,costs = [],[]
        for method in methods:
            selected = [(p,m) for c in cases for p in c["results"] for m in p["methods"] if p["protocol"] == protocol and m["method"] == method]
            # A failed prefix is never averaged against a complete horizon.
            if not all(m["status"] == "measured" for _,m in selected):
                values.append(np.nan); costs.append(np.nan); continue
            values.append(np.mean([m["relative_stress_trajectory_error"] for _,m in selected]))
            costs.append(np.mean([m["online_seconds_including_enrichment"]/p["reference_online_total_s"] for p,m in selected]))
        axes[0].semilogx(values,np.arange(len(methods)),"o-",label=protocol)
        axes[1].semilogx(costs,np.arange(len(methods)),"o-",label=protocol)
    axes[0].set_yticks(np.arange(len(methods)),methods)
    axes[0].set_xlabel("mean stress trajectory relative error");axes[1].set_xlabel("online cost / reused full coupled solve")
    for ax in axes: ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.savefig("energy_projection_comparison.png",dpi=170);plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True)
    args = parser.parse_args();workspace = Workspace(Path(os.environ["MORQEX_WORKSPACE"]))
    if workspace.root not in Path.cwd().parents: raise ValueError("External workspace run directory required")
    cfg = tomllib.loads(args.config.read_text());base_path = args.config.parent/cfg["geometry_config"]
    base = tomllib.loads(base_path.read_text())
    if cfg["schema"] != "morqex-energy-adaptivity-1" or cfg["steps"] > 120 or np.prod(base["shape_zyx"]) > 216 or len(base["seeds"]) > 5:
        raise ValueError("Unsupported protocol or research budget exceeded")
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root/"src/morqex").glob("*.py")),*sorted((root/"src/morqex/mechanics").glob("*.py")),
        args.config,base_path,Path(__file__),Path(__file__).with_name("study_controlled_transport.py")]
    output = {"schema":"morqex-energy-adaptivity-result-1","config":cfg,"geometry_config":base,
        "source_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "scope":"fixed-C coupled diffusion/elasticity, quadratic chemistry, prescribed flux and macro strain; dense offline N<=512; no BV/electrolyte/DFN/CT/novelty certification",
        "port_scope":"hotspot is absent from initial basis but declared in the certificate port library; activation is online, not arbitrary unknown geometry registration",
        "static_scope":"same basis for both static projections; reused linear factorization; unqualified static trial trajectories retained for analysis, not returned by guarded production step",
        "cases":[]}
    for seed in base["seeds"]:
        system,solver,builder,patterns,offline,concentration_setup,geometry = setup(seed,base,workspace)
        started = time.perf_counter(); error_space = system.error_space(cfg["retained_error_modes"])
        certificate_setup = time.perf_counter()-started
        initial = ConcentrationBasis.boundary_resolvent(system.diffusion,system.diffusion.n_components+cfg["initial_nonconstant_modes"],patterns[:,:4])
        case = {"seed":seed,"geometry_fingerprint":geometry,"active_voxels":system.n,
            "full_mechanical_response_offline_s":offline,
            "full_concentration_hessian_setup_s":concentration_setup,
            "shared_spectral_certificate_offline_s":certificate_setup,
            "shared_spectral_certificate_array_bytes":sum(a.nbytes for a in (error_space.slow,error_space.tail,error_space.slow_operator)),
            "dense_concentration_hessian_bytes":system.H.nbytes,
            "offline_scope":"shared dense full-response/Hessian and spectral setup; per-method setup is separate; no large-N scalability claim",
            "protocols":cfg["protocols"],"results":[]}
        for protocol in cfg["protocols"]:
            result,truth,stress = protocol_study(system,initial,seed,cfg,protocol)
            case["results"].append(result)
            _,strains = controls(seed,cfg,protocol)
            independent = solver.solve(np.einsum("eim,m->ei",builder.modes,truth[-1]),macro_strain=strains[-1])
            difference = np.linalg.norm(independent.stress-stress[-1])/max(np.linalg.norm(independent.stress),1e-30)
            assert difference < 1e-10
            result["independent_FEM_stress_difference"] = float(difference)
        output["cases"].append(case)
        Path("energy_adaptivity.json").write_text(json.dumps(output,indent=2,allow_nan=False))
        print(json.dumps({"seed":seed,"completed":True}),flush=True)
    figures(output["cases"])
    print(json.dumps({"result":str(Path.cwd()/"energy_adaptivity.json")}))


if __name__ == "__main__":main()
