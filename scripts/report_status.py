"""Summarize actual local verification artifacts and their source revision."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WS = Path(os.environ["MORQEX_WORKSPACE"])


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main():
    records = {}
    labels = ["final_tests", "installed_package", "resource_limits", "integrity", "package_wheel", "package_install",
              "reference_convergence", "mechanics_backends", "linear_study", "controlled_study", "energy_adaptivity", "sparse_scaling", "test_html", "html_validation"]
    for label in labels:
        record = json.loads((WS / "reports" / (label + "_latest.json")).read_text())
        stdout = (Path(record["run"]) / "stdout.txt").read_text()
        assert record["exit_code"] == 0 and not record["stopped_because"], label
        records[label] = {k: record[k] for k in ("run", "exit_code", "wall_seconds", "peak_group_rss_bytes_sampled")}
        if "tests" in label:
            records[label]["pytest_summary"] = stdout.strip().splitlines()[-1]
    assert re.search(r"\b111 passed\b", records["final_tests"]["pytest_summary"])
    historical = json.loads((WS / "reports/integrated_upstream_tests_latest.json").read_text())
    assert historical["exit_code"] == 0
    upstream_summary = (Path(historical["run"]) / "stdout.txt").read_text().strip().splitlines()[-1]
    assert re.search(r"\b7 passed, 9 skipped\b", upstream_summary)
    experiments = {}
    for label, filename in [("reference_convergence", "qualification.json"), ("mechanics_backends", "mechanics_backends.json"), ("linear_study", "linear_study.json"), ("controlled_study", "controlled_study.json"), ("energy_adaptivity", "energy_adaptivity.json"), ("sparse_scaling", "sparse_scaling.json")]:
        path = Path(records[label]["run"]) / filename
        evidence = json.loads(path.read_text())
        for relative, digest in evidence["source_sha256"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, (label, relative, "stale experimental source")
        experiments[label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    study = json.loads(Path(experiments["linear_study"]["path"]).read_text())
    assert len(study["cases"]) == 3
    reference = json.loads(Path(experiments["reference_convergence"]["path"]).read_text())
    controlled = json.loads(Path(experiments["controlled_study"]["path"]).read_text())
    adaptive = json.loads(Path(experiments["energy_adaptivity"]["path"]).read_text())
    adaptive_selected = [(c["seed"],p,m) for c in adaptive["cases"] for p in c["results"] for m in p["methods"] if m["method"] == "adaptive-spectral-certificate"]
    assert len(adaptive_selected) == 6 and all(m["qualified_steps"] == adaptive["config"]["steps"] and m["bound_coverage"] == 1 for seed,p,m in adaptive_selected)
    scaling=json.loads(Path(experiments["sparse_scaling"]["path"]).read_text())
    assert len(scaling["cases"])==9
    scaling_selected=[(c,next((m for m in c["methods"] if m["basis_family"]==c["selected_family"] and m["total_modes"]==c["selected_modes"]),None)) for c in scaling["cases"]]
    assert all(m is not None and m["qualified_steps"]==scaling["config"]["steps"] for c,m in scaling_selected)
    html_manifest = json.loads((WS/"reports/test_report_manifest.json").read_text())
    html_validation = json.loads((WS/"reports/test_report_validation.json").read_text())
    digest = hashlib.sha256(Path(html_manifest["html"]).read_bytes()).hexdigest()
    assert digest == html_manifest["sha256"] == html_validation["sha256"]
    for path in [ROOT / "setup.py", *(ROOT / "scripts").glob("*.py"), *(ROOT / "benchmarks").glob("*.py"), *(ROOT / "tests").glob("*.py")]:
        ast.parse(path.read_text(), filename=str(path))
    # This workspace is shared. Concurrent work in the original repository must
    # be preserved, not reset or misreported as our changes or as a clean tree.
    original_status = git(ROOT.parent / "ZynJoule", "status", "--porcelain")
    artifacts = []
    for path in (WS / "wheels").glob("*.whl"):
        artifacts.append({"path": str(path), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = {"source_commit": git(ROOT, "rev-parse", "HEAD"), "source_dirty": bool(git(ROOT, "status", "--porcelain")),
               "stage": "0.0.5 sparse coupled scaling, matrix-free concentration Hessian action, compact online/statistics kernels; no dense H/full identity stress map; fixed quadratic only, no DFN/CT/novelty qualification",
               "source": str(ROOT), "workspace": str(WS),
               "retained_failed_scaling_run":json.loads((WS/"reports/sparse_scaling_18_latest.json").read_text()), "records": records,
               "wheels": artifacts, "experiments": experiments, "html_report": html_manifest,
               "html_validation": html_validation,
               "original_repository": {"head": git(ROOT.parent / "ZynJoule", "rev-parse", "HEAD"),
                   "worktree_clean": not bool(original_status), "observed_worktree_status": original_status,
                   "task_scope": "This MORQEX task made no writes to ZynJoule; concurrent changes observed and preserved; no pretraining inspection"},
               "historical_integrated_tests": {"run": historical["run"], "summary": upstream_summary, "rerun_this_milestone": False,
                   "scope": "274 integrated files unchanged; optional backends remain unqualified"},
               "measurements": {"sparse_scaling_selected":[{"N":c["N"],"seed":c["seed"],"modes":m["total_modes"],"online_speedup":m["qualified_speedup"],"stress_L2":c["selected_fields"]["relative_stress_L2"]} for c,m in scaling_selected],"coupled_adaptive_selected": [{"seed":seed,"protocol":p["protocol"],"concentration_modes":m["final_modes"],"additional_error_modes":m["retained_error_modes"],"full_correction_queries":m["high_fidelity_correction_queries"],"stress_trajectory_error":m["relative_stress_trajectory_error"],"online_cost_over_reused_full":m["online_seconds_including_enrichment"]/p["reference_online_total_s"]} for seed,p,m in adaptive_selected],
                   "sphere_fitted_latest": reference["fitted_sphere"]["mesh_sequence"][-1],
                   "same_output_small_field_speedups": [c["timing"]["full_fields_speedup"] for c in study["cases"]],
                   "worst_closed_energy_balance_J_m3": max(abs(d["energy_balance_density"]) for c in study["cases"] for d in c["closed_diagnostics"]),
                   "controlled_min_bound_coverage": min(r["bound_coverage"] for c in controlled["cases"] for r in c["comparison"] if r["status"] == "measured"),
                   "controlled_rebuilt_hotspot_stress_trajectory_L2": [r["stress_trajectory_L2"] for c in controlled["cases"] for r in c["comparison"] if r["status"] == "measured" and r["method"] == "resolvent-rebuilt-with-hotspot" and r["domain"] == "held-out-surface-hotspot" and r["modes"] == c["diffusion_components"]+20]},
               "limitations": ["9 optional-backend tests historically skipped", "n24 sphere stress error 11%; geometry/discretization error remains",
                   "Original arbitrary-initial-field and unknown-hotspot spaces remain inaccurate; rebuilding uses newly observed flux shape",
                   "Legacy adaptive certificate remains dense N<=512; new sparse certificate fixed quadratic only; neither certifies nonlinear/DFN",
                   "Query-adaptive P99 intervals narrower but tree slower than vectorized full statistics on small grids",
                   "HTML DOM/assets/QuickJS fixture checked; no full browser rendering available",
                   "Quadratic microstructure coupling/adaptivity only; no DFN/electrolyte/charge/voltage",
                   "Sparse scaling is not fixed-geometry convergence; large-N strong-coupling/contrast/CT/RVE/material/advanced baselines and novelty remain unqualified",
                   "Small same-output field timing only; no validated FE2 or near-DFN speedup"]}
    path = WS / "reports/verification_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": str(path), "source_commit": summary["source_commit"],
                      "source_dirty": summary["source_dirty"], "successful_checks": len(labels),
                      "original_worktree_clean": not bool(original_status),
                      "concurrent_original_changes_preserved": bool(original_status)}, indent=2))


if __name__ == "__main__":
    main()
