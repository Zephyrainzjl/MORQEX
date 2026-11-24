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
              "reference_convergence", "mechanics_backends", "linear_study"]
    for label in labels:
        record = json.loads((WS / "reports" / (label + "_latest.json")).read_text())
        stdout = (Path(record["run"]) / "stdout.txt").read_text()
        assert record["exit_code"] == 0 and not record["stopped_because"], label
        records[label] = {k: record[k] for k in ("run", "exit_code", "wall_seconds", "peak_group_rss_bytes_sampled")}
        if "tests" in label:
            records[label]["pytest_summary"] = stdout.strip().splitlines()[-1]
    assert re.search(r"\b53 passed\b", records["final_tests"]["pytest_summary"])
    historical = json.loads((WS / "reports/integrated_upstream_tests_latest.json").read_text())
    assert historical["exit_code"] == 0
    upstream_summary = (Path(historical["run"]) / "stdout.txt").read_text().strip().splitlines()[-1]
    assert re.search(r"\b7 passed, 9 skipped\b", upstream_summary)
    experiments = {}
    for label, filename in [("reference_convergence", "qualification.json"), ("mechanics_backends", "mechanics_backends.json"), ("linear_study", "linear_study.json")]:
        path = Path(records[label]["run"]) / filename
        evidence = json.loads(path.read_text())
        for relative, digest in evidence["source_sha256"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, (label, relative, "stale experimental source")
        experiments[label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    study = json.loads(Path(experiments["linear_study"]["path"]).read_text())
    assert len(study["cases"]) == 3
    reference = json.loads(Path(experiments["reference_convergence"]["path"]).read_text())
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
               "stage": "0.0.2 verified fixed-C linear prototype; concentration compression and useful tail bounds failed; no DFN/full-cell certification",
               "source": str(ROOT), "workspace": str(WS), "records": records,
               "wheels": artifacts, "experiments": experiments,
               "original_repository": {"head": git(ROOT.parent / "ZynJoule", "rev-parse", "HEAD"),
                   "worktree_clean": not bool(original_status), "observed_worktree_status": original_status,
                   "task_scope": "This MORQEX task made no writes to ZynJoule; concurrent changes observed and preserved; no pretraining inspection"},
               "historical_integrated_tests": {"run": historical["run"], "summary": upstream_summary, "rerun_this_milestone": False,
                   "scope": "274 integrated files unchanged; optional backends remain unqualified"},
               "measurements": {"sphere_fitted_latest": reference["fitted_sphere"]["mesh_sequence"][-1],
                   "same_output_small_field_speedups": [c["timing"]["full_fields_speedup"] for c in study["cases"]],
                   "worst_closed_energy_balance_J_m3": max(abs(d["energy_balance_density"]) for c in study["cases"] for d in c["closed_diagnostics"])},
               "limitations": ["9 optional-backend tests historically skipped", "n24 sphere stress error 11%; geometry/discretization error remains",
                   "Joint POD did not outperform reliably; small bases failed held-out stress tolerance", "64-point P99 envelope too wide",
                   "Quadratic microstructure reservoir coupling only; no DFN/electrolyte/charge/voltage",
                   "No CT/RVE representativity/material calibration/complete advanced baselines or established novelty",
                   "Small same-output field timing only; no validated FE2 or near-DFN speedup"]}
    path = WS / "reports/verification_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": str(path), "source_commit": summary["source_commit"],
                      "source_dirty": summary["source_dirty"], "successful_checks": len(labels),
                      "original_worktree_clean": not bool(original_status),
                      "concurrent_original_changes_preserved": bool(original_status)}, indent=2))


if __name__ == "__main__":
    main()
