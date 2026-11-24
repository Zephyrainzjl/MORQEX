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
    labels = ["final_tests", "integrated_upstream_tests", "energy_algebra", "installed_package", "resource_limits", "integrity", "source_audit", "package_wheel"]
    for label in labels:
        record = json.loads((WS / "reports" / (label + "_latest.json")).read_text())
        stdout = (Path(record["run"]) / "stdout.txt").read_text()
        assert record["exit_code"] == 0 and not record["stopped_because"], label
        records[label] = {k: record[k] for k in ("run", "exit_code", "wall_seconds", "peak_group_rss_bytes_sampled")}
        if "tests" in label:
            records[label]["pytest_summary"] = stdout.strip().splitlines()[-1]
    assert re.search(r"\b11 passed\b", records["final_tests"]["pytest_summary"])
    assert re.search(r"\b7 passed, 9 skipped\b", records["integrated_upstream_tests"]["pytest_summary"])
    for path in [ROOT / "setup.py", *(ROOT / "scripts").glob("*.py"), *(ROOT / "benchmarks").glob("*.py"), *(ROOT / "tests").glob("*.py")]:
        ast.parse(path.read_text(), filename=str(path))
    assert git(ROOT.parent / "ZynJoule", "status", "--porcelain") == "", "Original repository changed"
    artifacts = []
    for path in (WS / "wheels").glob("*.whl"):
        artifacts.append({"path": str(path), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = {"source_commit": git(ROOT, "rev-parse", "HEAD"), "source_dirty": bool(git(ROOT, "status", "--porcelain")),
               "stage": "Phase 0/I foundation and integrated microstructure source; no certified RVE/DFN coupling",
               "source": str(ROOT), "workspace": str(WS), "records": records,
               "wheels": artifacts, "original_zynjoule_unchanged": True,
               "limitations": ["9 optional-backend tests skipped", "No full RVE mesh convergence", "No established novelty or measured algorithmic speedup"]}
    path = WS / "reports/verification_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": str(path), "source_commit": summary["source_commit"],
                      "source_dirty": summary["source_dirty"], "successful_checks": len(labels),
                      "original_zynjoule_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
