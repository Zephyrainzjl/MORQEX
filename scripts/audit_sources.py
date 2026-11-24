"""Read source/doc inventories and record reproducible static audit evidence."""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ["MORQEX_WORKSPACE"])
ZYN = ROOT.parent / "ZynJoule"
GEM = WORKSPACE / "upstream/GEMsim"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def inventory(root, paths):
    rows = []
    for path in sorted(set(paths)):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        content = path.read_bytes()
        row = {"path": str(path.relative_to(root)), "bytes": len(content),
               "sha256": hashlib.sha256(content).hexdigest()}
        if path.suffix in {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".H", ".cpp", ".cxx", ".pyx"} or path.name.startswith("inputs"):
            text = content.decode("utf8", errors="replace")
            row["lines"] = len(text.splitlines())
            if path.suffix == ".py":
                try:
                    tree = ast.parse(text)
                    row["definitions"] = [{"name": n.name, "line": n.lineno, "end_line": n.end_lineno}
                                          for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
                except SyntaxError as exc:
                    row["syntax_error"] = str(exc)
        rows.append(row)
    return rows


def main():
    # All current simulation/geometry source and their configs, all root docs,
    # README/RESUME/log/config discovery across the current repository; no old run snapshots.
    relevant = list((ZYN / "src/zynjoule/zynsim").rglob("*"))
    relevant += list((ZYN / "src/zynjoule/zynmorph").rglob("*"))
    relevant += list((ZYN / "src/zynjoule/physics").rglob("*"))
    relevant += list((ZYN / "configs").rglob("*"))
    relevant += list(ZYN.glob("*.md")) + [ZYN / "pyproject.toml", ZYN / "CMakeLists.txt"]
    docs = [p for p in ZYN.rglob("*") if p.is_file() and ".git" not in p.parts and
            (p.name.startswith(("README", "RESUME", "RESEARCH_LOG")) or p.suffix in {".toml", ".yaml", ".yml"})]
    tracked_gem = [GEM / p for p in git(GEM, "ls-files").splitlines()]
    versions = {}
    for package in ("numpy", "scipy", "pytest", "scikit-image", "tifffile", "meshio", "ngsolve", "pybamm", "pyamg", "petsc4py", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"scope": "Current ZynJoule simulation/geometry/physics + root docs/configs; complete pinned GEMsim",
              "caveat": "Inventory/hashing is not semantic validation; key manually reviewed files are cited in CODE_AUDIT.md",
              "exclusions": "lw historical run backups, training data and unrelated atomistic ML implementations",
              "zynjoule_commit": git(ZYN, "rev-parse", "HEAD"), "zynjoule_status": git(ZYN, "status", "--porcelain"),
              "gemsim_commit": git(GEM, "rev-parse", "HEAD"), "gemsim_status": git(GEM, "status", "--porcelain"),
              "python": platform.python_version(), "packages": versions,
              "zynjoule_files": inventory(ZYN, relevant), "zynjoule_docs_discovery": inventory(ZYN, docs),
              "gemsim_files": inventory(GEM, tracked_gem)}
    report["morqex_integrated_source"] = inventory(ROOT, (ROOT / "src").rglob("*"))
    destination = WORKSPACE / "evidence/source_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"report": str(destination), "zynjoule_files": len(report["zynjoule_files"]),
                      "gemsim_files": len(report["gemsim_files"]), "packages": versions}, indent=2))


if __name__ == "__main__":
    main()
