#!/usr/bin/env python3
"""Materialize immutable upstream source in the external workspace, without building."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from morqex.workspace import Workspace


def main():
    workspace = Workspace(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "morph_workspace")
    lock = json.loads((ROOT / "upstream.lock.json").read_text())["gemsim"]
    target = workspace.path(lock["source_relative_to_workspace"])
    if target.exists():
        commit = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True).strip()
        if commit != lock["commit"] or status:
            raise RuntimeError("Preserve and audit changed upstream checkout; refusing reset")
        print("Verified audit-only upstream GEMsim", commit)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", lock["url"], str(target)], check=True, timeout=90)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", lock["commit"]], check=True, timeout=90)
    print("Materialized GEMsim", lock["commit"], "at", target)


if __name__ == "__main__":
    main()

