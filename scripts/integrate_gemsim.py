"""Reproducibly integrate pinned GEMsim text source under morqex.microstructure.

This is a one-time migration tool, not a runtime dependency or an automatic updater.
It refuses to overwrite existing files. Review intentional changes in provenance.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent / "morph_workspace"
PREFIX = "morqex.microstructure"


def transform(relative, data):
    text = data.decode("utf8")
    if relative.suffix in {".py", ".pyx", ".toml"}:
        text = re.sub(r"\bgem\.", PREFIX + ".", text)
        text = re.sub(r"\bfrom gem import\b", "from " + PREFIX + " import", text)
        text = text.replace("files('gem')", "files('" + PREFIX + "')")
        if relative.as_posix() == "io.py":
            # Also accept artifacts serialized under the original gem namespace.
            text = text.replace("        module=modules.get(module,module)",
                "        if module == 'gem' or module.startswith('gem.'):\n"
                "            module='morqex.microstructure'+module[3:]\n"
                "        module=modules.get(module,module)")
        if relative.as_posix() == "workspace.py":
            text = text.replace("parents[2]", "parents[3]")
            text = text.replace("Path.home() / 'GEMWorkspace'",
                                "Path(__file__).resolve().parents[4] / 'morph_workspace'")
            text = text.replace("os.environ.get('GEM_WORKSPACE',", "os.environ.get('MORQEX_WORKSPACE',")
        if relative.as_posix() in {"generate/v1.py", "generate/spatial.py"}:
            text = text.replace("parents[2]", "parents[4]")
        if relative.as_posix() == "cli.py":
            text = text.replace("prog='gem'", "prog='morqex'").replace("default='cuda'", "default='cpu'")
    return text.encode("utf8")


def main():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())["gemsim"]
    source = WORKSPACE / lock["source_relative_to_workspace"]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if commit != lock["commit"] or status:
        raise RuntimeError("Upstream must match the clean pinned commit")
    target = ROOT / "src/morqex/microstructure"
    if target.exists():
        raise RuntimeError("Destination exists; integration never overwrites local source")
    target.mkdir()
    manifest = []
    for path in sorted((source / "src/gem").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source / "src/gem")
        original = path.read_bytes()
        result = transform(relative, original)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(result)
        manifest.append({"upstream": "src/gem/" + relative.as_posix(),
                         "integrated": str(destination.relative_to(ROOT)), "bytes": len(result),
                         "sha256_original": hashlib.sha256(original).hexdigest(),
                         "sha256_after_namespace_migration": hashlib.sha256(result).hexdigest(),
                         "namespace_migration_changed": result != original})
    (ROOT / "provenance").mkdir(exist_ok=True)
    (ROOT / "provenance/gemsim_migration.json").write_text(json.dumps({
        "repository": lock["url"], "commit": commit,
        "policy": "All src/gem source integrated; no weights, datasets or binaries",
        "changes": "namespace/importlib resource paths; source boundary; workspace default; CLI CPU default; legacy import mapping",
        "files": manifest}, indent=2))
    for name in ("README.md", "pyproject.toml", "setup.py", "docs/api_inventory.json", "docs/migration_manifest.json"):
        destination = ROOT / "provenance/gemsim_upstream" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / name).read_bytes())
    print(json.dumps({"files": len(manifest), "bytes": sum(r["bytes"] for r in manifest), "destination": str(target)}))


if __name__ == "__main__":
    main()
