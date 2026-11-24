"""Verify integrated source, document links, literature schema and artifact hygiene."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Record intentional final integrated hashes")
    args = parser.parse_args()
    migration = json.loads((ROOT / "provenance/gemsim_migration.json").read_text())
    assert len(migration["files"]) == 274
    rows = []
    for row in migration["files"]:
        path = ROOT / row["integrated"]
        assert path.is_file(), path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": row["integrated"], "sha256": digest,
                     "additional_local_adaptation": digest != row["sha256_after_namespace_migration"]})
    destination = ROOT / "provenance/integration_state.json"
    state = {"upstream_commit": migration["commit"], "files": rows,
             "additional_changes": {"src/morqex/microstructure/simulate/native/pyproject.toml":
                                    "Removed in-source scikit-build build-dir; launcher TMPDIR is external"}}
    if args.record:
        destination.write_text(json.dumps(state, indent=2))
    else:
        assert json.loads(destination.read_text()) == state, "Integrated source differs; review and record intentional changes"
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "gem" and not (node.module or "").startswith("gem."), path
            elif isinstance(node, ast.Import):
                assert all(alias.name != "gem" and not alias.name.startswith("gem.") for alias in node.names), path
    forbidden = {".npz", ".npy", ".tif", ".tiff", ".pdf", ".so", ".o", ".pyc", ".whl", ".pt", ".pth"}
    large = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        assert path.suffix not in forbidden, path
        if path.stat().st_size > 2 * 2**20:
            large.append(str(path))
    assert not large, large
    link_count = 0
    for path in [*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")]:
        markdown = re.sub(r"```[\s\S]*?```", "", path.read_text())
        markdown = re.sub(r"`[^`\n]*`", "", markdown)
        for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", markdown):
            target = match.group(1).split("#", 1)[0]
            if not target or "://" in target:
                continue
            assert (path.parent / target).exists(), (path, target)
            link_count += 1
    with (ROOT / "literature/references.tsv").open() as f:
        refs = list(csv.DictReader(f, delimiter="\t"))
    with (ROOT / "literature/capabilities.tsv").open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        methods = list(reader)
        assert len(reader.fieldnames) == 25
    assert len(refs) == 45 and len({r["id"] for r in refs}) == 45
    assert len(methods) == 28
    assert all(r["source_id"] == "THEORY/RESEARCH_PLAN" or r["source_id"] in {x["id"] for x in refs} for r in methods)
    print(json.dumps({"integrated_files": len(rows), "intentional_additional_adaptations": sum(r["additional_local_adaptation"] for r in rows),
                      "references": len(refs), "methods": len(methods), "capability_fields": 25,
                      "local_document_links": link_count, "forbidden_artifacts": 0}, indent=2))


if __name__ == "__main__":
    main()
