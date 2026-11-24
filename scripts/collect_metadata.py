"""Archive primary Crossref records; title-search hits require manual confirmation."""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request

DOIS = [
    "10.1016/j.jmps.2018.12.005", "10.1016/j.jpowsour.2021.230465",
    "10.1016/j.jpowsour.2022.232050", "10.1002/nme.70313", "10.1002/nme.70378",
    "10.1016/j.cma.2021.114334", "10.1002/nme.7547", "10.1002/nme.7501",
    "10.1016/j.cma.2026.118772", "10.1016/S0020-7683(03)00346-9",
    "10.1016/j.cma.2016.04.004", "10.1002/nme.4820", "10.1016/j.cma.2016.10.022",
    "10.1137/070688791", "10.1149/1.2221597", "10.1149/1.2054684",
    "10.1016/j.cma.2024.116932", "10.1016/j.jpowsour.2017.05.115",
]
TITLES = [
    "NTFA-enabled goal-oriented adaptive space-time finite elements for micro-heterogeneous elastoplasticity problems",
    "A numerical method for computing the overall response of nonlinear composites with complex microstructure",
    "Nonlinear Model Reduction via Discrete Empirical Interpolation",
    "A Sparse Matrix Arithmetic Based on H-Matrices. Part I: Introduction to H-Matrices",
    "A Multiscale Finite Element Method for Elliptic Problems in Composite Materials and Porous Media",
    "On a priori error analysis of fully discrete heterogeneous multiscale FEM",
    "Transformation field analysis of inelastic composite materials",
    "A computational homogenization approach for Li-ion battery cells: Part 1 - Formulation",
    "Eigenstrain-based reduced order homogenization for polycrystalline materials",
    "Parameterisation of a lithium-ion battery: a computer-aided methodology",
    "Multilevel finite element analysis of complex loading: multi-scale and micro-macro modelling of materials with microstructure",
    "A posteriori error estimation in finite element analysis",
    "A finite strain model of stress, diffusion, plastic flow and electrochemical reactions in a lithium-ion half-cell",
    "Computational homogenization of microstructures and RVE-based fracture modeling of lithium-ion battery electrodes",
]


def fetch(job):
    kind, query = job
    if kind == "doi":
        url = "https://api.crossref.org/works/" + urllib.parse.quote(query, safe="")
    else:
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.title": query, "rows": 2})
    req = urllib.request.Request(url, headers={"User-Agent": "MORQEX-local-research-audit/0.0.1"})
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            record = json.load(response)["message"]
        return {"kind": kind, "query": query, "url": url, "record": record}
    except Exception as exc:
        return {"kind": kind, "query": query, "url": url, "error": str(exc)}


def main():
    root = Path(os.environ["MORQEX_WORKSPACE"]) / "evidence/literature"
    root.mkdir(parents=True, exist_ok=True)
    jobs = [("doi", x) for x in DOIS] + [("title", x) for x in TITLES]
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, jobs))
    (root / "crossref_raw.json").write_text(json.dumps({"retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "records": records}, indent=2))
    summary = []
    for row in records:
        candidates = [row["record"]] if row.get("kind") == "doi" and "record" in row else row.get("record", {}).get("items", [])
        summary.append({"query": row["query"], "error": row.get("error"),
                        "candidates": [{key: r.get(key) for key in ("title", "DOI", "author", "container-title", "volume", "issue", "page", "article-number", "published", "URL", "link")} for r in candidates]})
    (root / "metadata_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"jobs": len(jobs), "success": sum("record" in r for r in records), "path": str(root)}, indent=2))


if __name__ == "__main__":
    main()
