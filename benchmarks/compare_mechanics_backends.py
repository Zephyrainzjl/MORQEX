"""Same Tet4 equations/precision/output: reused sparse LU vs element CG/Jacobi."""
import json
import hashlib
from pathlib import Path
import time
import numpy as np
from morqex.mechanics import TetMesh, ElasticitySolver, MatrixFreeElasticitySolver, isotropic_stiffness
from morqex.microstructure import Volume


def main():
    records = []
    for n in (4, 8):
        mesh = TetMesh.from_volume(Volume(np.ones((n,)*3, dtype=np.uint8), spacing=(1/n,)*3))
        for contrast in (1., 100.):
            base = isotropic_stiffness(1e9, .25)
            stiffness = np.broadcast_to(base, (len(mesh.cells), 6, 6)).copy()
            stiffness[mesh.centers[:, 0] < .5] /= contrast
            rng = np.random.default_rng(571)
            inherent = rng.normal(scale=.001, size=(len(mesh.cells), 6))
            states = []
            for name, backend in (("sparse-LU", ElasticitySolver), ("element-CG-Jacobi", MatrixFreeElasticitySolver)):
                start = time.perf_counter(); solver = backend(mesh, stiffness)
                setup = time.perf_counter()-start
                samples = []
                for _ in range(3):
                    start = time.perf_counter(); state = solver.solve(inherent); samples.append(time.perf_counter()-start)
                states.append(state)
                records.append({"n": n, "elements": len(mesh.cells), "contrast": contrast, "backend": name,
                    "setup_s": setup, "median_solve_and_fields_s": float(np.median(samples)),
                    "iterations": getattr(solver, "iterations", None), "residual": state.equilibrium_residual,
                    "retained_stiffness_bytes": int(solver.K.data.nbytes+solver.K.indices.nbytes+solver.K.indptr.nbytes+solver.factor.L.data.nbytes+solver.factor.L.indices.nbytes+solver.factor.L.indptr.nbytes+solver.factor.U.data.nbytes+solver.factor.U.indices.nbytes+solver.factor.U.indptr.nbytes)
                        if name == "sparse-LU" else int(solver.element_stiffness.nbytes),
                    "storage_scope": "stiffness and LU data/indices/pointers OR element stiffness only; excludes D/B/C/constraints/transient allocations, not peak RSS"})
            error = np.linalg.norm(states[0].stress-states[1].stress)/np.linalg.norm(states[0].stress)
            assert error < 1e-9
            records[-1]["stress_relative_difference_from_LU"] = float(error)
    source = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [*sorted((source/"src/morqex/mechanics").glob("*.py")), Path(__file__)]}
    Path("mechanics_backends.json").write_text(json.dumps({"records": records, "source_sha256": hashes,
        "scope": "fixed KUBC Tet4; iterative tolerance1e-11; no AMG or FFT comparison yet"}, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__": main()
