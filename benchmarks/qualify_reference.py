"""Actual manufactured/analytic convergence, not a ROM self-comparison."""
import json
import hashlib
from pathlib import Path
import time
import numpy as np
from qualification_meshes import fitted_ball
from morqex.mechanics import TetMesh, ElasticitySolver, isotropic_stiffness, to_mandel
from morqex.microstructure import Volume


def sphere(fitted=False, refinement=16):
    young, poisson, beta, delta, radius = 1e9, .25, 1e-6, 1000., .4
    exact_energy = 16*np.pi*young*beta**2*delta**2*radius**3/(175*(1-poisson))
    records = []
    refinements = (4, 8, 12, 16) + tuple(n for n in (24, 28, 32) if fitted and n <= refinement)
    for n in refinements:
        started = time.perf_counter()
        grid = (np.indices((n,)*3).transpose(1, 2, 3, 0)+.5)/n-.5
        labels = (np.linalg.norm(grid, axis=-1) < radius).astype(np.uint8)
        mesh = fitted_ball(n, radius) if fitted else TetMesh.from_volume(Volume(labels, spacing=(1/n,)*3), solid_phases=[1])
        solver = ElasticitySolver(mesh, isotropic_stiffness(young, poisson), boundary="neumann")
        x = mesh.centers-.5
        r = np.linalg.norm(x, axis=1)
        c = 5000+delta*(r/radius)**2
        inherent = np.zeros((len(r), 6)); inherent[:, :3] = beta*c[:, None]
        state = solver.solve(inherent)
        prefactor = 2*young*beta*delta/(5*(1-poisson))
        rr = prefactor*(1-(r/radius)**2)
        tt = prefactor*(1-2*(r/radius)**2)
        normal = np.divide(x, r[:, None], out=np.zeros_like(x), where=r[:, None] > 0)
        analytic = to_mandel(tt[:, None, None]*np.eye(3)+(rr-tt)[:, None, None]*normal[:, :, None]*normal[:, None, :])
        inside = r < radius
        interior = r < .65*radius
        def error(mask):
            difference = state.stress[mask]-analytic[mask]
            return float(np.sqrt(np.einsum("e,ei,ei->", mesh.volumes[mask], difference, difference)/np.einsum("e,ei,ei->", mesh.volumes[mask], analytic[mask], analytic[mask])))
        nodal_x = mesh.points-.5
        s = np.sum(nodal_x**2, axis=1)/radius**2
        exact_u = nodal_x*(beta*5000+beta*delta*(2*(1-2*poisson)+(1+poisson)*s)/(5*(1-poisson)))[:, None]
        rigid = np.zeros((len(mesh.points), 3, 6)); rigid[:, :, :3] = np.eye(3)
        for k in range(3): rigid[:, :, k+3] = np.cross(np.eye(3)[k], nodal_x)
        difference = state.displacement-exact_u
        alignment = np.linalg.lstsq(rigid.reshape(-1, 6), difference.ravel(), rcond=None)[0]
        difference -= (rigid @ alignment)
        records.append({"n": n, "nodes": len(mesh.points), "elements": len(mesh.cells),
            "solid_volume_error": float(abs(mesh.volumes.sum()-4*np.pi*radius**3/3)/(4*np.pi*radius**3/3)),
            "stress_L2_intersection": error(inside), "stress_L2_fixed_interior": error(interior),
            "displacement_relative_error_aligned": float(np.linalg.norm(difference)/np.linalg.norm(exact_u)),
            "energy_relative_error": float(abs(state.elastic_energy-exact_energy)/exact_energy),
            "equilibrium_residual": state.equilibrium_residual,
            "assembly_seconds": solver.assembly_seconds, "factorization_seconds": solver.factorization_seconds,
            "wall_seconds": time.perf_counter()-started})
        if n == (refinement if fitted else 16):
            np.savez_compressed("sphere_fitted_fields.npz" if fitted else "sphere_fields.npz", centers=mesh.centers, stress=state.stress, analytic=analytic,
                                displacement=state.displacement, points=mesh.points, cells=mesh.cells)
    # Only an observed refinement requirement, not an assumed asymptotic order on staircase geometry.
    assert records[-1]["stress_L2_fixed_interior"] < records[0]["stress_L2_fixed_interior"]
    return {"parameters": {"E_Pa": young, "nu": poisson, "beta_m3_mol": beta, "delta_c_mol_m3": delta,
                            "radius_m": radius, "provenance": "synthetic verification parameters, not material calibration"},
            "analytic_elastic_energy_J": exact_energy, "mesh_sequence": records,
            "scope": ("fitted convex-hull sphere" if fitted else "true-void voxel sphere")+"; both geometry and FE error vary; no ROM speedup claim"}


def manufactured():
    records = []
    c = isotropic_stiffness(100., .25)
    for n in (2, 4, 8):
        mesh = TetMesh.from_volume(Volume(np.ones((n,)*3, dtype=np.uint8), spacing=(1/n,)*3))
        solver = ElasticitySolver(mesh, c)
        exact_u = np.zeros_like(mesh.points); exact_u[:, 0] = .01*mesh.points[:, 0]**2
        state = solver.solve(prescribed_displacement=exact_u, force=mesh.body_load([-.02*c[0, 0], 0, 0]))
        exact = np.zeros_like(state.strain); exact[:, 0] = .02*mesh.centers[:, 0]
        error = state.strain-exact
        records.append({"n": n, "strain_L2": float(np.sqrt(np.einsum("e,ei,ei->", mesh.volumes, error, error))),
                        "nodal_displacement_L2": float(np.linalg.norm(state.displacement-exact_u)/np.sqrt(len(exact_u))),
                        "residual": state.equilibrium_residual})
    assert all(a["strain_L2"] > b["strain_L2"] for a, b in zip(records, records[1:]))
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--fitted-refinement", type=int, choices=(16, 24, 28), default=24)
    args = parser.parse_args()
    result = {"manufactured": manufactured(), "sphere": sphere(), "fitted_sphere": sphere(fitted=True, refinement=args.fitted_refinement)}
    source = Path(__file__).resolve().parents[1]
    paths = [*sorted((source/"src/morqex/mechanics").glob("*.py")), Path(__file__), Path(__file__).with_name("qualification_meshes.py")]
    result["source_sha256"] = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    Path("qualification.json").write_text(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    a, b = result["manufactured"], result["sphere"]["mesh_sequence"]
    axes[0].loglog([x["n"] for x in a], [x["strain_L2"] for x in a], "o-", label="Tet4 manufactured strain")
    for key, label in [("stress_L2_fixed_interior", "stress: fixed interior"), ("energy_relative_error", "elastic energy"), ("solid_volume_error", "geometry volume")]:
        axes[1].loglog([x["n"] for x in b], [x[key] for x in b], "o-", label=label)
    fitted = result["fitted_sphere"]["mesh_sequence"]
    axes[1].loglog([x["n"] for x in fitted], [x["stress_L2_intersection"] for x in fitted], "s--", label="fitted mesh: full stress")
    for ax in axes:
        ax.set_xlabel("voxels per RVE edge"); ax.set_ylabel("error"); ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    fig.savefig("reference_convergence.png", dpi=180)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
