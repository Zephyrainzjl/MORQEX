"""Small synthetic algebra checks. These DO NOT validate a continuum RVE solver."""
import json
from pathlib import Path
import numpy as np


def verify():
    rng = np.random.default_rng(20261124)
    # Elastic sample space, constrained displacement space, imposed strain/eigenstrain space.
    n, nu, nq = 30, 12, 9
    D = rng.normal(size=(n, nu))
    F = rng.normal(size=(n, nq))
    C = np.diag(np.exp(rng.uniform(-1, 1, n)))
    volume = 2.3
    K = D.T @ C @ D
    B = -D.T @ C @ F
    U = np.linalg.solve(K, B)
    H = (F.T @ C @ F - B.T @ U) / volume
    q = rng.normal(size=nq)
    u = np.linalg.solve(K, B @ q)
    elastic = D @ u + F @ q
    direct_energy = .5 * elastic @ C @ elastic / volume
    def reduced_energy(x):
        return .5 * x @ H @ x
    h = 1e-5
    fd = np.array([(reduced_energy(q+h*e)-reduced_energy(q-h*e))/(2*h) for e in np.eye(nq)])
    # Deliberately truncated displacement space: residual dual norm is an exact discrete identity.
    V, _ = np.linalg.qr(U[:, :3])
    Ur = V @ np.linalg.solve(V.T @ K @ V, V.T @ B)
    R = B - K @ Ur
    gram = R.T @ np.linalg.solve(K, R)
    error = u - Ur @ q
    actual_error2 = error @ K @ error
    estimated_error2 = q @ gram @ q
    # Generalized concentration derivative -> projected chemical potential using mass matrix.
    Phi = rng.normal(size=(16, 3))
    weights = rng.uniform(.5, 1.5, 16)
    weights /= weights.sum()  # normalized RVE volume measure
    M = Phi.T @ (weights[:, None] * Phi)
    g = (H @ q)[6:]
    coefficients = np.linalg.solve(M, g)
    projection_error = np.linalg.norm(Phi.T @ (weights * (Phi @ coefficients)) - g)
    # A scalar concentration-dependent stiffness: omitting C' is detectably wrong.
    strain, beta, c, stiffness0, slope = .12, .03, .4, 7., .8
    def psi(x):
        return .5 * stiffness0 * (1+slope*x) * (strain-beta*x)**2
    e = strain-beta*c
    mu_full = -stiffness0*(1+slope*c)*e*beta + .5*stiffness0*slope*e**2
    mu_incomplete = -stiffness0*(1+slope*c)*e*beta
    mu_fd = (psi(c+h)-psi(c-h))/(2*h)
    metrics = {
        "scope": "synthetic FP64 algebra; no RVE/DFN accuracy or speed claim",
        "seed": 20261124,
        "response_relative_error": float(np.linalg.norm(u-U@q)/np.linalg.norm(u)),
        "energy_relative_error": float(abs(direct_energy-reduced_energy(q))/direct_energy),
        "gradient_relative_error": float(np.linalg.norm(fd-H@q)/np.linalg.norm(H@q)),
        "H_symmetry_relative_error": float(np.linalg.norm(H-H.T)/np.linalg.norm(H)),
        "H_min_eigenvalue": float(np.linalg.eigvalsh(H).min()),
        "residual_identity_relative_error": float(abs(actual_error2-estimated_error2)/actual_error2),
        "mass_projection_error": float(projection_error),
        "variable_stiffness_mu_error": float(abs(mu_full-mu_fd)),
        "omitted_stiffness_derivative_error": float(abs(mu_incomplete-mu_fd)),
    }
    for key in ("response_relative_error", "energy_relative_error", "gradient_relative_error",
                "H_symmetry_relative_error", "residual_identity_relative_error", "mass_projection_error",
                "variable_stiffness_mu_error"):
        assert metrics[key] < 1e-8, (key, metrics[key])
    assert metrics["H_min_eigenvalue"] >= -1e-10
    assert metrics["omitted_stiffness_derivative_error"] > 1e-3
    return metrics


if __name__ == "__main__":
    metrics = verify()
    Path("energy_algebra.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

