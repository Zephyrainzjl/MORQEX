"""Orthonormal Mandel order: xx,yy,zz,sqrt(2)yz,sqrt(2)xz,sqrt(2)xy."""
import numpy as np

PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
MANDEL_BASIS = np.zeros((6, 3, 3))
for k, (i, j) in enumerate(PAIRS):
    MANDEL_BASIS[k, i, j] = MANDEL_BASIS[k, j, i] = 1 if i == j else 1 / np.sqrt(2)


def to_mandel(tensor):
    a = np.asarray(tensor, dtype=float)
    if a.shape[-2:] != (3, 3) or not np.isfinite(a).all():
        raise ValueError("Expected finite symmetric (...,3,3) tensor")
    if not np.allclose(a, np.swapaxes(a, -1, -2), atol=1e-14, rtol=1e-12):
        raise ValueError("Tensor must be symmetric")
    return np.einsum("kij,...ij->...k", MANDEL_BASIS, a)


def from_mandel(vector):
    a = np.asarray(vector, dtype=float)
    if a.shape[-1:] != (6,) or not np.isfinite(a).all():
        raise ValueError("Expected finite (...,6) Mandel vector")
    return np.einsum("...k,kij->...ij", a, MANDEL_BASIS)


def isotropic_stiffness(young, poisson):
    if not np.isfinite([young, poisson]).all() or young <= 0 or not -1 < poisson < .5:
        raise ValueError("Require E>0 and -1<nu<0.5")
    shear = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    return 2 * shear * np.eye(6) + lame * np.outer([1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0])


def validate_stiffness(stiffness):
    c = np.asarray(stiffness, dtype=float)
    if c.shape[-2:] != (6, 6) or not np.isfinite(c).all():
        raise ValueError("Expected finite (...,6,6) stiffness")
    scale = np.max(np.abs(c), axis=(-2, -1))
    asym = np.max(np.abs(c - np.swapaxes(c, -1, -2)), axis=(-2, -1))
    if np.any(asym > 1e-12 * scale) or np.any(np.linalg.eigvalsh(c)[..., 0] <= 0):
        raise ValueError("Stiffness must be symmetric positive definite")
    return c


def rotate_stiffness(stiffness, rotation):
    c = validate_stiffness(stiffness)
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), atol=1e-12) or np.linalg.det(r) < 0:
        raise ValueError("Rotation must be proper orthogonal")
    transformed = np.einsum("ia,kab,jb->kij", r, MANDEL_BASIS, r)
    t = np.einsum("aij,bij->ab", MANDEL_BASIS, transformed)
    return np.einsum("ai,...ij,bj->...ab", t, c, t)


def elastic_energy_and_mu(strain, concentration, stiffness, expansion, *, stiffness_derivative=None,
                          eigenstrain_reference=None, reference_concentration=0.):
    """Local fixed-strain energy and mechanical mu, c in mol/m^3, beta in m^3/mol.

    This derivative accepts an explicitly supplied C'(c); state evaluation of C
    is the caller's responsibility. No hidden material law is inferred.
    """
    eps = np.asarray(strain, dtype=float)
    beta = np.asarray(expansion, dtype=float)
    inherent = beta * (np.asarray(concentration) - reference_concentration)[..., None]
    if eigenstrain_reference is not None:
        inherent = inherent + eigenstrain_reference
    elastic = eps - inherent
    stress = np.einsum("...ij,...j->...i", validate_stiffness(stiffness), elastic)
    energy = .5 * np.sum(elastic * stress, axis=-1)
    mu = -np.sum(stress * beta, axis=-1)
    if stiffness_derivative is not None:
        prime = np.asarray(stiffness_derivative, dtype=float)
        if prime.shape[-2:] != (6, 6) or not np.isfinite(prime).all():
            raise ValueError("C' must be a finite Mandel tensor")
        mu = mu + .5 * np.einsum("...i,...ij,...j->...", elastic, prime, elastic)
    return energy, stress, mu
