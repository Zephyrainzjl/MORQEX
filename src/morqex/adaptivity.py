"""Stateless, residual-certified Galerkin response enrichment.

Enrichment changes approximation energy. Its correction is reported explicitly;
this class does not silently change a transient coupled model's stored energy.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class AdaptiveResponseResult:
    response: object
    error_bound_density: float
    enriched: bool
    model_energy_correction_density: float
    rank: int


class AdaptiveResponse:
    def __init__(self, builder, basis):
        self.builder = builder
        raw = np.asarray(basis, dtype=float)
        if raw.ndim != 2 or not np.isfinite(raw).all(): raise ValueError("Invalid enrichment basis")
        if raw.shape[1] and np.linalg.matrix_rank(raw) != raw.shape[1]: raise ValueError("Dependent enrichment basis")
        self.basis = np.linalg.qr(raw, mode="reduced")[0]
        self.response, self.certificate = builder.galerkin(self.basis)

    def evaluate(self, q, *, absolute_energy_tolerance):
        if absolute_energy_tolerance <= 0: raise ValueError("Positive energy-density error tolerance required")
        q = np.asarray(q, dtype=float)
        estimate = float(self.certificate.potential_error_density(q))
        before = float(self.response.energy_density(q))
        enriched = estimate > absolute_energy_tolerance
        if enriched:
            direction = self.builder.solver.solve_reduced(self.builder.rhs @ q+self.builder.offset_rhs)
            for _ in range(2): direction -= self.basis @ (self.basis.T @ direction)
            norm = np.linalg.norm(direction)
            if norm <= np.finfo(float).tiny: raise RuntimeError("Residual enrichment direction vanished")
            self.basis = np.column_stack([self.basis, direction/norm])
            self.response, self.certificate = self.builder.galerkin(self.basis)
            estimate = float(self.certificate.potential_error_density(q))
            if estimate > absolute_energy_tolerance: raise RuntimeError("Enrichment did not meet the requested discrete error tolerance")
        correction = float(self.response.energy_density(q))-before
        return AdaptiveResponseResult(self.response, estimate, enriched, correction, self.basis.shape[1])
