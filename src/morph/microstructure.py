"""Validated geometry contract; phase semantics are never inferred from labels."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

PHASE_KINDS = frozenset({"active", "cbd", "pore", "inert"})


@dataclass(frozen=True)
class Microstructure:
    labels: np.ndarray
    spacing_zyx_m: tuple[float, float, float]
    origin_zyx_m: tuple[float, float, float]
    phase_kinds: dict[int, str]
    provenance: dict

    def __post_init__(self):
        a = np.asarray(self.labels)
        if a.ndim != 3 or not a.size or a.dtype.kind not in "iu":
            raise ValueError("Expected nonempty integer phase labels in z,y,x order")
        spacing = np.asarray(self.spacing_zyx_m, dtype=float)
        origin = np.asarray(self.origin_zyx_m, dtype=float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Spacing must contain three positive SI lengths")
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Origin must contain three finite SI coordinates")
        kinds = dict(self.phase_kinds)
        if any(k not in PHASE_KINDS for k in kinds.values()):
            raise ValueError("Unknown phase kind")
        if set(map(int, np.unique(a))) != set(kinds):
            raise ValueError("Provide exactly one explicit semantic entry per present phase")
        # Own the geometry: caller mutations cannot silently invalidate its cache key.
        a = np.array(a, copy=True, order="C")
        a.flags.writeable = False
        object.__setattr__(self, "labels", a)
        object.__setattr__(self, "spacing_zyx_m", tuple(spacing))
        object.__setattr__(self, "origin_zyx_m", tuple(origin))
        object.__setattr__(self, "phase_kinds", kinds)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.labels.tobytes())
        h.update(json.dumps({"shape": self.labels.shape, "dtype": self.labels.dtype.str,
                             "spacing": self.spacing_zyx_m, "origin": self.origin_zyx_m,
                             "phases": sorted(self.phase_kinds.items())}, sort_keys=True).encode())
        return h.hexdigest()

