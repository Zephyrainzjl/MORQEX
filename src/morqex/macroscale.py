"""Layer stack with common lateral strain and continuous normal/shear traction."""
from dataclasses import dataclass
import numpy as np


@dataclass
class StackState:
    strain: np.ndarray
    stress: np.ndarray
    thickness_change: float
    axial_stress: float
    energy_per_area: float


class LayerStack:
    def __init__(self, responses, thicknesses):
        self.responses = tuple(responses)
        self.thicknesses = np.asarray(thicknesses, dtype=float)
        if self.thicknesses.shape != (len(self.responses),) or not len(self.responses) or np.any(self.thicknesses <= 0) or not np.isfinite(self.thicknesses).all():
            raise ValueError("Positive layer thicknesses required")
        self.a, self.b = np.array([0, 4, 5]), np.array([1, 2, 3])
        self.inverse = []
        for response in self.responses:
            c = response.H[np.ix_(self.a, self.a)]
            if np.linalg.eigvalsh(c).min() <= np.linalg.norm(c)*1e-12: raise ValueError("Layer lacks axial/shear load-bearing stiffness")
            self.inverse.append(np.linalg.inv(c))

    def solve(self, coefficients, *, lateral_strain=(0., 0., 0.), pressure=None, thickness_change=None):
        if (pressure is None) == (thickness_change is None): raise ValueError("Specify pressure OR total thickness change")
        if len(coefficients) != len(self.responses): raise ValueError("One chemical state per layer required")
        lateral = np.asarray(lateral_strain, dtype=float)
        if lateral.shape != (3,) or not np.isfinite(lateral).all() or not np.isfinite(pressure if pressure is not None else thickness_change):
            raise ValueError("Finite lateral strain and external constraint required")
        offsets = []
        for response, inverse, a in zip(self.responses, self.inverse, coefficients):
            a = np.asarray(a, dtype=float)
            if a.shape != (response.dimension-6,): raise ValueError("Layer concentration dimension mismatch")
            offsets.append(-inverse @ (response.H[np.ix_(self.a, self.b)] @ lateral+response.H[self.a, 6:] @ a+response.h[self.a]))
        compliance = sum(t*inv[0, 0] for t, inv in zip(self.thicknesses, self.inverse))
        free_change = sum(t*offset[0] for t, offset in zip(self.thicknesses, offsets))
        axial = -float(pressure) if pressure is not None else (float(thickness_change)-free_change)/compliance
        strains, stresses, energies = [], [], []
        for response, inverse, offset, a in zip(self.responses, self.inverse, offsets, coefficients):
            e = np.zeros(6); e[self.b] = lateral; e[self.a] = inverse @ np.array([axial, 0., 0.])+offset
            q = np.r_[e, a]
            strains.append(e); stresses.append(response.macro_stress(q)); energies.append(response.energy_density(q))
        strains, stresses = np.array(strains), np.array(stresses)
        return StackState(strains, stresses, float(self.thicknesses @ strains[:, 0]), float(axial), float(self.thicknesses @ energies))
