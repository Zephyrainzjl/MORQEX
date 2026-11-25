"""Explicit ionic/electronic overlap policy for unresolved microporous CBD."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class PorousCBD:
    microporosity:float=.47
    ionic_efficiency:float=.05
    def __post_init__(self):
        if not 0<self.ionic_efficiency<=self.microporosity<1:raise ValueError('CBD efficiency must not exceed microporosity')
    def ionic_phases(self,structure):return [p for p,k in structure.phase_kinds.items() if k in {'pore','cbd'}]
    def ionic_efficiencies(self,structure):return {p:self.ionic_efficiency if k=='cbd' else 1. for p,k in structure.phase_kinds.items() if k in {'pore','cbd'}}
    def porosity(self,structure):
        return sum(float(np.mean(structure.labels==p))*(self.microporosity if k=='cbd' else 1.) for p,k in structure.phase_kinds.items() if k in {'pore','cbd'})
    def metadata(self):return asdict(self)|{'model':'overlapping ionic/electronic microporous CBD continuum','scope':'assumed uncalibrated CBD parameters; not CT-resolved micropores'}
