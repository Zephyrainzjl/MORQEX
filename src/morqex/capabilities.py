"""Backend compatibility contracts: extension interfaces do not imply validation."""
from dataclasses import dataclass

@dataclass(frozen=True)
class BackendCapabilities:
    name:str
    features:frozenset[str]
    validation_scope:str
    def require(self,features):
        missing=set(features)-self.features
        if missing:raise ValueError(f'{self.name} does not support: '+', '.join(sorted(missing)))

REGIONAL_DFN=BackendCapabilities('regional-DFN/Tet4',frozenset({'voxel','fixed-geometry','fixed-anisotropic-elasticity',
    'linear-anisotropic-eigenstrain','diagonal-diffusion','single-host-chemistry','multiphase','porous-CBD','KUBC',
    'lumped-thermal','cohesive-scalar-damage','SEI','plating-stripping','dead-lithium','custom-partition','custom-energy-kernel'}),
    'Small numerical cases only. CT-derived microscopic SVEs; no universal geometry, material or degradation calibration.')
