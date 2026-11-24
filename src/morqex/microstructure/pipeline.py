"""Unified fluent facade with the same Volume object at every stage."""
from dataclasses import dataclass
from .volume import Volume

@dataclass
class Microstructure:
    volume: Volume

    @classmethod
    def load(cls,path,**kwargs): return cls(Volume.load(path,**kwargs))

    @classmethod
    def generate(cls,checkpoint,**kwargs):
        from .generate.spatial import draw_volume
        return cls(draw_volume(checkpoint,**kwargs))

    def save(self,path): self.volume.save(path);return self

    def mesh(self,kind='tetra',phases=None):
        from .mesh import voxel_to_tetra,voxel_to_hex
        if kind not in ('tetra','hex'): raise ValueError('kind=tetra/hex')
        return (voxel_to_tetra if kind == 'tetra' else voxel_to_hex)(self.volume,phases)

    def describe(self):
        from .analysis import describe
        return describe(self.volume)

    def show(self):
        from .view import slices
        return slices(self.volume)

    def homogenize(self,coefficients,order=2):
        from .simulate.homogenization import homogenize
        return homogenize(self.volume,coefficients,order)
