"""Auditable segmented CT import. No implicit thresholding, filling or phase repair."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from .geometry import MicrostructureData

@dataclass(frozen=True)
class SegmentedCT:
    path:Path
    sha256:str
    spacing_m:tuple[float,float,float]
    phase_kinds:dict[int,str]
    axes_to_zyx:tuple[int,int,int]=(0,1,2)
    crop:tuple[tuple[int,int],tuple[int,int],tuple[int,int]]|None=None
    stride:int=1
    source_url:str=''

    def build(self,workspace=None):
        import tifffile
        path=Path(self.path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.sha256:raise ValueError('CT checksum mismatch')
        if sorted(self.axes_to_zyx)!=[0,1,2] or self.stride!=1:raise ValueError('Explicit axis permutation required; implicit CT downsampling is not permitted')
        with tifffile.TiffFile(path) as f:
            shape=f.series[0].shape
            if len(shape)!=3 or np.prod(shape)*np.dtype(f.series[0].dtype).itemsize>128*2**20:raise ValueError('CT decoded volume exceeds 128 MiB or is not 3D')
            labels=f.asarray(maxworkers=1).transpose(self.axes_to_zyx)
        original_shape=labels.shape
        if self.crop is not None:
            if len(self.crop)!=3 or any(a<0 or b<=a or b>n for (a,b),n in zip(self.crop,labels.shape)):raise ValueError('Invalid CT crop')
            labels=labels[tuple(slice(a,b) for a,b in self.crop)]
        present=set(map(int,np.unique(labels)))
        if not present<=self.phase_kinds.keys():raise ValueError('Unknown CT labels; no implicit remapping')
        origin=tuple(a*s for (a,_),s in zip(self.crop,self.spacing_m)) if self.crop else (0.,)*3
        return MicrostructureData(labels,self.spacing_m,origin,{k:self.phase_kinds[k] for k in sorted(present)},
            {'provider':'segmented-CT','source_url':self.source_url,'source_sha256':self.sha256,'source_shape_zyx':original_shape,
             'axes_to_zyx':self.axes_to_zyx,'crop_zyx':self.crop,'resampling':'none','scope':'cropped SVE, no RVE representativeness claim'})

@dataclass(frozen=True)
class ArrayGeometry:
    structure:MicrostructureData
    def build(self,workspace=None):return self.structure
