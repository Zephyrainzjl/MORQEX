"""Stable public facade for validated model bundles stored outside the source tree."""
from dataclasses import dataclass
from pathlib import Path
import json
from morqex.microstructure.volume import Volume

@dataclass
class Reconstructor:
    checkpoint:Path
    references:tuple
    method:str='diffusion'
    options:dict=None
    validation:dict=None

    @classmethod
    def load(cls,manifest,workspace=None):
        path=Path(manifest).resolve()
        data=json.loads(path.read_text(encoding='utf8'))
        base=Path(workspace).resolve() if workspace else path.parent
        def resolve(p):
            p=Path(p)
            return p if p.is_absolute() else base/p
        return cls(resolve(data['checkpoint']),tuple(resolve(p) for p in data['references']),
                   data['method'],data.get('options',{}),data.get('validation',{}))

    def generate(self,fractions=None,size=64,seed=42,device='cuda',spacing=(1.,1.,1.),**overrides):
        options={**(self.options or {}),**overrides}
        coating=options.pop('coating',None)
        if self.method=='hybrid':
            from .hybrid import sample_student
            volume=sample_student(self.checkpoint,paths=self.references,fractions=fractions,
                size=size,seed=seed,device=device,**options)
            volume.spacing=spacing
        elif self.method=='volume_diffusion':
            if fractions is not None:
                raise ValueError('This model reconstructs microstructure without a numerical property condition')
            from .spatial import draw_volume
            volume=draw_volume(self.checkpoint,size=size,seed=seed,device=device,spacing=spacing,**options)
        elif self.method=='diffusion':
            from .v1 import sample
            volume=sample(self.checkpoint,paths=self.references,fractions=fractions,
                size=size,seed=seed,device=device,spacing=spacing,**options)
        else:raise ValueError('Unknown reconstruction method')
        if coating:
            from .constraints import surface_coating
            target=volume.metadata.get('requested_fractions')
            phase_ids=coating['phases']
            coating_id=phase_ids.index(coating['coating'])
            fraction=coating.get('fraction',target[coating_id] if target is not None else None)
            if fraction is None:raise ValueError('Specify the material-prior coating fraction')
            volume=surface_coating(volume,coating['substrate'],coating['coating'],coating['matrix'],fraction,seed=seed)
        volume.metadata['model_bundle']={'method':self.method,'checkpoint':str(self.checkpoint)}
        return volume
