"""Three explicit local-RVE bindings; each macro location owns its state.

Slab mode samples an actual native-resolution crop *inside* each thickness slab.
It is a representative-unit approximation, not a claim that the sampled crops
resolve the whole physical slab. No silent interpolation, stitching or repair.
"""
from dataclasses import dataclass,field
import json
from pathlib import Path
import numpy as np
from ..geometry import MicrostructureData
from ..generation.statistics import StatisticsCard,characterize,generate,connectivity

@dataclass(frozen=True)
class BindingSpec:
    mode:str='shared'
    shape:tuple=(16,16,16)
    seed:int=104
    spacing_m:tuple|None=None
    candidates:int=27
    require_percolation:bool=True
    statistics:StatisticsCard|None=None
    generator:str='gaussian'
    generator_options:dict=field(default_factory=dict)
    def __post_init__(self):
        if self.mode not in {'shared','slab','statistical'} or len(self.shape)!=3 or min(self.shape)<2 or any(int(n)!=n for n in self.shape) or self.candidates<1:raise ValueError('Explicit shared/slab/statistical mode, integer grid and sampling budget required')
        if self.generator not in {'gaussian','microstructpy','custom'}:raise ValueError('Select a named generation backend')

@dataclass(frozen=True)
class NodeGeometry:
    domain:str
    index:int
    x_fraction:float
    macro_weight:float
    structure:MicrostructureData
    selection:dict
    @property
    def operator_geometry(self):
        # Translation is not a constitutive parameter. Source coordinates live
        # in selection; canonical local coordinates improve operator reuse.
        s=self.structure
        return MicrostructureData(s.labels,s.spacing_zyx_m,(0.,)*3,s.phase_kinds,s.provenance)

def default_reference(workspace,domain):
    import tifffile
    root=workspace.root
    if domain=='positive':
        source=json.loads((root/'data/ct/full_electrodes.json').read_text())['nmc1cal'];axes=source['axes_to_zyx']
    elif domain=='negative':
        source=json.loads((root/'data/ct/sources.json').read_text())['graphite5cal'];axes=[1,2,0]
    else:raise ValueError('Electrode domain required')
    from ..visualization.voxel_fields import file_hash
    if file_hash(source['path'])!=source['sha256']:raise ValueError('Default CT checksum mismatch')
    labels=tifffile.imread(source['path'],maxworkers=1).transpose(axes);present=set(map(int,np.unique(labels)))
    return MicrostructureData(labels,(source['voxel_size_m'],)*3,(0.,)*3,{p:{0:'pore',1:'active',2:'cbd'}[p] for p in present},{'source':source,'provider':'complete supplied CT','axes_to_zyx':axes})

def _crop(reference,shape,slab,candidates,require_percolation):
    a=reference.labels;shape=np.asarray(shape,int);start,end=slab
    if np.any(shape>np.array(a.shape)) or shape[2]>end-start:raise ValueError('Representative crop must fit inside its thickness slab; reduce node count or crop size explicitly')
    target=a[:,:,start:end];phases=sorted(reference.phase_kinds);fractions=np.array([np.mean(target==p) for p in phases]);grid=max(1,int(np.ceil(candidates**(1/3))))
    choices=[np.unique(np.linspace(0,a.shape[k]-shape[k],grid).astype(int)) for k in [0,1]]
    choices.append(np.unique(np.linspace(start,end-shape[2],grid).astype(int)));best=None;parent_fingerprint=reference.fingerprint
    for z in choices[0]:
        for y in choices[1]:
            for x in choices[2]:
                origin=(int(z),int(y),int(x));labels=a[tuple(slice(v,v+n) for v,n in zip(origin,shape))]
                if set(map(int,np.unique(labels)))!=set(phases):continue
                data=MicrostructureData(labels,reference.spacing_zyx_m,tuple(o+v*h for o,v,h in zip(reference.origin_zyx_m,origin,reference.spacing_zyx_m)),reference.phase_kinds,
                    {'provider':'native CT representative crop','parent_fingerprint':parent_fingerprint,'crop_zyx':[[int(v),int(v+n)] for v,n in zip(origin,shape)],'source':reference.provenance})
                if require_percolation and not all(connectivity(data).values()):continue
                error=float(np.linalg.norm(np.array([np.mean(labels==p) for p in phases])-fractions));centre_error=abs((x+shape[2]/2)-(start+end)/2)/a.shape[2]
                score=error+.02*centre_error
                if best is None or score<best[0]:best=(score,data,error)
    if best is None:raise ValueError('No admissible native CT representative in this slab; no silent phase repair or fallback')
    return best[1],{'slab_x_voxels':[start,end],'phase_fraction_absolute_L2_error':best[2],'selection_score':best[0],'representativeness':'phase-fraction-ranked finite candidate set; not a certified RVE'}

def bind_microstructures(spec,nodes,domain,workspace,*,reference=None,template=None,overrides=None,node_edges=None,generator=None):
    if domain not in {'negative','positive'} or not isinstance(nodes,int) or nodes<1:raise ValueError('Electrode and positive macro node count required')
    overrides=overrides or {}
    if not set(overrides)<=set(range(nodes)):raise ValueError('Override node index outside the macro grid')
    source=reference;bindings=[]
    if source is None and len(overrides)==nodes:source=overrides[0]
    needs_source=spec.mode=='slab' or spec.mode=='shared' and template is None or spec.mode=='statistical' and template is None and (spec.statistics is None or spec.spacing_m is None)
    if source is None and needs_source:source=default_reference(workspace,domain)
    edges=np.linspace(0,1,nodes+1) if node_edges is None else np.asarray(node_edges,float)
    if edges.shape!=(nodes+1,) or not np.isfinite(edges).all() or edges[0]!=0 or edges[-1]!=1 or min(np.diff(edges))<=0:raise ValueError('Ordered macro cell edges must span [0,1]')
    if template is not None and template.labels.shape!=tuple(spec.shape):raise ValueError('Explicit template shape mismatch; resizing must be an explicit separate operation')
    if spec.mode=='shared':
        chosen,selection=(template,{'selection':'explicit template'}) if template is not None else _crop(source,spec.shape,(0,source.labels.shape[2]),spec.candidates,spec.require_percolation)
    elif spec.mode=='statistical':
        card=spec.statistics if spec.statistics is not None else characterize(source or template)[0]
        spacing=spec.spacing_m or (source or template).spacing_zyx_m
        if template is not None:chosen=template
        elif spec.generator=='gaussian':chosen=generate(card,spec.shape,spacing,seed=spec.seed,require_percolation=spec.require_percolation,**spec.generator_options)
        elif spec.generator=='microstructpy':
            from ..generation.upstream import microstructpy_voxels
            if len(set(spacing))!=1:raise ValueError('MicroStructPy raster adapter requires isotropic voxel spacing')
            options=dict(spec.generator_options)
            if 'phases' not in options:raise ValueError('MicroStructPy requires explicit particle shape/size distributions; phase correlations do not uniquely determine them')
            chosen=microstructpy_voxels(phase_kinds=card.phase_kinds,shape=spec.shape,spacing_m=spacing[0],seed=spec.seed,require_percolation=spec.require_percolation,**options)
        else:
            if generator is None:raise ValueError('A custom generation callable is required')
            chosen=generator(card=card,shape=spec.shape,spacing_m=spacing,seed=spec.seed,**spec.generator_options)
        selection={'selection':'one global-statistics draw shared across macro locations','statistics':card.metadata(),'seed':spec.seed}
    for i in range(nodes):
        if i in overrides:geometry=overrides[i];details={'selection':'explicit node override'}
        elif spec.mode=='slab':
            begin=int(edges[i]*source.labels.shape[2]);end=int(edges[i+1]*source.labels.shape[2])
            geometry,details=_crop(source,spec.shape,(begin,end),spec.candidates,spec.require_percolation)
        else:geometry,details=chosen,dict(selection)
        if geometry.labels.shape!=tuple(spec.shape):raise ValueError('Node geometry shape mismatch')
        if spec.require_percolation and not all(connectivity(geometry).values()):raise ValueError('Explicit geometry lacks a through-thickness physical transport channel')
        bindings.append(NodeGeometry(domain,i,float((edges[i]+edges[i+1])/2),float(edges[i+1]-edges[i]),geometry,details|{'mode':spec.mode,'macro_cell_fraction':edges[i:i+2].tolist()}))
    if spec.mode=='slab' and len({b.operator_geometry.fingerprint for b in bindings})!=nodes:raise ValueError('Slabs selected duplicate geometric templates; increase candidate budget or provide explicit distinct overrides')
    return bindings
