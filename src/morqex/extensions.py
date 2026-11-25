"""Explicit, versioned Python extension contracts; no implicit plugin imports."""
from dataclasses import dataclass,field
from typing import Protocol,runtime_checkable,Callable,Any
import hashlib
import inspect
import json
import numpy as np

@runtime_checkable
class GeometryProvider(Protocol):
    def build(self,workspace):...

@runtime_checkable
class BasisBuilder(Protocol):
    def build(self,electrode,target:int)->np.ndarray:...

@runtime_checkable
class OperatorBuilder(Protocol):
    def build(self,electrode,partition:np.ndarray):...

@runtime_checkable
class Mechanism(Protocol):
    """A symbolic law called during assembly; must not mutate its context."""
    name:str
    def evaluate(self,context):...

@dataclass(frozen=True)
class ComponentSpec:
    name:str
    factory:Callable
    schema:int=1
    assumptions:tuple[str,...]=()
    units:dict[str,str]=field(default_factory=dict)

    def fingerprint(self):
        try:source=inspect.getsource(self.factory)
        except (OSError,TypeError):raise ValueError('Custom factory needs inspectable source for reproducibility')
        return hashlib.sha256(json.dumps({'name':self.name,'schema':self.schema,'source':source,
            'assumptions':self.assumptions,'units':self.units},sort_keys=True).encode()).hexdigest()

class Registry:
    """Separate registries can be passed by the caller; no mutable global singleton."""
    kinds=('geometry','basis','operator','energy','materials','topology','mesh','solver','sei','plating','damage','thermal','kinetics')
    def __init__(self):self._items={k:{} for k in self.kinds}
    def register(self,kind,spec):
        if kind not in self._items:raise ValueError('Unknown extension kind')
        if not isinstance(spec,ComponentSpec) or not spec.name or spec.schema!=1 or not callable(spec.factory):raise ValueError('Invalid component specification')
        if spec.name in self._items[kind]:raise ValueError('Duplicate component registration')
        spec.fingerprint();self._items[kind][spec.name]=spec;return self
    def create(self,kind,name,**configuration):return self._items[kind][name].factory(**configuration)
    def manifest(self):return {k:{n:s.fingerprint() for n,s in v.items()} for k,v in self._items.items()}

def validate_partition(electrode,labels):
    labels=np.asarray(labels)
    if labels.dtype.kind not in 'iu' or labels.shape!=(electrode.n,) or np.any(labels<0):raise ValueError('Integer partition over all active cells required')
    if not np.array_equal(np.unique(labels),np.arange(labels.max()+1)):raise ValueError('Contiguous nonempty regions required')
    from scipy.sparse.csgraph import connected_components
    for k in np.unique(labels):
        ids=np.flatnonzero(labels==k)
        if connected_components(electrode.diffusion.stiffness[ids][:,ids],directed=False)[0]!=1:raise ValueError('Every region must be face connected')
    return labels.copy()

def build_operator(electrode,basis:BasisBuilder,operator:OperatorBuilder,target):
    result=operator.build(electrode,validate_partition(electrode,basis.build(electrode,int(target))))
    op=result.detach() if hasattr(result,'detach') else result
    if not np.allclose(op.H,op.H.T,rtol=1e-11,atol=1e-8):raise ValueError('Energy operator must be symmetric')
    diagonal=np.maximum(np.abs(op.H.diagonal()),1e-30);normalized=op.H/np.sqrt(diagonal[:,None]*diagonal[None,:])
    if np.linalg.eigvalsh(normalized).min() < -1e-8:raise ValueError('Energy operator must be positive semidefinite')
    if abs(op.weights.sum()-electrode.diffusion.weights.sum())>1e-12:raise ValueError('Operator changed lithium-bearing volume')
    return result
