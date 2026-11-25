"""Public composition API: geometry, basis, operator, topology and submodels."""
from dataclasses import dataclass,field
from pathlib import Path
from .extensions import build_operator
from .gemsim import GEMsim
from .electrochemistry.microstructure_operator import MicrostructureElectrode
from .cohesive import build_cohesive
from .electrochemistry.aging_cache import save_aging_operator,load_aging_kernel
from .electrochemistry.aging_dfn import build_aging_dfn
from .electrochemistry.microstructure_dfn import MicrostructureStack

@dataclass
class ElectrodeRecipe:
    geometry:object
    basis:object
    operator:object
    regions:int
    topology:object=None
    electrode_factory:object=MicrostructureElectrode
    cohesive_factory:object=build_cohesive
    mechanical_parameters:dict=field(default_factory=dict)
    cohesive_parameters:dict=field(default_factory=dict)

    def prepare(self,workspace,parameters,domain,name):
        if domain not in {'negative','positive'}:raise ValueError('Electrode domain required')
        data=self.geometry.build(workspace)
        full=self.electrode_factory(data,GEMsim(workspace),parameters,domain,topology=self.topology,**self.mechanical_parameters)
        op=build_operator(full,self.basis,self.operator,self.regions)
        cohesive=self.cohesive_factory(full,op.prolongation,**self.cohesive_parameters)
        return save_aging_operator(workspace.path('operators','workflow',name,domain),op,cohesive)

@dataclass
class CoupledRecipe:
    negative:ElectrodeRecipe
    positive:ElectrodeRecipe
    degradation:object
    submodel_factories:dict=field(default_factory=dict)
    feedback:float=1.

    def prepare(self,workspace,parameters,name):
        import gc
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):raise ValueError('Safe cache name required')
        paths={}
        for d in ['negative','positive']:
            paths[d]=getattr(self,d).prepare(workspace,parameters,d,name);gc.collect()
        return paths

    def model(self,paths,parameters):
        kernels={d:load_aging_kernel(paths[d]) for d in ['negative','positive']}
        stack=MicrostructureStack(kernels['negative'].operator,kernels['positive'].operator,parameters)
        stack.update_parameters(parameters)
        return build_aging_dfn(stack,kernels,self.degradation,feedback=self.feedback,factories=self.submodel_factories),stack,kernels
