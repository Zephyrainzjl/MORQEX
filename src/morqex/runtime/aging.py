"""Warm degradation stepping and numeric restart of all differential reservoirs."""
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import pybamm as pb
from morqex.electrochemistry.microstructure_runtime import MicrostructureDFNRuntime,_parameter_digest
from morqex.electrochemistry.microstructure_dfn import MicrostructureStack,key
from morqex.electrochemistry.aging_cache import load_aging_kernel,AgingLocalization
from morqex.electrochemistry.aging_dfn import build_aging_dfn,age
from morqex.solvers.pybamm_backend import solver,provenance
from morqex.electrochemistry.microstructure_validation import macro_weights

class AgingRuntime(MicrostructureDFNRuntime):
    def __init__(self,operator_paths,parameters,config,*,nx=3,feedback=1.,configuration_manifest=None):
        self.paths={d:Path(operator_paths[d]) for d in ['negative','positive']};kernels={d:load_aging_kernel(p) for d,p in self.paths.items()}
        self.parameters=parameters.copy();self.stack=MicrostructureStack(kernels['negative'].operator,kernels['positive'].operator,self.parameters)
        self.stack.update_parameters(self.parameters);self.parameters.update({'Current function [A]':'[input]'})
        self.nx=nx;self.feedback=feedback;self.constraint='pressure';self.config=config;self.model=build_aging_dfn(self.stack,kernels,config,feedback=feedback)
        self.simulation=pb.Simulation(self.model,parameter_values=self.parameters,solver=solver(),var_pts={**self.model.default_var_pts,'x_n':nx,'x_s':nx,'x_p':nx,'r_n':2,'r_p':2})
        self._last=None;self._current=None;self._localizers={};self.last_step_seconds=None
        if configuration_manifest is None:
            try:
                laws=[*config.negative,*config.positive,config.damage]
                configuration_manifest={'values':asdict(config),'law_types':[
                    {'type':type(law).__module__+'.'+type(law).__qualname__,'source_sha256':hashlib.sha256(inspect.getsource(type(law)).encode()).hexdigest()} for law in laws]}
            except TypeError:raise ValueError('Custom laws need an explicit serializable configuration manifest')
        package=Path(__file__).parents[1]
        files=[Path(__file__),*package.joinpath('electrochemistry').glob('*.py'),*package.glob('*.py')]
        payload={'parameters':_parameter_digest(self.parameters),'config':configuration_manifest,'nx':nx,'feedback':feedback,
            'backend':provenance(),
            'source_sha256':{str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'kernels':{d:hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest() for d,path in self.paths.items()}}
        self.fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    def advance(self,dt,current_A):
        previous=self._last;previous_current=self._current
        try:
            result=super().advance(dt,current_A);s=self._last
            inventory=np.asarray(s['Total lithium [mol]'].entries).copy()
            A=float(self.parameters['Electrode width [m]'])*float(self.parameters['Electrode height [m]'])*float(self.parameters['Number of electrodes connected in parallel to make a cell'])
            residual=0.
            for d,op in self.stack.electrodes.items():
                for law in getattr(self.config,d):
                    for r in law.reservoirs():
                        stocks=[s[age(d,r.name+' [mol.m-3]',k)].entries for k in range(op.dimension)]
                        if any(np.min(stock)<-1e-9 for stock in stocks):raise RuntimeError('Negative degradation reservoir rejected')
                        stock=sum(stocks)
                        wx=macro_weights(s[age(d,r.name+' [mol.m-3]',0)]);inventory+=A*self.stack.lengths[d]*(wx@stock)
                for k in range(op.dimension):residual=max(residual,float(np.max(abs(s[age(d,'reservoir balance residual [mol.m-3.s-1]',k)].entries))))
            drift=float(np.ptp(inventory)/max(abs(inventory[0]),1e-30))
            if drift>1e-9 or residual>1e-9:raise RuntimeError('Custom degradation law/step violated the lithium reservoir contract')
            result['validation']={'relative_total_lithium_drift':drift,'reservoir_rate_residual_mol_m3_s':residual}
            return result
        except Exception:
            self._last=previous;self._current=previous_current;self.simulation._solution=previous
            raise
    def _localizer(self,domain):
        if domain not in self._localizers:self._localizers[domain]=AgingLocalization(self.paths[domain])
        return self._localizers[domain]
    def recover(self,domain,*,x_index=0,selection=None):
        if self._last is None:raise ValueError('Advance before recovery')
        started=time.perf_counter();op=self.stack.electrodes[domain];s=self._last
        c=np.array([s[key(domain,k,'concentration [mol.m-3]')].entries[x_index,-1] for k in range(op.dimension)])
        E=np.array(s[f'MORQEX {domain} axial strain'].entries[x_index,-1]);theta=np.array(s['Volume-averaged cell temperature [K]'].entries[-1]-op.metadata()['reference_temperature_K'])
        damage=np.array(s[age(domain,'damage')].entries[x_index,-1]);fields=self._localizer(domain).recover(c,E,theta,damage)
        if selection is not None:
            for name in ['stress','strain']:fields[name]=fields[name][selection]
        fields['damage']=float(damage);fields['wall_seconds']=time.perf_counter()-started;return fields
    def reaction_bound(self,*args,**kwargs):raise NotImplementedError('Fixed-H reaction certificate does not cover evolving cohesive damage')
    def estimate_defect(self,*args,**kwargs):raise NotImplementedError('Evolving-contact trajectory estimator requires a separate qualified plugin')
    def enrich(self,*args,**kwargs):raise NotImplementedError('Degradation enrichment must transfer side reservoirs and cohesive history; legacy concentration-only transfer is refused')
