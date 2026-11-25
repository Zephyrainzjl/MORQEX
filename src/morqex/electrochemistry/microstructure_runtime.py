"""Stateful online DFN/microstructure stepping, localization and nested transfer."""
import hashlib
import json
import inspect
from functools import partial
from pathlib import Path
import time
import numpy as np
import pybamm as pb
from .microstructure_cache import OnlineMicrostructureOperator,LocalizationMap
from .microstructure_dfn import MicrostructureStack,build_microstructure_dfn,key
from ..solvers.pybamm_backend import solver


def _state_signature(model):
    return [(var.name,sl[0].start,sl[0].stop) for var,sl in model.y_slices.items() if isinstance(var,pb.Variable)]


def _parameter_digest(parameters):
    def encode(value):
        if isinstance(value,pb.Symbol):return {'symbol_type':type(value).__name__,'expression':str(value),'children':[encode(child) for child in value.children]}
        if isinstance(value,np.ndarray):return {'array_sha256':hashlib.sha256(value.tobytes()).hexdigest(),'shape':value.shape,'dtype':str(value.dtype)}
        if isinstance(value,partial):return {'partial':encode(value.func),'keywords':encode(value.keywords),'args':encode(value.args)}
        if callable(value):return {'function':value.__module__+'.'+value.__qualname__,'source':inspect.getsource(value)}
        if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [encode(v) for v in value]
        if isinstance(value,np.generic):return value.item()
        return value
    return hashlib.sha256(json.dumps({k:encode(v) for k,v in parameters.items()},sort_keys=True).encode()).hexdigest()


class MicrostructureDFNRuntime:
    """Warm online stepping retains compact operators, never a fine FEM solver.

    Call recover() only when a micro field/ROI is requested. Checkpoints contain
    numeric state plus hashes, not executable pickle objects. Enrichment uses
    nested partitions and preserves represented c, lithium and elastic energy.
    """
    def __init__(self,operator_paths,parameters,*,nx=4,feedback=1.,constraint='pressure'):
        self.paths={d:Path(operator_paths[d]) for d in ['negative','positive']}
        operators={d:OnlineMicrostructureOperator.load(path) for d,path in self.paths.items()}
        self.parameters=parameters.copy();self.stack=MicrostructureStack(operators['negative'],operators['positive'],self.parameters,constraint=constraint)
        self.stack.update_parameters(self.parameters);self.parameters.update({'Current function [A]':'[input]'})
        self.nx=int(nx);self.feedback=feedback;self.constraint=constraint
        self.model=build_microstructure_dfn(self.stack,feedback=feedback)
        self.simulation=pb.Simulation(self.model,parameter_values=self.parameters,solver=solver(),
            var_pts={**self.model.default_var_pts,'x_n':nx,'x_s':nx,'x_p':nx,'r_n':2,'r_p':2})
        self._last=None;self._current=None;self._localizers={};self.last_step_seconds=None
        payload={d:op.descriptor['fingerprint'] for d,op in operators.items()}|{'nx':nx,'feedback':feedback,'constraint':constraint,
            'parameters':_parameter_digest(self.parameters),
            'runtime_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'physics_sha256':hashlib.sha256(Path(__file__).with_name('microstructure_dfn.py').read_bytes()).hexdigest()}
        self.fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

    @property
    def solution(self):return self._last

    def advance(self,dt,current_A):
        if not np.isfinite([dt,current_A]).all() or dt<=0:raise ValueError('Finite current and positive dt required')
        before=0. if self._last is None else float(self._last.t[-1]);started=time.perf_counter()
        self._last=self.simulation.step(dt,starting_solution=self._last,inputs={'Current function [A]':float(current_A)},save=False)
        self._current=float(current_A)
        end=float(self._last.t[-1])
        voltage=float(self._last['Voltage [V]'].entries[-1]);temperature=float(self._last['Volume-averaged cell temperature [K]'].entries[-1])
        self.last_step_seconds=time.perf_counter()-started
        return {'time_s':end,'requested_end_s':before+dt,'completed':bool(abs(end-before-dt)<1e-6),
            'termination':self._last.termination,'voltage_V':voltage,'temperature_K':temperature,'wall_seconds':self.last_step_seconds}

    def _localizer(self,domain):
        if domain not in self._localizers:self._localizers[domain]=LocalizationMap(self.paths[domain])
        return self._localizers[domain]

    def recover(self,domain,*,x_index=0,selection=None):
        if self._last is None:raise ValueError('Advance the coupled state first')
        started=time.perf_counter()
        op=self.stack.electrodes[domain]
        concentration=np.array([self._last[key(domain,k,'concentration [mol.m-3]')].entries[x_index,-1] for k in range(op.dimension)])
        strain=self._last[f'MORQEX {domain} axial strain'].entries[x_index,-1]
        temperature=self._last['Volume-averaged cell temperature [K]'].entries[-1]
        field=self._localizer(domain).recover(concentration,strain,temperature-op.descriptor['reference_temperature_K'],selection=selection)
        field['wall_seconds']=time.perf_counter()-started;return field

    def checkpoint(self,path):
        if self._last is None:raise ValueError('No state to save')
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        state=np.asarray(self._last.last_state.y).ravel()
        np.savez_compressed(path/'state.npz',state=state,time=float(self._last.t[-1]),current=self._current)
        meta={'schema':'morqex-microstructure-dfn-state-1','fingerprint':self.fingerprint,
            'signature':_state_signature(self.simulation.built_model),'sha256':hashlib.sha256((path/'state.npz').read_bytes()).hexdigest()}
        (path/'manifest.json').write_text(json.dumps(meta,indent=2));return path

    def estimate_defect(self,domain,*,x_index=0,lookahead_s=10.):
        """Fine FV residual at the represented state, without a new FEM solve.

        This is an empirical local defect indicator, NOT a certified nonlinear
        trajectory/pointwise-stress error bound. Fine work is done only on demand.
        """
        if self._last is None:raise ValueError('Advance before estimating the defect')
        if 'phase_catalog' in self.stack.electrodes[domain].metadata()['material']:
            raise ValueError('Legacy isotropic fine-defect estimator cannot certify phase-catalog anisotropic chemistry')
        started=time.perf_counter();op=self.stack.electrodes[domain];local=self._localizer(domain);a=local.arrays
        field=self.recover(domain,x_index=x_index);c=field['concentration'];stress=field['stress'];active=a['element_active_cell']>=0
        mu_el=np.bincount(a['element_active_cell'][active],weights=-op.descriptor['material']['molar_volume_m3_mol']/3*
            np.sum(stress[active,:3],axis=1)*a['volumes'][active],minlength=len(c))/a['active_volumes']
        T=float(self._last['Volume-averaged cell temperature [K]'].entries[-1]);D=domain.capitalize()
        ce=float(self._last[f'{D} electrolyte concentration [mol.m-3]'].entries[x_index,-1])
        delta=float(self._last[f'{D} electrode surface potential difference [V]'].entries[x_index,-1]);phase=self.model.param.domain_params[domain].prim
        F=float(pb.constants.F.value);R=float(pb.constants.R.value);cmax=float(self.parameters[f'Maximum concentration in {domain} electrode [mol.m-3]'])
        def evaluate(symbol):return np.asarray(self.parameters.process_symbol(symbol).evaluate()).ravel()
        U=evaluate(phase.U(pb.Array(c/cmax),pb.Scalar(T)));mu=-F*U+self.feedback*mu_el
        j0=evaluate(phase.j0(pb.Scalar(ce),pb.Array(c),pb.Scalar(T)))
        current=2*j0*np.sinh(F*(delta-U+self.feedback*mu_el/F)/(2*R*T))
        massdot=-a['reaction_area_full']*current/F
        i,j=a['edge_i'],a['edge_j'];mid=(c[i]+c[j])/2
        variable=pb.InputParameter('MORQEX defect c',expected_size=len(mid))
        derivative=self.parameters.process_symbol(-F*phase.U(variable/phase.c_max,pb.Scalar(T)).diff(variable))
        curvature=np.asarray(derivative.evaluate(inputs={'MORQEX defect c':mid})).ravel()
        dc=c[i]-c[j];regularizer=(1e-7*cmax)**2
        secant=((-F*U[i]+F*U[j])*dc+regularizer*curvature)/(dc*dc+regularizer)
        diffusivity=evaluate(phase.D(pb.Array(mid),pb.Scalar(T)))
        flux=a['edge_conductance']*diffusivity/secant*(mu[i]-mu[j])
        np.add.at(massdot,i,-flux);np.add.at(massdot,j,flux)
        coarse=np.array([self._last[key(domain,k,'rate [mol.m-3.s-1]')].entries[x_index,-1] for k in range(op.dimension)])
        defect=a['P']@coarse-massdot/a['weights_full']
        coarse_j=np.array([self._last[key(domain,k,'reaction current [A.m-2]')].entries[x_index,-1] for k in range(op.dimension)])
        port_error=float(a['reaction_area_full']@current-op.reaction_area@coarse_j)
        return {'relative_concentration_defect':float(lookahead_s*np.sqrt(np.dot(a['weights_full'],defect*defect)/op.active_fraction)/cmax),
            'absolute_reaction_port_defect_A_m3':abs(port_error),
            'relative_reaction_port_defect':abs(port_error)/max(1.,abs(op.reaction_area@coarse_j)),
            'wall_seconds':time.perf_counter()-started,'kind':'empirical local residual, not a rigorous trajectory bound'}

    def reaction_bound(self,domain,*,x_index=0):
        if self._last is None:raise ValueError('Advance before querying the reaction bound')
        started=time.perf_counter();op=self.stack.electrodes[domain]
        c=np.array([self._last[key(domain,k,'concentration [mol.m-3]')].entries[x_index,-1] for k in range(op.dimension)])
        j0=np.array([self._last[key(domain,k,'exchange current [A.m-2]')].entries[x_index,-1] for k in range(op.dimension)])
        eta=np.array([self._last[key(domain,k,'reaction overpotential [V]')].entries[x_index,-1] for k in range(op.dimension)])
        T=float(self._last['Volume-averaged cell temperature [K]'].entries[-1]);eps=float(self._last[f'MORQEX {domain} axial strain'].entries[x_index,-1])
        q=np.r_[eps,c-op.reference_concentration,T-op.descriptor['reference_temperature_K']]
        bound=op.reaction_port_bound(q,j0,eta,T,feedback=self.feedback)
        A=float(self.parameters['Electrode width [m]'])*float(self.parameters['Electrode height [m]'])*float(self.parameters['Number of electrodes connected in parallel to make a cell'])
        reference=float(self.parameters['Nominal cell capacity [A.h]'])/(A*self.stack.lengths[domain])
        bound['relative_to_1C']=bound['absolute_bound_A_m3']/reference;bound['wall_seconds']=time.perf_counter()-started
        return bound

    def enrich_if_needed(self,new_paths,*,relative_port_tolerance=.01,concentration_defect_tolerance=None,lookahead_s=30.):
        if relative_port_tolerance<=0:raise ValueError('Positive reaction-port tolerance required')
        bounds={d:max(self.reaction_bound(d,x_index=i)['relative_to_1C'] for i in range(self.nx)) for d in self.stack.electrodes}
        defects={} if concentration_defect_tolerance is None else {d:max(self.estimate_defect(d,x_index=i,lookahead_s=lookahead_s)['relative_concentration_defect'] for i in range(self.nx)) for d in self.stack.electrodes}
        triggered=max(bounds.values())>relative_port_tolerance or bool(defects and max(defects.values())>concentration_defect_tolerance)
        started=time.perf_counter();updated=self.enrich(new_paths) if triggered else self
        return updated,{'triggered':triggered,'local_port_bounds_relative_to_1C':bounds,'tolerance':relative_port_tolerance,
            'empirical_concentration_defects':defects,'empirical_concentration_tolerance':concentration_defect_tolerance,
            'enrichment_seconds':time.perf_counter()-started,'scope':'local represented-state reaction closure; no trajectory certificate'}

    def restore(self,path):
        path=Path(path);meta=json.loads((path/'manifest.json').read_text())
        if meta['schema']!='morqex-microstructure-dfn-state-1' or meta['fingerprint']!=self.fingerprint:raise ValueError('Checkpoint/operator/model mismatch')
        if hashlib.sha256((path/'state.npz').read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Checkpoint checksum mismatch')
        self.simulation.build();model=self.simulation.built_model
        if json.loads(json.dumps(_state_signature(model)))!=meta['signature']:raise ValueError('Checkpoint state layout mismatch')
        with np.load(path/'state.npz',allow_pickle=False) as a:
            self._current=float(a['current']);self._last=pb.Solution(np.array([float(a['time'])]),a['state'][:,None],model,
                {'Current function [A]':self._current},check_solution=False)
        return self

    def enrich(self,new_paths):
        if self._last is None:raise ValueError('No state to enrich')
        new=type(self)(new_paths,self.parameters,nx=self.nx,feedback=self.feedback,constraint=self.constraint)
        new.simulation.build();model=new.simulation.built_model;old_model=self.simulation.built_model
        physical={};y=np.asarray(self._last.last_state.y).ravel()
        for var,sl in old_model.y_slices.items():
            if isinstance(var,pb.Variable):
                scale=np.asarray(var.scale.evaluate()).ravel();reference=np.asarray(var.reference.evaluate()).ravel()
                physical[var.name]=(reference+scale*y[sl[0]])[:,None]
        for d,op in self.stack.electrodes.items():
            old_map=self._localizer(d);new_map=new._localizer(d);P=old_map.arrays['P'];Pn=new_map.arrays['P'];Rn=new_map.arrays['R']
            if op.descriptor['geometry_fingerprint']!=new.stack.electrodes[d].descriptor['geometry_fingerprint']:raise ValueError('Enrichment geometry/material mismatch')
            if not np.allclose(Pn@Rn@P,P,rtol=0,atol=1e-12):raise ValueError('Enrichment must be nested; no lossy coarsening')
            previous=np.column_stack([physical[key(d,k,'concentration [mol.m-3]')][:,0] for k in range(op.dimension)])
            updated=previous@P.T@Rn.T
            for k in range(updated.shape[1]):physical[key(d,k,'concentration [mol.m-3]')]=updated[:,k,None]
        model.set_initial_conditions_from(physical)
        state=np.asarray(model.concatenated_initial_conditions.evaluate()).reshape(-1,1)
        new._current=self._current;new._last=pb.Solution(np.array([float(self._last.t[-1])]),state,model,
            {'Current function [A]':self._current},check_solution=False)
        return new
