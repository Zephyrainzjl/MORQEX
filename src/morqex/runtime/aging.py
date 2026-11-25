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
    def __init__(self,operator_paths,parameters,config,*,nx=3,feedback=1.,configuration_manifest=None,rtol=1e-7,atol=1e-8):
        if nx<3 or not np.isfinite([rtol,atol]).all() or min(rtol,atol)<=0:raise ValueError('At least three macro cells and positive finite solver tolerances required')
        self.paths={d:Path(operator_paths[d]) for d in ['negative','positive']};kernels={d:load_aging_kernel(p) for d,p in self.paths.items()}
        self._kernels=kernels
        self._kernel_manifest_hashes={d:hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest() for d,p in self.paths.items()}
        self.parameters=parameters.copy();self.stack=MicrostructureStack(kernels['negative'].operator,kernels['positive'].operator,self.parameters)
        self.stack.update_parameters(self.parameters);self.parameters.update({'Current function [A]':'[input]'})
        self.nx=nx;self.feedback=feedback;self.constraint='pressure';self.config=config;self.model=build_aging_dfn(self.stack,kernels,config,feedback=feedback)
        self.rtol=rtol;self.atol=atol
        self.simulation=pb.Simulation(self.model,parameter_values=self.parameters,solver=solver(rtol=rtol,atol=atol),var_pts={**self.model.default_var_pts,'x_n':nx,'x_s':nx,'x_p':nx,'r_n':2,'r_p':2})
        self._last=None;self._current=None;self._localizers={};self.last_step_seconds=None
        self._certificates={};self._reaction_certificates={}
        if configuration_manifest is None:
            try:
                laws=[*config.negative,*config.positive,config.damage]
                configuration_manifest={'values':asdict(config),'law_types':[
                    {'type':type(law).__module__+'.'+type(law).__qualname__,'source_sha256':hashlib.sha256(inspect.getsource(type(law)).encode()).hexdigest()} for law in laws]}
            except TypeError:raise ValueError('Custom laws need an explicit serializable configuration manifest')
        package=Path(__file__).parents[1]
        files=[Path(__file__),*package.joinpath('electrochemistry').glob('*.py'),*package.glob('*.py')]
        payload={'parameters':_parameter_digest(self.parameters),'config':configuration_manifest,'nx':nx,'feedback':feedback,'rtol':rtol,'atol':atol,
            'backend':provenance(),
            'source_sha256':{str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'kernels':{d:hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest() for d,path in self.paths.items()}}
        self.fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        self.configuration_manifest=configuration_manifest
    def advance(self,dt,current_A):
        previous=self._last;previous_current=self._current
        try:
            result=super().advance(dt,current_A);s=self._last
            inventory=np.asarray(s['Total lithium [mol]'].entries).copy()
            A=float(self.parameters['Electrode width [m]'])*float(self.parameters['Electrode height [m]'])*float(self.parameters['Number of electrodes connected in parallel to make a cell'])
            residual=0.
            for d,op in self.stack.electrodes.items():
                damage=np.asarray(s[age(d,'damage')].entries)
                if not np.isfinite(damage).all() or np.min(damage)<-1e-9 or np.max(damage)>self.config.damage.maximum+1e-9 or np.min(np.diff(damage,axis=-1))<-1e-9:raise RuntimeError('Contact damage irreversibility/range contract violated; tighten tolerances or reject this law')
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
        if domain not in self._localizers:
            if hashlib.sha256((self.paths[domain]/'manifest.json').read_bytes()).hexdigest()!=self._kernel_manifest_hashes[domain]:raise ValueError('Operator cache changed during runtime; use a new immutable cache path')
            self._localizers[domain]=AgingLocalization(self.paths[domain])
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
    def reaction_bound(self,domain,*,x_index=0):
        if self._last is None:raise ValueError('Advance before querying the reaction bound')
        from morqex.degradation import DiffusionLimitedSEI,PlatingStrippingDeadLi
        if any(type(law) not in {DiffusionLimitedSEI,PlatingStrippingDeadLi} for law in getattr(self.config,domain)):raise NotImplementedError('Custom side laws require an independently verified monotonicity contract')
        from morqex.error_estimation.aging_port import AgingReactionCertificate
        if domain not in self._reaction_certificates:
            path=self.paths[domain]/'reaction_certificate'
            if not (path/'manifest.json').exists():raise NotImplementedError('Compile the evolving-kernel reaction certificate offline before online queries')
            self._reaction_certificates[domain]=AgingReactionCertificate(path,self.paths[domain])
        started=time.perf_counter();s=self._last;op=self.stack.electrodes[domain]
        c=np.array([s[key(domain,k,'concentration [mol.m-3]')].entries[x_index,-1] for k in range(op.dimension)])
        T=float(s['Volume-averaged cell temperature [K]'].entries[-1]);E=float(s[f'MORQEX {domain} axial strain'].entries[x_index,-1]);dd=float(s[age(domain,'damage')].entries[x_index,-1])
        q=np.r_[E,c-op.reference_concentration,T-op.metadata()['reference_temperature_K']]
        exact=(self._kernels[domain].hessian(dd)@q)[1:-1]/op.weights
        j0=[];eta=[];mu_res=[];film_res=[];resistance=[]
        for k,A in enumerate(op.reaction_area):
            j0.append(float(s[key(domain,k,'exchange current [A.m-2]')].entries[x_index,-1]));eta.append(float(s[key(domain,k,'reaction overpotential [V]')].entries[x_index,-1]))
            mu_res.append(float(s[key(domain,k,'elastic chemical potential [J.mol-1]')].entries[x_index,-1])-exact[k])
            film_res.append(float(s[age(domain,'film current residual [A.m-2]',k)].entries[x_index,-1]))
            stocks={r.name:pb.Scalar(float(s[age(domain,r.name+' [mol.m-3]',k)].entries[x_index,-1])) for law in getattr(self.config,domain) for r in law.reservoirs()}
            value=sum((law.resistance(stocks,float(A)) for law in getattr(self.config,domain)),0.)
            resistance.append(float(value.evaluate()) if hasattr(value,'evaluate') else float(value))
        result=self._reaction_certificates[domain].bound(q,dd,j0,eta,T,feedback=self.feedback,mechanical_residual=np.asarray(mu_res),film_residual=np.asarray(film_res),film_resistance=np.asarray(resistance))
        A=float(self.parameters['Electrode width [m]'])*float(self.parameters['Electrode height [m]'])*float(self.parameters['Number of electrodes connected in parallel to make a cell'])
        result['relative_to_1C']=result['absolute_bound_A_m3']/(float(self.parameters['Nominal cell capacity [A.h]'])/(A*self.stack.lengths[domain]));result['wall_seconds']=time.perf_counter()-started
        return result
    def estimate_defect(self,*args,**kwargs):raise NotImplementedError('Evolving-contact trajectory estimator requires a separate qualified plugin')
    def checkpoint(self,path):
        path=super().checkpoint(path)
        if hasattr(self,'transfer_audit'):
            manifest=path/'manifest.json';metadata=json.loads(manifest.read_text());metadata['transfer_audit']=self.transfer_audit
            manifest.write_text(json.dumps(metadata,indent=2))
        return path
    def restore(self,path):
        super().restore(path)
        metadata=json.loads((Path(path)/'manifest.json').read_text())
        if 'transfer_audit' in metadata:self.transfer_audit=metadata['transfer_audit']
        return self
    def cohesive_error_bound(self,domain,*,x_index=0):
        """Fixed-state compression bound; never a nonlinear trajectory bound."""
        if self._last is None:raise ValueError('Advance before querying a certificate')
        from morqex.electrochemistry.aging_cache import load_cohesive_certificate
        from morqex.cohesive_reduction import cohesive_certificate
        if domain not in self._certificates:
            if hashlib.sha256((self.paths[domain]/'manifest.json').read_bytes()).hexdigest()!=self._kernel_manifest_hashes[domain]:raise ValueError('Operator cache changed during runtime')
            self._certificates[domain]=load_cohesive_certificate(self.paths[domain])
        kernel=self._kernels[domain];op=kernel.operator;s=self._last
        c=np.array([s[key(domain,k,'concentration [mol.m-3]')].entries[x_index,-1] for k in range(op.dimension)])
        q=np.r_[s[f'MORQEX {domain} axial strain'].entries[x_index,-1],c-op.reference_concentration,
            s['Volume-averaged cell temperature [K]'].entries[-1]-op.metadata()['reference_temperature_K']]
        damage=float(s[age(domain,'damage')].entries[x_index,-1])
        result=cohesive_certificate(kernel.cohesive,self._certificates[domain],q,damage)
        result['chemical_potential_absolute_upper_J_mol']=result['generalized_force_absolute_upper'][1:-1]/op.weights
        return result

    def enrich(self,new_paths,*,relative_energy_tolerance=1e-5):
        """Nested concentration and reaction-area reservoir transfer.

        Geometry, material and constitutive laws must match. Damage and all macro
        differential states are retained. Consistent algebraic states are solved
        on the next advance. Failed checks leave this runtime untouched.
        """
        if self._last is None:raise ValueError('No state to enrich')
        if not np.isfinite(relative_energy_tolerance) or relative_energy_tolerance<0:raise ValueError('Nonnegative energy transfer budget required')
        started=time.perf_counter()
        new=type(self)(new_paths,self.parameters,self.config,nx=self.nx,feedback=self.feedback,configuration_manifest=self.configuration_manifest,rtol=self.rtol,atol=self.atol)
        new.simulation.build();model=new.simulation.built_model;old_model=self.simulation.built_model
        physical={};y=np.asarray(self._last.last_state.y).ravel()
        for var,sl in old_model.y_slices.items():
            if isinstance(var,pb.Variable):physical[var.name]=(np.asarray(var.reference.evaluate())+np.asarray(var.scale.evaluate())*y[sl[0]])[:,None]
        audit={}
        theta=float(self._last['Volume-averaged cell temperature [K]'].entries[-1]-self.parameters['Reference temperature [K]'])
        for d,op in self.stack.electrodes.items():
            previous=self._localizer(d);following=new._localizer(d);next_op=new.stack.electrodes[d]
            if op.metadata()['geometry_fingerprint']!=next_op.metadata()['geometry_fingerprint']:raise ValueError('Enrichment geometry/material mismatch')
            old_meta=json.loads((self.paths[d]/'manifest.json').read_text());new_meta=json.loads((new.paths[d]/'manifest.json').read_text())
            for name in ['normal_Pa_m','tangent_Pa_m']:
                if old_meta['cohesive'][name]!=new_meta['cohesive'][name]:raise ValueError('Enrichment changes interface law')
            if previous.kernel.cohesive.residual!=following.kernel.cohesive.residual:raise ValueError('Enrichment changes residual stiffness')
            P=previous.arrays['P'];Pn=following.arrays['P'];Rn=following.arrays['R']
            if not np.array_equal(previous.arrays['voxel_ids'],following.arrays['voxel_ids']) or not np.allclose(Pn@Rn@P,P,rtol=0,atol=1e-12):raise ValueError('Only nested same-geometry enrichment is allowed')
            oldchem=getattr(op,'regional_chemistry',None);nextchem=getattr(next_op,'regional_chemistry',None)
            if (oldchem is None)!=(nextchem is None):raise ValueError('Enrichment changes dopant chemistry')
            if oldchem is not None and (vars(oldchem.law)!=vars(nextchem.law) or not np.allclose(P@oldchem.fraction,Pn@nextchem.fraction,atol=1e-12,rtol=0)):raise ValueError('Enrichment changes composition field or law')
            c=np.column_stack([physical[key(d,k,'concentration [mol.m-3]')][:,0] for k in range(op.dimension)]);cn=c@P.T@Rn.T
            for k in range(next_op.dimension):physical[key(d,k,'concentration [mol.m-3]')]=cn[:,k,None]
            area=previous.arrays['area_full'];transfer=(Pn.T@(area[:,None]*P))/np.maximum(op.reaction_area[None,:],1e-30)
            mass_error=float(np.max(abs(c@op.weights-cn@next_op.weights)))
            stock_error=0.
            for law in getattr(self.config,d):
                for r in law.reservoirs():
                    stock=np.column_stack([physical[age(d,r.name+' [mol.m-3]',k)][:,0] for k in range(op.dimension)])
                    updated=stock@transfer.T;stock_error=max(stock_error,float(np.max(abs(stock.sum(axis=1)-updated.sum(axis=1)))))
                    for k in range(next_op.dimension):physical[age(d,r.name+' [mol.m-3]',k)]=updated[:,k,None]
            strain=physical[f'MORQEX {d} axial strain'][:,0];damage=physical[age(d,'damage')][:,0]
            energy_error=0.
            for x,(E,dd) in enumerate(zip(strain,damage)):
                q=np.r_[E,c[x]-op.reference_concentration,theta];qn=np.r_[E,cn[x]-next_op.reference_concentration,theta]
                before=.5*q@previous.kernel.hessian(dd)@q;after=.5*qn@following.kernel.hessian(dd)@qn
                energy_error=max(energy_error,float(abs(after-before)/max(1.,abs(before))))
            if mass_error>1e-8 or stock_error>1e-8 or energy_error>relative_energy_tolerance:raise ValueError('Enrichment mass/energy budget failed')
            audit[d]={'host_mass_difference_mol_m3_RVE':mass_error,'reservoir_difference_mol_m3_RVE':stock_error,'relative_elastic_energy_jump':energy_error}
        # Defaults are guesses for new algebraic variables only.
        for var in model.initial_conditions:
            if isinstance(var,pb.Concatenation):continue
            if var.name not in physical:
                if var in model.rhs:raise ValueError('Unmapped differential state in enrichment: '+var.name)
                sl=model.y_slices[var][0];physical[var.name]=np.zeros((sl.stop-sl.start,1))
        model.set_initial_conditions_from(physical);state=np.asarray(model.concatenated_initial_conditions.evaluate()).reshape(-1,1)
        new._current=self._current;new._last=pb.Solution(np.array([float(self._last.t[-1])]),state,model,{'Current function [A]':self._current},check_solution=False)
        new.transfer_audit={'electrodes':audit,'seconds':time.perf_counter()-started,'scope':'nested same geometry/composition; algebraic reinitialization required on next advance'}
        return new
