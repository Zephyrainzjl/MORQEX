"""Small online energy/port operators built from sparse Hessian applications.

The analytic local clamped energy bounds the condensed elastic energy from
above. This gives a conservative whole-error certificate without dense H or a
full spectral error space. No sharpness, continuum-error or novelty claim.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from .concentration import ConcentrationBasis,component_constants,_orthogonal
from .gradient_transport import immutable,EnergyAccuracyError
from .transport import ConcentrationInfeasible
from .energy import _digest


@dataclass(frozen=True)
class StreamedState:
    coefficients: np.ndarray
    macro_strain: np.ndarray
    time: float
    energy: float
    inventory: np.ndarray
    error_bound: float
    fingerprint: str

    def __post_init__(self):
        for name in ('coefficients','macro_strain','inventory'):
            object.__setattr__(self,name,immutable(getattr(self,name)))


class OnlineEnergyKernel:
    ARRAYS=('H','mobility','source','G','macro_H','inventory','inventory_source','baseline_inventory','suprema','residual_root')
    SCALARS=('reference','maximum','margin','kappa','pointwise_constant','roundoff','fingerprint','system_fingerprint','projection')

    @property
    def dimension(self): return len(self.H)

    def energy(self,a,e): return float(.5*a @ self.H @ a+a @ (self.G @ e)+.5*e @ self.macro_H @ e)

    def accept(self,state,tolerance):
        if state.fingerprint != self.fingerprint or state.coefficients.shape != (self.dimension,) or state.macro_strain.shape != (6,): raise ValueError('Wrong ROM state')
        if not np.isfinite(np.r_[state.coefficients,state.macro_strain,state.error_bound,state.time,tolerance]).all() or state.error_bound < 0 or state.time < 0 or tolerance <= 0: raise ValueError('Invalid certified state/tolerance')
        if state.error_bound > tolerance: raise EnergyAccuracyError(state.error_bound,tolerance)
        if self.suprema @ np.abs(state.coefficients)+self.pointwise_constant*state.error_bound > self.margin:
            raise ConcentrationInfeasible('Certified physical concentration enclosure failed')

    def trial(self,previous,dt,u,*,macro_strain=None):
        if previous.fingerprint != self.fingerprint: raise ValueError('Wrong ROM state')
        e=previous.macro_strain if macro_strain is None else np.asarray(macro_strain,dtype=float)
        u=np.asarray(u,dtype=float)
        if e.shape != (6,) or u.shape != (self.source.shape[1],) or not np.isfinite(np.r_[e,u,dt]).all() or dt <= 0: raise ValueError('Invalid ROM input')
        if self._dt != dt:
            self._factor=lu_factor(np.eye(self.dimension)+dt*self.mobility @ self.H);self._dt=dt
        old=previous.coefficients
        a=lu_solve(self._factor,old+dt*(self.source @ u-self.mobility @ (self.G @ e)))
        coordinates=np.r_[a-old,dt*a,dt*u,dt*e]
        residual=float(np.linalg.norm(self.residual_root @ coordinates))
        padding=256*np.finfo(float).eps*self.roundoff*np.linalg.norm(coordinates)
        bound=previous.error_bound+residual+padding
        energy=self.energy(a,e);inventory=self.baseline_inventory+self.inventory @ a
        gradient=self.H @ a+self.G @ e
        dissipation=float(gradient @ self.mobility @ gradient);power=float(gradient @ self.source @ u)
        work=self.energy(old,e)-previous.energy
        numerical=float(.5*(a-old) @ self.H @ (a-old))
        state=StreamedState(a,e,previous.time+dt,energy,inventory,float(bound),self.fingerprint)
        return state,{'energy_balance':float(energy-previous.energy-work-dt*(power-dissipation)+numerical),
            'component_mass_error':inventory-previous.inventory-dt*self.inventory_source @ u,
            'diffusion_dissipation':dissipation,'external_chemical_power':power,'mechanical_work':float(work),
            'numerical_dissipation':numerical,'residual_upper_norm':residual}

    def step(self,previous,dt,u,*,error_tolerance,macro_strain=None):
        state,diagnostic=self.trial(previous,dt,u,macro_strain=macro_strain)
        self.accept(state,error_tolerance)
        return state,diagnostic

    def online_array_bytes(self):
        names=('H','mobility','source','G','macro_H','inventory','inventory_source','baseline_inventory','suprema','residual_root')
        return sum(getattr(self,k).nbytes for k in names)

    def macro_stress(self,state): return self.macro_H @ state.macro_strain+self.G.T @ state.coefficients

    def batch_macro(self,coefficients,strains):
        a,e=np.asarray(coefficients),np.asarray(strains)
        if a.ndim != 2 or a.shape[1] != self.dimension or e.shape != (len(a),6) or not np.isfinite(a).all() or not np.isfinite(e).all(): raise ValueError('Invalid batch')
        return a @ self.G+e @ self.macro_H.T
    def save(self,workspace,name,state=None):
        from pathlib import Path
        import hashlib,json
        path=workspace.path('operators',name);path.mkdir(parents=True,exist_ok=True)
        arrays={k:getattr(self,k) for k in self.ARRAYS}
        metadata={k:getattr(self,k) for k in self.SCALARS}
        if state is not None:
            self.accept(state,max(1.,state.error_bound*1.01))
            arrays.update(state_a=state.coefficients,state_e=state.macro_strain,state_inventory=state.inventory)
            metadata.update(state_time=state.time,state_energy=state.energy,state_error=state.error_bound)
        destination=path/'kernel.npz';np.savez_compressed(destination,**arrays)
        manifest={'schema':'morqex-online-energy-kernel-1','parameters':metadata,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'compact online kernel only; no bulk/localization or offline basis'}
        (path/'manifest.json').write_text(json.dumps(manifest,indent=2))
        return path

    @classmethod
    def load(cls,path):
        from pathlib import Path
        import hashlib,json
        path=Path(path);manifest=json.loads((path/'manifest.json').read_text());metadata=manifest['parameters']
        if manifest['schema'] != 'morqex-online-energy-kernel-1' or manifest['implementation_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest(): raise ValueError('Kernel implementation/schema mismatch')
        if hashlib.sha256((path/'kernel.npz').read_bytes()).hexdigest() != manifest['sha256']: raise ValueError('Kernel checksum mismatch')
        kernel=cls();kernel._dt,kernel._factor=None,None
        for k in cls.SCALARS:setattr(kernel,k,metadata[k])
        with np.load(path/'kernel.npz',allow_pickle=False) as a:
            for k in cls.ARRAYS:setattr(kernel,k,immutable(a[k]))
            state=None
            if 'state_time' in metadata:
                state=StreamedState(a['state_a'],a['state_e'],metadata['state_time'],metadata['state_energy'],a['state_inventory'],metadata['state_error'],kernel.fingerprint)
                kernel.accept(state,max(1.,state.error_bound*1.01))
                if not np.isclose(state.energy,kernel.energy(state.coefficients,state.macro_strain),rtol=1e-12,atol=1e-12) or not np.allclose(state.inventory,kernel.baseline_inventory+kernel.inventory @ state.coefficients,rtol=1e-12,atol=0): raise ValueError('Kernel state energy/inventory inconsistent')
        return kernel,state


class StreamedEnergyROM(OnlineEnergyKernel):
    def __init__(self,system,basis,*,projection='energy',add_mass_riesz=True,add_macro_lifting=True):
        if projection not in {'energy','mass'}: raise ValueError('Unknown projection')
        if basis.values.shape[0] != system.n or not np.allclose(basis.weights,system.weights,rtol=1e-12,atol=0): raise ValueError('Different concentration domain')
        if not np.isfinite(basis.values).all() or not np.allclose(basis.mass,np.eye(basis.values.shape[1]),atol=1e-9,rtol=1e-9): raise ValueError('Mass-orthonormal finite initial basis required')
        constants=component_constants(system.diffusion)
        values=_orthogonal(constants,system.weights,initial=basis.values)
        # Mass covectors in z=sqrt(kappa M)c coordinates.
        covectors=np.sqrt(system.weights)[:,None]*constants/np.sqrt(system.kappa)
        before=system.mechanical_solves
        if projection == 'energy' and add_mass_riesz:
            directions=np.column_stack([system.solve_h(covectors[:,j])/system.scale_c for j in range(covectors.shape[1])])
            values=_orthogonal(directions,system.weights,initial=values)
        self.riesz_mechanical_queries=system.mechanical_solves-before
        before=system.mechanical_solves
        if add_macro_lifting:
            directions=np.column_stack([system.solve_h(system.G[:,j])/system.scale_c for j in range(6)])
            values=_orthogonal(directions,system.weights,initial=values)
        self.lifting_mechanical_queries=system.mechanical_solves-before
        self.basis=ConcentrationBasis(immutable(values),immutable(system.weights),basis.method+'+streamed',system.diffusion.n_components)
        vz=system.scale_c[:,None]*values
        # At most one response column is resident in the Hessian-action solve.
        hz=np.column_stack([system.h_action(column) for column in vz.T])
        self.H=immutable(.5*(vz.T @ hz+hz.T @ vz))
        mass=self.basis.mass
        if projection == 'energy': w=np.linalg.solve(self.H,hz.T).T
        else: w=np.linalg.solve(mass,(np.sqrt(system.weights/system.kappa)[:,None]*values).T).T
        error=np.linalg.norm(w @ (vz.T @ covectors)-covectors)/max(np.linalg.norm(covectors),1e-30)
        if error > 1e-9: raise ValueError('Mass Riesz/constant directions missing')
        self.mass_projector_error=float(error)
        self.mobility=immutable(.5*(w.T @ (system.L @ w)+(w.T @ (system.L @ w)).T))
        self.source=immutable(w.T @ system.b)
        self.G=immutable(vz.T @ system.G)
        self.macro_H=immutable(system.macro_H)
        self.inventory=immutable(system.inventory_rows @ values)
        self.inventory_source=immutable((system.inventory_rows/system.scale_c) @ system.b)
        self.baseline_inventory=immutable(system.baseline_inventory)
        self.suprema=immutable(np.max(np.abs(values),axis=0))
        self.reference,self.maximum=system.reference,system.maximum
        self.margin=min(system.reference,system.maximum-system.reference)
        self.kappa=system.kappa
        self.pointwise_constant=float(1/np.min(system.scale_c)) # Hz >= I
        columns=np.column_stack([vz,system.L @ hz,-system.b,system.L @ system.G])
        weighted=np.sqrt(system.local_upper)[:,None]*columns
        self.residual_root=immutable(np.linalg.qr(weighted,mode='r'))
        self.roundoff=float(np.linalg.norm(weighted,ord='fro'))
        self.fingerprint=_digest(values,self.H,self.G,np.frombuffer((system.fingerprint+projection).encode(),dtype=np.uint8))
        self.system_fingerprint=system.fingerprint
        self.projection=projection
        self._dt,self._factor=None,None
        # No system/solver/full-N operators are retained by the online kernel.
        self.offline_normalized_basis=immutable(vz)
        self.offline_hessian_columns=immutable(hz)

    def initialize(self,initial,system,*,macro_strain=None,error_tolerance):
        if system.fingerprint != self.system_fingerprint: raise ValueError('Wrong full system')
        c=np.asarray(initial,dtype=float);e=np.zeros(6) if macro_strain is None else np.asarray(macro_strain,dtype=float)
        if c.shape != (system.n,) or e.shape != (6,) or not np.isfinite(np.r_[c,e]).all(): raise ValueError('Invalid initial field')
        right=self.offline_hessian_columns.T @ (system.scale_c*c)
        a=np.linalg.solve(self.H,right)
        difference=system.scale_c*(c-self.basis.values @ a)
        bound=float(np.sqrt(np.dot(system.local_upper*difference,difference)))
        state=StreamedState(a,e,0.,self.energy(a,e),self.baseline_inventory+self.inventory @ a,bound,self.fingerprint)
        self.accept(state,error_tolerance)
        return state

    def detach_online(self):
        kernel=OnlineEnergyKernel()
        for name in OnlineEnergyKernel.ARRAYS:setattr(kernel,name,immutable(getattr(self,name)))
        for name in OnlineEnergyKernel.SCALARS:setattr(kernel,name,getattr(self,name))
        kernel._dt,kernel._factor=None,None
        return kernel


class StressLocalization:
    """Optional stress-only output columns; construction is streamed by mode.

    Compact ROM contains no localization columns. This object has O(Nrequested*m)
    storage and recovery cost, and cannot masquerade as an N-independent output.
    """
    def __init__(self,system,rom,selection,*,maximum_bytes=128*2**20):
        if system.fingerprint != rom.system_fingerprint: raise ValueError('Wrong system')
        selected=np.asarray(selection,dtype=int)
        n=len(system.solver.mesh.cells)
        if selected.ndim != 1 or not len(selected) or np.any(selected < 0) or np.any(selected >= n) or len(np.unique(selected)) != len(selected): raise ValueError('Unique nonempty element selection required')
        required=len(selected)*6*(rom.dimension+6)*8
        if required > maximum_bytes: raise MemoryError('Requested localization exceeds declared byte budget')
        self.selection=immutable(selected).astype(int);self.fingerprint=rom.fingerprint
        self.operator=np.empty((len(selected),6,rom.dimension+6))
        for j in range(6):
            e=np.zeros(6);e[j]=1.
            self.operator[:,:,j]=system.fields(np.zeros(system.n),e).stress[selected]
        for j,column in enumerate(rom.basis.values.T):
            self.operator[:,:,j+6]=system.fields(column,np.zeros(6)).stress[selected]
        self.volumes=system.solver.mesh.volumes[selected].copy()
        self.phases=system.solver.mesh.phases[selected].copy()
        self.rve_volume=system.solver.mesh.rve_volume
        self.mean_operator=np.einsum('e,eid->id',self.volumes,self.operator)/self.volumes.sum()
        self.component_second_moments=np.einsum('e,eid,eif->idf',self.volumes,self.operator,self.operator)/self.volumes.sum()
        self.von_mises_second_moment=1.5*(np.sum(self.component_second_moments,axis=0)-np.einsum('e,ed,ef->df',self.volumes,np.sum(self.operator[:,:3],axis=1),np.sum(self.operator[:,:3],axis=1))/self.volumes.sum()/3)
        self.operator.flags.writeable=False

    def stress(self,state):
        if state.fingerprint != self.fingerprint: raise ValueError('Wrong localization state')
        return np.einsum('eid,d->ei',self.operator,np.r_[state.macro_strain,state.coefficients])

    def moments(self,state):
        if state.fingerprint != self.fingerprint: raise ValueError('Wrong localization state')
        q=np.r_[state.macro_strain,state.coefficients]
        mean=self.mean_operator @ q
        variance=np.einsum('d,idf,f->i',q,self.component_second_moments,q)-mean**2
        return mean,np.maximum(variance,0.),float(q @ self.von_mises_second_moment @ q)

    def detach_statistics(self):
        return QuadraticStressStatistics(self.mean_operator,self.component_second_moments,self.von_mises_second_moment,self.fingerprint)


class QuadraticStressStatistics:
    """Exact selected-volume linear means and quadratic moments, O(m²) storage.

    No percentile/maximum stress is represented as a quadratic form.
    """
    def __init__(self,mean,second,von_mises_second,fingerprint):
        self.mean=immutable(mean);self.second=immutable(second)
        self.von_mises_second=immutable(von_mises_second);self.fingerprint=fingerprint

    def moments(self,state):
        if state.fingerprint != self.fingerprint: raise ValueError('Wrong statistics state')
        q=np.r_[state.macro_strain,state.coefficients];mean=self.mean @ q
        variance=np.einsum('d,idf,f->i',q,self.second,q)-mean**2
        return mean,np.maximum(variance,0.),float(q @ self.von_mises_second @ q)
