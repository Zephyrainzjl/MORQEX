"""Full-grid diffuse-fatigue / DFN feedback candidate with accepted checkpoints.

This is an explicitly limited numerical candidate: scalar conductivity follows
a declared local quadratic damage law, not a validated aperture/contact law.
Crack threshold components are diagnostic, never pore or lost-lithium volume.
Radial DFN localization is restricted to this equal-size sphere pack and one
specified cathode node. It is not arbitrary-microstructure diffusion closure.
"""
from dataclasses import dataclass,asdict
import hashlib,json,time,tomllib
from pathlib import Path
import numpy as np
from scipy import ndimage
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.mechanics import TetMesh
from morqex.mechanics.tensors import MANDEL_BASIS
from .polycrystal import PolycrystalData
from .backend import NativeConfig,MFEMScalarBackend,require_local_budget
from .assembly_cache import CachedP1ScalarBackend
from .phase_field import AT2Damage,FatigueHistory
from .coupled_reference import SplitElasticity,staggered_step
from .epoch_dfn import EpochDFN,DFNFeedback
from .voxel_conductivity import effective_conductivity


@dataclass(frozen=True)
class DenseCycleConfig:
    geometry:str
    host:str
    cycles:int
    increments_per_half_cycle:int
    depth_fraction_nominal_capacity:float
    temperature_K:float
    active_young_Pa:float
    cbd_young_Pa:float
    poisson:float
    active_Gc_J_m2:float
    cbd_Gc_J_m2:float
    length_m:float
    chemical_strain_per_stoichiometry:list
    active_conductivity_S_m:float
    cbd_conductivity_S_m:float
    transport_residual:float
    stagger_tolerance:float
    precrack_radius_m:float
    protocol:str='partial-depth'
    mechanics_model:str='small-strain'

    @classmethod
    def read(cls,path):
        data=tomllib.loads(Path(path).read_text())
        if set(data)!={'schema_version','cycle'} or data['schema_version']!=1:raise ValueError('Expected version-1 cycle TOML')
        obj=cls(**data['cycle'])
        if obj.mechanics_model not in ('small-strain','finite-strain'):raise ValueError('Unknown mechanics model')
        if obj.protocol not in ('partial-depth','voltage-window'):raise ValueError('Unknown cycling protocol')
        if type(obj.cycles) is not int or obj.cycles<1 or type(obj.increments_per_half_cycle) is not int or obj.increments_per_half_cycle<2:raise ValueError('Resolved cycle counts required')
        if not 0<obj.depth_fraction_nominal_capacity<=.4 or not 0<obj.transport_residual<1 or not 0<obj.stagger_tolerance<1e-3:raise ValueError('Invalid cycle or numerical controls')
        if np.asarray(obj.chemical_strain_per_stoichiometry).shape!=(3,) or not np.isfinite(obj.chemical_strain_per_stoichiometry).all():raise ValueError('Three finite chemical strain coefficients required')
        amplitude=.4 if obj.protocol=='voltage-window' else .65*obj.depth_fraction_nominal_capacity
        if obj.mechanics_model=='small-strain' and max(abs(np.asarray(obj.chemical_strain_per_stoichiometry)))*amplitude>.02:raise ValueError('Small-strain amplitude exceeded')
        return obj


def _voxel_average(mesh,value,shape):
    values=np.asarray(value)
    weights=np.bincount(mesh.voxel_ids,weights=mesh.volumes,minlength=int(np.prod(shape)))
    if values.ndim==1:
        out=np.bincount(mesh.voxel_ids,weights=mesh.volumes*values,minlength=len(weights))
        return np.divide(out,weights,out=np.zeros_like(out),where=weights>0).reshape(shape)
    return np.stack([_voxel_average(mesh,values[:,i],shape) for i in range(values.shape[1])],axis=-1)


def _localized_concentration(data,runtime,x_index=2):
    radii,profile=runtime.radial_profile(x_index)
    c=np.zeros(data.structure.labels.shape)
    average=float(np.asarray(runtime.last['R-averaged positive particle concentration [mol.m-3]'].entries)[x_index,-1])
    maximum_correction=0.
    for p in data.structure.provenance['particles']:
        coords=np.argwhere(data.particle_ids==p['id']);r=np.linalg.norm(coords+.5-np.asarray(p['centre_zyx_voxels']),axis=1)/p['radius_voxels']
        local=np.interp(r,radii,profile);shift=average-float(np.mean(local));local+=shift
        maximum_correction=max(maximum_correction,abs(shift))
        if min(local)<=0 or max(local)>=runtime.cmax:raise RuntimeError('Conservative radial localization violates concentration bounds')
        c[tuple(coords.T)]=local
    return c,maximum_correction


def run_dense_cycle(config_path,workspace,*,environment='gentle',steps=1,output=None):
    import h5py
    require_local_budget();cfg=DenseCycleConfig.read(config_path)
    environments={'gentle':(.5,'kubc'),'fast':(2.,'kubc'),'released':(.5,'neumann')}
    if environment not in environments or type(steps) is not int or steps<1:raise ValueError('Unknown environment or invalid steps')
    rate,boundary=environments[environment];ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    data=PolycrystalData.load_hdf5(ws.path('results/top_damage/microstructures',cfg.geometry,'initial.h5'))
    if cfg.length_m<2*max(data.structure.spacing_zyx_m):raise ValueError('Phase-field length requires h <= ell/2 before convergence study')
    if data.metadata['minimum_grain_voxels']<8:raise ValueError('Unresolved grain fragments cannot enter the mechanics reference')
    directory=ws.path('results/top_damage/dense_cycles',output or environment);directory.mkdir(parents=True,exist_ok=True)
    sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
             ['dense_cycle.py','epoch_dfn.py','electrolyte_boundary.py','voxel_conductivity.py','coupled_reference.py','phase_field.py','assembly_cache.py']}
    finite=cfg.mechanics_model=='finite-strain'
    finite_backend=None
    if finite:
        from .hyperelastic import HyperelasticBackend,Hyperelasticity,finite_staggered_step,chemical_potential
        finite_backend=HyperelasticBackend(NativeConfig(),ws)
        sources['hyperelastic.py']=hashlib.sha256(Path(__file__).with_name('hyperelastic.py').read_bytes()).hexdigest()
        sources['compiled_hyperelastic']=finite_backend.manifest['identity']
    identity=hashlib.sha256(json.dumps({'config':asdict(cfg),'environment':environment,'geometry':data.fingerprint,'sources':sources},sort_keys=True).encode()).hexdigest()
    manifest_path=directory/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['identity']!=identity:raise ValueError('Dense-cycle identity changed; choose a new output, never silently resume')
    labels=data.structure.labels;shape=labels.shape;active=labels==1
    runtime=EpochDFN(host=cfg.host,active_fraction=float(active.mean()),porosity=float(np.mean(labels==0)),
        radius_m=float(data.structure.provenance['particles'][0]['radius_m']),temperature_K=cfg.temperature_K)
    started=time.perf_counter();mesh=TetMesh.from_volume(Volume(labels,spacing=data.structure.spacing_zyx_m),solid_phases=[1,2])
    backend=CachedP1ScalarBackend(mesh,MFEMScalarBackend(NativeConfig(),ws))
    damage_model=AT2Damage(mesh,backend,np.where(mesh.phases==1,cfg.active_Gc_J_m2,cfg.cbd_Gc_J_m2),cfg.length_m,linear_solver='jacobi-cg')
    young=np.where(mesh.phases==1,cfg.active_young_Pa,cfg.cbd_young_Pa)
    mechanics=(Hyperelasticity(mesh,finite_backend,young,cfg.poisson,boundary=boundary,linear_solver='matrix-free-amg-minres') if finite else
               SplitElasticity(damage_model,young,cfg.poisson,boundary=boundary,linear_solver='amg-cg'))
    grains=data.grain_ids.ravel()[mesh.voxel_ids];selected=mesh.phases==1
    beta=np.zeros((len(mesh.cells),6));rotated=data.rotated_tensor_table(np.diag(cfg.chemical_strain_per_stoichiometry))
    beta[selected]=np.einsum('kij,eij->ek',MANDEL_BASIS,rotated[grains[selected]-1])
    if finite:
        beta_tensor=np.einsum('ek,kij->eij',beta,MANDEL_BASIS)
        eigenvalues,eigenvectors=np.linalg.eigh(beta_tensor)
    checkpoints=sorted(directory.glob('step_*.h5'))
    if checkpoints:
        with h5py.File(checkpoints[-1],'r') as f:
            if f.attrs['identity']!=identity or not f.attrs['complete']:raise ValueError('Invalid accepted checkpoint')
            record=json.loads(f.attrs['record']);previous=int(record['step']);d=f['damage'][:];u=f['displacement_m'][:]
            history=FatigueHistory(f['fatigue'][:],f['alpha_previous'][:]);feedback=DFNFeedback(**record['next_feedback'])
            phase=record.get('next_phase','conditioning_charge');completed=record.get('completed_cycles',0);phase_start_q=record.get('phase_start_capacity_Ah',0.)
            if previous>0:runtime.restore_state(record['time_s'],f['dfn_state'][:],json.loads(f.attrs['dfn_inputs']))
    else:
        previous=-1;d=np.zeros(len(mesh.points));history=FatigueHistory.zero(len(mesh.cells));u=None;feedback=None
        phase='conditioning_charge';completed=0;phase_start_q=0.
        with h5py.File(directory/'mesh.h5','w') as f:
            for name,a in [('points_m',mesh.points),('tetrahedra',mesh.cells),('phases',mesh.phases),('voxel_ids',mesh.voxel_ids)]:f.create_dataset(name,data=a,compression='gzip')
        data.save_hdf5(directory/'initial_geometry.h5')
    total=2*cfg.cycles*cfg.increments_per_half_cycle if cfg.protocol=='partial-depth' else (2*cfg.cycles+1)*cfg.increments_per_half_cycle*3
    manifest={'identity':identity,'config':asdict(cfg),'environment':environment,'C_rate':rate,'boundary':boundary,
        'geometry':data.fingerprint,'sources':sources,'requested_steps':total,'cells':len(mesh.cells),'nodes':len(mesh.points),
        'publication_qualified':False,'physical_topology_qualified':False,
        'mechanics_model':cfg.mechanics_model,'stress_measure':'Cauchy, reference-volume statistics',
        'chemical_kinematics':'exp(beta * delta_stoichiometry)' if finite else 'additive eigenstrain',
        'limitations':['Numerical mechanics parameters are not calibrated NMC811 fatigue data.',
            'Radial localization at cathode node 2; same profile in equal-size spherical particles.',
            'Local quadratic diffuse conductivity is a sensitivity candidate, not a calibrated crack opening/contact law.',
            'Thresholded components diagnose damage; no material or lithium is deleted, no new electrolyte pore is fabricated.',
            'Lagged mean chemical-potential port; resolved stress-driven diffusion and spatial DFN feedback remain unqualified.',
            'Short numerical cycling does not constitute calibrated long-term life prediction.']}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    for step in range(previous+1,min(total+1,previous+1+steps)):
        if cfg.protocol=='voltage-window' and completed>=cfg.cycles:break
        tick=time.perf_counter()
        used_phase=phase;phase_capacity=None
        if step==0:
            p=data.structure.provenance['particles'][0];centre=np.asarray(p['centre_zyx_voxels'])[::-1]*np.asarray(data.structure.spacing_zyx_m)[::-1]
            plane=np.round(centre[0]/data.structure.spacing_zyx_m[2])*data.structure.spacing_zyx_m[2]
            mask=(abs(mesh.points[:,0]-plane)<1e-14)&(np.linalg.norm(mesh.points[:,1:]-centre[None,1:],axis=1)<=cfg.precrack_radius_m)
            if not mask.any():raise ValueError('Initial precrack not resolved')
            d[mask]=1.;d=damage_model.solve(np.zeros(len(mesh.cells)),d).damage
            concentration=np.zeros(shape);concentration[active]=runtime.reference_c;correction=0.;electro={'time_s':0.,'voltage_V':None,'inventory_relative_defect':0.,'relative_state_jump':0.,'net_capacity_Ah':0.}
            if finite:
                fc=np.broadcast_to(np.eye(3),(len(mesh.cells),3,3)).copy()
                state=mechanics.solve(fc,damage_model.mean_degradation(d))
            else:state=mechanics.solve(d)
            kkt=0.;iterations=0
        else:
            if cfg.protocol=='partial-depth':
                half=(step-1)//cfg.increments_per_half_cycle;current=(1 if half%2==0 else -1)*rate*runtime.capacity_Ah
                dt=3600/rate*cfg.depth_fraction_nominal_capacity/cfg.increments_per_half_cycle
            else:
                current=(1 if phase=='discharge' else -1)*rate*runtime.capacity_Ah
                dt=3600/rate/cfg.increments_per_half_cycle
            electro=runtime.advance(dt,current,feedback)
            if electro['termination']!='final time':
                if cfg.protocol=='partial-depth':raise RuntimeError('Partial-depth protocol hit a cutoff; trial is incomplete')
                required='minimum' if phase=='discharge' else 'maximum'
                cutoff=float(runtime.parameters[('Lower' if phase=='discharge' else 'Upper')+' voltage cut-off [V]'])
                if required not in electro['termination'].lower() or abs(electro['voltage_V']-cutoff)>1e-4:
                    raise RuntimeError('Protocol ended at an incorrect physical event')
                phase_capacity=abs(electro['net_capacity_Ah']-phase_start_q)
                phase_start_q=electro['net_capacity_Ah']
                if phase=='conditioning_charge':phase='discharge'
                elif phase=='discharge':phase='charge'
                else:completed+=1;phase='discharge'
            concentration,correction=_localized_concentration(data,runtime)
            eigenstrain=beta*((concentration.ravel()[mesh.voxel_ids]-runtime.reference_c)/runtime.cmax)[:,None]
            if finite:
                theta=(concentration.ravel()[mesh.voxel_ids]-runtime.reference_c)/runtime.cmax
                fc=np.einsum('eik,ek,ejk->eij',eigenvectors,np.exp(eigenvalues*theta[:,None]),eigenvectors)
                result=finite_staggered_step(mechanics,damage_model,d,history,fc,initial_displacement=u,tolerance=cfg.stagger_tolerance,max_iterations=100)
            else:result=staggered_step(mechanics,d,history,eigenstrain,initial_displacement=u,tolerance=cfg.stagger_tolerance,max_iterations=100)
            d=result.damage;history=result.fatigue;state=result.mechanics;kkt=result.damage_kkt;iterations=result.stagger_iterations
        u=state.displacement_m;d_voxel=_voxel_average(mesh,d[mesh.cells].mean(axis=1),shape)
        cell_stress=(np.einsum('kij,eij->ek',MANDEL_BASIS,state.cauchy_stress_Pa) if finite else state.mean_cell_stress_Pa)
        stress=_voxel_average(mesh,cell_stress,shape)
        # Exact element mean of the quadratic conductivity law; g(mean(d))
        # would incorrectly discard the positive within-element variance.
        g_cell=(damage_model.mean_degradation(d)-damage_model.k)/(1-damage_model.k)
        g=cfg.transport_residual+(1-cfg.transport_residual)*_voxel_average(mesh,g_cell,shape)
        conductivity=np.where(active,cfg.active_conductivity_S_m,np.where(labels==2,cfg.cbd_conductivity_S_m,0.))*g
        electronic=effective_conductivity(conductivity,data.structure.spacing_zyx_m)
        # Resolved pores stay pores. Damage is not converted to liquid volume.
        if step==0:
            ionic=effective_conductivity((labels==0).astype(float),data.structure.spacing_zyx_m)
            ionic_efficiency=ionic['effective']
        else:ionic_efficiency=feedback.electrolyte_efficiency
        if finite:
            derivative=beta_tensor@fc/runtime.cmax
            mechanical_mu=chemical_potential(state.first_piola_Pa,state.deformation_gradient,fc,derivative)
            mu=float(mesh.volumes[selected]@mechanical_mu[selected]/mesh.volumes[selected].sum())
        else:mu=float(np.sum(-state.mean_cell_stress_Pa[selected]*beta[selected]*mesh.volumes[selected,None])/runtime.cmax/mesh.volumes[selected].sum())
        feedback=DFNFeedback(electronic['effective'],ionic_efficiency,1.,mu)
        topology={}
        for threshold in (.8,.9,.95):
            _,components=ndimage.label(active&(d_voxel<threshold));topology[str(threshold)]=int(components)
        alpha_voxel=_voxel_average(mesh,history.accumulated_J_m3,shape)
        component_labels,component_count=ndimage.label(active&(d_voxel<.95))
        record={**electro,'step':step,'mechanical_cycle':step/(2*cfg.increments_per_half_cycle),
            'protocol':cfg.protocol,'protocol_phase':used_phase,'next_phase':phase,'completed_cycles':completed,
            'phase_start_capacity_Ah':phase_start_q,'completed_phase_capacity_Ah':phase_capacity,
            'mean_active_damage':float(d_voxel[active].mean()),'maximum_damage':float(d.max()),
            'active_volume_m3':float(active.sum()*np.prod(data.structure.spacing_zyx_m)),
            'max_principal_stress_P99_MPa':float(np.quantile(np.linalg.eigvalsh(np.einsum('ek,kij->eij',stress[active],MANDEL_BASIS))[:,-1],.99)/1e6),
            'damage_kkt':kkt,'equilibrium_residual':state.equilibrium_residual,'stagger_iterations':iterations,
            'elastic_energy_J':state.elastic_energy_J,'concentration_projection_correction_mol_m3':correction,
            'diagnostic_intact_components':topology,'next_feedback':asdict(feedback),
            'transport_residual':electronic['residual'],'transport_energy_defect':electronic['energy_relative_defect'],
            'wall_seconds':time.perf_counter()-tick}
        if cfg.protocol=='voltage-window':record['mechanical_cycle']=float(completed)+(.5 if phase=='charge' else 0.)
        if finite:
            record.update(mechanics_model=cfg.mechanics_model,minimum_jacobian=state.minimum_jacobian,
                          finite_newton_iterations=state.newton_iterations,
                          finite_total_newton_iterations=sum(s['newton_iterations'] for s in mechanics.solve_diagnostics),
                          finite_roundoff_merit_acceptances=sum(s['roundoff_merit_acceptances'] for s in mechanics.solve_diagnostics),
                          finite_line_search_backtracks=sum(s['backtracks'] for s in mechanics.solve_diagnostics))
        nodal=d[mesh.cells];grad=np.einsum('ei,eij->ej',nodal,mesh.gradients)
        d2=(nodal.sum(axis=1)**2+(nodal**2).sum(axis=1))/20
        record['crack_surface_functional_m2']=float(mesh.volumes@(.5*d2/cfg.length_m+.5*cfg.length_m*np.sum(grad**2,axis=1)))
        target=directory/f'step_{step:05d}.h5';temporary=target.with_suffix('.pending')
        with h5py.File(temporary,'w') as f:
            f.attrs.update(identity=identity,complete=True,record=json.dumps(record),dfn_inputs=json.dumps(runtime.last_inputs))
            for key,value in [('damage',d),('displacement_m',u),('fatigue',history.accumulated_J_m3),('alpha_previous',history.previous_alpha_J_m3),
                ('voxel/damage',d_voxel),('voxel/concentration_mol_m3',concentration),('voxel/stress_Mandel_Pa',stress),
                ('voxel/fatigue_J_m3',alpha_voxel),('voxel/electronic_potential_unit_load',electronic['potential']),
                ('voxel/diagnostic_intact_component_d095',component_labels),('voxel/phase_labels',labels)]:
                f.create_dataset(key,data=value,compression='gzip')
            if finite:
                for name,value in [('deformation_gradient',state.deformation_gradient),('chemical_deformation_gradient',fc),
                                   ('first_piola_Pa',state.first_piola_Pa),('cauchy_stress_Pa',state.cauchy_stress_Pa)]:
                    f.create_dataset('finite/'+name,data=value,compression='gzip')
                f.create_dataset('finite/mechanical_chemical_potential_J_mol',data=mechanical_mu,compression='gzip')
            if step>0:
                f.create_dataset('dfn_state',data=np.asarray(runtime.last.last_state.y).ravel())
                for name,value in [('time_s',runtime.last.t),('voltage_V',runtime.last['Voltage [V]'].entries),
                                   ('current_A',runtime.last['Current [A]'].entries),('net_capacity_Ah',runtime.last['Discharge capacity [A.h]'].entries)]:
                    f.create_dataset('curve/'+name,data=value)
        temporary.replace(target);previous=step
        print(json.dumps(record),flush=True)
    manifest.update(last_accepted_step=previous,completed_cycles=completed,
        protocol_complete=(previous==total if cfg.protocol=='partial-depth' else completed>=cfg.cycles),wall_seconds=time.perf_counter()-started)
    manifest_path.write_text(json.dumps(manifest,indent=2));return manifest|{'directory':str(directory)}
