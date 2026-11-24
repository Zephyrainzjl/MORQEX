"""Sparse coupled reference and compact energy ROM, same output and tolerance.

Resolution/seed scan is NOT fixed-geometry mesh convergence. Synthetic inputs,
no DFN, no RVE representativity or originality claims. Full solvers reuse LU.
"""
import argparse,gc,hashlib,json,os,time,tomllib
from pathlib import Path
import numpy as np
from morqex import Workspace
from morqex.gemsim import GEMsim
from morqex.mechanics import TetMesh,ElasticitySolver,isotropic_stiffness
from morqex.transport import VoxelDiffusion,ConcentrationInfeasible
from morqex.concentration import ConcentrationBasis
from morqex.sparse_coupled import SparseChemoelasticSystem,sparse_bytes
from morqex.streamed_rom import StreamedEnergyROM,StressLocalization,OnlineEnergyKernel
from morqex.gradient_transport import EnergyAccuracyError
from morqex.statistics import principal_tension,weighted_quantile
from study_controlled_transport import patterns_for

ROOT=Path(__file__).resolve().parents[1]


def source_hashes(config):
    paths=[*sorted((ROOT/'src/morqex').glob('*.py')),*sorted((ROOT/'src/morqex/mechanics').glob('*.py')),Path(__file__),Path(__file__).with_name('study_controlled_transport.py'),config]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def case(size,seed,cfg,workspace):
    started=time.perf_counter();provider=GEMsim(workspace)
    geometry=provider.random_field(shape=(size,)*3,fractions=tuple(cfg['fractions']),length_scale=tuple(cfg['length_scale_voxels']),
        seed=seed,spacing=(cfg['extent_m']/size,)*3,phase_kinds={0:'pore',1:'active',2:'cbd'})
    volume=provider.to_volume(geometry)
    diffusion=VoxelDiffusion(volume,active_phases=[1],diffusivity=cfg['diffusivity'],reactive_phases=[0])
    mesh=TetMesh.from_volume(volume,solid_phases=[1,2]);materials={1:isotropic_stiffness(1e9,.25),2:isotropic_stiffness(2e8,.3)}
    solver=ElasticitySolver(mesh,np.array([materials[int(p)] for p in mesh.phases]))
    system=SparseChemoelasticSystem(solver,diffusion,np.array(cfg['expansion']),patterns_for(diffusion),chemical_curvature=cfg['chemical_curvature'],
        reference_concentration=cfg['reference_concentration'],maximum_concentration=cfg['maximum_concentration'])
    shared_setup=time.perf_counter()-started
    t=(np.arange(cfg['steps'])+1)*cfg['dt_s'];phase=np.random.default_rng(seed+951).uniform(0,2*np.pi,4)
    controls=np.zeros((len(t),5));controls[:,0]=1+.2*np.sin(.4*t+phase[0]);controls[:,1:4]=.25*np.sin(t[:,None]*np.array([.2,.35,.5])+phase[1:])
    controls[len(t)//2:,4]=2.5*(1+.1*np.cos(.5*t[len(t)//2:]))
    controls*=cfg['flux_scale'];strains=np.zeros((len(t),6));strains[:,0]=cfg['macro_amplitude']*np.sin(.2*t)
    record={'size':size,'seed':seed,'N':system.n,'tetrahedra':len(mesh.cells),'mechanical_dofs':solver.Kr.shape[0],
        'diffusion_components':diffusion.n_components,'geometry_fingerprint':geometry.fingerprint,'shared_mechanical_system_setup_s':shared_setup,
        'sparse_system_array_bytes':system.retained_sparse_bytes(),'dense_H_hypothetical_bytes':8*system.n**2,'identity_full_stress_hypothetical_bytes':8*6*len(mesh.cells)*(system.n+6),
        'mechanical_LU_array_bytes':sum(sparse_bytes(a) for a in (solver.factor.L,solver.factor.U)) if solver.factor else 0,'references':[],'methods':[]}
    references={}
    for method in ('mixed-LU','Schur-GMRES'):
        begin=time.perf_counter();system.prepare(cfg['dt_s'],method);setup=time.perf_counter()-begin
        full=system.initialize(np.zeros(system.n));history=[];times=[];maxmass=maxenergy=maxres=0.;iterations=0
        for u,e in zip(controls,strains):
            begin=time.perf_counter();full,diagnostic=system.step(full,cfg['dt_s'],u,macro_strain=e,method=method,relative_tolerance=cfg['relative_solver_tolerance'])
            times.append(time.perf_counter()-begin);history.append(full)
            maxmass=max(maxmass,float(np.max(np.abs(diagnostic['component_mass_error']))));maxenergy=max(maxenergy,abs(diagnostic['energy_balance']));maxres=max(maxres,diagnostic['relative_algebraic_residual']);iterations+=diagnostic['iterations']
        references[method]=history
        record['references'].append({'method':method,'setup_s':setup,'online_s':sum(times),'maximum_component_mass_error_mol':maxmass,
            'maximum_energy_balance_J_m3':maxenergy,'maximum_relative_residual':maxres,'iterations':iterations,
            'factor_array_bytes':sum(sparse_bytes(a) for a in (system._mixed_factor.L,system._mixed_factor.U)) if method=='mixed-LU' else sum(sparse_bytes(a) for a in (system._krylov_preconditioner.L,system._krylov_preconditioner.U))})
    truth=references['mixed-LU']
    cross=max(np.linalg.norm(a.deviation-b.deviation)/max(np.linalg.norm(a.deviation),1e-30) for a,b in zip(truth,references['Schur-GMRES']))
    assert cross < 1e-8
    record['full_reference_cross_difference']=float(cross)
    best_reference=min(record['references'],key=lambda r:r['online_s'])
    record['fastest_full_reference']=best_reference['method']
    candidates=[]
    coupled_basis,coupled_work=system.coupled_resolvent_basis(min(system.n,diffusion.n_components+max(cfg['nonconstant_modes'])),moments=cfg['resolvent_moments'])
    record['coupled_basis_library_work']=coupled_work
    for family in ('Fick-resolvent','coupled-resolvent'):
      for modes in cfg['nonconstant_modes']:
        begin=time.perf_counter()
        if family == 'Fick-resolvent':
            basis=ConcentrationBasis.boundary_resolvent(diffusion,min(system.n,diffusion.n_components+modes),patterns_for(diffusion),moments=cfg['resolvent_moments'])
        else:
            basis=ConcentrationBasis(coupled_basis.values[:,:min(system.n,diffusion.n_components+modes)],coupled_basis.weights,'coupled-resolvent',diffusion.n_components)
        rom=StreamedEnergyROM(system,basis);kernel=rom.detach_online();setup=time.perf_counter()-begin+(coupled_work['complete_library_seconds'] if family=='coupled-resolvent' else 0.)
        state=rom.initialize(np.zeros(system.n),system,error_tolerance=cfg['energy_error_tolerance'])
        history=[];elapsed=0.;maxmass=maxenergy=0.;qualified=0;error2=truth2=0.;coverage=True
        for k,(u,e) in enumerate(zip(controls,strains)):
            begin=time.perf_counter();state,diag=kernel.trial(state,cfg['dt_s'],u,macro_strain=e)
            try:kernel.accept(state,cfg['energy_error_tolerance']);ok=True
            except (EnergyAccuracyError,ConcentrationInfeasible):ok=False
            elapsed+=time.perf_counter()-begin;qualified+=int(ok)
            difference=system.scale_c*(truth[k].deviation-rom.basis.values @ state.coefficients)
            actual=float(np.sqrt(max(difference @ system.h_action(difference),0.)))
            coverage &= actual <= state.error_bound+1e-7
            maxmass=max(maxmass,float(np.max(np.abs(diag['component_mass_error']))));maxenergy=max(maxenergy,abs(diag['energy_balance']))
            error2+=float(np.dot(difference,difference));truth2+=float(np.dot(system.scale_c*truth[k].deviation,system.scale_c*truth[k].deviation))
            history.append({'time_s':state.time,'actual_energy_error':actual,'bound':state.error_bound,'qualified':ok})
        assert coverage and maxmass < 1e-24 and maxenergy < 1e-6
        method={'basis_family':family,'initial_nonconstant_modes':modes,'total_modes':rom.dimension,'setup_s':setup,'online_s':elapsed,'qualified_steps':qualified,
            'bound_coverage':bool(coverage),'maximum_bound':max(r['bound'] for r in history),'relative_concentration_trajectory_L2':float(np.sqrt(error2/max(truth2,1e-30))),
            'online_array_bytes':kernel.online_array_bytes(),'offline_basis_and_action_bytes':rom.basis.values.nbytes+rom.offline_normalized_basis.nbytes+rom.offline_hessian_columns.nbytes,
            'maximum_component_mass_error_mol':maxmass,'maximum_energy_balance_J_m3':maxenergy,'measured_speedup':best_reference['online_s']/elapsed,
            'qualified_speedup':best_reference['online_s']/elapsed if qualified==cfg['steps'] else None,'history':history,
            'mass_riesz_mechanical_queries':rom.riesz_mechanical_queries,'macro_lifting_mechanical_queries':rom.lifting_mechanical_queries}
        saving=best_reference['online_s']-elapsed
        # Includes all ROM build costs; common mechanics cancels in both totals.
        method['offline_break_even_runs']=max(0.,(setup-best_reference['setup_s'])/saving) if saving > 0 else None
        record['methods'].append(method);candidates.append((rom,kernel,state,method))
    accepted=[x for x in candidates if x[3]['qualified_steps']==cfg['steps']]
    selected=min(accepted,key=lambda x:x[0].dimension) if accepted else None
    record['selected_modes']=selected[0].dimension if selected else None
    record['selected_family']=selected[3]['basis_family'] if selected else None
    if selected:
        rom,kernel,state,method=selected
        start=time.perf_counter();local=StressLocalization(system,rom,np.arange(len(mesh.cells)),maximum_bytes=cfg['field_byte_budget']);field_setup=time.perf_counter()-start
        start=time.perf_counter();actual_full=system.fields(truth[-1].deviation,strains[-1]);fom_field=time.perf_counter()-start
        start=time.perf_counter();stress=local.stress(state);rom_field=time.perf_counter()-start
        difference=stress-actual_full.stress;weights=mesh.volumes/mesh.rve_volume
        stress_error=float(np.sqrt(np.einsum('e,ei,ei->',weights,difference,difference)/max(np.einsum('e,ei,ei->',weights,actual_full.stress,actual_full.stress),1e-30)))
        c=rom.basis.values @ state.coefficients
        independent=system.fields(c,state.macro_strain)
        response_cross=np.linalg.norm(stress-independent.stress)/max(np.linalg.norm(independent.stress),1e-30)
        assert response_cross < 1e-9
        q99=weighted_quantile(principal_tension(stress),mesh.volumes,.99);q99_full=weighted_quantile(principal_tension(actual_full.stress),mesh.volumes,.99)
        statistics=local.detach_statistics()
        start=time.perf_counter();mean,var,vm2=statistics.moments(state);statistics_time=time.perf_counter()-start;mean_difference=np.linalg.norm(kernel.macro_stress(state)-np.einsum('e,ei->i',mesh.volumes,stress)/mesh.rve_volume)
        assert mean_difference < 1e-6
        record['selected_fields']={'setup_s':field_setup,'full_FEM_recovery_s':fom_field,'ROM_recovery_s':rom_field,'operator_bytes':local.operator.nbytes,
            'relative_stress_L2':stress_error,'P99_relative_error':float(abs(q99-q99_full)/max(abs(q99_full),1e3)),
            'independent_FEM_same_input_difference':float(response_cross),'macro_stress_average_difference_Pa':float(mean_difference),
            'sample_volume_mean':mean.tolist(),'sample_component_variance':var.tolist(),'von_mises_second_moment':vm2,'statistics_online_s':statistics_time,'statistics_operator_bytes':statistics.mean.nbytes+statistics.second.nbytes+statistics.von_mises_second.nbytes,
            'scope':'final time only; exact moments of the recovered discrete field; P99 error validated against full coupled final field, not a P99 certificate'}
        batches=[]
        for p in cfg['batch_points']:
            values=np.tile(state.coefficients,(p,1));es=np.tile(state.macro_strain,(p,1));times=[]
            for _ in range(5):
                start=time.perf_counter();kernel.batch_macro(values,es);times.append(time.perf_counter()-start)
            batches.append({'macro_points':p,'median_s':float(np.median(times)),'operator_instances':1,'input_bytes':values.nbytes+es.nbytes,'output_bytes':p*6*8})
        record['macro_batches']=batches
        saved=kernel.save(workspace,str(Path.cwd().relative_to(workspace.root)/f'kernel_{size}_{seed}'),state)
        loaded,restored=OnlineEnergyKernel.load(saved)
        np.testing.assert_allclose(loaded.macro_stress(restored),kernel.macro_stress(state),atol=0,rtol=0)
        record['compact_kernel_path']=str(saved)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--size',type=int,required=True);parser.add_argument('--seed',type=int);args=parser.parse_args()
    cfg=tomllib.loads(args.config.read_text());ws=Workspace(Path(os.environ['MORQEX_WORKSPACE']))
    if args.size not in cfg['sizes'] or args.size > 18 or cfg['steps'] > 80 or len(cfg['seeds']) > 3: raise ValueError('Scaling study budget exceeded')
    if args.seed is not None and args.seed not in cfg['seeds']: raise ValueError('Unknown seed')
    if args.size >= 18 and args.seed is None: raise ValueError('Large cases require one seed per bounded process')
    result={'schema':cfg['schema'],'config':cfg,'source_sha256':source_hashes(args.config),'cases':[],
        'scope':'fixed quadratic coupled model; new random geometry at each size, not continuum mesh convergence; synthetic values; no DFN/novelty qualification',
        'timing_scope':'same concentration step, energy and component inventories; ROM certificate included; full fields/tail outputs and offline costs reported separately; fastest of two full references'}
    for seed in ([args.seed] if args.seed is not None else cfg['seeds']):
        result['cases'].append(case(args.size,seed,cfg,ws));gc.collect()
        Path('sparse_scaling.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'size':args.size,'seed':seed,'done':True}),flush=True)


if __name__=='__main__':main()
