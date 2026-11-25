"""One local, resource-limited DFN–microstructure case, with real field evidence."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import tomllib
import numpy as np
import pybamm as pb
from morqex import Workspace,MicrostructureData
from morqex.gemsim import GEMsim
from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode
from morqex.electrochemistry.microstructure_cache import OnlineMicrostructureOperator,LocalizationMap
from morqex.electrochemistry.microstructure_dfn import MicrostructureStack,build_microstructure_dfn,key
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
from morqex.electrochemistry.microstructure_validation import thermodynamic_observables
from morqex.electrochemistry.validation import check_protocol
from morqex.solvers.pybamm_backend import solver,provenance

ROOT=Path(__file__).resolve().parents[1]


def source_hashes():
    files=[*(ROOT/'src/morqex/electrochemistry'/name for name in ['microstructure_operator.py','microstructure_cache.py','microstructure_dfn.py',
        'microstructure_parameters.py','microstructure_validation.py']),ROOT/'src/morqex/electrochemistry/parameters.py',
        ROOT/'src/morqex/electrochemistry/validation.py',ROOT/'src/morqex/solvers/pybamm_backend.py',
        ROOT/'src/morqex/transport.py',ROOT/'src/morqex/energy.py',*ROOT.joinpath('src/morqex/mechanics').glob('*.py'),
        ROOT/'src/morqex/gemsim.py',ROOT/'configs/microstructure_dfn.toml',Path(__file__)]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def prepare(workspace,parameters,args,cfg):
    provider=GEMsim(workspace);paths={};records={};timing={}
    for domain,seed in [('negative',args.seed),('positive',args.positive_seed if args.positive_seed is not None else args.seed)]:
        started=time.perf_counter()
        data=provider.random_field(shape=(args.shape,)*3,fractions=(cfg['pore_fraction_requested'],1-cfg['pore_fraction_requested']),
            length_scale=tuple(cfg['correlation_voxels']),seed=seed,spacing=(cfg['spacing_m'],)*3,phase_kinds={0:'pore',1:'active'})
        if args.refine>1:
            labels=data.labels
            for axis in range(3):labels=np.repeat(labels,args.refine,axis=axis)
            data=MicrostructureData(labels,tuple(x/args.refine for x in data.spacing_zyx_m),data.origin_zyx_m,data.phase_kinds,
                data.provenance|{'exact_geometry_refinement':args.refine})
        full=MicrostructureElectrode(data,provider,parameters,domain,thermal_expansion=cfg['thermal_expansion_per_K'])
        op=full.reduce() if args.regions==0 else full.partition(args.regions,args.strategy)
        paths[domain]=op.save(workspace,args.case+'_'+domain)
        records[domain]=op.metadata()|{'seed':seed,'grid_shape':list(data.labels.shape),'rve_volume_m3':full.mesh.rve_volume,
            'full_diffusion_components':int(full.diffusion.n_components),'requested_regions':args.regions,
            'offline_mechanical_factorizations':int(full.solver.factorizations)}
        timing[domain]=time.perf_counter()-started
        del full,op;gc.collect()
    return paths,records,timing


def main():
    cfg=tomllib.loads((ROOT/'configs/microstructure_dfn.toml').read_text());parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True);parser.add_argument('--regions',type=int,default=cfg['regions'])
    parser.add_argument('--strategy',choices=['geometry','mechanics'],default='geometry');parser.add_argument('--shape',type=int,default=cfg['shape'][0])
    parser.add_argument('--seed',type=int,default=cfg['negative_seed']);parser.add_argument('--positive-seed',type=int)
    parser.add_argument('--refine',type=int,default=1);parser.add_argument('--nx',type=int,default=cfg['x_points'])
    parser.add_argument('--rate',type=float,default=1.);parser.add_argument('--feedback',type=float,default=1.)
    parser.add_argument('--constraint',choices=['pressure','fixed-thickness'],default='pressure')
    parser.add_argument('--protocol',choices=['discharge','cycles','short'],default='discharge');parser.add_argument('--cycles',type=int,default=cfg['cycles'])
    args=parser.parse_args()
    if not args.case or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.case):raise ValueError('Safe case label required')
    hashes=source_hashes();backend=provenance();workspace=Workspace(Path(os.environ['MORQEX_WORKSPACE']))
    output=workspace.path('results','microstructure_dfn',args.case);output.mkdir(parents=True,exist_ok=True)
    p,audit=microstructure_parameters()
    for domain in ['Negative','Positive']:
        name=f'{domain} electrode partial molar volume [m3.mol-1]';p.update({name:p[name]*cfg['partial_molar_volume_scale']})
    paths,geometry,offline=prepare(workspace,p,args,cfg)
    operators={d:OnlineMicrostructureOperator.load(path) for d,path in paths.items()}
    assert all(not hasattr(op,'full') and not hasattr(op,'solver') for op in operators.values())
    stack=MicrostructureStack(operators['negative'],operators['positive'],p,constraint=args.constraint);stack.update_parameters(p)
    model=build_microstructure_dfn(stack,feedback=args.feedback)
    if any('particle' in dom for var in model.rhs for dom in var.domain):raise RuntimeError('Radial particle states were not fully replaced')
    lower=float(p['Lower voltage cut-off [V]']);upper=float(p['Upper voltage cut-off [V]']);nominal=float(p['Nominal cell capacity [A.h]'])
    discharge=f'Discharge at {args.rate}C until {lower} V'
    if args.protocol=='cycles':steps=[(discharge,f"Rest for {cfg['rest_s']} seconds",f"Charge at {cfg['charge_rate']}C until {upper} V",f'Hold at {upper} V until C/20',f"Rest for {cfg['rest_s']} seconds")]*args.cycles
    elif args.protocol=='short':steps=[f'Discharge at {args.rate}C for 120 seconds']
    else:steps=[discharge]
    sim=pb.Simulation(model,parameter_values=p,solver=solver(rtol=cfg['relative_tolerance'],atol=cfg['absolute_tolerance']),
        experiment=pb.Experiment(steps,period=f"{cfg['output_period_s']} seconds"),
        var_pts={**model.default_var_pts,'x_n':args.nx,'x_s':args.nx,'x_p':args.nx,'r_n':2,'r_p':2})
    started=time.perf_counter();solution=sim.solve();solve_seconds=time.perf_counter()-started
    if args.protocol=='short':
        if abs(solution.t[-1]-120)>1e-5:raise RuntimeError('Short case incomplete')
        protocol={'completed':True}
    else:protocol=check_protocol(solution,cycles=args.cycles if args.protocol=='cycles' else 1,steps_per_cycle=5 if args.protocol=='cycles' else 1,
        lower_V=lower,upper_V=upper,current_A=nominal)
    micro_state_jump=0.;previous=None
    for cycle in solution.cycles:
        for step in cycle.steps:
            if previous is not None:
                for d,op in operators.items():
                    cmax=float(p[f'Maximum concentration in {d} electrode [mol.m-3]'])
                    for k in range(op.dimension):
                        jump=np.max(abs(step[key(d,k,'concentration [mol.m-3]')].entries[:,0]-previous[key(d,k,'concentration [mol.m-3]')].entries[:,-1]))/cmax
                        micro_state_jump=max(micro_state_jump,float(jump))
            previous=step
    energy=thermodynamic_observables(solution,stack,p,feedback=args.feedback);metrics=energy['metrics']
    charge={}
    A=float(p['Electrode width [m]'])*float(p['Electrode height [m]'])*float(p['Number of electrodes connected in parallel to make a cell'])
    current=np.asarray(solution['Current [A]'].entries)
    for d in operators:
        integrated=A*stack.lengths[d]*np.asarray(solution[f'X-averaged {d} electrode volumetric interfacial current density [A.m-3]'].entries)
        charge[d]=float(np.max(abs(integrated-(1 if d=='negative' else -1)*current))/max(1.,np.max(abs(current))))
    metrics.update(electrode_charge_residual=charge,offline_build_seconds=offline,build_and_solve_seconds=solve_seconds,
        relative_micro_state_step_jump=micro_state_jump,
        end_time_s=float(solution.t[-1]),temperature_max_K=float(np.max(solution['Volume-averaged cell temperature [K]'].entries)),
        differential_state_count=int(sim.built_model.len_rhs) if sim.built_model is not None else int(solution.all_models[0].len_rhs))
    curves={'time_s':solution.t.tolist()}
    names={'voltage_V':'Voltage [V]','current_A':'Current [A]','temperature_K':'Volume-averaged cell temperature [K]',
        'stack_stress_Pa':'MORQEX stack axial stress [Pa]','thickness_m':'MORQEX sandwich thickness change [m]',
        'total_dissipation_W':'MORQEX total dissipation [W]','micro_diffusion_dissipation_W':'MORQEX micro diffusion dissipation [W]',
        'micro_reaction_dissipation_W':'MORQEX micro reaction dissipation [W]','macro_conversion_W':'MORQEX macro electric conversion heat [W]',
        'native_ohmic_W':'MORQEX native ohmic heating [W]','electrical_work_J':'MORQEX accumulated electrical work [J]',
        'cooling_work_J':'MORQEX accumulated cooling work [J]'}
    for name,variable in names.items():curves[name]=np.asarray(solution[variable].entries).tolist()
    for name in ['internal_energy_J','chemical_energy_J','elastic_internal_energy_J','entropy_J_K','energy_residual_J','entropy_residual_J_K']:curves[name]=energy[name].tolist()
    selected=np.unique(np.r_[0,np.argmin(abs(current)),len(solution.t)//2,len(solution.t)-1]).astype(int)
    micro={}
    for d,op in operators.items():
        concentrations=np.stack([solution[key(d,k,'concentration [mol.m-3]')].entries for k in range(op.dimension)],axis=-1)
        reactions=np.stack([solution[key(d,k,'reaction current [A.m-2]')].entries for k in range(op.dimension)],axis=-1)
        strain=np.asarray(solution[f'MORQEX {d} axial strain'].entries)
        temperature=np.asarray(curves['temperature_K']);theta=temperature-float(p['Reference temperature [K]'])
        q=np.concatenate([strain[...,None],concentrations-op.reference_concentration,np.broadcast_to(theta,(args.nx,len(theta)))[...,None]],axis=-1)
        stats=op.moments(q);local=LocalizationMap(paths[d])
        field=local.recover(concentrations[:,selected,:],strain[:,selected],theta[selected])
        np.savez_compressed(output/(d+'_fields.npz'),time_s=solution.t[selected],concentration=field['concentration'],stress=field['stress'],
            strain=field['strain'],displacement=field['displacement'],q=q[:,selected,:])
        micro[d]={'concentration_regions':concentrations.tolist(),'reaction_currents_regions_A_m2':reactions.tolist(),
            'potential_difference_V':np.asarray(solution[f'{d.capitalize()} electrode surface potential difference [V]'].entries).tolist(),
            'electrolyte_concentration_mol_m3':np.asarray(solution[f'{d.capitalize()} electrolyte concentration [mol.m-3]'].entries).tolist(),
            'volumetric_reaction_current_A_m3':np.asarray(solution[f'{d.capitalize()} electrode volumetric interfacial current density [A.m-3]'].entries).tolist(),
            'x_m':solution[key(d,0,'concentration [mol.m-3]')].mesh.nodes.tolist(),
            'axial_strain':strain.tolist(),'macro_mean_stress_Pa':stats['volume_mean_stress'].tolist(),
            'solid_stress_variance_Pa2':stats['solid_conditional_variance'].tolist(),
            'stoichiometry_range':[float(concentrations.min()/p[f'Maximum concentration in {d} electrode [mol.m-3]']),float(concentrations.max()/p[f'Maximum concentration in {d} electrode [mol.m-3]'])],
            'operator_path':str(paths[d]),'snapshot_times_s':solution.t[selected].tolist()}
    checks={'protocol':True,'microstate_continuity':micro_state_jump<1e-8,'lithium':metrics['relative_lithium_drift']<1e-10,'energy':metrics['relative_internal_energy_residual']<cfg['energy_tolerance'],
        'entropy':metrics['relative_entropy_residual']<cfg['entropy_tolerance'],'dissipation':metrics['minimum_dissipation_W']>=-1e-7,
        'port_identity':max(metrics['port_identity_relative_residual'].values())<cfg['port_identity_tolerance'],
        'charge':max(charge.values())<5e-5}
    if source_hashes()!=hashes or provenance()!=backend:raise RuntimeError('Physics source changed during this case')
    result={'schema':'morqex-microstructure-dfn-result-1','case':args.case,'arguments':vars(args),'parameters':audit,
        'geometry':geometry,'operators':{d:op.metadata() for d,op in operators.items()},'source_sha256':hashes,'backend':backend,
        'protocol':protocol,'metrics':metrics,'checks':{k:bool(v) for k,v in checks.items()},'curves':curves,'microstructure':micro,
        'scope':'Fixed geometry/linear elasticity; ideal electrolyte; nonuniform solid micro FV states; 1D DFN + 3D SVE, not a full pore-resolved electrolyte reference',
        'parameter_limits':'Explicit convex/compact OCP, Omega x0.1, synthetic GEM and alpha; no calibrated real-cell mechanism conclusion'}
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'case':args.case,'checks':checks,'metrics':metrics,'output':str(output)},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x),flush=True)
    if not all(checks.values()):raise SystemExit('Cross-scale numerical gate failed; retain this case and inspect')


if __name__=='__main__':main()
