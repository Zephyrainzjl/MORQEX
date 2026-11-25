"""Resource-limited, one-case full DFN validation. All artifacts are external.

Run through scripts/run_local.py. Case identifiers are explicit CLI arguments;
raw-reference and modified thermodynamic-closure results are separate tracks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import tomllib
import numpy as np
import pybamm as pb
from morqex import Workspace
from morqex.electrochemistry.mechanical_closure import ElectrodeClosure,AxialStackClosure
from morqex.electrochemistry.parameters import convex_parameters
from morqex.electrochemistry.dfn import build_model
from morqex.electrochemistry.validation import check_protocol
from morqex.solvers.pybamm_backend import solver,provenance

ROOT=Path(__file__).resolve().parents[1]


def source_hashes():
    files=[*ROOT.joinpath('src/morqex/electrochemistry').glob('*.py'),*ROOT.joinpath('src/morqex/solvers').glob('*.py'),*ROOT.joinpath('src/morqex/mechanics').glob('*.py'),Path(__file__),ROOT/'configs/dfn_validation.toml',ROOT/'src/morqex/energy.py',ROOT/'src/morqex/gemsim.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    cfg=tomllib.loads((ROOT/'configs/dfn_validation.toml').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    parser.add_argument('--mode',choices=['stock','oneway','coupled'],default='coupled')
    parser.add_argument('--protocol',choices=['discharge','cycles','comsol','crosscheck'],default='discharge')
    parser.add_argument('--parameters',default=cfg['parameter_set'])
    parser.add_argument('--ocp',choices=['raw','convex'],default=cfg['ocp_branch'])
    parser.add_argument('--constraint',choices=['pressure','fixed-thickness'],default='pressure')
    parser.add_argument('--pressure',type=float,default=0.)
    parser.add_argument('--rate',type=float,default=1.)
    parser.add_argument('--cycles',type=int,default=cfg['cycles'])
    parser.add_argument('--omega-scale',type=float,default=cfg['partial_molar_volume_scale'])
    parser.add_argument('--nx',type=int,default=cfg['x_points']);parser.add_argument('--nr',type=int,default=cfg['r_points'])
    parser.add_argument('--backend',choices=['idaklu','casadi'],default='idaklu')
    parser.add_argument('--algebraic-step-tol',type=float,default=1e-12)
    args=parser.parse_args()
    hashes=source_hashes()
    backend_sources=provenance()
    if not args.case or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.case):raise ValueError('Safe case identifier required')
    ws=Workspace(Path(os.environ['MORQEX_WORKSPACE']));out=ws.path('results','dfn',args.case);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();audit={'source_parameter_set':args.parameters,'modification':'none'}
    p,audit=convex_parameters(args.parameters) if args.ocp=='convex' else (pb.ParameterValues(args.parameters),audit)
    stack=None
    if args.mode!='stock':
        if args.omega_scale<=0:raise ValueError('Positive explicitly declared expansion scale required')
        for d in ['Negative','Positive']:
            key=f'{d} electrode partial molar volume [m3.mol-1]';p.update({key:p[key]*args.omega_scale})
        a=ElectrodeClosure.from_gemsim(ws,p,'negative',seed=104);b=ElectrodeClosure.from_gemsim(ws,p,'positive',seed=205)
        a.save(ws,args.case+'_negative');b.save(ws,args.case+'_positive')
        stack=AxialStackClosure(a,b,p,constraint=args.constraint,pressure_Pa=args.pressure)
        model=build_model(stack,feedback=float(args.mode=='coupled'))
        audit['mechanics']={'partial_molar_volume_scale':args.omega_scale,'purpose':'declared small-strain numerical verification, not a calibrated material prediction',
            'negative':a.metadata,'positive':b.metadata,'constraint':args.constraint,'pressure_Pa':args.pressure}
    else:model=pb.lithium_ion.DFN(options={'thermal':'lumped','heat of mixing':'true'})
    preparation=time.perf_counter()-started
    backend=solver(rtol=cfg['relative_tolerance'],atol=cfg['absolute_tolerance']) if args.backend=='idaklu' else pb.CasadiSolver(mode='safe',rtol=cfg['relative_tolerance'],atol=cfg['absolute_tolerance'],root_tol=1e-9,
        root_method=pb.CasadiAlgebraicSolver(tol=1e-10,step_tol=args.algebraic_step_tol),perturb_algebraic_initial_conditions=False)
    n=args.nx;r=args.nr;var_pts={**model.default_var_pts,'x_n':n,'x_s':n,'x_p':n,'r_n':r,'r_p':r}
    lower=float(p['Lower voltage cut-off [V]']);upper=float(p['Upper voltage cut-off [V]']);nominal=float(p['Nominal cell capacity [A.h]'])
    reference=None
    if args.protocol=='comsol':
        if args.mode!='stock' or args.parameters!='Marquis2019' or args.ocp!='raw':raise ValueError('COMSOL requires stock/raw/Marquis2019')
        p.update({'Electrode width [m]':1.,'Electrode height [m]':1.,'Negative electrode conductivity [S.m-1]':126.,'Positive electrode conductivity [S.m-1]':16.6,'Current function [A]':24.})
        model=pb.lithium_ion.DFN();sim=pb.Simulation(model,parameter_values=p,var_pts=var_pts,solver=backend)
        reference=json.loads(ws.path('cache','pybamm','v1.0.1','comsol_1C.json').read_text());tref=np.array(reference['time'])
        sol=sim.solve(tref,t_interp=tref) if args.backend=='idaklu' else sim.solve(tref)
        protocol={'completed':bool(sol.t[-1]>=tref[-1]-1e-6),'comparison':'official COMSOL example: 1m² area, 24 A, conductivities 126/16.6 S/m'}
        if not protocol['completed']:raise RuntimeError('COMSOL common time window not completed')
    else:
        discharge=f'Discharge at {args.rate}C until {lower} V'
        steps=[f'Discharge at {args.rate}C for 120 seconds'] if args.protocol=='crosscheck' else [discharge] if args.protocol=='discharge' else [(discharge,'Rest for 10 minutes',f'Charge at 1C until {upper} V',f'Hold at {upper} V until C/20','Rest for 10 minutes')]*args.cycles
        sim=pb.Simulation(model,parameter_values=p,var_pts=var_pts,solver=backend,experiment=pb.Experiment(steps,period=f"{cfg['output_period_s']} seconds"))
        sol=sim.solve()
        if args.protocol=='crosscheck':
            if abs(sol.t[-1]-120)>1e-5:raise RuntimeError('Cross-solver 120s interval incomplete')
            protocol={'completed':True,'scope':'120-second cross-solver check, not full discharge'}
        else:protocol=check_protocol(sol,cycles=1 if args.protocol=='discharge' else args.cycles,steps_per_cycle=1 if args.protocol=='discharge' else 5,lower_V=lower,upper_V=upper,current_A=nominal)
    solved=time.perf_counter();data={'time_s':sol.t.tolist()}
    names={'voltage_V':'Voltage [V]','temperature_K':'Volume-averaged cell temperature [K]','current_A':'Current [A]','lithium_mol':'Total lithium [mol]',
        'capacity_Ah':'Discharge capacity [A.h]','heat_W_m3':'Volume-averaged total heating [W.m-3]','heat_capacity_J_m3K':'Volume-averaged effective heat capacity [J.K-1.m-3]'}
    if stack:names.update({'stack_stress_Pa':'MORQEX stack axial stress [Pa]','thickness_m':'MORQEX sandwich thickness change [m]',
        'mechanical_energy_J_m2':'MORQEX mechanical energy per area [J.m-2]','thermal_residual_J_m3':'MORQEX thermal balance residual [J.m-3]',
        'thermal_storage_J_m3':'MORQEX thermal storage change [J.m-3]','thermal_heating_J_m3':'MORQEX accumulated thermal heating [J.m-3]',
        'thermal_cooling_J_m3':'MORQEX accumulated thermal cooling [J.m-3]','effective_heat_capacity_J_m3K':'MORQEX thermoelastic effective heat capacity [J.K-1.m-3]'})
    for key,name in names.items():
        if name not in sol.all_models[0].variables:continue
        value=np.asarray(sol[name].entries);data[key]=np.broadcast_to(value,sol.t.shape).tolist()
    charge_residuals={}
    for d in ['negative','positive']:
        D=d.capitalize();mean=np.asarray(sol[f'R-averaged {d} particle concentration [mol.m-3]'].entries)
        data[f'{d}_mean_concentration_mol_m3']=np.mean(mean,axis=0).tolist()
        concentration=np.asarray(sol[f'{D} particle concentration [mol.m-3]'].entries)
        cmax=float(p[f'Maximum concentration in {d} electrode [mol.m-3]'])
        data[f'{d}_stoichiometry_range']=[float(concentration.min()/cmax),float(concentration.max()/cmax)]
        volumetric=np.asarray(sol[f'X-averaged {d} electrode volumetric interfacial current density [A.m-3]'].entries)
        integrated=volumetric*float(p[f'{D} electrode thickness [m]'])*float(p['Electrode width [m]'])*float(p['Electrode height [m]'])*float(p['Number of electrodes connected in parallel to make a cell'])
        sign=1 if d=='negative' else -1
        charge_residuals[d]=float(np.max(abs(integrated-sign*np.array(data['current_A'])))/max(1.,max(abs(np.array(data['current_A'])))))
        if stack:
            for short,long in [('hoop_stress_Pa','particle surface tangential stress [Pa]'),('surface_mu_J_mol','surface mechanical chemical potential [J.mol-1]'),('axial_strain','axial strain')]:
                v=np.asarray(sol[f'MORQEX {d} {long}'].entries);data[f'{d}_{short}']=np.mean(v,axis=0).tolist();data[f'{d}_{short}_maxabs']=np.max(abs(v),axis=0).tolist()
            e=stack.electrodes[d];temperature=np.array(data['temperature_K']);q=np.zeros((len(sol.t),8));q[:,0]=data[f'{d}_axial_strain'];q[:,6]=np.mean(mean,axis=0)-e.reference_concentration;q[:,7]=temperature-float(p['Reference temperature [K]'])
            # Volume-fraction correction rescales the complete quadratic energy
            # and its localized stresses together, preserving the average.
            snapshot=np.array([0,int(np.argmax(abs(q[:,6]))),len(sol.t)-1]);stress=e.response.stress(q[snapshot])*e.dfn_active_fraction/e.active_fraction
            np.savez_compressed(out/f'{d}_mean_SVE_snapshots.npz',time_s=sol.t[snapshot],q=q[snapshot],stress_Mandel_Pa=stress,volumes_m3=e.response.volumes,phases=e.response.phases)
            data[f'{d}_mean_SVE_p99_absolute_component_Pa']=[float(np.quantile(abs(s[:,0]),.99)) for s in stress]
    metrics={'end_time_s':float(sol.t[-1]),'end_voltage_V':data['voltage_V'][-1],'temperature_max_K':max(data['temperature_K']),
        'relative_lithium_drift':float(np.ptp(data['lithium_mol'])/data['lithium_mol'][0]),'preparation_s':preparation,'build_and_solve_s':solved-started-preparation,
        'constant_heat_capacity_relative_range':float(np.ptp(data['heat_capacity_J_m3K'])/np.mean(data['heat_capacity_J_m3K'])) if 'heat_capacity_J_m3K' in data else None,
        'relative_electrode_charge_balance_residual':charge_residuals}
    if stack:
        scale=max(1.,max(abs(np.array(data['thermal_heating_J_m3']))),max(abs(np.array(data['thermal_cooling_J_m3']))))
        metrics['thermal_ODE_relative_ledger_residual']=max(abs(np.array(data['thermal_residual_J_m3'])))/scale
        if metrics['constant_heat_capacity_relative_range']>1e-10:raise RuntimeError('Constant-Cp ledger is inapplicable')
    if reference:
        error=np.asarray(sol['Voltage [V]'](tref))-np.asarray(reference['voltage']);metrics.update(comsol_voltage_rmse_V=float(np.mean(error**2)**.5),comsol_voltage_max_error_V=float(max(abs(error))))
        data['reference_time_s']=tref.tolist();data['reference_voltage_V']=reference['voltage']
        metrics['comsol_field_errors']={}
        for key,variable,coordinate in [('c_n_surf','Negative particle surface concentration [mol.m-3]','x_n'),('c_p_surf','Positive particle surface concentration [mol.m-3]','x_p'),('c_e','Electrolyte concentration [mol.m-3]','x'),('phi_n','Negative electrode potential [V]','x_n'),('phi_p','Positive electrode potential [V]','x_p'),('phi_e','Electrolyte potential [V]','x')]:
            exact=np.array(reference[key]);value=np.asarray(sol[variable](t=tref,x=np.array(reference[coordinate])));difference=value-exact
            metrics['comsol_field_errors'][key]={'relative_L2':float(np.linalg.norm(difference)/max(np.linalg.norm(exact),1e-12)),'absolute_max':float(max(abs(difference).ravel()))}
    if source_hashes()!=hashes or provenance()!=backend_sources:raise RuntimeError('Source changed during numerical experiment')
    checks={'protocol_complete':protocol['completed'],'lithium_drift_below_1e-10':metrics['relative_lithium_drift']<1e-10,
        'charge_residual_below_5e-5':max(charge_residuals.values())<5e-5,
        'thermal_ODE_residual_below_1e-6':metrics.get('thermal_ODE_relative_ledger_residual',0.)<1e-6}
    checks={name:bool(value) for name,value in checks.items()}
    result={'schema':'morqex-dfn-result-1','case':args.case,'arguments':vars(args),'parameters':audit,'solver':backend_sources|{'class':type(backend).__name__,'rtol':float(backend.rtol),'atol':float(backend.atol),'root_tol':float(backend.root_tol)},'source_sha256':hashes,
        'protocol':protocol,'numerical_checks':checks,'metrics':metrics,'data':data,'limitations':['Mean-concentration KUBC SVE plus separate isolated-sphere diffusion energy; no arbitrary-particle concentration reconstruction',
        'Synthetic GEM geometry, CBD/alpha/separator assumptions; no material calibration','Thermal ODE ledger only; no full DFN chemical/electrical/elastic energy certificate',
        'Reversible cycles; no degradation or capacity-fade prediction','CPU-only resource-limited validation; no GPU/algorithmic speedup claim']}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'case':args.case,'metrics':metrics,'protocol':protocol,'output':str(out)},indent=2),flush=True)


if __name__=='__main__':main()
