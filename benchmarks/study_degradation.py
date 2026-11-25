"""One budget-limited CT/cohesive/SEI/plating case; all evidence is external."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pybamm as pb
from morqex import Workspace
from morqex.ct import SegmentedCT
from morqex.gemsim import GEMsim
from morqex.topology import PorousCBD
from morqex.bases import GeometricBasis,PortBasis,ExactBasis,VariationalOperator
from morqex.extensions import build_operator
from morqex.cohesive import build_cohesive
from morqex.degradation import DegradationOptions,DiffusionLimitedSEI,PlatingStrippingDeadLi,EnergyDrivenDebonding,NoDamage
from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode
from morqex.electrochemistry.microstructure_dfn import MicrostructureStack,key
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
from morqex.electrochemistry.microstructure_validation import thermodynamic_observables
from morqex.electrochemistry.aging_dfn import build_aging_dfn,age
from morqex.electrochemistry.aging_cache import save_aging_operator,load_aging_kernel,AgingLocalization
from morqex.solvers.pybamm_backend import solver,provenance

ROOT=Path(__file__).resolve().parents[1]

def hashes():
    files=[*ROOT.joinpath('src/morqex/electrochemistry').glob('*.py'),*ROOT.joinpath('src/morqex/mechanics').glob('*.py'),
        *ROOT.joinpath('src/morqex').glob('*.py'),Path(__file__)]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def prepare(ws,p,args):
    sources=json.loads(ws.path('data','ct','sources.json').read_text());crops=json.loads(ws.path('data','ct','selected_crops.json').read_text());records={};paths={}
    for d in ['negative','positive']:
        start=time.perf_counter();info=crops[d];s=sources[info['dataset']]
        crop=tuple((int((a+b)//2-args.crop_size//2),int((a+b)//2-args.crop_size//2+args.crop_size)) for a,b in info['crop'])
        data=SegmentedCT(Path(s['path']),s['sha256'],(s['voxel_size_m'],)*3,{0:'pore',1:'active',2:'cbd'},axes_to_zyx=tuple(info['axes_to_zyx']),crop=crop,source_url=s['source_page']).build(ws)
        f=MicrostructureElectrode(data,GEMsim(ws),p,d,topology=PorousCBD())
        basis={'geometry':GeometricBasis(),'port':PortBasis(),'exact':ExactBasis()}[args.basis]
        op=build_operator(f,basis,VariationalOperator(),max(args.regions,f.diffusion.n_components));coh=build_cohesive(f,op.prolongation)
        paths[d]=save_aging_operator(ws.path('operators','aging',args.case,d),op,coh)
        records[d]={'data':info|{'actual_crop':crop},'operator':op.metadata(),'cohesive':coh.metadata,'offline_seconds':time.perf_counter()-start,'source':s}
        del f,op,coh;gc.collect()
    return paths,records

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',required=True);parser.add_argument('--regions',type=int,default=4)
    parser.add_argument('--basis',choices=['geometry','port','exact'],default='port');parser.add_argument('--mechanisms',choices=['all','none','sei','plating','damage'],default='all')
    parser.add_argument('--protocol',choices=['short','cycles','pulse'],default='short');parser.add_argument('--cycles',type=int,default=3);parser.add_argument('--rate',type=float,default=1.)
    parser.add_argument('--temperature',type=float,default=298.15);parser.add_argument('--damage-rate',type=float,default=2e-6);parser.add_argument('--damage-threshold',type=float,default=2e4)
    parser.add_argument('--duration',type=float,default=120.)
    parser.add_argument('--lower-voltage',type=float,default=2.5);parser.add_argument('--upper-voltage',type=float,default=4.2)
    parser.add_argument('--initial-plated',type=float,default=0.)
    parser.add_argument('--initial-state',choices=['reference','mid'],default='reference');parser.add_argument('--thin-electrodes',action='store_true')
    parser.add_argument('--crop-size',type=int,choices=[4,6,8],default=4)
    parser.add_argument('--rtol',type=float,default=1e-7);parser.add_argument('--atol',type=float,default=1e-8)
    args=parser.parse_args();start_hashes=hashes();ws=Workspace(Path(os.environ['MORQEX_WORKSPACE']));out=ws.path('results','aging',args.case);out.mkdir(parents=True,exist_ok=True)
    p,audit=microstructure_parameters();p.update({'Initial temperature [K]':args.temperature,'Ambient temperature [K]':args.temperature})
    for D in ['Negative','Positive']:
        k=f'{D} electrode partial molar volume [m3.mol-1]';p.update({k:p[k]*.1})
    if args.initial_state=='mid':
        for d,sto in [('negative',.6),('positive',.5)]:p.update({f'Initial concentration in {d} electrode [mol.m-3]':sto*p[f'Maximum concentration in {d} electrode [mol.m-3]']})
    if args.thin_electrodes:p.update({'Negative electrode thickness [m]':1e-5,'Positive electrode thickness [m]':1e-5})
    audit['scope']='Explicit modified convex/compact chemistry, Omega x0.1; degradation laws configured separately. Not calibrated CT chemistry or lifetime.'
    audit['initial_conditions']={d:p[f'Initial concentration in {d} electrode [mol.m-3]'] for d in ['negative','positive']}
    paths,geometry=prepare(ws,p,args);kernels={d:load_aging_kernel(path) for d,path in paths.items()}
    config=DegradationOptions(negative=tuple([DiffusionLimitedSEI()] if args.mechanisms in ['all','sei'] else [])+tuple([PlatingStrippingDeadLi(initial_plated_mol_m3=args.initial_plated)] if args.mechanisms in ['all','plating'] else []),
        damage=EnergyDrivenDebonding(rate_s=args.damage_rate,threshold_J_m3=args.damage_threshold) if args.mechanisms in ['all','damage'] else NoDamage())
    stack=MicrostructureStack(kernels['negative'].operator,kernels['positive'].operator,p);stack.update_parameters(p)
    model=build_aging_dfn(stack,kernels,config)
    steps=[f'Discharge at {args.rate}C for {args.duration} seconds'] if args.protocol=='short' else [(f'Discharge at {args.rate}C until {args.lower_voltage} V','Rest for 300 seconds',f'Charge at {args.rate}C until {args.upper_voltage} V',f'Hold at {args.upper_voltage} V until C/20','Rest for 300 seconds')]*args.cycles
    if args.protocol=='pulse':steps=[(f'Charge at {args.rate}C for {args.duration} seconds','Rest for 1200 seconds','Discharge at 0.1C for 60 seconds')]
    sim=pb.Simulation(model,parameter_values=p,solver=solver(rtol=args.rtol,atol=args.atol),experiment=pb.Experiment(steps,period='30 seconds'),var_pts={**model.default_var_pts,'x_n':3,'x_s':3,'x_p':3,'r_n':2,'r_p':2})
    start=time.perf_counter();sol=sim.solve();elapsed=time.perf_counter()-start
    if args.protocol=='short' and abs(sol.t[-1]-args.duration)>1e-6:raise ValueError('Short protocol incomplete')
    if args.protocol=='cycles':
        from morqex.electrochemistry.validation import check_protocol
        try:protocol=check_protocol(sol,cycles=args.cycles,steps_per_cycle=5,lower_V=args.lower_voltage,upper_V=args.upper_voltage,current_A=float(p['Nominal cell capacity [A.h]']),rest_s=300.)
        except RuntimeError:
            (out/'rejected_protocol.json').write_text(json.dumps({'arguments':vars(args),'complete':False,'end_time_s':float(sol.t[-1]),
                'end_voltage_V':float(sol['Voltage [V]'].entries[-1]),'termination':sol.termination,'source_sha256':start_hashes},indent=2));raise
    elif args.protocol=='pulse':
        if len(sol.cycles)!=1 or len(sol.cycles[0].steps)!=3 or any(abs(float(s.t[-1]-s.t[0])-t)>1e-3 or 'MORQEX' in s.termination for s,t in zip(sol.cycles[0].steps,[args.duration,1200.,60.])):raise ValueError('Pulse/rest protocol incomplete')
        protocol={'completed':True,'steps':[{'duration_s':float(s.t[-1]-s.t[0]),'termination':s.termination} for s in sol.cycles[0].steps]}
    else:protocol={'completed':True}
    previous=None;jump=0.
    state_names=[]
    for d,op in stack.electrodes.items():
        state_names.append(age(d,'damage'))
        for k in range(op.dimension):
            state_names.append(key(d,k,'concentration [mol.m-3]'))
            state_names.extend(age(d,r.name+' [mol.m-3]',k) for law in getattr(config,d) for r in law.reservoirs())
    for cycle in sol.cycles:
        for step in cycle.steps:
            if previous is not None:
                for name in state_names:
                    a=np.asarray(step[name].entries)[...,0];b=np.asarray(previous[name].entries)[...,-1]
                    jump=max(jump,float(np.max(abs(a-b))/max(1.,np.max(abs(b)))))
            previous=step
    # Independent host state free energy; add explicitly owned side reservoirs.
    energy=thermodynamic_observables(sol,stack,p);side_stock=np.zeros(len(sol.t));side_energy=np.zeros(len(sol.t));reservoirs={};micro={}
    A=float(p['Electrode width [m]'])*float(p['Electrode height [m]'])*float(p['Number of electrodes connected in parallel to make a cell'])
    for d,op in stack.electrodes.items():
        m={};rates=[]
        for law in getattr(config,d):
            for r in law.reservoirs():
                stock=np.stack([sol[age(d,r.name+' [mol.m-3]',k)].entries for k in range(op.dimension)],axis=-1)
                from morqex.electrochemistry.microstructure_validation import macro_weights
                wx=macro_weights(sol[age(d,r.name+' [mol.m-3]',0)]);total=A*stack.lengths[d]*np.einsum('x,xtk->t',wx,stock)
                side_stock+=total;side_energy+=r.chemical_potential_J_mol*total;reservoirs[d+'_'+r.name]=total.tolist();m[r.name]=stock.tolist()
                rates.extend(float(np.max(abs(sol[age(d,'reservoir balance residual [mol.m-3.s-1]',k)].entries))) for k in range(op.dimension))
        concentrations=np.stack([sol[key(d,k,'concentration [mol.m-3]')].entries for k in range(op.dimension)],axis=-1);strain=np.asarray(sol[f'MORQEX {d} axial strain'].entries);damage=np.asarray(sol[age(d,'damage')].entries)
        m.update(concentration=concentrations.tolist(),strain=strain.tolist(),damage=damage.tolist(),kernel_path=str(paths[d]),
            release=np.asarray(sol[age(d,'energy release [J.m-3]')].entries).tolist())
        micro[d]=m
    lithium=np.asarray(sol['Total lithium [mol]'].entries)+side_stock;U=energy['internal_energy_J']+side_energy;work=np.asarray(sol['MORQEX accumulated electrical work [J]'].entries);cool=np.asarray(sol['MORQEX accumulated cooling work [J]'].entries)
    residual=U-U[0]-(work-work[0])-(cool-cool[0]);metrics=energy['metrics']
    metrics['relative_lithium_including_reservoirs_drift']=float(np.ptp(lithium)/lithium[0]);metrics['relative_total_internal_energy_residual']=float(np.max(abs(residual))/max(1,np.max(abs(work-work[0]))));metrics['build_solve_seconds']=elapsed
    metrics['damage_end']={d:float(np.max(m['damage'])) for d,m in micro.items()};metrics['end_time_s']=float(sol.t[-1]);metrics['reservoir_end_mol']={k:v[-1] for k,v in reservoirs.items()}
    I=np.asarray(sol['Current [A]'].entries);charge={}
    for d in ['negative','positive']:
        integrated=A*stack.lengths[d]*np.asarray(sol[f'Sum of x-averaged {d} electrode volumetric interfacial current densities [A.m-3]'].entries)
        charge[d]=float(np.max(abs(integrated-(1 if d=='negative' else -1)*I))/max(1.,np.max(abs(I))))
    metrics['charge_residual']=charge
    metrics['differential_state_step_jump']=jump
    metrics['reservoir_max_mol']={k:float(np.max(v)) for k,v in reservoirs.items()}
    cycle_metrics=[]
    if args.protocol=='cycles':
        for i,cycle in enumerate(sol.cycles):
            s=cycle.steps[0];cycle_metrics.append({'cycle':i+1,'discharge_Ah':float(np.trapezoid(s['Current [A]'].entries,s.t)/3600),
                'end_time_s':float(cycle.t[-1]),'damage':{d:float(np.max(cycle[age(d,'damage')].entries[:,-1])) for d in ['negative','positive']}})
    metrics['cycle_metrics']=cycle_metrics
    checks={'protocol':True,'lithium':metrics['relative_lithium_including_reservoirs_drift']<1e-9,'energy':metrics['relative_total_internal_energy_residual']<1e-5,
        'entropy':metrics['relative_entropy_residual']<1e-5,'dissipation':metrics['minimum_dissipation_W']>=-1e-7,
        'local_power':max(metrics['port_identity_relative_residual'].values())<1e-8,
        'nonnegative_reservoirs':all(np.min(v)>=-1e-9 for m in micro.values() for k,v in m.items() if k in ['sei','plating','dead_lithium']),
        'damage_irreversible':all(np.min(np.diff(np.asarray(m['damage']),axis=1))>=-1e-9 for m in micro.values()),'charge':max(charge.values())<5e-5,'state_continuity':jump<1e-8}
    curves={'time_s':sol.t.tolist(),'voltage_V':sol['Voltage [V]'].entries.tolist(),'current_A':sol['Current [A]'].entries.tolist(),
        'temperature_K':sol['Volume-averaged cell temperature [K]'].entries.tolist(),'thickness_m':sol['MORQEX sandwich thickness change [m]'].entries.tolist(),
        'lithium_total_mol':lithium.tolist(),'internal_energy_residual_J':residual.tolist(),'entropy_residual_J_K':energy['entropy_residual_J_K'].tolist()}
    result={'schema':'morqex-aging-study-1','case':args.case,'arguments':vars(args),'source_sha256':start_hashes,'backend':provenance(),
        'geometry':geometry,'config':asdict(config),'parameter_audit':audit,'protocol':protocol,'metrics':metrics,'checks':{k:bool(v) for k,v in checks.items()},
        'curves':curves,'reservoirs_mol':reservoirs,'microstructure':micro,
        'scope':'Tiny CT-derived SVEs, reconstructed CBD, assumed microporosity/cohesive/side-law parameters. Not calibrated degradation or RVE-scale validation.'}
    if hashes()!=start_hashes:raise ValueError('Source changed during run')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'case':args.case,'metrics':metrics,'checks':checks},indent=2),flush=True)
    if not all(checks.values()):raise SystemExit('Degradation numerical gate failed')

if __name__=='__main__':main()
