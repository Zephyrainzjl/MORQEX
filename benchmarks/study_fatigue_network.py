"""Reproducible staged study; all outputs external, failures remain visible.

Run via scripts/run_local.py. Cases share explicit numerical mechanics inputs;
cross-chemistry runs validate software interfaces, not calibrated material life.
"""
import argparse,json,os,re,time,traceback
from pathlib import Path
from morqex.workspace import Workspace
from morqex.top_damage.network_cycle import run_network_cycle
from morqex.top_damage.network_report import build_network_report

CASES={
    'rate01':{'C_rate':.1},'rate05':{'C_rate':.5},'rate2':{'C_rate':2.},
    'rate5':{'C_rate':5.},'rate8':{'C_rate':8.},
    'seed105':{'C_rate':.5,'seed':105,'grain_seed':20261128},
    'seed106':{'C_rate':.5,'seed':106,'grain_seed':20261129},
    'nmc532':{'host':'NMC532'},'lfp':{'host':'LFP'},
    'time48':{'increments_per_half_cycle':48},'time96':{'increments_per_half_cycle':96},
    'no_feedback':{'topology_feedback':False},'exact_graph':{'graph_update_tolerance':0.},
    'long20':{'cycles':20},
    'regions64':{'cycles':1,'concentration_regions':64},
    'regions128':{'cycles':1,'concentration_regions':128},
    'regions_full':{'cycles':1,'concentration_regions':-1},
    'regions_grain':{'cycles':1},
    'rate8_symmetric':{'C_rate':8.,'charge_C_rate':8.},
    # Separate case keeps the failed NMC-sized/LFP-window attempt visible.
    # Prada2013 uses a 50 nm active radius and D=5.9e-18 m2/s; do not silently
    # combine that D with micron NMC particles. Mechanics remain uncalibrated.
    'lfp_matched':{'host':'LFP','stoichiometry_window':[.01,.99],
        'spacing_m':2e-8,'length_m':4e-8,'gradient_coefficient_J_m5_mol2':0.},
    'seed106_time48':{'seed':106,'grain_seed':20261129,'increments_per_half_cycle':48},
    'regions_full_local':{'cycles':1,'concentration_regions':-1,'local_mobility':True},
    'regions64_local':{'cycles':1,'concentration_regions':64,'local_mobility':True},
    'regions128_local':{'cycles':1,'concentration_regions':128,'local_mobility':True},
    'regions_grain_local':{'cycles':1,'local_mobility':True},
    'regions_full_batched':{'cycles':1,'concentration_regions':-1,'local_mobility':True},
    'regions256_local':{'cycles':1,'concentration_regions':256,'local_mobility':True},
    'long20_256':{'cycles':20,'concentration_regions':256,'local_mobility':True},
    'long20_full':{'cycles':20,'concentration_regions':-1,'local_mobility':True},
    'full8_time24':{'cycles':1,'C_rate':8.,'charge_C_rate':8.,'concentration_regions':-1,'local_mobility':True},
    'full8_time48':{'cycles':1,'C_rate':8.,'charge_C_rate':8.,'concentration_regions':-1,'local_mobility':True,'increments_per_half_cycle':48},
    'full8_time96':{'cycles':1,'C_rate':8.,'charge_C_rate':8.,'concentration_regions':-1,'local_mobility':True,'increments_per_half_cycle':96},
    'rate8_no_mechanics':{'C_rate':8.,'charge_C_rate':8.,'mechanics_enabled':False},
    'charge_gate_probe':{'cycles':1,'concentration_regions':256,'local_mobility':True},
    'long20_checked':{'cycles':20,'concentration_regions':256,'local_mobility':True},
    'full_projected':{'cycles':1,'concentration_regions':-1,'local_mobility':True,'project_algebraic_output':True,'dfn_rtol':1e-8,'dfn_atol':1e-9},
    'full8_projected':{'cycles':1,'C_rate':8.,'charge_C_rate':8.,'concentration_regions':-1,'local_mobility':True,'project_algebraic_output':True,'dfn_rtol':1e-8,'dfn_atol':1e-9},
    'full8_projected48':{'cycles':1,'C_rate':8.,'charge_C_rate':8.,'concentration_regions':-1,'local_mobility':True,'project_algebraic_output':True,'dfn_rtol':1e-8,'dfn_atol':1e-9,'increments_per_half_cycle':48},
    'full8_projected96':{'cycles':1,'C_rate':8.,'charge_C_rate':8.,'concentration_regions':-1,'local_mobility':True,'project_algebraic_output':True,'dfn_rtol':1e-8,'dfn_atol':1e-9,'increments_per_half_cycle':96},
    'chemical_rate01':{'cycles':1,'C_rate':.1,'charge_C_rate':.1,'concentration_regions':-1,'local_mobility':True,'diffusivity_interpretation':'chemical_diffusivity','gradient_coefficient_J_m5_mol2':0.},
    'chemical_rate8':{'cycles':1,'C_rate':8.,'charge_C_rate':8.,'concentration_regions':-1,'local_mobility':True,'diffusivity_interpretation':'chemical_diffusivity','gradient_coefficient_J_m5_mol2':0.},
}


def write_case(name,ws):
    raw=(Path(__file__).parents[1]/'configs/top_damage_network_cycle.toml').read_text()
    values={'output':'dual_'+name+'_v1',**CASES[name]}
    for key,value in values.items():
        literal=json.dumps(value)
        pattern=r'(?m)^'+re.escape(key)+r'\s*=.*$'
        if re.search(pattern,raw):raw=re.sub(pattern,key+' = '+literal,raw)
        else:raw=raw.replace('[cycle]','[cycle]\n'+key+' = '+literal)
    target=ws.path('experiments/fatigue_network',name+'.toml');target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_text()!=raw:raise ValueError('Study configuration changed; version the study before overwriting')
    target.write_text(raw);return target


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',choices=CASES,required=True);p.add_argument('--steps',type=int,default=150)
    args=p.parse_args();ws=Workspace(os.environ['MORQEX_WORKSPACE']);results=[];failed=False
    for name in args.cases:
        start=time.perf_counter();config=write_case(name,ws)
        try:
            state=run_network_cycle(config,ws,steps=args.steps)
            report=build_network_report(state['directory'],ws,output='MORQEX_dual_'+name+'.html')
            result={'case':name,'state':'complete' if report['protocol_complete'] else 'accepted_checkpoint',
                'report':report,'seconds':time.perf_counter()-start}
        except Exception as exc:
            # Symbolic solvers can attach enormous DAGs as exception arguments
            # or chained contexts. Never expand those into multi-GB messages.
            frames=traceback.extract_tb(exc.__traceback__)
            result={'case':name,'state':'failed','error_type':type(exc).__name__,
                'error_args':[a[:2000] if isinstance(a,str) else type(a).__name__ for a in exc.args],
                'traceback':traceback.format_list(frames[-30:]),'seconds':time.perf_counter()-start};failed=True
        path=ws.path('experiments/fatigue_network',name+'_status.json');path.write_text(json.dumps(result,indent=2))
        results.append(result);print(json.dumps(result),flush=True)
        if failed:break
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
