"""Resolved mechanical cycles: history-step refinement and cache ablation."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from morqex import Workspace
from morqex.top_damage.reference_run import run_reference, read_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--provider',default='vendored',choices=['vendored','pymfem-wheel']);p.add_argument('--prefix',default='phase_B_path');args=p.parse_args()
    ws=Workspace(os.environ['MORQEX_WORKSPACE'])
    original=(Path(__file__).resolve().parents[1]/'configs/top_damage_notched_reference.toml').read_text()
    original=original.replace('provider = "vendored"',f'provider = "{args.provider}"')
    base=Path.cwd();cases=[]
    for increments in (8,16,32,64):
        config=base/f'path_{increments}.toml'
        config.write_text(original.replace('increments_per_cycle = 8',f'increments_per_cycle = {increments}')+'\ncache_forms = true\n')
        result=run_reference(config,ws,steps=3*increments+1,output=f'{args.prefix}_{args.provider}_{increments}')
        if not result['complete_specimen_protocol']:raise RuntimeError('Incomplete mechanical protocol')
        records=read_records(result['directory'])
        peak=[records[(2*cycle+1)*increments//2]['mean_stress_yy_MPa'] for cycle in range(3)]
        cases.append({'increments_per_cycle':increments,'directory':result['directory'],
                      'final_mean_damage':records[-1]['mean_damage'],'peak_stress_MPa':peak,
                      'max_kkt':max(r['damage_kkt'] for r in records),
                      'max_equilibrium_residual':max(r['mechanical_residual'] for r in records),
                      'solver_seconds':sum(r['solve_seconds'] for r in records)})
    reference=cases[-1]
    for case in cases:
        case['final_mean_damage_relative_difference']=abs(case['final_mean_damage']/reference['final_mean_damage']-1)
        case['peak_stress_relative_difference']=float(np.linalg.norm(np.array(case['peak_stress_MPa'])-reference['peak_stress_MPa'])/np.linalg.norm(reference['peak_stress_MPa']))
    # Same mesh/protocol/fatigue, reassembled MFEM vs exact geometry cache.
    config=base/'uncached.toml';config.write_text(original.replace('increments_per_cycle = 8','increments_per_cycle = 32')+'\ncache_forms = false\n')
    result=run_reference(config,ws,steps=97,output=f'{args.prefix}_{args.provider}_uncached32')
    uncached=read_records(result['directory']);cached=read_records(cases[2]['directory'])
    difference=max(float(np.max(np.abs(np.array(a['fields']['damage'])-b['fields']['damage']))) for a,b in zip(uncached,cached))
    timing=sum(r['solve_seconds'] for r in uncached)
    summary={'scope':'Three mechanical specimen cycles, NOT battery charge/discharge. Finest path is a numerical comparison, not an exact solution.',
             'cases':cases,'cached_vs_MFEM_max_voxel_damage_difference':difference,
             'cached_solver_speedup_at_32_increments':timing/cases[2]['solver_seconds'],
             'uncached_solver_seconds':timing,'cache_timing_scope':'Solve times only; excludes one-time cache setup and per-step HDF5/VTU/HTML output. Includes the same mechanical and damage solver gates.',
             'path_32_vs_64_under_2percent':max(cases[2]['final_mean_damage_relative_difference'],cases[2]['peak_stress_relative_difference'])<.02,
             'cache_field_gate_passed':difference<1e-7}
    target=ws.path('reports',f'top_damage_path_{args.provider}.json');target.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if not summary['cache_field_gate_passed']:raise RuntimeError('Cache ablation changed physical solution')


if __name__=='__main__':main()
