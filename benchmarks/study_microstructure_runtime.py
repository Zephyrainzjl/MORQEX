"""Measured live forward/backward ports, restart, bound and nested enrichment."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import tomllib
from types import SimpleNamespace
import numpy as np
from study_microstructure_dfn import prepare,source_hashes,ROOT
from morqex import Workspace
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
from morqex.electrochemistry.microstructure_runtime import MicrostructureDFNRuntime
from morqex.electrochemistry.microstructure_validation import thermodynamic_observables
from morqex.solvers.pybamm_backend import provenance

def main():
    ws=Workspace(Path(os.environ['MORQEX_WORKSPACE']));output=ws.path('results','microstructure_runtime');output.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes()
    for path in [ROOT/'src/morqex/electrochemistry/microstructure_runtime.py',Path(__file__)]:hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    cfg=tomllib.loads((ROOT/'configs/microstructure_dfn.toml').read_text());p,audit=microstructure_parameters()
    for d in ['Negative','Positive']:
        name=f'{d} electrode partial molar volume [m3.mol-1]';p.update({name:p[name]*cfg['partial_molar_volume_scale']})
    paths={};offline={}
    for r in [4,12]:
        args=SimpleNamespace(shape=4,seed=105,positive_seed=None,refine=1,regions=r,strategy='geometry',case=f'live{r}')
        paths[r],_,offline[r]=prepare(ws,p,args,cfg)
    runtime=MicrostructureDFNRuntime(paths[4],p,nx=3);records=[];metrics=[];bounds=[]
    assert not runtime._localizers
    # Discharge before the charge pulse: the reference state starts near 100% SOC.
    for dt,current in [(600.,5.),(20.,5.),(20.,0.),(20.,0.),(20.,-2.5),(20.,-2.5),(20.,5.)]:
        record=runtime.advance(dt,current);assert record['completed'];records.append(record)
        metrics.append(thermodynamic_observables(runtime.solution,runtime.stack,runtime.parameters)['metrics'])
        for d in ['negative','positive']:
            for x in range(3):
                b=runtime.reaction_bound(d,x_index=x)
                assert not runtime._localizers
                bounds.append({'domain':d,'x':x,'time_s':record['time_s'],'bound':b['absolute_bound_A_m3'],
                    'relative_to_1C':b['relative_to_1C'],'wall_seconds':b['wall_seconds']})
    # Compact bound queries above must not load localization or fine mesh arrays.
    cold=runtime.recover('negative');warm=[runtime.recover('negative')['wall_seconds'] for _ in range(10)]
    roi=np.arange(0,len(cold['stress']),4);regional=runtime.recover('negative',selection=roi)
    np.testing.assert_allclose(regional['stress'],cold['stress'][roi],atol=1e-9)
    for b in bounds[-6:]:
        defect=runtime.estimate_defect(b['domain'],x_index=b['x'],lookahead_s=30.)
        b.update(actual_defect=defect['absolute_reaction_port_defect_A_m3'],empirical_concentration_defect=defect['relative_concentration_defect'])
        assert b['actual_defect']<=b['bound']*(1+1e-10)+1e-6
    checkpoint=runtime.checkpoint(ws.path('checkpoints','microstructure_live'))
    restored=MicrostructureDFNRuntime(paths[4],p,nx=3).restore(checkpoint)
    a=runtime.advance(20.,5.);b=restored.advance(20.,5.)
    restart_voltage=abs(a['voltage_V']-b['voltage_V']);restart_temperature=abs(a['temperature_K']-b['temperature_K'])
    del restored;gc.collect();old={d:runtime.recover(d) for d in ['negative','positive']}
    before=thermodynamic_observables(runtime.solution,runtime.stack,runtime.parameters)
    upgraded,adapt=runtime.enrich_if_needed(paths[12],relative_port_tolerance=1e-9,concentration_defect_tolerance=.002,lookahead_s=30.)
    assert adapt['triggered'];transfer={}
    for d in ['negative','positive']:
        new=upgraded.recover(d)
        transfer[d]={'concentration_max_absolute':float(np.max(abs(new['concentration']-old[d]['concentration']))),
            'stress_max_absolute_Pa':float(np.max(abs(new['stress']-old[d]['stress'])))}
    after=thermodynamic_observables(upgraded.solution,upgraded.stack,upgraded.parameters)
    energy_jump=float(abs(after['internal_energy_J'][-1]-before['internal_energy_J'][-1]))
    entropy_jump=float(abs(after['entropy_J_K'][-1]-before['entropy_J_K'][-1]))
    enriched_records=[upgraded.advance(20.,5.) for _ in range(4)]
    checks={'completed_steps':all(r['completed'] for r in records+enriched_records),'restart':restart_voltage<1e-9 and restart_temperature<1e-9,
        'nested_concentration':max(x['concentration_max_absolute'] for x in transfer.values())<1e-8,
        'nested_stress':max(x['stress_max_absolute_Pa'] for x in transfer.values())<1e-4,
        'nested_energy_entropy':energy_jump<1e-7 and entropy_jump<1e-8,
        'energy':max(m['relative_internal_energy_residual'] for m in metrics)<1e-6,
        'entropy':max(m['relative_entropy_residual'] for m in metrics)<1e-5,
        'port_bound_coverage':all(b['actual_defect']<=b['bound']*(1+1e-10)+1e-6 for b in bounds if 'actual_defect' in b)}
    result={'schema':'morqex-live-ports-1','source_sha256':hashes,'backend':provenance(),'parameter_audit':audit,'offline_seconds':offline,
        'steps':records,'enriched_steps':enriched_records,'recovery_cold_seconds':cold['wall_seconds'],
        'recovery_warm_seconds':warm,'roi_recovery_seconds':regional['wall_seconds'],'bounds':bounds,'adaptivity':adapt,
        'transfer':transfer,'energy_jump_J':energy_jump,'entropy_jump_J_K':entropy_jump,
        'restart_voltage_difference_V':restart_voltage,'restart_temperature_difference_K':restart_temperature,
        'thermodynamic_metrics':metrics,'checks':checks,
        'scope':'warm measured soft real-time latency, no deadline guarantee; local BV closure bound, not nonlinear trajectory bound'}
    for name,value in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))
    if not all(checks.values()):raise SystemExit('Live mapping validation failed')

if __name__=='__main__':main()
