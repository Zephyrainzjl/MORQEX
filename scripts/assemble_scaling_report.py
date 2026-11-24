"""Aggregate independent bounded scaling jobs; no silent stale/failed fallback."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
WS=Path(os.environ['MORQEX_WORKSPACE'])


def main():
    cases=[];runs=[];sources={};cfg=None
    for label in ('sparse_scaling_8','sparse_scaling_14','sparse_scaling_18_104','sparse_scaling_18_205','sparse_scaling_18_306'):
        m=json.loads((WS/'reports'/f'{label}_latest.json').read_text())
        if m['exit_code'] != 0 or m['stopped_because']:raise ValueError(f'Latest {label} failed; retain failure, do not select an old success')
        path=Path(m['run'])/'sparse_scaling.json';data=json.loads(path.read_text());cfg=data['config']
        for name,digest in data['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Stale scaling source: {name}')
            if name in sources and sources[name]!=digest:raise ValueError('Mixed numerical implementations')
            sources[name]=digest
        cases.extend(data['cases']);runs.append({'label':label,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'manifest':m})
    if len(cases)!=9:raise ValueError('Nine independent size/seed cases required')
    sources[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result={'schema':'morqex-sparse-scaling-combined-1','cases':cases,'config':cfg,'runs':runs,'source_sha256':sources,
        'retained_failed_budget_run':json.loads((WS/'reports/sparse_scaling_18_latest.json').read_text()),
        'scope':'fixed quadratic chemistry/linear elasticity and prescribed ports; three new random geometries per size, not fixed-geometry convergence or industrial-scale/DFN qualification'}
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for seed in cfg['seeds']:
        group=sorted([c for c in cases if c['seed']==seed],key=lambda c:c['N']);n=[c['N'] for c in group]
        selected=[next((r for r in c['methods'] if r['basis_family']==c['selected_family'] and r['total_modes']==c['selected_modes']),None) for c in group]
        axes[0].loglog(n,[r['qualified_speedup'] if r else np.nan for r in selected],'o-',label=str(seed))
        axes[1].loglog(n,[c.get('selected_fields',{}).get('relative_stress_L2',np.nan) for c in group],'o-',label=str(seed))
        axes[2].loglog(n,[c.get('selected_fields',{}).get('operator_bytes',np.nan)/2**20 for c in group],'o-',label=f'localization {seed}')
        axes[2].loglog(n,[c['identity_full_stress_hypothetical_bytes']/2**20 for c in group],'--',label=f'identity estimate {seed}')
    axes[0].set_ylabel('qualified online speedup / fastest full solve')
    axes[1].set_ylabel('final-time full-field stress relative L2 error')
    axes[2].set_ylabel('stress operator storage [MiB]')
    for ax in axes:ax.set_xlabel('active concentration DOFs');ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.savefig('sparse_scaling.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for seed in cfg['seeds']:
        c=next(c for c in cases if c['seed']==seed and c['size']==18)
        for family in ('Fick-resolvent','coupled-resolvent'):
            rows=[r for r in c['methods'] if r['basis_family']==family]
            axes[0].semilogy([r['total_modes'] for r in rows],[r['maximum_bound'] for r in rows],'o-' if family=='coupled-resolvent' else 'x--',label=f'{family} {seed}')
        for r in c['methods']:
            if r['basis_family']==c['selected_family'] and r['total_modes']==c['selected_modes']:
                axes[1].semilogy([h['time_s'] for h in r['history']],[max(h['actual_energy_error'],1e-12) for h in r['history']],label=f'actual {seed}')
                axes[1].semilogy([h['time_s'] for h in r['history']],[max(h['bound'],1e-12) for h in r['history']],'--',label=f'bound {seed}')
    axes[0].axhline(cfg['energy_error_tolerance'],color='black',ls=':',label='fixed tolerance')
    axes[0].set_xlabel('total concentration modes incl. Riesz/lifting');axes[0].set_ylabel('maximum whole-energy error bound')
    axes[1].set_xlabel('time [s]');axes[1].set_ylabel('coupled H error [sqrt(J/m3)]')
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.savefig('sparse_certificate.png',dpi=165);plt.close(fig)
    Path('sparse_scaling.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'cases':len(cases),'candidate_rows':sum(len(c['methods']) for c in cases),'data':str(Path.cwd()/'sparse_scaling.json')}))


if __name__=='__main__':main()
