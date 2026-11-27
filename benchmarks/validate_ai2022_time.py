"""Bounded time-refinement study; convergence does not waive concentration gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import h5py
from morqex.workspace import Workspace
from morqex.top_damage.paper_particle_run import run_particle
from morqex.top_damage.paper_particle import PaperParticle, ParticleMaterial
from skfem import MeshTri


def compare(coarse, fine):
    paths=[Path(coarse),Path(fine)]
    meta=[json.loads((p/'run.json').read_text()) for p in paths]
    a,b=[np.load(p/'mesh.npz') for p in paths]
    for k in ('points','cells'):
        if not np.array_equal(a[k],b[k]):raise ValueError('Time refinement must use identical mesh')
    solver=Path(__file__).parents[1]/'src/morqex/top_damage/paper_particle.py'
    digest=hashlib.sha256(solver.read_bytes()).hexdigest()
    if any(m['source']['paper_particle.py']!=digest for m in meta):raise ValueError('Solver revision mismatch')
    configs=json.loads(json.dumps([m['config'] for m in meta]))
    for cfg in configs:
        cfg['experiment'].pop('steps_per_half_cycle');cfg['experiment'].pop('name')
    if configs[0]!=configs[1]:raise ValueError('Physical configurations differ')
    model=PaperParticle(MeshTri(a['points'],a['cells']),ParticleMaterial(**meta[0]['config']['material']))
    with h5py.File(paths[0]/'checkpoint.h5') as h, h5py.File(paths[1]/'checkpoint.h5') as g:
        if abs(h.attrs['time_s']-g.attrs['time_s'])>1e-7:raise ValueError('End times differ')
        c0,c1=h['c'][:],g['c'][:];d0,d1=h['d'][:],g['d'][:]
        w=model.lump
        rms=lambda x:float(np.sqrt(w@(x*x)/w.sum()))
        error=dict(concentration_L2_relative=rms(c0-c1)/rms(c1),
            concentration_Linf_absolute=float(np.max(np.abs(c0-c1))),
            damage_L2_relative=rms(d0-d1)/rms(d1),
            crack_fraction_absolute=abs(model.cracked_fraction(d0)-model.cracked_fraction(d1)),
            fatigue_L2_relative=float(np.sqrt(model.w@((h['fatigue'][:]-g['fatigue'][:])**2)) /
                max(np.sqrt(model.w@(g['fatigue'][:]**2)),1e-30)))
    histories=[[json.loads(line) for line in (p/'history.jsonl').read_text().splitlines()] for p in paths]
    return dict(coarse=str(coarse),fine=str(fine),errors=error,
        end_cycle=histories[1][-1]['cycle'],physical_concentration_passed=all(
            r['physical_concentration_passed'] for history in histories for r in history),
        peak_soc=[max(r['max_soc'] for r in history) for history in histories],
        max_mass_error=[max(r['mass_error'] for r in history) for history in histories],
        full_paper_reproduction_passed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resolution',type=int,required=True)
    p.add_argument('--coarse',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();ws=Workspace(os.environ['MORQEX_WORKSPACE'])
    if args.resolution not in (80,160):raise ValueError('Bounded two-cycle qualification only')
    template=Path(__file__).parents[1]/'configs/top_damage_ai2022_author_controls.toml'
    raw=template.read_text().replace('steps_per_half_cycle = 40',f'steps_per_half_cycle = {args.resolution}')
    cfg=ws.path('configs',f'ai_time_{args.resolution}.toml');cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(raw)
    result=run_particle(cfg,ws,steps=4*args.resolution,output=args.output)
    comparison=compare(args.coarse,result['directory'])
    target=ws.path('reports',f'ai_time_{args.resolution}.json');target.write_text(json.dumps(comparison,indent=2))
    print(json.dumps(comparison,indent=2))
