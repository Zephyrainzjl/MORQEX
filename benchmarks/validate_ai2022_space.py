"""Memory-bounded, common-domain P1 field comparison on two particle meshes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import h5py
import numpy as np
from matplotlib.tri import Triangulation
from skfem import MeshTri
from morqex.workspace import Workspace
from morqex.top_damage.paper_particle import PaperParticle, ParticleMaterial
from morqex.top_damage.backend import require_local_budget


def compare(coarse, fine, step):
    require_local_budget()
    paths=[Path(coarse),Path(fine)]
    meta=[json.loads((p/'run.json').read_text()) for p in paths]
    source=Path(__file__).parents[1]/'src/morqex/top_damage/paper_particle.py'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if any(m['source']['paper_particle.py']!=digest for m in meta):raise ValueError('Solver revision mismatch')
    if any(meta[0]['config'][key]!=meta[1]['config'][key] for key in ('material','experiment')):
        raise ValueError('Spatial comparison must retain material and protocol')
    meshes=[np.load(p/'mesh.npz') for p in paths];a,b=meshes
    material=ParticleMaterial(**meta[0]['config']['material'])
    model=PaperParticle(MeshTri(a['points'],a['cells']),material,geometry=meta[0]['config']['experiment']['geometry'])
    finder=Triangulation(*b['points'],triangles=b['cells'].T).get_trifinder()
    points=model.quadrature_coordinates
    element=finder(*points);valid=element>=0
    # The two polygonal approximations can have different boundaries. Report
    # the excluded coarse volume rather than extrapolating an unknown field.
    excluded=float(model.w[~valid].sum()/model.volume)
    if excluded>.001:raise ValueError('Common domain loses more than 0.1% of volume')
    triangles=b['cells'][:,element[valid]].T
    vertices=b['points'][:,triangles].transpose(1,2,0)
    dx=points[:,valid].T-vertices[:,0]
    v1=vertices[:,1]-vertices[:,0];v2=vertices[:,2]-vertices[:,0]
    det=v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0]
    l1=(dx[:,0]*v2[:,1]-dx[:,1]*v2[:,0])/det
    l2=(v1[:,0]*dx[:,1]-v1[:,1]*dx[:,0])/det
    shape=np.column_stack((1-l1-l2,l1,l2))
    if shape.min()<-1e-8 or shape.max()>1+1e-8:raise ValueError('Invalid triangle localization')
    w=model.w[valid];rms=lambda x:np.sqrt(w@(x*x)/w.sum())
    errors={}
    with h5py.File(paths[0]/f'frame_{step:06d}.h5') as h,h5py.File(paths[1]/f'frame_{step:06d}.h5') as g:
        if abs(h.attrs['time_s']-g.attrs['time_s'])>1e-7:raise ValueError('Snapshot times differ')
        for state,identity in ((h,meta[0]['identity']),(g,meta[1]['identity'])):
            if not state.attrs['complete'] or state.attrs['identity']!=identity:raise ValueError('Snapshot identity mismatch')
        for key in ('c','d'):
            u=(model.Q@h[key][:])[valid];v=np.sum(g[key][:][triangles]*shape,axis=1)
            errors[key+'_L2_relative']=float(rms(u-v)/max(rms(v),1e-30))
        end=float(h.attrs['time_s'])
    histories=[[json.loads(line) for line in (p/'history.jsonl').read_text().splitlines()] for p in paths]
    return dict(coarse=str(coarse),fine=str(fine),time_s=end,errors=errors,
        excluded_coarse_volume_fraction=excluded,
        peak_soc=[max(r['max_soc'] for r in history if r['time_s']<=end+1e-7) for history in histories],
        physical_concentration_passed=all(r['physical_concentration_passed'] for history in histories for r in history if r['time_s']<=end+1e-7),
        spatial_convergence_passed=False,full_paper_reproduction_passed=False,
        note='Two-grid comparison only, not an asymptotic order or certified error bound.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('coarse');parser.add_argument('fine')
    parser.add_argument('--step',type=int,default=80)
    args=parser.parse_args();result=compare(args.coarse,args.fine,args.step)
    ws=Workspace(os.environ['MORQEX_WORKSPACE'])
    target=ws.path('reports','ai_spatial_comparison.json');target.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
