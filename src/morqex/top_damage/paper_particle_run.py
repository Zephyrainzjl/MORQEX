"""Bounded, resumable paper-equation particle experiments; no DFN claim."""
import hashlib
import json
from pathlib import Path
import time
import tomllib
import numpy as np
from .paper_particle import ParticleMaterial, ParticleState, PaperParticle
from .backend import require_local_budget
from morqex.workspace import Workspace


def quarter_particle_mesh(ell_over_radius, *, crack_h_over_ell=.5, bulk_h_over_radius=.08,
                          seed_h_over_ell=None, initial_crack_fraction=.02):
    """Gmsh quarter circle, graded along the entire prospective ring-crack plane.

    Refine the path, not just the original tip. All dimensions are scaled by R.
    Mesh size is a target; actual geometry/gradient convergence remains required.
    """
    require_local_budget()
    import gmsh
    from skfem import MeshTri
    if not 0<crack_h_over_ell<=.5 or not 0<bulk_h_over_radius<=.2:raise ValueError('Invalid mesh resolution')
    if seed_h_over_ell is not None and not 0<seed_h_over_ell<=crack_h_over_ell:raise ValueError('Invalid initial-crack refinement')
    if not 0<initial_crack_fraction<1:raise ValueError('Invalid seed extent')
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0);gmsh.option.setNumber('General.NumThreads',1)
        gmsh.model.add('ai2022_quarter_particle')
        geo=gmsh.model.geo
        origin=geo.addPoint(0,0,0);right=geo.addPoint(1,0,0);top=geo.addPoint(0,1,0)
        tip=geo.addPoint(1-initial_crack_fraction,0,0)
        bottom=geo.addLine(origin,tip);seed=geo.addLine(tip,right)
        arc=geo.addCircleArc(right,origin,top);left=geo.addLine(top,origin)
        loop=geo.addCurveLoop([bottom,seed,arc,left]);geo.addPlaneSurface([loop]);geo.synchronize()
        field=gmsh.model.mesh.field
        distance=field.add('Distance');field.setNumbers(distance,'CurvesList',[bottom,seed]);field.setNumber(distance,'Sampling',1000)
        threshold=field.add('Threshold');field.setNumber(threshold,'InField',distance)
        field.setNumber(threshold,'SizeMin',ell_over_radius*crack_h_over_ell)
        field.setNumber(threshold,'SizeMax',bulk_h_over_radius)
        field.setNumber(threshold,'DistMin',ell_over_radius)
        field.setNumber(threshold,'DistMax',.06)
        background=threshold
        if seed_h_over_ell is not None:
            near=field.add('Distance');field.setNumbers(near,'CurvesList',[seed]);field.setNumber(near,'Sampling',200)
            refine=field.add('Threshold');field.setNumber(refine,'InField',near)
            field.setNumber(refine,'SizeMin',ell_over_radius*seed_h_over_ell);field.setNumber(refine,'SizeMax',bulk_h_over_radius)
            field.setNumber(refine,'DistMin',ell_over_radius);field.setNumber(refine,'DistMax',4*ell_over_radius)
            background=field.add('Min');field.setNumbers(background,'FieldsList',[threshold,refine])
        field.setAsBackgroundMesh(background)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        gmsh.option.setNumber('Mesh.Algorithm',6)
        gmsh.model.mesh.generate(2)
        tags,points,_=gmsh.model.mesh.getNodes();points=points.reshape(-1,3)
        _,connect=gmsh.model.mesh.getElementsByType(2)
        sorter=np.argsort(tags);cells=sorter[np.searchsorted(tags[sorter],connect)].reshape(-1,3)
        return MeshTri(points[:,:2].T,cells.T)
    finally:gmsh.finalize()


def _state_write(path,state,identity,step):
    import h5py
    temp=path.with_suffix('.pending')
    with h5py.File(temp,'w') as f:
        f.attrs.update(identity=identity,time_s=state.time_s,step=step,complete=True)
        for k in ('u','c','d','history','fatigue','previous_alpha'):f.create_dataset(k,data=getattr(state,k),compression='gzip')
    temp.replace(path)


def _recover_history(path):
    """Repair only a torn final append; never skip a corrupt committed record."""
    if not path.exists():return []
    raw=path.read_text();lines=raw.splitlines(keepends=True);records=[]
    for i,line in enumerate(lines):
        try:record=json.loads(line)
        except json.JSONDecodeError:
            if i!=len(lines)-1 or line.endswith('\n'):raise ValueError('Corrupt committed particle history')
            break
        if record.get('step')!=len(records)+1:raise ValueError('Noncontiguous particle history')
        records.append(record)
    if raw and not raw.endswith('\n'):
        temp=path.with_suffix('.repair')
        temp.write_text(''.join(json.dumps(r)+'\n' for r in records));temp.replace(path)
    return records


def run_particle(config_path,workspace,*,steps=20,resume=False,output=None):
    import h5py
    from skfem import MeshTri
    require_local_budget()
    if type(steps) is not int or steps<1:raise ValueError('Positive bounded steps required')
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    raw=Path(config_path).read_bytes();cfg=tomllib.loads(raw.decode())
    if set(cfg)!={'schema_version','material','experiment','mesh'} or cfg['schema_version']!=1:raise ValueError('Paper particle schema 1 required')
    material=ParticleMaterial(**cfg['material']);experiment=cfg['experiment']
    allowed={'name','geometry','c_rate','cycles','steps_per_half_cycle','initial_soc','crack_fraction','fatigue','stress_feedback','save_stride','newton_tolerance','damage_constraint','linearization','boundary_policy','concentration_domain','phase_form','toughness_floor','seed_width_over_length'}
    if set(experiment)-allowed:raise ValueError('Unknown experiment setting')
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('paper_particle.py'))}
    import importlib.metadata
    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-fem','gmsh')}
    identity=hashlib.sha256(json.dumps(dict(config=cfg,source=source,versions=versions),sort_keys=True).encode()).hexdigest()
    name=output or experiment['name']
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):raise ValueError('Invalid output name')
    directory=ws.path('results','top_damage','paper_reproduction',name);directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'run.json';checkpoint=directory/'checkpoint.h5';records_path=directory/'history.jsonl'
    if manifest.exists():
        if not resume or json.loads(manifest.read_text())['identity']!=identity:raise ValueError('Explicit resume and matching source/config required')
        data=np.load(directory/'mesh.npz');mesh=MeshTri(data['points'],data['cells'])
    else:
        if resume:raise ValueError('No checkpoint to resume')
        mesh=quarter_particle_mesh(material.length_m/material.radius_m,**cfg['mesh'])
        np.savez_compressed(directory/'mesh.npz',points=mesh.p,cells=mesh.t)
        manifest.write_text(json.dumps(dict(identity=identity,config=cfg,source=source,versions=versions,
            scope='Independent paper-equation qualification; published curves not yet reproduced',
            differences=['P1 FE with explicit quadrature; author public model uses higher order',
                         'Backward Euler with exact monolithic tangent; paper uses BDF',
                         'Sharp switching resolved at every half-cycle; paper smooths 0.1% of cycle',
                         'Phase equation: '+experiment.get('phase_form','paper_divergence')+'; paper divergence and author Helmholtz differ for spatially varying toughness',
                         'Optional primal-dual active-set damage bounds, distinct from the paper history-only method',
                         'Boundary policy: '+experiment.get('boundary_policy','fixed_flux')+'; surface complementarity is an independent closure, not verified author cutoff',
                         'Concentration domain: '+experiment.get('concentration_domain','bounded')+'; diagnostic excesses never qualify as physical acceptance',
                         'Author crosswind transport stabilization is not implemented'],
            full_paper_reproduction_passed=False,dfn_feedback_passed=False),indent=2))
        (directory/'config.toml').write_bytes(raw)
    options={k:experiment[k] for k in ('geometry','fatigue','stress_feedback','damage_constraint','linearization','boundary_policy','concentration_domain','phase_form','toughness_floor','seed_width_over_length') if k in experiment}
    model=PaperParticle(mesh,material,**options)
    if checkpoint.exists():
        with h5py.File(checkpoint,'r') as f:
            if not f.attrs['complete'] or f.attrs['identity']!=identity:raise ValueError('Checkpoint identity mismatch')
            state=ParticleState(**{k:f[k][:] for k in ('u','c','d','history','fatigue','previous_alpha')},time_s=float(f.attrs['time_s']))
            index=int(f.attrs['step'])
        # The checkpoint is committed before its metric record. Recover a
        # missing last record using the adjacent transactional record file.
        records=_recover_history(records_path)
        pending=directory/'last_record.json'
        if (not records or records[-1]['step']<index) and pending.exists():
            record=json.loads(pending.read_text())
            if record['step']!=index:raise ValueError('Incomplete checkpoint transaction')
            with records_path.open('a') as stream:stream.write(json.dumps(record)+'\n')
        elif records and records[-1]['step']!=index:raise ValueError('History/checkpoint mismatch')
    else:
        state=model.initial_state(experiment.get('initial_soc',.9),experiment.get('crack_fraction',.02));index=0
        _state_write(checkpoint,state,identity,index);_state_write(directory/'frame_000000.h5',state,identity,index)
    rate=float(experiment['c_rate']);cycles=int(experiment['cycles']);resolution=int(experiment['steps_per_half_cycle'])
    if rate<=0 or cycles<1 or resolution<4:raise ValueError('Invalid cycling protocol')
    half=.7*3600/rate;nominal_dt=half/resolution;end=2*cycles*half
    flux=material.cmax_mol_m3*material.radius_m*(model.volume/model.surface_area)*rate/3600
    for _ in range(steps):
        if state.time_s>=end-1e-8:break
        half_index=int(np.floor((state.time_s+1e-8)/half));next_event=min((half_index+1)*half,end)
        dt=min(nominal_dt,next_event-state.time_s);sign=1 if half_index%2==0 else -1
        start=time.perf_counter();failures=[]
        for cutback in range(15):
            try:
                trial,record=model.advance(state,dt,sign*flux,tolerance=experiment.get('newton_tolerance',1e-9))
                break
            except RuntimeError as exc:
                failures.append(dict(dt_s=dt,reason=str(exc)));dt*=.5
        else:
            (directory/'failure.json').write_text(json.dumps(dict(step=index+1,time_s=state.time_s,attempts=failures),indent=2))
            raise RuntimeError('All bounded cutbacks failed; accepted checkpoint preserved')
        index+=1;state=trial
        record.update(step=index,time_s=state.time_s,cycle=state.time_s/(2*half),dt_s=dt,outward_flux_mol_m2_s=sign*flux,
            seconds=time.perf_counter()-start,cutbacks=failures,nodes=model.n,triangles=mesh.t.shape[1])
        (directory/'last_record.json').write_text(json.dumps(record))
        _state_write(checkpoint,state,identity,index)
        if index%experiment.get('save_stride',resolution)==0 or abs(state.time_s-next_event)<1e-8:
            _state_write(directory/f'frame_{index:06d}.h5',state,identity,index)
        with records_path.open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    return dict(directory=str(directory),step=index,cycle=state.time_s/(2*half),complete_protocol=state.time_s>=end-1e-8,
                full_paper_reproduction_passed=False)
