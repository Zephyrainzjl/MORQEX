"""Python API for upstream echemAMR's AMReX finite-volume engine.

    AMR Cartesian finite volumes and conforming FEM are distinct discretizations.
    Both consume the same Volume; no tetrahedron-to-AMR approximation is hidden.
"""
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import numpy as np

def backend_info():
    """Report the installed native build, independently of numerical validation."""
    import _gem_echemamr
    if hasattr(_gem_echemamr,'capabilities'):return dict(_gem_echemamr.capabilities())
    return {'capabilities_unavailable':True,'note':'Rebuild native extension for capability metadata'}

def assemble_cell(anode,cathode,separator_voxels,anode_map,cathode_map):
    """Assemble cathode|separator|anode along x, matching upstream boundary conventions."""
    from morqex.microstructure.volume import Volume
    if anode.data.shape[:2] != cathode.data.shape[:2] or not np.allclose(anode.spacing,cathode.spacing,rtol=1e-9,atol=0):
        raise ValueError('Incompatible transverse dimensions or voxel spacing')
    if separator_voxels < 1: raise ValueError('Separator requires >=1 voxel')
    def mapped(v,m,allowed):
        if not np.isin(v.data,list(m)).all() or not set(m.values()) <= allowed:
            raise ValueError('Incomplete or invalid electrode phase mapping')
        out=np.zeros(v.data.shape,np.uint8)
        for a,b in m.items():out[v.data == a]=b
        return out
    a=mapped(anode,anode_map,{1,2,3});c=mapped(cathode,cathode_map,{5,6,7})
    s=np.full((*a.shape[:2],separator_voxels),4,np.uint8)
    return Volume(np.concatenate((c,s,a),axis=2),anode.spacing,
                  metadata={'assembly_axis':'x','ordering':'cathode|separator|anode','echem_phase_ids':True})

def prepare_case(volume,directory,phase_map,parameters=None,template=None):
    """Write geometry, complete upstream Microstructure input template, and overrides."""
    from importlib.resources import files
    d=Path(directory).resolve();d.mkdir(parents=True,exist_ok=True)
    paths=export_geometry(volume,d,phase_map)
    source=Path(template).read_text() if template else files('morqex.microstructure').joinpath('resources/echem_microstructure.inputs').read_text()
    inputfile=d/'inputs'
    extent=(np.asarray(volume.data.shape)*volume.spacing)[::-1]
    overrides={
        # ASCII relative paths keep upstream narrow-stream I/O portable on Windows,
        # including workspaces whose parent path contains Chinese characters.
        'prob.geometry_filename':paths['geometry'].name,
        'prob.nanoporosity_filename':paths['nanoporosity'].name,
        'prob.macmullin_filename':paths['macmullin'].name,
        'geometry.prob_lo':[0.,0.,0.], 'geometry.prob_hi':extent.tolist(),
        'amr.n_cell':list(volume.data.shape[::-1]),'amr.blocking_factor':1,
        'echemamr.use_hypre':0,
        # AMReX otherwise reserves ~75% of the entire GPU at initialization.
        'amrex.the_arena_init_size':67108864,
    }
    overrides.update(parameters or {})
    extra=AMRConfig(overrides).write(d/'gem_overrides.inputs').read_text()
    # Replace active definitions rather than duplicate ParmParse keys.
    keys=set(overrides)
    source='\n'.join(line for line in source.splitlines() if line.split('=',1)[0].strip() not in keys)
    inputfile.write_text(source+'\n# GEM explicit overrides\n'+extra)
    volume.save(d/'input_volume.npz')
    return inputfile

def export_geometry(volume, directory, phase_map, nanoporosity=1., macmullin=1.):
    """Upstream x-fastest format: nx ny nz then phase labels.

    echem phase IDs: 1 anode, 2 anode electrolyte, 3 anode CBD,
    4 separator, 5 cathode, 6 cathode electrolyte, 7 cathode CBD.
    Explicit map prevents interpreting arbitrary CT labels as physics domains.
    """
    d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    a = volume.data
    if not np.isin(np.unique(a), list(phase_map)).all() or not set(phase_map.values()) <= set(range(1,8)):
        raise ValueError('Map every CT label explicitly to an echem phase ID 1..7')
    labels = np.zeros(a.shape, np.int32)
    for old,new in phase_map.items(): labels[a == old] = new
    paths = dict(geometry=d/'geometry.txt', nanoporosity=d/'nanoporosity.txt', macmullin=d/'macmullin.txt')
    with paths['geometry'].open('w') as f:
        f.write(' '.join(map(str,a.shape[::-1]))+'\n')
        np.savetxt(f,labels.ravel(),fmt='%d')
    for name,value in [('nanoporosity',nanoporosity),('macmullin',macmullin)]:
        array = np.broadcast_to(value,a.shape)
        if not np.isfinite(array).all() or np.min(array) <= 0:
            raise ValueError(f'{name} must be finite and positive')
        if name == 'nanoporosity' and np.max(array) > 1:
            raise ValueError('nanoporosity <=1')
        np.savetxt(paths[name],array.ravel(),fmt='%.17g')
    return paths

@dataclass
class AMRConfig:
    parameters: dict = field(default_factory=dict)

    def write(self,path):
        p = Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        def encode(v):
            if isinstance(v,(list,tuple)): return ' '.join(map(str,v))
            if isinstance(v,str): return '"'+v.replace('"','')+'"'
            if isinstance(v,bool): return str(int(v))
            return str(v)
        p.write_text('\n'.join(f'{k} = {encode(v)}' for k,v in self.parameters.items())+'\n',encoding='utf8')
        return p

def run(inputs, workdir, timeout=None, overrides=None):
    """Native Python binding runs in an isolated process (AMReX global lifecycle)."""
    inputs = Path(inputs).resolve()
    workdir = Path(workdir).resolve();workdir.mkdir(parents=True,exist_ok=True)
    if inputs.parent != workdir:
        import shutil
        staged=workdir/'gem_inputs'
        shutil.copy2(inputs,staged)
        inputs=staged
    args = [inputs.name] + [f'{k}={v}' for k,v in (overrides or {}).items()]
    import json
    command = [sys.executable,'-c','import _gem_echemamr,json,sys; _gem_echemamr.run(json.loads(sys.argv[1]))',json.dumps(args)]
    with (workdir/'solver.log').open('w',encoding='utf8') as log:
        subprocess.run(command,cwd=workdir,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
    return workdir/'solver.log'

@dataclass(frozen=True)
class DischargeSettings:
    c_rate:float=1.
    duration:float=3600.
    timestep:float=2.
    cutoff_voltage:float=3.
    anode_soc:float=.9
    cathode_soc:float=.4
    temperature:float=298.15
    nonlinear_tolerance:float=1e-8
    linear_tolerance:float=1e-10
    max_level:int=0

@dataclass
class DischargeCurve:
    time_s:np.ndarray
    voltage_V:np.ndarray
    current_A:np.ndarray
    capacity_Ah:np.ndarray
    source:Path
    diagnostics:dict=field(default_factory=dict)

    def plot(self):
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        ax.plot(self.time_s,self.voltage_V)
        ax.set_xlabel('Time [s]');ax.set_ylabel('Terminal voltage [V]');fig.tight_layout()
        return fig

def read_discharge(directory):
    from scipy.integrate import cumulative_trapezoid
    path=Path(directory)/'discharge.csv'
    a=np.atleast_1d(np.genfromtxt(path,names=True,delimiter=','))
    if not all(np.isfinite(a[k]).all() for k in ('time_s','voltage_V','current_A')):
        raise RuntimeError('Nonfinite native discharge result; inspect solver.log')
    capacity=cumulative_trapezoid(abs(a['current_A']),a['time_s'],initial=0)/3600
    return DischargeCurve(a['time_s'],a['voltage_V'],a['current_A'],capacity,path,
                          audit_solver(directory))


def audit_solver(directory):
    """Read actual convergence/mass evidence; never infer convergence from a curve."""
    import re
    directory=Path(directory);log=directory/'solver.log'
    if not log.exists():return {'status':'log unavailable','nonlinear_convergence_confirmed':False}
    text=log.read_text(encoding='utf8',errors='replace')
    values=re.findall(r'BV NON-LINEAR RESIDUAL \(rel,abs\):\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)',text)
    last=tuple(map(float,values[-1])) if values else None
    errors={}
    for phase in ('Anode','Cathode','Electrolyte'):
        hits=re.findall(phase+r'\s*:\s*current[^\n]*?error\s*=\s*([\d.eE+\-]+)\s*%',text)
        if hits:errors[phase]=float(hits[-1])
    # Defaults match DischargeSettings. Read actual case tolerance overrides.
    tolerance=1e-8;absolute_tolerance=1e-12
    inputs=directory/'inputs'
    if inputs.exists():
        hits=re.findall(r'^\s*echemamr.bv_nonlinear_reltol\s*=\s*([\d.eE+\-]+)',inputs.read_text(),re.M)
        if hits:tolerance=float(hits[-1])
        hits=re.findall(r'^\s*echemamr.bv_nonlinear_abstol\s*=\s*([\d.eE+\-]+)',inputs.read_text(),re.M)
        if hits:absolute_tolerance=float(hits[-1])
    failed=any(marker in text for marker in ('Failed to converge','Failing to converge','amrex::Abort','SIGSEGV','SIGABRT'))
    # A later converged solve cannot conceal an earlier unconverged state.
    starts=text.count('MORQEX BV BEGIN')
    accepted=re.findall(r'MORQEX BV ACCEPT\s+(\S+)\s+(\S+)',text)
    try:
        all_accepted=starts>0 and starts==len(accepted) and all(
            np.isfinite([float(r),float(a)]).all() and (float(r)<=tolerance or float(a)<=absolute_tolerance)
            for r,a in accepted)
    except ValueError:all_accepted=False
    return dict(status='solver failure' if failed else 'run completed',
        nonlinear_convergence_confirmed=bool(all_accepted and not failed),
        nonlinear_solves_started=starts,nonlinear_solves_accepted=len(accepted),
        last_nonlinear_relative_residual=last[0] if last else None,
        last_nonlinear_absolute_residual=last[1] if last else None,
        requested_nonlinear_relative_tolerance=tolerance,mass_error_percent=errors,
        interpretation='A finite curve and small step-to-step updates do not certify nonlinear or mesh convergence')

def discharge_3d(anode,cathode,directory,anode_map,cathode_map,
                 separator_voxels=4,settings=None,timeout=None,parameters=None):
    """Fully coupled upstream microstructure-scale discharge; SI units required.

    Default kinetics/transport use upstream real-material functions. Mesh, time
    step and nonlinear tolerances remain part of the required convergence study.
    """
    cfg=settings or DischargeSettings()
    if cfg.c_rate<=0 or cfg.duration<=0 or cfg.timestep<=0:
        raise ValueError('Positive rate, duration and timestep required')
    cell=assemble_cell(anode,cathode,separator_voxels,anode_map,cathode_map)
    params={
        'max_step':int(np.ceil(cfg.duration/cfg.timestep)), 'stop_time':cfg.duration,
        # Upstream Microstructure uses positive current for charging.
        # GEM's positive c_rate is discharge, therefore reverse at this boundary.
        'prob.Crate':-cfg.c_rate,'prob.soc_anode_t0':cfg.anode_soc,
        'prob.soc_cathode_t0':cfg.cathode_soc,'prob.Temperature':cfg.temperature,
        'echemamr.fixed_timestep':1,'echemamr.dtmin':cfg.timestep,'echemamr.dtmax':cfg.timestep,
        'echemamr.gem_voltage_cutoff':cfg.cutoff_voltage,
        'echemamr.bv_nonlinear_reltol':cfg.nonlinear_tolerance,
        'echemamr.linsolve_reltol':cfg.linear_tolerance,
        'amr.max_level':cfg.max_level,'amr.plot_int':-1,'amr.line_plot_int':-1,
        'prob.mater_anode':2,'prob.mater_cathode':2,
        'prob.mater_electrolyte':2,'prob.mater_electrolyte_tpAc':2,
    }
    params.update(parameters or {})
    inputs=prepare_case(cell,directory,{i:i for i in range(1,8)},parameters=params)
    run(inputs,directory,timeout=timeout)
    return read_discharge(directory)
