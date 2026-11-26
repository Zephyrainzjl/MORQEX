"""Bounded, source-tracked resolved 3D echemAMR reference backend.

This is separate from the DFN closure. The qualified interface rejects porous
CBD dual potentials and AMR flattening: those upstream paths are incomplete.
It does not silently substitute NMC532 for a requested MORQEX host material.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,sysconfig,time,tomllib
import numpy as np
from morqex import Workspace
from .backend import require_local_budget

NATIVE=Path(__file__).resolve().parents[1]/'microstructure/simulate/native'

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_restart_geometry(volume,labels,checkpoint):
    """A native restart retains its material mesh; it is not topology remapping."""
    from morqex.microstructure import Volume
    checkpoint=Path(checkpoint).resolve();root=checkpoint.parent
    if not (checkpoint/'Header').is_file() or not (root/'input_volume.npz').is_file() or not (root/'geometry.txt').is_file():
        raise ValueError('Restart requires its original checkpoint and geometry metadata')
    original=Volume.load(root/'input_volume.npz')
    if original.data.shape!=volume.data.shape or not np.array_equal(original.spacing,volume.spacing) or not np.array_equal(original.origin,volume.origin):
        raise ValueError('Restart lattice changed; explicit conservative state remapping is required')
    with (root/'geometry.txt').open() as handle:
        shape=tuple(map(int,handle.readline().split()))
        native=np.fromfile(handle,dtype=np.int64,count=labels.size+1,sep=' ')
    if shape!=labels.shape[::-1] or native.size!=labels.size or not np.array_equal(native,labels.ravel()):
        raise ValueError('Restart phase topology changed; native checkpoints cannot silently remap material')
    return checkpoint

def check_potential_connectivity(labels):
    """Reject floating domains; never add weak conductors or delete material.

    Graph edges are conducting phase contacts or active/electrolyte BV faces.
    Dry CBD/electrolyte faces block current. The x-high solid boundary fixes the
    voltage gauge in this native model. This is a necessary geometry check,
    not a substitute for actual nonlinear/linear convergence.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    labels=np.asarray(labels)
    if labels.ndim!=3 or not np.isin(labels,range(1,8)).all():raise ValueError('Explicit native phase IDs 1..7 on a 3D lattice required')
    ids=np.arange(labels.size).reshape(labels.shape);ionic=np.isin(labels,[2,4,6]);cbd=np.isin(labels,[3,7]);edges=[]
    for axis in range(3):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None);a=tuple(a);b=tuple(b)
        valid=~((cbd[a]&ionic[b])|(ionic[a]&cbd[b]))
        edges.append(np.column_stack([ids[a][valid],ids[b][valid]]))
    pairs=np.concatenate(edges);graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(labels.size,labels.size)).tocsr()
    count,components=connected_components(graph,directed=False)
    anchors=np.unique(components.reshape(labels.shape)[:,:,-1][~ionic[:,:,-1]])
    missing=np.setdiff1d(np.arange(count),anchors)
    if len(missing):raise ValueError(f'{len(missing)} floating potential components require explicit gauge/inactive-domain treatment; geometry was not modified')
    return {'potential_components':int(count),'anchored_components':len(anchors)}

@dataclass(frozen=True)
class ResolvedBuildConfig:
    device: str = 'cpu'
    precision: str = 'float64'
    jobs: int = 1
    slice_seconds: int = 55
    hypre: bool = False

    def __post_init__(self):
        if self.device!='cpu' or self.precision!='float64' or self.jobs!=1:
            raise ValueError('Resolved reference is qualified only for serial CPU FP64')
        if type(self.slice_seconds) is not int or not 5<=self.slice_seconds<=55:
            raise ValueError('Build slices must be 5..55 seconds')
        if type(self.hypre) is not bool:raise ValueError('Explicit Boolean HYPRE selection required')

    @classmethod
    def read(cls,path):
        d=tomllib.loads(Path(path).read_text())
        if set(d)!={'schema_version','resolved'} or d['schema_version']!=1:raise ValueError('Resolved build TOML schema 1 required')
        return cls(**d['resolved'])

def source_identity():
    files={str(p.relative_to(NATIVE)):_sha(p) for p in sorted(NATIVE.rglob('*')) if p.is_file() and (p.suffix in {'.cpp','.H','.h','.json'} or p.name in {'CMakeLists.txt','LICENSE'})}
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files

def build_resolved(config,workspace):
    """Resume single-job compilation outside src; each call has a finite slice."""
    import pybind11
    require_local_budget();cfg=config if isinstance(config,ResolvedBuildConfig) else ResolvedBuildConfig.read(config)
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    directory=ws.path('build/echemamr_cpu_candidate');directory.mkdir(parents=True,exist_ok=True)
    source,files=source_identity();tick=time.monotonic()
    command=['cmake','-S',str(NATIVE),'-B',str(directory),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
      '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF','-DCMAKE_MAKE_PROGRAM='+str(Path(sys.executable).parent/'ninja'),
      '-Dpybind11_DIR='+pybind11.get_cmake_dir(),'-DPYTHON_EXECUTABLE='+sys.executable,
      '-DAMReX_GPU_BACKEND=NONE','-DAMReX_PRECISION=DOUBLE','-DAMReX_MPI=OFF','-DAMReX_OMP=OFF',
      '-DAMReX_HYPRE='+('ON' if cfg.hypre else 'OFF')]
    hypre=ws.path('build/hypre-2.32.0-install')
    if cfg.hypre:
        from .hypre_dependency import build_hypre
        dependency=build_hypre(ws,slice_seconds=cfg.slice_seconds)
        if dependency['state']!='complete':return {'state':'dependency_pending','dependency':dependency}
        if '#define HYPRE_SEQUENTIAL 1' not in (hypre/'include/HYPRE_config.h').read_text():raise ValueError('Serial HYPRE build required')
        command.extend(['-DHYPRE_INCLUDE_DIRS='+str(hypre/'include'),'-DHYPRE_LIBRARIES='+str(hypre/'lib/libHYPRE.a'),'-DCMAKE_PREFIX_PATH='+str(hypre)])
    with (directory/'configure.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    receipt=directory/'morqex_build.json';receipt.write_text(json.dumps({'state':'building','source_identity':source}))
    with (directory/'morqex_build.log').open('a') as log:
        proc=subprocess.Popen([str(Path(sys.executable).parent/'ninja'),'-C',str(directory),'-j1'],stdout=log,stderr=subprocess.STDOUT)
        try:code=proc.wait(timeout=cfg.slice_seconds)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT);proc.wait(timeout=15);state='pending'
        else:state='complete' if code==0 else 'failed'
    library=directory/('_gem_echemamr'+sysconfig.get_config_var('EXT_SUFFIX'))
    result={'state':state,'config':asdict(cfg),'source_identity':source,'source_files':files,'directory':str(directory),
      'library':str(library),'library_sha256':_sha(library) if state=='complete' else None,'seconds_this_slice':time.monotonic()-tick,
      'python':sys.version,'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
      'amrex_commit':'9ed35ab0e58462f951528462376cc379ba834826','numerical_qualification':False}
    result['dependency_license_sha256']=_sha(directory/'_deps/amrex-src/LICENSE')
    if cfg.hypre:result.update(hypre_library=str(hypre/'lib/libHYPRE.a'),hypre_library_sha256=_sha(hypre/'lib/libHYPRE.a'))
    receipt.write_text(json.dumps(result,indent=2))
    if state=='failed':raise RuntimeError('Native build failed; inspect '+str(directory/'morqex_build.log'))
    return result

class ResolvedElectrochemistry:
    """Isolated native lifecycles, complete fields, explicit material provenance."""
    def __init__(self,workspace):
        require_local_budget();self.ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
        self.directory=self.ws.path('build/echemamr_cpu_candidate')
        self.receipt=json.loads((self.directory/'morqex_build.json').read_text())
        self._verify()

    def _verify(self):
        r=self.receipt
        if r['state']!='complete' or r['source_identity']!=source_identity()[0] or _sha(r['library'])!=r['library_sha256']:
            raise ValueError('Missing, stale or modified resolved electrochemistry backend')
        if r.get('hypre_library') and _sha(r['hypre_library'])!=r['hypre_library_sha256']:raise ValueError('HYPRE dependency changed')

    def _child(self,code,arguments,log,timeout=120):
        self._verify();env=dict(os.environ);env['PYTHONPATH']=str(self.directory)+os.pathsep+env.get('PYTHONPATH','')
        with Path(log).open('w') as handle:
            subprocess.run([sys.executable,'-c',code,*map(str,arguments)],env=env,cwd=Path(log).parent,
                           stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=timeout)

    def run(self,volume,directory,phase_map,parameters,*,restart=None,timeout=120):
        """Run one signed-current segment; native positive Crate means charging.

        Materials remain upstream NMC532/graphite or its explicit test variants.
        This API does not claim custom chemistry, phase-field or lifetime closure.
        """
        from morqex.microstructure.simulate.electrochemistry import prepare_case,read_discharge
        directory=self.ws.path(str(directory))
        if directory.exists() and (directory/'inputs').exists():raise ValueError('Choose a new segment directory; never overwrite accepted evidence')
        p=dict(parameters)
        required={'max_step','stop_time','prob.Crate','echemamr.dtmin','echemamr.dtmax'}
        if not required<=set(p):raise ValueError('Explicit time horizon, timestep and signed native Crate required')
        if not np.isfinite([p[k] for k in required]).all() or int(p['max_step'])!=p['max_step'] or p['max_step']<1 or p['stop_time']<=0 or not 0<p['echemamr.dtmin']<=p['echemamr.dtmax']:raise ValueError('Admissible finite time and current parameters required')
        if p.get('echemamr.use_hypre',0) and not self.receipt['config']['hypre']:raise ValueError('HYPRE requested but not compiled')
        fixed={'geometry.is_periodic':[0,0,0],'echemamr.lo_bc_spec':[2,2,2],'echemamr.hi_bc_spec':[2,2,2],
               'echemamr.lo_bc_pot':[6,2,2],'echemamr.hi_bc_pot':[5,2,2],
               'echemamr.bound_specden':0,'echemamr.reset_species_in_solid':0,'echemamr.interface_update_iters':0,'amr.max_level':0,'prob.CBD_transport':0,'prob.ELEisCBD':0,
               'echemamr.morqex_potential_offset':1,'echemamr.morqex_sync_potential':1,'echemamr.morqex_conservative_migration':1,'echemamr.morqex_strict_concentration_bounds':1,'echemamr.potential_solve':1,'echemamr.potential_solve_int':1,
               'echemamr.species_implicit_solve':1,'echemamr.mechanics_solve':0}
        for key,value in fixed.items():
            if key in p and not np.array_equal(p[key],value):raise ValueError('Unqualified resolved reference option: '+key)
            p[key]=value
        if not np.isin(volume.data,list(phase_map)).all():raise ValueError('Explicit mapping for every voxel phase required')
        if np.any(np.asarray(volume.origin)!=0):raise ValueError('Resolved reference requires explicit zero-origin coordinates')
        labels=np.zeros(volume.data.shape,np.int8)
        for a,b in phase_map.items():labels[volume.data==a]=b
        connectivity=check_potential_connectivity(labels)
        if not np.isin(labels[:,:,-1],[1,3]).all():raise ValueError('Offset reference requires an anode/CBD x-high collector and no cathode Dirichlet face')
        if restart is not None:p['amr.restart']=str(check_restart_geometry(volume,labels,restart))
        inputs=prepare_case(volume,directory,phase_map,p)
        manifest={'schema':'morqex-resolved-segment-1','native_source':self.receipt['source_identity'],
          'binary_sha256':self.receipt['library_sha256'],'inputs_sha256':_sha(inputs),
          'geometry_sha256':_sha(directory/'input_volume.npz'),'parameters':p,'state':'running','connectivity':connectivity,
          'scope':'Resolved three-dimensional electrochemistry; no implicit phase-field/topology feedback'}
        destination=directory/'morqex_segment.json';destination.write_text(json.dumps(manifest,indent=2))
        try:
            self._child('import _gem_echemamr,sys; _gem_echemamr.run([sys.argv[1]])',[inputs.name],directory/'solver.log',timeout)
            curve=read_discharge(directory)
            if not curve.diagnostics['nonlinear_convergence_confirmed']:raise RuntimeError('Every native nonlinear solve must have accepted residual evidence')
        except Exception as exc:
            manifest.update(state='failed',failure=str(exc));destination.write_text(json.dumps(manifest,indent=2));raise
        manifest.update(state='complete',diagnostics=curve.diagnostics);destination.write_text(json.dumps(manifest,indent=2));return curve

    def export_fields(self,plot,destination,variables,*,maximum_voxels=1000000):
        """Exact FP64 single-level AMReX -> ZYX HDF5 (no interpolation)."""
        import h5py
        if type(maximum_voxels) is not int or maximum_voxels<1:raise ValueError('Explicit positive voxel budget required')
        destination=self.ws.path(str(destination));destination.parent.mkdir(parents=True,exist_ok=True)
        code='''import _gem_echemamr,json,sys,h5py,numpy as np
from pathlib import Path
r=_gem_echemamr.read_plot(sys.argv[1],json.loads(sys.argv[3]),int(sys.argv[4]))
p=Path(sys.argv[2]);temporary=p.with_suffix('.pending')
with h5py.File(temporary,'w') as f:
 for k,v in r.items():
  if k!='fields':f.attrs[k]=v
 for k,a in r['fields'].items():
  if not np.isfinite(a).all():raise ValueError('Nonfinite native field: '+k)
  f.create_dataset(k,data=a,compression='gzip')
 f.attrs['schema']='morqex-resolved-fields-1';f.attrs['axis_order']='ZYX'
temporary.replace(p)
'''
        self._child(code,[Path(plot).resolve(),destination,json.dumps(list(variables)),maximum_voxels],destination.with_suffix('.log'))
        with h5py.File(destination) as f:
            return {'path':str(destination),'time_s':float(f.attrs['time_s']),'shape':list(f[next(iter(f))].shape),'fields':list(f)}
