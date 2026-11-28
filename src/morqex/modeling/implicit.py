"""Feature-aware CGAL meshing of fixed-grid continuous multiphase geometry."""
from dataclasses import dataclass,fields,replace
from pathlib import Path
import ctypes,hashlib,json,os,subprocess,sys,tempfile,time
import numpy as np
from scipy import ndimage
from .preparation import SmoothingOptions,PreparedGeometry,smooth_voxels
from .features import analyze_features
from .progress import progress_stage,TimingHistory


@dataclass(frozen=True)
class ImplicitMeshOptions:
    precision:str='balanced'
    surface_size_m:float|None=None
    bulk_size_m:float|None=None
    outer_surface_size_m:float|None=None
    facet_distance_m:float|None=None
    surface_minimum_angle_degrees:float=25.
    oracle_tolerance_voxels:float=.01
    gradation:float=.35
    grading:str='surface_driven'
    radius_edge_limit:float=2.
    native_radius_edge_ratio:float=1.5
    minimum_dihedral_degrees:float=3.
    optimization_target_degrees:float=3.5
    optimization:str='joint'
    optimization_seconds:float=60.
    maximum_phase_volume_error:float=.08
    maximum_box_volume_error:float=.03
    maximum_nodes:int=1500000
    maximum_cells:int=8000000
    initial_random_points:int=100
    seed_separation_voxels:float=.02
    seed:int=20261128
    timeout_seconds:float=300.
    maximum_quality_retries:int=1
    maximum_local_repair_cycles:int=2
    require_quality:bool=True
    require_preparation_fidelity:bool=True
    isolated:bool=True
    size_field:object=None

    def __post_init__(self):
        if self.precision not in ('coarse','balanced','fine','very_fine'):raise ValueError('Unknown implicit mesh precision')
        for name in ('surface_size_m','bulk_size_m','outer_surface_size_m','facet_distance_m'):
            value=getattr(self,name)
            if value is not None and (not np.isfinite(value) or value<=0):raise ValueError('Positive finite SI mesh sizes required')
        for name in ('gradation','radius_edge_limit','native_radius_edge_ratio','optimization_seconds','timeout_seconds'):
            value=getattr(self,name)
            if not np.isfinite(value) or value<0 or (name!='optimization_seconds' and value==0):raise ValueError('Finite positive mesh controls required')
        if self.radius_edge_limit<1 or not 1<=self.native_radius_edge_ratio<=self.radius_edge_limit:raise ValueError('Native radius-edge refinement must be >=1 and no looser than acceptance limit')
        if not np.isfinite(self.surface_minimum_angle_degrees) or not 0<=self.surface_minimum_angle_degrees<30:raise ValueError('Surface refinement angle in[0,30) required;0 disables this separate surface criterion')
        if not np.isfinite(self.oracle_tolerance_voxels) or not 0<self.oracle_tolerance_voxels<=.1:raise ValueError('Geometry-oracle absolute tolerance in(0,0.1] voxel required')
        if not 0<=self.minimum_dihedral_degrees<=20 or not 1<=self.optimization_target_degrees<=20:raise ValueError('Valid dihedral goals required')
        if self.optimization not in ('none','perturb','exude','perturb_exude','joint','joint_exude'):raise ValueError('Unknown mesh optimizer')
        if self.grading not in ('surface_driven','distance'):raise ValueError('grading is surface_driven or distance')
        if not 0<self.maximum_phase_volume_error<1 or not 0<self.maximum_box_volume_error<1:raise ValueError('Positive geometry-volume tolerances required')
        for name in ('maximum_nodes','maximum_cells','initial_random_points'):
            if type(getattr(self,name)) is not int or not 1<=getattr(self,name)<2**31:raise ValueError('Positive bounded native budgets required')
        if type(self.seed) is not int or not 0<=self.seed<2**32:raise ValueError('Unsigned32-bit seed required')
        if not np.isfinite(self.seed_separation_voxels) or not 0<=self.seed_separation_voxels<=.25:raise ValueError('Initialization sample separation in0..0.25 voxel required')
        if type(self.maximum_quality_retries) is not int or not 0<=self.maximum_quality_retries<=3:raise ValueError('Quality retries in0..3 required')
        if type(self.maximum_local_repair_cycles) is not int or not 0<=self.maximum_local_repair_cycles<=5:raise ValueError('Local quality repair cycles in0..5 required')


def recommend_mesh_sizes(geometry,*,progress=None):
    """Broad BULK range plus thin-feature guidance, not a safe uniform bound."""
    model=geometry.model if isinstance(geometry,PreparedGeometry) else geometry
    report=analyze_features(model,progress=progress)
    h=float(min(model.spacing));length=float(min(np.array(model.regions.shape)*model.spacing))
    report['recommended_sizes'].update({
        'surface_request_range_m':[.5*h,max(h,min(8*h,length/6))],
        'balanced_surface_request_m':max(h,min(3*h,length/12)),
        'balanced_bulk_request_m':max(h,min(10*h,length/8)),
        'caution':'Large bulk sizes do not permit erasing thin phases. Local grading, geometric distance, region/component checks and measured shape criteria remain active. Refinement below a voxel does not create new measured geometry.'})
    return report


def _record(options):
    return {f.name:getattr(options,f.name) for f in fields(options) if f.name!='size_field'}|{
        'custom_size_field':'callback sampled at voxel centres' if callable(options.size_field) else 'array' if options.size_field is not None else None}


def _sizes(model,o,advice):
    h=min(model.spacing);r=advice['recommended_sizes'];factor={'coarse':1.65,'balanced':1.,'fine':2/3,'very_fine':.4}[o.precision]
    surface=o.surface_size_m or r['balanced_surface_request_m']*factor
    bulk=o.bulk_size_m or max(surface,r['balanced_bulk_request_m']*factor)
    distance=o.facet_distance_m or h*{'coarse':.8,'balanced':.5,'fine':.25,'very_fine':.125}[o.precision]
    return float(surface),float(bulk),float(distance),float(o.outer_surface_size_m or bulk)


def _sizing_field(model,o,surface,bulk,progress):
    if o.size_field is not None:
        if callable(o.size_field):
            result=np.empty(model.regions.shape,np.float32)
            for start in range(0,result.size,65536):
                ids=np.arange(start,min(start+65536,result.size));zyx=np.column_stack(np.unravel_index(ids,result.shape))
                xyz=((zyx+.5)*model.spacing+model.origin)[:,::-1]
                values=np.asarray(o.size_field(xyz),float)
                if values.shape!=(len(ids),):raise ValueError('size_field callback must return one SI size per XYZ point')
                result.ravel()[ids]=values
        else:result=np.asarray(o.size_field,np.float32)
        if result.shape!=model.regions.shape or not np.isfinite(result).all() or np.any(result<=0):raise ValueError('Positive finite size_field on the processed ZYX grid required')
        return np.ascontiguousarray(result)
    result=np.full(model.regions.shape,bulk,np.float32);phases=np.unique(model.regions)
    if o.grading=='surface_driven':return result
    with progress_stage('graded sizing / 特征分级尺寸场',progress,estimated_seconds=max(.1,model.regions.size*len(phases)/10e6)) as stage:
        if len(phases)>1:
            for j,k in enumerate(phases):
                mask=model.regions==k
                # No exterior padding: flat observation-box walls should not
                # be treated as internal thin material features.
                distance=ndimage.distance_transform_edt(mask,sampling=model.spacing)
                result[mask]=np.minimum(bulk,surface+o.gradation*np.maximum(0.,distance[mask]-max(model.spacing)/2))
                stage.update(j+1,len(phases),f'region {int(k)}')
        else:stage.update(1,1,'uniform material')
    return result


def _component_seed_edges(model):
    """Six actual label-crossing rays per face-connected component.

    These initialize CGAL interfaces; they are not voxel-centroid material
    assignments to finished tetrahedra. CGAL bisects the continuous score
    labeling function to find the actual interface on each supplied segment.
    """
    shape=np.array(model.regions.shape);edges=[]
    for k in np.unique(model.regions):
        components,n=ndimage.label(model.regions==k);slices=ndimage.find_objects(components)
        for j,box in enumerate(slices,1):
            if box is None:continue
            mask=components[box]==j;distance=ndimage.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1,1:-1]
            centre=np.array(np.unravel_index(np.argmax(distance),distance.shape))+[s.start for s in box]
            for axis in range(3):
                for direction in (-1,1):
                    a=centre.copy();b=a.copy()
                    while True:
                        b[axis]+=direction
                        if np.any(b<0) or np.any(b>=shape) or model.regions[tuple(b)]!=k:break
                        a=b.copy()
                    edges.append(np.stack(((a+.5)*model.spacing,(b+.5)*model.spacing))[:,::-1])
    return np.asarray(edges,np.float64).reshape(-1,2,3)


class _NativeOutput(tuple):
    def __new__(cls,arrays,diagnostics):
        obj=super().__new__(cls,arrays);obj.diagnostics=diagnostics;return obj


def _native(scores,sizing,spacing,config,limits,seeds,workspace,callback=None):
    from .cgal_backend import library_path,identity
    f32=ctypes.POINTER(ctypes.c_float);f64=ctypes.POINTER(ctypes.c_double);u64=ctypes.POINTER(ctypes.c_uint64);i64=ctypes.POINTER(ctypes.c_int64)
    u32=ctypes.POINTER(ctypes.c_uint32);i32=ctypes.POINTER(ctypes.c_int32)
    cbtype=ctypes.CFUNCTYPE(None,ctypes.c_int,ctypes.c_uint64,ctypes.c_uint64)
    lib=ctypes.CDLL(str(library_path(workspace)))
    try:lib.mq_cgal_source_id.restype=ctypes.c_char_p;compiled=lib.mq_cgal_source_id().decode('ascii')
    except AttributeError as error:raise RuntimeError('CGAL library has no source identity; explicitly rebuild this backend') from error
    if compiled!=identity()['source_sha256']:raise RuntimeError('CGAL binary/source identity mismatch; rebuild or select a matching library')
    lib.mq_cgal_mesh.argtypes=[f32,f32,u64,f64,f64,i64,f64,ctypes.c_uint64,cbtype,ctypes.c_char_p,ctypes.c_int];lib.mq_cgal_mesh.restype=ctypes.c_void_p
    lib.mq_cgal_counts.argtypes=[ctypes.c_void_p,u64];lib.mq_cgal_counts.restype=ctypes.c_int
    lib.mq_cgal_copy.argtypes=[ctypes.c_void_p,f64,u32,i32,u64];lib.mq_cgal_copy.restype=ctypes.c_int
    lib.mq_cgal_free.argtypes=[ctypes.c_void_p];lib.mq_cgal_free.restype=None
    lib.mq_cgal_diagnostics.argtypes=[ctypes.c_void_p,f64];lib.mq_cgal_diagnostics.restype=ctypes.c_int
    shape=np.array((*scores.shape[:0:-1],scores.shape[0]),np.uint64)
    scores=np.ascontiguousarray(scores,np.float32);sizing=np.ascontiguousarray(sizing,np.float32)
    spacing=np.ascontiguousarray(spacing,np.float64);config=np.ascontiguousarray(config,np.float64);limits=np.ascontiguousarray(limits,np.int64);seeds=np.ascontiguousarray(seeds,np.float64)
    cb=cbtype(callback or (lambda *args:None));error=ctypes.create_string_buffer(4096)
    handle=lib.mq_cgal_mesh(scores.ctypes.data_as(f32),sizing.ctypes.data_as(f32),shape.ctypes.data_as(u64),spacing.ctypes.data_as(f64),
        config.ctypes.data_as(f64),limits.ctypes.data_as(i64),seeds.ctypes.data_as(f64),len(seeds),cb,error,len(error))
    if not handle:raise RuntimeError('CGAL native meshing rejected: '+error.value.decode(errors='replace'))
    try:
        counts=np.zeros(3,np.uint64)
        if not lib.mq_cgal_counts(handle,counts.ctypes.data_as(u64)):raise RuntimeError('CGAL counts unavailable')
        np_,nt,nl=map(int,counts)
        if not 0<np_<=limits[0] or not 0<nt<=limits[1] or nl!=scores.shape[0]:raise RuntimeError('CGAL output budget/label mismatch')
        p=np.empty((np_,3),np.float64);c=np.empty((nt,4),np.uint32);r=np.empty(nt,np.int32);components=np.empty(nl,np.uint64)
        if not lib.mq_cgal_copy(handle,p.ctypes.data_as(f64),c.ctypes.data_as(u32),r.ctypes.data_as(i32),components.ctypes.data_as(u64)):raise RuntimeError('CGAL output copy failed')
        values=np.zeros(6,np.float64)
        if not lib.mq_cgal_diagnostics(handle,values.ctypes.data_as(f64)):raise RuntimeError('Native quality diagnostics unavailable')
        diagnostics=dict(zip(('initial_radius_edge_max','initial_minimum_dihedral_degrees','final_radius_edge_max','final_minimum_dihedral_degrees','local_refinement_cycles','optimizer_return_code'),map(float,values)))
        return _NativeOutput((p,c,r,components),diagnostics)
    finally:lib.mq_cgal_free(handle)


def _isolated(scores,sizing,spacing,config,limits,seeds,workspace,timeout,progress_stage_):
    import h5py
    from .voxel import external_path
    from .cgal_backend import library_path
    root=external_path(workspace or os.environ.get('MORQEX_WORKSPACE',str(Path(tempfile.gettempdir())/'morqex')))
    folder=root/'cache/implicit_calls'/f'{time.time_ns()}';folder.mkdir(parents=True)
    source=folder/'input.h5';target=folder/'raw_output.h5'
    with h5py.File(source,'w') as f:
        for name,a in [('scores',scores),('sizing',sizing),('spacing',spacing),('config',config),('limits',limits),('seeds',seeds)]:
            f.create_dataset(name,data=a,compression='lzf' if name in ('scores','sizing') else None)
    env=os.environ.copy();env['MORQEX_CGAL_LIBRARY']=str(library_path(workspace));env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
    command=[sys.executable,'-m','morqex.modeling.implicit_worker',str(source),str(target)]
    events=[];start=time.perf_counter()
    with (folder/'stdout.txt').open('w') as out,(folder/'stderr.txt').open('w') as err:
        process=subprocess.Popen(command,env=env,stdout=out,stderr=err)
        while process.poll() is None:
            if time.perf_counter()-start>timeout:
                process.kill();process.wait();raise RuntimeError('CGAL native timeout; checkpoint '+str(folder))
            lines=(folder/'stdout.txt').read_text().splitlines()
            try:last=json.loads(lines[-1]) if lines else None
            except json.JSONDecodeError:last=None
            if last:
                name={0:'Delaunay refinement',1:'joint quality optimization',2:'quality measurement / packing',3:'component/array packing',4:'native complete'}.get(last['native_stage'],'local quality refinement' if last['native_stage']>=10 else 'native work')
                progress_stage_.message=f"{name}; {last['nodes']:,} nodes / {last['cells']:,} cells at last completed native stage"
            else:progress_stage_.message='native initialization; internal work count unknown'
            time.sleep(.1)
    if process.returncode or not target.exists():raise RuntimeError('CGAL worker failed; diagnostic checkpoint '+str(folder)+': '+(folder/'stderr.txt').read_text()[-2000:])
    for line in (folder/'stdout.txt').read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    with h5py.File(target) as f:
        if not f.attrs.get('complete',False):raise RuntimeError('Incomplete native CGAL output')
        arrays=tuple(f[key][:] for key in ('points','cells','labels','components'))
        diagnostics=json.loads(f.attrs.get('native_diagnostics','{}'))
        if diagnostics:events.append({'native_diagnostics':diagnostics})
    # Keep small diagnostic logs; successful input/raw buffers are temporary.
    source.unlink();target.unlink()
    return arrays,events,str(folder)


def implicit_mesh(geometry,*,options=None,smoothing=None,workspace=None,progress=None,regions=None):
    """Full-box mesh. Prepared input is never resmoothed or downsampled.

    Raw input uses balanced preparation unless smoothing is explicit. FEM
    precision never implicitly changes the geometry regularization strength.

    CGAL material indices come from the continuous labeled-domain mesher.
    This is not post-hoc source-voxel centroid labeling of unconstrained cells.
    """
    from .voxel import VoxelModel
    from .tetgen import TetGenMesh,tetra_quality
    from .imports import material_interfaces
    from .cgal_backend import identity,library_path
    o=options or ImplicitMeshOptions()
    if not isinstance(o,ImplicitMeshOptions):raise TypeError('ImplicitMeshOptions required')
    present=set(geometry.field_labels) if isinstance(geometry,PreparedGeometry) else set(map(int,np.unique(geometry.regions))) if isinstance(geometry,VoxelModel) else set()
    if regions is not None and set(map(int,regions))!=present:raise ValueError('Implicit whole-box meshing retains every phase; use the existing selected-region TetGen API for solid-only PLCs')
    library_path(workspace)  # explicit build required, before costly preparation
    start=time.perf_counter()
    if isinstance(geometry,VoxelModel):
        prepared=smooth_voxels(geometry,options=smoothing or SmoothingOptions(),progress=progress)
    elif isinstance(geometry,PreparedGeometry):
        if smoothing is not None:raise ValueError('Prepared geometry is not automatically resmoothed; call smooth_voxels explicitly')
        prepared=geometry.validate()
    else:raise TypeError('VoxelModel or PreparedGeometry required')
    if o.require_preparation_fidelity and not prepared.report['fidelity_limits_satisfied']:raise ArithmeticError('Preparation fidelity not accepted; revise explicit smoothing options')
    model=prepared.model;advice=recommend_mesh_sizes(prepared,progress=progress);surface,bulk,distance,outer=_sizes(model,o,advice)
    sizing=_sizing_field(model,o,surface,bulk,progress);scale=float(min(model.spacing))
    # CGAL6.0.1 uses half the diagonal of the sphere's bounding box.
    # Keep an explicitly voxel-sized tolerance, below the
    # requested geometric facet distance, rather than a fixed scale-free1e-7.
    domain_radius=float(np.sqrt(.26*np.sum((np.array(model.regions.shape)*model.spacing/scale)**2)))
    absolute_oracle=min(o.oracle_tolerance_voxels,distance/scale/10)
    relative_oracle=absolute_oracle/(np.sqrt(3)*domain_radius)
    with progress_stage('component interface initialization / 连通域界面初始化',progress,estimated_seconds=max(.1,model.regions.size/10e6)):
        seeds=_component_seed_edges(model)/scale
    labels=np.array(prepared.field_labels,np.int64);target=max(o.minimum_dihedral_degrees,o.optimization_target_degrees)
    history=None
    if workspace:
        history=TimingHistory(Path(workspace)/'cache/geometry_timing.json')
    units=float(sum(advice['contact_areas_m2'].values())/max(surface**2,1e-300)+np.prod(np.array(model.regions.shape)*model.spacing)/bulk**3)
    estimate,estimate_source=history.estimate('implicit_cgal',max(1.,units),8000.) if history else (max(1.,units/8000.),'uncalibrated workload heuristic; approximate')
    attempts=[];native_start=time.perf_counter()
    expected={str(int(k)):float(np.sum(model.regions==k)*np.prod(model.spacing)) for k in labels}
    radius_factor=1.
    for attempt in range(o.maximum_quality_retries+1):
        absolute_oracle=min(o.oracle_tolerance_voxels,distance/scale/10)
        relative_oracle=absolute_oracle/(np.sqrt(3)*domain_radius)
        seconds=o.optimization_seconds*(1.5**attempt);native_ratio=max(1.,o.native_radius_edge_ratio*radius_factor)
        perturb=seconds if o.optimization in ('joint','joint_exude','perturb','perturb_exude') else 0.
        exude=seconds if o.optimization in ('exude','perturb_exude','joint_exude') else 0.
        config=np.array([o.surface_minimum_angle_degrees,surface/scale,distance/scale,bulk/scale,native_ratio,perturb,exude,target,relative_oracle,1.,o.radius_edge_limit,outer/scale,o.seed_separation_voxels,o.minimum_dihedral_degrees],np.float64)
        limits=np.array([o.maximum_nodes,o.maximum_cells,o.seed,o.initial_random_points,int(o.optimization in ('joint','joint_exude')),o.maximum_local_repair_cycles],np.int64)
        with progress_stage(f'implicit volume meshing / 隐式域体网格 [attempt {attempt+1}]',progress,estimated_seconds=estimate,estimate_source=estimate_source) as stage:
            if o.isolated:
                (p,c,r,components),events,checkpoint=_isolated(prepared.scores,sizing/scale,np.array(model.spacing[::-1])/scale,config,limits,seeds,workspace,o.timeout_seconds,stage)
            else:
                events=[]
                def event(code,np_,nt):
                    events.append({'native_stage':code,'nodes':np_,'cells':nt,'elapsed_seconds':time.perf_counter()-native_start})
                    stage.message=f'native stage {code}; {np_:,} nodes / {nt:,} cells at last completed stage'
                native_output=_native(prepared.scores,sizing/scale,np.array(model.spacing[::-1])/scale,config,limits,seeds,workspace,event)
                p,c,r,components=native_output;checkpoint=None
                if getattr(native_output,'diagnostics',None):events.append({'native_diagnostics':native_output.diagnostics})
        p=p*scale+np.array(model.origin[::-1])
        if np.any(r<1) or np.any(r>len(labels)):raise ArithmeticError('CGAL returned unknown material indices')
        phase=labels[r-1]
        if set(map(int,np.unique(phase)))!=set(map(int,labels)):raise ArithmeticError('Implicit meshing omitted a requested material')
        for first in range(0,len(c),50000):
            block=c[first:first+50000];reverse=np.linalg.det(p[block[:,1:]]-p[block[:,:1]])<0;block[reverse]=block[reverse][:,[1,0,2,3]]
        with progress_stage('quality validation / 独立单元质量验证',progress):q,volumes=tetra_quality(p,c)
        accepted=q['radius_edge_max']<=o.radius_edge_limit*(1+1e-6) and q['minimum_dihedral_degrees']+1e-6>=o.minimum_dihedral_degrees
        phase_volumes={str(int(k)):float(volumes[phase==k].sum()) for k in labels}
        errors={k:abs(v/expected[k]-1) for k,v in phase_volumes.items() if expected[k]>0}
        box_error=abs(float(volumes.sum())/float(np.prod(np.array(model.regions.shape)*model.spacing))-1)
        geometry_accepted=max(errors.values(),default=0)<=o.maximum_phase_volume_error and box_error<=o.maximum_box_volume_error
        attempts.append({'attempt':attempt,'native_radius_edge_ratio':native_ratio,'optimization_seconds':seconds,'quality':q,
            'facet_distance_m':distance,'shape_passed':bool(accepted),'geometry_passed':bool(geometry_accepted),
            'phase_volume_errors':errors,'box_volume_error':float(box_error),'native_events':events,'checkpoint':checkpoint})
        if (accepted or not o.require_quality) and geometry_accepted:break
        if not geometry_accepted:distance*=.5
        if not accepted and o.require_quality:radius_factor*=.9
        if attempt<o.maximum_quality_retries:del p,c,r,components,phase,volumes
    if (not accepted and o.require_quality) or not geometry_accepted:
        raise ArithmeticError('Implicit mesh misses unchanged shape/geometry gates after bounded retries: '+json.dumps(attempts))
    if history:history.record('implicit_cgal',max(1.,units),time.perf_counter()-native_start)
    with progress_stage('physical interfaces / 真实材料界面恢复',progress):facets,markers,pairs=material_interfaces(c,phase)
    report={'mesher':'source-integrated CGAL6.0.1 Mesh_3','source_identity':identity(),'nodes':len(p),'tetrahedra':len(c),
        'quality':q,'quality_limits_satisfied':bool(accepted),'geometry_limits_satisfied':bool(geometry_accepted),
        'quality_request':{'radius_edge_max':o.radius_edge_limit,'minimum_dihedral_degrees':o.minimum_dihedral_degrees},
        'geometry':{'surface_method':'continuous multiphase score-domain refinement','region_names':model.region_names,'interface_pairs':pairs,
            'source_shape_zyx':list(model.regions.shape),'spacing_zyx_m':list(model.spacing),'origin_zyx_m':list(model.origin),
            'full_grid_retained':True,'resolution_reduction_applied':False,
            'input_was_explicitly_reduced':bool(model.metadata.get('resolution_reduction_applied',False)),
            'selection_groups':model.metadata.get('selection_groups',{})},
        'preparation':prepared.report,'feature_analysis':advice,'mesh_options':_record(o),
        'sizing':{'surface_size_m':surface,'bulk_size_m':bulk,'outer_surface_size_m':outer,'facet_distance_m':distance,
            'oracle_absolute_tolerance_m':absolute_oracle*scale,'oracle_relative_error_bound':relative_oracle,
            'size_field_sha256':hashlib.sha256(np.ascontiguousarray(sizing).view(np.uint8)).hexdigest(),
            'scope':'CGAL sizing requests, not strict maximum edge-length bounds; nominal sizes do not force deletion of thinner features'},
        'region_volumes_m3':phase_volumes,'phase_volumes_m3':phase_volumes,'processed_voxel_phase_volume_errors':errors,'physical_box_volume_relative_error':float(box_error),
        'native_face_connected_components':{str(int(k)):int(components[j]) for j,k in enumerate(labels)},
        'component_initialization_segments':len(seeds),'attempts':attempts,'elapsed_seconds':time.perf_counter()-start,
        'geometry_fingerprint':prepared.report['processed_fingerprint'],'geometry_score_sha256':prepared.report['score_sha256'],
        'mapping':'CGAL labeled-domain cell indices; no unique parent voxel; continuous score geometry differs from the voxel staircase',
        'limitations':['geometry/shape checks do not certify PDE solution error','smoothing and continuous reconstruction may change topology; measured voxel and mesh component counts are reported','outer box surfaces are approximated within geometric meshing controls, not an exact planar PLC contract']}
    prepared.validate()
    return TetGenMesh(p,c,phase,facets=facets,facet_markers=markers,report=report)
