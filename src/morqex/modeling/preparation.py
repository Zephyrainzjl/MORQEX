"""Separate fixed-grid smoothing from explicit resolution reduction.

Label scores define geometry only; they are not species concentrations or
calibrated material probabilities. Original data is never overwritten.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,os,time
import numpy as np
from scipy import ndimage
from .voxel import VoxelModel,from_regions,external_path
from .features import voxel_fingerprint,analyze_features,contact_areas
from .progress import progress_stage


@dataclass(frozen=True)
class SmoothingOptions:
    method:str='gaussian'
    reconstruction:str='filtered_scores'
    continuous_volume_calibration:bool=False
    calibration_sample_power:int=18
    calibration_iterations:int=12
    calibration_relative_tolerance:float=.02
    calibration_seed:int=20261128
    iterations:int=2
    minimum_feature_voxels:float=2.
    minimum_component_voxels:int|None=None
    final_component_cleanup:bool=True
    sigma_voxels:object=None
    sigma_m:object=None
    preserve_phase_volumes:bool=True
    volume_balance_tolerance:float=.01
    volume_balance_iterations:int=8
    maximum_changed_fraction:float=.12
    maximum_phase_volume_error:float=.12
    minimum_phase_iou:float=.65
    topology_policy:str='report'
    allow_phase_loss:bool=False
    acceptance:str='strict'
    maximum_score_memory_mib:int=768

    def __post_init__(self):
        if self.method not in ('none','gaussian','majority','signed_distance','morphology'):
            raise ValueError('Smoothing method: none, gaussian, majority, signed_distance or morphology')
        if self.reconstruction not in ('filtered_scores','signed_distance'):raise ValueError('Reconstruction uses filtered_scores or signed_distance')
        if type(self.calibration_sample_power) is not int or not 12<=self.calibration_sample_power<=21:raise ValueError('Calibration Sobol power in12..21 required')
        if type(self.calibration_iterations) is not int or not 1<=self.calibration_iterations<=40:raise ValueError('Calibration iterations in1..40 required')
        if not 0<self.calibration_relative_tolerance<1:raise ValueError('Positive calibration tolerance required')
        if type(self.calibration_seed) is not int or not 0<=self.calibration_seed<2**32:raise ValueError('Unsigned32-bit calibration seed required')
        if type(self.iterations) is not int or not 1<=self.iterations<=8:raise ValueError('Fixed-grid smoothing iterations must be1..8')
        if not np.isfinite(self.minimum_feature_voxels) or self.minimum_feature_voxels<1:
            raise ValueError('Minimum feature target is finite and >=1 voxel')
        if self.minimum_component_voxels is not None and (type(self.minimum_component_voxels) is not int or self.minimum_component_voxels<0):raise ValueError('Nonnegative component-volume floor required')
        if self.sigma_m is not None and self.sigma_voxels is not None:raise ValueError('Specify sigma_voxels OR sigma_m')
        for name in ('sigma_voxels','sigma_m'):
            value=getattr(self,name)
            if value is not None:
                a=np.asarray(value,float)
                if a.shape not in ((),(3,)) or not np.isfinite(a).all() or np.any(a<0):raise ValueError('Nonnegative finite scalar or ZYX sigma required')
                object.__setattr__(self,name,float(a) if a.ndim==0 else tuple(map(float,a)))
        if not 0<=self.maximum_changed_fraction<=1 or not 0<=self.maximum_phase_volume_error<=1 or not 0<=self.minimum_phase_iou<=1:
            raise ValueError('Fidelity tolerances must lie in0..1')
        if self.topology_policy not in ('report','preserve_components_and_contacts'):raise ValueError('Unknown topology policy')
        if self.acceptance not in ('strict','report'):raise ValueError('acceptance is strict or report')
        if not 0<self.volume_balance_tolerance<1 or type(self.volume_balance_iterations) is not int or not 1<=self.volume_balance_iterations<=40:
            raise ValueError('Positive phase-volume balance tolerance and bounded iterations required')
        if type(self.maximum_score_memory_mib) is not int or self.maximum_score_memory_mib<1:raise ValueError('Positive score-memory budget required')

    @classmethod
    def precision(cls,name,**overrides):
        choices={'very_fine':dict(iterations=1,minimum_feature_voxels=1,sigma_voxels=.25),
            'fine':dict(iterations=1,minimum_feature_voxels=1.5,sigma_voxels=.5),
            'balanced':dict(iterations=2,minimum_feature_voxels=2,sigma_voxels=.7),
            'coarse':dict(iterations=3,minimum_feature_voxels=4,sigma_voxels=1.4)}
        if name not in choices:raise ValueError('Unknown smoothing precision')
        parameters=choices[name]|overrides
        if overrides.get('sigma_m') is not None and 'sigma_voxels' not in overrides:parameters['sigma_voxels']=None
        return cls(**parameters)


@dataclass
class PreparedGeometry:
    model:VoxelModel
    scores:np.ndarray
    field_labels:tuple
    report:dict

    def validate(self):
        if self.scores.shape!=(len(self.field_labels),*self.model.regions.shape) or not np.isfinite(self.scores).all():
            raise ValueError('One finite geometry score field per label required')
        if voxel_fingerprint(self.model)!=self.report['processed_fingerprint']:raise ValueError('Prepared labels, named selections or physical coordinates changed; prepare again')
        if hashlib.sha256(np.ascontiguousarray(self.scores).view(np.uint8)).hexdigest()!=self.report['score_sha256']:
            raise ValueError('Prepared geometry fields changed; prepare again')
        return self

    def analyze(self,*,progress=None):return analyze_features(self.model,progress=progress)

    def save(self,path,*,progress=None):
        """Complete grid and continuous geometry scores in external HDF5."""
        import h5py
        self.validate();path=external_path(path)
        if path.exists():raise FileExistsError('Do not overwrite a geometry preparation')
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f'.{os.getpid()}.pending')
        with progress_stage('save prepared geometry / 保存平滑几何',progress) as stage:
            with h5py.File(temporary,'w') as f:
                f.create_dataset('phase_labels',data=self.model.regions,compression='gzip',shuffle=True)
                f.create_dataset('geometry_scores',data=self.scores,compression='gzip',shuffle=True)
                f.attrs.update(spacing_zyx_m=self.model.spacing,origin_zyx_m=self.model.origin,
                    region_names=json.dumps(self.model.region_names),metadata=json.dumps(self.model.metadata),
                    field_labels=self.field_labels,preparation_report=json.dumps(self.report),complete=True)
                stage.update(1,1)
            temporary.replace(path)
        return path

    @classmethod
    def load(cls,path):
        import h5py
        with h5py.File(path) as f:
            if not f.attrs.get('complete',False):raise ValueError('Incomplete prepared geometry')
            model=from_regions(f['phase_labels'][:],spacing=tuple(f.attrs['spacing_zyx_m']),origin=tuple(f.attrs['origin_zyx_m']),
                region_names=json.loads(f.attrs['region_names']),metadata=json.loads(f.attrs['metadata']))
            result=cls(model,f['geometry_scores'][:],tuple(map(int,f.attrs['field_labels'])),json.loads(f.attrs['preparation_report']))
        return result.validate()


def _scores(model,options,progress):
    a=model.regions;labels=tuple(map(int,np.unique(a)))
    if a.size*len(labels)*4>options.maximum_score_memory_mib*2**20:raise MemoryError('Geometry scores exceed explicit memory budget; increase the budget or process fewer phases')
    scores=np.empty((len(labels),*a.shape),np.float32)
    sigma=options.sigma_voxels if options.sigma_voxels is not None else .35*options.minimum_feature_voxels
    # Index-space sigma and physical SI sigma are separate explicit choices.
    sigmas=np.broadcast_to(np.asarray(options.sigma_m,float)/model.spacing if options.sigma_m is not None else sigma,(3,))
    radius=(options.minimum_feature_voxels-1)/2
    reach=int(np.ceil(radius));ball=None
    if options.method=='morphology':
        if options.minimum_feature_voxels>min(a.shape):raise ValueError('Morphological feature target exceeds the observation grid')
        coordinates=np.ogrid[tuple(slice(-reach,reach+1) for _ in range(3))]
        ball=sum((c*model.spacing[j]/min(model.spacing))**2 for j,c in enumerate(coordinates))<=max(radius,1e-12)**2
    with progress_stage('fixed-grid smoothing / 原尺寸体素平滑',progress,estimated_seconds=max(.1,a.size*len(labels)/15e6)) as stage:
        for j,k in enumerate(labels):
            mask=a==k
            if options.method=='none':scores[j]=mask
            elif options.method=='gaussian':ndimage.gaussian_filter(mask.astype(np.float32),sigmas,output=scores[j],mode='nearest')
            elif options.method=='majority':
                widths=tuple(max(1,2*int(np.ceil(radius*min(model.spacing)/h))+1) for h in model.spacing)
                ndimage.uniform_filter(mask.astype(np.float32),size=widths,output=scores[j],mode='nearest')
            elif options.method=='signed_distance':
                inside=ndimage.distance_transform_edt(np.pad(mask,1),sampling=model.spacing)[1:-1,1:-1,1:-1]
                outside=ndimage.distance_transform_edt(np.pad(~mask,1),sampling=model.spacing)[1:-1,1:-1,1:-1]
                field=((inside-outside)/min(model.spacing)).astype(np.float32)
                ndimage.gaussian_filter(field,sigmas,output=scores[j],mode='nearest')
            else:
                # Nearest padding avoids eroding material merely because the
                # chosen physical observation box ends at a planar boundary.
                padded=np.pad(mask,reach,mode='edge') if reach else mask
                clean=ndimage.binary_closing(ndimage.binary_opening(padded,structure=ball),structure=ball)
                clean=clean[reach:-reach,reach:-reach,reach:-reach] if reach else clean
                ndimage.gaussian_filter(clean.astype(np.float32),sigmas,output=scores[j],mode='nearest')
            stage.update(j+1,len(labels),f'region {k}')
    return scores,labels


def _regularized_source(model,options,progress):
    """Optional face-component area opening before continuous smoothing.

    The largest component of every label is retained. This is a declared
    geometric change, not a claim that removed isolated pieces were noise.
    Their volume can be redistributed by the separate capacity constraint.
    """
    floor=options.minimum_component_voxels
    if floor is None:floor=int(round(options.minimum_feature_voxels**3))
    if options.method=='none' or floor<=1:return model,{'applied':False,'component_volume_floor_voxels':floor}
    a=model.regions;discard=np.zeros(a.shape,bool);records={};labels=np.unique(a)
    with progress_stage('small-feature regularization / 小特征规则化',progress,estimated_seconds=max(.1,a.size*len(labels)/12e6)) as stage:
        for j,k in enumerate(labels):
            components,n=ndimage.label(a==k);counts=np.bincount(components.ravel())
            small=counts<floor;small[0]=False;small[1+np.argmax(counts[1:])]=False
            mask=small[components];discard|=mask
            records[str(int(k))]={'selected_components':int(small.sum()),'selected_voxels':int(mask.sum()),'largest_component_retained':True}
            stage.update(j+1,2*len(labels),f'component selection {int(k)}')
        b=a.copy()
        if discard.any():
            for j,k in enumerate(labels):
                mask=discard&(a==k)
                if mask.any():
                    retained=(a!=k)&~discard
                    indices=ndimage.distance_transform_edt(~retained,sampling=model.spacing,return_distances=False,return_indices=True)
                    b[mask]=a[tuple(indices[:,mask])]
                    del indices
                stage.update(len(labels)+j+1,2*len(labels),f'reassignment {int(k)}')
        else:stage.update(2*len(labels),2*len(labels),'no selected components')
    return from_regions(b,spacing=model.spacing,origin=model.origin,region_names=model.region_names,metadata=model.metadata),{
        'applied':bool(discard.any()),'component_volume_floor_voxels':floor,'regions':records,
        'scope':'fixed-grid component regularization; target scale, not guaranteed minimum neck thickness; original source retained and all total changes checked'}


def _winner(scores,labels,original=None):
    # Original-label tie breaking removes an otherwise arbitrary ID bias.
    best=np.max(scores,axis=0);winner=np.argmax(scores,axis=0)
    if original is not None:
        for j,k in enumerate(labels):
            keep=(original==k)&np.isclose(scores[j],best,rtol=0,atol=1e-7)
            winner[keep]=j
    return np.asarray(labels,dtype=original.dtype if original is not None else np.int32)[winner]


def _balance_scores(model,scores,labels,options,progress):
    """Coordinate threshold shifts for approximately prescribed label counts.

    For fixed competing scores, a quantile of score_j-max(other scores)
    determines the shift that attains the requested count. Coupled coordinate
    sweeps are stopped at the stated tolerance/budget, never declared an exact
    optimum. A tiny original-label prior resolves quantized score plateaus.
    """
    target=np.array([np.count_nonzero(model.regions==k) for k in labels]);bias=np.zeros(len(labels));errors=np.zeros(len(labels));completed=0
    if options.method=='none' or not options.preserve_phase_volumes or len(labels)==1:
        return {'applied':False,'scope':'no volume-constrained score shifts'}
    for j,k in enumerate(labels):scores[j]+=(model.regions==k)*np.float32(1e-6)
    with progress_stage('phase volume balance / 分相体积约束',progress,
        estimated_seconds=max(.1,model.regions.size*len(labels)/5e6)) as stage:
        for iteration in range(options.volume_balance_iterations):
            winner=_winner(scores,labels,model.regions)
            actual=np.array([np.count_nonzero(winner==k) for k in labels]);errors=abs(actual/target-1)
            if errors.max()<=options.volume_balance_tolerance:break
            for j in range(len(labels)):
                other=np.full(model.regions.shape,-np.inf,np.float32)
                for k in range(len(labels)):
                    if k!=j:np.maximum(other,scores[k],out=other)
                margin=(scores[j]-other).ravel();rank=int(len(margin)-target[j])
                if not 0<rank<len(margin):continue
                partition=np.partition(margin,(rank-1,rank));shift=-float((np.float64(partition[rank-1])+partition[rank])/2)
                scores[j]+=shift;bias[j]+=shift
            completed=iteration+1
            stage.update(iteration+1,options.volume_balance_iterations,f'max label-volume error before sweep {errors.max():.3%}')
        winner=_winner(scores,labels,model.regions)
        errors=abs(np.array([np.count_nonzero(winner==k) for k in labels])/target-1)
        stage.update(max(1,completed),max(1,completed),f'final max label-volume error {errors.max():.3%}; {completed} sweeps')
    return {'applied':True,'iterations':completed,'target_relative_tolerance':options.volume_balance_tolerance,
        'tolerance_met':bool(errors.max()<=options.volume_balance_tolerance),
        'relative_errors':{str(k):float(errors[j]) for j,k in enumerate(labels)},'score_shifts':{str(k):float(bias[j]) for j,k in enumerate(labels)},
        'scope':'approximate fixed-grid label-capacity balancing; not exact continuous-domain volume conservation or a certified optimization solution'}


def _comparison(source,result):
    """Same-grid fidelity: exact changed voxels, IoU and face contacts."""
    regions={};topology_changed=False;loss=[]
    for k in np.unique(source.regions):
        old=source.regions==k;new=result.regions==k;before,n0=ndimage.label(old);after,n1=ndimage.label(new)
        del before,after
        if not new.any():loss.append(int(k))
        error=abs(int(new.sum())/int(old.sum())-1);union=np.count_nonzero(old|new)
        regions[str(int(k))]={'volume_relative_error':float(error),'iou':float(np.count_nonzero(old&new)/union),
            'before_voxels':int(old.sum()),'after_voxels':int(new.sum()),'before_components_6':int(n0),'after_components_6':int(n1)}
        topology_changed|=n0!=n1
    a=contact_areas(source);b=contact_areas(result);added=sorted(set(b)-set(a));removed=sorted(set(a)-set(b))
    return {'changed_voxels':int(np.count_nonzero(source.regions!=result.regions)),
        'changed_fraction':float(np.mean(source.regions!=result.regions)),'regions':regions,'lost_regions':loss,
        'contact_areas_before_m2':a,'contact_areas_after_m2':b,'added_contact_pairs':added,'removed_contact_pairs':removed,
        'checked_topology_changed':bool(topology_changed or added or removed),
        'topology_scope':'component counts and phase contact graph only; equal descriptors are not a proof of identical topology'}


def _calibrate_continuous_volume(model,scores,labels,options,progress):
    """Approximate geometric-volume calibration, with FIXED nodal labels.

    Nonnegative gains multiply each phase's nodal indicator. At a voxel
    centre only the winning channel increases, so no nodal label is removed.
    Scrambled Sobol integration is an estimator, not a certified volume bound.
    The complete fields are retained; no voxel grid is reduced or replaced.
    """
    if not options.continuous_volume_calibration:return {'applied':False}
    from scipy.stats import qmc
    shape=np.array(model.regions.shape);count=2**options.calibration_sample_power
    target=np.array([np.mean(model.regions==k) for k in labels])
    def samples(seed):
        unit=qmc.Sobol(d=3,scramble=True,seed=seed).random_base2(options.calibration_sample_power)
        coordinates=(unit*shape-.5).T
        values=np.stack([ndimage.map_coordinates(field,coordinates,order=1,mode='nearest',prefilter=False) for field in scores]).astype(float)
        support=np.stack([ndimage.map_coordinates((model.regions==k).astype(np.float32),coordinates,order=1,mode='nearest',prefilter=False) for k in labels]).astype(float)
        return values,support
    with progress_stage('continuous volume calibration / 连续几何体积标定',progress,estimated_seconds=max(.1,model.regions.size*len(labels)/5e6)) as stage:
        base,support=samples(options.calibration_seed);gain=np.zeros(len(labels));target_count=np.rint(target*count).astype(int)
        valid=bool(np.all(target_count>0) and np.all(np.count_nonzero(support>1e-10,axis=1)>=target_count))
        before=np.bincount(np.argmax(base,axis=0),minlength=len(labels))/count
        completed=0
        if valid and len(labels)>1:
            for iteration in range(options.calibration_iterations):
                values=base+gain[:,None]*support
                actual=np.bincount(np.argmax(values,axis=0),minlength=len(labels))/count
                if np.max(abs(actual/target-1))<=options.calibration_relative_tolerance:break
                for j in range(len(labels)):
                    other=np.max(np.delete(base+gain[:,None]*support,j,axis=0),axis=0)
                    active=support[j]>1e-10;fixed=int(np.count_nonzero(~active&(base[j]>=other)))
                    wanted=int(np.clip(target_count[j]-fixed,0,active.sum()))
                    margin=(other[active]-base[j,active])/support[j,active]
                    rank=wanted
                    if rank<=0:value=0.
                    elif rank>=len(margin):value=float(margin.max()+1e-6)
                    else:
                        part=np.partition(margin,(rank-1,rank));value=float((part[rank-1]+part[rank])/2)
                    gain[j]=max(0.,value)
                completed=iteration+1;stage.update(completed,options.calibration_iterations,'coordinate volume calibration')
        for j,k in enumerate(labels):scores[j][model.regions==k]+=gain[j]
        # Independent scramble after the final complete-field update.
        validation,_=samples((options.calibration_seed+1)%2**32)
        after=np.bincount(np.argmax(validation,axis=0),minlength=len(labels))/count
        errors=abs(after/target-1);passed=valid and bool(errors.max()<=options.calibration_relative_tolerance)
        stage.update(1,1,f'independent sampled relative volume error {errors.max():.3%}')
    actual_labels=np.asarray(labels)[np.argmax(scores,axis=0)]
    if not np.array_equal(actual_labels,model.regions):raise ArithmeticError('Continuous volume calibration changed a fixed nodal label')
    return {'applied':True,'samples_per_scramble':count,'fit_seed':options.calibration_seed,'validation_seed':(options.calibration_seed+1)%2**32,
        'iterations':completed,'nonnegative_nodal_gains':{str(k):float(gain[j]) for j,k in enumerate(labels)},
        'target_fractions':{str(k):float(target[j]) for j,k in enumerate(labels)},
        'sampled_fractions_before':{str(k):float(before[j]) for j,k in enumerate(labels)},
        'independent_sampled_fractions_after':{str(k):float(after[j]) for j,k in enumerate(labels)},
        'independent_sampled_relative_errors':{str(k):float(errors[j]) for j,k in enumerate(labels)},
        'tolerance_met':passed,'nodal_labels_unchanged':True,
        'scope':'approximate subvoxel geometric-volume calibration; scrambled Sobol estimates are not certified bounds; mesh volumes are independently checked'}


def smooth_voxels(model,*,options=None,progress=None):
    """Change interfaces on the SAME full grid. Never change spacing/extent.

    minimum_feature_voxels is an operation scale, not a mathematical guarantee
    that every curved voxel stair or preserved thin phase exceeds that width.
    Actual feature statistics are available through result.analyze().
    """
    if not isinstance(model,VoxelModel):raise TypeError('VoxelModel required')
    o=options or SmoothingOptions();start=time.perf_counter();source=voxel_fingerprint(model)
    if not isinstance(o,SmoothingOptions):raise TypeError('SmoothingOptions required')
    current=model;iteration_records=[]
    for iteration in range(1 if o.method=='none' else o.iterations):
        working,regularization=_regularized_source(current,o,progress)
        scores,labels=_scores(working,o,progress);balance=_balance_scores(model,scores,labels,o,progress);a=_winner(scores,labels,model.regions)
        iteration_records.append({'iteration':iteration+1,'volume_balance':balance,'component_regularization':regularization,
            'changed_fraction_this_iteration':float(np.mean(a!=current.regions))})
        present=set(map(int,np.unique(a)))
        current=from_regions(a,spacing=model.spacing,origin=model.origin,region_names={k:v for k,v in model.region_names.items() if k in present},metadata=model.metadata)
    present=set(map(int,np.unique(a)));names={k:v for k,v in model.region_names.items() if k in present}
    final_regularization={'applied':False}
    if o.final_component_cleanup:
        cleaned,final_regularization=_regularized_source(current,o,progress)
        changed=cleaned.regions!=current.regions
        if changed.any():
            with progress_stage('final feature-field consistency / 最终小特征及评分一致化',progress) as stage:
                for j,k in enumerate(labels):
                    chosen=changed&(cleaned.regions==k)
                    if chosen.any():
                        retained=(current.regions==k)&~changed
                        indices=ndimage.distance_transform_edt(~retained,sampling=model.spacing,return_distances=False,return_indices=True)
                        source_indices=tuple(indices[:,chosen]);scores[:,chosen]=scores[(slice(None),*source_indices)]
                        del indices
                    stage.update(j+1,len(labels),f'region {k}')
            a=cleaned.regions
            if balance.get('applied'):
                errors={str(k):float(abs(np.count_nonzero(a==k)/np.count_nonzero(model.regions==k)-1)) for k in labels}
                balance=balance|{'relative_errors_after_final_cleanup':errors,'tolerance_met':max(errors.values())<=o.volume_balance_tolerance}
    processed=from_regions(a,spacing=model.spacing,origin=model.origin,region_names=names,metadata=model.metadata|{
        'geometry_operation':'fixed-grid smoothing','original_fingerprint':source,'smoothing':asdict(o)})
    if o.reconstruction=='signed_distance':
        # Reinitialize the continuous geometry from the final accepted labels.
        # This changes neither a hard label nor its physical coordinates and
        # avoids interpreting tiny margins in capacity-shifted scores as a
        # physically resolved geometric feature. Subvoxel geometry is a model.
        with progress_stage('distance reconstruction / 距离场几何重构',progress,estimated_seconds=max(.1,a.size*len(labels)/4e6)) as stage:
            for j,k in enumerate(labels):
                mask=a==k
                inside=ndimage.distance_transform_edt(np.pad(mask,1),sampling=model.spacing)[1:-1,1:-1,1:-1]
                outside=ndimage.distance_transform_edt(np.pad(~mask,1),sampling=model.spacing)[1:-1,1:-1,1:-1]
                scores[j]=(inside-outside)/min(model.spacing)
                stage.update(j+1,len(labels),f'region {k}')
    nodal_labels=np.asarray(labels,dtype=a.dtype)[np.argmax(scores,axis=0)];wrong=nodal_labels!=a;nodal_corrections=int(wrong.sum())
    if nodal_corrections:
        # The label tie-breaker uses the original material. Enforce that same
        # choice in the continuous oracle with the smallest representable
        # positive score increment; never silently change a hard label.
        best=np.max(scores[:,wrong],axis=0);chosen=a[wrong];values=scores[:,wrong]
        for j,k in enumerate(labels):values[j,chosen==k]=np.nextafter(best[chosen==k],np.float32(np.inf))
        scores[:,wrong]=values
    del nodal_labels,wrong
    calibration=_calibrate_continuous_volume(processed,scores,labels,o,progress)
    comparison=_comparison(model,processed)
    passed=(comparison['changed_fraction']<=o.maximum_changed_fraction and
        all(v['volume_relative_error']<=o.maximum_phase_volume_error and v['iou']>=o.minimum_phase_iou for v in comparison['regions'].values()) and
        (o.allow_phase_loss or not comparison['lost_regions']) and
        (o.topology_policy=='report' or not comparison['checked_topology_changed']) and
        (not balance.get('applied') or balance['tolerance_met']) and
        (not calibration.get('applied') or calibration['tolerance_met']))
    if voxel_fingerprint(model)!=source:raise ArithmeticError('Smoothing modified original source')
    if comparison['lost_regions'] and o.allow_phase_loss:
        keep=[j for j,k in enumerate(labels) if k in present];scores=scores[keep];labels=tuple(labels[j] for j in keep)
    report={'operation':'fixed_grid_smoothing','source_fingerprint':source,'processed_fingerprint':voxel_fingerprint(processed),
        'source_shape_zyx':list(model.regions.shape),'processed_shape_zyx':list(processed.regions.shape),
        'spacing_unchanged':True,'physical_extent_unchanged':True,'original_array_unmodified':True,
        'options':json.loads(json.dumps(asdict(o))),'fidelity':comparison,'fidelity_limits_satisfied':bool(passed),
        'volume_balance':balance,
        'continuous_volume_calibration':calibration,
        'nodal_label_tie_corrections':nodal_corrections,
        'component_regularization':regularization,
        'post_smoothing_regularization':final_regularization,
        'preparation_iterations':iteration_records,
        'score_sha256':hashlib.sha256(scores.view(np.uint8)).hexdigest(),'elapsed_seconds':time.perf_counter()-start,
        'minimum_feature_scope':'requested smoothing scale, not guaranteed strict minimum feature thickness; analyze the measured processed geometry',
        'field_scope':'continuous geometry scores; no physical concentration/probability interpretation'}
    if not passed and o.acceptance=='strict':raise ArithmeticError('Smoothing misses explicit fidelity/topology limits: '+json.dumps({'fidelity':comparison,'volume_balance':balance,'continuous_volume_calibration':calibration}))
    return PreparedGeometry(processed,scores,labels,report)


def lower_resolution(model,*,shape=None,factor=None,method='area',allow_phase_loss=False,progress=None):
    """Explicitly fewer voxels, SAME physical box. This is NOT smoothing.

    Area-overlap averaging preserves each phase's integrated score exactly;
    hard output labels can have different fractions and lose thin regions.
    Those effects are reported, and phase loss is rejected by default.
    """
    if not isinstance(model,VoxelModel):raise TypeError('VoxelModel required')
    if (shape is None)==(factor is None):raise ValueError('Specify a target shape OR a reduction factor')
    old=np.array(model.regions.shape)
    if factor is not None:
        if not np.isfinite(factor) or factor<1:raise ValueError('Resolution-reduction factor must be >=1')
        new=np.maximum(1,np.rint(old/factor)).astype(int)
    else:
        if len(shape)!=3 or any(type(x) not in (int,np.int64,np.int32) or x<1 for x in shape):raise ValueError('Explicit positive target integer shape required')
        new=np.array(shape,dtype=int)
    if np.any(new>old):raise ValueError('lower_resolution does not upsample')
    if method not in ('area','nearest'):raise ValueError('Use area or nearest resampling')
    labels=tuple(map(int,np.unique(model.regions)));fields=np.empty((len(labels),*new),np.float32);start=time.perf_counter()
    from scipy import sparse
    matrices=[]
    for before,after in zip(old,new):
        row=[];col=[];values=[]
        for j in range(after):
            left=j*before/after;right=(j+1)*before/after
            for i in range(int(np.floor(left)),min(before,int(np.ceil(right)))):
                overlap=min(right,i+1)-max(left,i)
                if overlap>0:row.append(j);col.append(i);values.append(overlap/(right-left))
        matrices.append(sparse.csr_matrix((values,(row,col)),shape=(after,before)))
    def reduce(a):
        for axis,matrix in enumerate(matrices):
            x=np.moveaxis(a,axis,0);a=np.moveaxis(np.asarray(matrix@x.reshape(x.shape[0],-1)).reshape((matrix.shape[0],*x.shape[1:])),0,axis)
        return a
    with progress_stage('lower resolution / 降低分辨率',progress,estimated_seconds=max(.1,model.regions.size*len(labels)/15e6)) as stage:
        for j,k in enumerate(labels):
            if method=='area':fields[j]=reduce((model.regions==k).astype(np.float32))
            else:
                indices=[np.minimum(old[i]-1,((np.arange(new[i])+.5)*old[i]/new[i]).astype(int)) for i in range(3)]
                fields[j]=model.regions[np.ix_(*indices)]==k
            stage.update(j+1,len(labels),f'region {k}')
    a=_winner(fields,labels).astype(model.regions.dtype,copy=False);present=set(map(int,np.unique(a)));lost=sorted(set(labels)-present)
    if lost and not allow_phase_loss:raise ArithmeticError(f'Resolution reduction removed labels {lost}; choose a finer grid or explicitly allow_phase_loss')
    h=old*np.asarray(model.spacing)/new
    result=from_regions(a,spacing=tuple(h),origin=model.origin,region_names={k:v for k,v in model.region_names.items() if k in present},
        metadata=model.metadata|{'geometry_operation':'lower_resolution','resolution_reduction_applied':True,'original_shape_zyx':old.tolist(),'resampling':method,'lost_regions':lost})
    contact_before=contact_areas(model);contact_after=contact_areas(result)
    components={str(k):{'before':int(ndimage.label(model.regions==k)[1]),'after':int(ndimage.label(a==k)[1])} for k in labels}
    return result,{'operation':'lower_resolution','source_shape_zyx':old.tolist(),'processed_shape_zyx':new.tolist(),
        'spacing_before_zyx_m':list(model.spacing),'spacing_after_zyx_m':h.tolist(),'physical_extent_unchanged':True,
        'lost_regions':lost,'phase_score_fractions':{str(k):float(fields[j].mean()) for j,k in enumerate(labels)},
        'source_phase_fractions':{str(k):float(np.mean(model.regions==k)) for k in labels},
        'phase_hard_fraction_relative_errors':{str(k):float(abs(np.mean(a==k)/np.mean(model.regions==k)-1)) for k in labels},
        'components_6':components,'contact_areas_before_m2':contact_before,'contact_areas_after_m2':contact_after,
        'added_contact_pairs':sorted(set(contact_after)-set(contact_before)),
        'removed_contact_pairs':sorted(set(contact_before)-set(contact_after)),
        'hard_label_fractions':{str(k):float(np.mean(a==k)) for k in labels},'elapsed_seconds':time.perf_counter()-start,
        'scope':'resolution reduction, independent of smoothing; area-averaged scores conserve phase fractions, hard labels need not'}


def generate_voxels(method,*,progress=None,**parameters):
    """Unified, timed geometry-generator entry. Existing APIs stay unchanged."""
    from .interfaces import generate
    size=parameters.get('size',64);count=np.prod(parameters.get('shape',(size,)*3));start=time.perf_counter()
    with progress_stage('voxel generation / 体素生成',progress,estimated_seconds=max(.1,float(count)/2e6)):
        result=generate(method,**parameters)
    result.metadata['generation_elapsed_seconds']=time.perf_counter()-start
    return result
