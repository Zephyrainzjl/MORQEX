"""Dense native3D electrodes and explicit multilayer domain composition.

The primitives use MORQEX's source-integrated Shape/MicroStructPy rasterizer.
Fractions are synthetic design inputs, never inferred chemical material labels.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
from .voxel import Shape,from_shapes,from_regions

DENSE_MORPHOLOGIES=('granular','rods','platelets','agglomerates','bicontinuous')


def _top_fraction(score,allowed,count):
    ids=np.flatnonzero(allowed);values=score.ravel()[ids]
    if not 0<count<len(ids):raise ValueError('Requested fraction leaves an empty material phase')
    chosen=ids[np.argpartition(values,len(values)-count)[-count:]]
    mask=np.zeros(allowed.shape,bool);mask.ravel()[chosen]=True;return mask


def dense_electrode(morphology='granular',*,size=256,spacing=5e-7,seed=20261127,
                    active_fraction=.55,cbd_fraction=.11,particle_scale_voxels=17.,irregularity=.22):
    if morphology not in DENSE_MORPHOLOGIES:raise ValueError('Unknown dense morphology')
    if type(size) is not int or not 48<=size<=256:raise ValueError('Dense generators support48..256³')
    if not 0<active_fraction<1 or not 0<cbd_fraction<1-active_fraction:raise ValueError('Positive phase fractions summing to less than1 required')
    if not np.isfinite(spacing) or spacing<=0 or not 3<=particle_scale_voxels<=256/5 or not 0<=irregularity<.35:
        raise ValueError('Positive SI spacing and resolved shape parameters required')
    rng=np.random.default_rng(seed);scale=particle_scale_voxels*size/256;scale=max(3.,scale);objects=[]
    # Poisson Boolean coverage estimate determines hundreds of true3D sources.
    # Exact requested phase counts are imposed by a recorded signed-distance
    # offset after rasterization, not by tiling/copying a smaller sample.
    volume=4*np.pi/3*scale**3
    count=max(12,int(-np.log1p(-active_fraction)*size**3/volume))
    def add(c,axes,kind='ellipsoid'):
        rotation=Rotation.random(random_state=rng).as_matrix()
        objects.append(Shape(kind,1,tuple(np.asarray(c)*spacing),tuple(np.asarray(axes)*spacing),
            tuple(map(tuple,rotation)),irregularity))
    if morphology!='bicontinuous':
        for _ in range(count):
            centre=rng.uniform(0,size,3);r=scale*np.clip(rng.lognormal(0,.25),.55,1.65)
            if morphology=='granular':add(centre,r*rng.uniform(.75,1.2,3))
            elif morphology=='rods':add(centre,[r*.5,r*.5,r*1.6],'capsule')
            elif morphology=='platelets':add(centre,[r*.32,r*1.2,r*1.6],'platelet')
            else:
                add(centre,[r*.9]*3)
                for _ in range(4):add(centre+rng.normal(0,r*.65,3),r*rng.uniform(.4,.7,3))
        initial=from_shapes(objects,shape=(size,)*3,spacing=(spacing,)*3).regions.astype(bool)
        field=(ndimage.distance_transform_edt(initial)-ndimage.distance_transform_edt(~initial)).astype(np.float32)
    else:
        field=rng.standard_normal((size,)*3,dtype=np.float32)
        field=ndimage.gaussian_filter(field,sigma=(scale*.28,scale*.45,scale*.65),mode='wrap')
        field/=max(float(field.std()),1e-20)
        z,y,x=np.ogrid[:size,:size,:size]
        field+=.65*np.sin(2*np.pi*x/(scale*3))*np.cos(2*np.pi*y/(scale*3))+.35*(z/size-.5)
    noise=ndimage.gaussian_filter(rng.standard_normal(field.shape,dtype=np.float32),max(1.,scale*.22),mode='wrap')
    noise/=max(float(noise.std()),1e-20)
    field+=noise*.025
    active=_top_fraction(field,np.ones(field.shape,bool),round(active_fraction*field.size))
    threshold=float(field[active].min());del field
    distance=ndimage.distance_transform_edt(~active).astype(np.float32)
    # Patchy CBD lies preferentially near particles, with genuine geometric
    # bridges. It is not an impermeable complete coating or a conductivity fit.
    cbd_score=noise*1.8-distance/max(2.,scale*.22)
    cbd=_top_fraction(cbd_score,~active,round(cbd_fraction*active.size))
    labels=np.zeros(active.shape,np.uint8);labels[active]=1;labels[cbd]=2
    return from_regions(labels,spacing=(spacing,)*3,region_names={0:'cathode_electrolyte',1:'cathode_active',2:'cathode_CBD'},
        metadata={'generator':'MORQEX native dense Shape/MicroStructPy and continuum-field generator',
            'morphology':morphology,'seed':seed,'native_shape_sources':len(objects),
            'particle_scale_voxels_at256':particle_scale_voxels,'irregularity':irregularity,
            'active_fraction_requested':active_fraction,'cbd_fraction_requested':cbd_fraction,
            'signed_distance_or_field_threshold':threshold,'no_tiling':True,'source_shape_zyx':[size]*3,
            'scope':'synthetic design geometry; not CT reconstruction or calibrated electrochemical material',
            'fraction_policy':'explicit signed-distance/field level-set calibration and near-particle patchy CBD'})


def porous_layer(shape,*,spacing,porosity=.45,correlation_voxels=3.,seed=0,name='separator'):
    if len(shape)!=3 or any(type(k) is not int or k<2 for k in shape) or np.prod(shape)>256**3:raise ValueError('Explicit3D layer budget required')
    if not 0<porosity<1 or correlation_voxels<=0:raise ValueError('Physical porosity and positive correlation required')
    rng=np.random.default_rng(seed)
    score=ndimage.gaussian_filter(rng.standard_normal(shape,dtype=np.float32),correlation_voxels,mode='wrap')
    pore=_top_fraction(score,np.ones(shape,bool),round(porosity*np.prod(shape)))
    labels=np.ones(shape,np.uint8);labels[pore]=0
    return from_regions(labels,spacing=spacing,region_names={0:name+'_electrolyte',1:name+'_solid'},
        metadata={'generator':'native correlated porous layer','seed':seed,'porosity_requested':porosity,'correlation_voxels':correlation_voxels})


def coated_electrode(morphology='granular',*,size=256,spacing=5e-7,seed=20261127,
                      active_fraction=.5,cbd_thickness_voxels=3,pore_buffer_voxels=7,particle_scale_voxels=17.):
    """Dense resolved coating geometry; independent of the patchy-CBD recipe.

    CBD is an effective porous composite domain, not automatically an ion
    blocking coating. Its volume fraction is measured, not prescribed.
    """
    if not 2<=cbd_thickness_voxels<pore_buffer_voxels<size/8:raise ValueError('Resolve the coating and outer pore buffer explicitly')
    base=dense_electrode(morphology,size=size,spacing=spacing,seed=seed,active_fraction=active_fraction,
        cbd_fraction=.05,particle_scale_voxels=particle_scale_voxels)
    initial=base.regions==1;field=(ndimage.distance_transform_edt(initial)-ndimage.distance_transform_edt(~initial)).astype(np.float32)
    rng=np.random.default_rng(seed+1024);noise=ndimage.gaussian_filter(rng.standard_normal(field.shape,dtype=np.float32),2)
    field+=noise*.02;allowed=np.zeros(field.shape,bool);b=pore_buffer_voxels;allowed[b:-b,b:-b,b:-b]=True
    active=_top_fraction(field,allowed,round(active_fraction*field.size))
    distance=ndimage.distance_transform_edt(~active)
    cbd=(distance<=cbd_thickness_voxels)&~active
    labels=np.zeros(active.shape,np.uint8);labels[active]=1;labels[cbd]=2
    component,n=ndimage.label(labels==0);counts=np.bincount(component.ravel())
    tiny=counts<27;tiny[0]=False;changed=tiny[component];labels[changed]=2
    return from_regions(labels,spacing=(spacing,)*3,region_names=base.region_names,
        metadata=base.metadata|{'generator':'native dense resolved-CBD-coating geometry',
            'phase_order':[0,2,1],'cbd_fraction_requested':None,'cbd_thickness_voxels':cbd_thickness_voxels,
            'pore_buffer_voxels':pore_buffer_voxels,'synthetic_unresolved_pore_voxels_filled_with_CBD':int(changed.sum()),
            'scope':'synthetic resolved geometry; CBD effective porous-composite coating; no ionic/electronic property inferred'})


def stack_layers(layers,*,axis=0,origin=(0.,0.,0.)):
    """Conforming voxel assembly; every layer×phase receives its own domain ID.

    layers is [(ASCII_name,VoxelModel),...]. No material type is guessed from
    a gray value. Electrolytes in different layers remain independently named.
    """
    import re
    from .voxel import VoxelModel
    if axis not in (0,1,2) or not layers or len(set(name for name,_ in layers))!=len(layers):raise ValueError('Distinct layer names and a ZYX stacking axis required')
    first=layers[0][1];chunks=[];names={};groups={};slices={};maps={};cursor=0;next_id=1
    total=sum(model.regions.size for _,model in layers)
    if total*2>128*2**20:raise ValueError('Stack label array exceeds128MiB budget')
    for name,model in layers:
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',name) or not isinstance(model,VoxelModel):raise ValueError('ASCII layer name and VoxelModel required')
        if not np.allclose(model.spacing,first.spacing,rtol=1e-12,atol=0) or any(model.regions.shape[a]!=first.regions.shape[a] for a in range(3) if a!=axis):
            raise ValueError('Layer spacing and transverse voxel shapes must match; no implicit resampling')
        labels=np.empty(model.regions.shape,np.uint16);mapping={}
        for k in np.unique(model.regions):
            k=int(k);mapping[k]=next_id;labels[model.regions==k]=next_id
            phase=model.region_names[k];names[next_id]=phase if phase==name or phase.startswith(name+'_') else name+'_'+phase
            next_id+=1
        groups[name]=list(mapping.values());maps[name]=mapping
        slices[name]=[cursor,cursor+model.regions.shape[axis]];cursor=slices[name][1];chunks.append(labels)
    result=np.concatenate(chunks,axis=axis)
    return from_regions(result,spacing=first.spacing,origin=origin,region_names=names,
        metadata={'generator':'explicit full voxel layer assembly','stack_axis_zyx':axis,'layer_slices':slices,
            'layer_region_ids':groups,'layer_source_region_maps':maps,
            'selection_groups':{name:{'dimension':3,'regions':ids} for name,ids in groups.items()},
            'source_shape_zyx':list(result.shape),
            'interface_policy':'one shared voxel grid; bonded conforming FE interfaces unless separate contact/cohesive topology is later constructed'})


def uniform_layer(shape,*,spacing,name='solid',region=1):
    if len(shape)!=3 or any(type(k) is not int or k<1 for k in shape) or np.prod(shape)>256**3:
        raise ValueError('Positive explicit layer shape within256³ voxels required')
    if type(region) is not int or not 0<=region<=65535:raise ValueError('Nonnegative16-bit region ID required')
    return from_regions(np.full(shape,region,np.uint16),spacing=spacing,region_names={region:name},metadata={'generator':'uniform named layer'})
