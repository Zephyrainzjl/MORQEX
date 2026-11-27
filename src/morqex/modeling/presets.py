"""Ten different irregular electrode morphology families, directly rasterized."""
import numpy as np
from scipy import ndimage
from .voxel import Shape,from_shapes,from_regions

PRESETS=[('granular','不规则颗粒堆积'),('rods','不规则长棒电极'),('platelets','弯曲片状电极'),
 ('dendritic','分枝树状骨架'),('hollow','多孔壳层电极'),('agglomerates','层级团聚电极'),
 ('fibres','缠结长纤维网络'),('bicontinuous','双连续随机电极'),('lamellar','梯度波纹层片'),
 ('channels','曲折定向孔道')]


def electrode(preset,*,size=256,spacing=5e-7,seed=20261127,cbd_fraction=.12):
    """Return (VoxelModel, source_region_ids), without any neural generator.

    All dimensions are evaluated on the requested grid; no voxel replication.
    Seeded implicit warping makes every morphology irregular. Sources may
    overlap and are geometrically bonded; IDs record final ownership only.
    """
    names=dict(PRESETS)
    if preset not in names:raise ValueError('Unknown preset: '+', '.join(names))
    if not 32<=size<=256 or not 0<cbd_fraction<.3:raise ValueError('Size32..256 and CBD fraction in(0,.3) required')
    rng=np.random.default_rng(seed);L=size*spacing;shapes=[];number=0
    def add(kind,centre,axes,angle=0.,region=None):
        nonlocal number
        number+=1;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        shapes.append(Shape(kind,number if region is None else region,tuple(np.asarray(centre)*L),tuple(np.asarray(axes)*L),tuple(map(tuple,R)),.22))
    if preset in ('granular','hollow'):
        for _ in range(120 if preset=='granular' else 65):
            centre=rng.uniform(.03,.97,3);r=rng.uniform(.065,.13)
            add('ellipsoid',centre,r*rng.uniform(.7,1.2,3),rng.uniform(-np.pi,np.pi))
            if preset=='hollow':add('ellipsoid',centre,np.full(3,r*.55),region=0)
    elif preset=='rods':
        for _ in range(75):add('capsule',rng.uniform(.02,.98,3),[.035,.035,rng.uniform(.16,.38)],rng.uniform(-.45,.45))
    elif preset=='platelets':
        for _ in range(80):add('platelet',rng.uniform(0,1,3),[rng.uniform(.012,.025),rng.uniform(.10,.19),rng.uniform(.12,.25)],rng.uniform(-.8,.8))
    elif preset in ('fibres','dendritic'):
        for trunk in range(24 if preset=='fibres' else 14):
            centre=rng.uniform(.05,.95,3);direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
            previous=centre.copy()
            for k in range(9):
                if preset=='fibres':direction=.8*direction+.3*rng.normal(size=3);direction/=np.linalg.norm(direction)
                end=previous+direction*.11;v=end-previous;axis=np.array([0,0,1.]);v/=np.linalg.norm(v)
                # Rodrigues maps the local capsule axis to the segment.
                cross=np.cross(axis,v);c=axis@v
                if c<-.999:rotation=np.diag([1.,-1.,-1.])
                else:
                    K=np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
                    rotation=np.eye(3)+K+K@K/(1+c)
                number+=1;radius=.023 if preset=='fibres' else .036
                shapes.append(Shape('capsule',number,tuple((previous+end)/2*L),(radius*L,radius*L,.06*L),tuple(map(tuple,rotation)),.2))
                if preset=='dendritic' and k%2==0:
                    add('capsule',end,[.021,.021,.16],rng.uniform(-2.8,2.8))
                previous=end
    elif preset=='agglomerates':
        for _ in range(22):
            centre=rng.uniform(.05,.95,3)
            for _ in range(12):add('ellipsoid',centre+rng.normal(0,.06,3),rng.uniform(.04,.075,3),rng.uniform(-3,3))
    if shapes:
        raw=from_shapes(shapes,shape=(size,)*3,spacing=(spacing,)*3)
        owners=raw.regions;active=owners>0
    else:
        z,y,x=np.ogrid[:size,:size,:size];z=z.astype(np.float32)/size;y=y.astype(np.float32)/size;x=x.astype(np.float32)/size
        if preset=='bicontinuous':
            # Use the bundled statistical generator's physical correlation law,
            # evaluated directly on the requested lattice, with exact counts.
            from morqex.generation.statistics import StatisticsCard,generate
            card=StatisticsCard({0:.30,1:.56,2:.14},{0:(L/18,)*3,1:(L/11,L/16,L/13),2:(L/25,)*3},
                {0:'pore',1:'active',2:'cbd'},'Irregular256 atlas: declared stochastic medium')
            generated=generate(card,(size,)*3,(spacing,)*3,seed=seed,require_percolation=False)
            labels=generated.labels.astype(np.uint8);owners=ndimage.label(labels==1)[0].astype(np.uint16)
            return from_regions(labels,spacing=(spacing,)*3,region_names={0:'pore',1:'active',2:'CBD'},metadata=generated.provenance),owners
        noise=ndimage.gaussian_filter(rng.standard_normal((size,)*3,dtype=np.float32),max(1,size/35))
        noise/=max(float(noise.std()),1e-12)
        if preset=='lamellar':
            field=np.sin(2*np.pi*(8*z+1.3*np.sin(2*np.pi*x)+.5*np.sin(3*np.pi*y)))-.22*noise+.35*(x-.5)
            active=field>-.15
        else:
            warp_z=z+.027*np.sin(4*np.pi*x)+.012*noise;warp_y=y+.025*np.sin(3*np.pi*x+.5)
            field=np.cos(2*np.pi*6*warp_z)+np.cos(2*np.pi*6*warp_y)
            active=field<.22
        owners=ndimage.label(active)[0].astype(np.uint16)
    distance=ndimage.distance_transform_edt(~active)
    noise=ndimage.gaussian_filter(rng.standard_normal(active.shape,dtype=np.float32),max(1,size/45))
    noise/=max(float(noise.std()),1e-12)
    score=distance+noise*(size/180)
    available=np.flatnonzero(~active);count=int(cbd_fraction*active.size)
    if count>=len(available):raise ValueError('Preset leaves insufficient pore space for CBD')
    selected=available[np.argpartition(score.ravel()[available],count-1)[:count]]
    labels=np.zeros(active.shape,np.uint8);labels[active]=1;labels.ravel()[selected]=2
    metadata={'preset':preset,'seed':seed,'size':size,'spacing_m':spacing,'source_shapes':len(shapes),
        'irregularity':.22,'generator':'MORQEX modeling Shape + bundled MicroStructPy geometry / integrated statistical generator',
        'scope':'Synthetic irregular geometry; intersecting sources are bonded. No calibrated electrode chemistry or guaranteed RVE.'}
    return from_regions(labels,spacing=(spacing,)*3,region_names={0:'pore',1:'active',2:'CBD'},metadata=metadata),owners
