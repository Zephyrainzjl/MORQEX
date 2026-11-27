"""Ten distinct256³ synthetic electrodes with explicitly resolved features.

Uses the integrated Shape/MicroStructPy rasterizer and continuum fields. These
are new recipes, not repaired CT data or replacements for the older presets.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from .voxel import Shape,from_shapes,from_regions
from .presets import PRESETS


def resolved_electrode(preset,*,size=256,spacing=5e-7,seed=20261127,
                       cbd_thickness_voxels=4,minimum_feature_voxels=4,cbd_bridges=True):
    if preset not in dict(PRESETS):raise ValueError('Unknown electrode preset')
    if type(size) is not int or not 96<=size<=256:raise ValueError('Resolved recipes require size96..256; use from_shapes for smaller custom geometry')
    if not np.isfinite(spacing) or spacing<=0:raise ValueError('Positive SI spacing required')
    if not 2<=minimum_feature_voxels<=size/12 or not 1<=cbd_thickness_voxels<=size/16:raise ValueError('Resolved feature / coating thickness outside budget')
    rng=np.random.default_rng(seed);L=size*spacing;shapes=[];centres=[]
    def add(kind,c,a,angle=0.,region=1):
        rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        shapes.append(Shape(kind,region,tuple(np.array(c)*L),tuple(np.array(a)*L),tuple(map(tuple,rotation)),.12))
        if region==1:centres.append(c)
    if preset in ('granular','hollow'):
        coordinates=[.30,.70] if preset=='hollow' else [.25,.5,.75]
        grid=np.array(np.meshgrid(*[coordinates]*3,indexing='ij')).reshape(3,-1).T
        for c in grid:
            c=c+rng.uniform(-.022,.022,3);radius=rng.uniform(.145,.16) if preset=='hollow' else rng.uniform(.082,.102)
            add('ellipsoid',c,[radius,radius*rng.uniform(.85,1.1),radius*rng.uniform(.85,1.1)],rng.uniform(-.5,.5))
            if preset=='hollow':add('ellipsoid',c,[radius*.48]*3,region=0)
    elif preset=='rods':
        for z in [.26,.5,.74]:
            for y in [.26,.5,.74]:add('capsule',[z,y,.5],[.055,.055,rng.uniform(.22,.27)],rng.uniform(-.17,.17))
    elif preset=='platelets':
        for z in [.25,.5,.75]:
            for y in [.3,.7]:add('platelet',[z,y,.5],[.045,.14,.26],rng.uniform(-.18,.18))
    elif preset=='dendritic':
        for z in [.3,.7]:
            add('capsule',[z,.5,.5],[.055,.055,.28])
            for x in [.3,.5,.7]:
                add('capsule',[z,.5,x],[.05,.05,.20],np.pi/2)
                for sign in [-1,1]:add('capsule',[z,.5+sign*.14,x],[.04,.04,.12],np.pi/2)
    elif preset=='agglomerates':
        for c in [[.3,.3,.3],[.7,.3,.7],[.3,.7,.7],[.7,.7,.3]]:
            add('ellipsoid',c,[.115,.10,.12])
            for offset in [[.08,0,0],[-.07,0,0],[0,.08,0],[0,0,.08]]:
                add('ellipsoid',np.array(c)+offset,[.09,.075,.085],rng.uniform(-.5,.5))
    else:
        z,y,x=np.ogrid[:size,:size,:size];z=(z+.5)/size;y=(y+.5)/size;x=(x+.5)/size
        if preset=='fibres':
            active=np.zeros((size,)*3,bool)
            for c in [.25,.5,.75]:
                active|=(z-c-.045*np.sin(2*np.pi*x))**2+(y-.5-.20*np.sin(2*np.pi*x+c*5))**2<.043**2
                active|=(y-c-.04*np.sin(2*np.pi*x+1))**2+(z-.5-.2*np.cos(2*np.pi*x+c*4))**2<.043**2
        elif preset=='bicontinuous':
            phase=[rng.uniform(-.4,.4) for _ in range(3)]
            Z=2*np.pi*2.2*z+phase[0];Y=2*np.pi*2.2*y+phase[1];X=2*np.pi*2.2*x+phase[2]
            field=np.sin(X)*np.cos(Y)+np.sin(Y)*np.cos(Z)+np.sin(Z)*np.cos(X)
            active=field>.12
        elif preset=='lamellar':
            active=np.sin(2*np.pi*(3.2*z+.22*np.sin(2*np.pi*x)+.10*np.cos(2*np.pi*y)))>.12
        else:
            active=np.ones((size,)*3,bool)
            for a in [.28,.5,.72]:
                for b in [.3,.7]:
                    active&=(z-a-.035*np.sin(2*np.pi*x))**2+(y-b-.04*np.sin(2*np.pi*x+1))**2>.065**2
        centres=[[.5,.5,.5]]
    if shapes:active=from_shapes(shapes,shape=(size,)*3,spacing=(spacing,)*3).regions==1
    # Generation regularity is explicit and occurs before saving the source
    # labels. No imported CT segmentation is silently smoothed by the mesher.
    smooth=max(1,int(minimum_feature_voxels)//2)
    structure=ndimage.generate_binary_structure(3,1)
    active=ndimage.binary_opening(active,structure=structure,iterations=smooth)
    active=ndimage.binary_closing(active,structure=structure,iterations=smooth)
    margin=max(int(cbd_thickness_voxels)+2,int(size*.065))
    for axis in range(3):
        view=np.moveaxis(active,axis,0);view[:margin]=False;view[-margin:]=False
    cbd=ndimage.distance_transform_edt(~active)<=cbd_thickness_voxels
    if cbd_bridges and len(centres)>1:
        c=np.asarray(centres);mst=minimum_spanning_tree(distance_matrix(c,c)).tocoo();bridges=[]
        for a,b in zip(mst.row,mst.col):
            delta=c[b]-c[a];length=np.linalg.norm(delta);axis=delta/length;local=np.array([0.,0.,1.]);cross=np.cross(local,axis);co=local@axis
            if co<-.999:rotation=np.diag([1.,-1.,-1.])
            else:
                k=np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
                rotation=np.eye(3)+k+k@k/max(1+co,1e-12)
            radius=max(2,cbd_thickness_voxels*.7)*spacing
            bridges.append(Shape('capsule',1,tuple((c[a]+c[b])/2*L),(radius,radius,length*L/2),tuple(map(tuple,rotation))))
        cbd|=from_shapes(bridges,shape=(size,)*3,spacing=(spacing,)*3).regions>0
    labels=np.zeros(active.shape,np.uint8);labels[cbd]=2;labels[active]=1
    pore_components,pore_count=ndimage.label(labels==0)
    counts=np.bincount(pore_components.ravel());small=counts<minimum_feature_voxels**3;small[0]=False
    filled=small[pore_components];filled_count=int(filled.sum());labels[filled]=2
    metadata={'generator':'resolved MORQEX Shape/MicroStructPy + implicit3D fields','preset':preset,'seed':seed,
        'phase_order':[0,2,1],
        'full_shape':(size,)*3,'minimum_feature_voxels':minimum_feature_voxels,
        'cbd_thickness_voxels':cbd_thickness_voxels,'cbd_bridges':cbd_bridges,'generation_open_close_iterations':smooth,
        'scope':'synthetic geometry with resolved features, not CT or calibrated material data; CBD is a separate effective phase',
        'cbd_geometry':'coating + optional physical capsule bridges; no conductivity or electrochemical accessibility inferred'}
    metadata['unresolved_enclosed_pore_voxels_filled_with_cbd']=filled_count
    return from_regions(labels,spacing=(spacing,)*3,region_names={0:'pore',1:'active',2:'CBD'},metadata=metadata)
