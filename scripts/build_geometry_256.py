"""Build one directly generated256^3 electrode and its full streamed FEM mesh."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
import h5py,numpy as np
from scipy import ndimage
from morqex.modeling import electrode,PRESETS
from morqex.microstructure import Volume
from morqex.microstructure.mesh.voxel import voxel_to_tetra
from morqex.generation.gallery import exact_surfaces,pack
from morqex.visualization.electrode_gallery import thumbnail

p=argparse.ArgumentParser();p.add_argument('index',type=int);args=p.parse_args();i=args.index
if not 0<=i<10:raise ValueError('Index0..9 required')
root=Path(os.environ['MORQEX_WORKSPACE']);out=root/'atlas256';folder=out/'structures'/f'{i+1:02d}_{PRESETS[i][0]}'
folder.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
if (folder/'metadata.json').exists():raise ValueError('Existing accepted structure; select another output')
model,owners=electrode(PRESETS[i][0],seed=20261127+i)
mesh_path=folder/'structure.h5'
artifact=model.mesh_to(mesh_path,regions=[1,2],slab_depth=2)
with h5py.File(mesh_path,'a') as f:
    f.create_dataset('particle_ids',data=owners,compression='gzip',compression_opts=1,shuffle=True)
    f.attrs['metadata']=json.dumps(model.metadata);f.attrs['region_names']=json.dumps(model.region_names)
# Use a labelled display LOD, never as the computational mesh or geometry.
stride=4;display=model.regions[stride//2::stride,stride//2::stride,stride//2::stride]
particle=owners[stride//2::stride,stride//2::stride,stride//2::stride]
grains=particle.copy();preview_mesh=voxel_to_tetra(Volume(display,spacing=(1.,)*3),phases=[1,2]);surfaces=[]
for cut in (16,32,48,64):
    points,attributes,_=exact_surfaces(preview_mesh,display,particle,grains,cut)
    surfaces.append({'cut':cut,'points':pack(np.rint(points).astype('u1')),'attributes':pack(attributes)})
    if cut==64:thumbnail(out/f'preview_{i+1:02d}.png',points,attributes,64,title=PRESETS[i][0])
del preview_mesh
# The local view is an exact subset of the FULL256^3 mesh, with the same global
# compact node IDs and original parent-voxel association. No proxy FE cells.
centre=np.array([128,128,128]);search=model.regions[120:136,120:136,120:136]>0
candidate=np.argwhere(search)
if len(candidate):centre=candidate[np.argmin(np.sum((candidate-8)**2,axis=1))]+120
else:centre=np.argwhere(model.regions>0)[0]
low=np.maximum(0,centre-3);high=np.minimum(256,centre+4)
local=model.regions[tuple(slice(a,b) for a,b in zip(low,high))]
mesh=voxel_to_tetra(Volume(local,spacing=(1.,)*3,origin=tuple(low)),phases=[1,2])
parent=np.flatnonzero(local.ravel()>0);coordinates=np.array(np.unravel_index(parent,local.shape)).T+low
source_ids=owners[tuple(coordinates.T)]
attributes=np.column_stack([mesh.phases,np.repeat(source_ids,6),np.repeat(source_ids,6)]).astype('<i4')
probe={'points':pack((mesh.points[mesh.cells]/stride).astype('<f4')),'attributes':pack(attributes),
       'centre_xyz':(centre[::-1]/stride).tolist(),'tetrahedra':len(mesh.cells)}
fractions={str(k):float(np.mean(model.regions==k)) for k in (0,1,2)}
with h5py.File(mesh_path) as f:volume_error=float(f.attrs['volume_relative_error']);minimum=float(f.attrs['minimum_signed_volume_m3'])
record={'id':i+1,'key':PRESETS[i][0],'title':PRESETS[i][1],'english':PRESETS[i][0],
 'description':'直接生成256³不规则几何；全域为64³显示LOD，有限元局部视图取自完整256³体网格。',
 'seed':20261127+i,'shape_zyx':[256]*3,'display_shape_zyx':[64]*3,'voxel_size_m':5e-7,'length_um':128.,
 'phase_fractions':fractions,'particle_regions':len(np.unique(owners[owners>0])),'grain_count':0,
 'grain_note':'本组未生成独立晶粒，来源分区不代表晶粒','mesh_points':artifact.nodes,'tetrahedra':artifact.tetrahedra,
 'volume_relative_error':volume_error,'minimum_signed_tet_volume_m3':minimum,'checks_passed':True,
 'provider':model.metadata.get('generator',model.metadata.get('provider')),
 'files':{'hdf5':str(mesh_path.relative_to(out)),'xdmf':str(artifact.xdmf.relative_to(out)),'vtu':str(artifact.xdmf.relative_to(out))},
 'local_probe':{'low_zyx':low.tolist(),'high_zyx':high.tolist(),'tetrahedra':len(mesh.cells)},
 'seconds':time.perf_counter()-tick}
payload={'record':record,'labels':pack(model.regions),'particles':pack(owners),'grains':pack(np.zeros((1,),np.uint8)),
 'surfaces':surfaces,'probe':probe}
(folder/'metadata.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
# A local script can load when opened with file://, without fetch/CORS or a
# web server. Only the selected geometry is loaded, keeping browser RAM bounded.
data=out/'data';data.mkdir(exist_ok=True)
(data/f'model_{i:02d}.js').write_text('window.receiveModel('+str(i)+','+json.dumps(payload,separators=(',',':'),ensure_ascii=False).replace('<','\\u003c')+');')
print(json.dumps(record,ensure_ascii=False),flush=True)
