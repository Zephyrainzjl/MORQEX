"""Unified TOML/array geometry and FEM command line."""
import argparse,json
from . import from_toml,VoxelModel,capabilities

p=argparse.ArgumentParser(prog='python -m morqex.modeling')
sub=p.add_subparsers(dest='action',required=True)
g=sub.add_parser('generate');g.add_argument('config');g.add_argument('--output',required=True)
g.add_argument('--mesh');g.add_argument('--regions',default='1,2');g.add_argument('--preview')
m=sub.add_parser('mesh');m.add_argument('geometry');m.add_argument('--output',required=True)
m.add_argument('--regions',default='1,2');m.add_argument('--slab-depth',type=int,default=2)
sub.add_parser('capabilities');a=p.parse_args()
if a.action=='capabilities':print(json.dumps(capabilities(),indent=2))
else:
    model=from_toml(a.config) if a.action=='generate' else VoxelModel.load(a.geometry)
    if a.action=='generate':
        model.save(a.output)
        if a.preview:model.viewer().save_html(a.preview)
    destination=a.mesh if a.action=='generate' else a.output
    if destination:
        artifact=model.mesh_to(destination,regions=[int(x) for x in a.regions.split(',')],slab_depth=getattr(a,'slab_depth',2))
        print(json.dumps({'hdf5':str(artifact.hdf5),'xdmf':str(artifact.xdmf),'nodes':artifact.nodes,'tetrahedra':artifact.tetrahedra}))
