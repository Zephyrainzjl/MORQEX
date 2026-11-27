"""Unified TOML/array geometry and FEM command line."""
import argparse,json,tomllib
from pathlib import Path
from . import from_toml,mesh_from_toml,VoxelModel,capabilities,mesh,build_tetgen,build_acvd,TetGenOptions,AdaptiveMeshOptions,export_comsol

p=argparse.ArgumentParser(prog='python -m morqex.modeling')
sub=p.add_subparsers(dest='action',required=True)
g=sub.add_parser('generate');g.add_argument('config');g.add_argument('--output',required=True)
g.add_argument('--mesh');g.add_argument('--regions',help='Comma-separated IDs or all; default all, or the TOML recipe selection');g.add_argument('--preview')
m=sub.add_parser('mesh');m.add_argument('geometry');m.add_argument('--output',required=True)
m.add_argument('--regions',help='Comma-separated IDs or all; default all');m.add_argument('--slab-depth',type=int,default=2)
for parser in (g,m):
    parser.add_argument('--method',choices=['streamed_tetrahedral','tetgen','gemsim_adaptive'])
    parser.add_argument('--precision',choices=['coarse','balanced','fine'],default='balanced')
    parser.add_argument('--mesh-preview')
    parser.add_argument('--surface-method',choices=['voxel_plc','marching_cubes'],default='voxel_plc')
    parser.add_argument('--step-size',type=int,default=1)
    parser.add_argument('--comsol',help='Additional COMSOL .mphtxt output')
b=sub.add_parser('build-tetgen');b.add_argument('--workspace',required=True)
c=sub.add_parser('build-acvd');c.add_argument('--workspace',required=True)
sub.add_parser('capabilities');a=p.parse_args()
if a.action=='capabilities':print(json.dumps(capabilities(),indent=2))
elif a.action=='build-tetgen':print(build_tetgen(a.workspace))
elif a.action=='build-acvd':print(build_acvd(a.workspace))
else:
    model=from_toml(a.config) if a.action=='generate' else VoxelModel.load(a.geometry)
    if a.action=='generate':
        model.save(a.output)
        if a.preview:model.viewer().save_html(a.preview)
    destination=a.mesh if a.action=='generate' else a.output
    if destination:
        regions=None if a.regions in (None,'all') else [int(x) for x in a.regions.split(',')]
        recipe=tomllib.loads(Path(a.config).read_text()).get('meshing') if a.action=='generate' and a.method is None else None
        if recipe:
            overrides={'path':destination} if recipe.get('method')=='streamed_tetrahedral' else {}
            if a.regions is not None:overrides['regions']=regions
            result=mesh_from_toml(model,a.config,overrides=overrides)
            if hasattr(result,'save_hdf5'):
                artifact=result.save_hdf5(destination)
                if a.mesh_preview:
                    (result.adaptive_viewer() if result.report.get('pipeline') else result.viewer()).save_html(a.mesh_preview)
            elif hasattr(result,'hdf5'):
                artifact=result
                if a.mesh_preview:artifact.viewer(sampling='uniform').save_html(a.mesh_preview)
            else:raise ValueError('CLI HDF5 export requires streamed_tetrahedral, tetgen or gemsim_adaptive')
        elif a.method=='gemsim_adaptive':
            result=mesh(model,method='gemsim_adaptive',regions=regions,options=AdaptiveMeshOptions.precision(a.precision))
            artifact=result.save_hdf5(destination)
            if a.mesh_preview:result.adaptive_viewer().save_html(a.mesh_preview)
        elif a.method=='tetgen':
            result=mesh(model,method='tetgen',regions=regions,surface_method=a.surface_method,step_size=a.step_size)
            artifact=result.save_hdf5(destination)
            if a.mesh_preview:result.adaptive_viewer().save_html(a.mesh_preview)
        else:
            artifact=model.mesh_to(destination,regions=regions,slab_depth=getattr(a,'slab_depth',2))
            if a.mesh_preview:artifact.viewer(sampling='uniform').save_html(a.mesh_preview)
        if a.comsol:export_comsol(artifact,a.comsol,region_names=model.region_names)
        print(json.dumps({'hdf5':str(artifact.hdf5),'xdmf':str(artifact.xdmf),'nodes':artifact.nodes,'tetrahedra':artifact.tetrahedra}))
