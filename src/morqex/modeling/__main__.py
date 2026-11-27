"""Unified TOML/array geometry and FEM command line."""
import argparse,json,tomllib
from pathlib import Path
from . import from_toml,mesh_from_toml,VoxelModel,capabilities,mesh,build_tetgen,build_acvd,TetGenOptions,AdaptiveMeshOptions,export_comsol,MeshResolution,controlled_tetgen


def preview_mesh(result,path):
    """Show real material interfaces when available, not a tiny cell subset."""
    has_interfaces=getattr(result,'facets',None) is not None
    if hasattr(result,'hdf5'):
        import h5py
        with h5py.File(result.hdf5) as h:has_interfaces='mesh/facets' in h
    view=(result.adaptive_viewer(maximum_embedded_faces=450000,node_cache_bytes=256*2**20)
        if has_interfaces else result.viewer(sampling='uniform'))
    view.save_html(path)


p=argparse.ArgumentParser(prog='python -m morqex.modeling')
sub=p.add_subparsers(dest='action',required=True)
g=sub.add_parser('generate');g.add_argument('config');g.add_argument('--output',required=True)
g.add_argument('--mesh');g.add_argument('--regions',help='Comma-separated IDs or all; default all, or the TOML recipe selection');g.add_argument('--preview')
m=sub.add_parser('mesh');m.add_argument('geometry');m.add_argument('--output',required=True)
m.add_argument('--regions',help='Comma-separated IDs or all; default all');m.add_argument('--slab-depth',type=int,default=2)
for parser in (g,m):
    parser.add_argument('--method',choices=['streamed_tetrahedral','tetgen','gemsim_adaptive','controlled_tetgen'])
    parser.add_argument('--precision',choices=['coarse','balanced','fine','very_fine'],default='balanced')
    parser.add_argument('--target-cells',type=int,help='Minimum density request via V/target_cells, not an exact cell count')
    parser.add_argument('--maximum-cell-volume',type=float,help='Maximum Tet4 volume in SI m3')
    parser.add_argument('--nominal-edge',type=float,help='SI equilateral-equivalent edge used to set a volume limit; not a strict edge-length bound')
    parser.add_argument('--surface-fraction',type=float,help='Adaptive retained surface-vertex fraction in(0,1]')
    parser.add_argument('--phase-volume-error',type=float,help='Maximum adaptive source phase-volume relative error')
    parser.add_argument('--maximum-cells',type=int,default=2000000,help='Hard native output budget')
    parser.add_argument('--mesh-preview')
    parser.add_argument('--surface-method',choices=['voxel_plc','marching_cubes','surface_nets','gemsim_adaptive'],default='voxel_plc')
    parser.add_argument('--step-size',type=int,default=1)
    parser.add_argument('--comsol',help='Additional COMSOL .mphtxt output')
    parser.add_argument('--length-unit',choices=['m','cm','mm','um','nm'],help='Numeric export unit; overrides TOML [exports], otherwise defaults to m; source geometry remains SI')
    parser.add_argument('--export-dir',help='Additional region-preserving COMSOL/VTU/Gmsh/Medit/Abaqus bundle')
b=sub.add_parser('build-tetgen');b.add_argument('--workspace',required=True)
c=sub.add_parser('build-acvd');c.add_argument('--workspace',required=True)
d=sub.add_parser('build-mmg');d.add_argument('--workspace',required=True)
e=sub.add_parser('build-backends');e.add_argument('config');e.add_argument('--workspace')
r=sub.add_parser('remesh');r.add_argument('mesh');r.add_argument('--config',required=True);r.add_argument('--output',required=True);r.add_argument('--preview');r.add_argument('--workspace')
q=sub.add_parser('partition');q.add_argument('geometry');q.add_argument('--directory',required=True);q.add_argument('--slab-depth',type=int,default=32);q.add_argument('--axis',type=int,choices=[0,1,2],default=0);q.add_argument('--surface-method',choices=['voxel_plc','surface_nets'],default='surface_nets');q.add_argument('--workspace');q.add_argument('--preview');q.add_argument('--content-addressed',action='store_true',help='Keep independent source/configuration/algorithm checkpoints under this directory')
sub.add_parser('capabilities');a=p.parse_args()
if a.action=='capabilities':print(json.dumps(capabilities(),indent=2))
elif a.action=='build-tetgen':print(build_tetgen(a.workspace))
elif a.action=='build-acvd':print(build_acvd(a.workspace))
elif a.action=='build-mmg':
    from .mmg_backend import build_mmg
    print(build_mmg(a.workspace))
elif a.action=='build-backends':
    from .native_build import build_native_from_toml
    print(json.dumps(build_native_from_toml(a.config,workspace=a.workspace),indent=2))
elif a.action=='remesh':
    import h5py
    from .meshing import MeshArtifact
    from .interfaces import remesh_from_toml
    artifact=remesh_from_toml(MeshArtifact.load(a.mesh),a.config,workspace=a.workspace,output=a.output)
    if a.preview:preview_mesh(artifact,a.preview)
    with h5py.File(artifact.hdf5) as h:print(h.attrs['report'])
elif a.action=='partition':
    from .partition_workflow import partitioned_tetgen
    result=partitioned_tetgen(VoxelModel.load(a.geometry),a.directory,surface_method=a.surface_method,slab_depth=a.slab_depth,axis=a.axis,workspace=a.workspace,content_addressed=a.content_addressed)
    if a.preview:
        from morqex.visualization.adaptive_mesh import AdaptiveMeshView
        AdaptiveMeshView(result,maximum_embedded_faces=450000,node_cache_bytes=256*2**20).save_html(a.preview)
    print(json.dumps({'hdf5':str(result.hdf5),'nodes':result.nodes,'tetrahedra':result.tetrahedra}))
else:
    model=from_toml(a.config) if a.action=='generate' else VoxelModel.load(a.geometry)
    export_settings=tomllib.loads(Path(a.config).read_text()).get('exports',{}) if a.action=='generate' else {}
    unit=a.length_unit or export_settings.get('length_unit','m')
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
                if a.mesh_preview:preview_mesh(result,a.mesh_preview)
            elif hasattr(result,'hdf5'):
                artifact=result
                if a.mesh_preview:preview_mesh(artifact,a.mesh_preview)
                if Path(destination).resolve()!=artifact.hdf5.resolve():
                    # A checkpointed mesher owns its cache directory. Publish
                    # the explicitly requested CLI artifact without changing
                    # that cache or silently ignoring --mesh.
                    import shutil
                    from .voxel import external_path
                    from .meshing import MeshArtifact
                    from xml.sax.saxutils import escape
                    target=external_path(destination);target.parent.mkdir(parents=True,exist_ok=True)
                    pending=target.with_name(target.name+'.pending')
                    if target.exists() or pending.exists() or target.with_suffix('.xdmf').exists():raise FileExistsError('Requested published mesh/temporary/XDMF already exists')
                    shutil.copyfile(artifact.hdf5,pending);pending.rename(target)
                    target.with_suffix('.xdmf').write_text(artifact.xdmf.read_text().replace(escape(artifact.hdf5.name)+':',escape(target.name)+':'))
                    artifact=MeshArtifact.load(target)
            else:raise ValueError('CLI HDF5 export requires streamed_tetrahedral, tetgen or gemsim_adaptive')
        elif a.method=='gemsim_adaptive':
            result=controlled_tetgen(model,regions=regions,resolution=MeshResolution(a.precision,a.target_cells,a.maximum_cell_volume,
                a.nominal_edge,a.surface_fraction,a.phase_volume_error),tetgen_options=TetGenOptions.for_adaptive(maximum_cells=a.maximum_cells))
            artifact=result.save_hdf5(destination)
            if a.mesh_preview:preview_mesh(result,a.mesh_preview)
        elif a.method in ('tetgen','controlled_tetgen'):
            result=controlled_tetgen(model,regions=regions,surface_method=a.surface_method,resolution=MeshResolution(a.precision,
                a.target_cells,a.maximum_cell_volume,a.nominal_edge,a.surface_fraction,a.phase_volume_error),
                tetgen_options=TetGenOptions.for_adaptive(maximum_cells=a.maximum_cells),step_size=a.step_size)
            artifact=result.save_hdf5(destination)
            if a.mesh_preview:preview_mesh(result,a.mesh_preview)
        else:
            artifact=model.mesh_to(destination,regions=regions,slab_depth=getattr(a,'slab_depth',2))
            if a.mesh_preview:preview_mesh(artifact,a.mesh_preview)
        if a.comsol:export_comsol(artifact,a.comsol,region_names=model.region_names,length_unit=unit,
            selections=model.metadata.get('selection_groups'))
        export_destination=a.export_dir
        if export_destination is None and export_settings.get('directory'):
            export_destination=(Path(a.config).resolve().parent/export_settings['directory']).resolve()
        if export_destination:
            from .exchange import export_bundle
            export_bundle(artifact,export_destination,region_names=model.region_names,length_unit=unit,
                selections=model.metadata.get('selection_groups'),formats=export_settings.get('formats',('comsol','vtu','gmsh','medit','abaqus')))
        import h5py
        with h5py.File(artifact.hdf5) as h:report=json.loads(h.attrs.get('report','{}'))
        print(json.dumps({'hdf5':str(artifact.hdf5),'xdmf':str(artifact.xdmf),'nodes':artifact.nodes,'tetrahedra':artifact.tetrahedra,
            'quality_limits_satisfied':report.get('quality_limits_satisfied'),'quality':report.get('quality'),
            'validation_scope':'mesh geometry and measured shape; no PDE convergence claim'}))
