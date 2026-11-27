"""General resumable whole-volume conforming TetGen pipeline.

This is the packaged API behind large-model workflows, not a list of named
gallery geometries. Run under an external process-tree resource guard. It
serializes native workers and never changes source labels to make meshing pass.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,os,subprocess,sys,time
import numpy as np


def _surface(path):
    import h5py
    from .tetgen import SurfacePLC
    with h5py.File(path) as f:
        return SurfacePLC(f['points'][:],f['triangles'][:],{int(k):v for k,v in json.loads(f.attrs['seeds']).items()},f['markers'][:],metadata=json.loads(f.attrs['metadata']))


def _configuration(model,method,slab_depth,axis,options,maximum_facets):
    from .tetgen_backend import identity
    digest=hashlib.sha256(np.ascontiguousarray(model.regions).view(np.uint8)).hexdigest()
    code=hashlib.sha256()
    for name in ('partition_workflow.py','partitioned.py','surface_nets.py','volume_projection.py','tetgen.py','interface_labelling.py'):
        code.update(name.encode()+Path(__file__).with_name(name).read_bytes())
    return {'protocol_version':1,'algorithm_source_sha256':code.hexdigest(),'native_backend':identity(),
        'source_voxel_sha256':digest,'shape_zyx':list(model.regions.shape),
        'spacing_zyx_m':list(map(float,model.spacing)),'origin_zyx_m':list(map(float,model.origin)),
        'region_names':{str(k):v for k,v in model.region_names.items()},'surface_method':method,
        'slab_depth':slab_depth,'axis_zyx':axis,'tetgen_options':asdict(options),'maximum_facets':maximum_facets}


def partitioned_tetgen(model,directory,*,surface_method='surface_nets',slab_depth=32,axis=0,
                       tetgen_options=None,maximum_facets=16000000,workspace=None,maximum_reconciliations=4,regions=None,
                       content_addressed=False):
    """Mesh a complete VoxelModel, including pore/electrolyte labels.

    axis uses ZYX; physical coordinates remain XYZ/metres. Finite slabs share
    actual material-labelled cap triangles. This is a geometric-conformity
    contract, not guaranteed success for arbitrary segmentation or mesh shape.
    Every input/config fingerprint must match before checkpoint reuse.
    """
    from .voxel import VoxelModel,external_path
    from .tetgen import TetGenOptions
    from .tetgen_backend import library_path
    from .meshing import MeshArtifact
    if not isinstance(model,VoxelModel):raise TypeError('VoxelModel required')
    if regions is not None and set(map(int,regions))!=set(map(int,np.unique(model.regions))):
        raise ValueError('Partitioning requires the complete labelled domain, including pore/electrolyte phases')
    if surface_method not in ('voxel_plc','surface_nets'):raise ValueError('Choose shared voxel_plc or surface_nets')
    if axis not in (0,1,2) or type(slab_depth) is not int or slab_depth<2:raise ValueError('ZYX axis and slab_depth>=2 required')
    if type(maximum_reconciliations) is not int or maximum_reconciliations<1:raise ValueError('Positive reconciliation budget required')
    if type(content_addressed) is not bool:raise ValueError('content_addressed must be a bool')
    options=tetgen_options or TetGenOptions.for_adaptive(preserve_surface=True,merge_coplanar=False,
        region_labelling='seeds' if surface_method=='voxel_plc' else 'interfaces',coplanarity_tolerance=1e-15,
        maximum_cells=8000000,maximum_steiner=800000,native_timeout_seconds=240)
    if not options.preserve_surface:raise ValueError('Conforming partitioning requires preserve_surface=True')
    config=_configuration(model,surface_method,slab_depth,axis,options,maximum_facets)
    # JSON normalization turns option dict keys/tuples into the same on-disk form.
    config=json.loads(json.dumps(config));signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    root=external_path(directory)
    if content_addressed:root=root/signature
    root.mkdir(parents=True,exist_ok=True);manifest=root/'configuration.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=config:raise ValueError('Partition checkpoint belongs to a different source/configuration; use a new directory')
    else:
        if any(root.iterdir()):raise ValueError('New partition directory must be empty')
        manifest.write_text(json.dumps(config,indent=2));model.save(root/'geometry.h5')
    if not (root/'geometry.h5').exists():model.save(root/'geometry.h5')
    env=os.environ.copy();env['MORQEX_TETGEN_LIBRARY']=str(library_path(workspace))
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
    if workspace is not None:env['MORQEX_WORKSPACE']=str(Path(workspace).resolve())
    def run(job,index=-1):
        log=root/'logs';log.mkdir(exist_ok=True);tag=f'{job}_{index}_{time.time_ns()}'
        with (log/(tag+'.stdout.txt')).open('w') as out,(log/(tag+'.stderr.txt')).open('w') as err:
            try:result=subprocess.run([sys.executable,'-m','morqex.modeling.native_geometry_worker','partition',str(root),job,str(index)],env=env,stdout=out,stderr=err,timeout=max(60.,options.native_timeout_seconds+120.))
            except subprocess.TimeoutExpired as e:raise RuntimeError(f'{job}/{index} exceeded worker budget; checkpoint retained') from e
        if result.returncode:raise RuntimeError(f'Partition {job}/{index} failed; inspect {log/(tag+".stderr.txt")}; no candidate accepted')
    if (root/'validation.json').exists():
        report=json.loads((root/'validation.json').read_text())
        if report.get('configuration_sha256')!=signature:raise ArithmeticError('Validation fingerprint mismatch')
        return MeshArtifact.load(root/'mesh.h5')
    if not (root/'surface.h5').exists():run('surface')
    n=model.regions.shape[axis];bounds=np.r_[0.,np.arange(slab_depth-.5,n,slab_depth),float(n)]
    if len(bounds)>2 and bounds[-1]-bounds[-2]<slab_depth*.25:bounds=np.delete(bounds,-2)
    # A computational cut may move slightly without moving any physical
    # interface. Avoid planes through original vertices automatically rather
    # than requiring a user-specific geometric repair.
    import h5py
    scale=min(model.spacing);step=model.spacing[axis]/scale
    with h5py.File(root/'surface.h5') as h:
        coordinates=np.sort((h['points'][:,2-axis]-model.origin[axis])/scale)
    for j in range(1,len(bounds)-1):
        preferred=float(bounds[j]);safe=None
        for shift in (0.,.001,-.001,.003,-.003,.011,-.011,.037,-.037):
            candidate=(preferred+shift)*step;position=np.searchsorted(coordinates,candidate)
            distances=np.abs(coordinates[max(0,position-1):position+1]-candidate)
            if distances.min()>1e-8 and bounds[j-1]<(preferred+shift)<bounds[j+1]:safe=preferred+shift;break
        if safe is None:raise ArithmeticError('No nondegenerate cut within the deterministic search budget')
        bounds[j]=safe
    del coordinates
    (root/'bounds.json').write_text(json.dumps(bounds.tolist()))
    def archive_part(i):
        old=root/'rejected'/str(time.time_ns());old.mkdir(parents=True)
        for suffix in ('.h5','.xdmf','.json'):
            p=root/f'part_{i:02}{suffix}'
            if p.exists():p.rename(old/p.name)
    for i in range(len(bounds)-1):
        if not (root/f'part_{i:02}.json').exists():
            if (root/f'part_{i:02}.h5').exists():archive_part(i)
            run('part',i)
    for _ in range(maximum_reconciliations):
        run('reconcile');changed=json.loads((root/'reconciliation.json').read_text())['changed_parts']
        if not changed:break
        for i in changed:archive_part(i);run('part',i)
    else:raise ArithmeticError('No common actual cap triangulation within the reconciliation budget')
    if (root/'mesh.h5').exists():
        rejected=root/'rejected'/str(time.time_ns());rejected.mkdir(parents=True)
        for name in ('mesh.h5','mesh.xdmf'):
            if (root/name).exists():(root/name).rename(rejected/name)
    run('assemble')
    return MeshArtifact.load(root/'mesh.h5')


def _worker(root,job,index):
    import h5py
    from .voxel import VoxelModel
    from .tetgen import SurfacePLC,TetGenOptions,tetgen_mesh,voxel_plc
    from .surface_nets import surface_nets_plc
    from .partitioned import cut_cap,slab_plc,artifact_cap,reconcile_cap,assemble_slabs
    config=json.loads((root/'configuration.json').read_text());axis=2-config['axis_zyx']
    scale=min(config['spacing_zyx_m']);origin=np.array(config['origin_zyx_m'][::-1]);step=config['spacing_zyx_m'][config['axis_zyx']]/scale
    if job=='surface':
        model=VoxelModel.load(root/'geometry.h5')
        if config['surface_method']=='voxel_plc':
            raw=voxel_plc(model,maximum_facets=config['maximum_facets']);q=np.asarray(raw.facets)
            surface=SurfacePLC(raw.points,np.stack((q[:,[0,1,2]],q[:,[0,2,3]]),axis=1).reshape(-1,3),raw.seeds,np.repeat(raw.markers,2),metadata=raw.metadata)
        else:surface=surface_nets_plc(model,maximum_facets=config['maximum_facets'],conserve_volumes=True)
        target=root/'surface.h5';pending=root/'surface.pending.h5'
        if pending.exists():pending.rename(root/('interrupted_surface_'+str(time.time_ns())+'.h5'))
        with h5py.File(pending,'x') as f:
            for k,v in [('points',surface.points),('triangles',surface.facets),('markers',surface.markers)]:f.create_dataset(k,data=v,compression='gzip',compression_opts=1)
            f.attrs.update(metadata=json.dumps(surface.metadata),seeds=json.dumps({k:v.tolist() for k,v in surface.seeds.items()}))
        pending.rename(target);return
    bounds=np.array(json.loads((root/'bounds.json').read_text()))*step
    if job=='part':
        surface=_surface(root/'surface.h5');grid=(surface.points-origin)/scale;caps=[]
        for j in (index,index+1):
            if j in (0,len(bounds)-1):caps.append(None);continue
            cap_path=root/f'cap_{j:02}.npz'
            if cap_path.exists():
                with np.load(cap_path,allow_pickle=False) as f:cap={k:f[k] for k in f.files}
            else:
                cap=cut_cap(grid,surface.facets,surface.markers,surface.metadata['interface_pairs'],bounds[j],axis=axis,minimum_angle=20.)
                np.savez_compressed(cap_path,**cap)
            caps.append(cap)
        part=slab_plc(surface,*bounds[index:index+2],axis=axis,lower_cap=caps[0],upper_cap=caps[1],length_scale=scale,origin=origin)
        del surface,grid,caps
        options=dict(config['tetgen_options']);options['region_maximum_volumes']={int(k):v for k,v in options['region_maximum_volumes'].items()}
        mesh=tetgen_mesh(part,options=TetGenOptions(**options),maximum_facets=config['maximum_facets'])
        mesh.save_hdf5(root/f'part_{index:02}.h5');(root/f'part_{index:02}.json').write_text(json.dumps(mesh.report,indent=2));return
    if job=='reconcile':
        changed=[]
        for j,plane in enumerate(bounds[1:-1],1):
            path=root/f'cap_{j:02}.npz'
            with np.load(path,allow_pickle=False) as f:cap={k:f[k] for k in f.files}
            left=artifact_cap(root/f'part_{j-1:02}.h5',plane,length_scale=scale,axis=axis)
            right=artifact_cap(root/f'part_{j:02}.h5',plane,length_scale=scale,axis=axis)
            common=reconcile_cap(cap,left,right)
            if common is not None:
                path.rename(path.with_name(path.stem+'_previous_'+str(time.time_ns())+'.npz'))
                np.savez_compressed(path,**common);changed.extend((j-1,j))
        (root/'reconciliation.json').write_text(json.dumps({'changed_parts':sorted(set(changed))}));return
    if job!='assemble':raise ValueError('Unknown partition worker job')
    from .adaptive import _region_volumes
    surface=_surface(root/'surface.h5');expected=_region_volumes(surface.points,surface.facets,surface.markers,surface.metadata['interface_pairs'],surface.seeds);del surface
    pending=root/'mesh.pending'
    if pending.exists():pending.rename(root/('rejected_assembly_'+str(time.time_ns())+'.h5'))
    artifact=assemble_slabs([root/f'part_{j:02}.h5' for j in range(len(bounds)-1)],root/'mesh.h5',length_scale=scale)
    with h5py.File(artifact.hdf5,'r+') as f:
        report=json.loads(f.attrs['report']);errors={str(k):abs(report['phase_volumes_m3'][str(k)]/v-1) for k,v in expected.items()}
        if max(errors.values())>1e-7:raise ArithmeticError('Whole assembled source PLC volume mismatch: '+str(errors))
        report.update(source_plc_phase_volume_errors=errors,configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
        f.attrs['report']=json.dumps(report)
    (root/'validation.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':_worker(Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]))
