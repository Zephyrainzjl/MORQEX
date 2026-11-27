"""Resume bounded conforming full-volume TetGen jobs; all output is external.

This is a serial controller. Each part gets its own resource-guarded process.
Never run concurrently with another run_local job in the same workspace.
"""
import argparse,json,os,sys,subprocess,time,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))


def partition_bounds(maximum,width=32.):
    import numpy as np
    bounds=np.r_[0.,np.arange(width-.5,maximum,width),float(maximum)]
    if len(bounds)>2 and bounds[-1]-bounds[-2]<width*.25:bounds=np.delete(bounds,-2)
    return bounds


def load_surface(path):
    import h5py
    from morqex.modeling import SurfacePLC
    with h5py.File(path) as f:
        return SurfacePLC(f['points'][:],f['triangles'][:],{int(k):v for k,v in json.loads(f.attrs['seeds']).items()},
            f['markers'][:],metadata=json.loads(f.attrs['metadata']))


def worker(args):
    import h5py,numpy as np
    from morqex.modeling import VoxelModel,surface_nets_plc,TetGenOptions,tetgen_mesh,export_comsol
    from morqex.modeling.partitioned import cut_cap,slab_plc,assemble_slabs,artifact_cap,reconcile_cap
    from morqex.visualization.adaptive_mesh import AdaptiveMeshView
    base=Path(args.output);folder=base/'nmc_tiff' if args.case in ('nmc_tiff','nmc_exact') else base/'native'/args.case
    out=folder/('exact_voxel_partitioned' if args.case=='nmc_exact' else 'partitioned');out.mkdir(exist_ok=True);h=.5e-6
    surface_path=folder/'projected_surface_nets.h5' if args.case=='nmc_tiff' else out/'source_surface.h5'
    if args.job=='surface':
        model=VoxelModel.load(folder/'geometry.h5')
        if args.case=='nmc_exact':
            from morqex.modeling import voxel_plc,SurfacePLC
            raw=voxel_plc(model,maximum_facets=8000000);q=np.asarray(raw.facets)
            f=np.stack((q[:,[0,1,2]],q[:,[0,2,3]]),axis=1).reshape(-1,3)
            surface=SurfacePLC(raw.points,f,raw.seeds,np.repeat(raw.markers,2),metadata=raw.metadata)
        else:surface=surface_nets_plc(model,maximum_facets=16000000,conserve_volumes=True)
        with h5py.File(surface_path,'x') as f:
            for name,data in [('points',surface.points),('triangles',surface.facets),('markers',surface.markers)]:f.create_dataset(name,data=data,compression='gzip',compression_opts=1)
            f.attrs['metadata']=json.dumps(surface.metadata);f.attrs['seeds']=json.dumps({k:v.tolist() for k,v in surface.seeds.items()})
        return
    if args.job=='reconcile':
        model=VoxelModel.load(folder/'geometry.h5');bounds=partition_bounds(model.regions.shape[0]);changed=[]
        for i,plane in enumerate(bounds[1:-1],1):
            name=out/('cap_z'+format(float(plane),'.12g')+'.npz')
            baseline=name if name.exists() else out/f'cap_{i:02}.npz'
            with np.load(baseline,allow_pickle=False) as f:cap={k:f[k] for k in f.files}
            left=artifact_cap(out/f'part_{i-1:02}.h5',plane,length_scale=h)
            right=artifact_cap(out/f'part_{i:02}.h5',plane,length_scale=h)
            common=reconcile_cap(cap,left,right)
            if common is not None:
                if name.exists():name.rename(name.with_name(name.stem+'_previous_'+str(time.time_ns())+'.npz'))
                np.savez_compressed(name,**common);changed.extend((i-1,i))
        (out/'reconciliation.json').write_text(json.dumps({'changed_parts':sorted(set(changed)),'all_caps_conforming':not changed},indent=2));return
    if args.job in ('part','stitch'):
        surface=load_surface(surface_path);points=surface.points/h
        maximum=float(points[:,2].max());bounds=partition_bounds(maximum)
        i=args.part;low,high=bounds[i:i+2];caps=[]
        refinement=out/'refinements'/f'part_{i:02}_{args.subdivisions}'
        if args.job=='stitch':
            del surface,points
            paths=[refinement/f'part_{k:02}.h5' for k in range(args.subdivisions)]
            artifact=assemble_slabs(paths,out/f'part_{i:02}.h5',length_scale=h)
            with h5py.File(artifact.hdf5,'r+') as f:
                report=json.loads(f.attrs['report']);report['geometry'].update(partition_axis_xyz=2,
                    slab_bounds_grid_units=[float(low),float(high)],length_scale_m=h,origin_xyz_m=[0.,0.,0.])
                report['adaptive_partition_subdivisions']=args.subdivisions;f.attrs['report']=json.dumps(report)
            (out/f'part_{i:02}.json').write_text(json.dumps(report,indent=2));return
        destination=out/f'part_{i:02}'
        if args.subdivisions>1:
            inner=np.linspace(low,high,args.subdivisions+1);low,high=inner[args.subpart:args.subpart+2]
            refinement.mkdir(parents=True,exist_ok=True);destination=refinement/f'part_{args.subpart:02}'
        for j,plane in ((i,low),(i+1,high)):
            if plane in (0.,maximum):caps.append(None);continue
            cap_path=out/('cap_z'+format(float(plane),'.12g')+'.npz')
            if cap_path.exists():
                with np.load(cap_path,allow_pickle=False) as data:cap={k:data[k] for k in data.files}
                cap['axis']=int(cap['axis']);cap['plane']=float(cap['plane']);cap['other']=list(map(int,cap['other']))
            else:
                cap=cut_cap(points,surface.facets,surface.markers,surface.metadata['interface_pairs'],float(plane),minimum_angle=20.)
                np.savez_compressed(cap_path,**cap)
            caps.append(cap)
        part=slab_plc(surface,float(low),float(high),lower_cap=caps[0],upper_cap=caps[1],length_scale=h)
        del surface,points,caps
        print(json.dumps({'part':i,'points':len(part.points),'facets':len(part.facets)}),flush=True)
        mesh=tetgen_mesh(part,options=TetGenOptions.for_adaptive(preserve_surface=True,merge_coplanar=False,
            region_labelling='seeds' if args.case=='nmc_exact' else 'interfaces',coplanarity_tolerance=1e-15,
            maximum_cells=8000000,maximum_steiner=800000,native_timeout_seconds=240),maximum_facets=5000000)
        mesh.save_hdf5(destination.with_suffix('.h5'))
        destination.with_suffix('.json').write_text(json.dumps(mesh.report,indent=2));return
    if args.job=='assemble':
        source=load_surface(surface_path)
        count=len(partition_bounds(float(source.points[:,2].max()/h)))-1
        paths=[out/f'part_{i:02}.h5' for i in range(count)]
        from morqex.modeling.adaptive import _region_volumes
        expected=_region_volumes(source.points,source.facets,source.markers,source.metadata['interface_pairs'],source.seeds)
        del source
        artifact=assemble_slabs(paths,out/'mesh.h5',length_scale=h)
        with h5py.File(artifact.hdf5) as f:report=json.loads(f.attrs['report'])
        error={str(k):abs(report['phase_volumes_m3'][str(k)]/v-1) for k,v in expected.items()}
        if max(error.values())>1e-7:raise ArithmeticError(f'Whole source PLC/volume mismatch {error}')
        report['source_plc_phase_volume_errors']=error
        (out/'validation.json').write_text(json.dumps(report,indent=2))
        return
    from morqex.modeling import MeshArtifact
    artifact=MeshArtifact.load(out/'mesh.h5')
    if args.job=='view':
        AdaptiveMeshView(artifact,maximum_embedded_faces=450000,node_cache_bytes=256*2**20,
            title=args.case+' / complete conforming TetGen volume').save_html(out/'mesh.html')
    if args.job=='export':
        with h5py.File(artifact.hdf5) as f:geo=json.loads(f.attrs['report'])['geometry']
        model=VoxelModel.load(folder/'geometry.h5')
        export_comsol(artifact,out/'mesh.mphtxt',length_unit=args.length_unit,
            region_names={int(k):v for k,v in geo['region_names'].items()},
            selections=model.metadata.get('selection_groups'),include_boundaries=False)


def controller(args):
    work=Path(args.workspace).resolve();base=Path(args.output or work/'nmc_complex_gallery').resolve()
    sys.path.insert(0,str(ROOT/'scripts'));from source_snapshot import snapshot
    snap=snapshot(ROOT,work);frozen=work/'frozen_geometry'/snap['tree_sha256'][:20]
    if not frozen.exists():
        frozen.mkdir(parents=True)
        with tarfile.open(snap['archive']) as archive:archive.extractall(frozen,filter='data')
    for case in args.cases:
        folder=base/'nmc_tiff' if case in ('nmc_tiff','nmc_exact') else base/'native'/case
        out=folder/('exact_voxel_partitioned' if case=='nmc_exact' else 'partitioned');out.mkdir(exist_ok=True)
        import h5py
        with h5py.File(folder/'geometry.h5') as f:
            # VoxelModel uses phase_labels, canonical axes ZYX.
            key='phase_labels' if 'phase_labels' in f else 'regions'
            maximum=f[key].shape[0]
        bounds=partition_bounds(maximum);count=len(bounds)-1
        # A previous partition plan may have left a thin trailing slab. Keep
        # every unchanged accepted part; archive only changed bounds.
        for i in range(count):
            report_path=out/f'part_{i:02}.json'
            artifact_path=out/f'part_{i:02}.h5'
            if artifact_path.exists() and not report_path.exists():
                archive=out/'superseded_partitions'/str(time.time_ns());archive.mkdir(parents=True)
                for suffix in ('.h5','.xdmf'):
                    item=(out/f'part_{i:02}').with_suffix(suffix)
                    if item.exists():item.rename(archive/item.name)
            if report_path.exists():
                old=json.loads(report_path.read_text())['geometry']['slab_bounds_grid_units']
                if old!=list(map(float,bounds[i:i+2])):
                    archive=out/'superseded_partitions'/str(time.time_ns());archive.mkdir(parents=True)
                    for suffix in ('.h5','.xdmf','.json'):
                        item=(out/f'part_{i:02}').with_suffix(suffix)
                        if item.exists():item.rename(archive/item.name)
        tasks=[]
        if case!='nmc_tiff' and not (out/'source_surface.h5').exists():tasks.append(('surface',-1))
        tasks.extend(('part',i) for i in range(count) if not (out/f'part_{i:02}.json').exists())
        if not (out/'validation.json').exists():tasks.append(('assemble',-1))
        if not (out/'mesh.html').exists():tasks.append(('view',-1))
        if not (out/'mesh.mphtxt').exists():tasks.append(('export',-1))
        for job,part in tasks:
            cmd=[sys.executable,str(frozen/'scripts/run_local.py'),'--workspace',str(work),'--reference-memory','--large-mesh-output',
                '--label',f'full_{case}_{job}_{part}'.replace('-1','all'),'--',str(work/'runtime/bin/python'),
                str(frozen/'scripts/partitioned_geometry_gallery.py'),'--case',case,'--job',job,'--part',str(part),
                '--output',str(base),'--length-unit',args.length_unit]
            if job=='assemble':
                for reconciliation in range(4):
                    inspect=cmd.copy();inspect[inspect.index('--job')+1]='reconcile'
                    result=subprocess.run(inspect)
                    if result.returncode:raise RuntimeError('Cap reconciliation rejected; inspect material area checks')
                    changed=json.loads((out/'reconciliation.json').read_text())['changed_parts']
                    if not changed:break
                    for i in changed:
                        archive=out/'superseded_partitions'/str(time.time_ns());archive.mkdir(parents=True)
                        for suffix in ('.h5','.xdmf','.json'):
                            path=(out/f'part_{i:02}').with_suffix(suffix)
                            if path.exists():path.rename(archive/path.name)
                        retry=cmd.copy();retry[retry.index('--job')+1]='part';retry[retry.index('--part')+1]=str(i)
                        if subprocess.run(retry).returncode:raise RuntimeError('Reconciled native slab requires correction')
                else:raise RuntimeError('Cap recovery did not reach a common triangulation within four rounds')
                pending=out/'mesh.pending'
                if pending.exists():pending.rename(out/('rejected_assembly_'+str(time.time_ns())+'.h5'))
            result=subprocess.run(cmd)
            if result.returncode and job=='part':
                repaired=False
                for subdivision in (2,4):
                    valid=True
                    for k in range(subdivision):
                        result=subprocess.run(cmd+['--subdivisions',str(subdivision),'--subpart',str(k)])
                        if result.returncode:valid=False;break
                    if valid:
                        stitch=cmd.copy();stitch[stitch.index('--job')+1]='stitch'
                        result=subprocess.run(stitch+['--subdivisions',str(subdivision)])
                        if not result.returncode:repaired=True;break
                if not repaired:raise RuntimeError(f'{case}/{part}: partition refinement exhausted; no failed geometry accepted')
            elif result.returncode:raise RuntimeError(f'{case}/{job}/{part} rejected; inspect resource and geometry logs')
        from complex_geometry_gallery import write_index
        write_index(base)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',default=str(ROOT.parent/'morph_workspace/microelectrode_gallery'));p.add_argument('--output')
    p.add_argument('--cases',nargs='+',default=['nmc_tiff']);p.add_argument('--case');p.add_argument('--job')
    p.add_argument('--part',type=int,default=-1);p.add_argument('--length-unit',default='um')
    p.add_argument('--subdivisions',type=int,default=1);p.add_argument('--subpart',type=int,default=0)
    args=p.parse_args();worker(args) if args.case else controller(args)
