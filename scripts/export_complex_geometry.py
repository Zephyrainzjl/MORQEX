"""Bounded serial exchange/readback of complete gallery meshes, not display LOD."""
import argparse,json,os,sys,tarfile,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))


def worker(args):
    import h5py
    from morqex.modeling import MeshArtifact,VoxelModel,export_bundle
    if args.validate_vtu:
        from morqex.modeling.vtu_stream import verify_with_vtk
        folder=Path(args.directory);report=json.loads((folder/'bundle.json').read_text())
        result=verify_with_vtk(folder/'mesh.vtu',args.mesh,length_unit=report['length_unit'])
        (folder/'native_vtk_readback.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    mesh=MeshArtifact.load(args.mesh)
    with h5py.File(mesh.hdf5) as f:report=json.loads(f.attrs['report'])
    names={int(k):v for k,v in report.get('geometry',{}).get('region_names',{}).items()} or None
    selections=None
    if args.geometry:selections=VoxelModel.load(args.geometry).metadata.get('selection_groups')
    elif report.get('source_selections'):
        selections={tag:{'regions':item['entities'],'label':item['label']} for tag,item in report['source_selections'].items() if item['dimension']==3}
        if names is None:names={int(item['entities'][0]):item['label'] for item in report['source_selections'].values() if item['dimension']==3 and len(item['entities'])==1}
    result=export_bundle(mesh,args.directory,region_names=names,selections=selections,length_unit=args.length_unit,formats=args.formats)
    print(json.dumps(result),flush=True)


def controller(args):
    work=Path(args.workspace).resolve();gallery=work/'nmc_complex_gallery'
    sys.path.insert(0,str(ROOT/'scripts'));from source_snapshot import snapshot
    snap=snapshot(ROOT,work);frozen=work/'frozen_geometry'/snap['tree_sha256'][:20]
    if not frozen.exists():
        frozen.mkdir(parents=True)
        with tarfile.open(snap['archive']) as archive:archive.extractall(frozen,filter='data')
    cases={'reference':gallery/'reference_comsol','nmc':gallery/'nmc_tiff/exact_voxel_partitioned'}
    cases.update({name:gallery/'native'/name/'partitioned' for name in ('granular','rods','platelets','agglomerates','bicontinuous','cell_stack')})
    checks=[]
    for case in args.cases or cases:
        folder=cases[case]
        for fmt in args.formats:
            target=folder/'software'/fmt;report=target/'bundle.json'
            if report.exists():
                accepted=json.loads(report.read_text())
                if accepted.get('passed') and accepted.get('length_unit')==args.length_unit:
                    if fmt!='vtu' or (target/'native_vtk_readback.json').exists():
                        checks.append({'case':case,'format':fmt,'directory':str(target),'passed':True});continue
                    # Validate an already accepted file in a separate process
                    # so writer-side arrays do not inflate reader memory.
                    cmd=[sys.executable,str(frozen/'scripts/run_local.py'),'--workspace',str(work),'--reference-memory','--large-mesh-output',
                        '--label',f'full_vtk_reader_{case}','--',str(work/'runtime/bin/python'),str(frozen/'scripts/export_complex_geometry.py'),
                        '--mesh',str(folder/'mesh.h5'),'--directory',str(target),'--validate-vtu']
                    if subprocess.run(cmd,stdout=subprocess.PIPE,text=True).returncode:raise RuntimeError('Native VTK full-file validation rejected')
                    checks.append({'case':case,'format':fmt,'directory':str(target),'passed':True});continue
            if target.exists():target.rename(target.with_name(fmt+'_rejected_'+str(time.time_ns())))
            cmd=[sys.executable,str(frozen/'scripts/run_local.py'),'--workspace',str(work),'--reference-memory','--large-mesh-output',
                '--label',f'full_exchange_{case}_{fmt}','--',str(work/'runtime/bin/python'),str(frozen/'scripts/export_complex_geometry.py'),
                '--mesh',str(folder/'mesh.h5'),'--directory',str(target),'--formats',fmt,'--length-unit',args.length_unit]
            geometry=folder.parent/'geometry.h5'
            if geometry.exists():cmd+=['--geometry',str(geometry)]
            result=subprocess.run(cmd,stdout=subprocess.PIPE,text=True)
            (gallery/'exchange_controller_latest.log').write_text(result.stdout)
            if result.returncode:raise RuntimeError(f'Full {case}/{fmt} export/readback rejected; original mesh retained')
            if fmt=='vtu':
                validation=cmd+['--validate-vtu']
                validation[validation.index('--label')+1]=f'full_vtk_reader_{case}'
                if subprocess.run(validation,stdout=subprocess.PIPE,text=True).returncode:raise RuntimeError('Native VTK full-file validation rejected')
            checks.append({'case':case,'format':fmt,'directory':str(target),'passed':True})
            (gallery/'exchange_validation.json').write_text(json.dumps({'passed':False,'complete':False,'source_snapshot':snap,'checks':checks},indent=2))
    (gallery/'exchange_validation.json').write_text(json.dumps({'passed':True,'complete':True,'requested_cases':list(args.cases or cases),'requested_formats':args.formats,'source_snapshot':snap,'checks':checks},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',default=str(ROOT.parent/'morph_workspace/microelectrode_gallery'))
    p.add_argument('--mesh');p.add_argument('--directory');p.add_argument('--geometry');p.add_argument('--gallery',action='store_true')
    p.add_argument('--validate-vtu',action='store_true')
    p.add_argument('--cases',nargs='+');p.add_argument('--formats',nargs='+',default=['vtu','gmsh','medit','abaqus','comsol']);p.add_argument('--length-unit',default='um')
    a=p.parse_args()
    if a.gallery:controller(a)
    elif a.mesh and a.directory:worker(a)
    else:p.error('Choose --gallery, or --mesh and --directory')
