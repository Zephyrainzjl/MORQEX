"""Independent numerical/readback or offline Chromium gallery checks."""
import argparse,json,os,re,sys,subprocess
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def readback(folder):
    import h5py,numpy as np
    from morqex.modeling import read_comsol
    from morqex.modeling.comsol import LENGTH_UNITS
    back=read_comsol(folder/'mesh.mphtxt');unit=back['length_unit'];blocks=back['elements'];labels=set()
    with h5py.File(folder/'mesh.h5') as f:
        points=f['mesh/points_m'];cells=f['mesh/tetrahedra'];phase=f['mesh/phase']
        assert len(back['points'])==len(points) and len(blocks['tet']['cells'])==len(cells)
        for start in range(0,len(points),50000):
            np.testing.assert_allclose(back['points'][start:start+50000]*LENGTH_UNITS[unit],points[start:start+50000],rtol=1e-13,atol=1e-20)
        for start in range(0,len(phase),50000):labels.update(map(int,np.unique(phase[start:start+50000])))
        labels=sorted(labels)
        for start in range(0,len(cells),50000):
            np.testing.assert_array_equal(blocks['tet']['cells'][start:start+50000],cells[start:start+50000])
            np.testing.assert_array_equal(blocks['tet']['entities'][start:start+50000],np.searchsorted(labels,phase[start:start+50000])+1)
        assert len(back['selections'])>=len(labels)
        report={'passed':True,'length_unit':unit,'nodes':len(points),'tetrahedra':len(cells),
            'source_regions':labels,'selection_objects':len(back['selections']),'scope':'all coordinates, all Tet4 connectivity, all domains and selections; no COMSOL GUI'}
    (folder/'comsol_readback.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


def browser(root,binary):
    output=root/'browser_checks';output.mkdir(exist_ok=True);checks=[]
    nmc=root/'nmc_tiff/exact_voxel_partitioned/mesh.html'
    targets=[root/'reference_comsol/mesh.html',nmc if nmc.exists() else root/'nmc_tiff/partitioned/mesh.html']
    targets.extend(root/'native'/case/'partitioned/mesh.html' for case in ('granular','rods','platelets','agglomerates','bicontinuous','cell_stack'))
    for target in targets:
        if not target.exists():raise FileNotFoundError(target)
        key='_'.join(target.relative_to(root).parts[:-1])
        for mode in ('surface','0'):
            png=output/(key+'_'+mode+'.png')
            result=subprocess.run([binary,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
                '--disable-background-networking','--virtual-time-budget=10000','--window-size=1440,1000',
                '--user-data-dir='+str(Path.cwd()/(key+'_'+mode)),
                '--screenshot='+str(png),'--dump-dom',target.as_uri()+'#view='+mode],capture_output=True,text=True,timeout=60)
            count=re.search(r'data-faces="(\d+)"',result.stdout)
            if result.returncode or 'data-error=' in result.stdout or 'data-rendered="true"' not in result.stdout or not count or int(count.group(1))<1:
                (output/(key+'_'+mode+'.failed.html')).write_text(result.stdout)
                raise RuntimeError('Offline rendering failed: '+key+'/'+mode)
            if png.stat().st_size<20000:raise RuntimeError('Unexpectedly empty screenshot')
            checks.append({'file':str(target),'view':mode,'actual_faces_or_polygons':int(count.group(1)),
                'screenshot':str(png),'passed':True});print(json.dumps(checks[-1]),flush=True)
    (root/'browser_validation.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(os.environ['MORQEX_WORKSPACE'])/'nmc_complex_gallery')
    p.add_argument('--folder',type=Path);p.add_argument('--browser');args=p.parse_args()
    if args.browser:browser(args.root,args.browser)
    elif args.folder:readback(args.folder)
    else:p.error('Select --browser or --folder')
