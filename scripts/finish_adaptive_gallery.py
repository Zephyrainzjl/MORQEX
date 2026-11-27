"""Refresh actual-mesh HTML and independently check whole-box Tet4 cuts.

Run through run_local; this reads external HDF5 meshes, never regenerates or
alters their geometry. Optional --browser checks run separately with --render.
"""
import argparse,json,os,subprocess,re
from pathlib import Path
import numpy as np


def finish(root):
    from morqex.modeling import MeshArtifact,VoxelModel
    from morqex.visualization.adaptive_mesh import AdaptiveMeshView
    from adaptive_geometry_gallery import NAMES,write_index
    checks=[];reports=[]
    for file in sorted(root.glob('*/*/validation.json')):
        report=json.loads(file.read_text());folder=file.parent
        assert report['passed_geometry_export'] and report['shape_zyx']==[256]*3
        artifact=MeshArtifact.load(folder/'mesh.h5');view=AdaptiveMeshView(artifact,title=report['case']+' / '+report['precision']+' / weighted ACVD + TetGen')
        # Payload is read independently from the serialized viewer. Check all
        # materials together: a complete box section must cover its area once.
        match=re.search(r'<script>const D=(.*?),\$=id=>',view.document)
        data=json.loads(match.group(1));model=VoxelModel.load(folder/'geometry.h5')
        length=np.asarray(model.regions.shape)[::-1]*np.asarray(model.spacing)[::-1]
        errors=[]
        for section in data['sections']:
            area=0.
            for polygon in section['polygons']:
                p=np.asarray(polygon);p=p-p[0]
                area+=abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))*.5
            expected=np.prod(length[section['axes']]);error=float(abs(area/expected-1))
            assert error<1e-8,(folder,section['axis'],area,expected)
            errors.append(error)
        view.save_html(folder/'mesh.html')
        checks.append({'case':report['case'],'precision':report['precision'],'passed':True,
            'section_total_area_relative_errors_xyz':errors,'mesh_unchanged':True})
        reports.append(report);print(json.dumps(checks[-1]),flush=True)
    assert set(NAMES)<=set(r['case'] for r in reports),'Ten complete morphology families required'
    write_index(root,'complete',256)
    (root/'validation.json').write_text(json.dumps({'all_geometry_export_checks_passed':True,
        'ten_morphologies_complete':True,'cases':reports,'size':256,
        'scope':'synthetic geometry/meshing, not battery or PDE validation'},indent=2))
    (root/'section_validation.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))


def browser_checks(root,browser):
    from adaptive_geometry_gallery import NAMES
    output=root/'browser_checks';output.mkdir(exist_ok=True);checks=[]
    jobs=[(name,'balanced','surface') for name in NAMES]
    jobs += [(name,'balanced',str(axis)) for name in ('granular','rods') for axis in range(3)]
    jobs += [('granular',mode,'surface') for mode in ('coarse','fine')]
    for name,precision,mode in jobs:
        target=root/name/precision/'mesh.html';key=f'{name}_{precision}_{mode}'
        png=output/(key+'.png')
        result=subprocess.run([browser,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
            '--disable-background-networking','--virtual-time-budget=6000','--window-size=1440,1000',
            '--user-data-dir='+str(Path.cwd()/key),'--screenshot='+str(png),'--dump-dom',
            target.as_uri()+'#view='+mode],capture_output=True,text=True,timeout=45)
        count=re.search(r'data-faces="(\d+)"',result.stdout)
        okay=(result.returncode==0 and 'data-rendered="true"' in result.stdout and
            f'data-view="{mode}"' in result.stdout and 'data-error=' not in result.stdout and
            count is not None and int(count.group(1))>0 and png.exists() and png.stat().st_size>20000)
        if not okay:
            (output/(key+'.failure.html')).write_text(result.stdout)
            (output/(key+'.failure.txt')).write_text(result.stderr)
            raise RuntimeError(f'Browser check failed: {key}')
        check={'case':name,'precision':precision,'view':mode,'passed':True,
            'actual_faces_or_polygons_shown':int(count.group(1)),'screenshot':str(png)}
        checks.append(check);print(json.dumps(check),flush=True)
    (root/'browser_validation.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(os.environ['MORQEX_WORKSPACE'])/'adaptive_gallery256_gemsim')
    p.add_argument('--browser',help='Headless Chromium path; use run_local --render')
    args=p.parse_args()
    browser_checks(args.root,args.browser) if args.browser else finish(args.root)
