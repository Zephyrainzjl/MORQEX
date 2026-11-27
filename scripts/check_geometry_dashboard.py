"""Offline, interactive gallery verification in one bounded browser session."""
import argparse,importlib.metadata,json,os,math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'))
    parser.add_argument('--browser',type=Path,required=True)
    parser.add_argument('--aligned',action='store_true',help='Check the exact voxel-aligned gallery, including actual 3D mesh faces')
    args=parser.parse_args()
    if args.workspace is None:parser.error('Specify an external workspace')
    from morqex.modeling.voxel import external_path
    from playwright.sync_api import sync_playwright
    root=external_path(args.workspace)/'nmc_complex_gallery'
    output=root/('aligned_dashboard_checks' if args.aligned else 'dashboard_checks');output.mkdir(exist_ok=True)
    checks=[];errors=[];roi_checks=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=str(args.browser),headless=True,args=[
            '--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking',
            '--renderer-process-limit=2','--js-flags=--max-old-space-size=768'])
        version=browser.version
        page=browser.new_page(viewport={'width':1600,'height':1250},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto((root/('voxel_aligned.html' if args.aligned else 'index.html')).as_uri()+'#case='+('nmc' if args.aligned else 'reference'),wait_until='domcontentloaded',timeout=90000)
        names=('nmc','granular','rods','platelets','agglomerates','bicontinuous','cell_stack') if args.aligned else ('reference','nmc','granular','rods','platelets','agglomerates','bicontinuous','cell_stack')
        for name in names:
            # select_option dispatches change even for the current value;
            # avoid starting a redundant second full CT load on initial entry.
            if page.evaluate('document.body.dataset.case')!=name:page.locator('#case').select_option(name)
            page.wait_for_function('(name)=>document.body.dataset.case===name',arg=name)
            page.locator('#views iframe').nth(1).wait_for()
            frames=[]
            for element in page.locator('#views iframe').element_handles():
                frame=element.content_frame()
                frame.wait_for_function("document.body&&(document.body.dataset.rendered==='true'||document.body.dataset.error)",timeout=90000)
                data=frame.evaluate('({...document.body.dataset})')
                if data.get('error'):raise RuntimeError(name+': '+data['error'])
                assert frame.locator('canvas').count()>0
                if 'faces' in data:assert int(data['faces'])>0
                frames.append({'url':frame.url,'data':data})
            assert page.locator('iframe').count()==2 and len(page.frames)==3
            if name!='reference':
                voxel=page.locator('#views iframe').nth(0).element_handle().content_frame()
                mesh=page.locator('#views iframe').nth(1).element_handle().content_frame()
                source_plane=float(voxel.evaluate('document.body.dataset.planeM'))
                mesh_plane=mesh.evaluate('D.sections.find(s=>s.axis===2).plane_m')
                assert math.isclose(source_plane,mesh_plane,rel_tol=1e-10,abs_tol=1e-16)
                assert mesh.evaluate('document.body.dataset.view')=='2'
                assert mesh.locator('#phase').input_value()=='-1'
                assert int(mesh.evaluate('document.body.dataset.faces'))==mesh.evaluate('D.sections.find(s=>s.axis===2).polygons.length')
                labels=voxel.evaluate('Object.keys(D.names).map(Number)')
                for label in labels:
                    assert voxel.evaluate('(id)=>rgb(id)',label)==mesh.evaluate('(id)=>color(id)',label)
            image=output/(name+'.png');page.screenshot(path=str(image),full_page=True)
            if image.stat().st_size<20000:raise RuntimeError('Unexpectedly empty dashboard screenshot')
            if args.aligned:
                mesh=page.locator('#views iframe').nth(1).element_handle().content_frame()
                assert mesh.evaluate('D.report.source_voxel_cell_centroids_checked===D.report.tetrahedra')
                mesh.locator('#mode').select_option('surface')
                mesh.wait_for_function("document.body.dataset.view==='surface'&&+document.body.dataset.faces>0",timeout=90000)
                page.screenshot(path=str(output/(name+'_surface.png')),full_page=True)
                mesh.locator('#mode').select_option('2')
                mesh.wait_for_function("document.body.dataset.view==='2'&&+document.body.dataset.faces>0")
            if name=='granular':
                voxel=page.locator('#views iframe').nth(0).element_handle().content_frame()
                voxel.locator('#show').select_option('0')
                voxel.wait_for_function("document.querySelector('#hide').value==='none'&&faces.length>0")
                assert voxel.evaluate('faces.every(f=>f.v===0)')
                voxel.locator('#axis').select_option('2')
                voxel.locator('#slice').evaluate("el=>{el.value='13';el.dispatchEvent(new Event('input',{bubbles:true}));}")
                assert voxel.evaluate("document.body.dataset.axis==='2'&&document.body.dataset.index==='13'")
                mesh=page.locator('#views iframe').nth(1).element_handle().content_frame()
                mesh.locator('#mode').select_option('2')
                mesh.wait_for_function("document.body.dataset.view==='2'&&+document.body.dataset.faces>0")
                mesh.locator('#phase').select_option('0')
                mesh.wait_for_function("+document.body.dataset.faces>0")
            checks.append({'case':name,'passed':True,'attached_iframes':2,'frames':frames,'screenshot':str(image)})
            print(json.dumps(checks[-1]),flush=True)
        if args.aligned:
            page.close();page=browser.new_page(viewport={'width':1450,'height':1050},device_scale_factor=1)
            page.on('pageerror',lambda error:errors.append(str(error)))
            for name in names:
                base=root/('nmc_tiff' if name=='nmc' else 'native/'+name)
                target=base/'voxel_aligned/roi_mesh.html'
                page.goto(target.as_uri(),wait_until='domcontentloaded',timeout=90000)
                page.wait_for_function("document.body.dataset.rendered==='true'||document.body.dataset.error",timeout=90000)
                assert not page.evaluate('document.body.dataset.error')
                assert page.evaluate('D.surface_bounds_m!==null&&D.face_count===D.complete_surface_face_count')
                assert page.evaluate('D.report.source_voxel_cell_centroids_checked===D.report.tetrahedra')
                page.locator('#phase').select_option('-1')
                page.wait_for_function("+document.body.dataset.faces===D.face_count",timeout=90000)
                assert page.locator('#budget').input_value()=='10000000'
                page.screenshot(path=str(output/(name+'_roi.png')),full_page=True)
                roi_checks.append({'case':name,'passed':True,'complete_actual_roi_faces':page.evaluate('D.face_count'),
                    'source_cells':page.evaluate('D.report.tetrahedra'),'display_roi_only':True})
        api=root/'nmc_tiff/current_api_aligned_optimized' if args.aligned else root/'native/rods/current_api_optimized'
        if args.aligned and not (api/'public_api_validation.json').exists():api=root/'nmc_tiff/current_api_aligned'
        if not args.aligned and not api.exists():api=root/'native/rods/current_api'
        page.goto((api/'mesh.html').as_uri(),wait_until='domcontentloaded',timeout=90000)
        page.wait_for_function("document.body.dataset.rendered==='true'||document.body.dataset.error",timeout=90000)
        assert not page.evaluate('document.body.dataset.error') and int(page.evaluate('document.body.dataset.faces'))>0
        from morqex.modeling import MeshArtifact
        expected=MeshArtifact.load(api/'custom_name.h5').tetrahedra
        assert page.evaluate('D.report.tetrahedra')==expected
        page.locator('#mode').select_option('0')
        page.wait_for_function("document.body.dataset.view==='0'&&+document.body.dataset.faces>0")
        page.screenshot(path=str(output/'current_public_api.png'))
        page.close()
        import numpy as np
        from morqex.modeling import from_regions
        model=from_regions(np.ones((4,6,8),np.uint8),spacing=(1e-6,2e-6,4e-6),origin=(10e-6,20e-6,30e-6))
        example=model.viewer(maximum_3d_size=8,slice_origin='lower').save_html(output/'anisotropic_slice.html')
        page=browser.new_page(viewport={'width':1100,'height':900},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(example.as_uri(),wait_until='domcontentloaded')
        page.wait_for_function("document.body.dataset.rendered==='true'",timeout=30000)
        data=page.evaluate('({...window.sliceMap})')
        assert math.isclose(data['stepX']/data['stepY'],2.,rel_tol=1e-12)
        assert math.isclose(float(page.evaluate('document.body.dataset.planeM')),12.5e-6,abs_tol=1e-16)
        page.locator('#two').click(position={'x':data['ox']+4.5*data['stepX'],
            'y':data['oy']+2.5*data['stepY']})
        assert 'ZYX (2,3,4)' in page.locator('#read').inner_text()
        page.screenshot(path=str(output/'anisotropic_slice.png'))
        if errors:raise RuntimeError('Browser JavaScript errors: '+str(errors))
        browser.close()
    result={'passed':True,'checks':checks,'current_public_api_preview_passed':True,
        'phase_zero_and_slice_controls_passed':True,'page_errors':errors,'browser_version':version,
        'default_physical_planes_and_phase_colours_match':True,'anisotropic_aspect_and_cartesian_click_passed':True,
        'playwright_version':importlib.metadata.version('playwright'),
        'actual_surface_checks':args.aligned,
        'roi_actual_face_checks':roi_checks,'all_default_section_phases_visible':True,
        'scope':f'one browser, {len(checks)} sequential structure switches, at most two attached iframes; actual stored data, offline CPU rendering'}
    (root/('aligned_dashboard_browser_validation.json' if args.aligned else 'dashboard_browser_validation.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'dashboard_cases':len(checks),'current_public_api':True}),flush=True)


if __name__=='__main__':main()
