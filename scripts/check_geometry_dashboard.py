"""Offline, interactive gallery verification in one bounded browser session."""
import argparse,importlib.metadata,json,os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'))
    parser.add_argument('--browser',type=Path,required=True)
    args=parser.parse_args()
    if args.workspace is None:parser.error('Specify an external workspace')
    from morqex.modeling.voxel import external_path
    from playwright.sync_api import sync_playwright
    root=external_path(args.workspace)/'nmc_complex_gallery';output=root/'dashboard_checks';output.mkdir(exist_ok=True)
    checks=[];errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=str(args.browser),headless=True,args=[
            '--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking',
            '--renderer-process-limit=2','--js-flags=--max-old-space-size=768'])
        version=browser.version
        page=browser.new_page(viewport={'width':1600,'height':1250},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto((root/'index.html').as_uri()+'#case=reference',wait_until='domcontentloaded',timeout=90000)
        for name in ('reference','nmc','granular','rods','platelets','agglomerates','bicontinuous','cell_stack'):
            page.locator('#case').select_option(name)
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
            image=output/(name+'.png');page.screenshot(path=str(image),full_page=True)
            if image.stat().st_size<20000:raise RuntimeError('Unexpectedly empty dashboard screenshot')
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
        page.goto((root/'native/rods/current_api/mesh.html').as_uri(),wait_until='domcontentloaded',timeout=90000)
        page.wait_for_function("document.body.dataset.rendered==='true'||document.body.dataset.error",timeout=90000)
        assert not page.evaluate('document.body.dataset.error') and int(page.evaluate('document.body.dataset.faces'))>0
        from morqex.modeling import MeshArtifact
        expected=MeshArtifact.load(root/'native/rods/current_api/custom_name.h5').tetrahedra
        assert page.evaluate('D.report.tetrahedra')==expected
        page.locator('#mode').select_option('0')
        page.wait_for_function("document.body.dataset.view==='0'&&+document.body.dataset.faces>0")
        page.screenshot(path=str(output/'current_public_api.png'))
        if errors:raise RuntimeError('Browser JavaScript errors: '+str(errors))
        browser.close()
    result={'passed':True,'checks':checks,'current_public_api_preview_passed':True,
        'phase_zero_and_slice_controls_passed':True,'page_errors':errors,'browser_version':version,
        'playwright_version':importlib.metadata.version('playwright'),
        'scope':'one browser, eight sequential structure switches, at most two attached iframes; actual stored data, offline CPU rendering'}
    (root/'dashboard_browser_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'dashboard_cases':len(checks),'current_public_api':True}),flush=True)


if __name__=='__main__':main()
