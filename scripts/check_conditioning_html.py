"""Offline browser checks for same-size smoothing and full FEM galleries."""
import argparse,json,os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'));parser.add_argument('--browser',type=Path,required=True);parser.add_argument('--study',default='rods');args=parser.parse_args()
    from playwright.sync_api import sync_playwright
    from morqex.modeling.voxel import external_path
    work=external_path(args.workspace);output=work/'implicit_geometry/browser_checks';output.mkdir(parents=True,exist_ok=True)
    errors=[];checks=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=str(args.browser),headless=True,args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--renderer-process-limit=2','--js-flags=--max-old-space-size=768'])
        page=browser.new_page(viewport={'width':1650,'height':1250},device_scale_factor=1);page.on('pageerror',lambda error:errors.append(str(error)))
        def frames():
            views=[]
            for element in page.locator('#views iframe').element_handles():
                frame=element.content_frame();frame.wait_for_function("document.body&&(document.body.dataset.rendered==='true'||document.body.dataset.error)",timeout=90000)
                state=frame.evaluate('({...document.body.dataset})')
                assert not state.get('error') and frame.locator('canvas').count()>0
                if 'faces' in state:assert int(state['faces'])>0
                views.append(frame)
            assert len(views)==2
            return views
        def ready_font():
            page.evaluate('async()=>{await document.fonts.ready;return true}')
            if (work/'data/report_font/NotoSansCJKsc-Regular.otf').exists():
                assert 'MORQEXReportCJK' in page.evaluate('getComputedStyle(document.body).fontFamily')
                assert page.evaluate("document.fonts.check('16px MORQEXReportCJK','平滑尺寸结构')")
        study=work/'conditioning_study'/args.study/'index.html';page.goto(study.as_uri(),wait_until='domcontentloaded');ready_font()
        for case in ('very_fine','fine','balanced','coarse','reduced'):
            if page.locator('#case').input_value()!=case:page.locator('#case').select_option(case)
            left,right=frames();before=left.evaluate('D.shape');after=right.evaluate('D.shape');h0=left.evaluate('D.spacing');h1=right.evaluate('D.spacing')
            if case=='reduced':
                assert all(b<a for a,b in zip(before,after))
                assert all(abs(a*ha-b*hb)<1e-12 for a,ha,b,hb in zip(before,h0,after,h1))
            else:assert before==after and h0==h1
            page.screenshot(path=str(output/('study_'+case+'.png')),full_page=True);checks.append({'study':case,'shape_before':before,'shape_after':after})
        page.goto((work/'implicit_geometry/index.html').as_uri(),wait_until='domcontentloaded');ready_font()
        options=page.locator('#case option').evaluate_all('(xs)=>xs.map(x=>x.value)')
        for case in options:
            if page.locator('#case').input_value()!=case:page.locator('#case').select_option(case)
            if not page.locator('#precision option').count():continue
            page.locator('#compare').select_option('mesh');voxel,mesh=frames()
            data=mesh.evaluate('D.report');assert data['quality_limits_satisfied'] and data['geometry_limits_satisfied']
            assert voxel.evaluate('D.shape')==data['geometry']['source_shape_zyx']
            assert mesh.evaluate('D.report.nodes')>0
            assert mesh.evaluate('document.body.dataset.view')=='2'
            page.screenshot(path=str(output/('mesh_'+case+'.png')),full_page=True);checks.append({'mesh':case,'nodes':data['nodes'],'cells':data['tetrahedra'],'quality':data['quality']})
        version=browser.version;browser.close()
    if errors:raise ArithmeticError('Browser errors: '+str(errors))
    report={'passed':True,'checks':checks,'browser':version,'javascript_errors':errors,'offline':True,'scope':'actual local voxel/FEM interactive rendering, not numerical PDE certification'}
    (output/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({'passed':True,'checks':len(checks),'report':str(output/'validation.json')}))

if __name__=='__main__':main()
