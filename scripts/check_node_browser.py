"""Offline controls and real-voxel rendering qualification, no browser GPU."""
import os,json,subprocess
from pathlib import Path
from html.parser import HTMLParser

CHECK='''<script>setTimeout(()=>{let r={errors:window.VIEW_ERRORS||[]};try{r.ready=!!window.MORQEX_VIEW;r.nodes=document.getElementById('positiveNodes').children.length;r.negative_nodes=document.getElementById('negativeNodes').children.length;r.voxels=window.MORQEX_VIEW.voxel_count;r.frames=window.MORQEX_VIEW.frames;
document.getElementById('positiveNodes').children[5].click();r.node6=window.MORQEX_VIEW.node===5;
document.getElementById('time').value=r.frames-1;document.getElementById('time').dispatchEvent(new Event('input'));r.time=window.MORQEX_VIEW.index===r.frames-1;
document.getElementById('field').value='vm';document.getElementById('field').dispatchEvent(new Event('input'));r.stress=Number.isFinite(window.MORQEX_VIEW.max)&&window.MORQEX_VIEW.max>0;
let before=document.getElementById('volume').toDataURL();document.getElementById('phase').value='cbd';document.getElementById('phase').dispatchEvent(new Event('input'));r.cbd=before!==document.getElementById('volume').toDataURL();
document.getElementById('cut').value=24;document.getElementById('cut').dispatchEvent(new Event('input'));r.cut=document.getElementById('cutText').textContent==='24/32';r.passed=r.ready&&r.nodes===6&&r.negative_nodes===0&&r.voxels===32768&&r.frames>=3&&r.node6&&r.time&&r.stress&&r.cbd&&r.cut&&!r.errors.length;
}catch(e){r.error=e.stack;r.passed=false}let p=document.createElement('pre');p.id='check';p.textContent=JSON.stringify(r);document.body.append(p)},2000);</script>'''

class Read(HTMLParser):
    def __init__(self):super().__init__();self.inside=False;self.value=''
    def handle_starttag(self,t,a):
        if t=='pre' and dict(a).get('id')=='check':self.inside=True
    def handle_endtag(self,t):
        if t=='pre':self.inside=False
    def handle_data(self,d):
        if self.inside:self.value+=d

if __name__=='__main__':
    ws=Path(os.environ['MORQEX_WORKSPACE']);source=(ws/'reports/MORQEX_sphere32_Li_short.html').read_text()
    source=source.replace('<script id="payload"','<script>window.VIEW_ERRORS=[];window.addEventListener("error",e=>window.VIEW_ERRORS.push(e.message));</script><script id="payload"',1)
    page=Path.cwd()/'check.html';page.write_text(source+CHECK)
    candidates=sorted((Path.home()/'.cache/ms-playwright').glob('chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell'))
    if not candidates:raise FileNotFoundError('Local headless Chromium required')
    screenshot=ws/'reports/MORQEX_sphere32_Li_short.png'
    command=[str(candidates[-1]),'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking',f'--user-data-dir={Path.cwd()/"profile"}',
        '--window-size=1440,1600','--virtual-time-budget=4500',f'--screenshot={screenshot}','--dump-dom',page.as_uri()]
    r=subprocess.run(command,capture_output=True,text=True,timeout=55)
    parser=Read();parser.feed(r.stdout)
    if not parser.value:raise RuntimeError(r.stderr[-3000:])
    v=json.loads(parser.value);(ws/'reports/sphere32_node_browser_check.json').write_text(json.dumps(v,indent=2));print(json.dumps(v));assert v['passed']
