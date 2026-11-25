"""Headless offline UI qualification with software WebGL and an isolated profile."""
import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess

CHECK=r"""
<script>
window.addEventListener('load',()=>setTimeout(()=>{
const result={};
try{
result.cases=document.querySelectorAll('#case option').length;
result.webgl=!!gl;result.instances=count;document.getElementById('mapping').value='representative';document.getElementById('mapping').dispatchEvent(new Event('change'));result.partition_unique=count===result.instances*5;document.getElementById('mapping').value='partition';document.getElementById('mapping').dispatchEvent(new Event('change'));result.fallback=document.getElementById('fallback').textContent;
const fingerprint=()=>{draw();let p=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,p);let sum=0,ink=0;for(let i=0;i<p.length;i+=16){sum=(sum+p[i]*3+p[i+1]*7+p[i+2]*11)%2147483647;if(Math.abs(p[i]-14)+Math.abs(p[i+1]-26)+Math.abs(p[i+2]-37)>12)ink++}return {sum,ink}};
let original=selected;document.getElementById('time').value=0;draw();let before=document.getElementById('timeLabel').textContent;
document.getElementById('time').value=700;document.getElementById('time').dispatchEvent(new Event('input'));result.time_control=before!==document.getElementById('timeLabel').textContent;
let visible=fingerprint();document.getElementById('cbd').checked=false;document.getElementById('cbd').dispatchEvent(new Event('change'));let hidden=fingerprint();result.cbd_toggle=visible.sum!==hidden.sum;result.colored_samples=visible.ink;
document.getElementById('cbd').checked=true;document.getElementById('case').value=(original+1)%D.cases.length;document.getElementById('case').dispatchEvent(new Event('change'));result.case_change=selected!==original;
document.getElementById('axis').value='capacity';document.getElementById('axis').dispatchEvent(new Event('change'));
document.getElementById('case').value=original;document.getElementById('case').dispatchEvent(new Event('change'));document.getElementById('time').value=500;draw();
result.stations=document.getElementById('stations').children.length;result.page_errors=window.REPORT_ERRORS||[];
result.passed=result.cases>=43&&result.webgl&&result.partition_unique&&result.instances>1000&&result.time_control&&result.cbd_toggle&&result.case_change&&result.stations===5&&result.colored_samples>100&&!result.fallback&&!result.page_errors.length;
}catch(error){result.error=error.stack;result.passed=false}
let pre=document.createElement('pre');pre.id='browser-check';pre.textContent=JSON.stringify(result);document.body.append(pre);
},100));
</script>
"""

class Result(HTMLParser):
    def __init__(self):super().__init__();self.inside=False;self.data=''
    def handle_starttag(self,tag,attrs):
        if tag=='pre' and dict(attrs).get('id')=='browser-check':self.inside=True
    def handle_endtag(self,tag):
        if tag=='pre':self.inside=False
    def handle_data(self,data):
        if self.inside:self.data+=data

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--browser',type=Path);args=parser.parse_args();workspace=Path(os.environ['MORQEX_WORKSPACE'])
    candidates=sorted((Path.home()/'.cache/ms-playwright').glob('chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell'))
    browser=args.browser or (candidates[-1] if candidates else None)
    if browser is None or not browser.exists():raise FileNotFoundError('Supply a local headless Chromium executable with --browser')
    source=(workspace/'reports/MORQEX_full_volume_report.html').read_text();source=source.replace('<script type="application/json"',"<script>window.REPORT_ERRORS=[];window.addEventListener('error',e=>window.REPORT_ERRORS.push(e.message));</script><script type=\"application/json\"",1)
    page=Path.cwd()/'browser_test.html';page.write_text(source+CHECK);dom=Path.cwd()/'browser_dom.html';screenshot=workspace/'reports/MORQEX_full_volume_preview.png'
    command=[str(browser),'--headless','--no-sandbox','--disable-gpu','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--disable-extensions','--disable-sync',f'--user-data-dir={Path.cwd()/"browser_profile"}',
        '--window-size=1440,2500','--virtual-time-budget=1800','--run-all-compositor-stages-before-draw',f'--screenshot={screenshot}','--dump-dom',page.as_uri()]
    with dom.open('w') as output:result=subprocess.run(command,stdout=output,stderr=subprocess.PIPE,text=True,timeout=45)
    parser=Result();parser.feed(dom.read_text())
    if not parser.data:raise RuntimeError('Browser did not return qualification data: '+result.stderr[-3000:])
    record=json.loads(parser.data);record.update(browser=str(browser),browser_exit_code=result.returncode,screenshot=str(screenshot),rendering='isolated headless software WebGL; physical RSS/CPU guarded, GPU disabled')
    (workspace/'reports/full_volume_browser_check.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
    if result.returncode or not record['passed']:raise RuntimeError('Browser qualification failed; inspect the captured result and screenshot')
if __name__=='__main__':main()
