import os,json,subprocess
from pathlib import Path
from html.parser import HTMLParser
ws=Path(os.environ['MORQEX_WORKSPACE']);source=(ws/'reports/MORQEX_sphere32_Li_geometry.html').read_text()
check='''<script>setTimeout(()=>{let r={ready:window.MORQEX_PACK_READY,nodes:document.querySelectorAll('.node').length,voxels:PACK.labels.length,particles:PACK.metadata.particle_count};try{document.querySelectorAll('.node')[5].click();r.node6=document.getElementById('title').textContent.includes('6');const hash=()=>document.getElementById('volume').toDataURL();let before=hash();document.getElementById('phase').value='cbd';document.getElementById('phase').dispatchEvent(new Event('input'));r.phase_changed=before!==hash();document.getElementById('phase').value='active';document.getElementById('cut').value=24;document.getElementById('cut').dispatchEvent(new Event('input'));r.cut=document.getElementById('cutText').textContent==='24/32';r.passed=r.ready&&r.nodes===6&&r.voxels===32768&&r.particles===56&&r.node6&&r.phase_changed&&r.cut;}catch(e){r.error=e.stack;r.passed=false}let p=document.createElement('pre');p.id='check';p.textContent=JSON.stringify(r);document.body.append(p)},500);</script>'''
p=Path.cwd()/'check.html';p.write_text(source+check)
browser=Path('/home/zephyrain/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell')
screenshot=ws/'reports/MORQEX_sphere32_Li_geometry.png'
r=subprocess.run([str(browser),'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking',f'--user-data-dir={Path.cwd()/"profile"}','--window-size=1440,1200','--virtual-time-budget=1600',f'--screenshot={screenshot}','--dump-dom',p.as_uri()],capture_output=True,text=True,timeout=40)
class Read(HTMLParser):
 def __init__(self):super().__init__();self.inside=False;self.value=''
 def handle_starttag(self,t,a):
  if t=='pre' and dict(a).get('id')=='check':self.inside=True
 def handle_endtag(self,t):
  if t=='pre':self.inside=False
 def handle_data(self,d):
  if self.inside:self.value+=d
out=Read();out.feed(r.stdout)
if not out.value:raise RuntimeError(r.stderr[-3000:])
v=json.loads(out.value);(ws/'reports/sphere32_browser_check.json').write_text(json.dumps(v,indent=2));print(json.dumps(v));assert v['passed']
