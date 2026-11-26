import json,subprocess
from pathlib import Path
root=Path('/home/zephyrain/lw/morph_workspace');source=root/'reports/MORQEX_resolved_3D_phase_field.html';page=Path.cwd()/'checked.html'
page.write_text(source.read_text()+'''<script>setTimeout(()=>{$('frame').value=String(DATA.frames.length-1);$('field').value='damage';showFrame();document.body.dataset.finalframe=Number($('frame').value)===DATA.frames.length-1;document.body.dataset.overflow=document.documentElement.scrollWidth>innerWidth;},800)</script>''')
browser='/home/zephyrain/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell'
checks=[]
for width in [1440,412]:
 name=f'resolved_3D_browser_{width}'
 r=subprocess.run([browser,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--virtual-time-budget=4000',f'--window-size={width},1200','--user-data-dir='+str(Path.cwd()/name),'--screenshot='+str(root/'reports'/f'{name}.png'),'--dump-dom',page.as_uri()],capture_output=True,text=True,timeout=55)
 row={'width':width,'exit_code':r.returncode,'last_frame':'data-finalframe="true"' in r.stdout,'field':'data-field="damage"' in r.stdout,'no_overflow':'data-overflow="false"' in r.stdout,'limits_visible':'尚未组成长期物理拓扑闭环' in r.stdout}
 row['passed']=r.returncode==0 and all(v for k,v in row.items() if k not in ['width','exit_code']);checks.append(row)
(root/'reports/resolved_3D_browser_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2));assert all(r['passed'] for r in checks)
