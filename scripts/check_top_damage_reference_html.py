"""Browser-render validation of the computed report and interactive field view."""
import json
import os
from pathlib import Path
import subprocess
from morqex import Workspace

ws=Workspace(os.environ['MORQEX_WORKSPACE'])
browser=Path('/home/zephyrain/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell')
study=json.loads(ws.path('reports','top_damage_path_vendored.json').read_text())
pages={'report':ws.path('reports','MORQEX_TOP_DAMAGE_reference.html'),
       'fields':Path(study['cases'][-1]['directory'])/'index.html'}
checks={}
for name,page in pages.items():
    png=ws.path('reports',f'MORQEX_TOP_DAMAGE_{name}.png')
    result=subprocess.run([str(browser),'--headless','--no-sandbox','--disable-gpu',
        '--disable-dev-shm-usage','--disable-background-networking','--virtual-time-budget=1500',
        '--user-data-dir='+str(Path.cwd()/('profile_'+name)),'--window-size=1440,1100',
        '--screenshot='+str(png),'--dump-dom',page.as_uri()],text=True,capture_output=True,timeout=45)
    check={'exit_code':result.returncode,'png_bytes':png.stat().st_size if png.exists() else 0}
    if name=='fields':
        check['canvas_script_executed']='data-rendered="true"' in result.stdout
        check['last_step_slider']='max="192"' in result.stdout
        check['no_false_battery_cycles']='not battery charge/discharge' in result.stdout
    else:
        check['svg_rendered']='<svg ' in result.stdout
        check['limitations_visible']='多晶正极标定' in result.stdout
    check['passed']=all(v for k,v in check.items() if k!='exit_code') and result.returncode==0
    checks[name]=check
target=ws.path('reports','top_damage_reference_browser.json');target.write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
if not all(c['passed'] for c in checks.values()):raise RuntimeError('Browser report qualification failed')
