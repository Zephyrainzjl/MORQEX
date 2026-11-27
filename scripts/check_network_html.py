"""Headless offline interaction checks; run through run_local.py --render."""
import argparse,json,os,re,subprocess
from pathlib import Path
from morqex.workspace import Workspace

p=argparse.ArgumentParser();p.add_argument('report');args=p.parse_args()
ws=Workspace(os.environ['MORQEX_WORKSPACE']);page=ws.path('reports',args.report)
browser=Path('/home/zephyrain/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell')
checks={}
for name,query,field,node in [('damage','','d',0),('initial','#frame=0&field=c&node=2&cut=4','c',2),
                              ('network','#field=electron&node=1&cbd=1','electron',1)]:
    png=ws.path('reports',page.stem+'_browser_'+name+'.png')
    result=subprocess.run([str(browser),'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
        '--disable-background-networking','--virtual-time-budget=1600','--user-data-dir='+str(Path.cwd()/name),
        '--window-size=1440,1100','--screenshot='+str(png),'--dump-dom',page.as_uri()+query],
        text=True,capture_output=True,timeout=40)
    ok=result.returncode==0 and 'data-rendered="true"' in result.stdout
    ok &= f'data-field="{field}"' in result.stdout and f'data-node="{node}"' in result.stdout
    if name=='initial':ok &= 'data-frame="0"' in result.stdout
    checks[name]={'passed':bool(ok),'screenshot':str(png),'bytes':png.stat().st_size if png.exists() else 0}
ws.path('reports','network_browser_checks_'+page.stem+'.json').write_text(json.dumps(checks,indent=2))
ws.path('reports/network_browser_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if not all(x['passed'] and x['bytes']>10000 for x in checks.values()):raise RuntimeError('Offline graph/voxel viewer failed')
