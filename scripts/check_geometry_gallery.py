"""Offline browser checks of ten real256³ datasets and actual Tet4 views."""
import os,json,subprocess
from pathlib import Path
root=Path(os.environ['MORQEX_WORKSPACE'])/'atlas256';html=root/'index.html'
browser='/home/zephyrain/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell'
output=root/'browser_checks';output.mkdir(exist_ok=True);checks=[]
cases=[(i,'voxel') for i in range(10)]+[(i,'mesh') for i in (0,1,4,6,8)]+[(1,'probe')]
for i,view in cases:
    png=output/f'{i+1:02d}_{view}.png'
    profile=Path.cwd()/f'profile_{i}_{view}'
    query=f'#model={i}&view={view}&cut=2'
    result=subprocess.run([browser,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
        '--disable-background-networking','--virtual-time-budget=6000','--window-size=1440,1700',
        '--user-data-dir='+str(profile),'--screenshot='+str(png),'--dump-dom',html.as_uri()+query],capture_output=True,text=True,timeout=60)
    okay=(result.returncode==0 and 'data-rendered="true"' in result.stdout and f'data-model="{i}"' in result.stdout
        and f'data-view="{view}"' in result.stdout and 'data-error=' not in result.stdout and png.exists() and png.stat().st_size>20000)
    checks.append({'model':i+1,'view':view,'passed':okay,'screenshot':str(png)})
    if not okay:
        (output/'failure_dom.html').write_text(result.stdout);(output/'failure_stderr.txt').write_text(result.stderr)
        raise RuntimeError(f'Offline browser check failed for model{i+1} view{view}')
    print(json.dumps(checks[-1]),flush=True)
(root/'browser_validation.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
