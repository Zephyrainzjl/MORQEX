"""Software-render the actual full voxel and clipped grain views offline."""
import json
import os
from pathlib import Path
import re
import subprocess
from morqex import Workspace

ws=Workspace(os.environ['MORQEX_WORKSPACE'])
page=ws.path('results/top_damage/microstructures/few_large_particles_32/index.html')
browser=Path('/home/zephyrain/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell')
checks={}
for name,query,mode,cut in [('particles','', 'particle',32),
                          ('grains','#cut=24&slice=23&mode=grain','grain',24),
                          ('cbd','#cbd=1&mode=phase','phase',32)]:
    png=ws.path('reports',f'MORQEX_TOP_DAMAGE_{name}.png')
    result=subprocess.run([str(browser),'--headless','--no-sandbox','--disable-gpu',
        '--disable-dev-shm-usage','--disable-background-networking','--virtual-time-budget=1800',
        '--user-data-dir='+str(Path.cwd()/('profile_'+name)),'--window-size=1440,1100',
        '--screenshot='+str(png),'--dump-dom',page.as_uri()+query],text=True,capture_output=True,timeout=45)
    faces=re.search(r'data-faces="(\d+)"',result.stdout)
    check={'exit_code':result.returncode,'png_bytes':png.stat().st_size if png.exists() else 0,
           'script_executed':'data-rendered="true"' in result.stdout,
           'expected_mode':f'data-mode="{mode}"' in result.stdout,
           'expected_cut':f'data-cut="{cut}"' in result.stdout,
           'rendered_faces':int(faces.group(1)) if faces else 0}
    check['passed']=result.returncode==0 and all(v for k,v in check.items() if k!='exit_code')
    checks[name]=check
checks['distinct_geometry_views']=len({c['rendered_faces'] for c in checks.values()})==3
ws.path('reports/top_damage_geometry_browser.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
if not checks['distinct_geometry_views'] or not all(c['passed'] for c in checks.values() if isinstance(c,dict)):
    raise RuntimeError('Full-voxel browser qualification failed')
