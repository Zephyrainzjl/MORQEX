"""Software-browser checks of actual stored TetGen display subsets."""
import os,json,subprocess
from pathlib import Path
work=Path(os.environ.get('MORQEX_TETGEN_OUTPUT',str(Path(os.environ['MORQEX_WORKSPACE'])/'tetgen_validation')));out=work/'browser_checks';out.mkdir(exist_ok=True)
browser='/home/zephyrain/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell'
checks=[]
for name in ['preset_granular','preset_rods','preset_fibres','preset_lamellar','full256_irregular','toml_multiphase']:
 target=work/name/'mesh.html';png=out/(name+'.png')
 p=subprocess.run([browser,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking',
  '--virtual-time-budget=3000','--window-size=1440,900','--user-data-dir='+str(Path.cwd()/name),
  '--screenshot='+str(png),'--dump-dom',target.as_uri()],capture_output=True,text=True,timeout=45)
 assert p.returncode==0 and 'data-rendered="true"' in p.stdout and 'data-cells="2500"' in p.stdout
 assert png.stat().st_size>20000
 checks.append({'case':name,'passed':True,'screenshot':str(png),'cells_displayed':2500})
(work/'browser_validation.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
print(json.dumps(checks))
