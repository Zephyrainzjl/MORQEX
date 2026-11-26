"""Software browser gate for computed three-dimensional dense-cycle fields."""
import argparse,json,os,subprocess
from pathlib import Path
from morqex import Workspace
ws=Workspace(os.environ['MORQEX_WORKSPACE'])
parser=argparse.ArgumentParser();parser.add_argument('--case',default='gentle_voltage_v3');parser.add_argument('--step',type=int,default=19);args=parser.parse_args()
page=ws.path('reports/top_damage_dense_cases',args.case+'.html')
browser=Path('/home/zephyrain/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell')
result=subprocess.run([str(browser),'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
    '--disable-background-networking','--virtual-time-budget=4000','--window-size=1440,1300',
    '--user-data-dir='+str(Path.cwd()/'profile'),
    '--screenshot='+str(ws.path('reports/MORQEX_TOP_DAMAGE_dense_cycles.png')),'--dump-dom',page.as_uri()],
    capture_output=True,text=True,timeout=55)
checks={'exit_code':result.returncode,'accepted_frame_rendered':f'data-accepted="{args.step}"' in result.stdout,
        'correct_environment':f'data-environment="{args.case}"' in result.stdout,
        'strain_screen_visible':'小应变适用性筛查未通过' in result.stdout,
        'no_false_topology_claim':'尚未通过物理拓扑与论文验收' in result.stdout,
        'png_bytes':ws.path('reports/MORQEX_TOP_DAMAGE_dense_cycles.png').stat().st_size}
checks['passed']=result.returncode==0 and all(v for k,v in checks.items() if k!='exit_code')
ws.path('reports/top_damage_dense_browser.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if not checks['passed']:raise RuntimeError('Dense field browser check failed')
