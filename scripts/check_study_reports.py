"""Static offline evidence reports: local asset integrity and software browser."""
import json,os,subprocess
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from morqex.workspace import Workspace

class Assets(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='img':self.links.append(attrs.get('src',''))
        if tag=='a':self.links.append(attrs.get('href',''))

ws=Workspace(os.environ['MORQEX_WORKSPACE'])
browser=Path('/home/zephyrain/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell')
checks=[]
for name in ('network_study','topology_study','mesh_study','article_draft'):
    page=ws.path('reports',name,'index.html');parser=Assets();parser.feed(page.read_text())
    missing=[]
    for value in parser.links:
        url=urlsplit(value)
        if not url.scheme and url.path and not (page.parent/unquote(url.path)).resolve().is_file():missing.append(value)
    if missing:raise ValueError('Broken report assets: '+str(missing))
    png=ws.path('reports',name,'browser.png')
    result=subprocess.run([str(browser),'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
        '--disable-background-networking','--virtual-time-budget=1800','--user-data-dir='+str(Path.cwd()/name),
        '--window-size=1440,1400','--screenshot='+str(png),'--dump-dom',page.as_uri()],
        capture_output=True,text=True,timeout=45)
    passed=result.returncode==0 and '<h1>' in result.stdout and png.exists() and png.stat().st_size>20000
    checks.append({'page':str(page),'passed':passed,'asset_count':len(parser.links),'screenshot':str(png)})
    if not passed:raise RuntimeError('Offline browser rendering failed: '+name)
ws.path('reports/study_browser_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
