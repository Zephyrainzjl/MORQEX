"""Qualify the real standalone report in a software-rendered browser."""
import json
import os
from pathlib import Path
import subprocess
from html.parser import HTMLParser

from morqex import Workspace
from morqex.top_damage.report import DOCUMENTS

ws = Workspace(os.environ['MORQEX_WORKSPACE'])
page = ws.path('reports', 'MORQEX_TOP_DAMAGE_phase_A.html')
browser = Path('/home/zephyrain/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell')
result = subprocess.run([str(browser), '--headless', '--no-sandbox', '--disable-gpu',
                         '--disable-dev-shm-usage', '--disable-background-networking',
                         '--user-data-dir='+str(Path.cwd()/'profile'), '--window-size=1440,1000',
                         '--screenshot='+str(ws.path('reports', 'MORQEX_TOP_DAMAGE_phase_A.png')),
                         '--dump-dom', page.as_uri()], text=True, capture_output=True, timeout=45)

class Inspect(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.links=[]; self.rows=0
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag == 'a': self.links.append(a.get('href', ''))
        if tag == 'tr': self.rows += 1

dom = Inspect(); dom.feed(result.stdout)
checks = {'exit_code': result.returncode, 'documents': len(dom.ids), 'literature_rows': dom.rows-1,
          'all_document_anchors': all(name in dom.ids and '#'+name in dom.links for name in DOCUMENTS),
          'unique_ids': len(dom.ids) == len(set(dom.ids)),
          'lifetime_unqualified_visible': 'can_predict_lifetime' in result.stdout and 'false' in result.stdout}
checks['passed'] = (result.returncode == 0 and checks['documents'] == 11 and checks['literature_rows'] == 13
                    and checks['all_document_anchors'] and checks['unique_ids'] and checks['lifetime_unqualified_visible'])
ws.path('reports', 'top_damage_browser_check.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
if not checks['passed']: raise RuntimeError('Report browser qualification failed: '+result.stderr[-2000:])
