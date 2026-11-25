"""Fetch a pinned, openly licensed CJK font to the external workspace."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

if __name__=='__main__':
    destination=Path(os.environ['MORQEX_WORKSPACE'])/'data/report_font';destination.mkdir(parents=True,exist_ok=True);records=[]
    base='https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/'
    for name,relative in [('NotoSansCJKsc-Regular.otf','Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf'),('LICENSE','LICENSE')]:
        target=destination/name
        if not target.exists():
            temporary=target.with_suffix('.download')
            with urllib.request.urlopen(base+relative,timeout=40) as response,temporary.open('wb') as output:
                while block:=response.read(2**20):output.write(block)
            temporary.replace(target)
        records.append({'path':str(target),'url':base+relative,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest=destination/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=records:raise ValueError('Pinned font cache changed; inspect instead of replacing provenance')
    manifest.write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
