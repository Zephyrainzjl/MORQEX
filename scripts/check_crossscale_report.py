"""Check embedded SVGs, provenance and all seven field-viewer modes offline."""
import base64
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import quickjs

WS=Path(os.environ['MORQEX_WORKSPACE'])
class Report(HTMLParser):
    def __init__(self):super().__init__();self.scripts={};self.active=None;self.figures=0;self.evidence=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='script':assert 'src' not in a;self.active=a['id'];self.scripts[self.active]=''
        if tag=='iframe':raise ValueError('Offline report required')
        if tag=='img':
            assert a['src'].startswith('data:image/svg+xml;base64,');ET.fromstring(base64.b64decode(a['src'].split(',',1)[1]));self.figures+=1
        if tag=='a' and a.get('download')=='MORQEX_crossscale_evidence.json':self.evidence=json.loads(base64.b64decode(a['href'].split(',',1)[1]))
    def handle_data(self,data):
        if self.active:self.scripts[self.active]+=data
    def handle_endtag(self,tag):
        if tag=='script':self.active=None

def main():
    path=WS/'reports/MORQEX_crossscale_cathodes_report.html';report=Report();report.feed(path.read_text());e=report.evidence
    assert report.figures>=6 and e['tests']>=180 and all(all(v.values()) for v in e['case_checks'].values())
    for p,h in e['artifacts'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    root=Path(__file__).resolve().parents[1]
    for p,h in e['release_source_sha256'].items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h
    assert {v['cathode_family'] for v in e['parameter_audits'].values()}=={'NMC','LFP','LCO'}
    context=quickjs.Context();context.set_memory_limit(128*2**20)
    context.eval('''const elements={},canvas={clearRect(){},fillRect(){},fillText(){}};const document={createElement(){return {}},getElementById(id){if(!elements[id])elements[id]={value:({domain:'negative',macro:'1',field:'soc',time:'0',depth:'0'})[id]||'',textContent:'',appendChild(x){if(!this.value)this.value=x.value},addEventListener(){},getContext(){return canvas}};return elements[id]}};''')
    context.eval("document.getElementById('viewer-data').textContent="+json.dumps(report.scripts['viewer-data']));context.eval(report.scripts['viewer-code'])
    result=json.loads(context.eval('''JSON.stringify((()=>{let checked=0,expected=0;for(const name of Object.keys(DATA)){el('case').value=name;const d=DATA[name];for(const domain of ['negative','positive']){el('domain').value=domain;const m=d.micro[domain];expected+=21*m.fields.soc.length;for(const ti of [0,Math.floor(d.time.length/2),d.time.length-1])for(let x=0;x<m.fields.soc.length;x++)for(const field of ['soc','vm','dopant','sei','plating','dead_lithium','damage']){el('time').value=String(ti);el('macro').value=String(x);el('field').value=field;draw();if(el('readout').textContent.includes('NaN')||m.fields[field][x][ti].some(v=>!Number.isFinite(v)))throw Error('Nonfinite field');checked++}}}return {checked,expected,cases:Object.keys(DATA).length};})())'''))
    assert result['checked']==result['expected']
    result.update(figures=report.figures,tests=e['tests'],scientific_cases=len(e['case_checks']),artifact_hashes='pass',html_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='QuickJS fixture, not browser rendering')
    (WS/'reports/crossscale_report_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
