"""Offline figure/provenance checks and execution of all field-viewer modes."""
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
    def __init__(self):super().__init__();self.scripts={};self.active=None;self.figures=0;self.download=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='script':assert 'src' not in a;self.active=a['id'];self.scripts[self.active]=''
        if tag=='iframe':raise ValueError('No remote report runtime')
        if tag=='img':
            assert a['src'].startswith('data:image/svg+xml;base64,');ET.fromstring(base64.b64decode(a['src'].split(',',1)[1]));self.figures+=1
        if tag=='a' and a.get('download')=='MORQEX_upgrade_evidence.json':self.download=json.loads(base64.b64decode(a['href'].split(',',1)[1]))
    def handle_data(self,text):
        if self.active:self.scripts[self.active]+=text
    def handle_endtag(self,tag):
        if tag=='script':self.active=None

def main():
    path=WS/'reports/MORQEX_degradation_CT_report.html';report=Report();report.feed(path.read_text());e=report.download
    assert report.figures>=10 and e['tests']>=166 and len(e['case_checks'])==11 and len(e['rejections'])==2
    assert all(all(v.values()) for v in e['case_checks'].values())
    for name,value in e['artifacts'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==value
    assert e['metrics']['ct_pulse']['reservoir_max_mol']['negative_plating']>0
    assert e['metrics']['ct_pulse']['reservoir_end_mol']['negative_dead_lithium']>0
    assert e['metrics']['ct_all']['damage_end']['positive']>0
    context=quickjs.Context();context.set_memory_limit(64*2**20)
    context.eval('''const elements={},canvas={clearRect(){},fillRect(){},fillText(){}};const document={createElement(){return {}},getElementById(id){if(!elements[id])elements[id]={value:({domain:'negative',macro:'1',field:'soc',time:'0',depth:'0'})[id]||'',textContent:'',appendChild(x){if(!this.value)this.value=x.value},addEventListener(){},getContext(){return canvas}};return elements[id]}};''')
    context.eval("document.getElementById('viewer-data').textContent="+json.dumps(report.scripts['viewer-data']));context.eval(report.scripts['viewer-code'])
    result=json.loads(context.eval('''JSON.stringify((()=>{let checked=0;for(const name of Object.keys(DATA)){el('case').value=name;const d=DATA[name];for(const domain of ['negative','positive']){el('domain').value=domain;const m=d.micro[domain];for(const ti of [0,Math.floor(d.time.length/2),d.time.length-1])for(let x=0;x<3;x++)for(const field of ['soc','vm','sei','plating','dead_lithium','damage']){el('time').value=String(ti);el('macro').value=String(x);el('field').value=field;draw();if(el('readout').textContent.includes('NaN')||m.fields[field][x][ti].some(v=>!Number.isFinite(v)))throw Error('Nonfinite field');checked++}}}return {checked};})())'''))
    assert result['checked']==324
    result.update(figures=report.figures,tests=e['tests'],valid_CT_cases=11,expected_physical_rejections=2,artifact_hashes='pass',
        html_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='QuickJS canvas/DOM fixture, not full browser rendering')
    (WS/'reports/upgrade_report_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
