"""Check numerical provenance, portable figures and execute field-viewer controls."""
import base64
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import quickjs

WS=Path(os.environ['MORQEX_WORKSPACE']);ROOT=Path(__file__).resolve().parents[1]

class Report(HTMLParser):
    def __init__(self):super().__init__();self.scripts={};self.active=None;self.images=0;self.download=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='script':
            assert 'src' not in a;self.active=a['id'];self.scripts[self.active]=''
        if tag=='iframe':raise ValueError('No remote runtime allowed')
        if tag=='img':
            assert a['src'].startswith('data:image/svg+xml;base64,');ET.fromstring(base64.b64decode(a['src'].split(',',1)[1]));self.images+=1
        if tag=='a' and a.get('download')=='MORQEX_microstructure_evidence.json':self.download=json.loads(base64.b64decode(a['href'].split(',',1)[1]))
    def handle_endtag(self,tag):
        if tag=='script':self.active=None
    def handle_data(self,data):
        if self.active:self.scripts[self.active]+=data

def main():
    report=Report();path=WS/'reports/MORQEX_microstructure_report.html';report.feed(path.read_text());e=report.download
    assert report.images>=9 and e['test_count']>=149
    assert len(e['case_checks'])==17 and all(all(v.values()) for v in e['case_checks'].values())
    for name,value in e['artifacts_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==value
    assert all(e['runtime']['checks'].values())
    context=quickjs.Context();context.set_memory_limit(64*2**20)
    context.eval('''const elements={};const context={clearRect(){},fillRect(){},fillText(){}};
      const document={createElement(){return {}},getElementById(id){if(!elements[id])elements[id]={value:({domain:'negative',macro:'0',field:'soc',time:'0',depth:'0'})[id]||'',textContent:'',addEventListener(){},appendChild(x){if(!this.value)this.value=x.value},getContext(){return context}};return elements[id]}};''')
    context.eval("document.getElementById('viewer-data').textContent="+json.dumps(report.scripts['viewer-data']))
    context.eval(report.scripts['viewer-code'])
    checks=context.eval('''JSON.stringify((()=>{let checked=0,worst=0;
      for(const name of Object.keys(DATA)){el('case').value=name;const d=DATA[name];
        for(const domain of ['negative','positive']){el('domain').value=domain;const m=d.micro[domain];
          for(const ti of [0,Math.floor(d.time.length/2),d.time.length-1])for(let xi=0;xi<3;xi++){
            const v=fields(m,ti,xi,d.temperature[ti]);if(v.some(row=>Object.values(row).some(x=>!Number.isFinite(x))))throw Error('Nonfinite field');
            const j=m.area.reduce((s,a,k)=>s+a*m.j[xi][ti][k],0),ref=m.jv[xi][ti];worst=Math.max(worst,Math.abs(j-ref)/Math.max(1,Math.abs(ref)));
            el('time').value=String(ti);el('macro').value=String(xi);
            for(const key of ['soc','xx','vm','mu']){el('field').value=key;draw();if(el('readout').textContent.includes('NaN'))throw Error('NaN viewer');checked++}
          }
        }
      }return {checked,worst_reaction_relative:worst};})())''')
    checks=json.loads(checks);assert checks['checked']==360 and checks['worst_reaction_relative']<1e-9
    checks.update(svg_figures=report.images,cases=len(e['case_checks']),tests=e['test_count'],hashes='pass',
        scope='QuickJS DOM/canvas fixture, not a full browser rendering qualification',html_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (WS/'reports/microstructure_report_validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
