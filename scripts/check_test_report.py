"""Validate the actual offline artifact and exercise its JS using a DOM fixture.

This is not a full browser rendering test. QuickJS is an optional external test
dependency; it is not required to build/read the HTML report or use MORQEX.
"""
import base64
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.images, self.rows, self.controls = [], [], [], {}
        self.scripts, self.current, self.assets, self.anchors = {}, None, [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a: self.ids.append(a["id"])
        if tag == "script":
            if "src" in a: self.assets.append(a["src"])
            self.current = a.get("id", "anonymous"); self.scripts[self.current] = ""
        if tag == "img": self.images.append(a)
        if tag == "link" and a.get("rel") == "stylesheet": self.assets.append(a.get("href"))
        if tag == "tr" and "class" in a: self.rows.append(a)
        if tag == "a" and a.get("href", "").startswith("#"): self.anchors.append(a["href"][1:])

    def handle_data(self, data):
        if self.current is not None: self.scripts[self.current] += data

    def handle_endtag(self, tag):
        if tag == "script": self.current = None


def main():
    ws = Path(os.environ["MORQEX_WORKSPACE"])
    path = ws/"reports/MORQEX_test_report.html"
    document = path.read_text()
    parser = ReportParser(); parser.feed(document)
    assert document.lower().startswith("<!doctype html>")
    assert len(parser.ids) == len(set(parser.ids)), "Duplicate document IDs"
    assert not parser.assets, "Report unexpectedly requires external script/style assets"
    assert set(parser.anchors) <= set(parser.ids), "Broken section navigation"
    for image in parser.images:
        assert image.get("alt")
        prefix, encoded = image["src"].split(",", 1)
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(encoded, validate=True).startswith(b"\x89PNG\r\n\x1a\n")
    evidence = json.loads(parser.scripts["evidence-json"])
    test_rows = [r for r in parser.rows if "test-row" in r["class"]]
    method_rows = [r for r in parser.rows if "method-row" in r["class"]]
    assert len(test_rows) == len(evidence["tests"])
    expected_rows = sum(r["status"] == "measured" for c in evidence["controlled"]["cases"] for r in c["comparison"])
    assert len(method_rows) == expected_rows
    adaptive_rows = [r for r in parser.rows if "adaptive-row" in r["class"]]
    expected_adaptive = sum(len(p["methods"]) for c in evidence["energy_adaptivity"]["cases"] for p in c["results"])
    assert len(adaptive_rows) == expected_adaptive == 48
    assert len(parser.images) == 9
    scaling_rows=[r for r in parser.rows if "scaling-row" in r["class"]]
    assert len(scaling_rows)==sum(len(c["methods"]) for c in evidence["sparse_scaling"]["cases"])==54
    assert any(g["status"] == "fail" for g in evidence["research_gates"]), "Known research failures were omitted"
    for source in evidence["provenance"]:
        assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]
    import quickjs
    runtime = quickjs.Context(); runtime.set_memory_limit(64*2**20); runtime.set_time_limit(3)
    fixture_rows = {"methods": [{"dataset": {k[5:]:v for k,v in row.items() if k.startswith("data-")}, "hidden":False} for row in method_rows],
        "tests": [{"dataset": {k[5:]:v for k,v in row.items() if k.startswith("data-")}, "hidden":False} for row in test_rows]}
    runtime.eval("const rows="+json.dumps(fixture_rows)+"; const elements={};")
    runtime.eval("""
const controlIds=['method-filter','domain-filter','seed-filter','test-search','test-status','method-count','test-count','download-evidence','evidence-json'];
controlIds.forEach(id=>elements[id]={value:id==='test-search'?'':'all',textContent:'{}',addEventListener:function(name,fn){this[name]=fn;}});
const document={getElementById:id=>elements[id],querySelectorAll:selector=>selector==='.method-row'?rows.methods:rows.tests,createElement:()=>({click:()=>{}})};
const Blob=function(data,options){this.data=data;};const URL={createObjectURL:()=>'',revokeObjectURL:()=>{}};
""")
    runtime.eval(parser.scripts["report-interactions"])
    assert runtime.eval("rows.methods.filter(r=>!r.hidden).length") == len(method_rows)
    runtime.eval("elements['method-filter'].value='boundary-resolvent';elements['domain-filter'].value='held-out-time';elements['seed-filter'].value='104';applyMethods();")
    expected = sum(r["data-method"] == "boundary-resolvent" and r["data-domain"] == "held-out-time" and r["data-seed"] == "104" for r in method_rows)
    assert runtime.eval("rows.methods.filter(r=>!r.hidden).length") == expected
    runtime.eval("elements['test-search'].value='certificate';applyTests();")
    expected_tests = sum("certificate" in r["data-search"] for r in test_rows)
    assert runtime.eval("rows.tests.filter(r=>!r.hidden).length") == expected_tests
    runtime.eval("elements['test-status'].value='failed';applyTests();")
    expected_failed = sum("certificate" in r["data-search"] and r["data-status"] == "failed" for r in test_rows)
    assert runtime.eval("rows.tests.filter(r=>!r.hidden).length") == expected_failed
    runtime.eval("elements['download-evidence'].click();")
    result = {"html":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "images_embedded":len(parser.images),"actual_test_rows":len(test_rows),"method_rows":len(method_rows),"coupled_candidate_rows":len(adaptive_rows),"scaling_candidate_rows":len(scaling_rows),
        "source_artifact_hashes_verified":True,"offline_assets_only":True,"javascript_parse_and_filter_checks":"passed (QuickJS DOM fixture)",
        "full_browser_rendering":"not performed; no local browser available"}
    (ws/"reports/test_report_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
