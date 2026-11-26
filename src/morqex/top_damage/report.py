"""Standalone evidence report; never manufacture numerical result curves."""
from html import escape
import json
from pathlib import Path
from .cli import status

DOCUMENTS = ('PROJECT_AUDIT', 'RESEARCH_PLAN', 'LITERATURE_REVIEW', 'NOVELTY_MATRIX',
             'PHYSICAL_MODEL', 'MATHEMATICAL_FRAMEWORK', 'SOFTWARE_ARCHITECTURE',
             'BENCHMARK_PROTOCOL', 'RESOURCE_BUDGET', 'VALIDATION_REPORT', 'RESUME')


def build_report(workspace):
    root = workspace.path('research_archive', 'top_damage')
    documents = [(name, (root/(name+'.md')).read_text()) for name in DOCUMENTS]
    # Require the complete phase report; missing evidence never becomes a green badge.
    source = Path(__file__).resolve().parents[3]
    matrix = json.loads((source/'literature/top_damage_matrix.json').read_text())
    rows = []
    for row in matrix['rows']:
        rows.append('<tr><td>'+escape(str(row['year']))+'</td><td><a href="'+escape(row['source'], quote=True)+'">'+escape(row['title'])+'</a></td><td>'+escape(row['doi'])+'</td><td>'+escape(row['evidence'])+'</td></tr>')
    toc = ''.join('<a href="#'+name+'">'+name+'</a>' for name, _ in documents)
    sections = ''.join('<section id="'+name+'"><h2>'+name+'</h2><pre>'+escape(body)+'</pre></section>' for name, body in documents)
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MORQEX · TOP-DAMAGE 研究证据</title>
<style>body{margin:0;background:#eef2f6;color:#16283c;font:16px/1.65 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1200px;margin:auto;padding:32px}header,section{background:white;border-radius:16px;padding:24px;margin-bottom:20px}h1{margin:0;color:#126b70}nav{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}nav a{background:#dfecef;padding:7px 12px;border-radius:7px}a{color:#126b70}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ccd6df;padding:10px;text-align:left;overflow-wrap:anywhere}.notice{border-left:5px solid #be770e;padding:12px;background:#fff4df}section{scroll-margin-top:15px}@media(max-width:700px){main{padding:12px}td,th{padding:5px;font-size:12px}}</style>
<main><header><h1>MORQEX · TOP-DAMAGE</h1><p>拓扑演化与长期退化子框架 · Phase A 证据与实施方案 · 2026-11-26</p>
<p class="notice">当前交付是审计、文献筛查、数学方案和基础接口。未获得三维疲劳裂纹—动态均匀化—DFN 长期循环结果，不展示合成容量或裂纹演化曲线。</p>
<pre>'''+escape(json.dumps(status(), ensure_ascii=False, indent=2))+'''</pre></header><nav>'''+toc+'''</nav>
<section><h2>已核实文献入口</h2><p>能力未知不等于不存在；文献发表不等于本地复现。</p><table><thead><tr><th>年份</th><th>文献</th><th>DOI</th><th>证据范围</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></section>'''+sections+'''</main></html>'''
    font = workspace.path('data', 'report_font')
    if (font/'NotoSansCJKsc-Regular.otf').exists() and (font/'LICENSE').exists():
        from morqex.visualization.report_font import embed_font
        html = embed_font(html, font/'NotoSansCJKsc-Regular.otf', (font/'LICENSE').read_text())
    destination = workspace.path('reports', 'MORQEX_TOP_DAMAGE_phase_A.html')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(html)
    temporary.replace(destination)
    return destination
