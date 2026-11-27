"""Export the external evidence-qualified manuscript; never upgrade claims."""
import html,json,os,re
from pathlib import Path
import markdown
from morqex import Workspace

ws=Workspace(os.environ['MORQEX_WORKSPACE'])
source=ws.path('manuscript/top_damage_2s/MANUSCRIPT.md')
contract=ws.path('research_archive/top_damage/ARTICLE_EVIDENCE_CONTRACT.json')
evidence=json.loads(contract.read_text())
if evidence.get('publication_qualified') is not False:
    raise ValueError('This exporter is for the explicitly unqualified research draft only')
text=source.read_text()
if 'not a completed or submission-qualified article' not in text:
    raise ValueError('The manuscript must retain its explicit draft qualification')
# Keep TeX intact offline instead of relying on a remote math-rendering CDN.
text=re.sub(r'\\\[(.*?)\\\]',lambda m:'\n\n<pre class="equation">'+html.escape(m[1].strip())+'</pre>\n\n',text,flags=re.S)
text=re.sub(r'\\\((.*?)\\\)',lambda m:'<code>'+html.escape(m[1])+'</code>',text,flags=re.S)
body=markdown.markdown(text,extensions=['extra','toc'])
out=ws.path('reports/article_draft');out.mkdir(exist_ok=True)
figures=[('../network_study/rate_curves.svg','Actually integrated first-discharge curves, 0.1–8C; fixed-grid candidate.'),
         ('../network_study/capacity_history.svg','Actual completed cycles; capacity drift is not automatically fatigue fade.'),
         ('../mesh_study/spatial_study.svg','Same physical geometry and loading; retained spatial-convergence failures.')]
for file,_ in figures:
    if not (out/file).resolve().is_file():raise FileNotFoundError('Generate the numerical evidence dashboards first: '+file)
gallery=''.join('<figure><img src="'+file+'" alt="'+caption+'"><figcaption>'+caption+'</figcaption></figure>' for file,caption in figures)
document='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MORQEX research manuscript · verified results and open gates</title><style>
body{font:17px/1.75 Georgia,serif;background:#edf2f5;color:#152b3a;margin:0}main{max-width:960px;margin:auto;background:white;padding:45px}h1{font-size:30px;line-height:1.3}h2{border-top:1px solid #d5e0e7;padding-top:24px}h3{font-size:20px}.notice{background:#fff3d8;border-left:5px solid #b77e17;padding:16px;font-family:MORQEXReportCJK,system-ui}a{color:#0d697b}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:14px;font-size:14px}code{font-size:.85em;overflow-wrap:anywhere}img{width:100%}figure{margin:30px 0}figcaption{font:14px/1.5 system-ui;color:#425b6b}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #d5e0e7}@media(max-width:700px){main{padding:18px}}@media print{body{background:white}main{padding:0}.notice{border:1px solid #999}img{break-inside:avoid}}
</style><main><aside class="notice"><b>研究稿，未通过全部论文验收</b><p>下文区分实际数值结果、失败检查和未完成的物理机制。没有声称完成裂纹侵液拓扑闭环、千次循环预测或完整文献复现。公式保留离线可读的 TeX 源形式。</p><a href="../network_study/index.html">循环与场报告</a> · <a href="../mesh_study/index.html">空间误差</a> · <a href="evidence_contract.json">验收状态</a> · <a href="manuscript.md">可编辑稿件</a></aside>'''+body+'<h2>Computed figures</h2>'+gallery+'</main>'
font=ws.path('data/report_font')
if (font/'NotoSansCJKsc-Regular.otf').exists():
    from morqex.visualization.report_font import embed_font
    document=embed_font(document,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
(out/'index.html').write_text(document)
(out/'manuscript.md').write_text(source.read_text())
(out/'evidence_contract.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps({'html':str(out/'index.html'),'publication_qualified':False,'markdown_version':markdown.__version__}))
