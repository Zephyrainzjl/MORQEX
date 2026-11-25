"""Self-contained HTML report with explicit provenance and mapping scope."""
import json
from pathlib import Path

def render_report(payload,destination,*,font_path=None,font_license=None):
    if payload.get('schema')!='morqex-full-volume-report-1':raise ValueError('Unknown report schema')
    root=Path(__file__).parent
    data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    html=(root/'full_volume.html').read_text().replace('@@DATA@@',data).replace('@@SCRIPT@@',(root/'full_volume.js').read_text())
    if font_path is not None:
        if font_license is None:raise ValueError('Embedded font requires its license notice')
        from .report_font import embed_font
        html=embed_font(html,font_path,font_license)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(html);return destination
