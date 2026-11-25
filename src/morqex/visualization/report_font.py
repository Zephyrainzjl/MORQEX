"""Embed a renamed OFL CJK glyph subset; no font binary lives in source."""
import base64
import io
from html import escape

def embed_font(html,path,license_text):
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font=TTFont(path);options=subset.Options();options.name_IDs=['*'];options.name_languages=['*'];options.name_legacy=True
    selected=subset.Subsetter(options=options);selected.populate(text=''.join(sorted(set(html))));selected.subset(font)
    for record in font['name'].names:
        if record.nameID in {1,4,6,16}:record.string='MORQEXReportCJK'.encode(record.getEncoding())
    if 'CFF ' in font:
        cff=font['CFF '].cff;cff.fontNames=['MORQEXReportCJK'];top=cff.topDictIndex[0];top.FullName='MORQEXReportCJK';top.FamilyName='MORQEXReportCJK'
    output=io.BytesIO();font.save(output);data=base64.b64encode(output.getvalue()).decode()
    style="<style>@font-face{font-family:MORQEXReportCJK;src:url(data:font/otf;base64,"+data+") format('opentype');font-weight:100 900;font-display:swap}body,select,button{font-family:MORQEXReportCJK,system-ui,sans-serif}</style>"
    notice='<details style="max-width:1400px;margin:auto;padding:20px"><summary>字体来源与 OFL 许可证</summary><p>Report glyph subset derived from Noto Sans CJK SC 2.004, renamed MORQEXReportCJK.</p><pre>'+escape(license_text)+'</pre></details>'
    return html.replace('</style>', '</style>'+style,1).replace('</main>','</main>'+notice,1)
