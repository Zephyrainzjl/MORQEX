import json
import re
from morqex.visualization import render_report

def test_standalone_report_escapes_embedded_data_and_names_baseline_scope(tmp_path):
    payload={'schema':'morqex-full-volume-report-1','source':'</script><script>wrong</script>'};path=render_report(payload,tmp_path/'report.html');text=path.read_text()
    embedded=re.search(r'<script type="application/json" id="data">(.*?)</script>',text,re.S).group(1)
    assert json.loads(embedded)==payload
    assert '@@DATA@@' not in text and '@@SCRIPT@@' not in text
    assert '不是已恢复的真实非均匀体素浓度' in text
