import json
from pathlib import Path
from html.parser import HTMLParser
from morqex import Workspace
from morqex.top_damage.cli import main, status
from morqex.top_damage.report import DOCUMENTS, build_report


def test_status_does_not_claim_fatigue_or_long_cycles(capsys):
    main(['status'])
    output = json.loads(capsys.readouterr().out)
    assert not output['can_predict_lifetime']
    assert not output['neural_models']
    assert 'full long-cycle closure' in output['not_qualified']


def test_report_escapes_input_and_requires_complete_evidence(tmp_path):
    import pytest
    ws = Workspace(tmp_path)
    folder = ws.path('research_archive', 'top_damage')
    folder.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        build_report(ws)
    for name in DOCUMENTS:
        (folder/(name+'.md')).write_text('<script>not-executable</script>')
    path = build_report(ws)
    html = path.read_text()
    HTMLParser().feed(html)
    assert '<script>not-executable</script>' not in html
    assert '&lt;script&gt;not-executable&lt;/script&gt;' in html
    assert all('id="'+name+'"' in html for name in DOCUMENTS)
    assert path.is_relative_to(tmp_path)


def test_unified_dispatch_preserves_existing_entrypoint(monkeypatch):
    import sys
    import morqex.cli
    import morqex.top_damage.cli
    called = []
    monkeypatch.setattr(sys, 'argv', ['morqex', 'top-damage', 'status'])
    monkeypatch.setattr(morqex.top_damage.cli, 'main', lambda argv: called.append(argv))
    morqex.cli.main()
    assert called == [['status']]
