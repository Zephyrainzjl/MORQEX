"""Check offline SVGs, embedded numerical evidence, hashes and complete cycles."""
import base64
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
WS=Path(os.environ['MORQEX_WORKSPACE'])


class Report(HTMLParser):
    def __init__(self):super().__init__();self.images=[];self.download=None
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag in {'script','iframe'}:raise ValueError('Report must not require executable or remote content')
        if tag=='img':
            assert attrs['src'].startswith('data:image/svg+xml;base64,')
            svg=base64.b64decode(attrs['src'].split(',',1)[1]);ET.fromstring(svg);self.images.append(svg)
        if tag=='a' and attrs.get('download')=='MORQEX_DFN_evidence.json':
            self.download=json.loads(base64.b64decode(attrs['href'].split(',',1)[1]))


def main():
    path=WS/'reports/MORQEX_DFN_report.html';report=Report();report.feed(path.read_text())
    assert len(report.images)>=12 and report.download
    evidence=report.download;cases=evidence['cases'];assert len(cases)==19
    assert len(cases['coupled_cycles']['protocol']['cycles'])==3
    assert all(len(c)==5 for c in cases['coupled_cycles']['protocol']['cycles'])
    for name,result in cases.items():
        assert result['protocol']['completed']
        path=WS/'results/dfn'/name/'result.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==result['artifact_sha256']
        assert result['metrics']['relative_lithium_drift']<1e-10
        charge=max(result['metrics']['relative_electrode_charge_balance_residual'].values())
        if name=='casadi_loose_ic':
            assert charge>5e-5 and not result['numerical_checks']['charge_residual_below_5e-5']
        else:assert charge<5e-5 and all(result['numerical_checks'].values())
        if 'thermal_ODE_relative_ledger_residual' in result['metrics']:assert result['metrics']['thermal_ODE_relative_ledger_residual']<1e-6
        for relative,digest in result['source_sha256'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    assert cases['comsol_fine']['metrics']['comsol_voltage_rmse_V']<.001
    assert cases['comsol_fine']['metrics']['comsol_voltage_max_error_V']<.005
    assert evidence['test_count']>=128
    assert len(evidence['experimental_comparisons'])==6
    assert cases['coupled_crosscheck']['protocol']['scope'].startswith('120-second')
    assert cases['coupled_crosscheck']['solver']['class']=='CasadiSolver'
    assert cases['coupled_crosscheck_idaklu']['solver']['class']=='IDAKLUSolver'
    assert evidence['installed_wheel_proof']['upstream_distribution_installed'] is False
    assert set(evidence['localization'])=={'negative','positive'}
    print(json.dumps({'offline_SVGs':len(report.images),'embedded_cases':len(cases),'complete_cycle_steps':15,'test_count':evidence['test_count'],'evidence_hashes':'pass','scope':'numerical gates, not material/innovation/publication certification'}))


if __name__=='__main__':main()
