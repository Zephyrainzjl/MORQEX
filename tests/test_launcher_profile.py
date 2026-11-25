"""Installation disk allowance must not enable a general command profile."""
from pathlib import Path
import subprocess
import sys


def test_dependency_profile_refuses_non_install_command_before_launch(tmp_path):
    launcher=Path(__file__).resolve().parents[1]/'scripts/run_local.py'
    marker=tmp_path/'must_not_exist'
    result=subprocess.run([sys.executable,str(launcher),'--label','invalid_profile','--dependency-install','--',
        sys.executable,'-c',f"open({str(marker)!r},'w').write('unexpected')"],capture_output=True,text=True)
    assert result.returncode==2
    assert 'restricted to python -m pip install' in result.stderr
    assert not marker.exists()
