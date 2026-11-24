"""Run original GEMsim tests with import-only migration in an external test directory."""
import os
from pathlib import Path
import re
import pytest

root = Path(__file__).resolve().parents[1]
workspace = Path(os.environ["MORQEX_WORKSPACE"])
original = workspace / "upstream/GEMsim/tests"
target = Path.cwd() / "upstream_tests"
target.mkdir()
for filename in ("test_pipeline.py", "test_physics.py", "test_periodic.py", "test_cell_contract.py"):
    text = (original / filename).read_text()
    text = re.sub(r"\bgem\b", "morqex.microstructure", text)
    (target / filename).write_text(text)
raise SystemExit(pytest.main([str(target), "-q", "--basetemp=" + str(Path.cwd() / "pytest_tmp"), "-p", "no:cacheprovider"]))

