import ast
from importlib.resources import files
import json
from pathlib import Path
import sys
import numpy as np
from morqex import microstructure


def test_namespace_is_self_contained():
    from morqex.gemsim import GEMsim
    from morqex.workspace import Workspace
    import os
    provider = GEMsim(Workspace(Path(os.environ["MORQEX_WORKSPACE"])))
    assert provider.volume_type is microstructure.Volume
    source = Path(__file__).resolve().parents[1] / "src"
    assert source in Path(microstructure.__file__).resolve().parents
    assert "gem" not in sys.modules
    assert not any("upstream/GEMsim/src" in p for p in sys.path)


def test_integrated_resources_and_case_export(tmp_path):
    from morqex.microstructure.simulate.electrochemistry import prepare_case
    template = files("morqex.microstructure").joinpath("resources/echem_microstructure.inputs")
    assert "geometry.is_periodic" in template.read_text()
    v = microstructure.Volume(np.ones((2, 3, 4), dtype=np.uint8), spacing=(1e-6, 2e-6, 3e-6))
    inputs = prepare_case(v, tmp_path, {1: 5})
    assert inputs.is_file()
    extent_line = next(line for line in inputs.read_text().splitlines() if line.startswith("geometry.prob_hi"))
    np.testing.assert_allclose([float(x) for x in extent_line.split("=", 1)[1].split()], [12e-6, 6e-6, 2e-6])
    assert (tmp_path / "input_volume.npz").is_file()


def test_all_integrated_python_is_syntactically_valid():
    for path in Path(microstructure.__file__).parent.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))


def test_migration_preserves_complete_source_inventory():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "provenance/gemsim_migration.json").read_text())
    assert len(manifest["files"]) == 274
    assert all((root / row["integrated"]).is_file() for row in manifest["files"])


def test_original_serialization_module_can_be_resolved_without_global_alias():
    from io import BytesIO
    from morqex.microstructure.io import _LegacyUnpickler
    assert _LegacyUnpickler(BytesIO()).find_class("gem.volume", "Volume") is microstructure.Volume
    assert "gem" not in sys.modules


def test_resource_launcher_import_path_does_not_initialize_numerical_libraries():
    import subprocess
    subprocess.run([sys.executable, "-c", "import sys; import morqex.workspace; assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules"], check=True)
