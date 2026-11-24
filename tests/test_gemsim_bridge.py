import os
from pathlib import Path
import numpy as np
import pytest
from morph.gemsim import GEMsim
from morph.microstructure import Microstructure
from morph.workspace import Workspace


@pytest.fixture
def gem():
    return GEMsim(Workspace(Path(os.environ["MORPH_WORKSPACE"])))


def test_real_upstream_generation_roundtrip_and_axis_order(gem, tmp_path):
    structure = gem.random_field(shape=(4, 5, 6), fractions=(.25, .5, .25),
                                 length_scale=(1., 1., 1.), seed=20261124,
                                 spacing=(2e-6, 3e-6, 4e-6),
                                 phase_kinds={0: "pore", 1: "active", 2: "cbd"})
    volume = gem.to_volume(structure)
    assert volume.fractions == {0: .25, 1: .5, 2: .25}
    volume.origin = (11e-6, 13e-6, 17e-6)
    structure = gem.from_volume(volume, phase_kinds=structure.phase_kinds)
    path = volume.save(tmp_path / "gemsim.npz")
    restored = gem.load(path, phase_kinds=structure.phase_kinds)
    assert restored.fingerprint == structure.fingerprint
    mesh = gem.mesh(restored)
    assert len(mesh.cells) == 6 * 4 * 5 * 6
    volumes = mesh.signed_volumes()
    assert np.all(volumes > 0)
    np.testing.assert_allclose(mesh.points.min(0), [17e-6, 13e-6, 11e-6], rtol=1e-13)
    np.testing.assert_allclose(mesh.points.max(0), [41e-6, 28e-6, 19e-6], rtol=1e-13)
    assert volumes.sum() == pytest.approx(120 * 2e-6 * 3e-6 * 4e-6, rel=1e-12)
    for label in (0, 1, 2):
        assert (mesh.phases == label).sum() == 6 * (structure.labels == label).sum()


def test_reject_missing_semantics_and_gray_values(gem):
    a = np.zeros((2, 3, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="semantic"):
        gem.from_volume(gem.volume_type(a), phase_kinds={})
    with pytest.raises(ValueError, match="integer"):
        gem.from_volume(gem.volume_type(a.astype(float)), phase_kinds={0: "pore"})


def test_geometry_owns_input_and_cache_identity():
    a = np.zeros((2, 3, 4), dtype=np.uint8)
    s = Microstructure(a, (1., 2., 3.), (0., 0., 0.), {0: "pore"}, {})
    key = s.fingerprint
    a[:] = 255
    assert s.fingerprint == key
    with pytest.raises(ValueError):
        s.labels[0, 0, 0] = 1
    assert Microstructure(np.zeros_like(a), (2., 2., 3.), (0., 0., 0.), {0: "pore"}, {}).fingerprint != key


def test_workspace_rejects_source_and_escape():
    source = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError):
        Workspace(source / "outputs")
    with pytest.raises(ValueError):
        Workspace(source.parent / "morph_workspace").path("..", "MORPH", "bad.npy")


def test_native_geometry_export_is_explicit_and_x_fast(gem, tmp_path):
    from gem.simulate.electrochemistry import export_geometry
    a = np.array([0, 128, 255, 255, 128, 0], dtype=np.uint8).reshape(1, 2, 3)
    v = gem.volume_type(a, spacing=(1e-6, 2e-6, 3e-6))
    files = export_geometry(v, tmp_path, {0: 6, 128: 5, 255: 7})
    assert files["geometry"].read_text().splitlines()[0] == "3 2 1"
    np.testing.assert_array_equal(np.loadtxt(files["geometry"], skiprows=1), [6, 5, 7, 7, 5, 6])
    with pytest.raises(ValueError):
        export_geometry(v, tmp_path / "invalid", {0: 6})

