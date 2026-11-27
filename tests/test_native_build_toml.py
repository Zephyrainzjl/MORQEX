"""Explicit compilation selection is independent of geometry generation."""
import os
from pathlib import Path
import pytest
from morqex.modeling import build_native_from_toml


def test_explicit_native_toml_builds_tetgen_into_external_cache(tmp_path):
    workspace=os.environ.get('MORQEX_WORKSPACE')
    if not workspace:pytest.skip('Run through the external resource guard')
    config=tmp_path/'build.toml';config.write_text('[native_build]\nbackends=["tetgen"]\n')
    libraries=build_native_from_toml(config,workspace=workspace)
    assert set(libraries)=={'tetgen'} and Path(libraries['tetgen']).is_file()
    assert Path(workspace).resolve() in Path(libraries['tetgen']).resolve().parents


@pytest.mark.parametrize('names',['[]','["unknown"]','["tetgen","tetgen"]','"tetgen"'])
def test_invalid_native_build_selection_has_no_compile_side_effect(tmp_path,names):
    config=tmp_path/'bad.toml';config.write_text('[native_build]\nbackends='+names+'\n')
    with pytest.raises(ValueError,match='backends'):build_native_from_toml(config,workspace=tmp_path/'cache')
    assert not (tmp_path/'cache').exists()
