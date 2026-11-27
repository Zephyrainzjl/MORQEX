"""Explicit TOML selection of source-integrated native geometry backends."""
from pathlib import Path
import tomllib


def build_native_from_toml(path,*,workspace=None):
    """Compile only explicitly listed backends, one compiler job at a time.

    Geometry generation never implicitly executes this build section. Use the
    external run_local guard for CPU/RAM/host-reserve enforcement.
    """
    from .voxel import external_path
    from .tetgen_backend import build_tetgen
    from .acvd_backend import build_acvd
    from .mmg_backend import build_mmg
    path=Path(path).resolve();raw=tomllib.loads(path.read_text())
    if 'native_build' not in raw:raise ValueError('Explicit [native_build] section required')
    config=dict(raw['native_build'])
    if set(config)-{'workspace','backends'}:raise ValueError('Unknown native build setting')
    names=config.get('backends');callbacks={'tetgen':build_tetgen,'acvd':build_acvd,'mmg':build_mmg}
    if not isinstance(names,list) or not names or any(not isinstance(n,str) or n not in callbacks for n in names) or len(set(names))!=len(names):
        raise ValueError('backends must explicitly list unique supported names: tetgen, acvd, mmg')
    destination=workspace or config.get('workspace')
    if destination is None:raise ValueError('Explicit external native-build workspace required')
    destination=external_path((path.parent/Path(destination)).resolve())
    return {name:str(callbacks[name](destination)) for name in names}
