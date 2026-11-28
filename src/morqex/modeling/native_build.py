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
    from .cgal_backend import build_cgal
    path=Path(path).resolve();raw=tomllib.loads(path.read_text())
    if 'native_build' not in raw:raise ValueError('Explicit [native_build] section required')
    config=dict(raw['native_build'])
    if set(config)-{'workspace','backends','cgal_dependency_prefix'}:raise ValueError('Unknown native build setting')
    names=config.get('backends');callbacks={'tetgen':build_tetgen,'acvd':build_acvd,'mmg':build_mmg,'cgal':build_cgal}
    if not isinstance(names,list) or not names or any(not isinstance(n,str) or n not in callbacks for n in names) or len(set(names))!=len(names):
        raise ValueError('backends must explicitly list unique supported names: tetgen, acvd, mmg, cgal')
    destination=workspace or config.get('workspace')
    if destination is None:raise ValueError('Explicit external native-build workspace required')
    destination=external_path((path.parent/Path(destination)).resolve())
    result={}
    for name in names:
        kwargs={}
        if name=='cgal' and config.get('cgal_dependency_prefix'):
            kwargs['dependency_prefix']=(path.parent/config['cgal_dependency_prefix']).resolve()
        result[name]=str(callbacks[name](destination,**kwargs))
    return result
