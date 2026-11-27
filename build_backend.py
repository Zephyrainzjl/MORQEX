"""PEP517 adapter: install Cython/NumPy build headers only for opt-in kernels."""
import os
from setuptools import build_meta as _backend


def _requirements(config_settings=None):
    if any(os.environ.get(k)=='1' for k in ('MORQEX_BUILD_ACVD','MORQEX_BUILD_GEOMETRY')):
        return ['Cython>=3.0','numpy>=2.1']
    return []


get_requires_for_build_wheel=_requirements
get_requires_for_build_sdist=_requirements
get_requires_for_build_editable=_requirements


def __getattr__(name):return getattr(_backend,name)
