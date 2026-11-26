"""TOML-controlled, out-of-tree MFEM compilation and scalar FE API.

Only bounded serial CPU compilation is qualified. Native MFEM is not thread
safe in this profile: API calls are serialized, with no hidden thread pool.
"""
from dataclasses import dataclass
import ctypes as ct
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
import tomllib

from morqex.workspace import Workspace

_LOCK = threading.RLock()
_LOADED_MFEM_PATH = None
_NATIVE = Path(__file__).resolve().parent / 'native'
_VENDOR = _NATIVE.parents[1] / 'vendor' / 'mfem'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_local_budget():
    if not os.environ.get('MORQEX_SOURCE_SNAPSHOT') or len(os.sched_getaffinity(0))!=1:
        raise RuntimeError('Use scripts/run_local.py for the qualified one-CPU local backend')


@dataclass(frozen=True)
class NativeConfig:
    provider: str = 'vendored'
    build_type: str = 'Release'
    jobs: int = 1
    slice_seconds: int = 55
    precision: str = 'float64'
    device: str = 'cpu'

    def __post_init__(self):
        if self.provider not in ('vendored', 'pymfem-wheel'):
            raise ValueError('Unknown MFEM provider')
        if self.build_type != 'Release' or self.jobs != 1:
            raise ValueError('Qualified local build uses Release and one compiler job')
        if not isinstance(self.slice_seconds, int) or not 5 <= self.slice_seconds <= 55:
            raise ValueError('Build slice must be 5..55 seconds')
        if self.precision != 'float64' or self.device != 'cpu':
            raise ValueError('This reference backend is qualified only for CPU FP64')

    @classmethod
    def from_toml(cls, path):
        data = tomllib.loads(Path(path).read_text())
        if not {'schema_version', 'native'} <= set(data) or set(data)-{'schema_version','native','reference'} or data['schema_version'] != 1:
            raise ValueError('Expected version 1 native-backend TOML')
        return cls(**data['native'])


def build_identity(config):
    inputs = {'adapter': {p.name: _sha(p) for p in sorted(_NATIVE.iterdir()) if p.is_file()},
              'provider': config.provider, 'build_type': config.build_type}
    if config.provider == 'vendored':
        provenance = json.loads((_VENDOR / 'provenance.json').read_text())
        for name, digest in provenance['files'].items():
            if _sha(_VENDOR / name) != digest:
                raise ValueError(f'Upstream source changed without provenance update: {name}')
        inputs['upstream'] = _sha(_VENDOR / 'provenance.json')
    else:
        dist = importlib.metadata.distribution('mfem')
        if dist.version != '4.10.0.1':
            raise ValueError('Cross-check wheel must be mfem==4.10.0.1')
        prefix = Path(dist.locate_file('mfem/external/ser')).resolve()
        inputs.update(prefix=str(prefix), library=_sha(prefix / 'lib/libmfem.so'))
    inputs['compiler'] = subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(), inputs


def build_backend(config, workspace):
    """Compile one bounded slice; return pending until Ninja finishes.

    Run through scripts/run_local.py. Repeating the same call resumes objects;
    no compiled code or upstream download is ever written in the source tree.
    """
    require_local_budget()
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    identity, inputs = build_identity(config)
    directory = ws.path('build', 'top_damage', identity[:20])
    directory.mkdir(parents=True, exist_ok=True)
    ninja = shutil.which('ninja') or str(Path(sys.executable).parent / 'ninja')
    command = ['cmake', '-S', str(_NATIVE), '-B', str(directory), '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON',
               '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
               '-DCMAKE_MAKE_PROGRAM=' + ninja,
               '-DMORQEX_MFEM_FROM_SOURCE=' + ('ON' if config.provider == 'vendored' else 'OFF')]
    if config.provider == 'pymfem-wheel':
        command.append('-DMFEM_ROOT=' + inputs['prefix'])
    started = time.monotonic()
    with (directory / 'configure.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (directory / 'build.log').open('a') as log:
        proc = subprocess.Popen([ninja, '-C', str(directory), '-j1', 'morqex_phase_field'],
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=config.slice_seconds)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            code = proc.wait(timeout=15)
            # Ninja 1.13 can return zero after an interrupted build. A bounded
            # slice is never a successful qualification; rerun to confirm.
            state = 'pending'
        else:
            state = 'complete' if code == 0 else 'failed'
    library = directory / 'libmorqex_phase_field.so'
    dependency = (directory/'mfem/libmfem.so').resolve() if config.provider == 'vendored' else (Path(inputs['prefix'])/'lib/libmfem.so').resolve()
    result = {'state': state, 'identity': identity, 'inputs': inputs,
              'directory': str(directory), 'library': str(library),
              'seconds_this_slice': time.monotonic()-started,
              'library_sha256': _sha(library) if state == 'complete' else None,
              'dependency': str(dependency),
              'dependency_sha256': _sha(dependency) if state == 'complete' else None,
              'provider': config.provider, 'mfem_version': '4.10', 'device': 'cpu'}
    (directory / 'manifest.json').write_text(json.dumps(result, indent=2))
    if state == 'failed':
        raise RuntimeError(f'MFEM build failed; inspect {directory / "build.log"}')
    return result


class MFEMScalarBackend:
    """Consistent P1 tetrahedron mass + full-tensor diffusion weak forms."""
    def __init__(self, config, workspace):
        global _LOADED_MFEM_PATH
        ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        identity, _ = build_identity(config)
        directory = ws.path('build', 'top_damage', identity[:20])
        manifest = json.loads((directory / 'manifest.json').read_text())
        library = directory / 'libmorqex_phase_field.so'
        if manifest['state'] != 'complete' or manifest['identity'] != identity or _sha(library) != manifest['library_sha256']:
            raise ValueError('Missing, stale or modified native backend')
        if _sha(manifest['dependency']) != manifest['dependency_sha256']:
            raise ValueError('Modified MFEM shared-library dependency')
        self.manifest = manifest
        with _LOCK:
            # Two builds share the upstream SONAME. Loading both in one process
            # can silently reuse the first one; cross-check in separate jobs.
            if _LOADED_MFEM_PATH not in (None, manifest['dependency']):
                raise RuntimeError('Compare MFEM providers in separate processes')
            self._lib = ct.CDLL(str(library))
            _LOADED_MFEM_PATH = manifest['dependency']
        self._lib.mq_mfem_version.restype = ct.c_int
        if self._lib.mq_mfem_version() != 41000:
            raise ValueError('Wrong MFEM ABI version')
        f64 = ct.POINTER(ct.c_double); i32 = ct.POINTER(ct.c_int)
        self._lib.mq_scalar_form.argtypes = [ct.c_int, ct.c_int, f64, i32, f64, f64, f64, ct.c_char_p, ct.c_int]
        self._lib.mq_scalar_form.restype = ct.c_void_p
        self._lib.mq_form_nnz.argtypes = [ct.c_void_p]
        self._lib.mq_form_nnz.restype = ct.c_int
        self._lib.mq_form_copy.argtypes = [ct.c_void_p, i32, i32, f64, f64]
        self._lib.mq_form_copy.restype = None
        self._lib.mq_form_free.argtypes = [ct.c_void_p]
        self._lib.mq_form_free.restype = None

    def assemble(self, mesh, reaction, diffusion, source):
        require_local_budget()
        import numpy as np
        from scipy.sparse import csr_matrix
        from morqex.mechanics.mesh import TetMesh
        if not isinstance(mesh, TetMesh):
            raise TypeError('Validated MORQEX TetMesh required')
        n, e = len(mesh.points), len(mesh.cells)
        if max(n, e) >= np.iinfo(np.int32).max:
            raise ValueError('MFEM adapter uses 32-bit indices')
        xyz = np.ascontiguousarray(mesh.points, dtype=np.float64)
        cells = np.ascontiguousarray(mesh.cells, dtype=np.int32)
        a = np.array(np.broadcast_to(reaction, (e,)), dtype=np.float64, order='C')
        D = np.array(np.broadcast_to(diffusion, (e, 3, 3)), dtype=np.float64, order='C')
        f = np.array(np.broadcast_to(source, (e,)), dtype=np.float64, order='C')
        if not all(np.isfinite(v).all() for v in (a, D, f)) or np.any(a < 0):
            raise ValueError('Finite fields and nonnegative reaction required')
        scale = max(float(np.max(np.abs(D))), np.finfo(float).tiny)
        if np.max(np.abs(D-D.swapaxes(1, 2))) > 1e-13*scale or np.min(np.linalg.eigvalsh(D)) < -1e-13*scale:
            raise ValueError('Diffusion must be symmetric positive semidefinite')
        dp = lambda v: v.ctypes.data_as(ct.POINTER(ct.c_double))
        ip = lambda v: v.ctypes.data_as(ct.POINTER(ct.c_int))
        error = ct.create_string_buffer(4096)
        with _LOCK:
            handle = self._lib.mq_scalar_form(n, e, dp(xyz), ip(cells), dp(a), dp(D), dp(f), error, len(error))
            if not handle:
                raise RuntimeError(error.value.decode(errors='replace'))
            try:
                nnz = self._lib.mq_form_nnz(handle)
                I = np.empty(n+1, dtype=np.int32); J = np.empty(nnz, dtype=np.int32)
                V = np.empty(nnz); b = np.empty(n)
                self._lib.mq_form_copy(handle, ip(I), ip(J), dp(V), dp(b))
            finally:
                self._lib.mq_form_free(handle)
        A = csr_matrix((V, J, I), shape=(n,n)); A.sort_indices()
        return A, b
