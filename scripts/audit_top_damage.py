"""Read-only repository/environment inventory; all output uses MORQEX Workspace."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from morqex import Workspace


def main():
    root = Path(__file__).resolve().parents[1]
    ws = Workspace(os.environ['MORQEX_WORKSPACE'])
    out = ws.path('research_archive', 'top_damage', 'evidence')
    out.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines()
    rows = []
    for name in tracked:
        p = root / name
        if not p.is_file():
            continue
        raw = p.read_bytes()
        row = dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if p.suffix == '.py' and '/vendor/' not in name:
            try:
                tree = ast.parse(raw, filename=name)
                row['definitions'] = [n.name for n in ast.walk(tree)
                                      if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
                row['imports'] = sorted({n.module for n in ast.walk(tree)
                                         if isinstance(n, ast.ImportFrom) and n.module}
                                        | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
            except (SyntaxError, UnicodeError) as exc:
                row['parse_error'] = str(exc)
        rows.append(row)
    packages = ['numpy', 'scipy', 'pybamm', 'h5py', 'meshio', 'dolfinx', 'petsc4py',
                'phasefieldx', 'mfem', 'mpi4py', 'gmsh', 'porespy', 'openpnm', 'cupy']
    installed = {name: bool(importlib.util.find_spec(name)) for name in packages}
    tools = {name: shutil.which(name) for name in ['cmake', 'g++', 'ninja', 'mpicc', 'mpirun', 'nvcc', 'matlab', 'octave', 'Rscript']}
    cpu = Path('/proc/cpuinfo').read_text()
    hardware = {'python': sys.version, 'python_executable': sys.executable,
                'cpu_model': next((l.split(':', 1)[1].strip() for l in cpu.splitlines() if l.startswith('model name')), 'unknown'),
                'memory': [l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:', 'MemAvailable:'))]}
    if shutil.which('nvidia-smi'):
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10)
        hardware['gpu_inventory_only'] = gpu.stdout.strip() if gpu.returncode == 0 else 'unavailable'
    logs = []
    for p in sorted(ws.path('research_archive').rglob('*.md')):
        if 'top_damage' not in p.parts:
            raw = p.read_bytes()
            logs.append(dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    evidence = {
        'schema': 'morqex-top-damage-audit-1',
        'base_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'branch': subprocess.check_output(['git', '-C', str(root), 'branch', '--show-current'], text=True).strip(),
        'scope': 'Complete tracked-file inventory and non-vendor-directory Python AST scan (including integrated upstream namespaces); not a mathematical proof or full vendor semantic audit.',
        'files': rows, 'archived_documents': logs,
        'configs': {p.name: tomllib.loads(p.read_text()) for p in sorted((root/'configs').glob('*.toml'))},
        'installed_module_specs': installed, 'tools': tools, 'hardware': hardware,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'resource_policy': tomllib.loads((root/'configs/local.toml').read_text()),
    }
    initial = out/'repository_audit.json'
    destination = initial if not initial.exists() else out/'repository_audit_current.json'
    destination.write_text(json.dumps(evidence, indent=2))
    print(json.dumps({'files': len(rows), 'python_parsed': sum('definitions' in r for r in rows),
                      'parse_errors': [r['path'] for r in rows if 'parse_error' in r],
                      'installed': installed, 'tools': tools, 'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
