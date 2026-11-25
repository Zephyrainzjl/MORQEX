#!/usr/bin/env python3
"""Single numerical job with explicit CPU/GPU budgets and source provenance."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
# This parent imports the workspace contract before spawning a limited child.
# Prevent even these launcher imports from leaving bytecode in the source tree.
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from morqex.workspace import Workspace
from source_snapshot import snapshot,tree


def memory_info():
    return {line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()}


def group_rss(pgid):
    total = 0
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if os.getpgid(int(p.name)) == pgid:
                for line in (p / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]) * 1024
        except (OSError, ProcessLookupError):
            pass
    return total

def gpu_snapshot(device,pgid=None):
    result=subprocess.run(['nvidia-smi',f'--id={device}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free,util=map(int,result.stdout.strip().split(','));used=0
    if pgid is not None:
        apps=subprocess.run(['nvidia-smi',f'--id={device}','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        for line in apps.stdout.strip().splitlines():
            try:
                pid,memory=map(int,line.split(','))
                if os.getpgid(pid)==pgid:used+=memory
            except (ValueError,ProcessLookupError):continue
    return {'free_mib':free,'utilization_percent':util,'process_mib':used}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT.parent / "morph_workspace")
    parser.add_argument("--dependency-install", action="store_true", help="Allow external pip installation disk space; CPU/RAM/GPU/time limits unchanged")
    parser.add_argument('--gpu',action='store_true',help='Explicit, monitored low-priority GPU profile; CPU default stays disabled')
    parser.add_argument('--render',action='store_true',help='Headless software-render qualification: allow browser virtual address reservations, keep physical RSS/CPU caps and GPU disabled')
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.label):
        parser.error("Label may contain only letters, digits, underscore and hyphen")
    settings=tomllib.loads((ROOT / 'configs/local.toml').read_text());cfg=settings['resources'];gpu=settings['gpu']
    if args.dependency_install:
        if len(command) < 4 or command[1:4] != ["-m", "pip", "install"]:
            parser.error("Dependency disk profile is restricted to python -m pip install")
        cfg.update(file_size_mib=256, run_disk_mib=1024)
    if cfg['gpu_enabled']:raise SystemExit('GPU must be selected explicitly by --gpu')
    before_gpu=None
    if args.render:
        if args.gpu or args.dependency_install:parser.error('Render, GPU and installation profiles are separate')
        cfg.update(address_space_mib=0)
    if args.gpu:
        if args.dependency_install:parser.error('GPU and dependency installation are separate jobs')
        before_gpu=gpu_snapshot(gpu['device'])
        if before_gpu['free_mib']<gpu['maximum_process_memory_mib']+gpu['minimum_free_memory_mib'] or before_gpu['utilization_percent']>gpu['maximum_start_utilization_percent']:
            raise SystemExit('GPU is busy or lacks the reserved memory; no GPU job started')
        # CUDA reserves a large virtual address range. Keep the physical RSS
        # guard unchanged instead of mistaking that reservation for host RAM.
        cfg.update(gpu_enabled=True,address_space_mib=0,file_size_mib=1024)
    ws = Workspace(args.workspace)
    ws.root.mkdir(parents=True, exist_ok=True)
    lock = ws.path(".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another MORQEX local job is running")
    if memory_info()["MemAvailable"] < cfg["minimum_available_memory_mib"] * 2**20:
        raise SystemExit("Insufficient available memory; job not started")
    allowed = sorted(os.sched_getaffinity(0))
    if os.getloadavg()[0] > len(allowed) * cfg["maximum_load_per_cpu"]:
        raise SystemExit("Host load exceeds research budget; job not started")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = ws.path("runs", stamp + "_" + args.label)
    run.mkdir(parents=True)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = str(cfg["cpu_threads"])
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYBAMM_DISABLE_TELEMETRY="true",
               MORQEX_WORKSPACE=str(ws.root),
               PYTHONPATH=str(ROOT / "src"), MPLCONFIGDIR=str(ws.path("cache/matplotlib")),
               XDG_CACHE_HOME=str(ws.path("cache")), HF_HOME=str(ws.path("cache/huggingface")),
               TORCH_HOME=str(ws.path("cache/torch")), TMPDIR=str(run / "tmp"),
               PYTHONPYCACHEPREFIX=str(ws.path("cache/pycache")))
    if args.gpu:env.update(CUDA_VISIBLE_DEVICES=str(gpu['device']),MORQEX_GPU_POLICY=json.dumps(gpu),CUPY_CACHE_DIR=str(ws.path('cache/cupy')),CUDA_CACHE_PATH=str(ws.path('cache/cuda')))
    (run / "tmp").mkdir()
    frozen_source=snapshot(ROOT,ws.root)
    env['MORQEX_SOURCE_SNAPSHOT']=json.dumps(frozen_source)
    cpus = allowed[-cfg["cpu_threads"]:]
    def limits():
        os.sched_setaffinity(0, cpus)
        os.nice(cfg["nice"])
        if cfg['address_space_mib']:resource.setrlimit(resource.RLIMIT_AS, (cfg["address_space_mib"] * 2**20,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cfg["cpu_seconds"], cfg["cpu_seconds"] + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (cfg["file_size_mib"] * 2**20,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    manifest = {"schema_version": 1, "command": command, "cwd": str(run), "resources": cfg,
                "cpu_affinity": cpus, "started_utc": stamp, "python": sys.version,
                "gpu_visible": args.gpu,'gpu_policy':gpu if args.gpu else None,'gpu_before':before_gpu, "source": str(ROOT),'source_snapshot':frozen_source}
    started = time.monotonic()
    reason = None
    peak = 0
    gpu_peak=0;gpu_checked=0.
    proc = None
    try:
        with (run / "stdout.txt").open("w") as out, (run / "stderr.txt").open("w") as err:
            proc = subprocess.Popen(command, cwd=run, env=env, stdout=out, stderr=err,
                                    start_new_session=True, preexec_fn=limits)
            try:
                subprocess.run(["ionice", "-c", "3", "-p", str(proc.pid)], check=False, capture_output=True)
            except FileNotFoundError:
                pass
            while proc.poll() is None:
                rss = group_rss(proc.pid)
                peak = max(peak, rss)
                disk = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
                if time.monotonic() - started > cfg["wall_seconds"]:
                    reason = "wall time limit"
                elif rss > cfg["resident_memory_mib"] * 2**20:
                    reason = "process group RSS limit"
                elif disk > cfg["run_disk_mib"] * 2**20:
                    reason = "run disk limit"
                elif memory_info()["MemAvailable"] < cfg["minimum_available_memory_mib"] * 2**20:
                    reason = "host memory reserve"
                if args.gpu and time.monotonic()-gpu_checked>=1.:
                    sample=gpu_snapshot(gpu['device'],proc.pid);gpu_checked=time.monotonic();gpu_peak=max(gpu_peak,sample['process_mib'])
                    if not sample['process_mib']:
                        # WSL can report [N/A] per-PID VRAM. Include all global
                        # growth conservatively rather than claiming zero use.
                        sample['process_mib']=max(0,before_gpu['free_mib']-sample['free_mib']);gpu_peak=max(gpu_peak,sample['process_mib'])
                        manifest['gpu_memory_accounting']='conservative global free-memory delta when per-PID data unavailable'
                    if sample['process_mib']>gpu['maximum_process_memory_mib']:reason='GPU process memory limit'
                    elif sample['free_mib']<gpu['minimum_free_memory_mib']:reason='GPU memory reserve'
                if reason:
                    os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(0.25)
            proc.wait()
    finally:
        if proc is not None:
            # Also reap orphan subprocesses left after the main command exits.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        manifest.update(wall_seconds=time.monotonic()-started, peak_group_rss_bytes_sampled=peak,
                        peak_gpu_process_mib_sampled=gpu_peak,
                        exit_code=None if proc is None else proc.returncode, stopped_because=reason)
        manifest['source_changed_during_run']=tree(ROOT)[0]!=frozen_source['tree_sha256']
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        ws.path("reports").mkdir(exist_ok=True)
        ws.path("reports", args.label + "_latest.json").write_text(json.dumps(manifest | {"run": str(run)}, indent=2))
    print(json.dumps(manifest | {"run": str(run)}, indent=2))
    if proc is None or proc.returncode != 0 or reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
