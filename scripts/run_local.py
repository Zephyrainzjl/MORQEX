#!/usr/bin/env python3
"""Single-job CPU-only Linux research launcher with process-tree resource limits."""
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
sys.path.insert(0, str(ROOT / "src"))
from morph.workspace import Workspace


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT.parent / "morph_workspace")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.label):
        parser.error("Label may contain only letters, digits, underscore and hyphen")
    cfg = tomllib.loads((ROOT / "configs/local.toml").read_text())["resources"]
    if cfg["gpu_enabled"]:
        raise SystemExit("GPU execution needs a separately validated resource policy")
    ws = Workspace(args.workspace)
    ws.root.mkdir(parents=True, exist_ok=True)
    lock = ws.path(".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another MORPH local job is running")
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
               MORPH_WORKSPACE=str(ws.root), GEMSIM_SOURCE=str(ws.path("upstream/GEMsim")),
               PYTHONPATH=str(ROOT / "src"), MPLCONFIGDIR=str(ws.path("cache/matplotlib")),
               XDG_CACHE_HOME=str(ws.path("cache")), HF_HOME=str(ws.path("cache/huggingface")),
               TORCH_HOME=str(ws.path("cache/torch")), TMPDIR=str(run / "tmp"),
               PYTHONPYCACHEPREFIX=str(ws.path("cache/pycache")))
    (run / "tmp").mkdir()
    cpus = allowed[-cfg["cpu_threads"]:]
    def limits():
        os.sched_setaffinity(0, cpus)
        os.nice(cfg["nice"])
        resource.setrlimit(resource.RLIMIT_AS, (cfg["address_space_mib"] * 2**20,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cfg["cpu_seconds"], cfg["cpu_seconds"] + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (cfg["file_size_mib"] * 2**20,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    manifest = {"schema_version": 1, "command": command, "cwd": str(run), "resources": cfg,
                "cpu_affinity": cpus, "started_utc": stamp, "python": sys.version,
                "gpu_visible": False, "source": str(ROOT)}
    started = time.monotonic()
    reason = None
    peak = 0
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
                        exit_code=None if proc is None else proc.returncode, stopped_because=reason)
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        ws.path("reports").mkdir(exist_ok=True)
        ws.path("reports", args.label + "_latest.json").write_text(json.dumps(manifest | {"run": str(run)}, indent=2))
    print(json.dumps(manifest | {"run": str(run)}, indent=2))
    if proc is None or proc.returncode != 0 or reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
