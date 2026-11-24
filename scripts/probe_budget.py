"""Verify enforced child limits rather than merely trusting configuration text."""
import json
import os
from pathlib import Path
import resource

assert len(os.sched_getaffinity(0)) == 1
assert os.getpriority(os.PRIO_PROCESS, 0) >= 15
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
assert os.environ["OMP_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == "1"
assert resource.getrlimit(resource.RLIMIT_AS)[0] == 3072 * 2**20
workspace = Path(os.environ["MORQEX_WORKSPACE"])
assert workspace in Path.cwd().parents
assert workspace in Path(os.environ["TMPDIR"]).parents
print(json.dumps({"affinity": sorted(os.sched_getaffinity(0)), "nice": os.getpriority(os.PRIO_PROCESS, 0),
                  "address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                  "gpu_visible": False, "cwd": str(Path.cwd()), "limits_verified": True}, indent=2))

