"""Thin isolated entry point, distinct from modules exported by the public API."""
import sys
from pathlib import Path

if __name__=='__main__':
    if sys.argv[1]=='mmg':
        from .remeshing import _worker_main
        _worker_main(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=='partition':
        from .partition_workflow import _worker
        _worker(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
    elif sys.argv[1]=='mmg-artifact':
        from .remeshing import _artifact_worker
        _artifact_worker(sys.argv[2],sys.argv[3],sys.argv[4])
    else:raise ValueError('Unknown native geometry worker')
