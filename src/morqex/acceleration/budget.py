"""Process-local GPU limits. Never change clocks, global MPS or other jobs."""
from contextlib import contextmanager
import json
import os
import time

class GpuBudget:
    def __init__(self):
        if 'MORQEX_GPU_POLICY' not in os.environ:raise RuntimeError('Use scripts/run_local.py --gpu for an authorized monitored GPU job')
        import cupy as cp
        self.cp=cp;self.policy=json.loads(os.environ['MORQEX_GPU_POLICY'])
        cp.get_default_memory_pool().set_limit(size=self.policy['pool_memory_mib']*2**20)
        cp.fft.config.get_plan_cache().set_size(0)
        # Separable transforms avoid oversized/driver-dependent ND plans for
        # odd tomography dimensions. They evaluate the same discrete FFT.
        cp.fft.config.enable_nd_planning=False
        # Default CUDA priority is 0 (normal/least on this device); do not use
        # version-specific CuPy wrappers or elevate compute priority.
        # CuPy FFT/reduction temporaries and raw kernels share one ordered
        # stream. Mixing a nonblocking raw stream with allocator temporaries
        # can let a temporary be reused before the raw kernel has consumed it.
        self.stream=cp.cuda.Stream.null
        self._dot=cp.ReductionKernel('T x, U y','float64 out','(double)x * (double)y','a+b','out=a','0','morqex_fp64_dot_mixed')
        self.gpu_seconds=0.;self.sleep_seconds=0.;self.batches=0
    @contextmanager
    def batch(self):
        cp=self.cp;begin=cp.cuda.Event();end=cp.cuda.Event()
        with self.stream:
            begin.record()
            try:yield
            finally:
                end.record();end.synchronize();elapsed=cp.cuda.get_elapsed_time(begin,end)/1000.
                pause=elapsed*(1/self.policy['duty_cycle']-1)
                self.gpu_seconds+=elapsed;self.sleep_seconds+=pause;self.batches+=1
        # Short sleeps remain interruptible by the launcher and host work.
        while pause>0:part=min(pause,.25);time.sleep(part);pause-=part
    def metadata(self):return {'policy':self.policy,'gpu_event_seconds':self.gpu_seconds,'cooperative_sleep_seconds':self.sleep_seconds,'batches':self.batches,'precision':'FP64 physical stencil and CG vectors; FP32 selected preconditioner'}
    def dot(self,a,b):return float(self._dot(a,b))
