"""Isolated optional CGAL call. Only complete native buffers are published."""
import json,sys,time
from pathlib import Path
import h5py
from .implicit import _native


def main(source,target):
    start=time.perf_counter()
    with h5py.File(source) as f:arrays=[f[name][:] for name in ('scores','sizing','spacing','config','limits','seeds')]
    def event(code,np_,nt):
        print(json.dumps({'native_stage':code,'nodes':np_,'cells':nt,'elapsed_seconds':time.perf_counter()-start}),flush=True)
    result=_native(*arrays,None,callback=event);p,c,r,components=result
    path=Path(target);temporary=path.with_suffix('.pending')
    with h5py.File(temporary,'w') as f:
        for name,a in [('points',p),('cells',c),('labels',r),('components',components)]:f.create_dataset(name,data=a)
        f.attrs['complete']=True
        f.attrs['native_diagnostics']=json.dumps(result.diagnostics)
    temporary.replace(path)


if __name__=='__main__':main(*sys.argv[1:])
