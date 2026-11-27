"""Bounded build-stage memory/time audit with periodic Python stack samples."""
import argparse,faulthandler,json,os,resource,time,traceback
from pathlib import Path
import pybamm as pb
from morqex.workspace import Workspace
from morqex.top_damage.network_cycle import run_network_cycle


def instrument(cls,name):
    original=getattr(cls,name)
    def call(self,*args,**kwargs):
        start=time.perf_counter()
        print(json.dumps({'stage':cls.__name__+'.'+name,'event':'start','rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
        try:return original(self,*args,**kwargs)
        finally:print(json.dumps({'stage':cls.__name__+'.'+name,'event':'finish','seconds':time.perf_counter()-start,
            'rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
    setattr(cls,name,call)


def main():
    p=argparse.ArgumentParser();p.add_argument('config');args=p.parse_args()
    for cls,name in [(pb.BaseBatteryModel,'build_model'),(pb.ParameterValues,'process_model'),
                     (pb.Discretisation,'process_model'),(pb.Simulation,'build')]:instrument(cls,name)
    faulthandler.dump_traceback_later(15,repeat=True)
    try:run_network_cycle(Path(args.config),Workspace(os.environ['MORQEX_WORKSPACE']),steps=1)
    except Exception as exc:
        # Symbolic KeyError arguments can recursively print an enormous tree.
        # Keep traceback frames, but do not stringify arbitrary expression DAGs.
        print('EXCEPTION_TYPE: '+type(exc).__name__,flush=True)
        for arg in exc.args:print(('ARG: '+arg[:1200]) if isinstance(arg,str) else 'ARG_TYPE: '+type(arg).__name__,flush=True)
        print(''.join(traceback.format_list(traceback.extract_tb(exc.__traceback__))),flush=True)
        raise SystemExit(1) from None
    finally:faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
