"""Measured stage progress and explicitly approximate, locally calibrated ETAs."""
from contextlib import contextmanager
from dataclasses import dataclass,asdict
from pathlib import Path
import json,math,sys,threading,time


@dataclass(frozen=True)
class ProgressEvent:
    stage:str
    elapsed_seconds:float
    completed:float|None=None
    total:float|None=None
    eta_seconds:float|None=None
    estimated_duration_seconds:float|None=None
    estimate_source:str='unavailable'
    message:str=''
    finished:bool=False
    status:str='running'

    def to_dict(self):return asdict(self)


class ProgressBar:
    """Console callback; native calls have a spinner, not fabricated counts.

    ``json_lines=True`` is useful for batch jobs. ETA is never a guarantee.
    """
    def __init__(self,stream=None,*,json_lines=False,interval=.5):
        self.stream=stream or sys.stderr;self.json_lines=json_lines
        self.interval=interval;self.previous=0.

    def __call__(self,event):
        now=time.monotonic()
        if not event.finished and now-self.previous<self.interval:return
        self.previous=now
        if self.json_lines:text=json.dumps(event.to_dict(),ensure_ascii=False)
        else:
            fraction=None if event.total is None else min(1.,event.completed/max(event.total,1e-300))
            bar=('[%s%s] %5.1f%%'%('#'*int(20*fraction),'.'*(20-int(20*fraction)),100*fraction)
                 if fraction is not None else '[working / 计算中]')
            eta='unknown' if event.eta_seconds is None else f'~{event.eta_seconds:.1f}s'
            text=f'{event.stage} {bar} elapsed {event.elapsed_seconds:.1f}s, ETA {eta} ({event.estimate_source}) {event.message}'
        print(text,file=self.stream,flush=True)


class NotebookProgress:
    """Optional IPython progress display; works without widget extensions."""
    def __init__(self):self.handle=None
    def __call__(self,event):
        from IPython.display import HTML,display
        from html import escape
        value='' if event.total is None else f' value="{min(1.,event.completed/max(event.total,1e-300))}"'
        eta='unknown / 暂未知' if event.eta_seconds is None else f'≈{event.eta_seconds:.1f}s'
        content=HTML(f'<div><b>{escape(event.stage)}</b> <progress max="1"{value}></progress> '
                     f'{event.elapsed_seconds:.1f}s · ETA {eta} · {escape(event.message)}</div>')
        if self.handle is None:self.handle=display(content,display_id=True)
        else:self.handle.update(content)


class _Stage:
    def __init__(self,stage,callback,estimate,source):
        self.stage=stage;self.callback=callback;self.estimate=estimate;self.source=source
        self.start=time.perf_counter();self.done=None;self.total=None;self.message=''
        self.lock=threading.RLock();self.stop=threading.Event()
        self.failed=False
    def emit(self,finished=False):
        with self.lock:
            elapsed=time.perf_counter()-self.start;eta=None
            if self.done is not None and self.total is not None and self.done>0:
                eta=max(0.,elapsed*(self.total-self.done)/self.done);source='measured completed work; approximate'
            else:
                eta=None if self.estimate is None else max(0.,self.estimate-elapsed)
                source=self.source
                if eta==0 and not finished:eta=None;source+='; estimate exceeded'
            if finished:eta=0.
            if self.callback:self.callback(ProgressEvent(self.stage,elapsed,self.done,self.total,eta,self.estimate,source,self.message,finished,
                'failed' if self.failed else 'completed' if finished else 'running'))
    def update(self,completed,total, message=''):
        if not all(math.isfinite(x) for x in (completed,total)) or not 0<=completed<=total or total<=0:
            raise ValueError('Finite completed work within a positive total required')
        with self.lock:self.done=completed;self.total=total;self.message=message;self.emit()
    def heartbeat(self):
        while not self.stop.wait(1.):self.emit()


@contextmanager
def progress_stage(name,callback=None,*,estimated_seconds=None,estimate_source='uncalibrated estimate'):
    """Opaque native work reports elapsed time; known loops call update()."""
    if estimated_seconds is not None and (not math.isfinite(estimated_seconds) or estimated_seconds<=0):
        raise ValueError('Positive finite duration estimate required')
    stage=_Stage(name,callback,estimated_seconds,estimate_source);stage.emit()
    thread=threading.Thread(target=stage.heartbeat,daemon=True)
    if callback:thread.start()
    try:yield stage
    except BaseException:
        stage.failed=True;stage.message='failed; no result accepted';raise
    else:
        if stage.total is not None:stage.done=stage.total
        stage.message=stage.message or 'complete'
    finally:
        stage.stop.set()
        if callback:thread.join(timeout=2.)
        stage.emit(finished=True)


class TimingHistory:
    """External rolling measurements. No calibrated-data claim on a first run."""
    def __init__(self,path):
        from .voxel import external_path
        self.path=external_path(path)
    def estimate(self,stage,units,fallback_rate):
        records=json.loads(self.path.read_text()) if self.path.exists() else {}
        rates=records.get(stage,[])
        import statistics
        return max(.01,units/(statistics.median(rates) if rates else fallback_rate)),('local rolling throughput; approximate' if rates else 'uncalibrated throughput; approximate')
    def record(self,stage,units,seconds):
        if units<=0 or seconds<=0:return
        records=json.loads(self.path.read_text()) if self.path.exists() else {}
        records[stage]=(records.get(stage,[])+[float(units/seconds)])[-12:]
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.path.with_name(self.path.name+f'.{time.time_ns()}.pending');temporary.write_text(json.dumps(records,indent=2));temporary.replace(self.path)


class HTMLProgress:
    """Small auto-refreshing LOCAL HTML; never reloads a large geometry view.

    Call finish() after the complete user workflow. A failed stage stops the
    refresh automatically and is not rendered as successful completion.
    """
    def __init__(self,path,*,title='MORQEX geometry progress / 几何处理进度'):
        from .voxel import external_path
        self.path=external_path(path);self.title=title;self.events={};self.terminal=False;self.message=''
        self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock()
    def __call__(self,event):
        with self.lock:
            self.events[event.stage]=event
            if event.status=='failed':self.terminal=True
            self._write()
    def finish(self,*,message='Workflow finished / 流程结束'):
        with self.lock:self.terminal=True;self.message=message;self._write()
    def _write(self):
        from html import escape
        rows=[]
        for e in self.events.values():
            value='' if e.total is None else f' value="{min(1.,e.completed/max(e.total,1e-300))}"'
            eta='unknown / 暂未知' if e.eta_seconds is None else f'≈{e.eta_seconds:.1f}s'
            rows.append(f'<tr><td>{escape(e.stage)}</td><td><progress max="1"{value}></progress></td><td>{escape(e.status)}</td><td>{e.elapsed_seconds:.1f}s</td><td>{eta}</td><td>{escape(e.message)}</td></tr>')
        refresh='' if self.terminal else '<meta http-equiv="refresh" content="2">'
        doc=f'''<!doctype html><html lang="en"><meta charset="utf-8">{refresh}<title>{escape(self.title)}</title>
<style>body{{font:15px system-ui;background:#edf3f6;color:#173b4b;margin:32px}}main{{max-width:1400px;margin:auto;background:white;padding:24px;border-radius:14px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #dce7ec;text-align:left}}progress{{width:180px}}pre{{white-space:pre-wrap}}</style>
<main><h1>{escape(self.title)}</h1><p>{escape(self.message)}</p><p>ETA is approximate; unknown native work is indeterminate. / ETA为粗略估计，原生求解内部工作量未知时不伪造百分比。</p>
<table><tr><th>Stage / 阶段</th><th>Measured progress / 已完成工作</th><th>Status</th><th>Elapsed</th><th>ETA</th><th>Detail</th></tr>{''.join(rows)}</table>
<details><summary>Recorded events / 记录</summary><pre>{escape(json.dumps([e.to_dict() for e in self.events.values()],ensure_ascii=False,indent=2))}</pre></details></main></html>'''
        tmp=self.path.with_name(self.path.name+f'.{time.time_ns()}.pending');tmp.write_text(doc);tmp.replace(self.path)
