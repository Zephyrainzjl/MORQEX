"""Reproducible, explicit validation ranking; selection does not certify superiority."""
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import numpy as np
from morqex.microstructure import Volume
from morqex.microstructure.analysis import describe,two_point

@dataclass(frozen=True)
class Candidate:
    name:str
    checkpoint:str
    method:str='diffusion'
    steps:int=50
    guidance:float=1.5
    exact_fractions:bool=False
    coating:bool=False
    composition_guidance:bool=False

def rve_references(ct,size=64,partition='validation',count=4,seed=42,observation_planes=None):
    """Disjoint z halves for validation/test; same-size statistical comparison."""
    a=ct.data
    if min(a.shape)<size or a.shape[0]<2*size:raise ValueError('CT must fit disjoint RVE splits')
    half=a.shape[0]//2
    lo,hi=(0,half) if partition=='validation' else (half,a.shape[0])
    rng=np.random.default_rng(seed)
    cubes=[]
    for _ in range(count):
        origins=[]
        for axis,(lower,upper) in enumerate(((lo,hi),(0,a.shape[1]),(0,a.shape[2]))):
            valid=[i for i in range(lower,upper-size+1)
                   if not any(i<=p<i+size for p in (observation_planes or {}).get(axis,()))]
            if not valid:raise ValueError('No RVE origin avoids the observed planes')
            origins.append(int(rng.choice(valid)))
        z,y,x=origins
        cubes.append(Volume(a[z:z+size,y:y+size,x:x+size],ct.spacing))
    return cubes

def descriptors(volume):
    d=describe(volume)
    return {str(p):{'fraction':v['fraction'],'largest':v['largest_component_fraction'],
        'spans':v['spans_zyx'],'s2':[two_point(volume,p,a).tolist() for a in range(3)]} for p,v in d.items()}

def error(reference,generated):
    errors={}
    for phase in reference:
        r,g=reference[phase],generated.get(phase,{'fraction':0.,'largest':0.,'spans':[False]*3,'s2':[[0.]*len(x) for x in reference[phase]['s2']]})
        errors[phase]={
            'fraction':abs(r['fraction']-g['fraction']),
            's2':float(np.sqrt(np.mean((np.asarray(r['s2'])-np.asarray(g['s2']))**2))),
            'largest':abs(r['largest']-g['largest']),
            'spans':float(np.mean(abs(np.asarray(r['spans'],float)-np.asarray(g['spans'],float))))}
    return errors

def benchmark(candidates,ct,output,seeds=(0,1,2),partition='validation',size=64,baseline=None,observation_planes=None):
    """Preset score weights; reports each metric and avoids hiding failures in score."""
    from .v1 import sample
    from .hybrid import sample_student
    from .constraints import surface_coating
    import tifffile
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    refs=[descriptors(v) for v in rve_references(ct,size,partition,observation_planes=observation_planes)]
    reference={p:{k:np.mean([r[p][k] for r in refs],axis=0).tolist()
                  for k in ('fraction','largest','spans','s2')} for p in refs[0]}
    records=[]
    for candidate in candidates:
        samples=[]
        for seed in seeds:
            if candidate.method=='diffusion':
                v=sample(candidate.checkpoint,steps=candidate.steps,guidance=candidate.guidance,
                         seed=seed,size=size,exact_fractions=candidate.exact_fractions,
                         composition_guidance=candidate.composition_guidance)
            elif candidate.method=='hybrid':
                v=sample_student(candidate.checkpoint,seed=seed,size=size,
                                 exact_fractions=candidate.exact_fractions,
                                 composition_guidance=candidate.composition_guidance)
            elif candidate.method=='volume_diffusion':
                from .spatial import draw_volume
                v=draw_volume(candidate.checkpoint,seed=seed,size=size)
            elif candidate.method=='volume_diffusion_clean':
                from .spatial import draw_volume
                from morqex.microstructure.image import clean
                v=clean(draw_volume(candidate.checkpoint,seed=seed,size=size),minimum_size=8)
            else:raise ValueError(candidate.method)
            if candidate.coating:
                # Phase meanings are explicit for the selected NMC dataset.
                target=v.metadata.get('requested_fractions') or v.metadata.get('training_fractions')
                if target is None:raise ValueError('A coating prior needs recorded training phase fractions')
                v=surface_coating(v,substrate=128,coating=255,matrix=0,
                                  coating_fraction=target[2],seed=seed)
            v.save(output/f'{candidate.name}_seed{seed}.npz')
            desc=descriptors(v)
            errors=[error(reference,desc)]
            per_metric={k:float(np.mean([e[p][k] for e in errors for p in e]))
                        for k in ('fraction','s2','largest','spans')}
            samples.append({'seed':seed,'metrics':per_metric,'descriptors':desc})
        metrics={k:float(np.mean([s['metrics'][k] for s in samples])) for k in ('fraction','s2','largest','spans')}
        score=metrics['fraction']/.03+metrics['s2']/.03+metrics['largest']/.15+metrics['spans']/.33
        records.append({'candidate':asdict(candidate),'score':score,'metrics':metrics,'samples':samples})
        (output/'partial.json').write_text(json.dumps(records,indent=2))
    records.sort(key=lambda r:r['score'])
    report={'partition':partition,'weights':{'fraction':1/.03,'s2':1/.03,'largest':1/.15,'spans':1/.33},
            'interpretation':'Best among measured candidates under stated score; no claim of universal superiority',
            'ranking':records,'reference_mean':reference,'excluded_observation_planes':observation_planes,
            'reference_rves':refs,'reference_protocol':'mean descriptors of equal-size RVEs; per-RVE variability reported'}
    if baseline:
        baselines=rve_references(baseline,size,'validation') if min(baseline.data.shape)>=2*size else [baseline]
        ds=[descriptors(v) for v in baselines]
        es=[error(reference,d) for d in ds]
        report['baseline_metrics']={k:float(np.mean([e[p][k] for e in es for p in e])) for k in ('fraction','s2','largest','spans')}
    (output/'ranking.json').write_text(json.dumps(report,indent=2))
    return report
