"""Repeated same-input / same-output full-spectrum and compressed localization."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import interp1d
from morqex.electrochemistry.aging_cache import AgingLocalization,load_cohesive_certificate
from morqex.cohesive_reduction import cohesive_certificate

WS=Path(os.environ['MORQEX_WORKSPACE'])

def timed(function,repeats=7):
    function();values=[]
    for _ in range(repeats):
        start=time.perf_counter();result=function();values.append(time.perf_counter()-start)
        assert all(np.isfinite(v).all() for v in result.values())
    return {'seconds':values,'median_seconds':float(np.median(values)),'minimum_seconds':float(np.min(values))}

def main():
    source=WS/'results/aging/closure_ct6_compressed/result.json';case=json.loads(source.read_text());result={}
    for d in ['negative','positive']:
        reduced=AgingLocalization(case['microstructure'][d]['kernel_path']);raw_path=WS/'operators/aging/closure_ct6_raw'/d
        exact=AgingLocalization(raw_path);assert exact.kernel.operator.metadata()['fingerprint']==reduced.kernel.operator.metadata()['fingerprint']
        t=np.linspace(0,case['curves']['time_s'][-1],16);m=case['microstructure'][d];source_t=case['curves']['time_s']
        c=interp1d(source_t,np.asarray(m['concentration'])[1],axis=0)(t)
        E=np.interp(t,source_t,np.asarray(m['strain'])[1]);theta=np.interp(t,source_t,case['curves']['temperature_K'])-reduced.kernel.operator.metadata()['reference_temperature_K']
        damage=np.linspace(0,.85,len(t))
        small=lambda:reduced.recover(c,E,theta,damage);full=lambda:exact.recover(c,E,theta,damage)
        a=small();b=full();error=float(np.linalg.norm(a['stress']-b['stress'])/max(1e-30,np.linalg.norm(b['stress'])))
        assert error<1e-4
        data=load_cohesive_certificate(case['microstructure'][d]['kernel_path']);bounds=[]
        for j in range(len(t)):
            q=np.r_[E[j],c[j]-reduced.kernel.operator.reference_concentration,theta[j]]
            bound=cohesive_certificate(reduced.kernel.cohesive,data,q,float(damage[j]))
            observed=float(np.max(np.linalg.norm(a['stress'][j]-b['stress'][j],axis=-1)))
            assert observed<=bound['stress_point_upper_Pa']+1e-3;bounds.append({'observed_Pa':observed,'bound_Pa':bound['stress_point_upper_Pa']})
        raw=timed(full);compressed=timed(small)
        result[d]={'raw':raw,'compressed':compressed,'median_speedup':raw['median_seconds']/compressed['median_seconds'],
            'stress_relative_L2':error,'raw_modes':len(exact.kernel.cohesive.eigenvalues),'compressed_modes':len(reduced.kernel.cohesive.eigenvalues),
            'output_shapes':{k:list(v.shape) for k,v in a.items()},'point_bounds':bounds,
            'kernel_manifest_sha256':{'raw':hashlib.sha256((raw_path/'manifest.json').read_bytes()).hexdigest(),'compressed':hashlib.sha256((Path(m['kernel_path'])/'manifest.json').read_bytes()).hexdigest()}}
    output={'schema':'morqex-localization-benchmark-1','queries':16,'warm_repeats':7,'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_result_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope':'same represented concentrations/strain/temperature/damage and all field outputs; warm cache; excludes offline, disk I/O and coupled evolution',
        'electrodes':result}
    (WS/'reports/crossscale_kernel_benchmark.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))

if __name__=='__main__':main()
