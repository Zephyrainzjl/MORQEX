import gc,json,os,time
from pathlib import Path
import numpy as np
from morqex.nodewise.cache import NodeLocalization
from morqex.nodewise.localization import BatchLocalization
ws=Path(os.environ['MORQEX_WORKSPACE']);r=json.loads((ws/'results/sphere32_li/short/result.json').read_text());f=r['segments'][-1]['fields']['positive'];states=[]
def last(a,i):
 a=np.asarray(a);return float(a[i,-1] if a.ndim==2 else a[-1])
local=NodeLocalization(r['paths']['positive'][0]);refT=local.kernel.operator.metadata()['reference_temperature_K']
raw_damage=[last(f['damage'],i) for i in range(6)]
assert r['configuration']['degradation'].startswith('disabled') and max(map(abs,raw_damage))<1e-12
# This benchmark uses the analytic NoDamage state d=0. IDAS roundoff of order
# 1e-23 must not become a physically negative contact-damage input.
for i in range(6):states.append((0,np.asarray(f['concentration'])[i,:,-1],last(f['strain'],i),last(f['temperature'],i)-refT,0.))
start=time.perf_counter();cpu=BatchLocalization([local],backend='cpu',parallel_nodes=6);setup_cpu=time.perf_counter()-start
truth=cpu.recover(states);start=time.perf_counter()
for _ in range(5):cpu.recover(states)
cpu_s=(time.perf_counter()-start)/5;del cpu;gc.collect()
start=time.perf_counter();gpu=BatchLocalization([local],backend='cuda',parallel_nodes=6);setup_gpu=time.perf_counter()-start
out=gpu.recover(states);errors={k:float(np.linalg.norm(out[k]-truth[k])/max(np.linalg.norm(truth[k]),1e-300)) for k in truth};assert max(errors.values())<1e-11
start=time.perf_counter()
for _ in range(5):gpu.recover(states)
gpu_s=(time.perf_counter()-start)/5
result={'nodes':6,'unique_geometries':1,'shape':[32,32,32],'concentration_states_per_node':len(states[0][1]),'errors':errors,'raw_NoDamage_roundoff':raw_damage,
 'cpu_wall_s':cpu_s,'cuda_wall_s_including_transfer_and_duty':gpu_s,'online_speedup':cpu_s/gpu_s,'setup_cpu_s':setup_cpu,'setup_cuda_s':setup_gpu,
 'metadata':gpu.metadata(),'scope':'full FP64 voxel stress/strain/displacement recovery for six independent physical states sharing one geometry; DFN DAE remains on CPU'}
(ws/'reports/sphere32_cuda.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
