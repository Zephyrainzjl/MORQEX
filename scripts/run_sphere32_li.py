"""Six true 32-cubed cathode blocks coupled to a planar lithium counterelectrode."""
import os,argparse,json,time
from pathlib import Path
import numpy as np
import pybamm as pb
from morqex import Workspace
from morqex.nodewise.halfcell import HalfCellStack,lithium_halfcell_parameters
from morqex.nodewise.cache import load_node_kernel
from morqex.electrochemistry.aging_dfn import AgingMechanics,age
from morqex.electrochemistry.open_particle import build_open_dfn
from morqex.electrochemistry.microstructure_dfn import key
from morqex.electrochemistry.protocol import shared_control,add_protocol_event,solve_protocol
from morqex.electrochemistry.open_validation import open_observables
from morqex.degradation import DegradationOptions,NoDamage
from morqex.solvers.pybamm_backend import solver
class ModalMechanics(AgingMechanics):modal_variables=True
ap=argparse.ArgumentParser();ap.add_argument('--short',action='store_true');ap.add_argument('--cycles',type=int,default=2);ap.add_argument('--rate',type=float,default=.5);ap.add_argument('--solver',choices=['idaklu','casadi'],default='idaklu');ap.add_argument('--chunk-seconds',type=float,default=1800.);args=ap.parse_args()
ws=Workspace(os.environ['MORQEX_WORKSPACE']);record=json.loads(ws.path('results','sphere32_li','operator.json').read_text());kernel=load_node_kernel(record['path'])
p,audit=lithium_halfcell_parameters(active_fraction=kernel.operator.active_fraction)
stack=HalfCellStack([kernel]*6,p,constraint='fixed-thickness',metal_sei='none');stack.update_parameters(p)
config=DegradationOptions(negative=(),positive=(),damage=NoDamage());paths={'positive':[record['path']]*6}
out=ws.path('results','sphere32_li','short' if args.short else f'cycles_{args.rate:g}C');out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();run_id=str(time.time_ns());frames=[]
print('BUILDING six nodes with',kernel.operator.dimension,'independent concentration states each',flush=True)
model=build_open_dfn(stack,stack.kernels,config,parameters=p,control=shared_control,factories={'mechanics':ModalMechanics});add_protocol_event(model)
backend=solver(rtol=1e-7,atol=1e-8) if args.solver=='idaklu' else pb.CasadiSolver(mode='safe',rtol=1e-7,atol=1e-8,dt_max=600.)
sim=pb.Simulation(model,parameter_values=p,solver=backend,var_pts=stack.var_pts(model))
def collect(step,segment):
 collection_start=time.perf_counter()
 d='positive';op=stack.electrodes[d]
 fields={d:{'concentration':np.stack([step[key(d,k,'concentration [mol.m-3]')].entries for k in range(op.dimension)],axis=1).tolist(),
  'strain':step['MORQEX positive axial strain'].entries.tolist(),'damage':step[age(d,'damage')].entries.tolist(),
  'phi_s':step['Positive electrode potential [V]'].entries.tolist(),'phi_e':step['Positive electrolyte potential [V]'].entries.tolist(),
  'ce':step['Positive electrolyte concentration [mol.m-3]'].entries.tolist(),'temperature':step['Positive electrode temperature [K]'].entries.tolist(),'reservoirs':{}}}
 frames.append({'segment':segment,'time_s':step.t.tolist(),'voltage_V':step['Voltage [V]'].entries.tolist(),'current_A':step['Current [A]'].entries.tolist(),
  'temperature_K':step['Volume-averaged cell temperature [K]'].entries.tolist(),'stress_Pa':step['MORQEX stack axial stress [Pa]'].entries.tolist(),
  'metal_inventory_mol':step['MORQEX Li metal inventory [mol]'].entries.tolist(),'fields':fields})
 tmp=out/'live.tmp';tmp.write_text(json.dumps({'status':'running','run_id':run_id,'paths':paths,'segments':frames}));tmp.replace(out/'live.json');print('FRAME',segment,'collection_s',time.perf_counter()-collection_start,'wall_s',time.perf_counter()-start,flush=True)
try:
 sol,protocol=solve_protocol(sim,kind='short' if args.short else 'cycles',cycles=args.cycles,current_A=args.rate*float(p['Nominal cell capacity [A.h]']),
  lower_V=float(p['Lower voltage cut-off [V]']),upper_V=float(p['Upper voltage cut-off [V]']),duration_s=120.,period_s=60.,rest_s=300.,
  cv_stop_A=float(p['Nominal cell capacity [A.h]'])/20,chunk_s=args.chunk_seconds,on_frame=collect)
 metrics=open_observables(sol,stack,p,config)['metrics']
 total=sol['Total lithium [mol]'].entries+sol['MORQEX Li metal inventory [mol]'].entries+sol['MORQEX metal SEI lithium loss [mol]'].entries
 metrics.update(relative_total_lithium_with_metal_drift=float(np.ptp(total)/total[0]),fixed_thickness_max_m=float(np.max(abs(sol['MORQEX sandwich thickness change [m]'].entries))),
  negative_particle_submodels=[name for name in model.submodels if 'negative' in name and 'particle' in name],minimum_metal_inventory_mol=float(np.min(sol['MORQEX Li metal inventory [mol]'].entries)))
 result={'run_id':run_id,'completed':True,'kind':'short' if args.short else 'cycles','counterelectrode':'planar lithium metal','thermal_mode':'lumped',
  'configuration':{'positive_nodes':6,'voxels_per_node':[32,32,32],'particles_per_node':record['particle_count'],'concentration_states_per_node':kernel.operator.dimension,
   'binding':'shared immutable geometry; independent states','degradation':'disabled for this intact reference qualification; evolving CBD contact requires cohesive backend'},
  'parameters':audit,'metrics':metrics,'protocol':protocol,'paths':paths,'segments':frames,'elapsed_s':time.perf_counter()-start}
except Exception as error:
 (out/'result.json').write_text(json.dumps({'completed':False,'error':repr(error),'paths':paths,'segments':frames}));raise
(out/'result.json').write_text(json.dumps(result));(out/'live.json').write_text(json.dumps(result));print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ['segments','parameters']}),flush=True)
