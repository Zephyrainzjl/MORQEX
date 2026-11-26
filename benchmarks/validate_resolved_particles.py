from pathlib import Path
import argparse,os
import json,time
import numpy as np,h5py
from scipy import ndimage
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.geometry import MicrostructureData
from morqex.top_damage.polycrystal import PolycrystalData
from morqex.top_damage.morphology import measure_geometry
from morqex.top_damage.resolved_echem import ResolvedElectrochemistry
parser=argparse.ArgumentParser();parser.add_argument('--output',default='resolved_particles_reference');parser.add_argument('--dt',type=float,default=1.);parser.add_argument('--stop',type=float,default=2.);parser.add_argument('--geometry',default='dense_large_particles_32_g4');parser.add_argument('--linear-iterations',type=int,default=30);parser.add_argument('--linear-budget',type=float,default=0.);parser.add_argument('--warm-start',action='store_true');args=parser.parse_args()
ws=Workspace(os.environ['MORQEX_WORKSPACE']);backend=ResolvedElectrochemistry(ws)
base=PolycrystalData.load_hdf5(ws.path('results/top_damage/microstructures',args.geometry,'initial.h5'))
if base.structure.labels.shape!=(32,32,32):raise ValueError('This fixed comparison fixture requires a 32-cubed parent; use the general Volume API for other sizes')
if not np.isfinite([args.dt,args.stop]).all() or args.dt<=0 or args.stop<=0 or not np.isclose(args.stop/args.dt,round(args.stop/args.dt)):raise ValueError('Positive whole-step reference horizon required')
if not 1<=args.linear_iterations<=200:raise ValueError('Bounded linear iteration budget required')
if not np.isfinite(args.linear_budget) or not 0<=args.linear_budget<=.25:raise ValueError('Linear residual fraction must be in [0,.25]')
labels=np.where(base.structure.labels==1,1,0).astype(np.uint8)
z,y,x=np.indices(labels.shape);cbd=np.zeros(labels.shape,bool);cbd[:,:,:2]=True
# Explicit collector-connected bridges, followed by a correlated partial CBD
# coating. This is a new synthetic initial morphology, not damage-created voids.
for particle in np.unique(base.particle_ids[base.particle_ids>0]):
 centre=np.argwhere(base.particle_ids==particle).mean(axis=0)
 cbd|=((z-centre[0])**2+(y-centre[1])**2<2.2**2)&(x<=centre[2])
cbd &= labels==0
target=round(.23*labels.size)
if cbd.sum()>target:raise ValueError('CBD bridge volume exceeds requested volume')
distance=ndimage.distance_transform_edt(labels!=1)
noise=ndimage.gaussian_filter(np.random.default_rng(1204).normal(size=labels.shape),1.2)
score=distance+4*noise/max(noise.std(),1e-30)
available=np.flatnonzero((labels==0)&~cbd)
cbd.ravel()[available[np.argsort(score.ravel()[available])[:target-int(cbd.sum())]]]=True
labels[cbd]=2
# A separate connected reference fixture: remove floating CBD agglomerates and
# fill closed solvent cavities at INITIAL generation only. Preserve every active
# voxel, record the resulting measured fractions, and do not use this as damage.
components,_=ndimage.label(labels>0);anchor=np.unique(components[:,:,0]);anchor=anchor[anchor>0]
floating=(labels>0)&~np.isin(components,anchor)
if np.any(floating&(labels==1)):raise ValueError('Active particle lost prescribed bridge')
labels[floating]=0
components,_=ndimage.label(labels==0);access=np.unique(components[:,:,-1]);access=access[access>0]
closed=(labels==0)&~np.isin(components,access);labels[closed]=2
if ndimage.label(labels>0)[1]!=1 or ndimage.label(labels==0)[1]!=1:raise ValueError('Connected initial reference required')
structure=MicrostructureData(labels,base.structure.spacing_zyx_m,base.structure.origin_zyx_m,base.structure.phase_kinds,
 base.structure.provenance|{'CBD_generation':'explicit x bridges and seeded correlated partial coating','CBD_seed':1204,'initial_floating_CBD_relocated_to_pore':int(floating.sum()),'initial_closed_pore_reassigned_CBD':int(closed.sum()),'parent_geometry':base.fingerprint})
data=PolycrystalData(structure,base.particle_ids,base.grain_ids,base.grain_rotations,base.grain_particles,base.metadata)
root=ws.path('results/top_damage',args.output)
if root.exists():raise ValueError('Choose a new result directory; never overwrite existing evidence')
root.mkdir(parents=True)
data.save_hdf5(root/'initial.h5');(root/'geometry.json').write_text(json.dumps(measure_geometry(data),indent=2))
full=np.concatenate([np.choose(labels,[6,5,7]),np.full((32,32,4),4),np.full((32,32,8),1)],axis=2).astype(np.uint8)
v=Volume(full,spacing=structure.spacing_zyx_m)
p={'max_step':int(round(args.stop/args.dt)),'stop_time':args.stop,'prob.Crate':-.5,'prob.halfcell_case':1,'prob.soc_anode_t0':.5,'prob.soc_cathode_t0':.5,
 'prob.Temperature':298.15,'prob.Sp_corr_anode':1.,'prob.Sp_corr_cathode':1.,'prob.Cphi_electrolyte_coupling':1,
 'echemamr.dtmin':args.dt,'echemamr.dtmax':args.dt,'echemamr.bv_nonlinear_iters':100,
 'echemamr.bv_nonlinear_reltol':1e-8,'echemamr.bv_nonlinear_abstol':1e-12,'echemamr.linsolve_reltol':1e-10,'echemamr.linsolve_maxiter':args.linear_iterations,
 'echemamr.linsolve_abstol':1e-12,'echemamr.linsolve_max_coarsening_level':0,'echemamr.use_hypre':1,'echemamr.hypre.bamg_coarsen_type':6,'echemamr.hypre.bamg_interp_type':0,'echemamr.hypre.bamg_relax_type':6,'echemamr.hypre.bamg_strong_threshold':.25,'echemamr.hypre.verbose':0,'echemamr.hypre.bamg_verbose':0,'echemamr.linsolve_bottom_maxiter':1000,'amr.plot_int':1,'amr.chk_int':1,'amr.line_plot_int':-1}
p['echemamr.morqex_linear_budget_fraction']=args.linear_budget
p['echemamr.potential_initial_guess']=int(args.warm_start)
tick=time.perf_counter();out=root/'discharge'
curve=backend.run(v,out,{i:i for i in [1,4,5,6,7]},p,timeout=240)
variables=['Concentration','Potential','Cathode_active_material','Cathode_CBD','Cathode_electrolyte','Anode_active_material','Separator','Nanoporosity']
for i in range(int(round(args.stop/args.dt))+1):backend.export_fields(out/f'plt{i:05d}',root/f'fields_{i:05d}.h5',variables)
result={'time_s':curve.time_s.tolist(),'voltage_V':curve.voltage_V.tolist(),'diagnostics':curve.diagnostics,'seconds':time.perf_counter()-tick,
 'scope':'Full 32^3 cathode with explicit partial CBD, NMC532 upstream constitutive card; short 3D electrochemical qualification, not lifetime or phase-field closure'}
with h5py.File(root/'fields_00000.h5') as f:
 c0=f['Concentration'][:];masks={name:f[name][:].astype(bool) for name in variables if name not in ('Concentration','Potential','Nanoporosity')};dv=np.prod(f.attrs['spacing_zyx_m'])
with h5py.File(root/f'fields_{int(round(args.stop/args.dt)):05d}.h5') as f:c=f['Concentration'][:]
charge=np.trapezoid(curve.current_A,curve.time_s)/96485.33212
checks={}
for name,sign in [('Cathode_active_material',-1),('Anode_active_material',1)]:
 mask=masks[name];change=float((c[mask]-c0[mask]).sum()*dv)
 checks[name+'_throughput_relative_error']=abs(change-sign*charge)/max(abs(charge),1e-30)
liquid=masks['Cathode_electrolyte']|masks['Separator']
checks['electrolyte_relative_error']=abs(float((c[liquid]-c0[liquid]).sum()/c0[liquid].sum()))
checks['dry_CBD_max_concentration']=float(np.max(abs(c[masks['Cathode_CBD']])))
result['inventory_checks']=checks
result['numerical_gates_passed']=all(v<1e-5 for k,v in checks.items() if 'throughput' in k) and checks['electrolyte_relative_error']<1e-8 and checks['dry_CBD_max_concentration']<1e-10
result['material_or_lifetime_qualified']=False
(root/'probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['numerical_gates_passed']
