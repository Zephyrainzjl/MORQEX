"""Equal dry interfacial crack area, different electronic connectivity.

A prescribed-geometry counterexample, not phase-field-generated crack growth.
One cut breaks a particle's sole CBD contact; another cuts one redundant CBD
edge. Both remove one physical face of the same area, without removing solid
volume or inventing a wetted crack. Full micro voxel inventories feed a DFN.
"""
import argparse,json,os,time,re
from pathlib import Path
import h5py,numpy as np
from morqex.workspace import Workspace
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.top_damage.region_port import RegionPort
from morqex.top_damage.dual_network import DualNetworkClosure
from morqex.top_damage.network_dfn import NetworkEpochDFN
from morqex.top_damage.epoch_dfn import DFNFeedback
from morqex.nodewise.halfcell import lithium_halfcell_parameters
from morqex.electrochemistry.open_particle import TransportLaw


def geometry():
    labels=np.zeros((6,6,10),np.uint8);particles=np.zeros_like(labels,dtype=np.int32)
    labels[0,:2,:]=2
    for particle,x in enumerate((1,6),1):
        labels[2:4,2:4,x:x+2]=1;particles[2:4,2:4,x:x+2]=particle
        labels[0:2,2,x]=2
    return labels,particles


def experiment(mode,rate,ws,tag):
    start=time.perf_counter();labels,particles=geometry();h=1e-6
    volume=Volume(labels,spacing=(h,)*3)
    fv=VoxelDiffusion(volume,active_phases=[1],diffusivity=1.,reactive_phases=[0],particle_ids=particles)
    p,_=lithium_halfcell_parameters(host='NMC811',active_fraction=float(np.mean(labels==1)))
    port=RegionPort.from_diffusion(fv,np.arange(len(fv.voxel_ids)),fv.reactive,.5*float(p['Maximum concentration in positive electrode [mol.m-3]']))
    closure=DualNetworkClosure(fv,port,update_tolerance=0.)
    electronic=np.where(labels==1,10.,np.where(labels==2,100.,0.));ionic=(labels==0).astype(float)
    face=np.ones((3,)+labels.shape)
    intact=closure.evaluate(electronic,ionic,electronic_face_factors=face)
    if mode=='isolate':face[0,1,2,1]=0.
    elif mode=='redundant':face[2,0,0,4]=0.
    elif mode!='intact':raise ValueError('Unknown physical cut')
    changed=closure.evaluate(electronic,ionic,electronic_face_factors=face)
    if changed.diagnostics['unresolved_split_regions']:raise RuntimeError('Full voxel regions cannot have unresolved internal splits')
    expected=8 if mode=='isolate' else 16
    assert int(np.count_nonzero(changed.electron_accessible&(labels==1)))==expected
    assert changed.electronic_conductivity_S_m>0 and changed.electrolyte_efficiency>0
    assert changed.diagnostics['material_volume_m3']==intact.diagnostics['material_volume_m3']
    out=ws.path('results/top_damage/topology_counterexample',tag,f'{mode}_{rate:g}C')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Version the experiment before overwriting accepted evidence')
    runtime=NetworkEpochDFN(port,host='NMC811',porosity=float(np.mean(labels==0)),nx=3,local_mobility=True,
        transport_law=TransportLaw(),diffusivity_interpretation='chemical_diffusivity')
    capacities=[];curves=[];records=[];snapshots=[];phase='conditioning_charge';cycle=0;phase_start=0.;time0=0.
    changed_inventory=None;max_trapped_drift=0.
    for step in range(1,301):
        net=intact if cycle==0 else changed
        mu=np.zeros((3,port.dimension));area=np.broadcast_to(net.reaction_fractions,mu.shape)
        isolated=~net.electron_accessible.ravel()[port.voxel_ids]
        if mode=='isolate' and cycle>0 and changed_inventory is None:
            changed_inventory=np.sum(runtime.concentrations()[:,isolated]*port.weights[isolated],axis=1)
        current=(-1 if phase!='discharge' else 1)*rate*runtime.capacity_Ah
        record=runtime.advance_network(3600/rate/24,current,DFNFeedback(net.electronic_conductivity_S_m,net.electrolyte_efficiency),mu,
            reaction_fractions=area,edge_factors=np.broadcast_to(net.diffusion_edge_factors,(3,len(runtime.edges))))
        c=runtime.concentrations()
        if mode=='isolate' and cycle>0:
            # Internal solid diffusion may redistribute trapped Li; its sum is
            # conserved. Do not demand pointwise frozen concentrations.
            trapped=np.sum(c[:,isolated]*port.weights[isolated],axis=1)
            max_trapped_drift=max(max_trapped_drift,float(np.max(abs(trapped-changed_inventory)))/max(float(np.max(abs(changed_inventory))),1e-30))
        sol=runtime.last
        curves.extend(np.column_stack([np.full(len(sol.t),cycle+1),sol.t,sol['Voltage [V]'].entries,
            sol['Current [A]'].entries,sol['Discharge capacity [A.h]'].entries]).tolist())
        record.update(step=step,phase=phase,cycle=cycle+1,accessible_active_voxels=int(np.count_nonzero(net.electron_accessible&(labels==1))))
        records.append(record)
        if record['termination']!='final time':
            required='minimum' if phase=='discharge' else 'maximum'
            if required not in record['termination'].lower():raise RuntimeError('Unexpected non-voltage termination')
            capacity=abs(record['net_capacity_Ah']-phase_start);phase_start=record['net_capacity_Ah']
            capacities.append({'phase':phase,'cycle':cycle+1,'capacity_Ah':capacity})
            with h5py.File(out/f'phase_{len(capacities):03d}.h5','w') as f:
                f.attrs.update(record=json.dumps(record),scope='Prescribed dry contact cuts; no phase-field evolution or new pore volume')
                for key,value in {'phase_labels':labels,'particle_ids':particles,'electronic_face_transmission':np.ones_like(face) if cycle==0 else face,
                    'electron_accessible':net.electron_accessible,'ion_accessible':net.ion_accessible,
                    'concentration_regions_mol_m3':c,'voxel_concentration_mol_m3':np.array([port.field(a,inactive=0.) for a in c]),
                    'dfn_state':np.asarray(sol.last_state.y).ravel()}.items():f.create_dataset(key,data=value)
            if phase=='conditioning_charge':phase='discharge'
            elif phase=='discharge':phase='charge'
            else:
                cycle+=1;phase='discharge'
                if cycle==2:break
    else:raise RuntimeError('Protocol incomplete at safety bound')
    if max_trapped_drift>1e-8:raise RuntimeError('Isolated material lost lithium')
    discharge=[a['capacity_Ah'] for a in capacities if a['phase']=='discharge']
    result={'mode':mode,'C_rate':rate,'complete_cycles':cycle,'regions':port.dimension,'phase_capacities':capacities,
        'discharge_capacity_Ah':discharge,'second_to_first_capacity_ratio':discharge[1]/discharge[0],
        'prescribed_dry_crack_area_m2':0. if mode=='intact' else h*h,
        'material_volume_m3':changed.diagnostics['material_volume_m3'],
        'accessible_active_voxels_before':16,'accessible_active_voxels_after':expected,
        'electronic_conductivity_before_S_m':intact.electronic_conductivity_S_m,
        'electronic_conductivity_after_S_m':changed.electronic_conductivity_S_m,
        'ionic_efficiency_before':intact.electrolyte_efficiency,'ionic_efficiency_after':changed.electrolyte_efficiency,
        'max_inventory_relative_defect':max(r['inventory_relative_defect'] for r in records),
        'max_charge_relative_defect':max(r['charge_relative_defect'] for r in records),
        'trapped_inventory_relative_drift':max_trapped_drift,'seconds':time.perf_counter()-start,
        'scope':'Analytical topology counterexample on a sparse numerical structure. Prescribed dry interfacial failure, not fatigue-generated crack evolution or calibrated material life.'}
    (out/'result.json').write_text(json.dumps(result,indent=2));(out/'records.json').write_text(json.dumps(records))
    np.savetxt(out/'curve.csv',curves,delimiter=',',header='cycle,time_s,voltage_V,current_A,net_capacity_Ah',comments='')
    print(json.dumps(result),flush=True);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rates',nargs='+',type=float,default=[.5,8.]);parser.add_argument('--tag',default='v1');args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',args.tag):raise ValueError('Safe experiment tag required')
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);results=[]
    for rate in args.rates:
        if not 0<rate<=8:raise ValueError('Positive rate <= 8C required')
        for mode in ('intact','redundant','isolate'):results.append(experiment(mode,rate,ws,args.tag))
    ws.path('reports','topology_counterexample_'+args.tag+'.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
