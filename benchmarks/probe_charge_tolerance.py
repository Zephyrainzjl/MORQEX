"""Replay one rejected electrochemical step from an immutable micro checkpoint."""
import argparse,json,os
from pathlib import Path
import h5py,numpy as np
from morqex.workspace import Workspace
from morqex.top_damage.polycrystal import PolycrystalData
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.top_damage.region_port import RegionPort
from morqex.top_damage.network_dfn import NetworkEpochDFN
from morqex.top_damage.epoch_dfn import DFNFeedback
from morqex.electrochemistry.open_particle import TransportLaw
p=argparse.ArgumentParser();p.add_argument('--root',type=float,required=True);p.add_argument('--rtol',type=float,default=1e-8);p.add_argument('--atol',type=float,default=1e-9)
a=p.parse_args();ws=Workspace(os.environ['MORQEX_WORKSPACE'])
directory=ws.path('results/top_damage/network_cycles/dual_charge_gate_probe_v1')
meta=json.loads((directory/'manifest.json').read_text());cfg=meta['config']
data=PolycrystalData.load_hdf5(directory/'initial.h5');labels=data.structure.labels
fv=VoxelDiffusion(Volume(labels,spacing=data.structure.spacing_zyx_m),active_phases=[1],diffusivity=1.,particle_ids=data.particle_ids,reactive_phases=[0,2])
with h5py.File(directory/'mesh.h5') as f:partition=f['partition'][:]
port=RegionPort.from_diffusion(fv,partition,fv.reactive,.5*meta['maximum_concentration_mol_m3'])
runtime=NetworkEpochDFN(port,host=cfg['host'],porosity=float(np.mean(labels==0)+cfg['cbd_microporosity']*np.mean(labels==2)),
    nx=cfg['nodes'],local_mobility=cfg['local_mobility'],transport_law=TransportLaw(gradient_coefficient_J_m5_mol2=cfg['gradient_coefficient_J_m5_mol2']))
s=runtime.simulation.solver;s.root_tol=a.root;s.root_method='casadi';s.rtol=a.rtol;s.atol=a.atol
captured=[];original=runtime.simulation.step
def capture(*args,**kwargs):
    sol=original(*args,**kwargs);captured.append(sol);return sol
runtime.simulation.step=capture
with h5py.File(directory/'step_000006.h5') as f:
    r=json.loads(f.attrs['record']);runtime.restore_state(r['time_s'],f['dfn_state'][:],json.loads(f.attrs['inputs']))
    sigma=f['next/electronic_S_m'][:];eta=f['next/ionic_efficiency'][:]
    try:
        result=runtime.advance_network(300.,-.5*runtime.capacity_Ah,DFNFeedback(float(sigma.mean()),float(eta.mean())),f['next/mu_J_mol'][:],
            reaction_fractions=f['next/reaction_fractions'][:],edge_factors=f['next/edge_factors'][:],
            electronic_conductivity_S_m=sigma,electrolyte_efficiency=eta)
    except RuntimeError as exc:result={'error':str(exc)}
sol=captured[-1];p=runtime.parameters
V=float(p['Electrode width [m]'])*float(p['Electrode height [m]'])*float(p['Positive electrode thickness [m]'])
I=np.asarray(sol['Current [A]'].entries);j=np.asarray(sol['Positive electrode volumetric interfacial current density [A.m-3]'].entries)
result.update(root_tol=a.root,rtol=a.rtol,atol=a.atol,time_s=sol.t.tolist(),charge_defect_trace=((V*j.mean(axis=0)+I)/np.max(abs(I))).tolist())
target=ws.path('reports',f'charge_probe_root{a.root}_rtol{a.rtol}.json');target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
