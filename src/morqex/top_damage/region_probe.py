"""Bounded six-node-compatible finite-mechanics/DFN feedback qualification.

This is a two-segment explicit-lag probe with a matched no-feedback control,
not a long-cycle topology prediction. Individual microstates can resume through
separate bounded calls. All arrays and reports remain in the external workspace.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,re,tomllib
import numpy as np
from morqex import Workspace


@dataclass(frozen=True)
class RegionProbeConfig:
    geometry: str = 'dense_large_particles_32_g4'
    output: str = 'region_finite_32_toml'
    host: str = 'NMC811'
    nodes: int = 6
    requested_regions: int = 64
    time_step_s: float = 30.
    C_rate: float = .5
    initial_stoichiometry: float = .5
    temperature_K: float = 298.15
    gradient_coefficient_J_m5_mol2: float = 1e-14
    cbd_microporosity: float = .47
    cbd_ionic_efficiency: float = .05
    electronic_conductivity_S_m: float = 10.
    active_young_Pa: float = 93e9
    cbd_young_Pa: float = 1e9
    poisson: float = .3
    active_Gc_J_m2: float = 10.
    cbd_Gc_J_m2: float = 10000.
    length_m: float = 1e-6
    precrack_radius_m: float = 1e-6
    chemical_strain_per_stoichiometry: tuple = (.012,.012,.040)
    stagger_tolerance: float = 1e-6

    def __post_init__(self):
        from morqex.topology import PorousCBD
        from morqex.electrochemistry.open_materials import HOSTS
        if any(not isinstance(v,str) or re.fullmatch(r'[A-Za-z0-9_-]+',v) is None for v in (self.geometry,self.output)):
            raise ValueError('Workspace dataset/output components must be simple identifiers')
        if self.host not in HOSTS:raise ValueError('Known host card required; use RegionEpochDFN API for separately validated chemistry')
        if type(self.nodes) is not int or self.nodes<3 or type(self.requested_regions) is not int or self.requested_regions<1:
            raise ValueError('At least three thickness nodes and a positive region budget required')
        positive=[self.time_step_s,self.C_rate,self.temperature_K,self.electronic_conductivity_S_m,self.active_young_Pa,self.cbd_young_Pa,self.active_Gc_J_m2,self.cbd_Gc_J_m2,self.length_m,self.precrack_radius_m,self.stagger_tolerance]
        if not np.isfinite(positive).all() or min(positive)<=0 or not -1<self.poisson<.5 or not .25<=self.initial_stoichiometry<.9:
            raise ValueError('Finite admissible physical parameters required')
        if not np.isfinite(self.gradient_coefficient_J_m5_mol2) or self.gradient_coefficient_J_m5_mol2<0:raise ValueError('Nonnegative gradient coefficient required')
        if len(self.chemical_strain_per_stoichiometry)!=3 or not np.isfinite(self.chemical_strain_per_stoichiometry).all():raise ValueError('Three finite crystal chemical stretches required')
        PorousCBD(self.cbd_microporosity,self.cbd_ionic_efficiency)

    @classmethod
    def read(cls,path):
        data=tomllib.loads(Path(path).read_text())
        if data.get('schema_version')!=1 or set(data)!={'schema_version','probe'}:raise ValueError('Region probe TOML schema 1 required')
        return cls(**data['probe'])


def _identity(cfg):
    paths=[Path(__file__),Path(__file__).with_name('region_port.py'),Path(__file__).with_name('region_dfn.py'),Path(__file__).with_name('hyperelastic.py')]
    directory=Path(__file__).parent
    paths.extend(directory/name for name in ('epoch_dfn.py','electrolyte_boundary.py','phase_field.py','native_hyperelastic/constitutive.cpp'))
    paths.extend(directory.parent/name for name in ('transport.py','electrochemistry/microstructure_operator.py','electrochemistry/microstructure_dfn.py','topology.py'))
    return hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()+b''.join(p.read_bytes() for p in paths)).hexdigest()


def _context(cfg,workspace,required=False):
    from .backend import require_local_budget
    require_local_budget();ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    out=ws.path('results/top_damage',cfg.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        meta=json.loads((out/'manifest.json').read_text())
        if meta.get('probe_identity')!=_identity(cfg):raise ValueError('Probe configuration/source changed; select a new output')
        from .polycrystal import PolycrystalData
        current=PolycrystalData.load_hdf5(ws.path('results/top_damage/microstructures',cfg.geometry,'initial.h5'))
        if current.fingerprint!=meta['geometry']:raise ValueError('Geometry changed; rebuild the concentration and mechanics states')
        if hashlib.sha256((out/'trial_macro.h5').read_bytes()).hexdigest()!=meta['macro_sha256']:raise ValueError('Regional DFN trial checksum changed')
    elif (out/'trial_macro.h5').exists():raise ValueError('Incomplete initial publication; select a new output instead of guessing checkpoint metadata')
    elif required:raise ValueError('Prepare the regional DFN trial before this stage')
    return ws,out


def run_probe(config,workspace,*,stage='all',node=None):
    cfg=config if isinstance(config,RegionProbeConfig) else RegionProbeConfig.read(config)
    if stage=='prepare':return prepare(cfg,workspace)
    if stage=='node':return solve_node(cfg,workspace,node=node)
    if stage=='feedback':return feedback_probe(cfg,workspace)
    from .region_report import build_region_report
    if stage=='report':return build_region_report(cfg,workspace)
    if stage!='all':raise ValueError('Unknown regional probe stage')
    prepare(cfg,workspace)
    for index in range(cfg.nodes):solve_node(cfg,workspace,node=index)
    result=feedback_probe(cfg,workspace)
    return result | build_region_report(cfg,workspace)


def prepare(cfg,workspace):
    import os,json,time
    from pathlib import Path
    import numpy as np,h5py
    from morqex import Workspace
    from morqex.gemsim import GEMsim
    from morqex.topology import PorousCBD
    from morqex.top_damage.polycrystal import PolycrystalData
    from morqex.top_damage.region_port import RegionPort
    from morqex.top_damage.region_dfn import RegionEpochDFN
    from morqex.top_damage.epoch_dfn import DFNFeedback
    from morqex.top_damage.backend import require_local_budget
    from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode
    from morqex.electrochemistry.open_particle import TransportLaw
    from morqex.nodewise.halfcell import lithium_halfcell_parameters
    ws,out=_context(cfg,workspace);tick=time.perf_counter()
    if (out/'trial_macro.h5').exists():return json.loads((out/'manifest.json').read_text())
    data=PolycrystalData.load_hdf5(ws.path('results/top_damage/microstructures',cfg.geometry,'initial.h5'))
    if data.structure.phase_kinds!={0:'pore',1:'active',2:'cbd'}:raise ValueError('This qualification fixture requires explicit pore/active/CBD phase labels 0/1/2')
    if cfg.length_m<2*max(data.structure.spacing_zyx_m):raise ValueError('Resolve ell with h <= ell/2 before this probe')
    active=float(np.mean(data.structure.labels==1));p,audit=lithium_halfcell_parameters(host=cfg.host,active_fraction=active)
    cmax=float(p['Maximum concentration in positive electrode [mol.m-3]']);p.update({'Initial concentration in positive electrode [mol.m-3]':cfg.initial_stoichiometry*cmax})
    full=MicrostructureElectrode(data.structure,GEMsim(ws),p,'positive',defer_mechanics=True,topology=PorousCBD(cfg.cbd_microporosity,cfg.cbd_ionic_efficiency))
    labels=full.partition_labels(cfg.requested_regions)
    # Enforce the new stricter connected-region contract without merging inventories.
    from scipy.sparse.csgraph import connected_components
    connected=np.empty_like(labels);offset=0
    for old_region in np.unique(labels):
        ids=np.flatnonzero(labels==old_region)
        count,local=connected_components(full.diffusion.stiffness[ids][:,ids],directed=False)
        connected[ids]=local+offset;offset+=count
    labels=connected
    port=RegionPort.from_diffusion(full.diffusion,labels,full.reactive,cfg.initial_stoichiometry*cmax)
    law=TransportLaw(gradient_coefficient_J_m5_mol2=cfg.gradient_coefficient_J_m5_mol2)
    runtime=RegionEpochDFN(port,porosity=full.porosity,nx=cfg.nodes,temperature_K=cfg.temperature_K,transport_law=law)
    feedback=DFNFeedback(cfg.electronic_conductivity_S_m,full.transport['electrolyte']['efficiency'])
    record=runtime.advance_regions(cfg.time_step_s,cfg.C_rate*runtime.capacity_Ah,feedback,np.zeros((cfg.nodes,port.dimension)))
    with h5py.File(out/'trial_macro.h5.pending','w') as f:
        f.attrs['status']='uncoupled trial; six microstates not yet accepted'
        f.attrs['record']=json.dumps(record);f.attrs['port_fingerprint']=port.fingerprint
        f.attrs['inputs']=json.dumps({k:np.asarray(v).tolist() for k,v in runtime.last_inputs.items()})
        for key,value in [('partition',labels),('voxel_ids',port.voxel_ids),('weights',port.weights),('reaction_area',port.reaction_area),('reactive_faces',full.reactive),
            ('concentration_mol_m3',runtime.concentrations()),('voxel_concentration_mol_m3',port.field(runtime.concentrations(),inactive=0.)),
            ('dfn_state',np.asarray(runtime.last.last_state.y).ravel()),('curve/time_s',runtime.last.t),('curve/voltage_V',runtime.last['Voltage [V]'].entries)]:
            f.create_dataset(key,data=value,compression='gzip')
    (out/'trial_macro.h5.pending').replace(out/'trial_macro.h5')
    metadata={'host':cfg.host,'geometry':data.fingerprint,'nodes':cfg.nodes,'microstructure_shape':list(data.structure.labels.shape),'regions':port.dimension,'requested_regions':cfg.requested_regions,
     'reference_concentration_mol_m3':port.reference_concentration,'cmax_mol_m3':cmax,'record':record,'seconds':time.perf_counter()-tick,
     'cbd_ionic_model':PorousCBD(cfg.cbd_microporosity,cfg.cbd_ionic_efficiency).metadata(),'effective_electrolyte_volume_fraction':full.porosity,'resolved_pore_volume_fraction':float(np.mean(data.structure.labels==0)),'feedback':vars(feedback),'transport_law':vars(law),'material_audit':audit,
     'scope':'Full active voxel coverage by conservative connected regions. Independent DFN concentrations at six nodes; frozen geometry and synthetic gradient coefficient. Macro trial only until all finite mechanics states and feedback are verified.'}
    metadata.update(probe_identity=_identity(cfg),configuration=asdict(cfg),port_fingerprint=port.fingerprint,
                    macro_sha256=hashlib.sha256((out/'trial_macro.h5').read_bytes()).hexdigest())
    pending=out/'manifest.json.pending';pending.write_text(json.dumps(metadata,indent=2));pending.replace(out/'manifest.json')
    return metadata


def solve_node(cfg,workspace,*,node):
    import os,json,time,argparse
    from pathlib import Path
    import numpy as np,h5py
    from morqex import Workspace
    from morqex.microstructure import Volume
    from morqex.mechanics.mesh import TetMesh
    from morqex.top_damage.polycrystal import PolycrystalData
    from morqex.top_damage.backend import require_local_budget,NativeConfig,MFEMScalarBackend
    from morqex.top_damage.assembly_cache import CachedP1ScalarBackend
    from morqex.top_damage.phase_field import AT2Damage,FatigueHistory
    from morqex.top_damage.hyperelastic import Hyperelasticity,HyperelasticBackend,finite_staggered_step,chemical_potential
    ws,out=_context(cfg,workspace,True);meta=json.loads((out/'manifest.json').read_text())
    if type(node) is not int or not 0<=node<cfg.nodes:raise ValueError('Node index outside configured cathode')
    if (out/f'trial_node_{node:02d}.h5').exists():
        with h5py.File(out/f'trial_node_{node:02d}.h5') as f:return json.loads(f.attrs['record'])
    data=PolycrystalData.load_hdf5(ws.path('results/top_damage/microstructures',cfg.geometry,'initial.h5'))
    with h5py.File(out/'trial_macro.h5') as f:
        concentration=f['voxel_concentration_mol_m3'][node];partition=f['partition'][:];active_ids=f['voxel_ids'][:];weights=f['weights'][:];fingerprint=f.attrs['port_fingerprint']
    mesh=TetMesh.from_volume(Volume(data.structure.labels,spacing=data.structure.spacing_zyx_m),solid_phases=[1,2]);tick=time.perf_counter()
    scalar=CachedP1ScalarBackend(mesh,MFEMScalarBackend(NativeConfig(),ws))
    damage=AT2Damage(mesh,scalar,np.where(mesh.phases==1,cfg.active_Gc_J_m2,cfg.cbd_Gc_J_m2),cfg.length_m,linear_solver='jacobi-cg')
    mechanics=Hyperelasticity(mesh,HyperelasticBackend(NativeConfig(),ws),np.where(mesh.phases==1,cfg.active_young_Pa,cfg.cbd_young_Pa),cfg.poisson,boundary='kubc',linear_solver='matrix-free-amg-minres')
    p=data.structure.provenance['particles'][0];centre=np.asarray(p['centre_zyx_voxels'])[::-1]*np.asarray(data.structure.spacing_zyx_m)[::-1]
    plane=np.round(centre[0]/data.structure.spacing_zyx_m[2])*data.structure.spacing_zyx_m[2]
    d=np.zeros(len(mesh.points));mask=(abs(mesh.points[:,0]-plane)<1e-14)&(np.linalg.norm(mesh.points[:,1:]-centre[None,1:],axis=1)<=cfg.precrack_radius_m)
    if not mask.any():raise ValueError('Precrack unresolved')
    d[mask]=1.;d=damage.solve(np.zeros(len(mesh.cells)),d).damage
    beta=np.zeros((len(mesh.cells),3,3));selected=mesh.phases==1;grains=data.grain_ids.ravel()[mesh.voxel_ids]
    rotated=data.rotated_tensor_table(np.diag(cfg.chemical_strain_per_stoichiometry));beta[selected]=rotated[grains[selected]-1]
    evals,evecs=np.linalg.eigh(beta);theta=(concentration.ravel()[mesh.voxel_ids]-meta['reference_concentration_mol_m3'])/meta['cmax_mol_m3']
    fc=np.einsum('eik,ek,ejk->eij',evecs,np.exp(evals*theta[:,None]),evecs)
    result=finite_staggered_step(mechanics,damage,d,FatigueHistory.zero(len(mesh.cells)),fc,tolerance=cfg.stagger_tolerance,max_iterations=100)
    state=result.mechanics;mu=chemical_potential(state.first_piola_Pa,state.deformation_gradient,fc,beta@fc/meta['cmax_mol_m3'])
    volume=np.bincount(mesh.voxel_ids,weights=mesh.volumes,minlength=data.structure.labels.size)
    voxel_mu=np.bincount(mesh.voxel_ids,weights=mesh.volumes*mu,minlength=data.structure.labels.size)/np.maximum(volume,np.finfo(float).tiny)
    regional_mu=np.bincount(partition,weights=voxel_mu[active_ids]*volume[active_ids],minlength=len(weights))/(weights*mesh.rve_volume)
    # Independently check the work-conjugate regional derivative.
    rng=np.random.default_rng(7);delta=rng.normal(size=len(weights));full_delta=np.zeros(data.structure.labels.size);full_delta[active_ids]=delta[partition]
    work_full=float(mesh.volumes@(mu*full_delta[mesh.voxel_ids]));work_region=float(mesh.rve_volume*(weights*regional_mu)@delta)
    work_error=abs(work_full-work_region)/max(abs(work_full),abs(work_region),1e-30)
    if work_error>1e-11:raise RuntimeError('Finite mechanics chemical virtual-work projection failed')
    record={'node':node,'seconds':time.perf_counter()-tick,'equilibrium_residual':state.equilibrium_residual,'equilibrium_residual_N':state.equilibrium_residual_N,
            'minimum_jacobian':state.minimum_jacobian,'damage_kkt':result.damage_kkt,'stagger_iterations':result.stagger_iterations,'chemical_virtual_work_relative_error':work_error,
            'scope':'One regional-DFN-driven finite microstate. Fixed template KUBC; synthetic fracture parameters, lagged coupling. Not a complete cyclic topology prediction.'}
    path=out/f'trial_node_{node:02d}.h5';temp=path.with_suffix('.pending')
    with h5py.File(temp,'w') as f:
        f.attrs['record']=json.dumps(record);f.attrs['port_fingerprint']=fingerprint
        for key,value in [('points_m',mesh.points),('tetrahedra',mesh.cells),('voxel_ids',mesh.voxel_ids),('displacement_m',state.displacement_m),('damage',result.damage),('fatigue_J_m3',result.fatigue.accumulated_J_m3),
           ('deformation_gradient',state.deformation_gradient),('chemical_deformation_gradient',fc),('cauchy_stress_Pa',state.cauchy_stress_Pa),('mechanical_chemical_potential_J_mol',mu),('regional_mechanical_potential_J_mol',regional_mu)]:
            f.create_dataset(key,data=value,compression='gzip')
    temp.replace(path);return record


def feedback_probe(cfg,workspace):
    import os,json,time
    from pathlib import Path
    import numpy as np,h5py
    from morqex import Workspace
    from morqex.gemsim import GEMsim
    from morqex.topology import PorousCBD
    from morqex.top_damage.backend import require_local_budget
    from morqex.top_damage.polycrystal import PolycrystalData
    from morqex.top_damage.region_port import RegionPort
    from morqex.top_damage.region_dfn import RegionEpochDFN
    from morqex.top_damage.epoch_dfn import DFNFeedback
    from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode
    from morqex.electrochemistry.open_particle import TransportLaw
    from morqex.nodewise.halfcell import lithium_halfcell_parameters
    ws,out=_context(cfg,workspace,True);meta=json.loads((out/'manifest.json').read_text());tick=time.perf_counter()
    if (out/'feedback_probe.json').exists():return json.loads((out/'feedback_probe.json').read_text())
    data=PolycrystalData.load_hdf5(ws.path('results/top_damage/microstructures',cfg.geometry,'initial.h5'))
    p,_=lithium_halfcell_parameters(host=cfg.host,active_fraction=float(np.mean(data.structure.labels==1)));p.update({'Initial concentration in positive electrode [mol.m-3]':meta['reference_concentration_mol_m3']})
    full=MicrostructureElectrode(data.structure,GEMsim(ws),p,'positive',defer_mechanics=True,topology=PorousCBD(cfg.cbd_microporosity,cfg.cbd_ionic_efficiency))
    with h5py.File(out/'trial_macro.h5') as f:
        labels=f['partition'][:];mask=f['reactive_faces'][:];previous_state=f['dfn_state'][:];inputs=json.loads(f.attrs['inputs']);fingerprint=f.attrs['port_fingerprint'];previous_c=f['concentration_mol_m3'][:]
    inputs={k:np.array(v) if isinstance(v,list) else v for k,v in inputs.items()}
    port=RegionPort.from_diffusion(full.diffusion,labels,mask,meta['reference_concentration_mol_m3'])
    if port.fingerprint!=fingerprint:raise RuntimeError('Stale regional operator')
    mu=[];node_records=[]
    for node in range(cfg.nodes):
        with h5py.File(out/f'trial_node_{node:02d}.h5') as f:
            if f.attrs['port_fingerprint']!=fingerprint:raise ValueError('Stale finite microstate')
            mu.append(f['regional_mechanical_potential_J_mol'][:]);node_records.append(json.loads(f.attrs['record']))
    mu=np.asarray(mu);results={}
    path=out/'feedback_probe.h5';temp=path.with_suffix('.pending')
    with h5py.File(temp,'w') as f:
        f.create_dataset('projected_mechanical_potential_J_mol',data=mu)
        for name,feedback_mu in [('finite_feedback',mu),('matched_no_feedback',np.zeros_like(mu))]:
            runtime=RegionEpochDFN(port,porosity=full.porosity,nx=cfg.nodes,temperature_K=cfg.temperature_K,transport_law=TransportLaw(**meta['transport_law']))
            runtime.restore_state(meta['record']['time_s'],previous_state,inputs)
            record=runtime.advance_regions(cfg.time_step_s,cfg.C_rate*runtime.capacity_Ah,DFNFeedback(**meta['feedback']),feedback_mu)
            results[name]=record
            for key,value in [('concentration_mol_m3',runtime.concentrations()),('voxel_concentration_mol_m3',port.field(runtime.concentrations(),inactive=0.)),('dfn_state',np.asarray(runtime.last.last_state.y).ravel()),('time_s',runtime.last.t),('voltage_V',runtime.last['Voltage [V]'].entries)]:
                f.create_dataset(name+'/'+key,data=value,compression='gzip')
        f.attrs['scope']='Two-segment, six-node local finite-mechanics feedback probe; not accepted cyclic topology evolution'
        f.attrs['port_fingerprint']=port.fingerprint
        f.attrs['results']=json.dumps(results)
    temp.replace(path)
    record={'nodes':cfg.nodes,'voxel_shape':list(data.structure.labels.shape),'regions':port.dimension,'microstate_records':node_records,'results':results,
     'feedback_voltage_difference_V':results['finite_feedback']['voltage_V']-results['matched_no_feedback']['voltage_V'],'seconds':time.perf_counter()-tick,
     'scope':'Conservative full-voxel regional concentrations at all six nodes -> six actual nonlinear fatigue microstates -> six independent projected chemical-potential inputs -> next DFN segment. Fixed geometry, KUBC, explicit lag. No physical open-pore conversion or calibrated lifetime prediction.',
     'physical_topology_qualified':False,'publication_qualified':False}
    (out/'feedback_probe.json').write_text(json.dumps(record,indent=2));return record


