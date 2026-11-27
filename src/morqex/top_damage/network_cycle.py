"""Accepted-state 3D finite-fatigue / dual-network / regional DFN driver.

The diffuse conductance law is an EXPLICIT numerical candidate, not a physical
open-crack/contact reconstruction. Immutable material volumes and Li states are
retained. This executable path is useful for validating coupling and identifying
failures; passing it alone does not qualify lifetime or arbitrary microstructure.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,time,tomllib,re
import numpy as np
from morqex.workspace import Workspace


@dataclass(frozen=True)
class NetworkCycleConfig:
    output: str = 'network_cycle_candidate'
    host: str = 'NMC811'
    stoichiometry_window: tuple = (.25,.9)
    C_rate: float = .5
    charge_C_rate: float = .5
    cycles: int = 2
    nodes: int = 3
    concentration_regions: int = 0
    local_mobility: bool = False
    diffusivity_interpretation: str = 'tracer_reference'
    increments_per_half_cycle: int = 24
    temperature_K: float = 298.15
    active_young_Pa: float = 93e9
    cbd_young_Pa: float = 1e9
    poisson: float = .3
    active_Gc_J_m2: float = 10.
    cbd_Gc_J_m2: float = 10000.
    length_m: float = 1.5e-6
    chemical_strain_per_stoichiometry: tuple = (.012,.012,.04)
    active_conductivity_S_m: float = 10.
    cbd_conductivity_S_m: float = 100.
    cbd_microporosity: float = .47
    cbd_ionic_efficiency: float = .05
    gradient_coefficient_J_m5_mol2: float = 1e-14
    conductance_floor: float = 1e-5
    graph_update_tolerance: float = .005
    stagger_tolerance: float = 1e-6
    dfn_rtol: float = 1e-10
    dfn_atol: float = 1e-11
    project_algebraic_output: bool = False
    mechanics_enabled: bool = True
    topology_feedback: bool = True

    def __post_init__(self):
        if not re.fullmatch('[A-Za-z0-9_-]+',self.output):raise ValueError('Safe output identifier required')
        if type(self.concentration_regions) is not int or self.concentration_regions < -1:
            raise ValueError('Regions: -1 full voxel, 0 grains, or positive connected aggregation target')
        if len(self.stoichiometry_window)!=2 or not .001<self.stoichiometry_window[0]<.5<self.stoichiometry_window[1]<.999:
            raise ValueError('Explicit OCP window must contain the initial stoichiometry 0.5')
        for name,minimum in [('cycles',1),('nodes',3),('increments_per_half_cycle',4)]:
            if type(getattr(self,name)) is not int or getattr(self,name)<minimum:raise ValueError('Invalid integer cycle/grid controls')
        positive=[self.C_rate,self.charge_C_rate,self.temperature_K,self.active_young_Pa,self.cbd_young_Pa,self.active_Gc_J_m2,self.cbd_Gc_J_m2,self.length_m,self.active_conductivity_S_m,self.cbd_conductivity_S_m,self.stagger_tolerance]
        if not np.isfinite(positive).all() or min(positive)<=0 or not -1<self.poisson<.5:raise ValueError('Finite positive physical coefficients required')
        if not 0<self.conductance_floor<1 or not 0<=self.graph_update_tolerance<1:raise ValueError('Explicit conductance floor and update tolerance required')
        if not np.isfinite([self.dfn_rtol,self.dfn_atol]).all() or not 0<min(self.dfn_rtol,self.dfn_atol)<=max(self.dfn_rtol,self.dfn_atol)<1e-3:
            raise ValueError('Finite positive DFN integration tolerances required')
        if type(self.project_algebraic_output) is not bool:raise ValueError('Boolean output projection control required')
        if self.diffusivity_interpretation not in ('tracer_reference','chemical_diffusivity'):
            raise ValueError('Explicit diffusivity interpretation required')
        if self.diffusivity_interpretation=='chemical_diffusivity' and not self.local_mobility:
            raise ValueError('Chemical diffusivity conversion requires local mobility')
        if not 0<self.cbd_ionic_efficiency<=self.cbd_microporosity<1:raise ValueError('Admissible porous CBD required')
        if len(self.chemical_strain_per_stoichiometry)!=3 or not np.isfinite(self.chemical_strain_per_stoichiometry).all():raise ValueError('Finite chemical stretches required')


def _json(obj):
    return json.dumps(obj,default=lambda x:np.asarray(x).tolist(),allow_nan=False)


def run_network_cycle(config_path,workspace,*,steps=1):
    import h5py
    from morqex.microstructure import Volume
    from morqex.transport import VoxelDiffusion
    from morqex.mechanics import TetMesh
    from .network_geometry import load_cycle_geometry
    from .region_port import RegionPort
    from .network_dfn import NetworkEpochDFN
    from .epoch_dfn import DFNFeedback
    from .dual_network import DualNetworkClosure
    from .graph_transport import voxel_network
    from .backend import require_local_budget,MFEMScalarBackend,NativeConfig
    from .assembly_cache import CachedP1ScalarBackend
    from .phase_field import AT2Damage,FatigueHistory
    from .hyperelastic import HyperelasticBackend,Hyperelasticity,finite_staggered_step,chemical_potential
    from .dense_cycle import _voxel_average
    from morqex.nodewise.halfcell import lithium_halfcell_parameters
    from morqex.electrochemistry.open_particle import TransportLaw
    require_local_budget()
    if type(steps) is not int or steps<1:raise ValueError('Positive accepted-step budget required')
    raw=tomllib.loads(Path(config_path).read_text())
    if set(raw)!={'schema_version','cycle','microstructure'} or raw['schema_version']!=1:raise ValueError('Network-cycle TOML schema 1 required')
    cfg=NetworkCycleConfig(**raw['cycle'])
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    data,geometry_metadata,solver_seed=load_cycle_geometry(raw['microstructure'],ws)
    out=ws.path('results/top_damage/network_cycles',cfg.output);out.mkdir(parents=True,exist_ok=True)
    # Hash runtime dependencies, including upstream solver code, for restart.
    root=Path(__file__).parents[1]
    paths=sorted((root/'top_damage').glob('*.py'))+sorted((root/'electrochemistry').glob('*.py'))
    paths += [root/'nodewise/halfcell.py',root/'transport.py',root/'generation/particles.py']
    paths += [root.parent/'pybamm/expression_tree/averages.py',root.parent/'pybamm/solvers/solution.py',root.parent/'pybamm/discretisations/discretisation.py']
    identity=hashlib.sha256(_json(raw).encode()+data.fingerprint.encode()+b''.join(p.read_bytes() for p in paths)).hexdigest()
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['identity']!=identity:
        raise ValueError('Source/config identity changed; choose a new output')
    labels=data.structure.labels;shape=labels.shape;h=data.structure.spacing_zyx_m
    if cfg.length_m<2*max(h):raise ValueError('Resolve ell with h <= ell/2 before refinement qualification')
    volume=Volume(labels,spacing=h,origin=data.structure.origin_zyx_m)
    fv=VoxelDiffusion(volume,active_phases=[1],diffusivity=1.,particle_ids=data.particle_ids,reactive_phases=[0,2])
    partition=data.grain_ids.ravel()[fv.voxel_ids]-1
    if cfg.concentration_regions:
        from .region_partition import connected_partition
        partition=connected_partition(fv,len(fv.voxel_ids) if cfg.concentration_regions==-1 else cfg.concentration_regions)
    p,audit=lithium_halfcell_parameters(host=cfg.host,active_fraction=float(np.mean(labels==1)),stoichiometry_window=cfg.stoichiometry_window)
    cmax=float(p['Maximum concentration in positive electrode [mol.m-3]'])
    port=RegionPort.from_diffusion(fv,partition,fv.reactive,.5*cmax)
    porosity=float(np.mean(labels==0)+cfg.cbd_microporosity*np.mean(labels==2))
    runtime=NetworkEpochDFN(port,host=cfg.host,porosity=porosity,nx=cfg.nodes,temperature_K=cfg.temperature_K,stoichiometry_window=cfg.stoichiometry_window,local_mobility=cfg.local_mobility,
        solver_rtol=cfg.dfn_rtol,solver_atol=cfg.dfn_atol,project_algebraic_output=cfg.project_algebraic_output,
        diffusivity_interpretation=cfg.diffusivity_interpretation,
        transport_law=TransportLaw(gradient_coefficient_J_m5_mol2=cfg.gradient_coefficient_J_m5_mol2))
    closures=[DualNetworkClosure(fv,port,update_tolerance=cfg.graph_update_tolerance) for _ in range(cfg.nodes)]
    base_e=np.where(labels==1,cfg.active_conductivity_S_m,np.where(labels==2,cfg.cbd_conductivity_S_m,0.))
    base_i=np.where(labels==0,1.,np.where(labels==2,cfg.cbd_ionic_efficiency,0.))
    mesh=TetMesh.from_volume(volume,solid_phases=[1,2])
    young=np.where(mesh.phases==1,cfg.active_young_Pa,cfg.cbd_young_Pa)
    backend=HyperelasticBackend(NativeConfig(),ws) if cfg.mechanics_enabled else None
    damage_model=mechanics=None
    if cfg.mechanics_enabled:
        scalar=CachedP1ScalarBackend(mesh,MFEMScalarBackend(NativeConfig(),ws))
        damage_model=AT2Damage(mesh,scalar,np.where(mesh.phases==1,cfg.active_Gc_J_m2,cfg.cbd_Gc_J_m2),cfg.length_m,linear_solver='jacobi-cg')
        # PyAMG estimates spectral radii with NumPy's legacy random generator.
        # Freeze only setup randomness and restore the caller's RNG afterwards.
        # This preserves the same preconditioner across checkpoint restarts.
        rng_state=np.random.get_state()
        try:
            np.random.seed(solver_seed)
            mechanics=Hyperelasticity(mesh,backend,young,cfg.poisson,boundary='kubc',linear_solver='matrix-free-amg-minres')
        finally:np.random.set_state(rng_state)
    beta=np.zeros((len(mesh.cells),3,3));active_cells=mesh.phases==1
    table=data.rotated_tensor_table(np.diag(cfg.chemical_strain_per_stoichiometry))
    beta[active_cells]=table[data.grain_ids.ravel()[mesh.voxel_ids[active_cells]]-1]
    eigenvalues,eigenvectors=np.linalg.eigh(beta)
    damage=np.zeros((cfg.nodes,len(mesh.points)));displacement=np.zeros((cfg.nodes,3*len(mesh.points)))
    fatigue=[FatigueHistory.zero(len(mesh.cells)) for _ in range(cfg.nodes)]
    mu=np.zeros((cfg.nodes,port.dimension));area=np.ones_like(mu)
    edges=np.ones((cfg.nodes,len(runtime.edges)));sigma=np.zeros(cfg.nodes);efficiency=np.zeros(cfg.nodes)
    checkpoints=sorted(out.glob('step_*.h5'))
    phase='conditioning_charge';cycles=0;phase_start_q=0.;last_step=-1
    if checkpoints:
        with h5py.File(checkpoints[-1],'r') as f:
            if f.attrs['identity']!=identity:raise ValueError('Checkpoint identity mismatch')
            rec=json.loads(f.attrs['record']);last_step=rec['step'];phase=rec['next_phase'];cycles=rec['completed_cycles'];phase_start_q=rec['phase_start_capacity_Ah']
            damage=f['damage'][:];displacement=f['displacement_m'][:]
            fatigue=[FatigueHistory(a,b) for a,b in zip(f['fatigue_J_m3'][:],f['alpha_previous_J_m3'][:])]
            mu=f['next/mu_J_mol'][:];area=f['next/reaction_fractions'][:];edges=f['next/edge_factors'][:]
            sigma=f['next/electronic_S_m'][:];efficiency=f['next/ionic_efficiency'][:]
            for node,closure in enumerate(closures):
                for name,cache,base,channel,side in [('electronic',closure.electronic_cache,base_e,'electronic',1),('ionic',closure.ionic_cache,base_i,'ionic',0)]:
                    saved=f[f'cache/{node}/{name}']
                    layout=voxel_network(base,h,channel=channel,reservoir_side=side)
                    cache.restore(layout,saved[:],saved.attrs['rebuilds'],saved.attrs['reuses'])
            if last_step>0:runtime.restore_state(rec['time_s'],f['dfn_state'][:],json.loads(f.attrs['inputs']))
    else:
        data.save_hdf5(out/'initial.h5')
        with h5py.File(out/'mesh.h5','w') as f:
            for name,value in [('points_m',mesh.points),('cells',mesh.cells),('voxel_ids',mesh.voxel_ids),('partition',partition),('material_voxel_ids',fv.voxel_ids)]:f.create_dataset(name,data=value,compression='gzip')
    manifest={'identity':identity,'config':asdict(cfg),'geometry':geometry_metadata,'material_audit':audit,
        'maximum_concentration_mol_m3':cmax,
        'regions':port.dimension,'tetrahedra':len(mesh.cells),'nodes':cfg.nodes,'protocol_complete':cycles>=cfg.cycles,
        'physical_topology_qualified':False,'publication_qualified':False,
        'limitations':['Diffuse quadratic conductance candidate, not calibrated crack aperture/CBD contact evolution.',
        'Fixed pore and porous-CBD ionic geometry; no invented crack wetting or volume loss.',
        'Each SVE is locally KUBC-clamped (macro F=I); not a force-equilibrated cell stack.',
        ('Native D is an Onsager mobility reference here, not calibrated chemical diffusivity for arbitrary OCP.'
         if cfg.diffusivity_interpretation=='tracer_reference' else
         'Chemical diffusivity uses verified positive inverse chemical curvature; Fick limit and spatial convergence remain separate gates.'),
        'Concentration region/full-voxel closure needs mesh refinement; not a resolved 3D electrolyte reference.',
        'Isothermal explicit stagger; split/time convergence and calibrated fatigue remain independent gates.',
        'Event certificate bounds passive graph response only, not accumulated DFN voltage/capacity error.',
        'No cycle jumping: every reported cycle is actually integrated.']}
    manifest_path.write_text(_json(manifest))
    started=time.perf_counter()
    for step in range(last_step+1,last_step+1+steps):
        if cycles>=cfg.cycles:break
        tick=time.perf_counter();used_phase=phase;phase_capacity=None
        if step==0:
            electro={'time_s':0.,'voltage_V':None,'inventory_relative_defect':0.,'relative_state_jump':0.,'net_capacity_Ah':0.,'termination':'initial'}
        else:
            rate=cfg.charge_C_rate if phase in ('conditioning_charge','charge') else cfg.C_rate
            current=(-1 if phase in ('conditioning_charge','charge') else 1)*rate*runtime.capacity_Ah
            # Rollback the full electrochemical transaction on a failed trial;
            # no fatigue or topology state is committed before this succeeds.
            electro=runtime.advance_network(3600/rate/cfg.increments_per_half_cycle,current,DFNFeedback(float(sigma.mean()),float(efficiency.mean())),mu,
                reaction_fractions=area,edge_factors=edges,electronic_conductivity_S_m=sigma,electrolyte_efficiency=efficiency)
            if electro['termination']!='final time':
                required='minimum' if phase=='discharge' else 'maximum'
                cutoff=float(runtime.parameters[('Lower' if phase=='discharge' else 'Upper')+' voltage cut-off [V]'])
                if required not in electro['termination'].lower() or abs(electro['voltage_V']-cutoff)>1e-4:raise RuntimeError('Unexpected event; cycle is incomplete')
                phase_capacity=abs(electro['net_capacity_Ah']-phase_start_q);phase_start_q=electro['net_capacity_Ah']
                if phase=='conditioning_charge':phase='discharge'
                elif phase=='discharge':phase='charge'
                else:cycles+=1;phase='discharge'
        concentrations=runtime.concentrations();fields=[];diagnostics=[]
        new_damage=[];new_displacement=[];new_fatigue=[];new_mu=[];new_area=[];new_edges=[];new_sigma=[];new_efficiency=[]
        for node in range(cfg.nodes):
            c=port.field(concentrations[node],inactive=0.)
            if cfg.mechanics_enabled and step>0:
                theta=(c.ravel()[mesh.voxel_ids]-port.reference_concentration)/cmax
                fc=np.einsum('eik,ek,ejk->eij',eigenvectors,np.exp(eigenvalues*theta[:,None]),eigenvectors)
                result=finite_staggered_step(mechanics,damage_model,damage[node],fatigue[node],fc,
                    initial_displacement=displacement[node],tolerance=cfg.stagger_tolerance)
                d=result.damage;u=result.mechanics.displacement_m;history=result.fatigue;state=result.mechanics
                if np.any(d<damage[node]-1e-12):raise ArithmeticError('Irreversibility violation')
                dv=_voxel_average(mesh,d[mesh.cells].mean(axis=1),shape)
                mechanical=chemical_potential(state.first_piola_Pa,state.deformation_gradient,fc,beta@fc/cmax)
                mu_voxel=_voxel_average(mesh,mechanical,shape)
                regional_mu=port.project_mechanical_potential(mu_voxel.ravel()[fv.voxel_ids])
                stress=_voxel_average(mesh,state.cauchy_stress_Pa.reshape(-1,9),shape)
                g=(damage_model.mean_degradation(d)-damage_model.k)/(1-damage_model.k)
                gv=cfg.conductance_floor+(1-cfg.conductance_floor)*_voxel_average(mesh,g,shape)
                mechanical_diag={'equilibrium_residual':state.equilibrium_residual,'damage_kkt':result.damage_kkt,
                    'minimum_jacobian':state.minimum_jacobian,'stagger_iterations':result.stagger_iterations,
                    'elastic_energy_J':state.elastic_energy_J,'max_damage':float(d.max()),'mean_active_damage':float(dv[labels==1].mean())}
                threshold=damage_model.Gc/(12*damage_model.ell)
                mechanical_diag.update(fracture_energy_J=damage_model.energies(d,state.tensile_energy_J_m3,history.toughness_factor(threshold))[1],
                    fatigue_fracture_energy_change_J=result.fatigue_fracture_energy_change_J)
                mechanics.solve_diagnostics.clear()
            else:
                d=damage[node];u=displacement[node];history=fatigue[node];dv=np.zeros(shape);regional_mu=np.zeros(port.dimension)
                stress=np.zeros(shape+(9,));gv=np.ones(shape);mechanical_diag={}
            factors=np.ones((3,)+shape)
            for a in range(3):
                lo=[slice(None)]*3;hi=lo.copy();lo[a]=slice(None,-1);hi[a]=slice(1,None);lo,hi=tuple(lo),tuple(hi)
                factors[a][lo]=2*gv[lo]*gv[hi]/(gv[lo]+gv[hi])
            network=closures[node].evaluate(base_e*(gv if cfg.topology_feedback else 1),base_i,
                solid_face_factors=factors if cfg.topology_feedback else None)
            if network.diagnostics['unresolved_split_regions']:raise RuntimeError('Topology split inside region: enrich before accepting')
            new_damage.append(d);new_displacement.append(u);new_fatigue.append(history)
            new_mu.append(regional_mu);new_area.append(network.reaction_fractions);new_edges.append(network.diffusion_edge_factors)
            new_sigma.append(network.electronic_conductivity_S_m);new_efficiency.append(network.electrolyte_efficiency)
            fields.append({'concentration_mol_m3':c,'damage':dv,'stress_Pa':stress,
                'electronic_potential_unit_load':network.electronic_potential,'ionic_potential_unit_load':network.ionic_potential,
                'electron_accessible':network.electron_accessible,'ion_accessible':network.ion_accessible,
                'fatigue_J_m3':_voxel_average(mesh,history.accumulated_J_m3,shape)})
            diagnostics.append(mechanical_diag|network.diagnostics)
        damage=np.array(new_damage);displacement=np.array(new_displacement);fatigue=new_fatigue
        mu=np.array(new_mu);area=np.array(new_area);edges=np.array(new_edges);sigma=np.array(new_sigma);efficiency=np.array(new_efficiency)
        record={**electro,'step':step,'protocol_phase':used_phase,'next_phase':phase,'completed_cycles':cycles,
            'phase_start_capacity_Ah':phase_start_q,'completed_phase_capacity_Ah':phase_capacity,
            'node_diagnostics':diagnostics,'seconds':time.perf_counter()-tick,
            'solid_lithium_mol_per_SVE':(concentrations@port.weights*fv.rve_volume).tolist()}
        target=out/f'step_{step:06d}.h5';pending=target.with_suffix('.pending')
        with h5py.File(pending,'w') as f:
            f.attrs.update(identity=identity,record=_json(record),inputs=_json(runtime.last_inputs))
            arrays={'damage':damage,'displacement_m':displacement,'fatigue_J_m3':np.array([a.accumulated_J_m3 for a in fatigue]),
                'alpha_previous_J_m3':np.array([a.previous_alpha_J_m3 for a in fatigue]),'concentration_regions_mol_m3':concentrations,
                'next/mu_J_mol':mu,'next/reaction_fractions':area,'next/edge_factors':edges,'next/electronic_S_m':sigma,'next/ionic_efficiency':efficiency}
            for name,value in arrays.items():f.create_dataset(name,data=value,compression='gzip')
            for name in fields[0]:f.create_dataset('voxel/'+name,data=np.array([a[name] for a in fields]),compression='gzip')
            for node,closure in enumerate(closures):
                for name,cache in [('electronic',closure.electronic_cache),('ionic',closure.ionic_cache)]:
                    saved=f.create_dataset(f'cache/{node}/{name}',data=cache.reference.network.conductance_S,compression='gzip')
                    saved.attrs.update(rebuilds=cache.rebuilds,reuses=cache.reuses)
            if step>0:
                f.create_dataset('dfn_state',data=np.asarray(runtime.last.last_state.y).ravel())
                for name,key in [('voltage_V','Voltage [V]'),('current_A','Current [A]'),('net_capacity_Ah','Discharge capacity [A.h]')]:f.create_dataset('curve/'+name,data=runtime.last[key].entries)
                f.create_dataset('curve/time_s',data=runtime.last.t)
        pending.replace(target);last_step=step
        manifest.update(last_step=step,completed_cycles=cycles,protocol_complete=cycles>=cfg.cycles,seconds_this_invocation=time.perf_counter()-started)
        temp=manifest_path.with_suffix('.pending');temp.write_text(_json(manifest));temp.replace(manifest_path)
        print(_json({'step':step,'time_s':record['time_s'],'V':record['voltage_V'],'cycles':cycles,'seconds':record['seconds']}),flush=True)
    return manifest|{'directory':str(out)}
