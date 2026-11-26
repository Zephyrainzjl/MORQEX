"""Conservative full-voxel resolved electrochemistry -> finite AT2 history.

This is a one-way reference bridge, not a claim of open-crack transport or
bidirectional resolved lifetime qualification. All active voxels are retained.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json
import h5py
import numpy as np
from morqex import Workspace
from .backend import require_local_budget,NativeConfig,MFEMScalarBackend
from .polycrystal import PolycrystalData

@dataclass(frozen=True)
class ResolvedMechanicsConfig:
    cmax_mol_m3:float=49600.
    reference_stoichiometry:float=.5
    active_young_Pa:float=93e9
    cbd_young_Pa:float=1e9
    poisson:float=.3
    active_Gc_J_m2:float=10.
    cbd_Gc_J_m2:float=10000.
    length_m:float=1e-6
    precrack_radius_m:float=1e-6
    chemical_strain_per_stoichiometry:tuple=(.012,.012,.040)
    stagger_tolerance:float=1e-6
    boundary:str='kubc'

    def __post_init__(self):
        positive=[self.cmax_mol_m3,self.active_young_Pa,self.cbd_young_Pa,self.active_Gc_J_m2,self.cbd_Gc_J_m2,self.length_m,self.precrack_radius_m,self.stagger_tolerance]
        if not np.isfinite(positive).all() or min(positive)<=0 or not 0<self.reference_stoichiometry<1 or not -1<self.poisson<.5:raise ValueError('Finite admissible SI material parameters required')
        if self.boundary not in ('kubc','neumann'):raise ValueError('Supported reference constraint required')
        if np.asarray(self.chemical_strain_per_stoichiometry).shape!=(3,) or not np.isfinite(self.chemical_strain_per_stoichiometry).all():raise ValueError('Three finite crystal expansion coefficients required')


def read_resolved_cathode(path,data,*,cmax_mol_m3):
    """Exact same-grid crop: the cathode occupies the full transverse section.

    Require provenance schema, SI spacing/origin and both solid phase masks.
    No spatial interpolation, radial cloning or silent material relabeling.
    """
    labels=data.structure.labels;shape=labels.shape
    if np.any(np.asarray(data.structure.origin_zyx_m)!=0):raise ValueError('This finite-mesh bridge requires explicit zero-origin material coordinates')
    with h5py.File(path) as f:
        if f.attrs.get('schema')!='morqex-resolved-fields-1' or f.attrs.get('axis_order')!='ZYX':raise ValueError('Resolved ZYX field schema required')
        if not np.allclose(f.attrs['spacing_zyx_m'],data.structure.spacing_zyx_m,rtol=0,atol=1e-20) or not np.allclose(f.attrs['origin_zyx_m'],data.structure.origin_zyx_m,rtol=0,atol=1e-20):raise ValueError('Resolved and material coordinates differ')
        if f['Concentration'].shape[:2]!=shape[:2] or f['Concentration'].shape[2]<shape[2]:raise ValueError('Complete cathode lattice required')
        crop=(slice(None),slice(None),slice(0,shape[2]))
        for name,phase in [('Cathode_active_material',1),('Cathode_CBD',2)]:
            if not np.array_equal(f[name][crop],labels==phase):raise ValueError('Resolved/material phase mismatch: '+name)
        c=f['Concentration'][crop];potential=f['Potential'][crop];time=float(f.attrs['time_s'])
    if not np.isfinite(c).all() or not np.isfinite(potential).all() or not np.isfinite(time) or time<0:raise ValueError('Finite synchronized fields required')
    if np.any((c[labels==1]<=0)|(c[labels==1]>=cmax_mol_m3)):raise ValueError('Active concentration outside physical bounds')
    return c,potential,time


def advance_resolved_phase_field(geometry,field,output,workspace,*,config=None,previous=None):
    """One accepted full-field state with explicit history/checkpoint input.

    Native electrochemical fluxes do not yet consume the returned mechanical
    chemical potential. Synthetic mechanical parameters require calibration.
    """
    require_local_budget();cfg=config or ResolvedMechanicsConfig();ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    output=ws.path(str(output))
    if output.exists():raise ValueError('Never overwrite an accepted resolved mechanical state')
    output.parent.mkdir(parents=True,exist_ok=True)
    data=PolycrystalData.load_hdf5(geometry);c,potential,time=read_resolved_cathode(field,data,cmax_mol_m3=cfg.cmax_mol_m3)
    if cfg.length_m<2*max(data.structure.spacing_zyx_m):raise ValueError('Resolve the phase-field length with h <= ell/2')
    from morqex.microstructure import Volume
    from morqex.mechanics.mesh import TetMesh
    from .assembly_cache import CachedP1ScalarBackend
    from .phase_field import AT2Damage,FatigueHistory
    from .hyperelastic import Hyperelasticity,HyperelasticBackend,finite_staggered_step,chemical_potential
    mesh=TetMesh.from_volume(Volume(data.structure.labels,spacing=data.structure.spacing_zyx_m),solid_phases=[1,2])
    backend=HyperelasticBackend(NativeConfig(),ws)
    source_names=['resolved_phase_field.py','phase_field.py','hyperelastic.py','assembly_cache.py']
    sources={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in source_names}
    identity=hashlib.sha256(json.dumps({'geometry':data.fingerprint,'config':asdict(cfg),'sources':sources,'native':backend.manifest['identity']},sort_keys=True).encode()).hexdigest()
    scalar=CachedP1ScalarBackend(mesh,MFEMScalarBackend(NativeConfig(),ws))
    damage=AT2Damage(mesh,scalar,np.where(mesh.phases==1,cfg.active_Gc_J_m2,cfg.cbd_Gc_J_m2),cfg.length_m,linear_solver='jacobi-cg')
    mechanics=Hyperelasticity(mesh,backend,np.where(mesh.phases==1,cfg.active_young_Pa,cfg.cbd_young_Pa),cfg.poisson,boundary=cfg.boundary,linear_solver='matrix-free-amg-minres')
    if previous is None:
        d=np.zeros(len(mesh.points));history=FatigueHistory.zero(len(mesh.cells));u=None
        particle=data.structure.provenance['particles'][0];centre=np.asarray(particle['centre_zyx_voxels'])[::-1]*np.asarray(data.structure.spacing_zyx_m)[::-1]
        plane=np.round(centre[0]/data.structure.spacing_zyx_m[2])*data.structure.spacing_zyx_m[2]
        mask=(abs(mesh.points[:,0]-plane)<1e-14)&(np.linalg.norm(mesh.points[:,1:]-centre[None,1:],axis=1)<=cfg.precrack_radius_m)
        if not mask.any():raise ValueError('Precrack unresolved')
        d[mask]=1.;d=damage.solve(np.zeros(len(mesh.cells)),d).damage
    else:
        with h5py.File(previous) as f:
            if f.attrs['identity']!=identity or f.attrs['time_s']>=time:raise ValueError('History identity mismatch or non-increasing physical time')
            d=f['damage'][:];u=f['displacement_m'][:]
            history=FatigueHistory(f['fatigue_J_m3'][:],f['previous_alpha_J_m3'][:])
    beta=np.zeros((len(mesh.cells),3,3));active=mesh.phases==1;grains=data.grain_ids.ravel()[mesh.voxel_ids]
    beta[active]=data.rotated_tensor_table(np.diag(cfg.chemical_strain_per_stoichiometry))[grains[active]-1]
    eigenvalues,eigenvectors=np.linalg.eigh(beta)
    theta=c.ravel()[mesh.voxel_ids]/cfg.cmax_mol_m3-cfg.reference_stoichiometry
    fc=np.einsum('eik,ek,ejk->eij',eigenvectors,np.exp(eigenvalues*theta[:,None]),eigenvectors)
    # Exact stress-free reference avoids eigensystem roundoff loading a
    # nominally zero-force solve. This is algebraic identity, not clipping.
    fc[theta==0]=np.eye(3)
    result=finite_staggered_step(mechanics,damage,d,history,fc,tolerance=cfg.stagger_tolerance,max_iterations=100,initial_displacement=u)
    state=result.mechanics
    mu=chemical_potential(state.first_piola_Pa,state.deformation_gradient,fc,beta@fc/cfg.cmax_mol_m3)
    solid_volume=np.bincount(mesh.voxel_ids,weights=mesh.volumes,minlength=data.structure.labels.size)
    voxel_mu=np.bincount(mesh.voxel_ids,weights=mesh.volumes*mu,minlength=data.structure.labels.size)/np.maximum(solid_volume,np.finfo(float).tiny)
    mass_micro=float(mesh.volumes[active]@c.ravel()[mesh.voxel_ids[active]])
    mass_voxel=float(c[data.structure.labels==1].sum()*np.prod(data.structure.spacing_zyx_m))
    transfer_error=abs(mass_micro-mass_voxel)/max(abs(mass_voxel),1e-30)
    if transfer_error>1e-12 or np.min(result.damage-d)<-1e-12:raise RuntimeError('Conservative transfer or damage irreversibility failed')
    record={'time_s':time,'identity':identity,'equilibrium_residual':state.equilibrium_residual,'damage_kkt':result.damage_kkt,'minimum_jacobian':state.minimum_jacobian,'inventory_transfer_relative_error':transfer_error,'stagger_iterations':result.stagger_iterations,'active_voxels':int((data.structure.labels==1).sum()),'tetrahedra':len(mesh.cells),'scope':'Resolved 3D concentration -> finite-strain fatigue AT2; one-way, fixed material topology, synthetic parameters; not qualified lifetime feedback'}
    temporary=output.with_suffix('.pending')
    with h5py.File(temporary,'w') as f:
        for k,v in record.items():f.attrs[k]=v
        f.attrs['field_sha256']=hashlib.sha256(Path(field).read_bytes()).hexdigest();f.attrs['previous']=str(previous or '')
        for key,value in [('points_m',mesh.points),('tetrahedra',mesh.cells),('voxel_ids',mesh.voxel_ids),('voxel_concentration_mol_m3',c),('voxel_potential_V',potential),('voxel_mechanical_potential_J_mol',voxel_mu.reshape(data.structure.labels.shape)),('displacement_m',state.displacement_m),('damage',result.damage),('fatigue_J_m3',result.fatigue.accumulated_J_m3),('previous_alpha_J_m3',result.fatigue.previous_alpha_J_m3),('chemical_deformation_gradient',fc),('deformation_gradient',state.deformation_gradient),('cauchy_stress_Pa',state.cauchy_stress_Pa)]:f.create_dataset(key,data=value,compression='gzip')
    temporary.replace(output);return record
