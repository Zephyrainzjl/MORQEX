"""Fixed-geometry adaptive static verification; no cycling qualification."""
import argparse,json,os,time
import numpy as np,h5py
from morqex import Workspace
from morqex.mechanics import TetMesh
from morqex.microstructure import Volume
from morqex.top_damage.polycrystal import PolycrystalData
from morqex.top_damage.backend import NativeConfig,MFEMScalarBackend
from morqex.top_damage.assembly_cache import CachedP1ScalarBackend
from morqex.top_damage.hyperelastic import HyperelasticBackend,Hyperelasticity,finite_staggered_step
from morqex.top_damage.phase_field import AT2Damage,FatigueHistory
from morqex.top_damage.refinement import RefinementBackend
from morqex.top_damage.refinement_indicator import coupled_indicator,mark_bulk


def main():
    p=argparse.ArgumentParser();p.add_argument('--level',type=int,required=True)
    p.add_argument('--fraction',type=float,default=.5);args=p.parse_args()
    if args.level<0:raise ValueError('Nonnegative level required')
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);base=ws.path('results/top_damage/network_cycles/dual_long20_checked_v1')
    data=PolycrystalData.load_hdf5(base/'initial.h5');cfg=json.loads((base/'manifest.json').read_text())['config']
    root=ws.path('results/top_damage/static_mesh_adaptive_v1');out=root/f'level{args.level}';out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Do not overwrite a prior experiment')
    native=NativeConfig();backend=HyperelasticBackend(native,ws);refiner=RefinementBackend(native,ws)
    start=time.perf_counter();u=None
    if args.level==0:
        mesh=TetMesh.from_volume(Volume(data.structure.labels,spacing=data.structure.spacing_zyx_m,origin=data.structure.origin_zyx_m),solid_phases=[1,2],separate_point_contacts=True)
    else:
        with h5py.File(root/f'level{args.level-1}'/'fields.h5') as f:
            previous=TetMesh(f['points_m'][:],f['cells'][:],f['phases'][:],f['box'][:],f['voxel_ids'][:])
            marked=mark_bulk(f['indicator/total_J'][:],args.fraction)
            if not len(marked):raise ValueError('Zero marking; inspect indicator before continuation')
            refined=refiner.refine(previous,marked);mesh=refined.mesh
            u=refined.nodal_field(f['nodal_displacement_m'][:])
        del previous,refined
    if len(mesh.cells)>1500000:raise ValueError('Adaptive reference exceeds the declared 1.5M-cell budget')
    scalar=CachedP1ScalarBackend(mesh,MFEMScalarBackend(native,ws))
    young=np.where(mesh.phases==1,cfg['active_young_Pa'],cfg['cbd_young_Pa'])
    np.random.seed(104)
    mechanics=Hyperelasticity(mesh,backend,young,cfg['poisson'],boundary='kubc',linear_solver='matrix-free-amg-minres')
    damage=AT2Damage(mesh,scalar,np.where(mesh.phases==1,cfg['active_Gc_J_m2'],cfg['cbd_Gc_J_m2']),cfg['length_m'],linear_solver='jacobi-cg')
    active=mesh.phases==1;beta=np.zeros((len(mesh.cells),3,3));table=data.rotated_tensor_table(np.diag(cfg['chemical_strain_per_stoichiometry']))
    beta[active]=table[data.grain_ids.ravel()[mesh.voxel_ids[active]]-1]
    ev,R=np.linalg.eigh(beta);Fc=np.einsum('eik,ek,ejk->eij',R,np.exp(-.2*ev),R)
    # Fresh zero history and damage on each mesh: do not inherit a coarse-mesh
    # irreversible overestimate as the lower bound for spatial convergence.
    result=finite_staggered_step(mechanics,damage,np.zeros(len(mesh.points)),FatigueHistory.zero(len(mesh.cells)),Fc,
        fatigue=False,tolerance=1e-7,initial_displacement=u)
    state=result.mechanics;d=result.damage
    indicators=coupled_indicator(mesh,state.first_piola_Pa,d,state.tensile_energy_J_m3,young,damage.Gc,damage.ell,damage.k)
    denom=np.bincount(mesh.voxel_ids,weights=mesh.volumes,minlength=data.structure.labels.size)
    def average(values):
        array=np.asarray(values).reshape(len(mesh.cells),-1);ans=np.zeros((len(denom),array.shape[1]))
        for j in range(array.shape[1]):
            ans[:,j]=np.divide(np.bincount(mesh.voxel_ids,weights=mesh.volumes*array[:,j],minlength=len(denom)),denom,out=np.zeros_like(denom),where=denom>0)
        return ans.reshape(*data.structure.labels.shape,*np.asarray(values).shape[1:])
    fracture=damage.energies(d,state.tensile_energy_J_m3)[1]
    record={'refinement':args.level,'fraction':args.fraction,'geometry_fingerprint':data.fingerprint,
        'length_m':cfg['length_m'],'mechanical_boundary':'KUBC F=I','chemical_log_stretch_change':-.2,
        'separate_point_contacts':True,'fatigue_enabled':False,'tetrahedra':len(mesh.cells),'points':len(mesh.points),
        'seconds':time.perf_counter()-start,'equilibrium_residual':state.equilibrium_residual,'damage_kkt':result.damage_kkt,
        'elastic_energy_J':state.elastic_energy_J,'fracture_energy_J':fracture,'max_damage':float(d.max()),
        'indicator_energy_J':float(indicators['total_J'].sum()),'indicator_is_certified_bound':False,
        'scope':'Conforming MFEM refinement, same geometry/materials/ell. Static spatial study, not lifetime validation.'}
    with h5py.File(out/'fields.h5','w') as f:
        f.attrs['record']=json.dumps(record)
        for key,val in {'points_m':mesh.points,'cells':mesh.cells,'phases':mesh.phases,'box':mesh.box,'voxel_ids':mesh.voxel_ids,
                        'nodal_damage':d,'nodal_displacement_m':state.displacement_m,
                        'common_voxel/damage':average(d[mesh.cells].mean(axis=1)),
                        'common_voxel/stress_Pa':average(state.cauchy_stress_Pa.reshape(-1,9)),
                        **{'indicator/'+k:v for k,v in indicators.items()}}.items():f.create_dataset(key,data=val,compression='gzip')
    (out/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
