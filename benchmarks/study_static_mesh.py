"""Same physical voxel geometry, refined Tet4 discretization, static AT2 check.

Replicating labels refines the mesh without changing the staircase boundary,
material/grain interfaces, physical loading or phase-field length. This is NOT
convergence to a smooth CT surface, cyclic fatigue or electrolyte convergence.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np,h5py
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.mechanics import TetMesh
from morqex.top_damage.polycrystal import PolycrystalData
from morqex.top_damage.backend import NativeConfig,MFEMScalarBackend
from morqex.top_damage.assembly_cache import CachedP1ScalarBackend
from morqex.top_damage.hyperelastic import HyperelasticBackend,Hyperelasticity,finite_staggered_step
from morqex.top_damage.phase_field import AT2Damage,FatigueHistory
from morqex.top_damage.dense_cycle import _voxel_average


def repeat(a,k):
    for axis in range(3):a=np.repeat(a,k,axis=axis)
    return a


def main():
    p=argparse.ArgumentParser();p.add_argument('--refinement',type=int,choices=[1,2,3,4,5,6],required=True)
    p.add_argument('--study',choices=['static_mesh_v1','static_mesh_storage_v2','static_mesh_face_contacts_v3','static_mesh_quadratic_v1','static_mesh_quadratic_storage_v2','static_mesh_quadratic_q7_v1'],default='static_mesh_v1');a=p.parse_args()
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);base=ws.path('results/top_damage/network_cycles/dual_long20_checked_v1')
    data=PolycrystalData.load_hdf5(base/'initial.h5');cfg=json.loads((base/'manifest.json').read_text())['config']
    k=a.refinement;out=ws.path('results/top_damage',a.study,f'level{k}');out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Version the study before overwriting')
    start=time.perf_counter();labels=repeat(data.structure.labels,k);grains=repeat(data.grain_ids,k)
    h=np.asarray(data.structure.spacing_zyx_m)/k
    quadratic=a.study.startswith('static_mesh_quadratic_')
    quadrature_order=7 if a.study=='static_mesh_quadratic_q7_v1' else 5
    separate=a.study=='static_mesh_face_contacts_v3' or quadratic
    mesh=TetMesh.from_volume(Volume(labels,spacing=h,origin=data.structure.origin_zyx_m),solid_phases=[1,2],separate_point_contacts=separate)
    native=NativeConfig();backend=HyperelasticBackend(native,ws)
    scalar=CachedP1ScalarBackend(mesh,MFEMScalarBackend(native,ws))
    young=np.where(mesh.phases==1,cfg['active_young_Pa'],cfg['cbd_young_Pa'])
    np.random.seed(104)
    gc=np.where(mesh.phases==1,cfg['active_Gc_J_m2'],cfg['cbd_Gc_J_m2'])
    if quadratic:
        from morqex.top_damage.refinement import RefinementBackend
        from morqex.top_damage.quadratic_reference import QuadraticGeometry,QuadraticHyperelasticity,QuadratureAT2,quadratic_staggered_step
        refiner=RefinementBackend(native,ws);geometry=QuadraticGeometry(mesh,refiner,order=quadrature_order)
        mechanics=QuadraticHyperelasticity(geometry,backend,refiner,young,cfg['poisson'])
        damage=QuadratureAT2(geometry,scalar,gc,cfg['length_m'])
    else:
        mechanics=Hyperelasticity(mesh,backend,young,cfg['poisson'],boundary='kubc',linear_solver='matrix-free-amg-minres')
        damage=AT2Damage(mesh,scalar,gc,cfg['length_m'],linear_solver='jacobi-cg')
    active=mesh.phases==1;beta=np.zeros((len(mesh.cells),3,3));table=data.rotated_tensor_table(np.diag(cfg['chemical_strain_per_stoichiometry']))
    beta[active]=table[grains.ravel()[mesh.voxel_ids[active]]-1]
    ev,R=np.linalg.eigh(beta);Fc=np.einsum('eik,ek,ejk->eij',R,np.exp(-.2*ev),R)
    if quadratic:
        result=quadratic_staggered_step(mechanics,damage,np.zeros(len(mesh.points)),FatigueHistory.zero(geometry.quadrature_count),geometry.cell_to_quadrature(Fc),fatigue=False,tolerance=1e-7)
    else:
        result=finite_staggered_step(mechanics,damage,np.zeros(len(mesh.points)),FatigueHistory.zero(len(mesh.cells)),Fc,
            fatigue=False,tolerance=1e-7)
    d=result.damage;state=result.mechanics
    stress=geometry.quadrature_average(state.cauchy_stress_Pa) if quadratic else state.cauchy_stress_Pa
    fields={'damage':_voxel_average(mesh,d[mesh.cells].mean(axis=1),labels.shape),
        'stress_Pa':_voxel_average(mesh,stress.reshape(-1,9),labels.shape)}
    def average(x):
        nz,ny,nx=data.structure.labels.shape
        return x.reshape(nz,k,ny,k,nx,k,*x.shape[3:]).mean(axis=(1,3,5))
    energies=damage.energies(d,state.tensile_energy_J_m3)
    record={'refinement':k,'geometry_fingerprint':data.fingerprint,'shape':list(labels.shape),'spacing_m':h.tolist(),
        'length_m':cfg['length_m'],'mechanical_boundary':'KUBC F=I','chemical_log_stretch_change':-.2,
        'separate_point_contacts':separate,'solid_components':int(mesh.n_components),
        'displacement_degree':2 if quadratic else 1,'damage_degree':1,'mechanical_points':len(mechanics.mesh.points),
        'quadrature_order':quadrature_order if quadratic else None,
        'fatigue_enabled':False,'tetrahedra':len(mesh.cells),'points':len(mesh.points),'seconds':time.perf_counter()-start,
        'equilibrium_residual':state.equilibrium_residual,'damage_kkt':result.damage_kkt,'stagger_iterations':result.stagger_iterations,
        'elastic_energy_J':state.elastic_energy_J,'fracture_energy_J':energies[1],'max_damage':float(d.max()),
        'scope':'Static fixed-staircase-geometry and fixed-ell convergence; no cyclic-fatigue or geometric convergence claim'}
    with h5py.File(out/'fields.h5','w') as f:
        f.attrs['record']=json.dumps(record);f.create_dataset('points_m',data=mesh.points,compression='gzip');f.create_dataset('cells',data=mesh.cells,compression='gzip')
        f.create_dataset('nodal_damage',data=d,compression='gzip');f.create_dataset('nodal_displacement_m',data=state.displacement_m,compression='gzip')
        if quadratic:
            f.create_dataset('mechanics_points_m',data=geometry.points,compression='gzip')
            f.create_dataset('mechanics_element_nodes',data=geometry.element_nodes,compression='gzip')
        for key,val in fields.items():f.create_dataset('voxel/'+key,data=val,compression='gzip');f.create_dataset('common_voxel/'+key,data=average(val),compression='gzip')
    (out/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':main()
