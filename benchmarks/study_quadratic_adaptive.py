"""P2/P1 spatial candidate with conforming MFEM refinement and warm starts.

Starts from the existing 24^3 same-geometry reference, never a changed load or
phase-field length. Each level solves from zero accepted damage/history.
"""
import argparse,json,os,time
import h5py
import numpy as np
from morqex import Workspace
from morqex.mechanics import TetMesh
from morqex.microstructure import Volume
from morqex.top_damage.polycrystal import PolycrystalData
from morqex.top_damage.backend import NativeConfig,MFEMScalarBackend
from morqex.top_damage.assembly_cache import CachedP1ScalarBackend
from morqex.top_damage.hyperelastic import HyperelasticBackend
from morqex.top_damage.phase_field import FatigueHistory
from morqex.top_damage.refinement import RefinementBackend
from morqex.top_damage.refinement_indicator import quadratic_indicator,mark_bulk
from morqex.top_damage.quadratic_reference import QuadraticGeometry,QuadraticHyperelasticity,QuadratureAT2,quadratic_staggered_step,prolong_quadratic_field


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--level',type=int,required=True)
    studies=['static_mesh_quadratic_adaptive_v1','static_mesh_quadratic_adaptive_budget_v1']
    parser.add_argument('--study',choices=studies,default=studies[0])
    parser.add_argument('--parent-study',choices=studies)
    parser.add_argument('--fraction',type=float,default=.5)
    args=parser.parse_args()
    if args.level<0 or not 0<args.fraction<=1:raise ValueError('Nonnegative level and fraction in (0,1] required')
    ws=Workspace(os.environ['MORQEX_WORKSPACE'])
    initial=ws.path('results/top_damage/network_cycles/dual_long20_checked_v1')
    data=PolycrystalData.load_hdf5(initial/'initial.h5')
    cfg=json.loads((initial/'manifest.json').read_text())['config']
    root=ws.path('results/top_damage',args.study)
    parent_root=ws.path('results/top_damage',args.parent_study or args.study)
    out=root/f'level{args.level}';out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Do not overwrite an existing spatial experiment')
    start=time.perf_counter();native=NativeConfig();refiner=RefinementBackend(native,ws)
    source_record=None
    if args.level==0:
        # Preserve the uniform reference's exact cell order for its warm start.
        labels=data.structure.labels
        for axis in range(3):labels=np.repeat(labels,2,axis=axis)
        mesh=TetMesh.from_volume(Volume(labels,spacing=np.asarray(data.structure.spacing_zyx_m)/2,
            origin=data.structure.origin_zyx_m),solid_phases=[1,2],separate_point_contacts=True)
        original=np.array(np.unravel_index(mesh.voxel_ids,labels.shape)).T//2
        voxel_ids=np.ravel_multi_index(original.T,data.structure.labels.shape)
        mesh=TetMesh(mesh.points,mesh.cells,mesh.phases,mesh.box,voxel_ids)
        geometry=QuadraticGeometry(mesh,refiner)
        seed=ws.path('results/top_damage/static_mesh_quadratic_v1/level2')
        source_record=json.loads((seed/'result.json').read_text())
        if source_record['geometry_fingerprint']!=data.fingerprint:raise ValueError('Seed geometry mismatch')
        with h5py.File(seed/'fields.h5') as f:
            if not np.array_equal(f['mechanics_points_m'][:],geometry.points):raise ValueError('Seed node ordering mismatch')
            u=f['nodal_displacement_m'][:]
    else:
        with h5py.File(parent_root/f'level{args.level-1}/fields.h5') as f:
            old=TetMesh(f['points_m'][:],f['cells'][:],f['phases'][:],f['box'][:],f['voxel_ids'][:])
            old_geometry=QuadraticGeometry(old,refiner)
            marked=mark_bulk(f['indicator/total_J'][:],args.fraction)
            if not len(marked):raise ValueError('No nonzero refinement indicator')
            refined=refiner.refine(old,marked);mesh=refined.mesh
            if len(mesh.cells)>180000:raise ValueError(f'P2 study has {len(mesh.cells)} cells, exceeding its declared 180000-parent-cell budget')
            geometry=QuadraticGeometry(mesh,refiner)
            u=prolong_quadratic_field(old_geometry,geometry,refined.parent_cells,f['nodal_displacement_m'][:])
        del old,old_geometry,refined
    np.random.seed(104)
    E=np.where(mesh.phases==1,cfg['active_young_Pa'],cfg['cbd_young_Pa'])
    gc=np.where(mesh.phases==1,cfg['active_Gc_J_m2'],cfg['cbd_Gc_J_m2'])
    mechanics=QuadraticHyperelasticity(geometry,HyperelasticBackend(native,ws),refiner,E,cfg['poisson'])
    scalar=CachedP1ScalarBackend(mesh,MFEMScalarBackend(native,ws))
    damage=QuadratureAT2(geometry,scalar,gc,cfg['length_m'])
    beta=np.zeros((len(mesh.cells),3,3));active=mesh.phases==1
    table=data.rotated_tensor_table(np.diag(cfg['chemical_strain_per_stoichiometry']))
    beta[active]=table[data.grain_ids.ravel()[mesh.voxel_ids[active]]-1]
    eig,R=np.linalg.eigh(beta);Fc=np.einsum('eik,ek,ejk->eij',R,np.exp(-.2*eig),R)
    result=quadratic_staggered_step(mechanics,damage,np.zeros(len(mesh.points)),
        FatigueHistory.zero(geometry.quadrature_count),geometry.cell_to_quadrature(Fc),
        fatigue=False,tolerance=1e-7,initial_displacement=u)
    state=result.mechanics;d=result.damage
    indicator=quadratic_indicator(geometry,state.first_piola_Pa,d,state.tensile_energy_J_m3,E,gc,cfg['length_m'],damage.k)
    denominator=np.bincount(mesh.voxel_ids,weights=mesh.volumes,minlength=data.structure.labels.size)
    def average(values):
        a=np.asarray(values).reshape(len(mesh.cells),-1);answer=np.zeros((len(denominator),a.shape[1]))
        for j in range(a.shape[1]):
            answer[:,j]=np.divide(np.bincount(mesh.voxel_ids,weights=mesh.volumes*a[:,j],minlength=len(denominator)),
                denominator,out=np.zeros_like(denominator),where=denominator>0)
        return answer.reshape(*data.structure.labels.shape,*np.asarray(values).shape[1:])
    record={'refinement':args.level,'fraction':args.fraction,'parent_study':args.parent_study or args.study,'geometry_fingerprint':data.fingerprint,
        'length_m':cfg['length_m'],'mechanical_boundary':'KUBC F=I','chemical_log_stretch_change':-.2,
        'separate_point_contacts':True,'fatigue_enabled':False,'tetrahedra':len(mesh.cells),'points':len(mesh.points),
        'displacement_degree':2,'damage_degree':1,'mechanical_points':len(geometry.points),'quadrature_order':5,
        'seconds':time.perf_counter()-start,'equilibrium_residual':state.equilibrium_residual,
        'damage_kkt':result.damage_kkt,'stagger_iterations':result.stagger_iterations,
        'elastic_energy_J':state.elastic_energy_J,'fracture_energy_J':damage.energies(d,state.tensile_energy_J_m3)[1],
        'max_damage':float(d.max()),'indicator_energy_J':float(indicator['total_J'].sum()),
        'indicator_is_certified_bound':False,'seed_reference':source_record,
        'scope':'Same geometry and ell; heuristic P2 marking, static fresh damage/history, no lifetime qualification.'}
    with h5py.File(out/'fields.h5','w') as f:
        f.attrs['record']=json.dumps(record)
        values={'points_m':mesh.points,'cells':mesh.cells,'phases':mesh.phases,'box':mesh.box,'voxel_ids':mesh.voxel_ids,
            'mechanics_points_m':geometry.points,'mechanics_element_nodes':geometry.cells,
            'nodal_damage':d,'nodal_displacement_m':state.displacement_m,
            'common_voxel/damage':average(d[mesh.cells].mean(axis=1)),
            'common_voxel/stress_Pa':average(geometry.quadrature_average(state.cauchy_stress_Pa).reshape(-1,9)),
            **{'indicator/'+k:v for k,v in indicator.items()}}
        for key,value in values.items():f.create_dataset(key,data=value,compression='gzip')
    (out/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
