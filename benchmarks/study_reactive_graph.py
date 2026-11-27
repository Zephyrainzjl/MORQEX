"""Resolved working potentials and nonlinear reaction-port Schur verification."""
import json,os,time
import h5py,numpy as np
import pybamm as pb
from morqex.workspace import Workspace
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.top_damage.reaction_network import voxel_reaction_problem
from morqex.nodewise.halfcell import lithium_halfcell_parameters
from study_topology_counterexample import geometry


def value(function,*args):
    a=function(*map(pb.Scalar,args)) if callable(function) else function
    return float(np.asarray(a.evaluate() if isinstance(a,pb.Symbol) else a).ravel()[0])


def main():
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);out=ws.path('results/top_damage/reactive_graph/v1');out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Version the benchmark before overwriting')
    labels,particles=geometry();h=1e-6
    fv=VoxelDiffusion(Volume(labels,spacing=(h,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0],particle_ids=particles)
    p,_=lithium_halfcell_parameters(host='NMC811',active_fraction=float(np.mean(labels==1)))
    cmax=float(p['Maximum concentration in positive electrode [mol.m-3]']);T=298.15
    U=value(p['Positive electrode OCP [V]'],.5)
    j0=value(p['Positive electrode exchange-current density [A.m-2]'],1000.,.5*cmax,cmax,T)
    kappa=value(p['Electrolyte conductivity [S.m-1]'],1000.,T)
    electronic=np.where(labels==1,10.,np.where(labels==2,100.,0.));ionic=kappa*(labels==0)
    results=[]
    for mode in ('intact','isolate'):
        faces=np.ones((3,)+labels.shape)
        if mode=='isolate':faces[0,1,2,1]=0.
        models=[];build=[]
        for reduced in (True,False):
            tick=time.perf_counter()
            models.append(voxel_reaction_problem(fv,electronic,ionic,np.full(len(fv.voxel_ids),U),j0,
                electronic_face_factors=faces,reduce=reduced,max_ports=512))
            build.append(time.perf_counter()-tick)
        for rate in (.5,8.):
            current=rate/3600*96485.33212*cmax*np.sum(fv.volumes)*.65
            states=[];cost=[]
            for problem,*_ in models:
                tick=time.perf_counter();states.append(problem.solve(current));cost.append(time.perf_counter()-tick)
            reduced,full=states;problem,e,i,selected=models[0]
            voltage_error=abs(reduced.voltage_V-full.voltage_V)
            reaction_error=float(np.linalg.norm(reduced.reaction_current_A-full.reaction_current_A)/np.linalg.norm(full.reaction_current_A))
            np.testing.assert_allclose(reduced.electronic_potential_V,full.electronic_potential_V,rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(reduced.ionic_potential_V,full.ionic_potential_V,rtol=1e-10,atol=1e-10)
            if voltage_error>1e-9 or reaction_error>1e-8:raise RuntimeError('Nonlinear port reduction disagrees with the full graph')
            total_by_particle=np.bincount(particles.ravel()[fv.voxel_ids[fv.face_cells[selected]]],weights=reduced.reaction_current_A,minlength=3)[1:]
            if mode=='isolate' and abs(total_by_particle[0])/current>1e-8:raise RuntimeError('Floating particle violates electronic charge conservation')
            with h5py.File(out/f'{mode}_{rate:g}C.h5','w') as f:
                f.attrs.update(scope='Actual quasi-static ohmic/BV potentials at fixed SOC. No electrolyte concentration gradients, diffusion integration or fatigue in this benchmark.',current_A=current,voltage_V=reduced.voltage_V,temperature_K=T)
                for key,layout,values in [('electronic_potential_V',e,reduced.electronic_potential_V),('ionic_potential_V',i,reduced.ionic_potential_V)]:
                    field=np.full(labels.size,np.nan);field[layout.voxel_ids]=values[:len(layout.voxel_ids)]
                    f.create_dataset(key,data=field.reshape(labels.shape))
                f.create_dataset('phase_labels',data=labels);f.create_dataset('particle_ids',data=particles)
                f.create_dataset('reaction_current_A',data=reduced.reaction_current_A)
                f.create_dataset('original_reaction_face_indices',data=selected)
            results.append({'mode':mode,'rate_C':rate,'current_A':current,'voltage_V':reduced.voltage_V,
                'voltage_difference_V':voltage_error,'reaction_relative_difference':reaction_error,
                'net_reaction_by_particle_A':total_by_particle.tolist(),'charge_relative_residual':reduced.relative_charge_residual,
                'power_defect_W':reduced.power_defect_W,'reaction_dissipation_W':reduced.reaction_dissipation_W,
                'reduced_build_s':build[0],'full_build_s':build[1],'reduced_solve_s':cost[0],'full_solve_s':cost[1],
                'retained_ports':[len(problem.eop.terminals),len(problem.iop.terminals)],
                'full_vertices':[e.network.vertex_count,i.network.vertex_count]})
    summary={'results':results,'scope':'Nonlinear reaction-port verification on prescribed dry-contact topology; not a DFN/fatigue transient or full concentrated-solution model.'}
    (out/'result.json').write_text(json.dumps(summary,indent=2));ws.path('reports/reactive_graph_v1.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
