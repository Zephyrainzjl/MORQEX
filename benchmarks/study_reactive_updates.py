"""Independent fresh-full-graph validation of nonlinear voltage event bounds."""
import json,os,time
import numpy as np
import pybamm as pb
from morqex.workspace import Workspace
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.top_damage.reaction_network import voxel_reaction_problem,ReactiveDualNetwork
from morqex.top_damage.contracts import ConductingNetwork
from morqex.nodewise.halfcell import lithium_halfcell_parameters
from study_topology_counterexample import geometry
from study_reactive_graph import value


def main():
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);labels,particles=geometry();h=1e-6
    fv=VoxelDiffusion(Volume(labels,spacing=(h,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0],particle_ids=particles)
    p,_=lithium_halfcell_parameters(host='NMC811',active_fraction=float(np.mean(labels==1)))
    cmax=float(p['Maximum concentration in positive electrode [mol.m-3]']);T=298.15
    U=value(p['Positive electrode OCP [V]'],.5)
    j0=value(p['Positive electrode exchange-current density [A.m-2]'],1000.,.5*cmax,cmax,T)
    kappa=value(p['Electrolyte conductivity [S.m-1]'],1000.,T)
    electronic=np.where(labels==1,10.,np.where(labels==2,100.,0.));ionic=kappa*(labels==0)
    records=[]
    for mode in ('intact','isolate'):
        faces=np.ones((3,)+labels.shape)
        if mode=='isolate':faces[0,1,2,1]=0.
        model,*_=voxel_reaction_problem(fv,electronic,ionic,np.full(len(fv.voxel_ids),U),j0,electronic_face_factors=faces)
        for rate in (.5,8.):
            I=rate/3600*96485.33212*cmax*np.sum(fv.volumes)*.65
            original=model.solve(I)
            for epsilon in (.001,.05,.4):
                print(json.dumps({'mode':mode,'rate_C':rate,'edge_change':epsilon}),flush=True)
                graphs=[]
                for graph,shift in [(model.electronic,0),(model.ionic,2)]:
                    factors=1+epsilon*np.cos(np.arange(len(graph.edges))*.37+shift)
                    graphs.append(ConductingNetwork(graph.vertex_count,graph.edges,graph.conductance_S*factors,
                        graph.anchors,graph.channel,'deterministic known edge perturbations'))
                start=time.perf_counter();bound=model.changed_conductance_bound(original,*graphs);estimate_s=time.perf_counter()-start
                start=time.perf_counter()
                trace=[]
                try:
                    fresh=ReactiveDualNetwork(*graphs,model.faces,electronic_collector=int(model.electronic.anchors[0]),
                        ionic_reservoir=int(model.ionic.anchors[0]),reduce=False,max_ports=512).solve(I,diagnostics=trace)
                except RuntimeError:
                    print(json.dumps({'failed_trace':trace[-15:]}),flush=True);raise
                full_s=time.perf_counter()-start
                error=abs(fresh.voltage_V-original.voltage_V)
                if error>bound.voltage_absolute_bound_V+1e-11:raise ArithmeticError('Voltage bound failed independent full-graph comparison')
                records.append(dict(mode=mode,rate_C=rate,relative_edge_change=bound.relative_edge_change,
                    actual_voltage_error_V=error,voltage_bound_V=bound.voltage_absolute_bound_V,
                    bound_to_actual_ratio=bound.voltage_absolute_bound_V/max(error,1e-30),
                    bound_seconds=estimate_s,fresh_full_build_and_solve_seconds=full_s,
                    below_1mV_event_tolerance=bound.voltage_absolute_bound_V<=.001))
    result={'passed':True,'records':records,'scope':'Same fixed concentrations/kinetics/current; edge support unchanged. FP64 evaluation of analytical bound; not interval arithmetic or accumulated cycling certification.'}
    target=ws.path('reports/reactive_voltage_bounds.json')
    if target.exists():raise ValueError('Do not overwrite accepted benchmark; version first')
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
