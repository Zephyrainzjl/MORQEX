"""Changed OCP/kinetics/current voltage certificates versus fresh full graphs.

These are supplied-state ohmic/BV snapshots, not transient DFN validation or
fatigue/wetting predictions. Different chemistry parameter interfaces are used.
"""
from dataclasses import replace
import argparse,json,os,time
import numpy as np
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.top_damage.reaction_network import voxel_reaction_problem,ReactiveDualNetwork
from morqex.top_damage.contracts import ConductingNetwork
from morqex.nodewise.halfcell import lithium_halfcell_parameters
from study_topology_counterexample import geometry
from study_reactive_graph import value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--with-rebinding',action='store_true');args=parser.parse_args()
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);target=ws.path('reports/reactive_state_voltage_bounds_v2.json' if args.with_rebinding else 'reports/reactive_state_voltage_bounds_v1.json')
    if target.exists():raise ValueError('Version the benchmark instead of overwriting evidence')
    labels,particles=geometry();fv=VoxelDiffusion(Volume(labels,spacing=(1e-6,)*3),
        active_phases=[1],diffusivity=1.,reactive_phases=[0],particle_ids=particles)
    rows=[]
    for host in ('NMC811','NMC532','LFP'):
        p,_=lithium_halfcell_parameters(host=host,active_fraction=float(np.mean(labels==1)))
        cmax=float(p['Maximum concentration in positive electrode [mol.m-3]']);T=298.15
        U=value(p['Positive electrode OCP [V]'],.5)
        j0=value(p['Positive electrode exchange-current density [A.m-2]'],1000.,.5*cmax,cmax,T)
        kappa=value(p['Electrolyte conductivity [S.m-1]'],1000.,T)
        for topology in ('intact','isolated'):
            factors=np.ones((3,)+labels.shape)
            if topology=='isolated':factors[0,1,2,1]=0.
            old,_,_,selected=voxel_reaction_problem(fv,np.where(labels==1,10.,np.where(labels==2,100.,0.)),
                kappa*(labels==0),np.full(len(fv.voxel_ids),U),j0,electronic_face_factors=factors)
            for rate in (.5,8.):
                I=rate/3600*96485.33212*cmax*float(fv.volumes.sum())*.65
                cached=old.solve(I,extended_precision=True)
                for variation in ('small','reversal'):
                    small=variation=='small';epsilon=.0001 if small else .05
                    theta=.5+(.00001 if small else .005)*np.cos(np.arange(len(fv.voxel_ids))*.47)
                    temperature=T+(.001 if small else 10.)
                    target_current=I*(.999 if small else -1.)
                    target_U=np.array([value(p['Positive electrode OCP [V]'],c) for c in theta])
                    target_j=np.array([value(p['Positive electrode exchange-current density [A.m-2]'],1000.,c*cmax,cmax,temperature) for c in theta])
                    cell=fv.face_cells[selected]
                    faces=replace(old.faces,equilibrium_V=target_U[cell],exchange_A_m2=target_j[cell],
                        area_m2=old.faces.area_m2*(.9999 if small else .97),
                        provenance='Explicit controlled reaction coverage and supplied concentration/temperature')
                    graphs=[]
                    for network,shift in ((old.electronic,0),(old.ionic,2)):
                        multiplier=1+epsilon*np.cos(np.arange(len(network.edges))*.37+shift)
                        graphs.append(ConductingNetwork(network.vertex_count,network.edges,network.conductance_S*multiplier,
                            network.anchors,network.channel,'Deterministic support-preserving perturbation'))
                    start=time.perf_counter()
                    bound=old.changed_problem_bound(cached,*graphs,faces=faces,temperature_K=temperature,discharge_current_A=target_current)
                    certificate_seconds=time.perf_counter()-start
                    start=time.perf_counter()
                    target_model=ReactiveDualNetwork(*graphs,faces,electronic_collector=int(graphs[0].anchors[0]),
                        ionic_reservoir=int(graphs[1].anchors[0]),temperature_K=temperature,reduce=False,max_ports=512)
                    # Both methods start from the same accepted full potentials.
                    # Shift to the target model's OCP reference, not its old one.
                    initial=target_model.initial_from(cached)
                    trace=[]
                    try:full=target_model.solve(target_current,initial=initial,diagnostics=trace,extended_precision=True)
                    except RuntimeError:
                        print(json.dumps({'failed_case':[host,topology,rate,variation],'trace':trace}),flush=True)
                        raise
                    full_seconds=time.perf_counter()-start
                    error=abs(cached.voltage_V-full.voltage_V)
                    if error>bound.voltage_absolute_bound_V+1e-11:raise ArithmeticError('Changed-state voltage certificate failed')
                    reuse={}
                    if args.with_rebinding:
                        start=time.perf_counter();rebound=old.with_reaction_state(faces,temperature_K=temperature)
                        solved=rebound.solve(target_current,initial=rebound.initial_from(cached),extended_precision=True)
                        rebind_seconds=time.perf_counter()-start
                        start=time.perf_counter();rebound_bound=rebound.changed_conductance_bound(solved,*graphs)
                        rebind_bound_seconds=time.perf_counter()-start
                        rebind_error=abs(solved.voltage_V-full.voltage_V)
                        if rebind_error>rebound_bound.voltage_absolute_bound_V+1e-11:raise ArithmeticError('Rebound voltage bound failed')
                        if abs(float(solved.reaction_current_A.sum())+target_current)>1e-9*max(abs(target_current),float(np.sum(abs(solved.reaction_current_A)))):
                            raise ArithmeticError('Rebound reaction currents violate imposed charge')
                        reuse={'cached_passive_rebound_seconds':rebind_seconds,'rebound_certificate_seconds':rebind_bound_seconds,
                            'rebound_voltage_error_V':rebind_error,'rebound_voltage_bound_V':rebound_bound.voltage_absolute_bound_V,
                            'rebound_charge_residual':solved.relative_charge_residual,
                            'rebound_below_1mV_event_tolerance':rebound_bound.voltage_absolute_bound_V<=.001}
                    rows.append({'host':host,'topology':topology,'rate_C':rate,'variation':variation,
                        'temperature_K':temperature,'target_current_A':target_current,'actual_voltage_error_V':error,
                        'voltage_bound_V':bound.voltage_absolute_bound_V,'bound_to_actual_ratio':bound.voltage_absolute_bound_V/max(error,1e-30),
                        'certificate_seconds':certificate_seconds,'fresh_full_build_and_solve_seconds':full_seconds,
                        'full_initial_guess':'same accepted full potentials, shifted to target OCP gauge',
                        'potential_precision':full.potential_precision,'tangent_precision':'float64',
                        'potential_significand_bits':int(np.finfo(np.longdouble).nmant+1),
                        'below_1mV_event_tolerance':bound.voltage_absolute_bound_V<=.001,
                        'full_charge_residual':full.relative_charge_residual,'full_power_defect_W':full.power_defect_W,**reuse})
                    print(json.dumps(rows[-1]),flush=True)
    result={'passed':True,'records':rows,'scope':'Known target OCP, exchange current, reaction coverage, T and I; fixed positive graph/reaction support. Ohmic/BV snapshots, not electrolyte diffusion, lifetime, automatic crack wetting or interval-arithmetic certification.'}
    target.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
