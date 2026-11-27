"""Accepted graph-update sequences versus fresh full nonlinear graphs.

Prescribed chemistry/contact sequences, NOT integrated diffusion or fatigue
cycles. Local BV source errors, charge balance and topology events are checked.
"""
from dataclasses import replace
import json,os,time
import numpy as np
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.top_damage.reaction_network import voxel_reaction_problem,ReactiveDualNetwork
from morqex.top_damage.reaction_cache import ReactionNetworkCache
from morqex.nodewise.halfcell import lithium_halfcell_parameters
from study_topology_counterexample import geometry
from study_reactive_graph import value


def main():
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);target=ws.path('reports/reactive_event_sequences_v1.json')
    if target.exists():raise ValueError('Version rather than overwrite accepted experiments')
    rows=[];labels,particles=geometry()
    for variant in ('single_contact','second_bridge','larger_second_particle'):
        phase=labels.copy();ids=particles.copy()
        if variant=='second_bridge':phase[0:2,3,2]=2
        if variant=='larger_second_particle':phase[2:5,2:4,6:9]=1;ids[2:5,2:4,6:9]=2
        fv=VoxelDiffusion(Volume(phase,spacing=(1e-6,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0],particle_ids=ids)
        for host in ('NMC811','NMC532','LFP'):
            p,_=lithium_halfcell_parameters(host=host,active_fraction=float(np.mean(phase==1)))
            cmax=float(p['Maximum concentration in positive electrode [mol.m-3]']);T=298.15
            U=value(p['Positive electrode OCP [V]'],.5)
            j0=value(p['Positive electrode exchange-current density [A.m-2]'],1000.,.5*cmax,cmax,T)
            kappa=value(p['Electrolyte conductivity [S.m-1]'],1000.,T)
            initial,layout,_,selected=voxel_reaction_problem(fv,
                np.where(phase==1,10.,np.where(phase==2,100.,0.)),kappa*(phase==0),np.full(len(fv.voxel_ids),U),j0)
            contact=np.sort(np.searchsorted(layout.voxel_ids,np.ravel_multi_index(np.array([[1,2,1],[2,2,1]]).T,phase.shape)))
            cut=np.flatnonzero(np.all(initial.electronic.edges==contact,axis=1))
            if len(cut)!=1:raise ValueError('The prescribed physical face must be an actual unique contact')
            for rate in (.1,2.,8.):
                cache=ReactionNetworkCache(relative_current_tolerance=.01,extended_precision=True)
                base_I=rate/3600*96485.33212*cmax*float(fv.volumes.sum())*.65
                for step in range(12):
                    e=initial.electronic;i=initial.ionic
                    conductance=e.conductance_S*(1-min(step,3)*1e-5)
                    if 6<=step<11:conductance[cut]=0.
                    e=replace(e,conductance_S=conductance)
                    if step>=8:i=replace(i,conductance_S=i.conductance_S*.7)
                    theta=.5+.015*np.sin(step*.4)+.0005*np.cos(np.arange(len(fv.voxel_ids))*.47)
                    temperature=T+.1*step
                    ocp=np.array([value(p['Positive electrode OCP [V]'],c) for c in theta])
                    exchange=np.array([value(p['Positive electrode exchange-current density [A.m-2]'],1000.,c*cmax,cmax,temperature) for c in theta])
                    cell=fv.face_cells[selected];j=exchange[cell]
                    if step==9:j[cell<4]=0.
                    faces=replace(initial.faces,equilibrium_V=ocp[cell],exchange_A_m2=j,
                        provenance='Explicit synthetic chemical-state path and physical contact-event verification')
                    current=base_I*(1 if step<5 else -1)
                    previous=None if cache.accepted is None else cache.accepted.state
                    tick=time.perf_counter()
                    accepted=cache.evaluate(e,i,faces,temperature_K=temperature,discharge_current_A=current)
                    cache_seconds=time.perf_counter()-tick
                    tick=time.perf_counter()
                    full=ReactiveDualNetwork(e,i,faces,electronic_collector=int(e.anchors[0]),
                        ionic_reservoir=int(i.anchors[0]),temperature_K=temperature,reduce=False,max_ports=512)
                    result=full.solve(current,initial=None if previous is None else full.initial_from(previous),extended_precision=True)
                    full_seconds=time.perf_counter()-tick
                    voltage_error=abs(result.voltage_V-accepted.state.voltage_V)
                    flux_errors=abs(result.reaction_current_A-accepted.state.reaction_current_A)
                    bound=accepted.certificate
                    current_scale=max(abs(current),float(np.sum(abs(accepted.state.reaction_current_A))),1e-30)
                    # Small slack covers floating-point evaluation, not physical
                    # model error or a relaxed nonlinear solve.
                    slack=1e-10*current_scale
                    checks={
                        'voltage_bound_contains_full':voltage_error<=bound.voltage_absolute_bound_V+1e-11,
                        'every_face_bound_contains_full':bool(np.all(flux_errors<=bound.reaction_current_absolute_bounds_A+slack)),
                        'voltage_below_1mV':voltage_error<=.001,
                        'reaction_L1_below_1percent':float(np.sum(flux_errors))/current_scale<=.01,
                        'charge_balance':max(result.relative_charge_residual,accepted.state.relative_charge_residual)<=1e-9,
                        'mandatory_support_rebuild':accepted.rebuilt if step in (0,6,9,10,11) else True}
                    row={'variant':variant,'host':host,'rate_C':rate,'state_index':step,'rebuilt':accepted.rebuilt,
                        'reason':accepted.reason,'voltage_error_V':voltage_error,'voltage_bound_V':bound.voltage_absolute_bound_V,
                        'relative_face_error_L1':float(np.sum(flux_errors))/current_scale,
                        'relative_face_current_bound':accepted.relative_face_current_bound,
                        'cache_update_seconds':cache_seconds,'fresh_full_seconds':full_seconds,
                        'charge_residual':accepted.state.relative_charge_residual,'checks':checks}
                    rows.append(row)
                    if not all(checks.values()):
                        print(json.dumps(row),flush=True);raise ArithmeticError('Reactive update sequence failed')
                print(json.dumps({'variant':variant,'host':host,'rate_C':rate,'rebuilds':cache.rebuilds,'reuses':cache.reuses}),flush=True)
    result={'passed':True,'records':rows,
            'scope':'324 supplied-state graph queries; 3 synthetic geometries, 3 chemistry interfaces, 0.1/2/8C. No integrated diffusion, automatic phase-field topology, cycle-jump, calibrated lifetime or interval arithmetic certification.'}
    target.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
