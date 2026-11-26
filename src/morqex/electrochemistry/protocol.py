"""One compiled DAE for CC/rest/CV protocols; no change to battery physics.

Native FunctionControl supplies one algebraic current. Discrete control inputs
switch its equation. Only the dedicated protocol event can be stepped past;
physical concentration/voltage/porosity events always reject the protocol.
"""
import numpy as np
import pybamm as pb


def shared_control(v):
    mode=pb.InputParameter('Protocol voltage control')
    return (1-mode)*(v['Current [A]']-pb.InputParameter('Protocol current [A]'))/pb.Parameter('Nominal cell capacity [A.h]')+mode*(v['Voltage [V]']-pb.InputParameter('Protocol voltage [V]'))


def add_protocol_event(model):
    ve=pb.InputParameter('Protocol voltage event');ie=pb.InputParameter('Protocol current event')
    direction=pb.InputParameter('Protocol voltage direction')
    stop=ve*direction*(model.variables['Voltage [V]']-pb.InputParameter('Protocol voltage [V]'))+ie*(-model.variables['Current [A]']-pb.InputParameter('Protocol stop current [A]'))+(1-ve-ie)
    model.events.append(pb.Event('Protocol target [experiment]',stop))


def solve_protocol(simulation,*,kind,cycles=1,current_A=1.,lower_V=3.5,upper_V=4.,duration_s=120.,period_s=30.,rest_s=300.,cv_stop_A=.25,pulse_discharge_A=.5,on_segment=None,event_margin_V=1e-5,chunk_s=None,on_frame=None):
    from pybamm.solvers.solution import make_cycle_solution
    if kind not in {'short','cycles','pulse'} or current_A<=0 or duration_s<=0 or cycles<1:raise ValueError('Invalid shared protocol')
    if period_s<=0 or chunk_s is not None and chunk_s<period_s:raise ValueError('Positive output period and chunk at least as long as one output period required')
    if not 0<=event_margin_V<(upper_V-lower_V)/2:raise ValueError('Positive event margin smaller than half the voltage window required')
    requested=(lower_V,upper_V)
    # A protocol event coincident with a native voltage limit has undefined
    # root ordering. Stop just *inside* the physical limits, rather than remove
    # a physical event or relabel its termination to step past it.
    margins=[0.,0.]
    if kind=='cycles':
        physical=[float(simulation.parameter_values[name]) for name in
            ['Lower voltage cut-off [V]','Upper voltage cut-off [V]']]
        # Interior protocol targets do not need displacement. This preserves
        # equivalence with native CC/rest/CV experiments at those same targets.
        margins=[event_margin_V if abs(target-limit)<1e-12 else 0.
                 for target,limit in zip(requested,physical)]
        lower_V+=margins[0];upper_V-=margins[1]
    if kind=='cycles':
        sequence=[(0,current_A,lower_V,1,0,1,36000.),(0,0.,lower_V,0,0,1,rest_s),
            (0,-current_A,upper_V,1,0,-1,36000.),(1,0.,upper_V,0,1,1,36000.),(0,0.,upper_V,0,0,1,rest_s)]
    elif kind=='pulse':sequence=[(0,-current_A,upper_V,0,0,1,duration_s),(0,0.,upper_V,0,0,1,1200.),(0,pulse_discharge_A,lower_V,0,0,1,60.)]
    else:sequence=[(0,current_A,lower_V,0,0,1,duration_s)]
    previous=None;cycle_solutions=[];records=[]
    for cycle in range(cycles if kind=='cycles' else 1):
        steps=[]
        for mode,current,voltage,ve,ie,direction,duration in sequence:
            inputs={'Protocol voltage control':mode,'Protocol current [A]':current,'Protocol voltage [V]':voltage,
                'Protocol voltage event':ve,'Protocol current event':ie,'Protocol voltage direction':direction,'Protocol stop current [A]':cv_stop_A}
            remaining=duration;step=None
            while remaining>1e-8:
                dt=min(remaining,chunk_s or remaining)
                t_interp=np.unique(np.r_[np.arange(0,dt,period_s),dt])
                part=simulation.step(dt,starting_solution=previous,inputs=inputs,save=False,t_eval=t_interp,t_interp=t_interp)
                if part.termination not in {'final time','event: Protocol target [experiment]'}:raise RuntimeError('Incomplete shared protocol: '+str(part.termination))
                elapsed=float(part.t[-1]-part.t[0])
                if elapsed<1e-10:raise RuntimeError('Protocol chunk made no temporal progress')
                if on_frame is not None:on_frame(part,{'cycle':cycle+1,'duration_s':elapsed,'termination':part.termination,'voltage_control':bool(mode)})
                step=part if step is None else step+part;previous=part;remaining-=elapsed
                if part.termination!='final time':break
            expected='event: Protocol target [experiment]' if ve or ie else 'final time'
            if step.termination!=expected:raise RuntimeError('Incomplete shared protocol: '+str(step.termination))
            if not (ve or ie) and abs(step.t[-1]-step.t[0]-duration)>1e-3:raise RuntimeError('Incomplete timed protocol segment')
            steps.append(step);previous=step
            records.append({'cycle':cycle+1,'duration_s':float(step.t[-1]-step.t[0]),'termination':step.termination,'voltage_control':bool(mode)})
            if on_segment is not None:on_segment(step,records[-1])
        result,_,_=make_cycle_solution(steps,esoh_solver=None);cycle_solutions.append(result)
    result=cycle_solutions[0]
    for cycle in cycle_solutions[1:]:result=result+cycle
    result.cycles=cycle_solutions
    return result,{'driver':'single compiled input-switched algebraic CC/rest/CV control','segments':records,'compiled_protocol_models':1,
        'requested_voltage_limits_V':list(requested),'protocol_voltage_targets_V':[lower_V,upper_V],
        'event_margin_V':max(margins),'event_margins_V':margins,'output_period_s':period_s,'streaming_chunk_s':chunk_s}
