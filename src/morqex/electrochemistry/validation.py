"""Protocol completion is checked independently of the solver return code."""
import numpy as np


def check_protocol(solution, *, cycles, steps_per_cycle, lower_V, upper_V, current_A,rest_s=600.):
    if len(solution.cycles)!=cycles:raise RuntimeError('Incomplete requested cycle count')
    records=[];continuity=0.
    previous=None
    for i,cycle in enumerate(solution.cycles):
        if cycle is None or len(cycle.steps)!=steps_per_cycle:raise RuntimeError('Incomplete cycle steps')
        row=[]
        for j,step in enumerate(cycle.steps):
            if step is None:raise RuntimeError('Skipped mandatory protocol step')
            if 'MORQEX' in step.termination:raise RuntimeError(f'Physical applicability event: {step.termination}')
            v=float(step['Voltage [V]'].entries[-1]);current=float(step['Current [A]'].entries[-1])
            duration=float(step.t[-1]-step.t[0])
            if j==0 and abs(v-lower_V)>1e-4:raise RuntimeError('Discharge did not reach requested voltage')
            if steps_per_cycle==5:
                if j==2 and abs(v-upper_V)>1e-4:raise RuntimeError('CC charge did not reach upper voltage')
                if j==3 and abs(abs(current)-current_A/20)>1e-4:raise RuntimeError('CV charge did not reach C/20')
                if j in [1,4] and abs(duration-rest_s)>1e-3:raise RuntimeError('Incomplete rest')
            for variable in ['Negative particle concentration [mol.m-3]','Positive particle concentration [mol.m-3]','Volume-averaged cell temperature [K]']:
                start=np.asarray(step[variable].entries)[...,0]
                if previous is not None:
                    end=np.asarray(previous[variable].entries)[...,-1]
                    continuity=max(continuity,float(np.max(abs(start-end))/max(1.,np.max(abs(end)))))
            row.append({'step':j,'duration_s':duration,'termination':step.termination,'voltage_end_V':v,'current_end_A':current})
            previous=step
        records.append(row)
    if continuity>1e-8:raise RuntimeError(f'Differential-state discontinuity: {continuity}')
    return {'completed':True,'cycles':records,'relative_step_state_jump':continuity}
