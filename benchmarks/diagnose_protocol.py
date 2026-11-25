"""Record physical endpoint diagnostics without changing equations or events."""
import json
import hashlib
import os
from pathlib import Path
import pybamm as pb
import numpy as np
import study_degradation as engine

original=pb.Simulation.step
def diagnosed(self,*args,**kwargs):
    step=original(self,*args,**kwargs)
    if 'MORQEX' in step.termination:
        record={'termination':step.termination,'time_s':float(step.t[-1]),'voltage_V':float(step['Voltage [V]'].entries[-1]),'regions':{}}
        for d in ['negative','positive']:
            cmax=float(self.parameter_values[f'Maximum concentration in {d} electrode [mol.m-3]'])
            k=0
            while engine.key(d,k,'concentration [mol.m-3]') in step.all_models[0].variables:
                concentration=step[engine.key(d,k,'concentration [mol.m-3]')].entries[:,-1]/cmax
                record['regions'][f'{d}_{k}']={'stoichiometry_by_X':concentration.tolist(),'reaction_current_A_m2':step[engine.key(d,k,'reaction current [A.m-2]')].entries[:,-1].tolist()};k+=1
            record[d+'_damage']=step[engine.age(d,'damage')].entries[:,-1].tolist()
        record.update(source_sha256=engine.hashes()|{str(Path(__file__).relative_to(engine.ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},backend=engine.provenance())
        path=Path(os.environ['MORQEX_WORKSPACE'])/'reports/protocol_diagnosis.json';path.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    return step

if __name__=='__main__':pb.Simulation.step=diagnosed;engine.main()
