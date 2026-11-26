import os,json
import numpy as np
import pybamm as pb
from morqex import Workspace
from morqex.top_damage.epoch_dfn import EpochDFN,DFNFeedback
ws=Workspace(os.environ['MORQEX_WORKSPACE'])
rt=EpochDFN(active_fraction=.420379638671875,porosity=.349609375,radius_m=3.725e-6)
inputs=DFNFeedback(10.,.15).inputs(.5*rt.capacity_Ah)
s=rt.simulation.step(540,inputs=inputs,save=False,t_interp=np.linspace(0,540,10))
out={'F':float(pb.constants.F.value),'analytical_initial':rt.initial_inventory,'capacity_Ah':rt.capacity_Ah}
for k in ['Total lithium [mol]','Total lithium in electrolyte [mol]','Total lithium in positive electrode [mol]','Discharge capacity [A.h]']:
    out[k]=np.asarray(s[k].entries).tolist()
for key in rt.model.variables:
    if 'flux' in key.lower() and 'electrolyte' in key.lower():
        a=np.asarray(s[key].entries);out[key]={'shape':a.shape,'left':a[0].tolist(),'right':a[-1].tolist()}
print(json.dumps(out,indent=2));ws.path('reports/top_damage_inventory_diagnosis.json').write_text(json.dumps(out,indent=2))
