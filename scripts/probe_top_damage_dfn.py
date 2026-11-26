"""Check persistence across transport updates using the real bundled DFN."""
import json,os
from morqex import Workspace
from morqex.top_damage.epoch_dfn import EpochDFN,DFNFeedback
ws=Workspace(os.environ['MORQEX_WORKSPACE'])
runtime=EpochDFN(active_fraction=.420379638671875,porosity=.349609375,radius_m=3.725e-6)
records=[]
for i,current in enumerate([.5,.5,-.5,-.5]):
    records.append(runtime.advance(60.,current*runtime.capacity_Ah,DFNFeedback(10. if i<2 else 5.,.15)))
print(json.dumps(records,indent=2))
ws.path('reports/top_damage_dfn_port_probe.json').write_text(json.dumps(records,indent=2))
