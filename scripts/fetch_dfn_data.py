"""Fetch pinned public validation data to the external workspace with hashes."""
import hashlib,json,os
from pathlib import Path
import pybamm

loader=pybamm.DataLoader()
requested=['comsol_1C.json','0.5C_discharge_U.txt','1C_discharge_U.txt','2C_discharge_U.txt','0.5C_discharge_T.txt','1C_discharge_T.txt','2C_discharge_T.txt','1C_discharge_displacement.txt']
output=[]
for name in requested:
    path=Path(loader.get_data(name)).resolve()
    workspace=Path(os.environ['MORQEX_WORKSPACE']).resolve()
    if workspace not in path.parents:raise ValueError('Reference cache must be outside source in declared workspace')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    expected=loader.files[name].removeprefix('sha256:')
    if digest!=expected:raise ValueError('Reference checksum failed')
    output.append({'name':name,'path':str(path),'sha256':digest,'url':str(loader.registry.base_url).rstrip('/')+'/'+name,'kind':'independent COMSOL simulation' if name.startswith('comsol') else 'published Enertech experimental data; Ai2020/Rieger2016'})
    print(name,path.stat().st_size,flush=True)
Path(workspace/'reports/dfn_reference_data.json').write_text(json.dumps({'release':loader.version,'files':output},indent=2))
