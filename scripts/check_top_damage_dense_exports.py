"""Independent XDMF reader and lossless VTI/HDF5 comparison on actual outputs."""
import base64,json,os,struct
import xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import meshio
import numpy as np
from morqex import Workspace

ws=Workspace(os.environ['MORQEX_WORKSPACE']);reports=[]
for name in ('gentle_voltage_v3','fast_voltage_v3','released_voltage_v3'):
    directory=ws.path('results/top_damage/dense_cycles',name)
    manifest=json.loads((directory/'export_manifest.json').read_text())
    with meshio.xdmf.TimeSeriesReader(directory/'mechanics_history.xdmf') as reader:
        points,cells=reader.read_points_cells()
        assert reader.num_steps==len(manifest['states'])
        for index in (0,reader.num_steps-1):
            t,point,cell=reader.read_data(index)
            state=manifest['states'][index]
            assert t==state['time_s']
            source=directory/f"step_{state['step']:05d}.h5"
            with h5py.File(source,'r') as f:
                np.testing.assert_array_equal(point['damage'],f['damage'][:])
                np.testing.assert_array_equal(point['displacement_m'],f['displacement_m'][:])
                np.testing.assert_array_equal(cell['fatigue_J_m3'][0],f['fatigue'][:])
                root=ET.parse(source.with_suffix('.vti')).getroot()
                for array in root.findall('./ImageData/Piece/CellData/DataArray'):
                    key=array.attrib['Name']
                    if key not in f['voxel']:continue
                    raw=base64.b64decode(array.text)
                    assert struct.unpack('<Q',raw[:8])[0]==len(raw)-8
                    values=np.frombuffer(raw[8:],dtype='<i8' if array.attrib['type']=='Int64' else '<f8')
                    np.testing.assert_array_equal(values,f['voxel/'+key][:].ravel())
    reports.append({'case':name,'states':len(manifest['states']),'xdmf_reader':'meshio 5.3.5','passed':True})
ws.path('reports/top_damage_export_checks.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
