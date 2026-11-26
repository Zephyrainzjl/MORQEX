"""Independent meshio XDMF and binary VTI consistency check for actual history."""
import argparse,base64,json,struct,xml.etree.ElementTree as ET
from pathlib import Path
import h5py,meshio,numpy as np
from morqex.top_damage.backend import require_local_budget
require_local_budget()
p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();root=Path(args.directory)
records=[]
with meshio.xdmf.TimeSeriesReader(root/'resolved_history.xdmf') as reader:
    points,cells=reader.read_points_cells()
    files=sorted(root.glob('mechanics_*.h5'));assert len(files)==reader.num_steps
    for i,path in enumerate(files):
        t,pd,cd=reader.read_data(i)
        with h5py.File(path) as f:
            np.testing.assert_array_equal(points,f['points_m'][:]);np.testing.assert_array_equal(cells[0].data,f['tetrahedra'][:])
            for key in ['damage','displacement_m']:np.testing.assert_array_equal(pd[key],f[key][:])
            for key in ['fatigue_J_m3','cauchy_stress_Pa']:np.testing.assert_array_equal(cd[key][0],f[key][:])
            assert t==f.attrs['time_s']
            arrays={}
            for node in ET.parse(path.with_suffix('.vti')).findall('.//CellData/DataArray'):
                block=base64.b64decode(node.text);size=struct.unpack('<Q',block[:8])[0];assert size==len(block)-8
                arrays[node.attrib['Name']]=np.frombuffer(block[8:],dtype='<i8' if node.attrib['type']=='Int64' else '<f8')
            for key,hkey in [('concentration_mol_m3','voxel_concentration_mol_m3'),('potential_V','voxel_potential_V'),('mechanical_chemical_potential_J_mol','voxel_mechanical_potential_J_mol')]:np.testing.assert_array_equal(arrays[key],f[hkey][:].ravel())
            assert len(arrays['damage'])==f['voxel_concentration_mol_m3'].size
            records.append({'time_s':t,'tetrahedra':len(cells[0].data),'full_voxels':len(arrays['damage']),'exact_export':True})
(root/'export_checks.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
