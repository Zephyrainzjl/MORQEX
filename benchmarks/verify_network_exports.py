"""Read every exported VTK state independently and compare to accepted HDF5."""
import argparse,base64,json,os,struct,time
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py,numpy as np
from morqex import Workspace
from morqex.top_damage.backend import require_local_budget


def arrays(element):
    result={}
    for array in element.findall('DataArray'):
        dtype={'Float64':'<f8','Int64':'<i8','Int32':'<i4','UInt8':'u1'}[array.attrib['type']]
        if array.attrib['format']=='binary':
            raw=base64.b64decode(array.text,validate=True)
            if struct.unpack('<Q',raw[:8])[0]!=len(raw)-8:raise ValueError('VTK binary length mismatch')
            values=np.frombuffer(raw[8:],dtype=dtype)
        else:values=np.fromstring(array.text,sep=' ',dtype=dtype)
        result[array.attrib['Name']]=values
    return result


def main():
    require_local_budget();p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args()
    w=Workspace(os.environ['MORQEX_WORKSPACE']);root=Path(args.directory).resolve()
    if not root.is_relative_to(w.root.resolve()):raise ValueError('External workspace required')
    folder=root/'paraview';manifest=json.loads((folder/'export_manifest.json').read_text())
    with h5py.File(root/'initial.h5') as f:
        static={name:f[name][:] for name in ('phase_labels','particle_ids','grain_ids')}
    with h5py.File(root/'mesh.h5') as f:
        points=f['points_m'][:];cells=f['cells'][:];voxels=f['voxel_ids'][:]
    start=time.perf_counter();verified=0;previous=-np.inf
    for step in manifest['steps']:
        with h5py.File(root/f'step_{step:06d}.h5') as f:
            if f.attrs['identity']!=manifest['identity']:raise ValueError('Mixed export identities')
            record=json.loads(f.attrs['record'])
            if record['time_s']<=previous:raise ValueError('Nonmonotonic export times')
            previous=record['time_s']
            for node in range(manifest['nodes']):
                prefix=f'node_{node:03d}_step_{step:06d}'
                image=ET.parse(folder/(prefix+'.vti')).getroot()
                data=arrays(image.find('ImageData/Piece/CellData'))
                for name in f['voxel']:
                    np.testing.assert_array_equal(data[name],f['voxel/'+name][node].ravel())
                for name,values in static.items():np.testing.assert_array_equal(data[name],values.ravel())
                np.testing.assert_array_equal(data['dfn_node_id'],np.full(static['phase_labels'].size,node))
                mesh=ET.parse(folder/(prefix+'.vtu')).getroot().find('UnstructuredGrid/Piece')
                nodal=arrays(mesh.find('PointData'));element=arrays(mesh.find('CellData'))
                np.testing.assert_array_equal(nodal['damage'],f['damage'][node].ravel())
                np.testing.assert_array_equal(nodal['displacement_m'],f['displacement_m'][node].ravel())
                np.testing.assert_array_equal(element['fatigue_J_m3'],f['fatigue_J_m3'][node].ravel())
                np.testing.assert_array_equal(element['phase_labels'],static['phase_labels'].ravel()[voxels])
                coordinates=next(iter(arrays(mesh.find('Points')).values()))
                np.testing.assert_array_equal(coordinates,points.ravel())
                np.testing.assert_array_equal(arrays(mesh.find('Cells'))['connectivity'],cells.ravel())
                verified+=2
    for node in range(manifest['nodes']):
        for suffix in ('','_mechanics'):
            entries=ET.parse(folder/f'node_{node:03d}{suffix}.pvd').getroot().findall('Collection/DataSet')
            if len(entries)!=len(manifest['steps']):raise ValueError('PVD frame count mismatch')
            times=np.array([float(entry.attrib['timestep']) for entry in entries])
            if np.any(np.diff(times)<=0):raise ValueError('PVD timeline is not ordered')
            for entry,step in zip(entries,manifest['steps']):
                expected=f'node_{node:03d}_step_{step:06d}'+('.vtu' if suffix else '.vti')
                if entry.attrib['file']!=expected or not (folder/expected).is_file():raise ValueError('Broken PVD association')
    report={'passed':True,'directory':str(folder),'identity':manifest['identity'],'states':len(manifest['steps']),
            'nodes':manifest['nodes'],'verified_vtk_files':verified,'seconds':time.perf_counter()-start,
            'scope':'Every exported concentration/stress/damage/fatigue/network field and exact Tet4 displacement/mesh matched to accepted HDF5; no physical topology or continuum qualification implied.'}
    (folder/'independent_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
