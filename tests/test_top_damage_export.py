import base64
import struct
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from morqex.top_damage.dense_export import write_field_vti
from morqex.top_damage.dense_fields import DisplacementRecovery


def test_vti_full_lattice_axis_component_and_fp64_roundtrip(tmp_path):
    shape=(2,3,4)
    scalar=np.arange(24,dtype=float).reshape(shape)/7
    vector=np.stack((scalar,-scalar,scalar+np.pi),axis=-1)
    path=tmp_path/'field.vti'
    write_field_vti(path,shape,(1.,2.,3.),(4.,5.,6.),
                    {'scalar':scalar,'vector':vector,'labels':scalar.astype(int)})
    root=ET.parse(path).getroot();grid=root.find('ImageData')
    assert grid.attrib=={'WholeExtent':'0 4 0 3 0 2','Origin':'6.0 5.0 4.0','Spacing':'3.0 2.0 1.0'}
    for element in grid.findall('./Piece/CellData/DataArray'):
        raw=base64.b64decode(element.text);length=struct.unpack('<Q',raw[:8])[0]
        assert length==len(raw)-8
        a=np.frombuffer(raw[8:],dtype='<i8' if element.attrib['type']=='Int64' else '<f8')
        original={'scalar':scalar,'vector':vector,'labels':scalar.astype(int)}[element.attrib['Name']]
        np.testing.assert_array_equal(a,original.ravel(order='C'))
        assert int(element.attrib['NumberOfComponents'])==(3 if element.attrib['Name']=='vector' else 1)
    with pytest.raises(ValueError):
        write_field_vti(path,shape,(1.,)*3,(0.,)*3,{'bad':np.full(shape,np.nan)})


def test_affine_displacement_recovery_preserves_tensor_and_coordinates(tmp_path):
    import h5py
    from morqex.mechanics import TetMesh
    from morqex.microstructure import Volume
    mesh=TetMesh.from_volume(Volume(np.ones((2,3,4),dtype=np.uint8),spacing=(1e-6,2e-6,3e-6)))
    with h5py.File(tmp_path/'mesh.h5','w') as f:
        for name,a in [('points_m',mesh.points),('tetrahedra',mesh.cells),('voxel_ids',mesh.voxel_ids)]:f[name]=a
    gradient=np.array([[.001,.002,.003],[-.004,.005,.001],[.002,-.003,.006]])
    offset=np.array([1e-8,-2e-8,3e-8])
    recovery=DisplacementRecovery(tmp_path,(2,3,4))
    strain,u=recovery.fields(mesh.points@gradient.T+offset)
    np.testing.assert_allclose(strain,np.broadcast_to((gradient+gradient.T)/2,strain.shape),atol=1e-16)
    z,y,x=np.meshgrid((np.arange(2)+.5)*1e-6,(np.arange(3)+.5)*2e-6,(np.arange(4)+.5)*3e-6,indexing='ij')
    exact=np.stack((x,y,z),axis=-1)@gradient.T+offset
    np.testing.assert_allclose(u,exact,rtol=1e-14,atol=1e-22)
