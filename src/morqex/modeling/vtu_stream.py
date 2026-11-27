"""VTK XML appended-raw Tet4/triangle exchange with bounded independent readback.

Standard VTK UnstructuredGrid, LittleEndian, UInt64 block headers. Raw appended
storage is explicitly supported by VTK/ParaView. It trades compression for low
peak memory and avoids concatenating all connectivity or base64 text in RAM.
"""
import struct
import xml.etree.ElementTree as ET
import numpy as np


def _array_chunks(a,dtype,rows=50000):
    for start in range(0,len(a),rows):yield np.asarray(a[start:start+rows],dtype=dtype).reshape(-1)


def _constant(count,value,dtype):
    for start in range(0,count,50000):yield np.full(min(50000,count-start),value,dtype=dtype)


def _specifications(points,cells,domains,facets,markers):
    nt=len(cells);nf=len(facets);nc=nt+nf
    def connectivity():
        yield from _array_chunks(cells,'<u4');yield from _array_chunks(facets,'<u4')
    def offsets():
        for start in range(0,nt,50000):yield np.arange(start+1,min(start+50000,nt)+1,dtype='<u8')*4
        for start in range(0,nf,50000):yield np.arange(start+1,min(start+50000,nf)+1,dtype='<u8')*3+nt*4
    def types():yield from _constant(nt,10,'u1');yield from _constant(nf,5,'u1')
    def materials():yield from _array_chunks(domains,'<i4');yield from _constant(nf,0,'<i4')
    def boundaries():yield from _constant(nt,0,'<i4');yield from _array_chunks(markers,'<i4')
    return [dict(name='points',vtk='Float64',dtype='<f8',components=3,count=len(points)*3,chunks=lambda:_array_chunks(points,'<f8')),
        dict(name='connectivity',vtk='UInt32',dtype='<u4',components=1,count=nt*4+nf*3,chunks=connectivity),
        dict(name='offsets',vtk='UInt64',dtype='<u8',components=1,count=nc,chunks=offsets),
        dict(name='types',vtk='UInt8',dtype='u1',components=1,count=nc,chunks=types),
        dict(name='domain_id',vtk='Int32',dtype='<i4',components=1,count=nc,chunks=materials),
        dict(name='boundary_id',vtk='Int32',dtype='<i4',components=1,count=nc,chunks=boundaries)]


def write_and_verify_vtu(path,points,cells,domains,facets,markers):
    if len(points)>np.iinfo(np.uint32).max:raise ValueError('Streaming VTU currently supports UInt32 node indices')
    specs=_specifications(points,cells,domains,facets,markers);offset=0
    for s in specs:s['offset']=offset;offset+=8+s['count']*np.dtype(s['dtype']).itemsize
    def tag(s):
        components=f' NumberOfComponents="{s["components"]}"' if s['components']!=1 else ''
        return f'<DataArray type="{s["vtk"]}" Name="{s["name"]}"{components} format="appended" offset="{s["offset"]}"/>'
    header='<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n<UnstructuredGrid>\n'
    header+=f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(cells)+len(facets)}">\n<Points>{tag(specs[0])}</Points>\n<Cells>'
    header+=''.join(tag(s) for s in specs[1:4])+'</Cells>\n<CellData>'
    header+=''.join(tag(s) for s in specs[4:])+'</CellData>\n</Piece>\n</UnstructuredGrid>\n<AppendedData encoding="raw">_'
    with path.open('xb') as stream:
        stream.write(header.encode())
        for s in specs:
            stream.write(struct.pack('<Q',s['count']*np.dtype(s['dtype']).itemsize));written=0
            for block in s['chunks']():stream.write(block.tobytes());written+=len(block)
            if written!=s['count']:raise ArithmeticError('Incomplete VTU array write')
        stream.write(b'\n</AppendedData>\n</VTKFile>\n')
    # Parse schema/offsets from the actual file, not from the writer's offsets.
    prefix=b'<AppendedData encoding="raw">_'
    with path.open('rb') as stream:
        initial=stream.read(16384);position=initial.find(prefix)
        if position<0:raise ArithmeticError('Missing VTU appended data header')
        document=ET.fromstring(initial[:position]+b'</VTKFile>');piece=document.find('UnstructuredGrid/Piece')
        if piece is None or int(piece.attrib['NumberOfPoints'])!=len(points) or int(piece.attrib['NumberOfCells'])!=len(cells)+len(facets):raise ArithmeticError('Wrong VTU point/cell counts')
        if document.attrib.get('byte_order')!='LittleEndian' or document.attrib.get('header_type')!='UInt64':raise ArithmeticError('Wrong VTU binary header convention')
        arrays={a.attrib['Name']:a for a in piece.iter('DataArray')}
        if set(arrays)!={s['name'] for s in specs}:raise ArithmeticError('VTU array schema mismatch')
        start=position+len(prefix)
        for s in specs:
            a=arrays[s['name']]
            if a.attrib['type']!=s['vtk'] or int(a.attrib.get('NumberOfComponents',1))!=s['components']:raise ArithmeticError('VTU array type mismatch')
            stream.seek(start+int(a.attrib['offset']));declared=struct.unpack('<Q',stream.read(8))[0]
            if declared!=s['count']*np.dtype(s['dtype']).itemsize:raise ArithmeticError('VTU binary block byte count mismatch')
            count=0
            for expected in s['chunks']():
                data=stream.read(expected.nbytes)
                if len(data)!=expected.nbytes:raise ArithmeticError('Truncated VTU binary data')
                np.testing.assert_array_equal(np.frombuffer(data,dtype=s['dtype']),expected);count+=len(expected)
            if count!=s['count']:raise ArithmeticError('Incomplete VTU readback')
    return {'readback':'independent bounded VTK XML binary parser; every point/connectivity/offset/type/domain/boundary entry',
        'storage':'VTK XML appended raw; UInt64 block headers, uncompressed','domain_readback_passed':True,'boundary_readback_passed':True}


def verify_with_vtk(path,source_hdf5,*,length_unit='m'):
    """Optional independent compiled-reader validation of a complete file."""
    import h5py
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
    from vtkmodules.vtkCommonCore import vtkVersion
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from .comsol import LENGTH_UNITS
    reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(path));reader.Update();grid=reader.GetOutput()
    if reader.GetErrorCode():raise ArithmeticError('Native VTK reader failed')
    with h5py.File(source_hdf5) as f:
        points=f['mesh/points_m'];cells=f['mesh/tetrahedra'];facets=f['mesh/facets'];phase=f['mesh/phase'];marks=f['mesh/facet_markers'];nt=len(cells);nf=len(facets)
        if grid.GetNumberOfPoints()!=len(points) or grid.GetNumberOfCells()!=nt+nf:raise ArithmeticError('Native VTK count mismatch')
        p=vtk_to_numpy(grid.GetPoints().GetData());c=vtk_to_numpy(grid.GetCells().GetConnectivityArray())
        offsets=vtk_to_numpy(grid.GetCells().GetOffsetsArray())
        domain=vtk_to_numpy(grid.GetCellData().GetArray('domain_id')).reshape(-1)
        boundary=vtk_to_numpy(grid.GetCellData().GetArray('boundary_id')).reshape(-1)
        types=vtk_to_numpy(grid.GetCellTypesArray())
        if len(c)!=4*nt+3*nf:raise ArithmeticError('Native VTK connectivity length mismatch')
        if len(offsets)!=nt+nf+1:raise ArithmeticError('Native VTK offset count mismatch')
        labels=set()
        for start in range(0,nt,50000):labels.update(map(int,np.unique(phase[start:start+50000])))
        labels=sorted(labels)
        for start in range(0,len(points),50000):np.testing.assert_array_equal(p[start:start+50000],points[start:start+50000]/LENGTH_UNITS[length_unit])
        for start in range(0,nt,50000):
            stop=min(start+50000,nt)
            np.testing.assert_array_equal(c[start*4:stop*4].reshape(-1,4),cells[start:stop])
            np.testing.assert_array_equal(offsets[start:stop],np.arange(start,stop,dtype=np.int64)*4)
            np.testing.assert_array_equal(domain[start:stop],np.searchsorted(labels,phase[start:stop])+1)
            np.testing.assert_array_equal(boundary[start:stop],np.zeros(stop-start,np.int32))
            np.testing.assert_array_equal(types[start:stop],np.full(stop-start,10,np.uint8))
        for start in range(0,nf,50000):
            stop=min(start+50000,nf)
            np.testing.assert_array_equal(c[4*nt+3*start:4*nt+3*stop].reshape(-1,3),facets[start:stop])
            np.testing.assert_array_equal(offsets[nt+start:nt+stop],4*nt+np.arange(start,stop,dtype=np.int64)*3)
            np.testing.assert_array_equal(domain[nt+start:nt+stop],np.zeros(stop-start,np.int32))
            np.testing.assert_array_equal(boundary[nt+start:nt+stop],marks[start:stop])
            np.testing.assert_array_equal(types[nt+start:nt+stop],np.full(stop-start,5,np.uint8))
        if int(offsets[-1])!=4*nt+3*nf:raise ArithmeticError('Native VTK terminal offset mismatch')
    return {'passed':True,'reader':'VTK'+vtkVersion.GetVTKVersion(),'points':len(p),'tetrahedra':nt,'triangles':nf,
        'length_unit':length_unit,'scope':'all coordinates/connectivity/cell types/domain tags/boundary tags; compiled VTK reader, no GUI solver'}
