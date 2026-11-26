"""Lossless accepted-state exports, without declaring diffuse damage a void.

VTI holds the full reference lattice and computed voxel fields. XDMF references
the original FP64 FEM displacement/damage and mesh HDF5 without copying them.
Use ParaView Warp By Vector with scale 1 for actual small-strain displacement;
the mesh connectivity is unchanged. Damage thresholds remain diagnostics.
"""
import base64
import json
import struct
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .polycrystal import PolycrystalData
from morqex.mechanics.tensors import MANDEL_BASIS
from .dense_fields import DisplacementRecovery


def write_field_vti(path, shape, spacing, origin, fields):
    nz, ny, nx = shape
    extent = f'0 {nx} 0 {ny} 0 {nz}'
    root = ET.Element('VTKFile', type='ImageData', version='1.0',
                      byte_order='LittleEndian', header_type='UInt64')
    grid = ET.SubElement(root, 'ImageData', WholeExtent=extent,
        Origin=' '.join(map(str, origin[::-1])), Spacing=' '.join(map(str, spacing[::-1])))
    cell = ET.SubElement(ET.SubElement(grid, 'Piece', Extent=extent), 'CellData')
    for name, value in fields.items():
        a = np.asarray(value)
        if a.shape[:3] != tuple(shape) or a.ndim not in (3, 4) or not np.isfinite(a).all():
            raise ValueError('Finite full-lattice scalar/vector field required')
        integer = np.issubdtype(a.dtype, np.integer)
        a = np.asarray(a, dtype='<i8' if integer else '<f8', order='C')
        raw = a.tobytes(order='C')
        array = ET.SubElement(cell, 'DataArray', Name=name, type='Int64' if integer else 'Float64',
            NumberOfComponents=str(a.shape[3] if a.ndim == 4 else 1), format='binary')
        array.text = base64.b64encode(struct.pack('<Q', len(raw))+raw).decode('ascii')
    path = Path(path)
    temporary = path.with_suffix('.pending')
    ET.ElementTree(root).write(temporary, encoding='utf-8', xml_declaration=True)
    temporary.replace(path)


def export_dense_history(directory):
    """Export every accepted state and temporal collections next to its HDF5."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    data = PolycrystalData.load_hdf5(directory/'initial_geometry.h5')
    recovery = DisplacementRecovery(directory,data.structure.labels.shape)
    with h5py.File(directory/'mesh.h5', 'r') as f:
        nodes, cells = len(f['points_m']), len(f['tetrahedra'])
    xroot = ET.Element('Xdmf', Version='3.0')
    temporal = ET.SubElement(ET.SubElement(xroot, 'Domain'), 'Grid',
        Name='Accepted small-strain states', GridType='Collection', CollectionType='Temporal')
    proot = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
    collection = ET.SubElement(proot, 'Collection')
    records = []
    for expected, path in enumerate(sorted(directory.glob('step_*.h5'))):
        with h5py.File(path, 'r') as f:
            r = json.loads(f.attrs['record'])
            if not f.attrs['complete'] or f.attrs['identity'] != manifest['identity'] or r['step'] != expected:
                raise ValueError('Missing, rejected or inconsistent accepted state')
            fields = {name: f['voxel/'+name][:] for name in f['voxel']}
            fields.update(particle_ids=data.particle_ids, grain_ids=data.grain_ids)
            stress = np.einsum('...k,kij->...ij', fields['stress_Mandel_Pa'], MANDEL_BASIS)
            fields['stress_tensor_Pa'] = stress.reshape(*data.structure.labels.shape, 9)
            fields['maximum_principal_stress_Pa'] = np.linalg.eigvalsh(stress)[..., -1]
            strain, displacement = recovery.fields(f['displacement_m'][:])
            fields['total_strain_tensor'] = strain.reshape(*data.structure.labels.shape,9)
            fields['displacement_m'] = displacement
            write_field_vti(path.with_suffix('.vti'), data.structure.labels.shape,
                            data.structure.spacing_zyx_m, data.structure.origin_zyx_m, fields)
        ET.SubElement(collection, 'DataSet', timestep=str(r['time_s']), group='', part='0', file=path.with_suffix('.vti').name)
        grid = ET.SubElement(temporal, 'Grid', Name=path.stem, GridType='Uniform')
        ET.SubElement(grid, 'Time', Value=str(r['time_s']))
        topology = ET.SubElement(grid, 'Topology', TopologyType='Tetrahedron', NumberOfElements=str(cells))
        ET.SubElement(topology, 'DataItem', Format='HDF', NumberType='Int', Dimensions=f'{cells} 4').text = 'mesh.h5:/tetrahedra'
        geometry = ET.SubElement(grid, 'Geometry', GeometryType='XYZ')
        ET.SubElement(geometry, 'DataItem', Format='HDF', NumberType='Float', Precision='8', Dimensions=f'{nodes} 3').text = 'mesh.h5:/points_m'
        for name, center, kind, dimensions, reference in [
            ('damage', 'Node', 'Scalar', str(nodes), path.name+':/damage'),
            ('displacement_m', 'Node', 'Vector', f'{nodes} 3', path.name+':/displacement_m'),
            ('fatigue_J_m3', 'Cell', 'Scalar', str(cells), path.name+':/fatigue'),
            ('phase_labels', 'Cell', 'Scalar', str(cells), 'mesh.h5:/phases')]:
            attribute = ET.SubElement(grid, 'Attribute', Name=name, Center=center, AttributeType=kind)
            ET.SubElement(attribute, 'DataItem', Format='HDF', Dimensions=dimensions,
                NumberType='Int' if name=='phase_labels' else 'Float', **({} if name=='phase_labels' else {'Precision':'8'})).text = reference
        records.append({'step':r['step'], 'time_s':r['time_s'], 'vti':path.with_suffix('.vti').name})
    if not records:
        raise ValueError('No accepted states to export')
    for root, target in [(xroot, directory/'mechanics_history.xdmf'), (proot, directory/'voxel_history.pvd')]:
        temp = target.with_suffix('.pending')
        ET.ElementTree(root).write(temp, encoding='utf-8', xml_declaration=True)
        temp.replace(target)
    report = {'identity':manifest['identity'], 'states':records,
        'voxel_fields':'FP64, full reference lattice; no downsampling',
        'mechanical_geometry':'Original Tet4 connectivity, FP64 displacement for optional scale-1 warp',
        'physical_topology_qualified':False,
        'scope':'Diffuse damage and threshold components are diagnostics, not reconstructed physical crack pores.'}
    (directory/'export_manifest.json').write_text(json.dumps(report, indent=2))
    return report
