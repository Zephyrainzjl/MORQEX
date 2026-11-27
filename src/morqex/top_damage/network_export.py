"""Full, lossless per-DFN-node VTI time series of accepted microstates.

Material labels remain material labels. A diffuse damage field is exported as
a field, not relabelled as a physical crack pore. Each node owns a separate PVD
collection so representative SVEs are not mistaken for a tiled whole electrode.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import h5py
from .dense_export import write_field_vti
from .network_report import audit_network_history
from .reference_run import write_vtu
from types import SimpleNamespace


def export_network_history(directory,workspace,*,steps=None):
    from .backend import require_local_budget
    from morqex.workspace import Workspace
    require_local_budget();ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ws.root.resolve()):raise ValueError('External workspace required')
    summary,records,_=audit_network_history(directory)
    meta=json.loads((directory/'manifest.json').read_text())
    chosen=list(range(len(records))) if steps is None else sorted(set(steps))
    if not chosen or any(type(s) is not int or s<0 or s>=len(records) for s in chosen):
        raise ValueError('Export only existing accepted step indices')
    with h5py.File(directory/'initial.h5') as f:
        labels=f['phase_labels'][:];particles=f['particle_ids'][:];grains=f['grain_ids'][:]
        spacing=np.asarray(f.attrs['spacing_zyx_m'])
        origin=np.asarray(f.attrs['origin_zyx_m'])
    with h5py.File(directory/'mesh.h5') as f:
        cells=f['cells'][:];voxels=f['voxel_ids'][:];points=f['points_m'][:]
    x=points[cells];volume=np.linalg.det(np.swapaxes(x[:,1:]-x[:,:1],1,2))/6
    weights=np.bincount(voxels,weights=volume,minlength=labels.size)
    output=directory/'paraview';output.mkdir(exist_ok=True)
    roots=[];collections=[];mechanical_roots=[];mechanical_collections=[];files=[]
    for node in range(meta['nodes']):
        root=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
        roots.append(root);collections.append(ET.SubElement(root,'Collection'))
        mechanical=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
        mechanical_roots.append(mechanical);mechanical_collections.append(ET.SubElement(mechanical,'Collection'))
    for step in chosen:
        with h5py.File(directory/f'step_{step:06d}.h5') as f:
            for node in range(meta['nodes']):
                fields={name:np.asarray(f['voxel/'+name][node]) for name in f['voxel']}
                # VTK has no Bool scalar type in this writer.
                fields={k:v.astype(np.uint8) if v.dtype.kind=='b' else v for k,v in fields.items()}
                u=f['displacement_m'][node].reshape(-1,3)
                uc=u[cells].mean(axis=1)
                mean=np.column_stack([np.bincount(voxels,weights=volume*uc[:,a],minlength=labels.size) for a in range(3)])
                mean=np.divide(mean,weights[:,None],out=np.zeros_like(mean),where=weights[:,None]>0)
                stress=fields['stress_Pa'].reshape(labels.shape+(3,3))
                fields.update(phase_labels=labels,particle_ids=particles,grain_ids=grains,
                    dfn_node_id=np.full(labels.shape,node,np.int32),displacement_m=mean.reshape(labels.shape+(3,)),
                    max_principal_stress_Pa=np.linalg.eigvalsh(stress)[...,-1])
                name=f'node_{node:03d}_step_{step:06d}.vti'
                write_field_vti(output/name,labels.shape,spacing,origin,fields)
                mechanical_name=f'node_{node:03d}_step_{step:06d}.vtu'
                write_vtu(output/mechanical_name,SimpleNamespace(points=points,cells=cells),
                    {'damage':f['damage'][node],'displacement_m':u},
                    {'fatigue_J_m3':f['fatigue_J_m3'][node],'phase_labels':labels.ravel()[voxels]})
                ET.SubElement(collections[node],'DataSet',timestep=str(records[step]['time_s']),
                    group=f'DFN node {node}',part='0',file=name)
                ET.SubElement(mechanical_collections[node],'DataSet',timestep=str(records[step]['time_s']),
                    group=f'DFN node {node} Tet4',part='0',file=mechanical_name)
                files.append({'file':name,'node':node,'step':step,'time_s':records[step]['time_s']})
    for node,root in enumerate(roots):
        target=output/f'node_{node:03d}.pvd';temp=target.with_suffix('.pending')
        ET.ElementTree(root).write(temp,encoding='utf-8',xml_declaration=True);temp.replace(target)
        target=output/f'node_{node:03d}_mechanics.pvd';temp=target.with_suffix('.pending')
        ET.ElementTree(mechanical_roots[node]).write(temp,encoding='utf-8',xml_declaration=True);temp.replace(target)
    result={'identity':summary['identity'],'directory':str(output),'nodes':meta['nodes'],'steps':chosen,'files':files,
        'precision':'FP64 fields, exact original reference voxel lattice; no downsampling',
        'scope':'Per-node SVE fields and volume-averaged displacement. No invented crack pores, moved phase labels or full-electrode spatial tiling.',
        'physical_topology_qualified':False}
    (output/'export_manifest.json').write_text(json.dumps(result,indent=2));return result
