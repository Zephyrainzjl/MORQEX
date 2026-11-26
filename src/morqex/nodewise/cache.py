"""Checksummed, externally chunked localization; compact online loading."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ..electrochemistry.aging_cache import CohesiveOnline
from ..electrochemistry.microstructure_cache import OnlineMicrostructureOperator
from ..electrochemistry.aging_dfn import EnergyKernel


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def save_node_kernel(path,op,response,*,chunk_mib=8):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);op.detach().save(path/'base')
    np.savez(path/'energy.npz',direct=response.direct,factors=response.factors,eigenvalues=response.eigenvalues,residual=response.residual)
    full=op.full
    arrays={name:getattr(response,name) for name in ['stress_direct','stress_modes','strain_direct','strain_modes','displacement_direct','displacement_modes','interface_pairs','interface_areas']}
    arrays.update(labels=full.structure.labels,P=op.prolongation,R=op.restriction,voxel_ids=full.diffusion.voxel_ids,
        element_voxel_ids=response.mesh.voxel_ids,volumes=response.mesh.volumes,points=response.mesh.points,cells=response.mesh.cells,
        spacing=np.array(full.structure.spacing_zyx_m),origin=np.array(full.structure.origin_zyx_m),area_full=full.area)
    chunks={};hashes={'energy.npz':digest(path/'energy.npz')}
    for name,array in arrays.items():
        folder=path/'fields'/name;folder.mkdir(parents=True,exist_ok=True);a=np.asarray(array)
        rows=max(1,int(chunk_mib*2**20/max(a[0:1].nbytes,1)));files=[]
        for begin in range(0,len(a),rows):
            f=folder/(str(begin)+'.npy');np.save(f,a[begin:begin+rows],allow_pickle=False)
            rel=str(f.relative_to(path));files.append(rel);hashes[rel]=digest(f)
        chunks[name]={'shape':list(a.shape),'dtype':a.dtype.str,'files':files}
    certificate=getattr(response,'sparse_certificate',None)
    if certificate:
        np.savez(path/'certificate.npz',**{n:v for n,v in certificate.items() if n!='scope'})
        hashes['certificate.npz']=digest(path/'certificate.npz')
    manifest={'schema':'morqex-node-kernel-1','base':op.metadata(),'cohesive':response.metadata,'field_location':response.metadata.get('field_location','tetra'),
        'phase_kinds':full.structure.phase_kinds,'source_geometry':full.structure.provenance,'chunks':chunks,'sha256':hashes}
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return path


def load_node_kernel(path):
    path=Path(path);meta=json.loads((path/'manifest.json').read_text())
    if meta['schema']!='morqex-node-kernel-1' or digest(path/'energy.npz')!=meta['sha256']['energy.npz']:raise ValueError('Invalid node kernel schema or checksum')
    with np.load(path/'energy.npz',allow_pickle=False) as a:
        coh=CohesiveOnline(a['direct'],a['factors'],a['eigenvalues'],float(a['residual']))
    op=OnlineMicrostructureOperator.load(path/'base')
    if 'spreading' in meta:
        name='spreading_online.npz'
        if digest(path/name)!=meta['sha256'][name]:raise ValueError('Spreading online checksum mismatch')
        with np.load(path/name,allow_pickle=False) as a:op.spreading={n:a[n] for n in a.files}
        op.descriptor['spreading']=meta['spreading']
    return EnergyKernel(op,coh)


class NodeLocalization:
    def __init__(self,path):
        self.path=Path(path);self.manifest=json.loads((self.path/'manifest.json').read_text())
        self.kernel=load_node_kernel(path);self.loaded={}
    def array(self,name):
        if name not in self.loaded:
            meta=self.manifest['chunks'][name];pieces=[]
            for rel in meta['files']:
                if digest(self.path/rel)!=self.manifest['sha256'][rel]:raise ValueError('Node localization checksum mismatch')
                pieces.append(np.load(self.path/rel,allow_pickle=False))
            a=np.concatenate(pieces,axis=0) if pieces else np.empty(meta['shape'],dtype=meta['dtype'])
            if list(a.shape)!=meta['shape'] or a.dtype.str!=meta['dtype']:raise ValueError('Node localization shape mismatch')
            self.loaded[name]=a
        return self.loaded[name]
    def recover(self,concentration,strain,theta,damage):
        q=np.r_[strain,np.asarray(concentration)-self.kernel.operator.reference_concentration,theta]
        if q.shape!=(self.kernel.operator.dimension+2,) or not np.isfinite(q).all() or not 0<=damage<=1:raise ValueError('Invalid local state')
        c=self.kernel.cohesive;g=c.residual+(1-c.residual)*(1-damage)**2
        z=c.factors@q/(1+(g-1)*c.eigenvalues)
        result={name:self.array(name+'_direct')@q+self.array(name+'_modes')@z for name in ['stress','strain','displacement']}
        pairs=self.array('interface_pairs');result['opening']=result['displacement'][pairs[:,1]]-result['displacement'][pairs[:,0]]
        result['concentration']=self.array('P')@concentration
        return result
    def voxel_average(self,element_field):
        """Volume average of an element quantity, with NaN in true voids."""
        a=np.asarray(element_field);ids=self.array('element_voxel_ids');w=self.array('volumes');shape=self.array('labels').shape
        value=np.zeros((np.prod(shape),*a.shape[1:]));volume=np.bincount(ids,weights=w,minlength=value.shape[0])
        np.add.at(value,ids,a*w.reshape((-1,)+(1,)*(a.ndim-1)))
        np.divide(value,volume.reshape((-1,)+(1,)*(a.ndim-1)),out=value,where=volume.reshape((-1,)+(1,)*(a.ndim-1))>0)
        value[volume==0]=np.nan
        return value.reshape((*shape,*a.shape[1:]))
