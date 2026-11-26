"""Energy-consistent nonlocal reaction-spreading resistance on real voxels.

Dirichlet local node potentials at X faces, insulated transverse faces. The
Green maps and B^T K^-1 B come from the same conductive cell. Regional BV sees
the area-averaged potential drop; nonlinear within-region reaction error remains
a separate closure error. This is a dissipative Schur complement, not a claim
that a macro node uniquely determines arbitrary microscopic electrochemistry.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from ..transport import VoxelDiffusion


def build_spreading(electrode,P,topology):
    full=electrode;fv=full.diffusion;mask=full.reactive;cells=fv.face_cells[mask];areas=fv.face_areas[mask]/fv.rve_volume
    neighbors=fv.coordinates_zyx[cells]+fv.face_normals[mask][:,::-1].astype(int)
    neighbor_ids=np.ravel_multi_index(neighbors.T,full.volume.data.shape)
    sources={};fields={};records={};m=P.shape[1]
    for name in ['electronic','ionic']:
        if name=='ionic':phases=topology.ionic_phases(full.structure);factors=topology.ionic_efficiencies(full.structure);ids=neighbor_ids
        else:
            phases=topology.electronic_phases(full.structure) if hasattr(topology,'electronic_phases') else [p for p,k in full.structure.phase_kinds.items() if k!='pore']
            factors=topology.electronic_efficiencies(full.structure) if hasattr(topology,'electronic_efficiencies') else {p:1. for p in phases};ids=fv.voxel_ids[cells]
        selected=np.isin(full.volume.data.ravel(),phases);coeff=np.array([factors[int(p)] for p in full.volume.data.ravel()[selected]])
        graph=VoxelDiffusion(full.volume,active_phases=phases,diffusivity=coeff[:,None])
        lookup=np.full(full.volume.data.size,-1,dtype=int);lookup[graph.voxel_ids]=np.arange(len(graph.weights));rows=lookup[ids]
        if min(rows)<0:raise ValueError('Reaction source deposited outside its physical conductor')
        B=np.zeros((len(graph.weights),m));np.add.at(B,rows,areas[:,None]*P[cells])
        if not np.allclose(B.sum(axis=0),P.T@full.area,rtol=1e-12,atol=1e-9):raise ValueError('Charge source/physical reaction area mismatch')
        h=full.volume.spacing[2];boundary=2*np.prod(full.volume.spacing)/h**2/graph.rve_volume
        local=coeff;left=(graph.coordinates_zyx[:,2]==0)*boundary*local;right=(graph.coordinates_zyx[:,2]==full.volume.data.shape[2]-1)*boundary*local
        anchored=np.isin(graph.components,np.unique(graph.components[left+right>0]))
        if np.max(abs(B[~anchored]),initial=0)>0:raise ValueError('Reaction on an electrically isolated component')
        K=(graph.stiffness+sparse.diags(left+right))[anchored][:,anchored].tocsc()
        factor=splu(K);solution=np.zeros_like(B);solution[anchored]=factor.solve(B[anchored])
        R=B.T@solution;R=(R+R.T)/2
        if np.linalg.eigvalsh(R).min()<-1e-10*max(1.,np.linalg.norm(R,2)):raise ValueError('Spreading resistance lost positive dissipation')
        field=np.zeros((full.volume.data.size,m));field[graph.voxel_ids]=solution
        phase=np.zeros(full.volume.data.size,bool);phase[graph.voxel_ids[anchored]]=True
        sources[name]=R;fields[name+'_map']=field;fields[name+'_mask']=phase
        records[name]={'cells':int(anchored.sum()),'reciprocity_relative_defect':float(np.linalg.norm(R-R.T)/max(np.linalg.norm(R),1e-30)),
            'linear_residual':float(np.linalg.norm(K@solution[anchored]-B[anchored])/max(np.linalg.norm(B),1e-30))}
    return sources,fields,{'model':'Dirichlet reaction-spreading Green ports; regional BV with surface-averaged local potentials',
        'boundary':'DFN node potential on both local X faces; insulated transverse faces; no imposed macro-gradient fluctuation in this closure',
        'validation':records,'scope':'conditional local conductive response, not a fully resolved nonlinear voxel-BV reaction solve'}


def attach_spreading(path,electrode,P,topology):
    from .cache import digest
    path=Path(path);online,fields,metadata=build_spreading(electrode,P,topology)
    np.savez_compressed(path/'spreading_online.npz',**online)
    np.savez_compressed(path/'spreading_fields.npz',**fields)
    manifest=json.loads((path/'manifest.json').read_text());manifest['spreading']=metadata
    for name in ['spreading_online.npz','spreading_fields.npz']:manifest['sha256'][name]=digest(path/name)
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2));return metadata


def recover_potentials(localizer,current_A_m2,damage,electronic_conductivity,ionic_conductivity,phi_s,phi_e):
    from .cache import digest
    meta=localizer.manifest
    if 'spreading' not in meta:raise ValueError('This operator has no resolved spreading maps')
    if electronic_conductivity<=0 or ionic_conductivity<=0 or not 0<=damage<=1:raise ValueError('Positive local conductor coefficients and admissible damage required')
    path=localizer.path/'spreading_fields.npz'
    if 'spreading_fields' not in localizer.loaded:
        if digest(path)!=meta['sha256'][path.name]:raise ValueError('Spreading localization checksum mismatch')
        with np.load(path,allow_pickle=False) as a:localizer.loaded['spreading_fields']={n:a[n] for n in a.files}
    fields=localizer.loaded['spreading_fields'];j=np.asarray(current_A_m2,float)*(1-damage)
    out={}
    for channel,base,conductivity,sign in [('electronic',phi_s,electronic_conductivity,-1),('ionic',phi_e,ionic_conductivity,1)]:
        a=base+sign*fields[channel+'_map']@j/conductivity;a[~fields[channel+'_mask']]=np.nan
        out[channel+'_potential_V']=a.reshape(localizer.array('labels').shape)
    return out
