"""Separate compact online operators from optional geometric localization data."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse


@dataclass
class OnlineMicrostructureOperator:
    """No mesh, fine diffusion graph, basis or mechanical solver retained."""
    H: np.ndarray
    weights: np.ndarray
    reaction_area: np.ndarray
    stiffness: object
    stress_mean: np.ndarray
    stress_second: np.ndarray
    port_gram: np.ndarray
    port_box: np.ndarray
    descriptor: dict

    def __post_init__(self):
        for name in ['H','weights','reaction_area','stress_mean','stress_second','port_gram','port_box']:
            value=np.array(getattr(self,name),copy=True);value.flags.writeable=False;setattr(self,name,value)

    @property
    def dimension(self):return len(self.weights)
    @property
    def active_fraction(self):return float(self.weights.sum())
    @property
    def area_density(self):return float(self.reaction_area.sum())
    @property
    def reference_concentration(self):return self.descriptor['reference_concentration']
    @property
    def transport(self):return self.descriptor['transport']
    @property
    def porosity(self):return self.descriptor['porosity']
    def metadata(self):return self.descriptor.copy()

    def moments(self,q):
        mean=np.asarray(q)@self.stress_mean.T
        conditional=mean/self.descriptor['solid_fraction']
        second=np.einsum('...i,kij,...j->...k',q,self.stress_second,q)
        return {'volume_mean_stress':mean,'solid_conditional_mean_stress':conditional,'solid_conditional_variance':second-conditional**2}

    def reaction_port_bound(self,q,exchange_current,overpotential,temperature,*,feedback=1.):
        """MVT/Cauchy upper bound for fine-vs-coarse BV current at P a.

        This bounds the instantaneous mechanical reaction-closure error at the
        represented concentration. It is NOT a trajectory or stress error bound.
        Complexity/storage depend on r, not the fine voxel count.
        """
        if self.descriptor.get('port_certificate')=='not the piecewise-constant certificate':raise NotImplementedError('POU reaction closure requires its own quadrature certificate')
        q=np.asarray(q);j0=np.asarray(exchange_current);eta=np.asarray(overpotential)
        if q.shape!=(self.dimension+2,) or j0.shape!=(self.dimension,) or eta.shape!=j0.shape or np.any(j0<0) or temperature<=0 or not 0<=feedback<=1:raise ValueError('Invalid reaction-bound inputs')
        RT=8.31446261815324*temperature
        spread=feedback*(self.port_box@abs(q))
        quadratic=np.einsum('i,kij,j->k',q,self.port_gram,q)
        roundoff=64*np.finfo(float).eps*np.einsum('i,kij,j->k',abs(q),abs(self.port_gram),abs(q))
        variance=np.maximum(0,quadratic+roundoff)
        regions=feedback*j0/RT*np.cosh(abs(96485.33212*eta/(2*RT))+spread/(2*RT))*np.sqrt(self.reaction_area*variance)
        return {'absolute_bound_A_m3':float(regions.sum()),'regional_bounds_A_m3':regions,
            'scope':'instantaneous BV reaction closure at represented concentration; not a nonlinear trajectory bound'}

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'online.npz',H=self.H,weights=self.weights,reaction_area=self.reaction_area,
            stiffness=self.stiffness.toarray(),stress_mean=self.stress_mean,stress_second=self.stress_second,port_gram=self.port_gram,port_box=self.port_box)
        (path/'manifest.json').write_text(json.dumps({'schema':'morqex-microstructure-port-2','metadata':self.descriptor,
            'online_sha256':hashlib.sha256((path/'online.npz').read_bytes()).hexdigest()},indent=2))

    @classmethod
    def load(cls,path):
        path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
        if manifest['schema']!='morqex-microstructure-port-2':raise ValueError('Unknown operator schema')
        if hashlib.sha256((path/'online.npz').read_bytes()).hexdigest()!=manifest['online_sha256']:raise ValueError('Online checksum mismatch')
        with np.load(path/'online.npz',allow_pickle=False) as a:
            result=cls(a['H'],a['weights'],a['reaction_area'],sparse.csr_matrix(a['stiffness']),a['stress_mean'],a['stress_second'],a['port_gram'],a['port_box'],manifest['metadata'])
        if result.H.shape!=(result.dimension+2,)*2 or np.any(result.weights<=0) or np.any(result.reaction_area<0):raise ValueError('Invalid operator dimensions or physical weights')
        return result


def save_operator(operator,workspace,name):
    path=workspace.path('operators','microstructure_dfn',name);path.mkdir(parents=True,exist_ok=True)
    operator.detach().save(path)
    full=operator.full
    np.savez_compressed(path/'localization.npz',P=operator.prolongation,R=operator.restriction,stress=operator.stress_map,
        strain=operator.strain_map,displacement=operator.displacement_map,weights_full=full.diffusion.weights,
        voxel_ids=full.diffusion.voxel_ids,labels=full.structure.labels,spacing=full.volume.spacing,
        points=full.mesh.points,cells=full.mesh.cells,volumes=full.mesh.volumes,phases=full.mesh.phases,
        element_active_cell=full._element_cells,active_volumes=full.diffusion.volumes,reaction_area_full=full.area,
        edge_i=sparse.triu(full.diffusion.stiffness,k=1).tocoo().row,
        edge_j=sparse.triu(full.diffusion.stiffness,k=1).tocoo().col,
        edge_conductance=-sparse.triu(full.diffusion.stiffness,k=1).tocoo().data)
    meta=json.loads((path/'manifest.json').read_text());meta['localization_sha256']=hashlib.sha256((path/'localization.npz').read_bytes()).hexdigest()
    (path/'manifest.json').write_text(json.dumps(meta,indent=2));return path


class LocalizationMap:
    def __init__(self,path):
        path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
        if hashlib.sha256((path/'localization.npz').read_bytes()).hexdigest()!=manifest['localization_sha256']:raise ValueError('Localization checksum mismatch')
        self.online=OnlineMicrostructureOperator.load(path)
        with np.load(path/'localization.npz',allow_pickle=False) as a:self.arrays={k:a[k] for k in a.files}

    def recover(self,concentration,axial_strain,temperature_deviation,*,selection=None):
        c=np.asarray(concentration);shape=c.shape[:-1]
        q=np.concatenate([np.broadcast_to(axial_strain,shape)[...,None],c-self.online.reference_concentration,
            np.broadcast_to(temperature_deviation,shape)[...,None]],axis=-1)
        a=self.arrays;stress=a['stress'] if selection is None else a['stress'][selection]
        strain=a['strain'] if selection is None else a['strain'][selection]
        return {'concentration':c@a['P'].T,'stress':np.einsum('eik,...k->...ei',stress,q),
            'strain':np.einsum('eik,...k->...ei',strain,q),'displacement':np.einsum('eik,...k->...ei',a['displacement'],q)}
