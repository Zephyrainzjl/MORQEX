"""Separate cohesive online scalars from optionally loaded field localization."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .microstructure_cache import OnlineMicrostructureOperator

@dataclass
class CohesiveOnline:
    direct:np.ndarray
    factors:np.ndarray
    eigenvalues:np.ndarray
    residual:float
    def __post_init__(self):
        for name in ['direct','factors','eigenvalues']:
            value=np.array(getattr(self,name),dtype=float,copy=True)
            if not np.isfinite(value).all():raise ValueError('Finite cohesive coefficients required')
            value.flags.writeable=False;setattr(self,name,value)
        n=len(self.direct)
        if self.direct.shape!=(n,n) or self.factors.ndim!=2 or self.factors.shape[1]!=n or self.eigenvalues.shape!=(len(self.factors),):raise ValueError('Invalid cohesive coefficient shapes')
        if np.any((self.eigenvalues<0)|(self.eigenvalues>1+1e-12)) or not 1e-6<=self.residual<1:raise ValueError('Invalid cohesive spectrum/residual stiffness')
        if not np.allclose(self.direct,self.direct.T,rtol=1e-11,atol=1e-7):raise ValueError('Cohesive energy reciprocity lost')
    def hessian(self,d):
        if not np.isfinite(d) or not 0<=d<=1:raise ValueError('Invalid damage')
        den=1+(self.residual+(1-self.residual)*(1-d)**2-1)*self.eigenvalues
        return self.direct-(self.factors.T/den)@self.factors

def save_aging_operator(path,op,cohesive):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);op.detach().save(path/'base')
    np.savez_compressed(path/'cohesive_online.npz',direct=cohesive.direct,factors=cohesive.factors,eigenvalues=cohesive.eigenvalues,residual=cohesive.residual)
    names=['displacement_direct','displacement_modes','strain_direct','strain_modes','stress_direct','stress_modes','interface_pairs','interface_areas']
    arrays={n:getattr(cohesive,n) for n in names}
    arrays.update(P=op.prolongation,R=op.restriction,weights_full=op.full.diffusion.weights,labels=op.full.structure.labels,
        voxel_ids=op.full.diffusion.voxel_ids,element_cells=op.full._element_cells,element_voxel_ids=cohesive.mesh.voxel_ids,points=cohesive.mesh.points,cells=cohesive.mesh.cells,volumes=cohesive.mesh.volumes,
        area_full=op.full.area,reaction_area=op.reaction_area)
    np.savez_compressed(path/'localization.npz',**arrays)
    files=['cohesive_online.npz','localization.npz']
    if hasattr(cohesive,'certificate_arrays'):
        np.savez_compressed(path/'certificate.npz',**cohesive.certificate_arrays);files.append('certificate.npz')
    metadata={'schema':'morqex-aging-kernel-1','cohesive':cohesive.metadata,'base':op.metadata(),
        'sha256':{n:hashlib.sha256((path/n).read_bytes()).hexdigest() for n in files}}
    if hasattr(op,'regional_chemistry'):metadata['regional_chemistry']=op.regional_chemistry.metadata()
    (path/'manifest.json').write_text(json.dumps(metadata,indent=2));return path

def load_aging_kernel(path):
    from .aging_dfn import EnergyKernel
    path=Path(path);metadata=json.loads((path/'manifest.json').read_text())
    if metadata['schema']!='morqex-aging-kernel-1':raise ValueError('Unknown kernel schema')
    if hashlib.sha256((path/'cohesive_online.npz').read_bytes()).hexdigest()!=metadata['sha256']['cohesive_online.npz']:raise ValueError('Cohesive kernel checksum mismatch')
    with np.load(path/'cohesive_online.npz',allow_pickle=False) as a:coh=CohesiveOnline(a['direct'],a['factors'],a['eigenvalues'],float(a['residual']))
    operator=OnlineMicrostructureOperator.load(path/'base')
    if 'regional_chemistry' in metadata:
        from ..doping import RegionalChemistry,DopantLaw
        profile=metadata['regional_chemistry'];operator.regional_chemistry=RegionalChemistry(profile['fraction'],DopantLaw(**profile['law']))
        if len(operator.regional_chemistry.fraction)!=operator.dimension:raise ValueError('Dopant profile/kernel dimension mismatch')
    return EnergyKernel(operator,coh)

class AgingLocalization:
    def __init__(self,path):
        path=Path(path);self.kernel=load_aging_kernel(path);metadata=json.loads((path/'manifest.json').read_text())
        if hashlib.sha256((path/'localization.npz').read_bytes()).hexdigest()!=metadata['sha256']['localization.npz']:raise ValueError('Field checksum mismatch')
        with np.load(path/'localization.npz',allow_pickle=False) as a:self.arrays={n:a[n] for n in a.files}
    def recover(self,c,E,theta,d):
        q=np.concatenate([np.asarray(E)[...,None],np.asarray(c)-self.kernel.operator.reference_concentration,np.broadcast_to(theta,np.asarray(E).shape)[...,None]],axis=-1)
        k=self.kernel.cohesive;den=1+(k.residual+(1-k.residual)*(1-np.asarray(d)[...,None])**2-1)*k.eigenvalues;z=q@k.factors.T/den
        fields={n:np.einsum('...j,eij->...ei',q,self.arrays[n+'_direct'])+np.einsum('...k,eik->...ei',z,self.arrays[n+'_modes']) for n in ['stress','strain','displacement']}
        pairs=self.arrays['interface_pairs'];fields['opening']=fields['displacement'][...,pairs[:,1],:]-fields['displacement'][...,pairs[:,0],:]
        fields['concentration']=np.asarray(c)@self.arrays['P'].T;return fields

def load_cohesive_certificate(path):
    """Load small residual coefficients without loading localization fields."""
    path=Path(path);metadata=json.loads((path/'manifest.json').read_text())
    expected=metadata['sha256'].get('certificate.npz')
    if expected is None:raise ValueError('Kernel has no cohesive compression certificate')
    if hashlib.sha256((path/'certificate.npz').read_bytes()).hexdigest()!=expected:raise ValueError('Certificate checksum mismatch')
    with np.load(path/'certificate.npz',allow_pickle=False) as data:return {n:data[n] for n in data.files}
