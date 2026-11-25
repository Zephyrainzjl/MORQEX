"""Instantaneous BV/implicit-film closure bound for evolving cohesive kernels.

The augmented state [q, Fq/(1+s rho)] restores linear localization in the state
coefficients. This is not a concentration/DFN trajectory error certificate.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import qr
from morqex.electrochemistry.aging_cache import AgingLocalization,load_aging_kernel


def _sources():
    root=Path(__file__).parents[1]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'electrochemistry/aging_dfn.py']}


def compile_reaction_certificate(kernel_path,destination=None):
    """Offline only: integrate conjugate stress and QR regional area norms."""
    kernel_path=Path(kernel_path);local=AgingLocalization(kernel_path);a=local.arrays;op=local.kernel.operator;c=local.kernel.cohesive
    P=a['P']
    if not np.all((P==0)|(P==1)) or not np.all(P.sum(axis=1)==1):raise ValueError('Piecewise constant physical concentration regions required')
    S=np.concatenate([a['stress_direct'],a['stress_modes']],axis=-1);cells=a['element_cells'];active=cells>=0
    material=op.metadata()['material'];catalog=material.get('phase_catalog')
    if catalog:
        phases=a['labels'].ravel()[a['element_voxel_ids'][active]]
        beta=np.array([catalog['phases'][str(int(p))]['chemical_strain_per_concentration'] for p in phases])
    else:beta=np.broadcast_to(np.r_[np.full(3,material['molar_volume_m3_mol']/3),np.zeros(3)],(int(active.sum()),6))
    conjugate=-np.einsum('ei,eij->ej',beta,S[active]);mu=np.zeros((len(P),S.shape[-1]))
    volumes=np.bincount(cells[active],weights=a['volumes'][active],minlength=len(P))
    for k in range(mu.shape[1]):np.add.at(mu[:,k],cells[active],a['volumes'][active]*conjugate[:,k])
    mu/=volumes[:,None]
    coarse=np.concatenate([c.direct[1:-1],-c.factors.T[1:-1]],axis=1)/op.weights[:,None]
    difference=mu-P@coarse;area=a['area_full'];arrays={'box':np.zeros((op.dimension,difference.shape[1]))}
    for k in range(op.dimension):
        selected=(P[:,k]>0)&(area>0)
        if selected.any():
            _,root=qr(np.sqrt(area[selected,None])*difference[selected],mode='economic');arrays[f'root_{k}']=root
            arrays['box'][k]=np.max(abs(difference[selected]),axis=0)
        else:arrays[f'root_{k}']=np.zeros((0,difference.shape[1]))
    destination=Path(destination) if destination is not None else kernel_path/'reaction_certificate';destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'online.npz',**arrays)
    record={'schema':'morqex-aging-reaction-port-1','kernel_manifest_sha256':hashlib.sha256((kernel_path/'manifest.json').read_bytes()).hexdigest(),
        'coefficient_sha256':hashlib.sha256((destination/'online.npz').read_bytes()).hexdigest(),'source_sha256':_sources(),
        'scope':'same represented c,E,T,d and localized mechanical kernel; passive monotone side laws and nonnegative common film resistance; no trajectory/kernel/space error'}
    (destination/'manifest.json').write_text(json.dumps(record,indent=2));return destination


class AgingReactionCertificate:
    def __init__(self,path,kernel_path):
        path=Path(path);kernel_path=Path(kernel_path);m=json.loads((path/'manifest.json').read_text())
        if m['schema']!='morqex-aging-reaction-port-1' or m['source_sha256']!=_sources():raise ValueError('Reaction certificate schema/source mismatch')
        if hashlib.sha256((kernel_path/'manifest.json').read_bytes()).hexdigest()!=m['kernel_manifest_sha256']:raise ValueError('Reaction certificate/kernel mismatch')
        if hashlib.sha256((path/'online.npz').read_bytes()).hexdigest()!=m['coefficient_sha256']:raise ValueError('Reaction certificate checksum mismatch')
        self.kernel=load_aging_kernel(kernel_path);self.scope=m['scope']
        with np.load(path/'online.npz',allow_pickle=False) as data:self.arrays={n:data[n] for n in data.files}
    def bound(self,q,damage,j0,eta,temperature,*,feedback=1.,mechanical_residual=None,film_residual=None,film_resistance=None):
        op=self.kernel.operator;c=self.kernel.cohesive;q=np.asarray(q,float);j0=np.asarray(j0,float);eta=np.asarray(eta,float);m=op.dimension
        if q.shape!=(m+2,) or j0.shape!=(m,) or eta.shape!=(m,) or not np.isfinite(np.r_[q,j0,eta,temperature,damage,feedback]).all() or np.any(j0<0) or temperature<=0 or not 0<=damage<=1 or not 0<=feedback<=1:raise ValueError('Invalid reaction certificate input')
        zero=np.zeros(m)
        correction=np.abs(zero if mechanical_residual is None else mechanical_residual)
        residual=np.abs(zero if film_residual is None else film_residual)
        resistance=np.asarray(zero if film_resistance is None else film_resistance)
        if any(v.shape!=(m,) or not np.isfinite(v).all() for v in [correction,residual,resistance]) or np.any(resistance<0):raise ValueError('Finite per-region residuals and nonnegative resistance required')
        retention=c.residual+(1-c.residual)*(1-damage)**2
        p=np.r_[q,c.factors@q/(1+(retention-1)*c.eigenvalues)]
        spread=feedback*(self.arrays['box']@abs(p)+correction);RT=8.31446261815324*temperature
        argument=np.abs(96485.33212*eta/(2*RT))+spread/(2*RT)+96485.33212*resistance*residual/(2*RT)
        regions=[]
        for k,A in enumerate(op.reaction_area):
            if not A:regions.append(0.);continue
            root=self.arrays[f'root_{k}'];value=root@p
            rounding=64*np.finfo(float).eps*np.linalg.norm(root)*np.linalg.norm(p)
            norm=feedback*(np.linalg.norm(value)+rounding+np.sqrt(A)*correction[k])
            gain=0. if j0[k]==0 else np.inf if argument[k]>700 else j0[k]/RT*np.cosh(argument[k])
            sensitivity=0. if norm==0 else gain*np.sqrt(A)*norm
            # Covers both total current and host-intercalation current when
            # the coarse algebraic film equation has a numerical residual.
            amplification=2. if resistance[k]==0 else max(2.,96485.33212*gain*resistance[k])
            algebraic=0. if residual[k]==0 else A*residual[k]*amplification
            regions.append((1-damage)*(sensitivity+algebraic))
        return {'absolute_bound_A_m3':float(np.sum(regions)),'regional_bounds_A_m3':np.asarray(regions),'scope':self.scope}
