"""Small-reference FEM construction of a positive, mass-lumped POU operator."""
from dataclasses import asdict
import hashlib
import numpy as np
from scipy import sparse
from scipy import ndimage
from ..electrochemistry.microstructure_operator import MicrostructureOperator
from ..electrochemistry.microstructure_cache import OnlineMicrostructureOperator
from .positive import ComponentBasis

class POUOperator(MicrostructureOperator):
    def metadata(self):
        data=super().metadata();data.update(basis=self.basis_recipe,transport_law=self.transport_law,
            state_meaning='positive POU nodal concentration, mol/m3, with lumped variational chemical quadrature',
            restriction='lumped projection; not RP=I',localization='positive componentwise partition of unity; exact FEM response for the represented field',
            port_certificate='not the piecewise-constant certificate')
        return data
    def detach(self):
        volumes=self.full.mesh.volumes;fraction=volumes.sum()/self.full.mesh.rve_volume;m=self.dimension+2
        mean=np.einsum('e,eij->ij',volumes/self.full.mesh.rve_volume,self.stress_map)
        second=np.einsum('e,eik,eil->ikl',volumes/volumes.sum(),self.stress_map,self.stress_map)
        descriptor=self.metadata()|{'reference_concentration':self.reference_concentration,'reference_temperature_K':self.full.reference_T,
            'porosity':self.porosity,'solid_fraction':float(fraction),'fingerprint':hashlib.sha256(self.full.fingerprint.encode()+self.prolongation.tobytes()).hexdigest()}
        return OnlineMicrostructureOperator(self.H,self.weights,self.reaction_area,self.stiffness,mean,second,np.zeros((self.dimension,m,m)),np.zeros((self.dimension,m)),descriptor)

def build_pou_operator(full,*,target,transport_law,relax_transverse=False):
    shape=full.structure.labels.shape;mask=np.zeros(full.structure.labels.size,bool);mask[full.diffusion.voxel_ids]=True;mask=mask.reshape(shape)
    distance=ndimage.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1,1:-1].ravel()[full.diffusion.voxel_ids]
    fv=full.diffusion;components=fv.components+1;features=np.zeros((3,full.n));features[0]=distance/max(1.,distance.max())
    # A local surface forcing feature is independent of distance on irregular
    # particles; a through-plane coordinate provides directional enrichment.
    features[1]=full.area/np.maximum(fv.weights,1e-30);features[1]/=max(1.,features[1].max())
    x=fv.centers[:,0];features[2]=(x-x.min())/max(float(np.ptp(x)),1e-30)
    basis=ComponentBasis.allocate(components,features,target=target);P=basis.evaluate(components,features)
    weights=P.T@fv.weights
    if np.any(weights<1e-14):raise ValueError('Degenerate POU nodes; refine feature selection rather than retaining zero-volume states')
    response=full._response(P);idx=[0,*range(6,6+P.shape[1]+1)];R=(P.T*fv.weights)/weights[:,None]
    transform=np.zeros((len(response.H),len(idx)));transform[idx]=np.eye(len(idx))
    if relax_transverse:
        free=np.arange(1,6);transform[free]=-np.linalg.solve(response.H[np.ix_(free,free)],response.H[np.ix_(free,idx)])
    result=POUOperator(full,np.argmax(P,axis=1),P,R,weights,P.T@full.area,sparse.csr_matrix(P.T@fv.stiffness@P),transform.T@response.H@transform,
        response.stress_operator@transform,response.strain_operator@transform,response.displacement_operator@transform,'positive-component-POU')
    result.basis_recipe=basis.metadata()|{'transverse_macroscopic_strains':'energy-minimized, zero mean transverse stress' if relax_transverse else 'fixed to zero',
        'fluctuation_boundary':'KUBC of the reference FEM; not a periodic micro boundary'}
    result.transport_law=asdict(transport_law);return result
