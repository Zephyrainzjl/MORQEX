"""Conservative microstructure restriction/localization and energy operators.

The micro concentration states REPLACE the DFN radial states. Connected-region
aggregation is a variational finite-volume approximation, not a unique map from
SOC. Clustering/static condensation are established methods, not novelty claims.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components,dijkstra
from scipy.sparse.linalg import spsolve
from morqex.transport import VoxelDiffusion
from morqex.mechanics import TetMesh,ElasticitySolver,isotropic_stiffness
from morqex.energy import ResponseBuilder


def phase_transport(volume,phases,axis_xyz=0):
    """Unit-conductivity TPFA cell problem, Dirichlet x ends/insulated sides.

    Disconnected floating components carry no current. Boundary half-cell
    distances are included. Returns a dimensionless directional efficiency.
    """
    if axis_xyz not in (0,1,2):raise ValueError('XYZ axis required')
    fv=VoxelDiffusion(volume,active_phases=phases,diffusivity=1.)
    axis=2-axis_xyz;shape=volume.data.shape;h=volume.spacing[axis]
    length=shape[axis]*h
    boundary=2*np.prod(volume.spacing)/h**2/fv.rve_volume
    left=(fv.coordinates_zyx[:,axis]==0)*boundary
    right=(fv.coordinates_zyx[:,axis]==shape[axis]-1)*boundary
    anchored=np.isin(fv.components,np.unique(fv.components[left+right>0]))
    potential=np.zeros(len(fv.weights));matrix=fv.stiffness+sparse.diags(left+right)
    if anchored.any():potential[anchored]=spsolve(matrix[anchored][:,anchored].tocsc(),right[anchored])
    current=float(right@(1-potential));efficiency=current*length**2
    energy=float(potential@(fv.stiffness@potential)+left@(potential**2)+right@((1-potential)**2))*length**2
    if not -1e-12<=efficiency<=fv.weights.sum()+1e-10:raise RuntimeError('Transport bounds violated')
    return {'efficiency':max(0.,efficiency),'energy_efficiency':energy,
        'relative_residual':float(np.linalg.norm((matrix@potential-right)[anchored])/max(np.linalg.norm(right),1e-30)),
        'phase_fraction':float(fv.weights.sum()),'boundary':'Dirichlet ends, insulated transverse faces'}


def _connected_to_x_boundary(volume,phases,side):
    fv=VoxelDiffusion(volume,active_phases=phases,diffusivity=1.)
    coordinate=0 if side=='left' else volume.data.shape[2]-1
    ids=np.unique(fv.components[fv.coordinates_zyx[:,2]==coordinate])
    result=np.zeros(volume.data.size,dtype=bool)
    result[fv.voxel_ids]=np.isin(fv.components,ids)
    return result.reshape(volume.data.shape)


@dataclass
class MicrostructureOperator:
    full: object
    partition: np.ndarray
    prolongation: np.ndarray
    restriction: np.ndarray
    weights: np.ndarray
    reaction_area: np.ndarray
    stiffness: object
    H: np.ndarray
    stress_map: np.ndarray
    strain_map: np.ndarray
    displacement_map: np.ndarray
    strategy: str

    @property
    def dimension(self):return len(self.weights)
    @property
    def active_fraction(self):return float(self.weights.sum())
    @property
    def area_density(self):return float(self.reaction_area.sum())
    @property
    def reference_concentration(self):return self.full.reference
    @property
    def transport(self):return self.full.transport
    @property
    def porosity(self):return self.full.porosity

    def restrict(self,values):return np.asarray(values)@self.restriction.T
    def lift(self,values):return np.asarray(values)@self.prolongation.T

    def chemical_potential(self,concentration,axial_strain,temperature_deviation):
        c=np.asarray(concentration)-self.full.reference
        return (c@self.H[1:-1,1:-1].T+np.asarray(axial_strain)[...,None]*self.H[1:-1,0]
            +np.asarray(temperature_deviation)[...,None]*self.H[1:-1,-1])/self.weights

    def recover(self,concentration,axial_strain,temperature_deviation):
        c=np.asarray(concentration);shape=c.shape[:-1]
        q=np.concatenate([np.broadcast_to(axial_strain,shape)[...,None],c-self.full.reference,
            np.broadcast_to(temperature_deviation,shape)[...,None]],axis=-1)
        return {'concentration':self.lift(c),'stress':np.einsum('eik,...k->...ei',self.stress_map,q),
            'strain':np.einsum('eik,...k->...ei',self.strain_map,q),
            'displacement':np.einsum('eik,...k->...ei',self.displacement_map,q),
            'elastic_energy_density':.5*np.einsum('...i,ij,...j->...',q,self.H,q)}

    def metadata(self):
        return {'micro_cells':self.full.n,'online_regions':self.dimension,'strategy':self.strategy,
            'reference_concentration':self.full.reference,'reference_temperature_K':self.full.reference_T,
            'active_fraction':self.active_fraction,'reaction_area_m_inv':self.area_density,
            'geometry_fingerprint':self.full.fingerprint,'transport':self.full.transport,'material':self.full.material_record,
            'state_meaning':'intrinsic region concentration, mol/m3; no additional radial lithium states',
            'restriction':'volume-weighted; R P=I; w_r^T R=w_full^T','localization':'piecewise-constant concentration and exact FEM response for that represented concentration'}


    def transfer(self,new_operator,values):
        if self.full.fingerprint!=new_operator.full.fingerprint:raise ValueError('Different microstructure/material operator')
        lifted=new_operator.prolongation@new_operator.restriction@self.prolongation
        if not np.allclose(lifted,self.prolongation,rtol=0,atol=1e-12):raise ValueError('Only nested enrichment preserves the represented field and energy')
        return new_operator.restrict(self.lift(values))

    def detach(self):
        from .microstructure_cache import OnlineMicrostructureOperator
        volumes=self.full.mesh.volumes;fraction=volumes.sum()/self.full.mesh.rve_volume
        mean=np.einsum('e,eij->ij',volumes/self.full.mesh.rve_volume,self.stress_map)
        second=np.einsum('e,eik,eil->ikl',volumes/volumes.sum(),self.stress_map,self.stress_map)
        cell=self.full._element_cells;active=cell>=0;mu_map=np.zeros((self.full.n,self.dimension+2))
        trace=self.stress_map[active,:3,:].sum(axis=1)
        for k in range(self.dimension+2):
            np.add.at(mu_map[:,k],cell[active],-self.full._omega/3*volumes[active]*trace[:,k])
        mu_map/=self.full.diffusion.volumes[:,None]
        delta=mu_map-self.H[1:-1,:][self.partition]/self.weights[self.partition,None]
        port_gram=np.zeros((self.dimension,self.dimension+2,self.dimension+2));port_box=np.zeros((self.dimension,self.dimension+2))
        for region in range(self.dimension):
            selected=(self.partition==region)&(self.full.area>0)
            if selected.any():
                port_gram[region]=np.einsum('n,ni,nj->ij',self.full.area[selected],delta[selected],delta[selected])
                port_box[region]=np.max(abs(delta[selected]),axis=0)
        descriptor=self.metadata()|{'reference_concentration':self.full.reference,'reference_temperature_K':self.full.reference_T,
            'porosity':self.porosity,'solid_fraction':float(fraction),'fingerprint':hashlib.sha256(self.full.fingerprint.encode()+self.partition.tobytes()).hexdigest()}
        return OnlineMicrostructureOperator(self.H.copy(),self.weights.copy(),self.reaction_area.copy(),self.stiffness.copy(),mean,second,port_gram,port_box,descriptor)

    def save(self,workspace,name):
        from .microstructure_cache import save_operator
        return save_operator(self,workspace,name)


class MicrostructureElectrode:
    """Full active-voxel diffusion and fixed-C FEM energy on the SAME geometry."""
    def __init__(self,structure,provider,parameters,domain,*,thermal_expansion=1e-5,cbd_young=1e9):
        if domain not in {'negative','positive'}:raise ValueError('Electrode domain required')
        self.structure=structure;self.volume=provider.to_volume(structure);self.domain=domain
        active=[p for p,k in structure.phase_kinds.items() if k=='active']
        pores=[p for p,k in structure.phase_kinds.items() if k=='pore']
        solid=[p for p,k in structure.phase_kinds.items() if k in {'active','cbd','inert'}]
        if not active or not pores:raise ValueError('Active and pore phases required')
        self.diffusion=VoxelDiffusion(self.volume,active_phases=active,diffusivity=1.,reactive_phases=pores)
        fv=self.diffusion;self.n=len(fv.weights)
        self.reference=float(parameters[f'Initial concentration in {domain} electrode [mol.m-3]'])
        self.reference_T=float(parameters['Reference temperature [K]'])
        Domain=domain.capitalize();young=float(parameters[f"{Domain} electrode Young's modulus [Pa]"])
        nu=float(parameters[f"{Domain} electrode Poisson's ratio"])
        omega=float(parameters[f'{Domain} electrode partial molar volume [m3.mol-1]'])
        self.material_record={'young_Pa':young,'poisson':nu,'molar_volume_m3_mol':omega,'thermal_expansion_per_K':thermal_expansion,
            'cbd_young_Pa':cbd_young,'scope':'fixed linear elasticity; synthetic CBD/thermal alpha, not material calibration'}
        self.transport={'electrolyte':phase_transport(self.volume,pores),'electrode':phase_transport(self.volume,solid)}
        if min(v['efficiency'] for v in self.transport.values())<1e-10:raise ValueError('SVE has no through-thickness ionic/electronic percolation')
        pore_access=_connected_to_x_boundary(self.volume,pores,'right' if domain=='negative' else 'left')
        solid_access=_connected_to_x_boundary(self.volume,solid,'left' if domain=='negative' else 'right')
        reactive=fv.reactive.copy()
        for face in np.flatnonzero(reactive):
            cell=fv.face_cells[face];coord=fv.coordinates_zyx[cell]
            other=coord+fv.face_normals[face][::-1].astype(int)
            reactive[face]=pore_access[tuple(other)] and solid_access[tuple(coord)]
        self.reactive=reactive
        self.area=np.bincount(fv.face_cells,weights=fv.face_areas*reactive,minlength=self.n)/fv.rve_volume
        if self.area.sum()<=0:raise ValueError('No ionically and electronically accessible reactive interfaces')
        self.mesh=TetMesh.from_volume(self.volume,solid_phases=solid)
        # Edge/corner touching voxels are not bonded by an area-bearing contact.
        # Duplicate their shared nodes across face-disconnected solid components.
        solid_graph=VoxelDiffusion(self.volume,active_phases=solid,diffusivity=1.)
        if solid_graph.n_components>1:
            labels=np.full(self.volume.data.size,-1,dtype=int);labels[solid_graph.voxel_ids]=solid_graph.components
            component=labels[self.mesh.voxel_ids]
            pairs,inverse=np.unique(np.column_stack([np.repeat(component,4),self.mesh.cells.ravel()]),axis=0,return_inverse=True)
            self.mesh=TetMesh(self.mesh.points[pairs[:,1]],inverse.reshape(-1,4),self.mesh.phases,self.mesh.box,self.mesh.voxel_ids)
        self.material_record['mechanical_contact']='face-connected solid; corner-only and edge-only components are not tied'
        tensor={p:isotropic_stiffness(young,nu) if p in active else isotropic_stiffness(cbd_young,.3) for p in solid}
        self.solver=ElasticitySolver(self.mesh,np.array([tensor[int(p)] for p in self.mesh.phases]),boundary='kubc')
        parent=np.full(self.volume.data.size,-1,dtype=int);parent[fv.voxel_ids]=np.arange(self.n)
        cell=parent[self.mesh.voxel_ids];selected=cell>=0
        self._element_cells=cell;self._selected=selected;self._omega=omega;self._alpha=thermal_expansion;self._full_response=None
        self.porosity=float(np.isin(structure.labels,pores).mean())
        self.fingerprint=hashlib.sha256((structure.fingerprint+self.mesh.fingerprint+domain).encode()+np.array([young,nu,omega,thermal_expansion,cbd_young,self.reference,self.reference_T]).tobytes()+Path(__file__).read_bytes()).hexdigest()

    def _response(self,P):
        m=P.shape[1];ne=len(self.mesh.cells)
        if ne*6*(m+7)*8>128*2**20:raise MemoryError('Field allocation exceeds 128 MiB; reduce regions or stream localization')
        modes=np.zeros((ne,6,m+1));cell=self._element_cells;selected=self._selected
        modes[selected,:3,:m]=self._omega/3*P[cell[selected],None,:]
        modes[:,:3,-1]=self._alpha
        response=ResponseBuilder(self.solver,modes).exact()
        response.metadata.update(input_units=['1']*6+['mol.m-3']*m+['K'],geometry_fingerprint=self.structure.fingerprint,
            closure='actual represented nonuniform concentration; no sphere mapping')
        return response

    @property
    def response(self):
        if self.n>128:raise MemoryError('Full identity reference capped at 128 cells; regional construction remains available')
        if self._full_response is None:self._full_response=self._response(np.eye(self.n))
        return self._full_response

    @property
    def H(self):
        idx=[0,*range(6,6+self.n+1)]
        return self.response.H[np.ix_(idx,idx)]

    def partition(self,target,strategy='geometry'):
        fv=self.diffusion
        if strategy not in {'geometry','mechanics'}:raise ValueError('Unknown partition strategy')
        if target<fv.n_components:raise ValueError('At least one region per disconnected diffusion component required')
        groups=[np.flatnonzero(fv.components==k) for k in range(fv.n_components)]
        features=fv.centers/np.ptp(fv.centers,axis=0).clip(min=1e-30)
        if strategy=='mechanics':
            # Uniform-eigenstrain stress features avoid an N-by-N response.
            response=self._response(np.ones((self.n,1)))
            mechanical=np.zeros((self.n,6));active=self._selected;cell=self._element_cells[active]
            for k in range(6):np.add.at(mechanical[:,k],cell,self.mesh.volumes[active]*response.stress_operator[active,k,6])
            mechanical/=fv.volumes[:,None]
            mechanical/=max(np.linalg.norm(mechanical,axis=0).max(),1e-30)
            mechanical*=np.sqrt(self.n)
            features=np.column_stack([features,mechanical])
        edges=fv.stiffness.tocoo();off=edges.row!=edges.col
        i,j=edges.row[off],edges.col[off]
        costs=np.linalg.norm(features[i]-features[j],axis=1)
        graph=sparse.coo_matrix((costs,(i,j)),shape=fv.stiffness.shape).tocsr()
        while len(groups)<min(target,self.n):
            choices=[(len(g),k) for k,g in enumerate(groups) if len(g)>1]
            _,k=max(choices);group=groups.pop(k);f=features[group];_,_,v=np.linalg.svd(f-f.mean(axis=0),full_matrices=False)
            ordered=np.argsort((f-f.mean(axis=0))@v[0],kind='stable')
            distances=dijkstra(graph[group][:,group],directed=False,indices=[ordered[0],ordered[-1]])
            labels=np.argmin(distances,axis=0)
            groups.extend(group[labels==j] for j in (0,1))
        labels=np.empty(self.n,dtype=int)
        for k,g in enumerate(groups):labels[g]=k
        return self.reduce(labels,strategy)

    def reduce(self,partition=None,strategy='full'):
        labels=np.arange(self.n) if partition is None else np.asarray(partition,dtype=int)
        if labels.shape!=(self.n,) or not np.array_equal(np.unique(labels),np.arange(labels.max()+1)):raise ValueError('Contiguous region IDs required')
        for k in np.unique(labels):
            if len(np.unique(self.diffusion.components[labels==k]))!=1:raise ValueError('Region mixes disconnected lithium inventories')
        if len(np.unique(labels))==self.n and self.n>128:raise MemoryError('Full identity reference capped at 128 active cells')
        P=np.eye(int(labels.max())+1)[labels];w=P.T@self.diffusion.weights
        R=(P.T*self.diffusion.weights)/w[:,None]
        response=self._response(P);idx=[0,*range(6,6+P.shape[1]+1)]
        return MicrostructureOperator(self,labels.copy(),P,R,w,P.T@self.area,
            sparse.csr_matrix(P.T@self.diffusion.stiffness@P),response.H[np.ix_(idx,idx)],
            response.stress_operator[:,:,idx],response.strain_operator[:,:,idx],response.displacement_operator[:,:,idx],strategy)
