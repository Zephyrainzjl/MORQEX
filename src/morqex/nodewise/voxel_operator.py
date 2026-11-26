"""Streamed, factor-reusing Tet4 response with full-grid voxel output maps."""
import time
import numpy as np
from scipy import sparse
from ..electrochemistry.microstructure_cache import OnlineMicrostructureOperator
from ..cohesive import CohesiveResponse


class VoxelOperator:
    def __init__(self,full,labels,P,weights,areas,K,H,stress,strain,displacement):
        self.full=full;self.partition=labels;self.prolongation=P;self.restriction=(P.T*full.diffusion.weights)/weights[:,None]
        self.weights=weights;self.reaction_area=areas;self.stiffness=K;self.H=H
        self.stress_map=stress;self.strain_map=strain;self.displacement_map=displacement
    @property
    def dimension(self):return len(self.weights)
    @property
    def active_fraction(self):return float(self.weights.sum())
    @property
    def area_density(self):return float(self.reaction_area.sum())
    @property
    def reference_concentration(self):return self.full.reference
    @property
    def porosity(self):return self.full.porosity
    @property
    def transport(self):return self.full.transport
    def metadata(self):
        from ..electrochemistry.microstructure_operator import MicrostructureOperator
        self.strategy='component-preserving geometric partition / streamed voxel FEM'
        out=MicrostructureOperator.metadata(self);out['localization']='exact Tet4 voxel-volume averages for represented concentration, fixed intact geometry/C'
        return out
    def detach(self):
        import hashlib
        N=self.full.structure.labels.size;solid=np.isin(self.full.structure.labels.ravel(),[p for p,k in self.full.structure.phase_kinds.items() if k!='pore'])
        mean=self.stress_map.sum(axis=0)/N;S=self.stress_map[solid]
        second=np.einsum('eik,eil->ikl',S,S)/solid.sum();m=self.dimension+2
        beta=self.full._chemical_tensors.reshape(-1,6,6)[:,0,:]
        material_voxels=self.full.mesh.voxel_ids[::6];mapping=np.full(N,-1,int);mapping[material_voxels]=np.arange(len(material_voxels))
        active=self.full.diffusion.voxel_ids
        mu=-np.einsum('ei,eij->ej',beta[mapping[active]],self.stress_map[active])
        delta=mu-self.H[1:-1,:][self.partition]/self.weights[self.partition,None]
        gram=np.zeros((self.dimension,m,m));box=np.zeros((self.dimension,m))
        for k in range(self.dimension):
            mask=(self.partition==k)&(self.full.area>0)
            if mask.any():gram[k]=np.einsum('n,ni,nj->ij',self.full.area[mask],delta[mask],delta[mask]);box[k]=np.max(abs(delta[mask]),axis=0)
        descriptor=self.metadata()|{'reference_concentration':self.full.reference,'reference_temperature_K':self.full.reference_T,
            'porosity':self.porosity,'solid_fraction':float(solid.mean()),'fingerprint':hashlib.sha256(self.full.fingerprint.encode()+self.partition.tobytes()).hexdigest()}
        return OnlineMicrostructureOperator(self.H,self.weights,self.reaction_area,self.stiffness,mean,second,gram,box,descriptor)


def nodal_to_voxel(mesh,nodal,shape,*,block=768):
    """Stream P1 element averages into parent voxels, including split nodes."""
    N=int(np.prod(shape));result=np.zeros((N,*nodal.shape[1:]));volumes=np.bincount(mesh.voxel_ids,weights=mesh.volumes,minlength=N)
    for begin in range(0,len(mesh.cells),block):
        end=min(len(mesh.cells),begin+block);a=np.asarray(nodal[mesh.cells[begin:end]]).mean(axis=1)
        np.add.at(result,mesh.voxel_ids[begin:end],a*mesh.volumes[begin:end].reshape((-1,)+(1,)*(a.ndim-1)))
    np.divide(result,volumes.reshape((-1,)+(1,)*(result.ndim-1)),out=result,where=volumes.reshape((-1,)+(1,)*(result.ndim-1))>0)
    return result


def build_voxel_intact_operator(full,target=112,*,load_batch=4,field_block=768):
    """No full element-by-mode tensor is allocated; reuse one sparse factor.

    This backend is fixed intact linear mechanics. Contact damage requires the
    cohesive backend, rather than a postprocessed multiplier on these fields.
    """
    from ..mechanics.reference import ElasticitySolver
    started=time.perf_counter();labels=full.partition_labels(target);P=np.eye(labels.max()+1)[labels]
    fv=full.diffusion;weights=P.T@fv.weights;areas=P.T@full.area;m=len(weights);q=m+2
    solver=ElasticitySolver(full.mesh,full.solver.stiffness,boundary='kubc',factorize=False)
    mesh=full.mesh;N=full.structure.labels.size;ids=mesh.voxel_ids[::6];ne=len(mesh.cells);V=mesh.rve_volume
    if not np.array_equal(mesh.voxel_ids,np.repeat(ids,6)):raise ValueError('Streamed response needs the validated six-Tet4 voxel subdivision')
    C=solver.stiffness.reshape(-1,6,6,6)[:,0].copy()
    chemical=full._chemical_tensors.reshape(-1,6,6)[:,0];thermal=full._thermal_tensors.reshape(-1,6,6)[:,0]
    # The compact builder only needs D, Kr and P during multi-RHS solves.
    # Do not retain duplicate assembled gradient/stiffness representations.
    del solver.B,solver.K,solver.constraint,solver.stiffness
    # F=affine strain-eigenstrain, constant in each material voxel.
    direct=np.zeros((q,q));B=np.zeros((solver.P.shape[1],q));u=np.zeros((len(mesh.points)*3,q));u[:,0]=solver.affine[:,0]
    cell_of_voxel=np.full(N,-1,int);cell_of_voxel[fv.voxel_ids]=np.arange(full.n);cell=cell_of_voxel[ids];is_active=cell>=0
    def f_columns(columns):
        out=np.zeros((len(ids),6,len(columns)))
        for k,j in enumerate(columns):
            if j==0:out[:,0,k]=1.
            elif j==q-1:out[:,:,k]=-thermal
            else:out[is_active,:,k]=-chemical[is_active]*P[cell[is_active],j-1,None]
        return out
    # Direct energy assembled in bounded voxel blocks.
    for begin in range(0,len(ids),256):
        end=min(len(ids),begin+256);f=np.zeros((end-begin,6,q));f[:,0,0]=1.;f[:,:,-1]=-thermal[begin:end]
        chosen=is_active[begin:end]
        f[chosen,:,1:-1]=-chemical[begin:end][chosen,:,None]*P[cell[begin:end][chosen],None,:]
        direct+=np.einsum('eik,eij,ejl->kl',f,C[begin:end],f)/N
    maximum_residual=0.
    for begin in range(0,q,load_batch):
        columns=list(range(begin,min(q,begin+load_batch)));f=f_columns(columns)
        stress=np.einsum('eij,ejk->eik',C,f);load=mesh.volumes[:,None,None]*np.repeat(stress,6,axis=0)
        force=-np.asarray(solver.P.T@(solver.D.T@load.reshape(6*ne,-1)));B[:,columns]=force
        z=solver.solve_reduced(force);u[:,columns]+=solver.P@z
        maximum_residual=max(maximum_residual,float(np.linalg.norm(solver.Kr@z-force)/max(np.linalg.norm(force),1e-300)))
    fluctuation=solver.P.T@u;fluctuation[:,0]-=solver.P.T@solver.affine[:,0]
    H=direct-B.T@fluctuation/V
    del fluctuation,B,force,z,load,stress,f
    if maximum_residual>1e-8:raise ValueError('Streamed intact operator failed direct FEM equilibrium')
    # Drop the sparse factor before localization maps occupy memory.
    solver.factor=None
    # Keep only the strain gradient after the solves. Factors, assembly and
    # constrained matrices do not participate in voxel field evaluation.
    dofs=solver.P.shape[1]
    del solver.Kr,solver.P,solver.affine
    strain=np.zeros((N,6,q))
    for begin in range(0,len(ids),max(1,field_block//6)):
        end=min(len(ids),begin+max(1,field_block//6));rows=slice(36*begin,36*end)
        strain[ids[begin:end]]=np.asarray(solver.D[rows]@u).reshape(end-begin,6,6,q).mean(axis=1)
    stress=np.zeros_like(strain)
    for begin in range(0,len(ids),256):
        end=min(len(ids),begin+256);local=strain[ids[begin:end]].copy();local[:,:,-1]-=thermal[begin:end]
        chosen=is_active[begin:end]
        local[chosen,:,1:-1]-=chemical[begin:end][chosen,:,None]*P[cell[begin:end][chosen],None,:]
        stress[ids[begin:end]]=np.einsum('eij,ejk->eik',C[begin:end],local)
    op=VoxelOperator(full,labels,P,weights,areas,sparse.csr_matrix(P.T@fv.stiffness@P),(H+H.T)/2,stress,strain,u.reshape(-1,3,q))
    op.full_reference_residual=maximum_residual
    empty=lambda n,c:np.empty((n,c,0))
    response=CohesiveResponse(op.H,np.zeros((0,q)),np.empty(0),op.displacement_map,empty(len(mesh.points),3),
        strain,empty(N,6),stress,empty(N,6),mesh,.001,
        {'model':'fixed intact Tet4 FEM, factor reuse, streamed full-voxel output','interfaces':0,'spring_modes':0,
            'field_location':'voxel','constrained_DOFs':dofs,'direct_equilibrium_residual':maximum_residual,
            'offline_seconds':time.perf_counter()-started,'scope':'no evolving mechanical contact degradation in this fixed intact backend'})
    response.interface_pairs=np.empty((0,2),int);response.interface_areas=np.empty(0)
    return op,response
