"""Exact scalar-damage cohesive-interface resolvent, with fixed bulk elasticity.

Active/CBD faces are split and tied by area-weighted normal/tangential springs.
Damage changes the springs, not a postprocessed stress multiplier. One damage
state per SVE; residual springs prevent newly created rigid mechanisms.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse,linalg
from .mechanics import TetMesh,ElasticitySolver
from .mechanics.constraints import constraints
from .transport import VoxelDiffusion

@dataclass
class CohesiveResponse:
    direct:np.ndarray
    factors:np.ndarray
    eigenvalues:np.ndarray
    displacement_direct:np.ndarray
    displacement_modes:np.ndarray
    strain_direct:np.ndarray
    strain_modes:np.ndarray
    stress_direct:np.ndarray
    stress_modes:np.ndarray
    mesh:object
    residual:float
    metadata:dict
    interface_pairs:np.ndarray=None
    interface_areas:np.ndarray=None

    def retention(self,d):return self.residual+(1-self.residual)*(1-np.asarray(d))**2
    def hessian(self,d):
        if not np.isfinite(d) or not 0<=d<=1:raise ValueError('Damage outside [0,1]')
        den=1+(self.retention(d)-1)*self.eigenvalues
        return self.direct-(self.factors.T/den)@self.factors
    def derivative(self,d):
        den=1+(self.retention(d)-1)*self.eigenvalues
        first=-2*(1-self.residual)*(1-d)
        return (self.factors.T*(first*self.eigenvalues/den**2))@self.factors
    def recover(self,q,d):
        q=np.asarray(q);den=1+(self.retention(d)[...,None]-1)*self.eigenvalues;z=(q@self.factors.T)/den
        fields={name:np.einsum('...j,eij->...ei',q,getattr(self,name+'_direct'))+
            np.einsum('...k,eik->...ei',z,getattr(self,name+'_modes')) for name in ['displacement','strain','stress']}
        if self.interface_pairs is not None:
            fields['opening']=fields['displacement'][...,self.interface_pairs[:,1],:]-fields['displacement'][...,self.interface_pairs[:,0],:]
        return fields

def build_cohesive(electrode,P,*,normal_stiffness=1e15,tangent_stiffness=5e14,residual=1e-3):
    if normal_stiffness<=0 or tangent_stiffness<=0 or not 1e-6<=residual<1:raise ValueError('Positive interface stiffness and residual in [1e-6,1) required')
    original=electrode.mesh;ne=len(original.cells);active=electrode._selected
    # Duplicate only the active/non-active boundary. Face-disconnected solid
    # components were already separated by the electrode geometry builder.
    kind=active.astype(int);parts=np.empty(ne,dtype=int);count=0
    for selected in [False,True]:
        phases=np.unique(original.phases[active==selected]);graph=VoxelDiffusion(electrode.volume,active_phases=phases,diffusivity=1.)
        lookup_voxel=np.full(electrode.volume.data.size,-1,dtype=int);lookup_voxel[graph.voxel_ids]=graph.components+count
        parts[active==selected]=lookup_voxel[original.voxel_ids[active==selected]];count+=graph.n_components
    pairs,inverse=np.unique(np.column_stack([np.repeat(parts,4),original.cells.ravel()]),axis=0,return_inverse=True)
    mesh=TetMesh(original.points[pairs[:,1]],inverse.reshape(-1,4),original.phases,original.box,original.voxel_ids)
    lookup={(int(k),int(n)):i for i,(k,n) in enumerate(pairs)}
    faces={};interfaces=[]
    for e,tet in enumerate(original.cells):
        for omitted in range(4):
            nodes=tuple(sorted(np.delete(tet,omitted)));previous=faces.pop(nodes,None)
            if previous is None:faces[nodes]=e
            elif active[previous]!=active[e]:interfaces.append((nodes,previous,e))
    rows=[];cols=[];values=[];interface_pairs=[];interface_areas=[]
    for face,ea,eb in interfaces:
        x=original.points[list(face)];cross=np.cross(x[1]-x[0],x[2]-x[0]);area=np.linalg.norm(cross)/2;normal=cross/(2*area)
        k=(tangent_stiffness*np.eye(3)+(normal_stiffness-tangent_stiffness)*np.outer(normal,normal))*area/3
        for node in face:
            a,b=lookup[(int(parts[ea]),node)],lookup[(int(parts[eb]),node)]
            interface_pairs.append((a,b));interface_areas.append(area/3)
            for i in range(3):
                for j in range(3):
                    for u,v,sign in [(a,a,1),(b,b,1),(a,b,-1),(b,a,-1)]:rows.append(3*u+i);cols.append(3*v+j);values.append(sign*k[i,j])
    if not interfaces:raise ValueError('Cohesive debonding requires explicitly resolved active/CBD or active/inert interfaces')
    C=sparse.coo_matrix((values,(rows,cols)),shape=(3*len(mesh.points),)*2).tocsr()
    bulk=ElasticitySolver(mesh,electrode.solver.stiffness,boundary='kubc')
    # Rigid modes are those of the bonded graph, rather than separately pinned
    # particles. Springs must be allowed to determine relative particle motion.
    class BondedTopology:
        points=mesh.points;box=mesh.box
        components=original.components[pairs[:,1]];n_components=original.n_components
        def outer_nodes(self):return mesh.outer_nodes()
    projection=constraints(BondedTopology(),'kubc').projection
    nd=projection.shape[1]
    if nd>1800:raise MemoryError('Dense cohesive spectral qualification limited to 1800 constrained DOFs')
    K0=(projection.T@(bulk.K+C)@projection).toarray();Cr=(projection.T@C@projection).toarray()
    scale=float(np.max(np.diag(K0)));L=linalg.cholesky(K0/scale,lower=True)
    normalized=linalg.solve_triangular(L,Cr/scale,lower=True)
    normalized=linalg.solve_triangular(L,normalized.T,lower=True).T
    rho,V=linalg.eigh((normalized+normalized.T)/2)
    if rho.min() < -1e-9 or rho.max()>1+1e-9:raise ValueError('Invalid cohesive spectrum')
    rho=np.clip(rho,0,1)
    m=P.shape[1];eigen=np.zeros((ne,6,m+2));cell=electrode._element_cells
    eigen[active,:,1:-1]=electrode._chemical_tensors[active,:,None]*P[cell[active],None,:];eigen[:,:,-1]=electrode._thermal_tensors
    affine=np.zeros((3*len(mesh.points),m+2));affine[:,0]=bulk.affine[:,0]
    f=np.asarray(bulk.D@affine).reshape(ne,6,-1)-eigen
    stress0=np.einsum('eij,ejk->eik',bulk.stiffness,f)
    direct=np.einsum('eik,eil,e->kl',f,stress0,mesh.volumes)/mesh.rve_volume
    B=-np.asarray(projection.T@(bulk.D.T@(mesh.volumes[:,None,None]*stress0).reshape(6*ne,-1)))
    # K/scale=L L.T, u=projection L^-T V [(V.T L^-1 B)/scale]/den.
    Q=linalg.solve_triangular(L.T,V,lower=False)
    F=Q.T@B/np.sqrt(scale*mesh.rve_volume)
    u_modes=np.asarray(projection@Q)*np.sqrt(mesh.rve_volume/scale)
    # Exact zero-spring modes have denominator one: combine them into direct
    # terms, reducing online work without approximating positive eigenvalues.
    keep=rho>1e-12;zero=~keep
    direct-=F[zero].T@F[zero];affine+=u_modes[:,zero]@F[zero]
    F=F[keep];rho=rho[keep];u_modes=u_modes[:,keep]
    strain0=np.asarray(bulk.D@affine).reshape(ne,6,-1);strain_modes=np.asarray(bulk.D@u_modes).reshape(ne,6,-1)
    stress0=np.einsum('eij,ejk->eik',bulk.stiffness,strain0-eigen);stress_modes=np.einsum('eij,ejk->eik',bulk.stiffness,strain_modes)
    response=CohesiveResponse((direct+direct.T)/2,F,rho,affine.reshape(-1,3,m+2),u_modes.reshape(-1,3,len(rho)),
        strain0,strain_modes,stress0,stress_modes,mesh,residual,
        {'interfaces':len(interfaces),'spring_modes':len(rho),'constrained_DOFs':nd,'normal_Pa_m':normal_stiffness,'tangent_Pa_m':tangent_stiffness,
         'model':'zero-thickness lumped cohesive springs; uniform scalar irreversible SVE damage; residual stiffness',
         'scope':'no unilateral closure, friction, explicit crack path or finite deformation'})
    response.interface_pairs=np.asarray(interface_pairs);response.interface_areas=np.asarray(interface_areas)
    # Independent direct solves at two retained-damage values, not interpolation.
    errors=[]
    for damage in [0.,.63]:
        K=projection.T@(bulk.K+response.retention(damage)*C)@projection
        u=affine.copy()  # Use the original affine lift, before zero-mode reduction.
        u=np.zeros_like(u);u[:,0]=bulk.affine[:,0]
        v=sparse.linalg.spsolve(K.tocsc(),B);u+=projection@v
        strain=np.asarray(bulk.D@u).reshape(ne,6,-1)-eigen
        H=np.einsum('eik,eij,ejl,e->kl',strain,bulk.stiffness,strain,mesh.volumes)/mesh.rve_volume
        H+=u.T@(response.retention(damage)*C@u)/mesh.rve_volume
        norm=np.maximum(np.abs(H.diagonal()),1e-30);errors.append(float(np.linalg.norm((response.hessian(damage)-H)/np.sqrt(norm[:,None]*norm[None,:]))))
    if max(errors)>1e-7:raise ValueError('Cohesive resolvent failed independent direct FEM energy comparison')
    response.metadata['direct_solve_scaled_energy_error']=max(errors)
    return response
