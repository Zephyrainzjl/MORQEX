"""P2 displacement/P1 damage reference with positive MFEM quadrature.

Same material geometry and finite chemical energy as the Tet4 reference.
Low-order refined (LOR) elasticity is only an AMG preconditioner; the solved
residual and tangent use the quadratic basis. Not the default cycling model.
"""
import hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator
from morqex.mechanics import TetMesh
from .hyperelastic import Hyperelasticity,FiniteCoupledStep
from .phase_field import AT2Damage


class QuadraticGeometry:
    def __init__(self,base,refiner,order=5):
        self.base=base;self.bary,self.weights=refiner.tetrahedron_rule(order);self.nq=len(self.weights)
        local_edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
        edges,inverse=np.unique(np.sort(base.cells[:,local_edges].reshape(-1,2),axis=1),axis=0,return_inverse=True)
        self.points=np.concatenate([base.points,base.points[edges].mean(axis=1)])
        self.element_nodes=np.column_stack([base.cells,len(base.points)+inverse.reshape(-1,6)])
        # Conforming 8-tet subdivision, used only for preconditioning.
        sub=np.array([[0,4,5,6],[4,1,7,8],[5,7,2,9],[6,8,9,3],
                      [4,5,6,9],[4,5,7,9],[4,7,8,9],[4,6,8,9]])
        cells=self.element_nodes[:,sub].reshape(-1,4).copy();xyz=self.points[cells]
        det=np.linalg.det(np.swapaxes(xyz[:,1:]-xyz[:,:1],1,2));flip=det<0
        cells[flip,0],cells[flip,1]=cells[flip,1].copy(),cells[flip,0].copy()
        self.lor=TetMesh(self.points,cells,np.repeat(base.phases,8),base.box,
            None if base.voxel_ids is None else np.repeat(base.voxel_ids,8))
        self.cells=self.element_nodes
        self.quadrature_count=len(base.cells)*self.nq
        self.shape_gradient=np.zeros((self.nq,10,4))
        for a in range(4):self.shape_gradient[:,a,a]=4*self.bary[:,a]-1
        for k,(a,b) in enumerate(local_edges):
            self.shape_gradient[:,4+k,a]=4*self.bary[:,b]
            self.shape_gradient[:,4+k,b]=4*self.bary[:,a]
        self.volumes=(base.volumes[:,None]*self.weights).ravel()
        self.dofs=np.ascontiguousarray((3*self.cells[:,:,None]+np.arange(3)).reshape(-1,30),dtype=np.int64)
        self.box=base.box;self.rve_volume=base.rve_volume
        self.components=self.lor.components;self.n_components=self.lor.n_components
        self.fingerprint=hashlib.sha256((base.fingerprint+f'/P2/{order}').encode()).hexdigest()
        for x in (self.points,self.cells,self.shape_gradient,self.volumes,self.dofs):x.flags.writeable=False

    def gradients_for_cells(self,start,stop):
        return np.einsum('qab,ebj->eqaj',self.shape_gradient,self.base.gradients[start:stop],optimize=True)

    def outer_nodes(self):return self.lor.outer_nodes()

    def cell_to_quadrature(self,value):
        a=np.asarray(value)
        if len(a)!=len(self.base.cells):raise ValueError('One value per base tetrahedron required')
        return np.repeat(a,self.nq,axis=0)

    def quadrature_average(self,value):
        a=np.asarray(value)
        if len(a)!=self.quadrature_count:raise ValueError('One value per quadrature point required')
        return np.einsum('q,eq...->e...',self.weights,a.reshape(len(self.base.cells),self.nq,*a.shape[1:]))


class QuadraticHyperelasticity(Hyperelasticity):
    def __init__(self,geometry,backend,refiner,young_Pa,poisson,*,boundary='kubc'):
        n=len(geometry.base.cells);E=np.broadcast_to(young_Pa,(n,));nu=np.broadcast_to(poisson,(n,))
        # The existing LOR operator is a preconditioner, never the solved model.
        super().__init__(geometry.lor,backend,np.repeat(E,8),np.repeat(nu,8),
            boundary=boundary,linear_solver='matrix-free-amg-minres')
        self.mesh=geometry;self.geometry=geometry;self.refiner=refiner
        self.young=geometry.cell_to_quadrature(E)
        self.shear=geometry.cell_to_quadrature(E/(2*(1+nu)))
        self.bulk=geometry.cell_to_quadrature(E/(3*(1-2*nu)))
        self._geometry_fingerprint=geometry.fingerprint

    def assemble(self,u,chemical_F,degradation=1.,*,tangent=True):
        mesh=self.mesh;u=np.asarray(u,float).reshape(-1)
        if u.shape!=(3*len(mesh.points),) or not np.isfinite(u).all():raise ValueError('Compatible finite quadratic displacement required')
        count=mesh.quadrature_count
        fc=np.broadcast_to(chemical_F,(count,3,3));g=np.broadcast_to(degradation,(count,))
        F=np.empty((count,3,3));p=np.empty_like(F);plus=np.empty(count);w=np.empty(count)
        c=np.empty((count,9,9)) if tangent else None
        force=np.zeros_like(u)
        for start in range(0,len(mesh.base.cells),256):
            stop=min(start+256,len(mesh.base.cells));s=slice(start*mesh.nq,stop*mesh.nq)
            grad=mesh.gradients_for_cells(start,stop)
            F[s]=(np.eye(3)+np.einsum('eai,eqaj->eqij',u[mesh.dofs[start:stop]].reshape(-1,10,3),grad)).reshape(-1,3,3)
            response=self.backend.evaluate(F[s],fc[s],self.shear[s],self.bulk[s],g[s],tangent=tangent)
            w[s],plus[s],p[s]=response[:3]
            if tangent:c[s]=response[3]
            del response
            local=np.einsum('eqij,eqaj,eq->eai',p[s].reshape(-1,mesh.nq,3,3),grad,mesh.volumes[s].reshape(-1,mesh.nq))
            np.add.at(force,mesh.dofs[start:stop].ravel(),local.ravel())
        if np.any(plus < -64*np.finfo(float).eps*(self.shear+self.bulk)):raise FloatingPointError('Negative tensile energy beyond arithmetic scale')
        plus=np.maximum(plus,0.)
        A=None
        if tangent:
            action=self.refiner.quadratic_tangent_action(mesh,c)
            A=LinearOperator((len(u),len(u)),matvec=action,rmatvec=action,dtype=float)
        return float(mesh.volumes@w),force,A,F,p,plus


class QuadratureAT2(AT2Damage):
    """P1 damage with nonconstant quadratic-displacement driving energy."""
    def __init__(self,geometry,backend,Gc,length,residual_stiffness=1e-5):
        super().__init__(geometry.base,backend,Gc,length,residual_stiffness,linear_solver='jacobi-cg')
        self.geometry=geometry;self.qGc=geometry.cell_to_quadrature(self.Gc);self.qell=geometry.cell_to_quadrature(self.ell)

    def mean_degradation(self,damage):
        d=self._damage(damage)[self.mesh.cells]@self.geometry.bary.T
        return (self.k+(1-self.k)*(1-d)**2).ravel()

    def _qfields(self,psi,factor):
        n=self.geometry.quadrature_count
        psi=np.asarray(np.broadcast_to(psi,(n,)),float);factor=np.asarray(np.broadcast_to(factor,(n,)),float)
        if not np.isfinite(psi).all() or np.any(psi<0) or not np.isfinite(factor).all() or np.any((factor<=0)|(factor>1)):
            raise ValueError('Finite positive quadrature fields required')
        return psi,self.qGc*factor

    def assemble(self,tensile_energy_J_m3,toughness_factor=1.):
        psi,gc=self._qfields(tensile_energy_J_m3,toughness_factor);g=self.geometry;mesh=self.mesh
        source=2*(1-self.k)*psi;reaction=source+gc/self.qell
        weighted=(g.volumes*reaction).reshape(-1,g.nq)
        local=np.einsum('eq,qa,qb->eab',weighted,g.bary,g.bary,optimize=True)
        rows=np.broadcast_to(mesh.cells[:,:,None],local.shape);cols=np.broadcast_to(mesh.cells[:,None,:],local.shape)
        mass=coo_matrix((local.ravel(),(rows.ravel(),cols.ravel())),shape=(len(mesh.points),)*2).tocsr()
        coef=g.quadrature_average(gc*self.qell)
        stiffness,_=self.backend.assemble(mesh,0.,coef[:,None,None]*np.eye(3),0.)
        nodal=(g.volumes*source).reshape(-1,g.nq)@g.bary
        rhs=np.bincount(mesh.cells.ravel(),weights=nodal.ravel(),minlength=len(mesh.points))
        return mass+stiffness,rhs

    def energies(self,damage,tensile_energy_J_m3,toughness_factor=1.):
        psi,gc=self._qfields(tensile_energy_J_m3,toughness_factor);g=self.geometry
        d=self._damage(damage);dq=(d[self.mesh.cells]@g.bary.T).ravel()
        grad=np.einsum('ei,eij->ej',d[self.mesh.cells],self.mesh.gradients)
        grad2=g.cell_to_quadrature(np.sum(grad**2,axis=1))
        fracture=float(g.volumes@(gc/2*(dq*dq/self.qell+self.qell*grad2)))
        elastic=float(g.volumes@(self.mean_degradation(d)*psi))
        return elastic,fracture


def prolong_quadratic_field(source,target,parent_cells,field):
    """Exact nested P2 evaluation using MFEM's parent cells, without searches.

    Coincident nodes on separate material sheets retain their own parent.
    This transfers a displacement guess, not an irreversible damage bound.
    """
    parent=np.asarray(parent_cells,dtype=np.int64);values=np.asarray(field,float)
    if parent.shape!=(len(target.base.cells),) or np.any((parent<0)|(parent>=len(source.base.cells))):
        raise ValueError('One valid source parent for each target cell required')
    if values.shape[0]!=len(source.points) or not np.isfinite(values).all():
        raise ValueError('Finite source quadratic nodal field required')
    nodes,first=np.unique(target.cells.ravel(),return_index=True)
    if not np.array_equal(nodes,np.arange(len(target.points))):raise ValueError('Unreferenced target nodes')
    owner=parent[first//10];x=target.points-source.base.points[source.base.cells[owner,0]]
    bary=np.einsum('ni,nai->na',x,source.base.gradients[owner]);bary[:,0]+=1
    if np.min(bary)<-1e-10 or np.max(bary)>1+1e-10:raise ValueError('Target node outside supplied parent')
    edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    shape=np.column_stack([bary*(2*bary-1),4*bary[:,edges[:,0]]*bary[:,edges[:,1]]])
    return np.einsum('na,na...->n...',shape,values[source.cells[owner]])


def quadratic_staggered_step(mechanics,damage_model,accepted_damage,history,chemical_F,*,fatigue=False,tolerance=1e-7,max_iterations=100,initial_displacement=None):
    """Fixed-geometry variational alternation; history resides at quadrature points."""
    if mechanics.geometry is not damage_model.geometry:raise ValueError('A shared quadratic geometry is required')
    if len(history.accumulated_J_m3)!=mechanics.geometry.quadrature_count:raise ValueError('Quadrature history required')
    threshold=damage_model.qGc/(12*damage_model.qell)
    factor=history.toughness_factor(threshold) if fatigue else 1.
    lower=damage_model._damage(accepted_damage).copy();d=lower.copy();u=initial_displacement;trace=[]
    for iteration in range(1,max_iterations+1):
        state=mechanics.solve(chemical_F,damage_model.mean_degradation(d),initial_displacement=u,tolerance=min(tolerance*.01,1e-9))
        psi=state.tensile_energy_J_m3
        energy=state.potential_energy_J+damage_model.energies(d,psi,factor)[1]
        if trace and energy>trace[-1]+1e-8*max(abs(trace[-1]),np.finfo(float).tiny):raise RuntimeError('P2/P1 alternating energy increased')
        trace.append(energy);solved=damage_model.solve(psi,lower,factor,tolerance=min(tolerance*.01,1e-10))
        if np.max(np.abs(solved.damage-d))<tolerance:
            A,b=damage_model.assemble(psi,factor)
            kkt=float(np.max(np.abs(d-np.clip(d-(A@d-b)/A.diagonal(),lower,1))))
            if kkt<tolerance:
                new=history.trial(damage_model.mean_degradation(d)*psi) if fatigue else history
                changed=new.toughness_factor(threshold) if fatigue else factor
                delta=damage_model.energies(d,psi,changed)[1]-damage_model.energies(d,psi,factor)[1]
                return FiniteCoupledStep(d,state,new,iteration,kkt,np.asarray(trace),delta)
        d=solved.damage.copy();u=state.displacement_m
    raise RuntimeError('Quadratic stagger did not converge; accepted state unchanged')
