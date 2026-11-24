"""Implicit-Euler heterogeneous diffusion using NGSolve assembly/direct solver."""
from dataclasses import dataclass
import numpy as np
from .transport import to_ngsolve

@dataclass
class DiffusionHistory:
    mesh:object
    field:object
    times:list
    mass:list
    vectors:list

def diffuse(mesh,coefficients,initial=1.,dt=.01,steps=10,order=2,dirichlet='',boundary_value=0.,save_every=1):
    import ngsolve as ng
    if dt<=0 or steps<1 or save_every<1:raise ValueError('Invalid integration settings')
    if any(coefficients[p]<=0 for p in np.unique(mesh.phases)):raise ValueError('Positive diffusivity required')
    m=to_ngsolve(mesh)
    k=m.MaterialCF({f'phase_{p}':float(coefficients[p]) for p in np.unique(mesh.phases)})
    fes=ng.H1(m,order=order,dirichlet=dirichlet)
    u,v=fes.TnT()
    mass=ng.BilinearForm(fes,symmetric=True);mass+=u*v*ng.dx;mass.Assemble()
    system=ng.BilinearForm(fes,symmetric=True)
    system+=(u*v+dt*k*ng.InnerProduct(ng.grad(u),ng.grad(v)))*ng.dx;system.Assemble()
    inv=system.mat.Inverse(fes.FreeDofs(),inverse='sparsecholesky')
    g=ng.GridFunction(fes);g.Set(initial)
    history=DiffusionHistory(m,g,[0.],[float(ng.Integrate(g,m))],[g.vec.FV().NumPy().copy()])
    rhs=g.vec.CreateVector()
    for step in range(1,steps+1):
        rhs.data=mass.mat*g.vec
        g.vec[:]=0
        if dirichlet:g.Set(boundary_value,definedon=m.Boundaries(dirichlet))
        rhs.data-=system.mat*g.vec
        g.vec.data+=inv*rhs
        if step%save_every==0 or step==steps:
            history.times.append(step*dt);history.mass.append(float(ng.Integrate(g,m)))
            history.vectors.append(g.vec.FV().NumPy().copy())
    return history
