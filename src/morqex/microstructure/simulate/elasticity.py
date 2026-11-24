"""Upstream NGSolve C++ linear elasticity and kinematic RVE homogenization."""
import numpy as np
from .transport import to_ngsolve

def homogenize_elasticity(mesh,properties,order=1):
    """Returns engineering-Voigt stiffness [xx,yy,zz,yz,xz,xy].

    Uniform strain displacement on all outer surfaces (KUBC), not periodic BC.
    properties maps phase -> (Young modulus in Pa, Poisson ratio).
    """
    import ngsolve as ng
    labels=np.unique(mesh.phases)
    for p in labels:
        e,nu=properties[p]
        if e<=0 or not -1<nu<.5:raise ValueError('Require E>0 and -1<nu<.5')
    m=to_ngsolve(mesh)
    lam=m.MaterialCF({f'phase_{p}':properties[p][0]*properties[p][1]/((1+properties[p][1])*(1-2*properties[p][1])) for p in labels})
    mu=m.MaterialCF({f'phase_{p}':properties[p][0]/(2*(1+properties[p][1])) for p in labels})
    fes=ng.VectorH1(m,order=order,dirichlet='.*')
    u,v=fes.TnT()
    def strain(u):return ng.Sym(ng.Grad(u))
    def stress(u):return 2*mu*strain(u)+lam*ng.Trace(strain(u))*ng.Id(3)
    a=ng.BilinearForm(fes,symmetric=True)
    a+=ng.InnerProduct(stress(u),strain(v))*ng.dx
    a.Assemble();inv=a.mat.Inverse(fes.FreeDofs(),inverse='sparsecholesky')
    coordinates=ng.CoefficientFunction((ng.x,ng.y,ng.z))
    voigt=[(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]
    stiffness=np.zeros((6,6));fields=[]
    volume=float(ng.Integrate(1,m))
    for j,(b,c) in enumerate(voigt):
        eps=np.zeros((3,3));eps[b,c]=eps[c,b]=1. if b == c else .5
        affine=ng.CoefficientFunction(tuple(eps.ravel()),dims=(3,3))*coordinates
        g=ng.GridFunction(fes);g.Set(affine,definedon=m.Boundaries('.*'))
        residual=g.vec.CreateVector();residual.data=-a.mat*g.vec
        g.vec.data+=inv*residual
        sig=stress(g)
        for i,(b,c) in enumerate(voigt):stiffness[i,j]=ng.Integrate(sig[b,c],m)/volume
        fields.append(g)
    return {'stiffness':stiffness,'fields':fields,'mesh':m,'boundary_condition':'KUBC',
            'voigt_order':['xx','yy','zz','yz','xz','xy'],
            'symmetry_error':float(np.linalg.norm(stiffness-stiffness.T)/max(np.linalg.norm(stiffness),1e-30))}
