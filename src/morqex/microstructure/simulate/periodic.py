"""Periodic RVE cell problems with C++ NGSolve; full 3x3 homogenized tensor."""
import numpy as np
from .transport import to_ngsolve
from morqex.microstructure.mesh import voxel_to_tetra

def conductivity_tensor(volume,coefficients,order=2):
    import ngsolve as ng
    if min(volume.data.shape)<2:raise ValueError('At least two elements per periodic direction')
    if any(coefficients[p]<=0 for p in np.unique(volume.data)):raise ValueError('Positive phase coefficients required')
    mesh=to_ngsolve(voxel_to_tetra(volume),periodic=True)
    space=ng.Periodic(ng.H1(mesh,order=order))
    u,v=space.TnT()
    k=mesh.MaterialCF({f'phase_{p}':float(coefficients[p]) for p in np.unique(volume.data)})
    form=ng.BilinearForm(space,symmetric=True)
    form+=k*ng.InnerProduct(ng.grad(u),ng.grad(v))*ng.dx
    form.Assemble()
    free=space.FreeDofs()
    for i in range(space.ndof):
        if free[i]:free[i]=False;break
    inverse=form.mat.Inverse(free,inverse='sparsecholesky')
    size=float(ng.Integrate(1,mesh));tensor=np.zeros((3,3));fields=[];residuals=[]
    for axis in range(3):
        direction=ng.CoefficientFunction(tuple(1 if i==axis else 0 for i in range(3)))
        rhs=ng.LinearForm(space);rhs+=-k*ng.InnerProduct(direction,ng.grad(v))*ng.dx;rhs.Assemble()
        w=ng.GridFunction(space);w.vec.data=inverse*rhs.vec
        flux=k*(ng.grad(w)+direction)
        for i in range(3):tensor[i,axis]=float(ng.Integrate(flux[i],mesh))/size
        residual=w.vec.CreateVector();residual.data=form.mat*w.vec-rhs.vec
        array=residual.FV().NumPy()
        mask=np.array([bool(free[i]) for i in range(space.ndof)])
        residuals.append(float(np.linalg.norm(array[mask])/max(np.linalg.norm(rhs.vec.FV().NumPy()),1e-30)))
        fields.append(w)
    return {'tensor_xyz':tensor,'correctors':fields,'mesh':mesh,'boundary_condition':'periodic corrector',
            'residual_xyz':residuals,'symmetry_error':float(np.linalg.norm(tensor-tensor.T)/np.linalg.norm(tensor))}
