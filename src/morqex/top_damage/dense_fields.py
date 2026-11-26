"""Exact Tet4 kinematics and volume-averaged full-lattice recovery."""
from pathlib import Path
import h5py
import numpy as np


class DisplacementRecovery:
    def __init__(self, directory, shape):
        self.shape=tuple(shape)
        with h5py.File(Path(directory)/'mesh.h5','r') as f:
            self.points=f['points_m'][:];self.cells=f['tetrahedra'][:];self.voxels=f['voxel_ids'][:]
        x=self.points[self.cells]
        jac=np.swapaxes(x[:,1:]-x[:,:1],1,2)
        self.volumes=np.linalg.det(jac)/6
        self.gradients=np.einsum('ij,ejk->eik',np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]]),np.linalg.inv(jac))
        self.weights=np.bincount(self.voxels,weights=self.volumes,minlength=int(np.prod(shape)))

    def average(self, values):
        a=np.asarray(values).reshape(len(self.cells),-1)
        result=np.stack([np.bincount(self.voxels,weights=self.volumes*a[:,i],minlength=len(self.weights)) for i in range(a.shape[1])],axis=1)
        result=np.divide(result,self.weights[:,None],out=np.zeros_like(result),where=self.weights[:,None]>0)
        return result.reshape(*self.shape,*np.asarray(values).shape[1:])

    def fields(self, displacement):
        u=np.asarray(displacement,float)
        if u.shape!=self.points.shape or not np.isfinite(u).all():raise ValueError('Finite XYZ nodal displacements required')
        grad=np.einsum('eai,eaj->eij',u[self.cells],self.gradients)
        strain=.5*(grad+np.swapaxes(grad,1,2))
        return self.average(strain),self.average(u[self.cells].mean(axis=1))
