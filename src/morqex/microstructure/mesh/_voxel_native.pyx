# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: language=c++
import numpy as np
cimport numpy as np

def full_hexa(np.ndarray[np.int64_t,ndim=3] data, spacing, origin):
    cdef long nz=data.shape[0],ny=data.shape[1],nx=data.shape[2]
    cdef long n=(nz+1)*(ny+1)*(nx+1)
    cdef np.ndarray[np.float64_t,ndim=2] points=np.empty((n,3),dtype=np.float64)
    cdef np.ndarray[np.int64_t,ndim=2] cells=np.empty((nz*ny*nx,8),dtype=np.int64)
    cdef long z,y,x,i,j,base,sx=nx+1,sz=(ny+1)*(nx+1)
    cdef double dz=spacing[0],dy=spacing[1],dx=spacing[2]
    cdef double oz=origin[0],oy=origin[1],ox=origin[2]
    with nogil:
        for z in range(nz+1):
            for y in range(ny+1):
                for x in range(nx+1):
                    i=z*sz+y*sx+x
                    points[i,0]=ox+x*dx
                    points[i,1]=oy+y*dy
                    points[i,2]=oz+z*dz
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    i=(z*ny+y)*nx+x
                    base=z*sz+y*sx+x
                    cells[i,0]=base;cells[i,1]=base+1
                    cells[i,2]=base+sx+1;cells[i,3]=base+sx
                    cells[i,4]=base+sz;cells[i,5]=base+sz+1
                    cells[i,6]=base+sz+sx+1;cells[i,7]=base+sz+sx
    return points,cells,data.ravel().copy()
