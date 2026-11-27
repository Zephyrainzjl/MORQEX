"""Exact boundary-face extraction and lossless offline geometry packing."""
import base64,gzip
import numpy as np

def exact_surfaces(mesh,labels,particle,grains,cut):
    """Expose actual Tet4 faces; phase interfaces retained for phase toggles.

    Derive each voxel-side pattern from this mesh's six real tetrahedra. No
    alternative display tessellation is substituted for the FEM connectivity.
    """
    selected=np.flatnonzero(labels.ravel()>0);voxel=np.array(np.unravel_index(selected,labels.shape)).T
    local=mesh.cells.reshape(-1,6,4)
    face_pattern=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    master=local[0][:,face_pattern].reshape(24,3)
    xyz=mesh.points[master];origin=xyz.min(axis=(0,1));extent=xyz.max(axis=(0,1))-origin
    coordinates=(xyz-origin)/extent
    output=[];values=[];layers=[]
    for axis in range(3):
        for side in (0,1):
            normal=np.zeros(3);normal[axis]=1 if side else -1
            mask=np.all(np.isclose(coordinates[:,:,axis],side),axis=1)
            pattern=np.flatnonzero(mask)
            if len(pattern)!=2:raise ArithmeticError('Expected two actual Tet4 faces per voxel side')
            zaxis=2-axis;other=voxel.copy();other[:,zaxis]+=1 if side else -1
            inside=np.all((other>=0)&(other<np.asarray(labels.shape)),axis=1)
            neighbor=np.zeros(len(voxel),labels.dtype);neighbor[inside]=labels[tuple(other[inside].T)]
            # Clip at an original voxel face, so every displayed triangle is
            # an actual mesh face, including the interior section.
            neighbor[other[:,0]>=cut]=0
            exposed=(voxel[:,0]<cut)&(neighbor!=labels.ravel()[selected])
            owners=np.flatnonzero(exposed)
            for p in pattern:
                tri=local[owners,p//4][:,face_pattern[p%4]]
                points=mesh.points[tri]/extent
                # Correct winding using the outward normal.
                orientation=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])@normal
                flip=orientation<0;tri[flip]=tri[flip][:,[0,2,1]]
                output.append(mesh.points[tri]/extent)
                ids=selected[owners]
                values.append(np.column_stack([labels.ravel()[ids],particle.ravel()[ids],grains.ravel()[ids]]))
                layers.append(tri)
    return np.concatenate(output).astype('<f4'),np.concatenate(values).astype('<i4'),np.concatenate(layers).astype('<u4')


def pack(array):
    a=np.ascontiguousarray(array)
    return {'shape':a.shape,'dtype':a.dtype.str,'gzip':base64.b64encode(gzip.compress(a.tobytes(),compresslevel=6)).decode()}

