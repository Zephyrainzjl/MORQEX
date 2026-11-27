"""Separate zero-area voxel contacts without deleting any material.

A solid FEM vertex is shared only between incident voxels connected through
solid faces in its local eight-octant star. Edge/corner proximity does not
create a weld. Actual finite-area face contacts remain perfectly bonded;
contact resistance, sliding and cohesive traction are separate models.
"""
import numpy as np


def _star_components():
    table=np.full((256,8),255,np.uint8)
    for mask in range(256):
        unseen={i for i in range(8) if mask&(1<<i)};group=0
        while unseen:
            root=min(unseen);unseen.remove(root);pending=[root]
            while pending:
                node=pending.pop();table[mask,node]=group
                for axis in (1,2,4):
                    other=node^axis
                    if other in unseen:unseen.remove(other);pending.append(other)
            group+=1
    return table


def separate_voxel_junctions(volume, points, cells, voxel_ids):
    """Return coordinates/connectivity with disconnected nodal stars split.

    Input must be the unmodified conforming voxel Tet4 mesh and exact parent
    voxel IDs. Coordinates, element volumes and phase/particle labels do not
    change. Components are local stars, not whole-body labels, so an artificial
    point shortcut is removed even when both sides connect elsewhere.
    """
    xyz=np.asarray(points,float);tet=np.asarray(cells);parent=np.asarray(voxel_ids)
    if tet.ndim!=2 or tet.shape[1]!=4 or parent.shape!=(len(tet),):
        raise ValueError('Tet4 cells with exact parent voxels required')
    shape=np.array(volume.data.shape);grid_shape=shape+1
    lattice=(xyz[:,::-1]-np.asarray(volume.origin))/np.asarray(volume.spacing)
    node=np.rint(lattice).astype(np.int64)
    if not np.allclose(lattice,node,rtol=0,atol=1e-7):raise ValueError('Only original voxel lattice nodes supported')
    if np.any(node<0) or np.any(node>shape):raise ValueError('Node outside voxel lattice')
    solid=np.zeros(volume.data.size,np.uint8);solid[parent]=1;solid=solid.reshape(tuple(shape))
    mask=np.zeros(tuple(grid_shape),np.uint8)
    for code in range(8):
        delta=np.array([code//4,(code//2)%2,code%2])
        region=tuple(slice(int(d),int(d+n)) for d,n in zip(delta,shape))
        mask[region]|=solid<<code
    vox=np.column_stack(np.unravel_index(parent,tuple(shape)))
    corners=node[tet];delta=corners-vox[:,None,:]
    if np.any((delta<0)|(delta>1)):raise ValueError('Element vertices disagree with parent voxel')
    codes=delta@np.array([4,2,1]);node_ids=np.ravel_multi_index(corners.reshape(-1,3).T,tuple(grid_shape)).reshape(tet.shape)
    groups=_star_components()[mask.ravel()[node_ids],codes]
    if np.any(groups==255):raise ValueError('Tet4 incident voxel missing from nodal star')
    keys=node_ids*8+groups
    unique,inverse=np.unique(keys,return_inverse=True)
    new_node=np.column_stack(np.unravel_index(unique//8,tuple(grid_shape)))
    new_xyz=(new_node*np.asarray(volume.spacing)+np.asarray(volume.origin))[:,::-1]
    return new_xyz,inverse.reshape(tet.shape)
