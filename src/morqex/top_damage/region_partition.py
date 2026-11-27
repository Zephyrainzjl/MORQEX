"""Deterministic connected, nested voxel aggregation for refinement studies.

Geometric bisection is a baseline, not a proposed optimal graph reduction.
Every original diffusion component retains independent inventory. At target=N
the identity partition recovers the unreduced voxel finite-volume equations.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components


def connected_partition(diffusion,target):
    coordinates=diffusion.coordinates_zyx;n=len(coordinates)
    if type(target) is not int or not 1<=target<=n:raise ValueError('Region target must lie between one and active voxel count')
    groups=[np.flatnonzero(diffusion.components==k) for k in np.unique(diffusion.components)]
    if target<len(groups):raise ValueError('Cannot merge disconnected diffusion inventories')
    while len(groups)<target:
        which=max(range(len(groups)),key=lambda k:(len(groups[k]),-int(groups[k][0])))
        ids=groups.pop(which);xyz=coordinates[ids];axis=int(np.argmax(np.ptp(xyz,axis=0)))
        order=np.lexsort((ids,xyz[:,axis]));middle=len(ids)//2
        children=[]
        for subset in (ids[order[:middle]],ids[order[middle:]]):
            count,labels=connected_components(diffusion.stiffness[subset][:,subset],directed=False)
            children.extend(np.sort(subset[labels==k]) for k in range(count))
        groups.extend(children)
    labels=np.empty(n,np.int64)
    for k,ids in enumerate(sorted(groups,key=lambda ids:int(ids[0]))):labels[ids]=k
    return labels
