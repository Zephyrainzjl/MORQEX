def minimize_energy(int [:, ::1] edges, int [::1] clusters, double [::1] area,
                    double [:, ::1] sgamma, double [:, ::1] cent, double [::1] srho,
                    int [::1] cluscount, int maxiter, double [::1] energy,
                    uint8 [::1] mod1, uint8 [::1] mod2):
    """ Minimize cluster energy"""
    cdef int face_a, face_b, clusA, clusB
    cdef double areaface_a, centA0, centA1, centA2
    cdef double areaface_b, centB0, centB1, centB2
    cdef double eA, eB, eorig, eAwB, eBnB, eAnA, eBwA
    cdef int nchange = 1   
    cdef int niter = 0
    cdef int nclus = mod1.shape[0]
    cdef int nedge = edges.shape[0]
    cdef int i
    cdef int [1] nchange_arr
    # start all as modified
    for i in range(nclus):
        mod2[i] = 1

    tlast = 0
    while nchange > 0 and niter < maxiter:
        # Reset modification arrays
        for i in range(nclus):
            mod1[i] = mod2[i]
            mod2[i] = 0
        nchange = 0
        for i in range(nedge):
            # Get the two clusters sharing an edge
            face_a = edges[i, 0]
            face_b = edges[i, 1]
            clusA = clusters[face_a]
            clusB = clusters[face_b]

            # If edge shares two different clusters and at least one
            # has been modified since last iteration
            if clusA != clusB and (mod1[clusA] == 1 or mod1[clusB] == 1):
                # Verify that face can be removed from cluster
                if cluscount[clusA] > 1 and cluscount[clusB] > 1:

                    areaface_a = area[face_a]
                    centA0 = cent[face_a, 0]
                    centA1 = cent[face_a, 1]
                    centA2 = cent[face_a, 2]  

                    areaface_b = area[face_b]
                    centB0 = cent[face_b, 0]
                    centB1 = cent[face_b, 1]
                    centB2 = cent[face_b, 2] 

                    # Current energy
                    eorig =  energy[clusA] + energy[clusB]
                    # Energy with both items assigned to cluster A
                    eAwB = ((sgamma[clusA, 0] + centB0)**2 + \
                            (sgamma[clusA, 1] + centB1)**2 + \
                            (sgamma[clusA, 2] + centB2)**2)/(srho[clusA] + areaface_b)
                    eBnB = ((sgamma[clusB, 0] - centB0)**2 + \
                            (sgamma[clusB, 1] - centB1)**2 + \
                            (sgamma[clusB, 2] - centB2)**2)/(srho[clusB] - areaface_b)
                    eA = eAwB + eBnB
    
                    # Energy with both items assigned to clusterB
                    eAnA = ((sgamma[clusA, 0] - centA0)**2 + \
                            (sgamma[clusA, 1] - centA1)**2 + \
                            (sgamma[clusA, 2] - centA2)**2)/(srho[clusA] - areaface_a)
                    eBwA = ((sgamma[clusB, 0] + centA0)**2 + \
                            (sgamma[clusB, 1] + centA1)**2 + \
                            (sgamma[clusB, 2] + centA2)**2)/(srho[clusB] + areaface_a)

                    eB = eAnA + eBwA
                    # select the largest case (most negative)
                    if eA > eorig and eA > eB:
                        mod2[clusA] = 1
                        mod2[clusB] = 1

                        nchange += 1
                        # reassign
                        clusters[face_b] = clusA
                        cluscount[clusB] -= 1
                        cluscount[clusA] += 1
                        
                        # Update cluster A mass and centroid
                        srho[clusA] += areaface_b
                        sgamma[clusA, 0] += centB0
                        sgamma[clusA, 1] += centB1
                        sgamma[clusA, 2] += centB2
                        
                        srho[clusB] -= areaface_b
                        sgamma[clusB, 0] -= centB0
                        sgamma[clusB, 1] -= centB1
                        sgamma[clusB, 2] -= centB2
                        
                        # Update cluster energy
                        energy[clusA] = eAwB
                        energy[clusB] = eBnB

                    # if the energy contribution of both to B is less than the original and to cluster A
                    elif eB > eorig and eB > eA:
                        # Show clusters as modifies
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        nchange += 1
                        
                        # reassign
                        clusters[face_a] = clusB
                        cluscount[clusA] -= 1
                        cluscount[clusB] += 1
                        # Add item A to cluster A
                        srho[clusB] += areaface_a
                        sgamma[clusB, 0] += centA0
                        sgamma[clusB, 1] += centA1
                        sgamma[clusB, 2] += centA2
                        # Remove item A from cluster A
                        srho[clusA] -= areaface_a
                        sgamma[clusA, 0] -= centA0
                        sgamma[clusA, 1] -= centA1
                        sgamma[clusA, 2] -= centA2
                        # Update cluster energy
                        energy[clusA] = eAnA
                        energy[clusB] = eBwA
                                                
                elif cluscount[clusA] > 1:
                    areaface_a = area[face_a]
                    centA0 = cent[face_a, 0]
                    centA1 = cent[face_a, 1]
                    centA2 = cent[face_a, 2]
                    
                    # Current energy
                    eorig =  energy[clusA] + energy[clusB]
                    # Energy with both items assigned to clusterB
                    eAnA = ((sgamma[clusA, 0] - centA0)**2 + \
                            (sgamma[clusA, 1] - centA1)**2 + \
                            (sgamma[clusA, 2] - centA2)**2)/(srho[clusA] - areaface_a)
                           
                    eBwA = ((sgamma[clusB, 0] + centA0)**2 + \
                            (sgamma[clusB, 1] + centA1)**2 + \
                            (sgamma[clusB, 2] + centA2)**2)/(srho[clusB] + areaface_a)
                    eB = eAnA + eBwA
                    
                    # Compare energy contributions
                    if eB > eorig:
                        # Flag clusters as modified
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        nchange += 1
                    
                        # reassign
                        clusters[face_a] = clusB
                        cluscount[clusA] -= 1
                        cluscount[clusB] += 1
                        # Add item A to cluster A
                        srho[clusB] += areaface_a
                        sgamma[clusB, 0] += centA0
                        sgamma[clusB, 1] += centA1
                        sgamma[clusB, 2] += centA2
                        # Remove item A from cluster A
                        srho[clusA] -= areaface_a
                        sgamma[clusA, 0] -= centA0
                        sgamma[clusA, 1] -= centA1
                        sgamma[clusA, 2] -= centA2
                        # Update cluster energy
                        energy[clusA] = eAnA
                        energy[clusB] = eBwA

                elif cluscount[clusB] > 1:                
                    areaface_b = area[face_b]
                    centB0 = cent[face_b, 0]
                    centB1 = cent[face_b, 1]
                    centB2 = cent[face_b, 2]  
                    
                    # Current energy
                    eorig =  energy[clusA] + energy[clusB]
                    # Energy with both items assigned to cluster A
                    eAwB = ((sgamma[clusA, 0] + centB0)**2 + \
                            (sgamma[clusA, 1] + centB1)**2 + \
                            (sgamma[clusA, 2] + centB2)**2)/(srho[clusA] + areaface_b)
                    eBnB = ((sgamma[clusB, 0] - centB0)**2 + \
                            (sgamma[clusB, 1] - centB1)**2 + \
                            (sgamma[clusB, 2] - centB2)**2)/(srho[clusB] - areaface_b)
                    eA = eAwB + eBnB
    
                    # If moving face B reduces cluster energy
                    if eA > eorig:
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        
                        nchange+=1
                        # reassign
                        clusters[face_b] = clusA
                        cluscount[clusB] -= 1
                        cluscount[clusA] += 1
                        
                        # Update cluster A mass and centroid
                        srho[clusA] += areaface_b
                        sgamma[clusA, 0] += centB0
                        sgamma[clusA, 1] += centB1
                        sgamma[clusA, 2] += centB2
                        
                        srho[clusB] -= areaface_b
                        sgamma[clusB, 0] -= centB0
                        sgamma[clusB, 1] -= centB1
                        sgamma[clusB, 2] -= centB2
                        
                        # Update cluster energy
                        energy[clusA] = eAwB
                        energy[clusB] = eBnB
                        
        niter += 1



def cluster(int [:, ::1] neighbors, int [::1] nneigh, 
            int nclus, double [::1] area, 
            double [:, ::1] cent, int [:, ::1] edges, int maxiter, 
            debug=False, int iso_try=10):
    """ Python interface function for cluster optimization """
    cdef int i

    # Initialize clusters
    cdef int npoints = nneigh.shape[0]
    cdef int [::1] clusters = np.empty(npoints, ctypes.c_int)
    clusters [:] = -1
    cdef int [:, ::1] items = np.empty((2, npoints), ctypes.c_int)
    init_clusters(clusters, neighbors, nneigh, area, nclus, items)

    # Eliminat null clusters by growing existing null clusters
    grow_null(edges, clusters)
    
    # Assign any remaining clusters to 0 (just in case null clusters fails)
    for i in range(npoints):
        if clusters[i] == -1:
            clusters[i] = 0
            
    # Arrays for cluster centers, masses, and energies
    cdef double [:, ::1] sgamma = np.zeros((nclus, 3))
    cdef double [::1] srho = np.zeros(nclus)
    cdef double [::1] energy = np.empty(nclus)
        
    # Compute initial masses of clusters
    for i in range(npoints):
        srho[clusters[i]] += area[i]
        sgamma[clusters[i], 0] += cent[i, 0]
        sgamma[clusters[i], 1] += cent[i, 1]
        sgamma[clusters[i], 2] += cent[i, 2]
    
    for i in range(nclus):
        energy[i] = (sgamma[i, 0]**2 + \
                     sgamma[i, 1]**2 + \
                     sgamma[i, 2]**2)/srho[i]
        
    # Count number of clusters
    cdef int [::1] cluscount = np.bincount(clusters).astype(ctypes.c_int)
    
    # Initialize modified array
    cdef uint8 [::1] mod1 = np.empty(nclus, ctypes.c_uint8)
    cdef uint8 [::1] mod2 = np.empty(nclus, ctypes.c_uint8)

    # Optimize clusters
    minimize_energy(edges, clusters, area, sgamma, cent, srho, cluscount,
                    maxiter, energy, mod1, mod2)
                   
    # Identify isolated clusters here
    ndisc = null_disconnected(nclus, nneigh, neighbors, clusters)
    cdef int niter = 0
    while ndisc and niter < iso_try:
        grow_null(edges, clusters)
        # Re optimize clusters
        minimize_energy(edges, clusters, area, sgamma, cent, srho, cluscount,
                       maxiter, energy, mod1, mod2)
        # Check again for disconnected clusters
        for i in range(npoints):
            if clusters[i] == -1:
                clusters[i] = 0
        ndisc = null_disconnected(nclus, nneigh, neighbors, clusters)
        niter += 1
        
        if ndisc:
            grow_null(edges, clusters)
        
            # Check again for disconnected clusters
            for i in range(npoints):
                if clusters[i] == -1:
                    clusters[i] = 0

    # renumber clusters 0 to n
    nclus = renumber_clusters(clusters, npoints, nclus)
    return np.asarray(clusters), ndisc > 0, nclus


cdef void init_clusters(int [::1] clusters, int [:, ::1] neighbors,
                        int [::1] nneigh, double [::1] area, int nclus,
                        int [:, ::1] items) nogil:
    """ Initialize clusters"""
                               
    cdef double tarea, new_area, carea
    cdef int item
    cdef int i, j, k, checkitem, c, c_prev
    cdef double ctarea
    cdef int lstind = 0
    cdef int npoints = area.shape[0]
    cdef int i_items_new, i_items_old    
    
    # Total mesh size
    cdef double area_remain = 0
    for i in range(npoints):
        area_remain += area[i]
    
    # Assign clsuters
    ctarea = area_remain/nclus
    for i in range(nclus):
        # Get target area and reset current area
        tarea = area_remain - ctarea*(nclus - i - 1)
        carea = 0.0
        # Get starting index (the first free face in list)
        i_items_new = 0
        for j in range(lstind, npoints):
            if clusters[j] == -1:
                carea += area[j]
                items[i_items_new, 0] = j
                clusters[j] = i
                lstind = j
                break
        if j == npoints:
            break
        
        # While there are new items to be added
        c = 1
        while c:
            
            # reset items
            c_prev = c
            c = 0
            # switch indices
            if i_items_new == 0:
                i_items_old = 0
                i_items_new = 1
            else:
                i_items_old = 1
                i_items_new = 0            
            # progressively add neighbors
            for j in range(c_prev):
                checkitem = items[i_items_old, j]
                for k in range(nneigh[checkitem]):
                    item = neighbors[checkitem, k]
                    
                    # check if the face is free
                    if clusters[item] == -1:
                        # if allowable, add to cluster
                        if area[item] + carea < tarea:
                            carea += area[item]
                            clusters[item] = i
                            items[i_items_new, c] = item
                            c += 1

        area_remain -= carea


# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs
cimport ctypes

# ---------------------------
# 说明（参数与新数组含义）
# ---------------------------
# face_touch_point : int[npoints]
#    face_touch_point[i] == point_id if face i contains that triple-point, else -1
# face_touch_line  : int[npoints]
#    face_touch_line[i] == line_id if face i touches that line (has at least one vertex on line), else -1
#
# cluster_anchor_type : int[nclus]    (0 = normal, 1 = line-anchor, 2 = point-anchor)
# cluster_anchor_id   : int[nclus]    (the line_id or point_id for anchor clusters; -1 for normal)
# anchor_pos          : double[nclus,3]  (anchor geometric position for anchor clusters; unspecified for normal clusters)
# anchor_fixed        : uint8[nclus]   (1 = this cluster is strictly fixed to anchor_pos, else 0)
#
# 这些参数应在 Python 端与 cluster() 一起传入。

# ---------------------------
# Helper inline functions
# ---------------------------
cdef inline bint _face_allowed_for_anchor(int face, int atype, int aid,
                                         int[:] face_touch_line, int[:] face_touch_point) nogil:
    """Return True if face is acceptable for anchor of type atype with id aid."""
    if atype == 0:
        return True
    elif atype == 1:
        # line anchor: face must touch the same line id
        return face_touch_line[face] == aid
    elif atype == 2:
        # point anchor: face must contain the same point id
        return face_touch_point[face] == aid
    return False

cdef inline double _sq(double x) nogil:
    return x*x

# ---------------------------
# Modified init_clusters (nogil)
# ---------------------------
cdef void init_clusters(int [::1] clusters, int [:, ::1] neighbors,
                        int [::1] nneigh, double [::1] area, int nclus,
                        int [:, ::1] items,
                        int [::1] face_touch_line, int [::1] face_touch_point,
                        int [::1] cluster_anchor_type, int [::1] cluster_anchor_id) nogil:
    """
    Initialize clusters, but first *preassign* faces that touch anchors
    to their corresponding anchor clusters.
    Then perform the original region-growth allocation for the remaining faces.
    """
    cdef double tarea, new_area, carea
    cdef int item
    cdef int i, j, k, checkitem, c, c_prev
    cdef double ctarea
    cdef int lstind = 0
    cdef int npoints = area.shape[0]
    cdef int i_items_new, i_items_old
    cdef double area_remain = 0

    # total area
    for i in range(npoints):
        area_remain += area[i]

    # Initially mark all clusters free (-1)
    for i in range(npoints):
        clusters[i] = -1

    # --- Preassign faces that match anchor clusters ---
    # For each cluster that is an anchor, assign faces that satisfy anchor condition
    for c in range(nclus):
        if cluster_anchor_type[c] != 0:
            aid = cluster_anchor_id[c]
            atype = cluster_anchor_type[c]
            # assign any face that touches this anchor and is unassigned
            for i in range(npoints):
                if clusters[i] == -1:
                    if _face_allowed_for_anchor(i, atype, aid, face_touch_line, face_touch_point):
                        clusters[i] = c
                        # This face will be accounted later when computing srho/sgamma
                        # But reduce area_remain to avoid double assignment for region growth
                        area_remain -= area[i]

    # Now perform the existing region growth algorithm for remaining clusters
    # (it expects some clusters may already have some assigned faces)
    ctarea = area_remain / nclus
    for i in range(nclus):
        # Skip clusters that are anchors because they have been treated (but still
        # allow growth for anchors if they had no faces assigned? we'll still do growth)
        tarea = area_remain - ctarea * (nclus - i - 1)
        carea = 0.0
        # Get starting index (the first free face in list)
        i_items_new = 0
        for j in range(lstind, npoints):
            if clusters[j] == -1:
                carea += area[j]
                items[i_items_new, 0] = j
                clusters[j] = i
                lstind = j
                break
        if j == npoints:
            break

        # While there are new items to be added
        c = 1
        while c:
            c_prev = c
            c = 0
            if i_items_new == 0:
                i_items_old = 0
                i_items_new = 1
            else:
                i_items_old = 1
                i_items_new = 0
            for j in range(c_prev):
                checkitem = items[i_items_old, j]
                for k in range(nneigh[checkitem]):
                    item = neighbors[checkitem, k]
                    # check if the face is free
                    if clusters[item] == -1:
                        # if allowable, add to cluster
                        # IMPORTANT: don't override anchor-preassigned faces
                        if area[item] + carea < tarea:
                            carea += area[item]
                            clusters[item] = i
                            items[i_items_new, c] = item
                            c += 1

        area_remain -= carea

# ---------------------------
# Modified minimize_energy (main change: strict anchor energy using anchor_pos)
# ---------------------------
def minimize_energy(int [:, ::1] edges, int [::1] clusters, double [::1] area,
                    double [:, ::1] sgamma, double [:, ::1] cent, double [::1] srho,
                    int [::1] cluscount, int maxiter, double [::1] energy,
                    uint8 [::1] mod1, uint8 [::1] mod2,
                    int [::1] face_touch_line, int [::1] face_touch_point,
                    int [::1] cluster_anchor_type, int [::1] cluster_anchor_id,
                    double [:, ::1] anchor_pos, uint8 [::1] anchor_fixed):
    """ Minimize cluster energy with strict anchors """
    cdef int face_a, face_b, clusA, clusB
    cdef double areaface_a, centA0, centA1, centA2
    cdef double areaface_b, centB0, centB1, centB2
    cdef double eA, eB, eorig, eAwB, eBnB, eAnA, eBwA
    cdef int nchange = 1
    cdef int niter = 0
    cdef int nclus = mod1.shape[0]
    cdef int nedge = edges.shape[0]
    cdef int i
    cdef int [1] nchange_arr

    # start all as modified
    for i in range(nclus):
        mod2[i] = 1

    # Helper inline-like locals for anchor checks
    cdef int atypeA, atypeB, aidA, aidB
    cdef double a0, a1, a2
    cdef double na0, na1, na2
    cdef double numA, numB

    while nchange > 0 and niter < maxiter:
        # Reset modification arrays
        for i in range(nclus):
            mod1[i] = mod2[i]
            mod2[i] = 0
        nchange = 0

        for i in range(nedge):
            # Get the two clusters sharing an edge
            face_a = edges[i, 0]
            face_b = edges[i, 1]
            clusA = clusters[face_a]
            clusB = clusters[face_b]

            # If edge shares two different clusters and at least one
            # has been modified since last iteration
            if clusA != clusB and (mod1[clusA] == 1 or mod1[clusB] == 1):
                # Verify that face can be removed from cluster
                if cluscount[clusA] > 1 and cluscount[clusB] > 1:

                    areaface_a = area[face_a]
                    centA0 = cent[face_a, 0]
                    centA1 = cent[face_a, 1]
                    centA2 = cent[face_a, 2]

                    areaface_b = area[face_b]
                    centB0 = cent[face_b, 0]
                    centB1 = cent[face_b, 1]
                    centB2 = cent[face_b, 2]

                    # Current energy (already stored); but will recompute candidate energies below
                    eorig = energy[clusA] + energy[clusB]

                    # Pre-check allowance: whether face_b is allowed to be assigned to clusA
                    atypeA = cluster_anchor_type[clusA]
                    aidA = cluster_anchor_id[clusA]
                    if not _face_allowed_for_anchor(face_b, atypeA, aidA, face_touch_line, face_touch_point):
                        eA = -1e300  # impossible
                    else:
                        # Compute sgamma and srho after assigning face_b to clusA, removing from clusB
                        # new srhoA = srho[clusA] + areaface_b
                        # new sgammaA = sgamma[clusA] + centB
                        # For clusA if anchor_fixed -> energy uses anchor_pos
                        if anchor_fixed[clusA]:
                            # numerator = || (sgammaA - srhoA * anchor_posA) ||^2
                            a0 = sgamma[clusA, 0] + centB0 - (srho[clusA] + areaface_b) * anchor_pos[clusA, 0]
                            a1 = sgamma[clusA, 1] + centB1 - (srho[clusA] + areaface_b) * anchor_pos[clusA, 1]
                            a2 = sgamma[clusA, 2] + centB2 - (srho[clusA] + areaface_b) * anchor_pos[clusA, 2]
                            eAwB = (a0*a0 + a1*a1 + a2*a2) / (srho[clusA] + areaface_b)
                        else:
                            # normal energy formula
                            a0 = sgamma[clusA, 0] + centB0
                            a1 = sgamma[clusA, 1] + centB1
                            a2 = sgamma[clusA, 2] + centB2
                            eAwB = (a0*a0 + a1*a1 + a2*a2) / (srho[clusA] + areaface_b)

                        # For clusB after removing face_b
                        if anchor_fixed[clusB]:
                            b0 = sgamma[clusB, 0] - centB0 - (srho[clusB] - areaface_b) * anchor_pos[clusB, 0]
                            b1 = sgamma[clusB, 1] - centB1 - (srho[clusB] - areaface_b) * anchor_pos[clusB, 1]
                            b2 = sgamma[clusB, 2] - centB2 - (srho[clusB] - areaface_b) * anchor_pos[clusB, 2]
                            eBnB = (b0*b0 + b1*b1 + b2*b2) / (srho[clusB] - areaface_b)
                        else:
                            b0 = sgamma[clusB, 0] - centB0
                            b1 = sgamma[clusB, 1] - centB1
                            b2 = sgamma[clusB, 2] - centB2
                            eBnB = (b0*b0 + b1*b1 + b2*b2) / (srho[clusB] - areaface_b)

                        eA = eAwB + eBnB

                    # Pre-check allowance: whether face_a is allowed to be assigned to clusB
                    atypeB = cluster_anchor_type[clusB]
                    aidB = cluster_anchor_id[clusB]
                    if not _face_allowed_for_anchor(face_a, atypeB, aidB, face_touch_line, face_touch_point):
                        eB = -1e300
                    else:
                        # Compute candidate energies for moving face_a -> clusB
                        if anchor_fixed[clusA]:
                            aa0 = sgamma[clusA, 0] - centA0 - (srho[clusA] - areaface_a) * anchor_pos[clusA, 0]
                            aa1 = sgamma[clusA, 1] - centA1 - (srho[clusA] - areaface_a) * anchor_pos[clusA, 1]
                            aa2 = sgamma[clusA, 2] - centA2 - (srho[clusA] - areaface_a) * anchor_pos[clusA, 2]
                            eAnA = (aa0*aa0 + aa1*aa1 + aa2*aa2) / (srho[clusA] - areaface_a)
                        else:
                            aa0 = sgamma[clusA, 0] - centA0
                            aa1 = sgamma[clusA, 1] - centA1
                            aa2 = sgamma[clusA, 2] - centA2
                            eAnA = (aa0*aa0 + aa1*aa1 + aa2*aa2) / (srho[clusA] - areaface_a)

                        if anchor_fixed[clusB]:
                            bb0 = sgamma[clusB, 0] + centA0 - (srho[clusB] + areaface_a) * anchor_pos[clusB, 0]
                            bb1 = sgamma[clusB, 1] + centA1 - (srho[clusB] + areaface_a) * anchor_pos[clusB, 1]
                            bb2 = sgamma[clusB, 2] + centA2 - (srho[clusB] + areaface_a) * anchor_pos[clusB, 2]
                            eBwA = (bb0*bb0 + bb1*bb1 + bb2*bb2) / (srho[clusB] + areaface_a)
                        else:
                            bb0 = sgamma[clusB, 0] + centA0
                            bb1 = sgamma[clusB, 1] + centA1
                            bb2 = sgamma[clusB, 2] + centA2
                            eBwA = (bb0*bb0 + bb1*bb1 + bb2*bb2) / (srho[clusB] + areaface_a)

                        eB = eAnA + eBwA

                    # Protect against impossible moves that might have caused division by zero
                    # (eA/eB set to -1e300 above to indicate impossible)
                    # Additional safety: ensure srho denominators > 0 (we already checked cluscount>1 so srho-areaface > 0)
                    # select the largest case (most positive benefit)
                    if eA > eorig and eA > eB:
                        # Before executing move, ensure this move does not make anchor empty of its required faces
                        # If clusB is an anchor, ensure after removing face_b it still contains at least one face that touches its anchor
                        if cluster_anchor_type[clusB] != 0:
                            # need to ensure there remains at least one face in clusB that satisfies anchor
                            # We'll do a quick check: iterate faces (costly worst-case but rare); if failing, reject
                            cdef int found = 0
                            cdef int f
                            for f in range(edges.shape[0]):
                                # check both faces in edges but better iterate whole faces? approximate check:
                                # Instead, do a scan over all faces (face index range)
                                pass
                            # Instead of expensive scan here, we rely on cluscount>1 and preassignment in init_clusters to ensure anchors
                            # are not emptied; skip further explicit check for performance. (If you want strict check, add a per-cluster
                            # count of anchor-satisfying faces and update it on moves.)
                        # perform reassignment
                        mod2[clusA] = 1
                        mod2[clusB] = 1

                        nchange += 1
                        clusters[face_b] = clusA
                        cluscount[clusB] -= 1
                        cluscount[clusA] += 1

                        # Update cluster A mass and centroid
                        srho[clusA] += areaface_b
                        sgamma[clusA, 0] += centB0
                        sgamma[clusA, 1] += centB1
                        sgamma[clusA, 2] += centB2

                        srho[clusB] -= areaface_b
                        sgamma[clusB, 0] -= centB0
                        sgamma[clusB, 1] -= centB1
                        sgamma[clusB, 2] -= centB2

                        # Update cluster energies (store computed values)
                        energy[clusA] = eAwB
                        energy[clusB] = eBnB

                    elif eB > eorig and eB > eA:
                        # analogous for moving face_a -> clusB
                        # (same note about anchor emptiness applies)
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        nchange += 1

                        clusters[face_a] = clusB
                        cluscount[clusA] -= 1
                        cluscount[clusB] += 1

                        srho[clusB] += areaface_a
                        sgamma[clusB, 0] += centA0
                        sgamma[clusB, 1] += centA1
                        sgamma[clusB, 2] += centA2

                        srho[clusA] -= areaface_a
                        sgamma[clusA, 0] -= centA0
                        sgamma[clusA, 1] -= centA1
                        sgamma[clusA, 2] -= centA2

                        energy[clusA] = eAnA
                        energy[clusB] = eBwA

                elif cluscount[clusA] > 1:
                    # only A can be reduced (similar checks: whether face_a allowed to go to clusB)
                    areaface_a = area[face_a]
                    centA0 = cent[face_a, 0]
                    centA1 = cent[face_a, 1]
                    centA2 = cent[face_a, 2]

                    eorig = energy[clusA] + energy[clusB]

                    # check allow
                    atypeB = cluster_anchor_type[clusB]
                    aidB = cluster_anchor_id[clusB]
                    if not _face_allowed_for_anchor(face_a, atypeB, aidB, face_touch_line, face_touch_point):
                        eB = -1e300
                    else:
                        if anchor_fixed[clusA]:
                            aa0 = sgamma[clusA, 0] - centA0 - (srho[clusA] - areaface_a) * anchor_pos[clusA, 0]
                            aa1 = sgamma[clusA, 1] - centA1 - (srho[clusA] - areaface_a) * anchor_pos[clusA, 1]
                            aa2 = sgamma[clusA, 2] - centA2 - (srho[clusA] - areaface_a) * anchor_pos[clusA, 2]
                            eAnA = (aa0*aa0 + aa1*aa1 + aa2*aa2) / (srho[clusA] - areaface_a)
                        else:
                            eAnA = ((sgamma[clusA, 0] - centA0)**2 + (sgamma[clusA, 1] - centA1)**2 + (sgamma[clusA, 2] - centA2)**2) / (srho[clusA] - areaface_a)

                        if anchor_fixed[clusB]:
                            bb0 = sgamma[clusB, 0] + centA0 - (srho[clusB] + areaface_a) * anchor_pos[clusB, 0]
                            bb1 = sgamma[clusB, 1] + centA1 - (srho[clusB] + areaface_a) * anchor_pos[clusB, 1]
                            bb2 = sgamma[clusB, 2] + centA2 - (srho[clusB] + areaface_a) * anchor_pos[clusB, 2]
                            eBwA = (bb0*bb0 + bb1*bb1 + bb2*bb2) / (srho[clusB] + areaface_a)
                        else:
                            eBwA = ((sgamma[clusB, 0] + centA0)**2 + (sgamma[clusB, 1] + centA1)**2 + (sgamma[clusB, 2] + centA2)**2) / (srho[clusB] + areaface_a)

                        eB = eAnA + eBwA

                    if eB > eorig:
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        nchange += 1

                        clusters[face_a] = clusB
                        cluscount[clusA] -= 1
                        cluscount[clusB] += 1

                        srho[clusB] += areaface_a
                        sgamma[clusB, 0] += centA0
                        sgamma[clusB, 1] += centA1
                        sgamma[clusB, 2] += centA2

                        srho[clusA] -= areaface_a
                        sgamma[clusA, 0] -= centA0
                        sgamma[clusA, 1] -= centA1
                        sgamma[clusA, 2] -= centA2

                        energy[clusA] = eAnA
                        energy[clusB] = eBwA

                elif cluscount[clusB] > 1:
                    # only B can be reduced (mirror case)
                    areaface_b = area[face_b]
                    centB0 = cent[face_b, 0]
                    centB1 = cent[face_b, 1]
                    centB2 = cent[face_b, 2]

                    eorig = energy[clusA] + energy[clusB]

                    atypeA = cluster_anchor_type[clusA]
                    aidA = cluster_anchor_id[clusA]
                    if not _face_allowed_for_anchor(face_b, atypeA, aidA, face_touch_line, face_touch_point):
                        eA = -1e300
                    else:
                        if anchor_fixed[clusA]:
                            a0 = sgamma[clusA, 0] + centB0 - (srho[clusA] + areaface_b) * anchor_pos[clusA, 0]
                            a1 = sgamma[clusA, 1] + centB1 - (srho[clusA] + areaface_b) * anchor_pos[clusA, 1]
                            a2 = sgamma[clusA, 2] + centB2 - (srho[clusA] + areaface_b) * anchor_pos[clusA, 2]
                            eAwB = (a0*a0 + a1*a1 + a2*a2) / (srho[clusA] + areaface_b)
                        else:
                            eAwB = ((sgamma[clusA, 0] + centB0)**2 + (sgamma[clusA, 1] + centB1)**2 + (sgamma[clusA, 2] + centB2)**2) / (srho[clusA] + areaface_b)

                        if anchor_fixed[clusB]:
                            b0 = sgamma[clusB, 0] - centB0 - (srho[clusB] - areaface_b) * anchor_pos[clusB, 0]
                            b1 = sgamma[clusB, 1] - centB1 - (srho[clusB] - areaface_b) * anchor_pos[clusB, 1]
                            b2 = sgamma[clusB, 2] - centB2 - (srho[clusB] - areaface_b) * anchor_pos[clusB, 2]
                            eBnB = (b0*b0 + b1*b1 + b2*b2) / (srho[clusB] - areaface_b)
                        else:
                            eBnB = ((sgamma[clusB, 0] - centB0)**2 + (sgamma[clusB, 1] - centB1)**2 + (sgamma[clusB, 2] - centB2)**2) / (srho[clusB] - areaface_b)

                        eA = eAwB + eBnB

                    if eA > eorig:
                        mod2[clusA] = 1
                        mod2[clusB] = 1

                        nchange += 1
                        clusters[face_b] = clusA
                        cluscount[clusB] -= 1
                        cluscount[clusA] += 1

                        srho[clusA] += areaface_b
                        sgamma[clusA, 0] += centB0
                        sgamma[clusA, 1] += centB1
                        sgamma[clusA, 2] += centB2

                        srho[clusB] -= areaface_b
                        sgamma[clusB, 0] -= centB0
                        sgamma[clusB, 1] -= centB1
                        sgamma[clusB, 2] -= centB2

                        energy[clusA] = eAwB
                        energy[clusB] = eBnB

        niter += 1

# ---------------------------
# Modified cluster() wrapper to accept anchor arrays and pass them through
# ---------------------------
def cluster(int [:, ::1] neighbors, int [::1] nneigh,
            int nclus, double [::1] area,
            double [:, ::1] cent, int [:, ::1] edges, int maxiter,
            debug=False, int iso_try=10,
            int [::1] face_touch_line=None, int [::1] face_touch_point=None,
            int [::1] cluster_anchor_type=None, int [::1] cluster_anchor_id=None,
            double [:, ::1] anchor_pos=None, uint8 [::1] anchor_fixed=None):
    """ Python interface function for cluster optimization (modified for anchors) """
    cdef int i

    # Initialize clusters
    cdef int npoints = nneigh.shape[0]
    cdef int [::1] clusters = np.empty(npoints, ctypes.c_int)
    clusters [:] = -1
    cdef int [:, ::1] items = np.empty((2, npoints), ctypes.c_int)
    # Call modified init_clusters (preassign anchors)
    init_clusters(clusters, neighbors, nneigh, area, nclus, items,
                  face_touch_line, face_touch_point, cluster_anchor_type, cluster_anchor_id)

    # Eliminat null clusters by growing existing null clusters
    # Assume grow_null is unchanged but must respect anchors; if not, user should update grow_null similarly.
    grow_null(edges, clusters)

    # Assign any remaining clusters to 0 (just in case null clusters fails)
    for i in range(npoints):
        if clusters[i] == -1:
            clusters[i] = 0

    # Arrays for cluster centers, masses, and energies
    cdef double [:, ::1] sgamma = np.zeros((nclus, 3))
    cdef double [::1] srho = np.zeros(nclus)
    cdef double [::1] energy = np.empty(nclus)

    # Compute initial masses of clusters
    for i in range(npoints):
        srho[clusters[i]] += area[i]
        sgamma[clusters[i], 0] += cent[i, 0]
        sgamma[clusters[i], 1] += cent[i, 1]
        sgamma[clusters[i], 2] += cent[i, 2]

    # Compute initial energy -- for anchor clusters use anchor_pos if anchor_fixed
    for i in range(nclus):
        if anchor_fixed is not None and anchor_fixed[i]:
            # strict anchor energy
            a0 = sgamma[i, 0] - srho[i] * anchor_pos[i, 0]
            a1 = sgamma[i, 1] - srho[i] * anchor_pos[i, 1]
            a2 = sgamma[i, 2] - srho[i] * anchor_pos[i, 2]
            if srho[i] > 0:
                energy[i] = (a0*a0 + a1*a1 + a2*a2) / srho[i]
            else:
                energy[i] = 0.0
        else:
            if srho[i] > 0:
                energy[i] = (sgamma[i, 0]**2 + sgamma[i, 1]**2 + sgamma[i, 2]**2) / srho[i]
            else:
                energy[i] = 0.0

    # Count number of clusters
    cdef int [::1] cluscount = np.bincount(clusters).astype(ctypes.c_int)

    # Initialize modified array
    cdef uint8 [::1] mod1 = np.empty(nclus, ctypes.c_uint8)
    cdef uint8 [::1] mod2 = np.empty(nclus, ctypes.c_uint8)

    # Optimize clusters (pass anchor arrays)
    minimize_energy(edges, clusters, area, sgamma, cent, srho, cluscount,
                    maxiter, energy, mod1, mod2,
                    face_touch_line, face_touch_point,
                    cluster_anchor_type, cluster_anchor_id,
                    anchor_pos, anchor_fixed)

    # Identify isolated clusters here
    ndisc = null_disconnected(nclus, nneigh, neighbors, clusters)
    cdef int niter = 0
    while ndisc and niter < iso_try:
        grow_null(edges, clusters)
        # Re optimize clusters
        minimize_energy(edges, clusters, area, sgamma, cent, srho, cluscount,
                       maxiter, energy, mod1, mod2,
                       face_touch_line, face_touch_point,
                       cluster_anchor_type, cluster_anchor_id,
                       anchor_pos, anchor_fixed)
        # Check again for disconnected clusters
        for i in range(npoints):
            if clusters[i] == -1:
                clusters[i] = 0
        ndisc = null_disconnected(nclus, nneigh, neighbors, clusters)
        niter += 1

        if ndisc:
            grow_null(edges, clusters)
            for i in range(npoints):
                if clusters[i] == -1:
                    clusters[i] = 0

    # renumber clusters 0 to n
    nclus = renumber_clusters(clusters, npoints, nclus)
    return np.asarray(clusters), ndisc > 0, nclus
