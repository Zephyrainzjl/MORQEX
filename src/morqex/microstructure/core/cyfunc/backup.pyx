%%cython 
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libcpp.vector cimport vector

# -------------------------
# 内部小函数（数值）
# -------------------------
cdef inline double edge_len(const double[:] a, const double[:] b):
    cdef double dx = a[0]-b[0]
    cdef double dy = a[1]-b[1]
    cdef double dz = a[2]-b[2]
    return sqrt(dx*dx + dy*dy + dz*dz)

cdef inline void vec_sub(const double[:] a, const double[:] b, double[:] out):
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2]

cdef inline double vec_dot(const double[:] a, const double[:] b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

cdef inline double vec_norm(const double[:] a):
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

cdef inline void vec_scale(double[:] a, double s):
    a[0] *= s; a[1] *= s; a[2] *= s

cdef inline void vec_add_inplace(double[:] a, const double[:] b):
    a[0] += b[0]; a[1] += b[1]; a[2] += b[2]

cdef inline void vec_copy(const double[:] a, double[:] out):
    out[0] = a[0]; out[1] = a[1]; out[2] = a[2]

cdef inline void normalize3_inplace(double[:] v):
    cdef double n = vec_norm(v)
    if n > 1e-12:
        v[0] /= n; v[1] /= n; v[2] /= n

# -------------------------
# 帮助构造拓扑（较直接的Python层容器）
# -------------------------
def build_topology(np.ndarray[np.int32_t, ndim=2] tris, int nV):
    """
    返回：
      neighbors: list of sets (vertex->neighbor vertices)
      edge_faces: dict {(min,max): [face_idx,...]}
      v_faces: list of incident faces per vertex
    """
    cdef int i, a, b, c
    neighbors = [set() for _ in range(nV)]
    edge_faces = dict()
    v_faces = [[] for _ in range(nV)]
    cdef int n_tris = tris.shape[0]
    for i in range(n_tris):
        a = int(tris[i,0]); b = int(tris[i,1]); c = int(tris[i,2])
        neighbors[a].add(b); neighbors[a].add(c)
        neighbors[b].add(a); neighbors[b].add(c)
        neighbors[c].add(a); neighbors[c].add(b)
        v_faces[a].append(i); v_faces[b].append(i); v_faces[c].append(i)
        def add_edge_face(i1, i2, face):
            key = (i1, i2) if i1 < i2 else (i2, i1)
            if key in edge_faces:
                edge_faces[key].append(face)
            else:
                edge_faces[key] = [face]
        add_edge_face(a,b,i); add_edge_face(b,c,i); add_edge_face(c,a,i)
    return neighbors, edge_faces, v_faces

# -------------------------
# 计算每个 feature vertex 的 feature-edge count（corner/crease 判断）
# feature_edges_set: python set of (min,max) tuples
# feature_vertex_mask: boolean array or list
# -------------------------
def _feature_vertex_counts(int nV, np.ndarray[np.int32_t, ndim=2] feature_edges):
    """
    返回 counts: length nV 的 list, counts[i] = feature-edge 与 i 相连数量
    """
    counts = [0]*nV
    cdef int n_edge = feature_edges.shape[0]
    cdef int i, j
    for i in range(n_edge):
        a = int(feature_edges[i,0]); b = int(feature_edges[i,1])
        counts[a] += 1
        counts[b] += 1
    return counts

# -------------------------
# Collapse (feature-aware)
# -------------------------
def collapse_short_edges_feature(np.ndarray[np.double_t, ndim=2] verts,
                                 np.ndarray[np.int32_t, ndim=2] tris,
                                 double L,
                                 np.ndarray[np.int32_t, ndim=2] feature_edges,
                                 feature_vertices):
    """
    Feature-aware collapse:
      - 禁止 collapse feature edges（feature_edges）
      - 禁止 collapse 若任一端是 feature vertex（保守策略）
    Inputs:
      feature_edges: (k,2) int array
      feature_vertices: either boolean mask (nV,) or list of indices
    """
    cdef int nV = verts.shape[0]
    cdef int nF = tris.shape[0]
    neighbors, edge_faces, v_faces = build_topology(tris, nV)

    # build feature edge set
    feature_set = set()
    cdef int n_edge = feature_edges.shape[0]
    cdef int i, j
    for i in range(n_edge):
        a = int(feature_edges[i,0]); b = int(feature_edges[i,1])
        key = (a,b) if a<b else (b,a)
        feature_set.add(key)

    # build feature vertex mask (boolean list)
    if isinstance(feature_vertices, np.ndarray):
        if feature_vertices.dtype == np.bool_:
            fv_mask = [bool(feature_vertices[i]) for i in range(nV)]
        else:
            # assume list of indices
            fv_mask = [False]*nV
            for idx in feature_vertices:
                fv_mask[int(idx)] = True
    else:
        # assume Python iterable of indices
        fv_mask = [False]*nV
        for idx in feature_vertices:
            fv_mask[int(idx)] = True

    # build edge list with lengths
    edge_lens = []
    for (i,j) in edge_faces.keys():
        # skip feature edges directly
        if (i,j) in feature_set:
            continue
        edge_lens.append((edge_len(verts[i], verts[j]), i, j))
    edge_lens.sort()  # ascending

    verts_list = [ [verts[i,0], verts[i,1], verts[i,2]] for i in range(nV) ]
    tris_list = [ (int(tris[i,0]), int(tris[i,1]), int(tris[i,2])) for i in range(nF) ]
    removed_vert = [False]*nV

    cdef double collapse_thresh = 4.0/3.0 * L
    for length, i, j in edge_lens:
        if length > collapse_thresh:
            break
        if removed_vert[i] or removed_vert[j]:
            continue
        # conservative: 如果任一端是 feature vertex, 禁止 collapse
        if fv_mask[i] or fv_mask[j]:
            continue
        # merge j into i (midpoint)
        mx = 0.5*(verts_list[i][0] + verts_list[j][0])
        my = 0.5*(verts_list[i][1] + verts_list[j][1])
        mz = 0.5*(verts_list[i][2] + verts_list[j][2])
        verts_list[i][0] = mx; verts_list[i][1] = my; verts_list[i][2] = mz
        # reassign j -> i in triangles, remove degenerate
        new_tris = []
        for (a,b,c) in tris_list:
            na = i if a==j else a
            nb = i if b==j else b
            nc = i if c==j else c
            if na==nb or nb==nc or nc==na:
                continue
            new_tris.append((na,nb,nc))
        tris_list = new_tris
        removed_vert[j] = True

    # compact vertices
    remap = [-1]*len(verts_list)
    new_vs = []
    for idx,flag in enumerate(removed_vert):
        if not flag:
            remap[idx] = len(new_vs)
            new_vs.append(tuple(verts_list[idx]))
    new_tris = []
    for (a,b,c) in tris_list:
        ra = remap[a]; rb = remap[b]; rc = remap[c]
        if ra==rb or rb==rc or rc==ra:
            continue
        new_tris.append((ra,rb,rc))
    return np.asarray(new_vs, dtype=np.double), np.asarray(new_tris, dtype=np.int32)

# -------------------------
# Split (feature-aware)
# -------------------------
def split_long_edges_feature(np.ndarray[np.double_t, ndim=2] verts,
                             np.ndarray[np.int32_t, ndim=2] tris,
                             double L,
                             np.ndarray[np.int32_t, ndim=2] feature_edges):
    """
    Splits edges longer than split_thresh * L, but **do not split feature edges**.
    Conservative implementaton: 不对 feature edges 做任何拆分。
    """
    cdef int nV = verts.shape[0]
    neighbors, edge_faces, v_faces = build_topology(tris, nV)
    feature_set = set()
    cdef int n_edge = feature_edges.shape[0]
    cdef int i, j
    for i in range(n_edge):
        a = int(feature_edges[i,0]); b = int(feature_edges[i,1])
        key = (a,b) if a<b else (b,a)
        feature_set.add(key)

    edge_newv = dict()
    new_verts_list = [tuple(verts[i]) for i in range(nV)]
    tris_list = [ (int(tris[i,0]), int(tris[i,1]), int(tris[i,2])) for i in range(tris.shape[0]) ]

    split_thresh = 4.0/5.0 * L
    for (i,j), faces in edge_faces.items():
        key = (i,j)
        if key in feature_set:
            continue
        a = verts[i]; b = verts[j]
        if edge_len(a,b) > split_thresh:
            mx = 0.5*(a[0]+b[0]); my = 0.5*(a[1]+b[1]); mz = 0.5*(a[2]+b[2])
            new_idx = len(new_verts_list)
            new_verts_list.append((mx,my,mz))
            edge_newv[key] = new_idx

    new_tris = []
    for (a,b,c) in tris_list:
        s_ab = edge_newv.get((a,b), edge_newv.get((b,a), -1))
        s_bc = edge_newv.get((b,c), edge_newv.get((c,b), -1))
        s_ca = edge_newv.get((c,a), edge_newv.get((a,c), -1))
        if s_ab == -1 and s_bc == -1 and s_ca == -1:
            new_tris.append((a,b,c)); continue
        # handle split cases robustly:
        # Use a standard triangle midpoint subdivision approach:
        # Create mapping from original edges to midpoints
        mid = dict()
        if s_ab != -1: mid[(a,b)] = s_ab; mid[(b,a)] = s_ab
        if s_bc != -1: mid[(b,c)] = s_bc; mid[(c,b)] = s_bc
        if s_ca != -1: mid[(c,a)] = s_ca; mid[(a,c)] = s_ca
        # number of new mids
        mcount = len(set(mid.values()))
        if mcount == 1:
            # one edge mid -> split into two triangles
            m = list(set(mid.values()))[0]
            # find which edge
            if s_ab != -1:
                new_tris.append((a,m,c)); new_tris.append((m,b,c))
            elif s_bc != -1:
                new_tris.append((a,b,m)); new_tris.append((a,m,c))
            else:
                new_tris.append((a,b,m)); new_tris.append((m,b,c))
            continue
        elif mcount == 2:
            # two edges -> split into three triangles
            mids = list(set(mid.values()))
            m1 = mids[0]; m2 = mids[1]
            # produce fan triangulation around the vertex opposite the missing split
            # find which edge was not split
            if s_ab == -1:
                # ab not split => other two split
                new_tris.append((a,b,m2))
                new_tris.append((a,m2,m1))
                new_tris.append((a,m1,c))
                continue
            if s_bc == -1:
                new_tris.append((b,c,m1))
                new_tris.append((b,m1,m2))
                new_tris.append((b,m2,a))
                continue
            if s_ca == -1:
                new_tris.append((c,a,m2))
                new_tris.append((c,m2,m1))
                new_tris.append((c,m1,b))
                continue
        else:
            # three splits -> central triangle + 3 outer
            m_ab = s_ab; m_bc = s_bc; m_ca = s_ca
            new_tris.append((a, m_ab, m_ca))
            new_tris.append((m_ab, b, m_bc))
            new_tris.append((m_ca, m_bc, c))
            new_tris.append((m_ab, m_bc, m_ca))
            continue
    newV = np.asarray(new_verts_list, dtype=np.double)
    newT = np.asarray(new_tris, dtype=np.int32)
    return newV, newT

# -------------------------
# Edge flip (feature-aware)
# -------------------------
def edge_flips_feature(np.ndarray[np.double_t, ndim=2] verts,
                       np.ndarray[np.int32_t, ndim=2] tris,
                       np.ndarray[np.int32_t, ndim=2] feature_edges):
    """
    Flip interior edges if it reduces valence error, but never flip feature edges,
    and avoid flips that would remove/create feature edges.
    """
    cdef int nV = verts.shape[0]
    neighbors, edge_faces, v_faces = build_topology(tris, nV)
    feature_set = set()
    cdef int n_edge = feature_edges.shape[0]
    cdef int i, j
    for i in range(n_edge):
        a = int(feature_edges[i,0]); b = int(feature_edges[i,1])
        key = (a,b) if a<b else (b,a)
        feature_set.add(key)

    tris_list = [ [int(tris[i,0]), int(tris[i,1]), int(tris[i,2])] for i in range(tris.shape[0]) ]

    def valence(idx):
        return len(neighbors[idx])

    # consider each interior edge
    for (i,j), faces in list(edge_faces.items()):
        if (i,j) in feature_set:
            continue
        if len(faces) != 2:
            continue
        f1 = faces[0]; f2 = faces[1]
        tri1 = tris_list[f1]; tri2 = tris_list[f2]
        opp1 = -1; opp2 = -1
        for v in tri1:
            if v != i and v != j:
                opp1 = v; break
        for v in tri2:
            if v != i and v != j:
                opp2 = v; break
        if opp1 == -1 or opp2 == -1:
            continue
        # if flipping would create a feature edge (opp1,opp2) which conflicts with existing feature set,
        # and that new edge would be a feature edge but we don't have it - to be safe, avoid flip if (opp1,opp2) is in feature_set
        newkey = (opp1,opp2) if opp1<opp2 else (opp2,opp1)
        if newkey in feature_set:
            continue
        # don't flip if flipping would remove an existing feature edge adjacent to this quad (conservative)
        # compute current valence error vs new valence error
        cur_err = (valence(i)-6)**2 + (valence(j)-6)**2 + (valence(opp1)-6)**2 + (valence(opp2)-6)**2
        new_err = ( (valence(i)-1-6)**2 + (valence(j)-1-6)**2 +
                    (valence(opp1)+1-6)**2 + (valence(opp2)+1-6)**2 )
        if new_err < cur_err:
            # apply flip
            tris_list[f1] = [opp1, opp2, i]
            tris_list[f2] = [opp2, opp1, j]
            # note: neighbors not updated (one-pass)
    return np.asarray(tris_list, dtype=np.int32)

# -------------------------
# Tangential relaxation (feature-aware)
# -------------------------
def tangential_relaxation_feature(np.ndarray[np.double_t, ndim=2] verts,
                                 np.ndarray[np.int32_t, ndim=2] tris,
                                 np.ndarray[np.int32_t, ndim=2] feature_edges,
                                 feature_vertices,
                                 int iterations = 1):
    """
    Feature-aware smoothing:
      - corner vertices (feature-edge-count >=3): fixed
      - crease vertices (count==2): compute feature-line direction and project movement onto that line
      - smooth vertices: usual tangential smoothing (barycenter projected to tangent plane)
    """
    cdef int nV = verts.shape[0]
    neighbors, edge_faces, v_faces = build_topology(tris, nV)
    # build feature structures
    feature_set = set()
    cdef int n_edges = feature_edges.shape[0]
    cdef int i, j
    for i in range(n_edges):
        a = int(feature_edges[i,0]); b = int(feature_edges[i,1])
        key = (a,b) if a<b else (b,a)
        feature_set.add(key)
    # feature vertex mask
    if isinstance(feature_vertices, np.ndarray):
        if feature_vertices.dtype == np.bool_:
            fv_mask = [bool(feature_vertices[i]) for i in range(nV)]
        else:
            fv_mask = [False]*nV
            for idx in feature_vertices:
                fv_mask[int(idx)] = True
    else:
        fv_mask = [False]*nV
        for idx in feature_vertices:
            fv_mask[int(idx)] = True
    # counts per vertex
    counts = [0]*nV
    feat_neighbors = [[] for _ in range(nV)]
    for (a,b) in feature_set:
        counts[a] += 1; counts[b] += 1
        feat_neighbors[a].append(b); feat_neighbors[b].append(a)

    vnorms, fnorms = compute_normals(verts, tris)
    verts_arr = verts.copy()
    cdef int it
    for it in range(iterations):
        newpos = np.empty_like(verts_arr)
        for i in range(nV):
            # corner: fixed
            if counts[i] >= 3:
                newpos[i,0] = verts_arr[i,0]; newpos[i,1] = verts_arr[i,1]; newpos[i,2] = verts_arr[i,2]
                continue
            # compute barycenter of neighbors
            if len(neighbors[i]) == 0:
                newpos[i,:] = verts_arr[i,:]; continue
            sx = 0.0; sy = 0.0; sz = 0.0
            for j in neighbors[i]:
                sx += verts_arr[j,0]; sy += verts_arr[j,1]; sz += verts_arr[j,2]
            invn = 1.0 / len(neighbors[i])
            cx = sx * invn; cy = sy * invn; cz = sz * invn
            # smooth vs crease handling
            if counts[i] == 2:
                # crease: restrict movement along feature direction
                # compute direction from the two feature neighbors (approx)
                n1 = feat_neighbors[i][0]; n2 = feat_neighbors[i][1] if len(feat_neighbors[i])>1 else feat_neighbors[i][0]
                dirvec = [ verts_arr[n1,0]-verts_arr[i,0], verts_arr[n1,1]-verts_arr[i,1], verts_arr[n1,2]-verts_arr[i,2] ]
                if n1 != n2:
                    # average two directions
                    d2 = [ verts_arr[n2,0]-verts_arr[i,0], verts_arr[n2,1]-verts_arr[i,1], verts_arr[n2,2]-verts_arr[i,2] ]
                    dirvec[0] = 0.5*(dirvec[0] + d2[0])
                    dirvec[1] = 0.5*(dirvec[1] + d2[1])
                    dirvec[2] = 0.5*(dirvec[2] + d2[2])
                # project (cx - xi) onto dirvec, produce new position xi + proj
                rx = cx - verts_arr[i,0]; ry = cy - verts_arr[i,1]; rz = cz - verts_arr[i,2]
                dv = dirvec
                dvnorm = sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])
                if dvnorm < 1e-12:
                    # fallback: don't move
                    newpos[i,0] = verts_arr[i,0]; newpos[i,1] = verts_arr[i,1]; newpos[i,2] = verts_arr[i,2]
                else:
                    invdn = 1.0/dvnorm
                    dv0 = dv[0]*invdn; dv1 = dv[1]*invdn; dv2 = dv[2]*invdn
                    dot = rx*dv0 + ry*dv1 + rz*dv2
                    newpos[i,0] = verts_arr[i,0] + dot*dv0
                    newpos[i,1] = verts_arr[i,1] + dot*dv1
                    newpos[i,2] = verts_arr[i,2] + dot*dv2
                continue
            else:
                # smooth vertex: tangential projection
                nx = vnorms[i,0]; ny = vnorms[i,1]; nz = vnorms[i,2]
                rx = cx - verts_arr[i,0]; ry = cy - verts_arr[i,1]; rz = cz - verts_arr[i,2]
                dot = rx*nx + ry*ny + rz*nz
                tx = cx - dot*nx; ty = cy - dot*ny; tz = cz - dot*nz
                newpos[i,0] = tx; newpos[i,1] = ty; newpos[i,2] = tz
        verts_arr = newpos
        vnorms, fnorms = compute_normals(verts_arr, tris)
    return verts_arr

# -------------------------
# compute normals (same as before)
# -------------------------
def compute_normals(np.ndarray[np.double_t, ndim=2] verts,
                    np.ndarray[np.int32_t, ndim=2] tris):
    cdef int nV = verts.shape[0]
    cdef int nF = tris.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] vnorms = np.zeros((nV,3), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] fnorms = np.zeros((nF,3), dtype=np.double)
    cdef int i, a, b, c
    cdef double ux,uy,uz, vx,vy,vz
    for i in range(nF):
        a = int(tris[i,0]); b = int(tris[i,1]); c = int(tris[i,2])
        ux = verts[b,0]-verts[a,0]; uy = verts[b,1]-verts[a,1]; uz = verts[b,2]-verts[a,2]
        vx = verts[c,0]-verts[a,0]; vy = verts[c,1]-verts[a,1]; vz = verts[c,2]-verts[a,2]
        fnorms[i,0] = uy*vz - uz*vy
        fnorms[i,1] = uz*vx - ux*vz
        fnorms[i,2] = ux*vy - uy*vx
        vnorms[a,0] += fnorms[i,0]; vnorms[a,1] += fnorms[i,1]; vnorms[a,2] += fnorms[i,2]
        vnorms[b,0] += fnorms[i,0]; vnorms[b,1] += fnorms[i,1]; vnorms[b,2] += fnorms[i,2]
        vnorms[c,0] += fnorms[i,0]; vnorms[c,1] += fnorms[i,1]; vnorms[c,2] += fnorms[i,2]
        # normalize face normal
        normalize3_inplace(fnorms[i])
    # normalize vertex normals
    cdef int vi
    for vi in range(nV):
        normalize3_inplace(vnorms[vi])
    return vnorms, fnorms

# -------------------------
# 高层 API：一次 remesh iteration (feature-preserving)
# -------------------------
def remesh_iteration_feature(np.ndarray[np.double_t, ndim=2] verts,
                             np.ndarray[np.int32_t, ndim=2] tris,
                             double L,
                             np.ndarray[np.int32_t, ndim=2] feature_edges,
                             feature_vertices,
                             int relax_iter = 1):
    """
    1) collapse (feature-aware)
    2) split (feature-aware)
    3) edge flips (feature-aware)
    4) tangential relaxation (feature-aware)
    """
    v1, t1 = collapse_short_edges_feature(verts, tris, L, feature_edges, feature_vertices)
    v2, t2 = split_long_edges_feature(v1, t1, L, feature_edges)
    t3 = edge_flips_feature(v2, t2, feature_edges)
    v3 = tangential_relaxation_feature(v2, t3, feature_edges, feature_vertices, iterations=relax_iter)
    return v3, t3
