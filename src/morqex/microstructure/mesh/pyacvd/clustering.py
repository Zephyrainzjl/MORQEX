"""Point based clustering module"""
import ctypes

import numpy as np
from . import _clustering
from scipy import sparse


class Clustering:
    """Uniform point clustering based on ACVD.
    """

    def __init__(self, points, faces, weights=[]):
        """Check inputs and initializes neighbors"""
        self.points = points
        self.faces = faces 
        face_head = np.full(faces.shape[0], 3, dtype=np.int32)[:, np.newaxis]
        self._faces = np.append(face_head, faces, axis=1)

        self.clusters = None
        self.nclus = None
        self.remesh = None
        self._area = None
        self._wcent = None
        self._neigh = None
        self._nneigh = None
        self._edges = None
        self.weights = weights
        if not np.any(weights):
            weights = np.ones(points.shape[0])
        self._update_data(weights)

    def _update_data(self, weights=None):
        # Compute point weights and weighted points
        self._area, self._wcent = weighted_points(self.points, self._faces, additional_weights=weights)
        self._area *= weights

        # neighbors and edges
        self._neigh, self._nneigh = neighbors_from_mesh(self.points, self._faces)
        self._edges = _clustering.edge_id(self._neigh, self._nneigh)

    def cluster(self, nclus, maxiter=100, debug=False, iso_try=10):
        """Cluster points"""
        self.clusters, _, self.nclus = _clustering.cluster(
            self._neigh,
            self._nneigh,
            nclus,
            self._area,
            self._wcent,
            self._edges,
            maxiter,
            debug,
            iso_try,
        )

        return self.clusters


    def create_mesh(self, flipnorm=True):
        """Generates mesh from clusters"""
        if flipnorm:
            cnorm = self.cluster_norm
        else:
            cnorm = None

        # Generate mesh
        return create_mesh(self.points, self._faces, self._area, self.clusters, cnorm, flipnorm)

    @property
    def cluster_norm(self):
        """Return cluster norms"""
        if not hasattr(self, "clusters"):
            raise Exception("No clusters available")

        # Normals of original mesh
        norm = compute_vertex_normals(self.points, self.faces)

        # Compute normalized mean cluster normals
        cnorm = np.empty((self.nclus, 3))
        cnorm[:, 0] = np.bincount(self.clusters, weights=norm[:, 0] * self._area)
        cnorm[:, 1] = np.bincount(self.clusters, weights=norm[:, 1] * self._area)
        cnorm[:, 2] = np.bincount(self.clusters, weights=norm[:, 2] * self._area)
        weights = ((cnorm * cnorm).sum(1) ** 0.5).reshape((-1, 1))
        weights[weights == 0] = 1
        cnorm /= weights
        return cnorm

    @property
    def cluster_centroid(self):
        """Computes an area normalized value for each cluster"""
        wval = self.points * self._area.reshape(-1, 1)
        cval = np.vstack(
            (
                np.bincount(self.clusters, weights=wval[:, 0]),
                np.bincount(self.clusters, weights=wval[:, 1]),
                np.bincount(self.clusters, weights=wval[:, 2]),
            )
        )
        weights = np.bincount(self.clusters, weights=self._area)
        weights[weights == 0] = 1
        cval /= weights
        return cval.T


def cluster_centroid(cent, area, clusters):
    """Computes an area normalized centroid for each cluster"""

    # Check if null cluster exists
    null_clusters = np.any(clusters == -1)
    if null_clusters:
        clusters = clusters.copy()
        clusters[clusters == -1] = clusters.max() + 1

    wval = cent * area.reshape(-1, 1)
    cweight = np.bincount(clusters, weights=area)
    cweight[cweight == 0] = 1

    cval = (
        np.vstack(
            (
                np.bincount(clusters, weights=wval[:, 0]),
                np.bincount(clusters, weights=wval[:, 1]),
                np.bincount(clusters, weights=wval[:, 2]),
            )
        )
        / cweight
    )

    if null_clusters:
        cval[:, -1] = np.inf

    return cval.T


def create_mesh(points, faces, area, clusters, cnorm, flipnorm=True):
    """Generates a new mesh given cluster data

    moveclus is a boolean flag to move cluster centers to the surface of their
    corresponding cluster

    """
    faces = faces.reshape(-1, 4)
    # points = mesh.points
    if points.dtype != np.double:
        points = points.astype(np.double)

    # Compute centroids
    ccent = np.ascontiguousarray(cluster_centroid(points, area, clusters))

    # Create sparse matrix storing the number of adjacent clusters a point has
    rng = np.arange(faces.shape[0]).reshape((-1, 1))
    a = np.hstack((rng, rng, rng)).ravel()
    b = clusters[faces[:, 1:]].ravel()  # take?
    c = np.ones(len(a), dtype="bool")

    boolmatrix = sparse.csr_matrix((c, (a, b)), dtype="bool")

    # Find all points with three neighboring clusters.  Each of the three
    # cluster neighbors becomes a point on a triangle
    nadjclus = boolmatrix.sum(1)
    adj = np.array(nadjclus == 3).nonzero()[0]
    idx = boolmatrix[adj].nonzero()[1]

    # Append these points and faces
    points = ccent
    faces = idx.reshape((-1, 3))

    # Remove duplicate faces
    faces = faces[unique_row_indices(np.sort(faces, 1))]

    #TODO 移除重叠面
    #TODO 从聚簇的表面生成方法

    # Mean normals of clusters each face is build from
    if flipnorm:
        adjcnorm = cnorm[faces].sum(1)
        adjcnorm /= np.linalg.norm(adjcnorm, axis=1).reshape(-1, 1)

        newnorm = compute_face_normals(points, faces)

        # If the dot is negative, reverse the order of those faces
        agg = (adjcnorm * newnorm).sum(1)  # dot product
        mask = agg < 0.0
        faces[mask] = faces[mask, ::-1]

    return points, faces


def unique_row_indices(a):
    """Indices of unique rows"""
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return idx


def weighted_points(points, faces, return_weighted=True, additional_weights=None):
    """Returns point weight based on area weight and weighted points.

    Points are weighted by adjacent area faces.

    Parameters
    ----------
    mesh : pv.PolyData
        All triangular surface mesh.

    return_weighted : bool, optional
        Returns vertices mutlipled by point weights.

    Returns
    -------
    pweight : np.ndarray, np.double
        Point weight array

    wvertex : np.ndarray, np.double
        Vertices mutlipled by their corresponding weights.  Returned only
        when return_weighted is True.

    """
    faces = faces.reshape(-1, 4)
    if faces.dtype != np.int32:
        faces = faces.astype(np.int32)
    points = points

    if additional_weights is not None:
        weights = additional_weights
        return_weighted = True
        if not weights.flags["C_CONTIGUOUS"]:
            weights = np.ascontiguousarray(weights, dtype=ctypes.c_double)
        elif weights.dtype != ctypes.c_double:
            weights = weights.astype(ctypes.c_double)

        if (weights < 0).any():
            raise Exception("Negative weights not allowed.")

    else:
        weights = np.array([])

    if points.dtype == np.float64:
        weighted_point_func = _clustering.weighted_points_double
    else:
        weighted_point_func = _clustering.weighted_points_float

    return weighted_point_func(points, faces, weights, return_weighted)


def neighbors_from_mesh(points, faces):
    """Assemble neighbor array.  Assumes all-triangular mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh to assemble neighbors from.

    Returns
    -------
    neigh : int np.ndarray [:, ::1]
        Indices of each neighboring node for each node.

    nneigh : int np.ndarray [::1]
        Number of neighbors for each node.
    """
    npoints = points.shape[0]
    faces = faces.reshape(-1, 4)
    if faces.dtype != np.int32:
        faces = faces.astype(np.int32)

    return _clustering.neighbors_from_faces(npoints, faces)



def compute_face_normals(vertices, faces):
    """
    计算三角形面的法向量
    """

    v1 = vertices[faces[:, 0], :]
    v2 = vertices[faces[:, 1], :]
    v3 = vertices[faces[:, 2], :]

    # 计算每个三角形的法向量
    face_normals = np.cross(v2 - v1, v3 - v1)
    face_normals /= np.sqrt(np.sum(face_normals ** 2, axis=1, keepdims=True))

    return face_normals


def compute_vertex_normals(vertices, faces):
    """
    计算三角表面网格的顶点法向量
    """

    # 计算每个三角形的法向量
    face_normals = compute_face_normals(vertices, faces)

    # 将每个三角形的法向量加到每个顶点的法向量上
    vertex_normals = np.zeros_like(vertices)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    vertex_normals /= np.sqrt(np.sum(vertex_normals ** 2, axis=1, keepdims=True))

    return vertex_normals


# from scipy import sparse

# faces = clus_mesh._faces
# points = clus_mesh.points
# clusters = clus_mesh.clusters  #每个顶点所属的簇索引
# area = clus_mesh._area  #每个顶点平均面积
# # neigh #每个顶点相连的顶点索引 预分配12个 超出可能卡死
# # faces = np.array([[3, 0, 1, 2], [3, 1, 3, 2], [3, 1, 4, 3], [3, 4, 5, 3]])
# # points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]])
# # clusters = np.array([0, 0, 0, 0, 1, 2]) 
# # clus_mesh._update_data()
# # area = np.ones(6)
# # points = mesh.points
# if points.dtype != np.double:
#     points = points.astype(np.double)

# # Compute centroids
# ccent = np.ascontiguousarray(cluster_centroid(points, area, clusters))  #簇中心坐标

# # Create sparse matrix storing the number of adjacent clusters a point has
# rng = np.arange(faces.shape[0]).reshape((-1, 1))  #0 0 0 1 1 1 ...
# a = np.hstack((rng, rng, rng)).ravel() #所有顶点所属三角面索引
# b = clusters[faces[:, 1:]].ravel() #面的每个顶点所属的簇
# c = np.ones(len(a), dtype="bool")  #相邻标记为1

# boolmatrix = sparse.csr_matrix((c, (a, b)), dtype="bool")  #数据 行索引 列索引

# # Find all points with three neighboring clusters.  Each of the three
# # cluster neighbors becomes a point on a triangle
# nadjclus = boolmatrix.sum(1)  #沿行求和取值1~3 一个面的三个顶点属于三个不同簇
# adj = np.array(nadjclus == 3).nonzero()[0]  #满足构成三角形的簇中心
# idx = boolmatrix[adj].nonzero()[1]  #满足相邻的三个簇中心索引

# # Append these points and faces
# points = ccent   #簇中心作为新顶点
# tri_faces = idx.reshape((-1, 3))  

# # Remove duplicate faces
# tri_faces = remove_dul_points(tri_faces)[0]


# cluster定义在定点上, 若一个三角形三个顶点属于不同的cluster则连接这三个cluster中心. 

# 原 mesh 点 → cluster 聚类 → 计算 cluster 的几何质心

# 对每个原 mesh 面：
#     取三顶点所属 cluster → 得到一个 cluster 集合
#     如果刚好是三个不同 cluster：
#         这三个 cluster 的质心形成一个新三角形

# 输出：
#     points = cluster_centroids
#     faces = {每个 cluster 三元组}


# 两个簇 C_i 和 C_j 真正相邻
# ⇔
# 1 ∃ 三角面 (v0, v1, v2) 使得该面三点都属于 {C_i OR C_j}  簇包含至少一个面
# 且
# 2 该面属于它们的边界（不是内部面） 该面是两个簇的边界, 两个簇同时包含这个面