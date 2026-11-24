# cython: language_level=3
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport cython
cimport numpy as np
# np.import_array()
from itertools import product

cdef tuple sorted_tuple3(int a, int b, int c):
    # 排序三个整数并返回元组
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return (a, b, c)

cdef list get_faces(int v0, int v1, int v2, int v3):
    # 生成四面体的四个面（已排序）
    return [
        sorted_tuple3(v0, v1, v2),
        sorted_tuple3(v0, v1, v3),
        sorted_tuple3(v0, v2, v3),
        sorted_tuple3(v1, v2, v3)
    ]
    
cpdef partition_domains(
    np.ndarray[np.float32_t, ndim=2] verts,
    np.ndarray[np.int32_t, ndim=2] cells,
    np.ndarray[np.int32_t, ndim=2] bound_faces
):
    # 预处理边界面集合
    cdef set boundary_set = set()
    cdef int i, a, b, c
    cdef np.int32_t[:, ::1] bound_faces_view = bound_faces
    for i in range(bound_faces.shape[0]):
        a = bound_faces_view[i, 0]
        b = bound_faces_view[i, 1]
        c = bound_faces_view[i, 2]
        boundary_set.add(sorted_tuple3(a, b, c))
    
    # 构建邻接字典（改用普通dict）
    cdef dict adjacency = {}
    cdef int cell_idx, v0, v1, v2, v3
    cdef np.int32_t[:, ::1] cells_view = cells
    cdef tuple face
    for cell_idx in range(cells.shape[0]):
        v0 = cells_view[cell_idx, 0]
        v1 = cells_view[cell_idx, 1]
        v2 = cells_view[cell_idx, 2]
        v3 = cells_view[cell_idx, 3]
        for face in get_faces(v0, v1, v2, v3):
            if face not in adjacency:
                adjacency[face] = []
            adjacency[face].append(cell_idx)
    
    # 构建邻接表
    cdef list cell_adj = [[] for _ in range(cells.shape[0])]
    for cell_idx in range(cells.shape[0]):
        v0 = cells_view[cell_idx, 0]
        v1 = cells_view[cell_idx, 1]
        v2 = cells_view[cell_idx, 2]
        v3 = cells_view[cell_idx, 3]
        for face in get_faces(v0, v1, v2, v3):
            if face in adjacency:  # 直接检查字典键
                for neighbor in adjacency[face]:
                    if neighbor != cell_idx:
                        cell_adj[cell_idx].append( (neighbor, face) )
    
    # 区域增长
    cdef list domains = []
    cdef np.ndarray[np.uint8_t, ndim=1] visited = np.zeros(cells.shape[0], dtype=np.uint8)
    cdef int current_cell, neighbor_cell
    cdef list queue
    for cell_idx in range(cells.shape[0]):
        if not visited[cell_idx]:
            queue = [cell_idx]
            visited[cell_idx] = 1
            domain = []
            while queue:
                current_cell = queue.pop(0)
                domain.append(current_cell)
                for item in cell_adj[current_cell]:
                    neighbor_cell = item[0]
                    face = item[1]
                    if not visited[neighbor_cell] and face not in boundary_set:
                        visited[neighbor_cell] = 1
                        queue.append(neighbor_cell)
            domains.append(tuple(domain))
    
    return domains


def find_tetrahedrons(np.ndarray[np.int32_t, ndim=2] faces, np.ndarray[np.float64_t, ndim=2] vertices=None):
    # 数据结构初始化
    cdef dict face_dict = {}
    cdef dict pair_dict = {}
    cdef set tetra_set = set()

    cdef int[:, :] faces_view = faces
    cdef int i, a, b, c, u, v, w, d
    cdef tuple sorted_face, pair_key

    # 构建面字典和边字典 通过顶点查询面和边
    for i in range(faces_view.shape[0]):
        a, b, c = faces_view[i, 0], faces_view[i, 1], faces_view[i, 2]
        sorted_face = tuple(sorted((a, b, c)))
        if sorted_face not in face_dict:
            face_dict[sorted_face] = []
        face_dict[sorted_face].append(i)  # 记录所有重复面索引

        # 更新边字典
        pairs = [(a, b), (a, c), (b, c)]
        third_vertices = [c, b, a]
        for (u, v), w in zip(pairs, third_vertices):
            if u > v:
                u, v = v, u
            pair_key = (u, v)
            if pair_key not in pair_dict:
                pair_dict[pair_key] = set()
            pair_dict[pair_key].add(w)

    # 四面体检测逻辑
    cdef tuple req_face0, req_face1, req_face2
    cdef list indices0, indices1, indices2, indices3
    cdef list candidates
    for sorted_face in face_dict:
        a, b, c = sorted_face
        for u, v in [(a, b), (a, c), (b, c)]:
            if u > v:
                u, v = v, u
            pair_key = (u, v)
            if pair_key not in pair_dict:
                continue
            for d in pair_dict[pair_key]:
                if d in (a, b, c):
                    continue
                # 验证其他边存在性
                if not ( ( (a < c and (a, c) in pair_dict and d in pair_dict[(a, c)]) or 
                          (c < a and (c, a) in pair_dict and d in pair_dict[(c, a)]) ) and
                        ( (b < c and (b, c) in pair_dict and d in pair_dict[(b, c)]) or 
                          (c < b and (c, b) in pair_dict and d in pair_dict[(c, b)]) ) ):
                    continue

                # 获取所有可能的面索引组合 
                #TODO 使用预先定义的映射进行列举
                req_face0 = tuple(sorted((a, b, d)))
                req_face1 = tuple(sorted((a, c, d)))
                req_face2 = tuple(sorted((b, c, d)))
                if req_face0 not in face_dict or req_face1 not in face_dict or req_face2 not in face_dict:
                    continue

                indices0 = face_dict[sorted_face]
                indices1 = face_dict[req_face0]
                indices2 = face_dict[req_face1]
                indices3 = face_dict[req_face2]

                # 生成所有可能的组合
                for combo in product(indices0, indices1, indices2, indices3):
                    if len(set(combo)) != 4:
                        continue  # 排除重复面索引
                    
                    # 记录有序结果
                    tetra_set.add( (
                        tuple(sorted(combo)), 
                        tuple(sorted([a, b, c, d])) 
                    ) )

    # 结果格式化
    if not tetra_set:
        return np.empty((0,4), dtype=np.int32), np.empty((0,4), dtype=np.int32)
    
    faces_arr = []
    verts_arr = []
    for face_indices, vertex_indices in sorted(tetra_set):
        faces_arr.append(face_indices)
        verts_arr.append(vertex_indices)
    
    return (
        np.array(verts_arr, dtype=np.int32),
        np.array(faces_arr, dtype=np.int32)
    )

def fill_holes_bound(np.ndarray[np.int32_t, ndim=2] faces):
    """
    填充由三条边界边构成的三角形孔洞(仅限边界边构成的三角形)
    参数：
        faces : 三角形面索引数组，形状(N,3)，数据类型np.int32
    返回：
        待填充的三角形面数组，形状(M,3)，数据类型np.int32
    """
    # 统计边出现次数
    cdef dict edge_counter = {}
    cdef int[:, :] faces_view = faces
    cdef int i, a, b, c
    cdef tuple edge
    
    # 遍历所有面统计边
    for i in range(faces.shape[0]):
        a = faces[i, 0]
        b = faces[i, 1]
        c = faces[i, 2]
        
        # 处理边 (a, b)
        edge = tuple(sorted((a, b)))
        if edge in edge_counter:
            edge_counter[edge] += 1
        else:
            edge_counter[edge] = 1
        
        # 处理边 (b, c)
        edge = tuple(sorted((b, c)))
        if edge in edge_counter:
            edge_counter[edge] += 1
        else:
            edge_counter[edge] = 1
        
        # 处理边 (c, a)
        edge = tuple(sorted((c, a)))
        if edge in edge_counter:
            edge_counter[edge] += 1
        else:
            edge_counter[edge] = 1
    
    # 收集边界边（出现1次的边）
    cdef set boundary_edges = set()
    for edge, cnt in edge_counter.items():
        if cnt == 1:
            boundary_edges.add(edge)
    
    # 构建邻接关系图
    cdef dict adj = {}
    cdef int v1, v2
    for edge in boundary_edges:
        v1, v2 = edge
        # 更新v1的邻接表
        if v1 in adj:
            adj[v1].append(v2)
        else:
            adj[v1] = [v2]
        # 更新v2的邻接表
        if v2 in adj:
            adj[v2].append(v1)
        else:
            adj[v2] = [v1]
    
    # 查找三角形孔洞
    cdef set holes = set()
    cdef list neighbors
    cdef int next_v, third_v
    cdef tuple sorted_tri
    
    for edge in boundary_edges:
        v1, v2 = edge
        # 获取v2的邻接顶点
        if v2 not in adj:
            continue
        neighbors = adj[v2]
        
        for next_v in neighbors:
            if next_v == v1:
                continue
            
            # 检查v1-next_v边是否也是边界边
            check_edge = tuple(sorted((v1, next_v)))
            if check_edge in boundary_edges:
                # 构成三角形孔洞
                sorted_tri = tuple(sorted([v1, v2, next_v]))
                holes.add(sorted_tri)
    
    # 转换为numpy数组
    if not holes:
        return np.empty((0,3), dtype=np.int32)
    return np.array(list(holes), dtype=np.int32)


def fill_holes(np.int32_t[:, :] faces):
    """填充由三个相邻顶点组成的缺失三角面
    参数:
        faces: 原始三角面数组，形状为(N,3)
    返回:
        新生成的三角面数组，形状为(M,3)
    """
    cdef:
        set existing_faces = set()  # 存储现有三角面的哈希集合
        dict adj = {}               # 邻接表字典
        list new_faces = []         # 结果容器
        int i, u, v, w, a, b
        tuple face_tuple, ring_tuple
        np.int32_t[:] face
    
    # 第一阶段：预处理原始数据
    for i in range(faces.shape[0]):
        face = faces[i]
        u, v, w = face[0], face[1], face[2]
        
        # 将面存入哈希集合（顶点排序去重）
        face_tuple = tuple(sorted((u, v, w)))
        existing_faces.add(face_tuple)
        
        # 构建双向邻接表
        for a, b in [(u, v), (v, w), (w, u)]:
            if a not in adj:
                adj[a] = set()
            adj[a].add(b)
            if b not in adj:
                adj[b] = set()
            adj[b].add(a)
    
    # 第二阶段：检测候选三角环
    cdef set processed = set()
    for u in adj:
        for v in adj[u]:
            if v <= u: continue  # 避免重复检查无向边
            for w in adj[v]:
                # 验证三角环条件
                if w in adj[u] and w not in (u, v):
                    # 顶点排序去重
                    vertices = sorted((u, v, w))
                    ring_tuple = tuple(vertices)
                    
                    # 查重并验证是否为缺失面
                    if ring_tuple not in processed:
                        processed.add(ring_tuple)
                        if ring_tuple not in existing_faces:
                            new_faces.append(vertices)
    
    # 第三阶段：返回结果
    if new_faces:
        return np.array(new_faces, dtype=np.int32)
    else:
        return np.empty((0, 3), dtype=np.int32)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_face_adjacency(np.ndarray[np.int32_t, ndim=2] cells):
    """
    计算四面体网格中三角面到邻接单元的映射关系
    参数：
        cells : 四面体单元顶点索引数组，形状(N,4)，dtype=np.int32
    返回：
        face_adj_dict : 字典 {(sorted_v1, sorted_v2, sorted_v3): [cell_idx1, cell_idx2]}
    """
    cdef:
        dict face_dict = {}
        int[:, :] cells_view = cells
        int n_cells = cells.shape[0]
        int cell_idx, i
        int v0, v1, v2, v3
        tuple face_key
        list cell_list
    
    # 遍历所有四面体单元
    for cell_idx in range(n_cells):
        # 获取当前四面体的4个顶点
        v0 = cells_view[cell_idx, 0]
        v1 = cells_view[cell_idx, 1]
        v2 = cells_view[cell_idx, 2]
        v3 = cells_view[cell_idx, 3]
        
        # 四面体的四个三角面
        # 每个面按顶点升序排序以保证唯一性
        face_keys = [
            tuple(sorted([v0, v1, v2])),
            tuple(sorted([v0, v1, v3])),
            tuple(sorted([v0, v2, v3])),
            tuple(sorted([v1, v2, v3]))
        ]
        
        # 将当前单元索引添加到每个面的邻接列表中
        for face_key in face_keys:
            if face_key in face_dict:
                face_dict[face_key].append(cell_idx)
            else:
                face_dict[face_key] = [cell_idx]
    
    # 创建固定长度的邻接字典（最大2个邻接单元）
    cdef dict face_adj_dict = {}
    cdef np.ndarray[np.int32_t, ndim=1] adj_array
    cdef int[:] array_view
    
    for face_key, cell_list in face_dict.items():
        # 创建初始化为-1的数组（长度2，因为每个面最多属于2个四面体）
        adj_array = np.full(2, -1, dtype=np.int32)
        
        # 填充有效数据
        for i in range(min(len(cell_list), 2)):
            adj_array[i] = cell_list[i]
        
        face_adj_dict[face_key] = adj_array
    
    return face_adj_dict

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def compute_face_adjacency(np.ndarray[np.int32_t, ndim=2] cells):
#     """
#     计算四面体网格中三角面到邻接单元的映射关系，同时返回四面体邻接关系表
    
#     参数：
#         cells : 四面体单元顶点索引数组，形状(N,4)，dtype=np.int32
    
#     返回：
#         face_adj_dict : 字典 {(sorted_v1, sorted_v2, sorted_v3): np.array([cell_idx1, cell_idx2], dtype=np.int32)}
#         cell_adj_array : np.ndarray 形状(N,5), 第一列是当前单元索引，后面4列是邻接单元索引，用-1填充
#     """
#     cdef:
#         dict face_dict = {}
#         int[:, :] cells_view = cells
#         int n_cells = cells.shape[0]
#         int cell_idx, i, j, neighbor_idx, existing_cell
#         int v0, v1, v2, v3
#         tuple face_key
#         list cell_list
    
#     # 初始化四面体邻接数组，形状(n_cells, 5)
#     # 第一列存储当前单元索引，后面4列存储邻接单元索引
#     cdef np.ndarray[np.int32_t, ndim=2] cell_adj_array = np.full((n_cells, 5), -1, dtype=np.int32)
#     cdef int[:, :] cell_adj_view = cell_adj_array
    
#     # 初始化第一列为单元索引
#     for cell_idx in range(n_cells):
#         cell_adj_view[cell_idx, 0] = cell_idx
    
#     # 遍历所有四面体单元
#     for cell_idx in range(n_cells):
#         # 获取当前四面体的4个顶点
#         v0 = cells_view[cell_idx, 0]
#         v1 = cells_view[cell_idx, 1]
#         v2 = cells_view[cell_idx, 2]
#         v3 = cells_view[cell_idx, 3]
        
#         # 四面体的四个三角面
#         # 每个面按顶点升序排序以保证唯一性
#         face_keys = [
#             tuple(sorted([v0, v1, v2])),
#             tuple(sorted([v0, v1, v3])),
#             tuple(sorted([v0, v2, v3])),
#             tuple(sorted([v1, v2, v3]))
#         ]
        
#         # 将当前单元索引添加到每个面的邻接列表中
#         for face_key in face_keys:
#             if face_key in face_dict:
#                 # 这个面已经被另一个四面体使用，添加邻接关系
#                 for existing_cell in face_dict[face_key]:
#                     # 将现有单元添加到当前单元的邻接列表
#                     for i in range(4):
#                         if cell_adj_view[cell_idx, i+1] == -1:
#                             cell_adj_view[cell_idx, i+1] = existing_cell
#                             break
                    
#                     # 将当前单元添加到现有单元的邻接列表
#                     for j in range(4):
#                         if cell_adj_view[existing_cell, j+1] == -1:
#                             cell_adj_view[existing_cell, j+1] = cell_idx
#                             break
                
#                 face_dict[face_key].append(cell_idx)
#             else:
#                 face_dict[face_key] = [cell_idx]
    
#     # 创建固定长度的邻接字典（最大2个邻接单元）
#     cdef dict face_adj_dict = {}
#     cdef np.ndarray[np.int32_t, ndim=1] adj_array
    
#     for face_key, cell_list in face_dict.items():
#         # 创建初始化为-1的数组（长度2，因为每个面最多属于2个四面体）
#         adj_array = np.full(2, -1, dtype=np.int32)
        
#         # 填充有效数据
#         for i in range(min(len(cell_list), 2)):
#             adj_array[i] = cell_list[i]
        
#         face_adj_dict[face_key] = adj_array
    
#     return face_adj_dict, cell_adj_array

def compute_edge_adjacency(np.ndarray[np.int32_t, ndim=2] faces):
    """
    计算边到邻接三角面的映射关系，返回固定长度邻接向量
    参数：
        faces : 三角形面顶点索引数组，形状(N,3)，dtype=np.int32
    返回：
        edge_face_dict : 字典 {(v1, v2): np.ndarray[12](面索引,-1填充)}
    """
    cdef:
        dict edge_dict = {}
        int[:, :] faces_view = faces
        int n_faces = faces.shape[0]
        int face_idx, a, b, c, u, v, min_uv, max_uv
        list face_list
        np.ndarray[np.int32_t, ndim=1] adj_array

    # 第一阶段：收集邻接面索引
    for face_idx in range(n_faces):
        a = faces_view[face_idx, 0]
        b = faces_view[face_idx, 1]
        c = faces_view[face_idx, 2]

        # 处理三条边
        for u, v in [(a, b), (b, c), (c, a)]:
            # 标准化边表示
            if u > v:
                min_uv, max_uv = v, u
            else:
                min_uv, max_uv = u, v
            
            # 更新字典
            edge_key = (min_uv, max_uv)
            if edge_key in edge_dict:
                edge_dict[edge_key].append(face_idx)
            else:
                edge_dict[edge_key] = [face_idx]

    # 第二阶段：转换为固定长度向量
    cdef dict edge_face_dict = {}
    cdef int[:] array_view
    cdef int i, arr_len

    for edge_key, face_list in edge_dict.items():
        # 创建初始化为-1的数组
        adj_array = np.full(12, -1, dtype=np.int32)
        array_view = adj_array
        
        # 填充有效数据
        arr_len = min(len(face_list), 12)
        for i in range(arr_len):
            array_view[i] = face_list[i]
        
        edge_face_dict[edge_key] = adj_array

    return edge_face_dict

# 获取边界面索引
cpdef get_boundary_faces(np.int32_t[:, ::1] cells, np.int32_t[:] idx):
    cdef dict face_counts = {}
    cdef int i, v0, v1, v2, v3  # 顶点变量改为int类型
    cdef tuple face, sorted_face

    # 遍历int32类型的单元索引
    for i in range(idx.shape[0]):
        # 提取当前四面体的4个顶点
        v0 = cells[idx[i], 0]
        v1 = cells[idx[i], 1]
        v2 = cells[idx[i], 2]
        v3 = cells[idx[i], 3]

        # 生成四面体的四个三角面
        faces = [
            (v0, v1, v2),
            (v0, v1, v3),
            (v0, v2, v3),
            (v1, v2, v3)
        ]

        # 统计每个面的出现次数
        for face in faces:
            sorted_face = tuple(sorted(face))
            face_counts[sorted_face] = face_counts.get(sorted_face, 0) + 1

    # 提取出现次数为1的边界面
    cdef list boundary = [face for face, cnt in face_counts.items() if cnt == 1]
    # 提取所有表面
    cdef list edges = [face for face, cnt in face_counts.items()]
    
    # 转换为np.int32类型的二维数组返回
    return np.array(boundary, dtype=np.int32, order='C'), np.array(edges, dtype=np.int32, order='C')


# # 测试网格：四面体 (4个面，每个面有3个邻接)
# faces = np.array([
#     [0,1,2], [0,1,3],
#     [0,2,3], [1,2,3]
# ], dtype=np.int32)
# 
# counts = count_face_neighbors(faces)
# print("邻接数量:", counts)
# # 输出: [3 3 3 3]
# 
# # 测试网格：立方体 (6个面，每个面有3个邻接)
# cube_faces = np.array([
#     [0,1,2], [0,2,3],  # 前面
#     [4,5,6], [4,6,7],  # 后面
#     [0,3,7], [0,7,4],  # 左面
#     [1,5,6], [1,6,2],  # 右面
#     [0,4,5], [0,5,1],  # 顶面
#     [2,6,7], [2,7,3]   # 底面
# ], dtype=np.int32)


from libcpp.vector cimport vector
from libcpp.utility cimport pair

ctypedef pair[int, int] Edge


# ------------------------------------------------------------
# Build edge-face adjacency
# ------------------------------------------------------------
cdef void add_edge(vector[Edge] &edges,
                   vector[int] &faces,
                   int v1, int v2, int f_id):
    cdef int a = v1
    cdef int b = v2
    if a > b:
        a, b = b, a
    edges.push_back(Edge(a, b))
    faces.push_back(f_id)


def detect_triple_features(np.ndarray[np.int32_t, ndim=2] F):
    """
    输入:
        F: (m, 3) int32 三角面片顶点索引

    输出:
        triple_edges: 共享 >=3 个面片的边 (edge index list)
        triple_points: 关联 >=3 条 triple_edge 的顶点索引
        edge_vertices: 所有边的 (v1, v2) 序列
    """

    cdef Py_ssize_t m = F.shape[0]
    cdef Py_ssize_t i

    # ============================================================
    # 1. 构建边列表（每个面贡献三条边）
    # ============================================================
    cdef vector[Edge] edges
    cdef vector[int] faces
    edges.reserve(m * 3)
    faces.reserve(m * 3)

    cdef int v0, v1, v2

    for i in range(m):
        v0 = F[i, 0]
        v1 = F[i, 1]
        v2 = F[i, 2]

        add_edge(edges, faces, v0, v1, i)
        add_edge(edges, faces, v1, v2, i)
        add_edge(edges, faces, v2, v0, i)

    # ------------------------------------------------------------
    # 2. 排序边 (按 (v1, v2))
    # ------------------------------------------------------------
    # edges[k] 与 faces[k] 始终保持同步
    cdef Py_ssize_t total = edges.size()

    # C++ sort for vector
    # 使用 edges 的 < 比较自动按 pair<int,int> 排序
    # 需要 faces 也一起重新排序
    # 为此构造 index 数组进行排序

    cdef vector[int] idx
    idx.resize(total)
    for i in range(total):
        idx[i] = i

    # 在 Python 中排序 idx，只比较 edges[idx[k]]
    sorted_idx = sorted(range(total), key=lambda x: (edges[x].first, edges[x].second))

    # 重新排列 edges, faces
    cdef vector[Edge] edges_sorted
    cdef vector[int] faces_sorted
    edges_sorted.resize(total)
    faces_sorted.resize(total)
    for i in range(total):
        edges_sorted[i] = edges[sorted_idx[i]]
        faces_sorted[i] = faces[sorted_idx[i]]

    edges = edges_sorted
    faces = faces_sorted

    # ------------------------------------------------------------
    # 3. 扫描所有相同边，统计其共享的面数
    # ------------------------------------------------------------
    cdef vector[int] triple_edges    # 存 edge index
    triple_edges.reserve(total // 10)

    cdef Py_ssize_t start = 0
    cdef Py_ssize_t j

    while start < total:
        # 当前边 (v1, v2)
        v0 = edges[start].first
        v1 = edges[start].second

        # 找到相同边的范围 [start, end)
        j = start + 1
        while j < total and edges[j].first == v0 and edges[j].second == v1:
            j += 1

        # 共享的面数
        if j - start >= 3:
            triple_edges.push_back(start)

        start = j

    # ------------------------------------------------------------
    # 4. 找三面交点：与 >=3 条 triple-edge 相连的顶点
    # ------------------------------------------------------------
    # 首先记录所有边的顶点（包括重复）
    # triple_edges 中存储的是 edges 的位置索引
    cdef int eidx, a, b
    cdef dict deg = {}     # 顶点 -> 数量

    for i in range(triple_edges.size()):
        eidx = triple_edges[i]
        a = edges[eidx].first
        b = edges[eidx].second
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1

    triple_points = [v for v, d in deg.items() if d >= 3]

    # ------------------------------------------------------------
    # 5. 返回所有边的 (v1, v2)，便于你后续构建约束
    # ------------------------------------------------------------
    edge_vertices = np.zeros((total, 2), dtype=np.int64)
    for i in range(total):
        edge_vertices[i, 0] = edges[i].first
        edge_vertices[i, 1] = edges[i].second

    return (
        np.array([triple_edges[i] for i in range(triple_edges.size())], dtype=np.int64),
        np.array(triple_points, dtype=np.int64),
        edge_vertices
    )


