# cython: language_level=3

import numpy as np
cimport cython
# np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef edt_dst(int[::1] pnt1, int[::1] pnt2):
    cdef float sq = 0.0
    cdef int i 
    for i in range(pnt1.size):
        sq += (pnt1[i] - pnt2[i])**2
    return sq**(1/2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cdt_dst(int[::1] pnt1, int[::1] pnt2):
    cdef float dst = 0.0
    cdef int i
    for i in range(pnt1.size):
        dst += abs(pnt1[i] - pnt2[i])
    return dst


@cython.boundscheck(False)
@cython.wraparound(False)
cdef edt_test_dst(int[:, :, :, ::1] idx_arr, float[:, :, ::1] dst, int i, int j, int k, 
        int it, int jt, int kt): 
    cdef float cur_dist = 0.0 
    cdef int[::1] pnt1 = np.array([i, j, k], dtype=np.int32)
    cur_dist = edt_dst(pnt1, idx_arr[it, jt, kt])
    if dst[i, j, k] > cur_dist:
        dst[i, j, k] = cur_dist
        idx_arr[i, j, k, :] = idx_arr[it, jt, kt, :]
    return dst 

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cdt_test_dst(int[:, :, :, ::1] idx_arr, float[:, :, ::1] dst, int i, int j, int k, 
        int it, int jt, int kt): 
    cdef float cur_dist = 0.0 
    cdef int[::1] pnt1 = np.array([i, j, k], dtype=np.int32)
    cur_dist = cdt_dst(pnt1, idx_arr[it, jt, kt])  #当前点到目标点的距离
    if dst[i, j, k] > cur_dist:
        dst[i, j, k] = cur_dist
        idx_arr[i, j, k, :] = idx_arr[it, jt, kt, :] #赋值为近邻目标点
    return dst 

@cython.boundscheck(False)
@cython.wraparound(False)
def period_distance_transform_op(char[:, :, ::1] input_array, char[::1] periods, mode="edt"):
    cdef Py_ssize_t i, j, k, prg_i  
    input_array = np.pad(input_array, 1)
    cdef Py_ssize_t rows = input_array.shape[0]
    cdef Py_ssize_t cols = input_array.shape[1]
    cdef Py_ssize_t depths = input_array.shape[2]
    
    # Initialize distance array with max value
    cdef float[:, :, ::1] distance = np.full((rows, cols, depths), np.Inf, dtype=np.float32)
    # 最近物体点坐标
    cdef int[:, :, :, ::1] idx = np.full((rows, cols, depths, 3), 0, dtype=np.int32)

    if mode == "edt":
        test = edt_test_dst
    elif mode == "cdt":
        test = cdt_test_dst
    else:
        raise ValueError("unkown input mode")
        
    # Forward pass
    for prg_i in range(2):
        if prg_i and not np.any(periods): break
        for k in range(1, depths-1):
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if input_array[i, j, k] == 0:
                        distance[i, j, k] = 0.0
                        idx[i, j, k, 0] = i
                        idx[i, j, k, 1] = j
                        idx[i, j, k, 2] = k
                    else:  #空位
                        if periods[0] and i-1 == 0:
                            idx[i-1, j, k, 0] = idx[rows-2, j, k, 0] - rows + 2
                            idx[i-1, j, k, 1] = idx[rows-2, j, k, 1]
                            idx[i-1, j, k, 2] = idx[rows-2, j, k, 2]
                            distance[i-1, j, k] = distance[rows-2, j, k]
                        test(idx, distance, i, j, k, i-1, j, k)
                        if periods[1] and j-1 == 0:
                            idx[i, j-1, k, 0] = idx[i, cols-2, k, 0]
                            idx[i, j-1, k, 1] = idx[i, cols-2, k, 1] - cols + 2
                            idx[i, j-1, k, 2] = idx[i, cols-2, k, 2]
                            distance[i, j-1, k] = distance[i, cols-2, k]
                        test(idx, distance, i, j, k, i, j-1, k)
                        if periods[2] and k-1 == 0:
                            idx[i, j, k-1, 0] = idx[i, j, depths-2, 0]
                            idx[i, j, k-1, 1] = idx[i, j, depths-2, 1]
                            idx[i, j, k-1, 2] = idx[i, j, depths-2, 2] - depths + 2 
                            distance[i, j, k-1] = distance[i, j, depths-2]
                        test(idx, distance, i, j, k, i, j, k-1)
                        
    # Backward pass
    for prg_i in range(2):
        if prg_i and not np.any(periods): break
        for k in reversed(range(1, depths-1)):
            for i in reversed(range(1, rows-1)):
                for j in reversed(range(1, cols-1)):
                    if periods[0] and i+1 == rows-1:
                        idx[rows-1, j, k, 0] = idx[1, j, k, 0] + rows - 2
                        idx[rows-1, j, k, 1] = idx[1, j, k, 1] 
                        idx[rows-1, j, k, 2] = idx[1, j, k, 2] 
                        distance[rows-1, j, k] = distance[1, j, k]
                    test(idx, distance, i, j, k, i+1, j, k)
                    if periods[1] and j+1 == cols-1:
                        idx[i, cols-1, k, 0] = idx[i, 1, k, 0]
                        idx[i, cols-1, k, 1] = idx[i, 1, k, 1] + cols - 2
                        idx[i, cols-1, k, 2] = idx[i, 1, k, 2]
                        distance[i, cols-1, k] = distance[i, 1, k]
                    test(idx, distance, i, j, k, i, j+1, k)
                    if periods[2] and k+1 == depths-1:
                        idx[i, j, depths-1, 0] = idx[i, j, 1, 0]
                        idx[i, j, depths-1, 1] = idx[i, j, 1, 1]
                        idx[i, j, depths-1, 2] = idx[i, j, 1, 2] + depths - 2
                        distance[i, j, depths-1] = distance[i, j, 1]
                    test(idx, distance, i, j, k, i, j, k+1)


    return np.asarray(distance[1:-1,1:-1,1:-1]), np.asarray(idx[1:-1,1:-1,1:-1])



@cython.boundscheck(False)
@cython.wraparound(False)
def period_distance_transform_ddt(char[:, :, ::1] input_array, char[::1] periods):
    cdef Py_ssize_t i, j, k, prg_i
    cdef Py_ssize_t rows = input_array.shape[0]
    cdef Py_ssize_t cols = input_array.shape[1]
    cdef Py_ssize_t depths = input_array.shape[2]
    
    # Initialize distance array with max value
    cdef float[:, :, ::1] distance = np.full((rows, cols, depths), np.Inf, dtype=np.float32)
    
    # Forward pass
    for prg_i in range(2):
        if prg_i and not np.any(periods):  #非周期性边界
            break
        for k in range(depths):
            for i in range(rows):
                for j in range(cols):
                    if input_array[i, j, k] == 0:
                        distance[i, j, k] = 0
                    else:
                        if periods[0]:
                            distance[i, j, k] = min(distance[i, j, k], distance[(i - 1)%rows, j, k] + 1)
                        elif i > 0: 
                            distance[i, j, k] = min(distance[i, j, k], distance[i - 1, j, k] + 1)
                        if periods[1]:
                            distance[i, j, k] = min(distance[i, j, k], distance[i, (j - 1)%cols, k] + 1)
                        elif j > 0: 
                            distance[i, j, k] = min(distance[i, j, k], distance[i, j - 1, k] + 1)
                        if periods[2]:
                            distance[i, j, k] = min(distance[i, j, k], distance[i, j, (k - 1)%depths] + 1)
                        elif k > 0:
                            distance[i, j, k] = min(distance[i, j, k], distance[i, j, k - 1] + 1)
                        
    # Backward pass
    for prg_i in range(2):
        if prg_i and not np.any(periods):
            break
        for k in reversed(range(depths)):
            for i in reversed(range(rows)):
                for j in reversed(range(cols)):
                    if periods[0]:
                        distance[i, j, k] = min(distance[i, j, k], distance[(i + 1)%rows, j, k] + 1)
                    elif i < rows - 1: 
                        distance[i, j, k] = min(distance[i, j, k], distance[i + 1, j, k] + 1)
                    if periods[1]:
                        distance[i, j, k] = min(distance[i, j, k], distance[i, (j + 1)%cols, k] + 1)
                    elif j < cols - 1: 
                        distance[i, j, k] = min(distance[i, j, k], distance[i, j + 1, k] + 1)
                    if periods[2]:
                        distance[i, j, k] = min(distance[i, j, k], distance[i, j, (k + 1)%depths] + 1)
                    elif k < depths -1:
                        distance[i, j, k] = min(distance[i, j, k], distance[i, j, k + 1] + 1)

    return np.asarray(distance)