# cython: language_level=3

cimport cython
from cython.parallel cimport prange
import numpy as np 

#根据预计算种子相对中心坐标进行种子的渲染 

@cython.boundscheck(False)
cdef discrete_section(int[:,:,:] vol, int[:,:] centers, 
        int[:] rads_min, int[:] rads_max, unsigned int[:,] marks, 
        int[:] periods, int[:,:] rcod_tab, unsigned int[:] cnt_tab):
    cdef:
        int i, j, k, dsc_mark, rcod
        unsigned int cnt = centers.shape[0] 
        unsigned int[:] vxl_cnt = np.zeros(centers.shape[0], dtype=np.uint32) 
        int[:] coord = np.zeros(3, dtype=np.int32)  #coord需要接受负值 声明为int
    
    #用于调试边界检测&堆栈
    #for i in prange(cnt+1, nogil=True): #所有种子
    #    rads_max[i] += 1

    for i in prange(cnt, nogil=True): #所有种子
        if rads_max[i] < 0: continue #跳过半径小于0, 被隐藏的种子
        for k in range(cnt_tab[rads_min[i]], cnt_tab[rads_max[i]+1]):  #所有体素
            dsc_mark = 1
            for j in range(3): #三个方向
                coord[j] = centers[i, j] + rcod_tab[j, k]  #查询相对坐标
                #边界检查 
                if coord[j] < 0 and coord[j] > - vol.shape[j]:   #vol.shape的类型不是int,而是py_size_t 保证进行%除法时,两者均为正
                    if periods[j]:  #周期性
                        coord[j] = (coord[j] + 10000*vol.shape[j]) % vol.shape[j]   #当vol.shape[j] = 1时是否仍然成立
                    else:  #无周期性:跳过
                        dsc_mark = 0
                elif coord[j] >= vol.shape[j]:
                    if periods[j]:  #周期性
                        coord[j] = coord[j] % vol.shape[j]
                    else:  
                        dsc_mark = 0
                elif coord[j] <= - vol.shape[j]:
                    dsc_mark = 0

            if dsc_mark:
                vxl_cnt[i] += 1  #种子内部体素计数
                vol[coord[0], coord[1], coord[2]] = marks[i]

    return vxl_cnt


@cython.boundscheck(False)
cpdef discrete_surface(int[:,:,:] vol, int[:,:] centers, 
        int[:] rads, unsigned int[:,] marks, 
        int[:] periods, int[:,:] rcod_tab, unsigned int[:] cnt_tab):
    return np.asarray(discrete_section(vol, centers, rads, rads, marks, periods, rcod_tab, cnt_tab))

@cython.boundscheck(False)
cpdef discrete_volume(int[:,:,:] vol, int[:,:] centers, 
        int[:] rads, unsigned int[:,] marks, 
        int[:] periods, int[:,:] rcod_tab, unsigned int[:] cnt_tab):
    cdef int[:] rads_min = np.zeros(rads.shape[0], dtype=np.int32)
    return np.asarray(discrete_section(vol, centers, rads_min, rads, marks, periods, rcod_tab, cnt_tab))


@cython.boundscheck(False)
cdef discrete_section_axis(int[:,:,:] vol, int[:,:] centers, 
        int[:] rads_min, int[:] rads_max, unsigned int[:] marks, unsigned int[:] axis, 
        int[:] periods, int[:,:] rcod_tab, unsigned int[:] cnt_tab):
    cdef:
        int i, j, k, dsc_mark, rcod
        unsigned int cnt = centers.shape[0] 
        unsigned int[:] vxl_cnt = np.zeros(centers.shape[0], dtype=np.uint32) 
        int[:] coord = np.zeros(3, dtype=np.int32)  #coord需要接受负值 声明为int

    for i in prange(cnt, nogil=True): #所有种子
        for k in range(cnt_tab[rads_min[i]], cnt_tab[rads_max[i]+1]):  #所有体素
            dsc_mark = 1
            for j in range(3): #三个方向
                if axis[i] == j and j != 2: #axis不为2时, 交换rcod的2号与j号索引(将j变为最后轴)
                    rcod = rcod_tab[2, k]
                elif j == 2:
                    rcod = rcod_tab[axis[i], k]  #当j取2时, 改为取axis[i]号轴
                else:
                    rcod = rcod_tab[j, k]
                coord[j] = centers[i, j] + rcod  #查询相对坐标

                #边界检查 
                if coord[j] < 0 and coord[j] > - vol.shape[j]:   #vol.shape的类型不是int,而是py_size_t 保证进行%除法时,两者均为正
                    if periods[j]:  #周期性
                        coord[j] = (coord[j] + 10000*vol.shape[j]) % vol.shape[j] 
                    else:  #无周期性:跳过
                        dsc_mark = 0
                elif coord[j] >= vol.shape[j]:
                    if periods[j]:  #周期性
                        coord[j] = coord[j] % vol.shape[j]
                    else:  
                        dsc_mark = 0
                elif coord[j] <= - vol.shape[j]:   #负值超出范围 无论是否为周期性边界 不渲染
                    dsc_mark = 0
            if dsc_mark:
                vxl_cnt[i] += 1  #种子内部体素计数
                vol[coord[0], coord[1], coord[2]] = marks[i]

    return vxl_cnt

#按轴方向的离散
@cython.boundscheck(False)
cpdef discrete_volume_axis(int[:,:,:] vol, int[:,:] centers, 
        int[:] rads, unsigned int[:] marks, unsigned int[:] axis, 
        int[:] periods, int[:,:] rcod_tab, unsigned int[:] cnt_tab):
    cdef int[:] rads_min = np.zeros(rads.shape[0], dtype=np.int32)
    return np.asarray(discrete_section_axis(vol, centers, rads_min, rads, marks, axis, periods, rcod_tab, cnt_tab))
