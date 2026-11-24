import numpy as np
from morqex.microstructure.core.cyfunc import cy_discrete as cyd


#-------------------------------------------------------------------------------------------------------------------------
# 体素化器及相关功能
#-------------------------------------------------------------------------------------------------------------------------

def shperef_mask(d):
    """获取一定直径球表面的体素掩膜"""
    grid = np.mgrid[0:d, 0:d, 0:d]
    coords = np.array(grid) + 0.5 - d/2 #体素中心坐标
    rpos_sq = np.sum(coords**2, axis=0)  #体素网格位置
    mask = np.bitwise_and(rpos_sq >= (d/2-0.5)**2, rpos_sq < (d/2+0.5)**2)
    return mask

def shperef_rcoord(d):
    """获取一定直径球表面的相对中心坐标"""
    mask = shperef_mask(d)
    rcoord = np.array(np.where(mask)) - d//2  #整数
    return rcoord

def circlef_mask(d):
    """获取一定直径圆表面的体素掩膜"""
    grid = np.mgrid[0:d, 0:d, 0:1]
    coords = np.array(grid) + np.array([0.5, 0.5, d/2])[:, None, None, None] - d/2
    rpos_sq = np.sum(coords**2, axis=0)  #体素网格位置
    mask = np.bitwise_and(rpos_sq >= (d/2-0.5)**2, rpos_sq < (d/2+0.5)**2)
    return mask

def circlef_rcoord(d):
    """获取一定直径圆表面的相对中心坐标"""
    mask = circlef_mask(d)
    rcoord = np.array(np.where(mask)) - np.array([d//2, d//2, 0])[:, None]  #整数
    return rcoord

# def squaref_mask(d):
#     """获取一定直径正八面体表表面的体素掩膜"""
#     grid = np.mgrid[0:d, 0:d, 0:d]
#     coords = np.array(grid) + 0.5 - d/2 #体素中心坐标
#     rpos_sq = np.sum(np.abs(coords), axis=0)  #体素网格位置
#     mask = np.bitwise_and(rpos_sq >= (d/2-0.5), rpos_sq < (d/2+0.5))
#     return mask

# def squaref_rcoord(d):
#     """获取一定直径正八面体表面的相对中心坐标"""
#     mask = squaref_mask(d)
#     rcoord = np.array(np.where(mask)) - d//2  #整数
#     return rcoord

def period_clip(bound_size, coords):
    return coords % bound_size

def bound_clip(bound_size, coords):
    return np.clip(coords, 0, bound_size-1)

class Voxelizer():
    """体素化器,为体素坐标种子进行体素化"""
    div_zero_value = 1e-32 
    coord_dtype = np.int32
    vol_dtype = np.int32 
    period_dtype = np.int32
    data_dtype = np.uint32  #中心,半径,标识类型 建议为uint32和cython接口中C++long一致
    rad_dtype = np.int32
    orit_dtype = np.float32
    cnt_dtype = np.uint32
    def __init__(self, max_r=20, ndim=3):
        """初始化相对坐标索引表和坐标计数表(默认体素单元边长为1)
        Args:
            max_r: 最大边长(体素数目)
            periods: 周期性边界列表
            
        Attributes:
            cnt_tab: 球内部体素数目的计数表,cnt_tab[r]表示半径r的球内部的体素数目
            rcod_tab: 形状为(3,n_cnt),球内部体素相对中心坐标表
        """
        #体素化周期性设置
        #预先计算相对半径坐标索引
        self.set_rcod_tab(max_r, ndim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(rmax:{self.max_r} ndim:{self.ndim})"
    
    def __repr__(self):
        return self.__str__()

    def set_ndim(self, ndim):
        self.set_rcod_tab(self.max_r, ndim)

    @property
    def max_r(self):
        return self.cnt_tab.size - 2
        
    def set_rcod_tab(self, max_r, ndim=3):
        """设置当前半径相对坐标表"""
        self.rcod_tab = np.zeros(shape=(3, int(4/3*np.pi*(max_r+2)**3)), dtype=self.__class__.coord_dtype)  #相对坐标,最多为(球形半径+2)区域体积,可为负
        # self.phi_tab = np.zeros(shape=(2, (max_r*2+1)**3), dtype=self.__class__.orit_dtype) #预计算方向信息
        self.cnt_tab = np.zeros(max_r+2, dtype=self.__class__.cnt_dtype)  #0号指向开始,1号指向0半径的末尾索引
        for r in range(max_r+1):
            if ndim == 3:
                rcod_d = shperef_rcoord(2*r+1)
            elif ndim == 2:
                rcod_d = circlef_rcoord(2*r+1)
            else:
                raise ValueError(f"set voxelizer rcod failed: invalid ndim: {ndim}")
            self.cnt_tab[r+1] = self.cnt_tab[r] + rcod_d.shape[1]
            #相对坐标更新
            self.rcod_tab[:, self.cnt_tab[r]:self.cnt_tab[r+1]] = rcod_d
            #相对方向更新
            # self.phi_tab[0, self.cnt_tab[r]:self.cnt_tab[r+1]] = np.arctan(rcod_d[1,:]/(rcod_d[0,:]+self.__class__.div_zero_value))
            # self.phi_tab[1, self.cnt_tab[r]:self.cnt_tab[r+1]] = np.arctan(np.sqrt(rcod_d[1,:]**2+rcod_d[0,:]**2)/(rcod_d[2,:]+self.__class__.div_zero_value))

        self.ndim = ndim

        
    # def set_rcod_cdt_tab(self, max_r):
    #     """设置当前棋盘网格相对坐标"""
    #     self.rcod_cdt_tab = np.zeros(shape=(3, int((2*max_r+4)**3)), dtype=self.__class__.coord_dtype)  #相对坐标,最多为(球形半径+2)区域体积,可为负
    #     self.cnt_cdt_tab = np.zeros(max_r+2, dtype=self.__class__.cnt_dtype)
    #     self.phi_cdt_tab = np.zeros(shape=(2, (max_r*2+1)**3), dtype=self.__class__.orit_dtype) #预计算方向信息
    #     for r in range(max_r+1):
    #         rcod_cdt_d = squaref_rcoord(2*r+1)
    #         self.cnt_cdt_tab[r+1] = self.cnt_cdt_tab[r] + rcod_cdt_d.shape[1]
    #         #相对坐标更新
    #         self.rcod_cdt_tab[:, self.cnt_cdt_tab[r]:self.cnt_cdt_tab[r+1]] = rcod_cdt_d
    #         self.rcod_cdt_tab[:, self.cnt_cdt_tab[r]:self.cnt_cdt_tab[r+1]] = rcod_cdt_d
    #         #相对方向更新
    #         self.phi_cdt_tab[0, self.cnt_cdt_tab[r]:self.cnt_cdt_tab[r+1]] = np.arctan(rcod_cdt_d[1,:]/(rcod_cdt_d[0,:]+self.__class__.div_zero_value))
    #         self.phi_cdt_tab[1, self.cnt_cdt_tab[r]:self.cnt_cdt_tab[r+1]] = np.arctan(np.sqrt(rcod_cdt_d[1,:]**2+rcod_cdt_d[0,:]**2)/(rcod_cdt_d[2,:]+self.__class__.div_zero_value))

    def check_rads(self, rads, cnt):
        """检查并处理即将渲染的半径"""
        if isinstance(rads, (int, np.integer)):
            rads = np.repeat(np.array([rads], dtype=self.__class__.rad_dtype), cnt)
        elif not isinstance(rads, np.ndarray):
            rads = np.array(rads, dtype=self.__class__.rad_dtype)
        if np.max(rads) > self.max_r:  #超出渲染范围
            print(f"render seeds warning: rads {np.max(rads)} is out of bound {self.max_r}, auto clip")
            rads = np.clip(rads, None, self.max_r-1)
        return rads 

    def check_centers(self, centers):
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers, dtype=self.__class__.data_dtype)
        return centers 

    def check_marks(self, marks, cnt):
        if isinstance(marks, (int, np.integer, float, np.floating)):  #支持整数输入和numpy计算结果np.int32输入
            marks = np.repeat(np.array([marks], dtype=self.__class__.data_dtype), cnt)
        else:
            marks = np.array(marks,  dtype=self.__class__.data_dtype)
        return marks

    def discrete_check(self, centers, rads, marks):
        """统一体素化函数,按照self.bound_funcs进行周期性边界处理
        """
        cnt = centers.shape[0]
        if cnt > 0:
            centers = self.check_centers(centers)
            rads = self.check_rads(rads, cnt)
            marks = self.check_marks(marks, cnt)
        else:
            print("discrete check failed: seeds is empty")
        return centers, rads, marks 

    def discrete_surface(self, vol, centers, rads, marks, periods):
        """球表面离散,输入均为体素坐标"""
        centers, rads, marks = self.discrete_check(centers, rads, marks)
        return cyd.discrete_surface(vol, centers, rads, marks, 
            np.array(periods, dtype=self.__class__.period_dtype), self.rcod_tab, self.cnt_tab)
        
    def discrete_volume(self, vol, centers, rads, marks, periods):
        """球体积离散,输入均为体素坐标
        Example:
            v = Voxelizer(30)
            vol = np.zeros([100,100,100])  #待渲染体积
            v.discrete_volume(vol,[[50,50,50],[80,80,80],[47,47,49]],[6,29,8],1)
            plt.imshow(vol[78,:,:])
        """
        # vol = vol.astype(self.vol_dtype)
        centers, rads, marks = self.discrete_check(centers, rads, marks)
        return cyd.discrete_volume(vol, centers, rads, marks, 
            np.array(periods, dtype=self.__class__.period_dtype), self.rcod_tab, self.cnt_tab)

    # def discrete_volume_check(self, vol, centers, rads, marks, periods):
    #     """球体积离散,输入均为体素坐标,并且提供渲染检查标记(当前种子中心是否被占用),返回检查结果
    #     Example:
    #         vol = np.zeros((10,10,10), dtype=np.int32)
    #         centers = np.array([[1,1,1],[5,5,5]], dtype=np.uint32)
    #         rads = np.array([2, 4], dtype=np.uint32)
    #         marks = np.array([1, 2], dtype=np.uint32)
    #         check_mark = np.zeros(2, dtype=np.uint32)
    #         periods = np.array([1,1,1], dtype=np.int32)
    #         volizer.discrete_volume_check(vol, centers, rads, marks)
    #     """
    #     centers, rads, marks = self.discrete_check(centers, rads, marks)
    #     seeds_mark = np.zeros(centers.shape[0], dtype=np.uint32)  #指示种子之间的重叠情况
    #     cyd.discrete_volume_check_center(vol, centers, rads, marks, seeds_mark,
    #         np.array(periods, dtype=self.__class__.period_dtype), self.rcod_tab, self.cnt_tab)
    #     return seeds_mark
        
    # def discrete_dist(self, vol, centers, rads_min, rads_max, periods):
    #     """中心距离离散
    #     Args:
    #         vol (np.ndarray np.uint32): 将被渲染的体素空间对象,初始为0,表示该处距离未定义,而非距离为0
    #         centers (np.ndarray np.uint32 shape=(-1,3)): 整数化的种子体素中心列表
    #         rads (np.array np.uint32): 种子的基础半径,置为0时计算中心场
    #         rads_min (np.array np.uint32): 整数化的种子最小相关半径列表,小于该值的位置不渲染半径
    #         rads_max (np.array np.uint32): 种子最大相关半径大小列表,大于该值的位置不渲染半径
    #         dist_add (bool): 是否相加距离
    #     Example:
    #         vol = np.zeros(elec_dmn.vol_shape, dtype=np.int32)
    #         uint_seeds = elec_dmn.uint_seeds(locd_seeds.phase_split_seeds(acm))
    #         volizer.discrete_dist(vol, uint_seeds.pos, 0, uint_seeds.r+3)
    #     """
    #     #前提检查
    #     cnt = centers.shape[0]
    #     centers = self.check_centers(centers)
        
    #     #渲染前最大半径边界检查
    #     rads_min = self.check_rads(rads_min, cnt)
    #     rads_max = self.check_rads(rads_max, cnt)
    #     rads_max = np.max([rads_min, rads_max], axis=0) #max大于等于min
    #     #渲染
    #     cyd.discrete_center_dist(vol, centers, rads_min, rads_max, 
    #         np.array(periods, dtype=self.__class__.period_dtype), self.rcod_tab, self.cnt_tab)

    def discrete_check_axis(self, centers, rads, axis, marks):
        """统一体素化函数,按照self.bound_funcs进行周期性边界处理
        """
        cnt = centers.shape[0]
        if cnt > 0:
            centers = self.check_centers(centers)
            rads = self.check_rads(rads, cnt)
            marks = self.check_marks(marks, cnt)
            axis = self.check_marks(axis, cnt)
        else:
            print("discrete check failed: seeds is empty")
        return centers, rads, marks, axis

    def discrete_volume_axis(self, vol, centers, rads, marks, axis, periods):
        # vol = vol.astype(self.vol_dtype)
        centers, rads, marks, axis = self.discrete_check_axis(centers, rads, axis, marks)
        return cyd.discrete_volume_axis(vol, centers, rads, marks, axis, 
            np.array(periods, dtype=self.__class__.period_dtype), self.rcod_tab, self.cnt_tab)