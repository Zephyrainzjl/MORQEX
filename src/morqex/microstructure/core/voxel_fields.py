from  __future__ import annotations
from typing import Any, Callable

import warnings 
import numpy as np 

from scipy.interpolate import interpn

from skimage.morphology import dilation, erosion 
from skimage.transform import rescale
from skimage.filters import gaussian, median 
from skimage.filters.rank import modal

np.set_printoptions(legacy="1.25")

#-------------------------------------------------------------------------------------#
# 距离变换
#-------------------------------------------------------------------------------------#


# from cyfunc.cy_morphology import period_distance_transform_op

# def op_distance_transform(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True, mode="edt"):
#     dsts, idxs = period_distance_transform_op(np.array(obj_vol, dtype=np.bool_), np.array(periods, dtype=np.bool_), mode)
#     if return_indices and return_distances:
#         return dsts, idxs
#     elif return_indices:
#         return idxs
#     elif return_distances:
#         return dsts
#     else: 
#         return 

# def edt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True):
#     return op_distance_transform(obj_vol, periods, return_indices, return_distances, "edt")

# def cdt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True):
#     return op_distance_transform(obj_vol, periods, return_indices, return_distances, "cdt")

from scipy.ndimage import distance_transform_cdt
from scipy.ndimage import distance_transform_edt
from morqex.microstructure.core.edt import edt as fastdt

def dist_trans(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True, measure_func=distance_transform_edt):
    npad = np.array(obj_vol.shape)/2 * periods
    dsts, idxs = measure_func(np.pad(obj_vol, np.array([npad, npad], dtype=np.int32).T, "wrap"), return_indices=True, return_distances=True)

    dst_cut_key = tuple([slice(int(pad), int(pad*3)) if period else slice(None) for pad, period in zip(npad, periods)])
    idx_cut_key = tuple([slice(None)] + [slice(int(pad), int(pad*3)) if period else slice(None) for pad, period in zip(npad, periods)])
    if return_indices and return_distances:
        return dsts[dst_cut_key], idxs[idx_cut_key]
    elif return_indices:
        return idxs[idx_cut_key]
    elif return_distances:
        return dsts[dst_cut_key]
    else: 
        return 

n_para = 6
def fedt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True, parallel=n_para):
    if return_indices==False and return_distances==True and np.prod(obj_vol.shape) > 1e5:  #10万规模以下用scipy的edt
        npad = np.array(obj_vol.shape)/2 * periods
        dsts = fastdt.edt(np.pad(obj_vol, np.array([npad, npad], dtype=np.int32).T, "wrap"), parallel=parallel)
        dst_cut_key = tuple([slice(int(pad), int(pad*3)) if period else slice(None) for pad, period in zip(npad, periods)])
        return dsts[dst_cut_key]
    else: 
        return dist_trans(obj_vol, periods=periods, return_indices=return_indices, return_distances=return_distances, measure_func=distance_transform_edt)

def invfedt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True, parallel=n_para):
    return fedt(1 - obj_vol, periods=periods, return_indices=return_indices, return_distances=return_distances, parallel=parallel)

def edt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True):
    return dist_trans(obj_vol, periods=periods, return_indices=return_indices, return_distances=return_distances, measure_func=distance_transform_edt)

# def invedt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True):
#     return dist_trans(1 - obj_vol, periods=periods, return_indices=return_indices, return_distances=return_distances, measure_func=distance_transform_edt)

def cdt(obj_vol, periods=[False, False, False], return_indices=False, return_distances=True):
    return dist_trans(obj_vol, periods=periods, return_indices=return_indices, return_distances=return_distances, measure_func=distance_transform_cdt)

def ndt(obj_vol, *args, **kwargs): 
    obj_vol = ObjVol(obj_vol)[:]
    return np.max(obj_vol.shape) * obj_vol.astype(np.float32)  #体素内部距离变换

def invdt(obj_vol, *args, **kwargs): 
    obj_vol = ObjVol(obj_vol)[:]
    return np.max(obj_vol.shape) * (1 - obj_vol).astype(np.float32)


#-------------------------------------------------------------------------------------#
# 辅助类
#-------------------------------------------------------------------------------------#

def hex_id(obj):
    return hex(id(obj))[-DataVol.id_len:]



#-------------------------------------------------------------------------------------#
# 数据数组
#-------------------------------------------------------------------------------------#

def auto_vol(func):
    def auto_vol_func(self: DataVol, *args, **kwargs):
        if len(args) and isinstance(args[0], DataVol): 
            return func(self, args[0].vol, *args[1:], **kwargs)
        else: 
            return func(self, *args, **kwargs)
    return auto_vol_func



class DataVol():
    """数据数组基类"""
    id_len = 4
    zero_th = 1e-32 
    sum_zero_th = zero_th * 1e6
    vol_dtype = np.float64

    @auto_vol
    def __init__(self, vol: np.ndarray=np.array([])) -> None:
        if vol.dtype == self.vol_dtype:
            self.vol = vol 
        else:
            self.vol = vol.astype(self.vol_dtype) 

    @classmethod
    def ones(cls, shape):
        return cls(np.ones(shape, dtype=cls.vol_dtype))

    @classmethod
    def zeros(cls, shape):
        return cls(np.zeros(shape, dtype=cls.vol_dtype))

    def copy(self):
        return self.__class__(self.vol.copy())

    def __getstate__(self):
        return {"vol": self.vol}
    
    def __setstate__(self, state):
        """更新 其中vol更新为numpy向量"""
        self.__dict__.update(state)
        self.vol = np.array(state["vol"], dtype=self.vol_dtype)

    @auto_vol
    def equals(self, value) -> np.ndarray:
        return self.__class__(self.vol.__eq__(value))

    @auto_vol
    def nequals(self, value) -> np.ndarray:
        return self.__class__(self.vol.__ne__(value))

    @auto_vol
    def __setitem__(self, key, value) -> np.ndarray:
        return self.vol.__setitem__(key, value)

    @auto_vol
    def __getitem__(self, key) -> np.ndarray:
        return self.vol.__getitem__(key)
    
    @auto_vol
    def __add__(self, value) -> DataVol:
        return self.__class__(self.vol.__add__(value)) 
    
    @auto_vol
    def __iadd__(self, value) -> DataVol:
        self.vol.__iadd__(value)
        return self 
    
    @auto_vol
    def __sub__(self, value) -> DataVol:
        return self.__class__(self.vol.__sub__(value)) 
    
    @auto_vol
    def __isub__(self, value) -> DataVol:
        self.vol.__isub__(value)
        return self 
    
        
    @auto_vol
    def __mul__(self, value) -> DataVol:
        return self.__class__(self.vol.__mul__(value)) 

    @auto_vol
    def __imul__(self, value) -> DataVol:
        self.vol.__imul__(value)
        return self 
    
    @auto_vol
    def __truediv__(self, value) -> DataVol:
        return self.__class__(self.vol.__truediv__(value)) 

    @auto_vol
    def __itruediv__(self, value) -> DataVol:
        self.vol.__itruediv__(value)
        return self 

    @property
    def ndim(self) -> int:
        return self.vol.ndim 
        
    def query(self, loc: np.ndarray):  
        """相对坐标->体素坐标 超出索引范围,值置为-1
        loc = np.array([[0.1,0.1,0.1],
                [0.8,0.8,0.8]])
        dv = DataVol(np.array(
                    [[[0,1],
                    [2,3]],
                    [[4,5],
                    [6,7]]]))
        dv.query(loc)
        """
        qry_loc = loc % (np.array(self.shape)[:3]/self.vxl_len)  #避免超出范围
        int_loc = (qry_loc * self.vxl_len).astype(np.int32)
        item_arr = self.__getitem__(tuple(int_loc.T))
        # item_arr[np.bitwise_or(np.any(loc < -1.0, axis=1), np.any(loc > 1.0, axis=1))] = -np.inf
        return item_arr
    
    def itp_query(self, loc: np.ndarray):
        """插值查询 坐标0.5表示体素中心"""
        loc = loc * self.vxl_len + 4 #坐标弥补pad的一个宽度
        itp_vol = np.pad(self[:], 4, "symmetric")  
        return interpn([np.arange(l) + 0.5 for l in itp_vol.shape], itp_vol, loc)  #网格插值坐标中心0.5 1.5

    def count_vol(self, delta=1.0):
        int_data = (self.vol / delta).astype(np.int32)
        _, inv, cnt = np.unique(int_data, return_counts=True, return_inverse=True)
        return DataVol(cnt[inv].reshape(self.vol.shape)) 

    @auto_vol
    def concatenate(self, vol: DataVol, axis: int):
        return self.__class__(np.concatenate([self.vol, vol], axis=axis)) 
        
    @property
    def max(self):
        if self.vol.size:
            return np.max(self.vol)
        else:
            raise ValueError(f"vol of {self.__str__()} is empty, check the region")
    
    @property
    def min(self):
        if self.vol.size:
            return np.min(self.vol)
        else:
            raise ValueError(f"vol of {self.__str__()} is empty, check the region")

    @property
    def sum(self):
        return np.sum(self.vol)

    @property 
    def is_uni(self) -> bool:
        return (self.max - self.min) < self.zero_th
    
    @property
    def is_zero(self) -> bool:
        return self.max < self.zero_th
    
    @property
    def is_pos(self) -> bool:
        return self.min > -self.zero_th
    
    @property
    def is_norm(self):
        return abs(self.sum - 1) < self.sum_zero_th
    
    @property
    def shape(self) -> tuple:
        return self.vol.shape 
    
    @property
    def vxl_len(self) -> int:
        return np.max(self.shape)
    
    @property
    def vol_len(self) -> float:
        return np.max(self.vol.shape) 
    
    @property
    def cls_id_str(self) -> str:
        return f"{self.cls_str}-{self.id_str}"
    
    @property
    def cls_str(self) -> str:
        return f"{self.__class__.__name__}"
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(vol:{hex_id(self.vol)})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vol:{hex_id(self.vol)})"

    @property
    def id_str(self) -> str:
        return f"{hex_id(self)}"

    def reshape(self, *args):
        return self.vol.reshape(*args)
    
    def rescale(self, scale) -> DataVol:
        """保持自身类型 缩放指定倍数"""
        return self.__class__(rescale(self.vol, scale)) 

    def resize(self, shape) -> DataVol:
        """插值重新采样进行放缩"""
        if np.all(np.array(shape) == self.shape):
            return self.__class__(self.vol.copy())
        # if not np.all(shape/np.max(shape) == np.array(self.shape)/self.vxl_len): 
        #     warnings.warn(f"vol resize warning: target shape {shape} not consist with {self.vol.shape}, auto cut vol")
        # loc_pnts = move_axes(np.mgrid[tuple([slice(None, i_len) for i_len in shape])]) #+ 0.5  #坐标中心0.5
        # #TODO 确认一致性
        # #BUG 报错:One of the requested xi is out of bounds in dimension 0
        # return self.__class__(self.itp_query(loc_pnts/np.max(shape))) 
        loc_pnts = (move_axes(np.mgrid[tuple([slice(None, i_len) for i_len in shape])]) + 0.5)* self.shape / shape
        #BUG resize后产生0. 出现在界面的边界, 因为是对每个相单独进行缩放, 不一定填充所有位置. 
        return self.__class__(self.itp_query(loc_pnts/ np.max(self.shape)))

    def trans_resize(self, shape) -> DataVol:
        from skimage.transform import resize 
        return self.__class__(resize(DataVol(self.vol)[:], shape, 1))

    def swapaxes(self, axis0: int, axis1: int):
        self.vol = np.swapaxes(self.vol, axis0, axis1)
        return self
               
    def smooth(self, sigma=1.2):
        return DataVol(gaussian(self.vol, sigma=sigma)) 

    def median_smooth(self): 
        """中值平滑 保持自身类型"""
        return self.__class__(median(self.vol))

    def modal_smooth(self, depth=1): 
        """众数平滑 uint8类型"""
        if depth <= 0: return self.__class__(self)
        npad = int(depth)  #网格剖分稍有改善
        vol = np.pad(self.vol, npad, "symmetric")
        pre_vol = modal(vol.astype(np.uint8), np.ones([3,3,3]))[npad:-npad, npad:-npad, npad:-npad]
        # pre_vol = modal(self.vol.astype(np.uint8), np.ones([3,3,3]))
        for i in range(depth-1):
            vol = modal(pre_vol, np.ones([3,3,3]))
            if np.all(vol[:] == pre_vol[:]): break
            pre_vol = vol 
        return self.__class__(pre_vol)

    def open_smooth(self, depth=1): 
        """开运算+众数平滑 uint8类型"""
        if depth <= 0: return self.__class__(self)
        vol_open = self.open(depth)
        vol_modal = self.modal_smooth(depth)
        mask = self.vol[:] != vol_open[:]
        vol_open[mask] = vol_modal[mask]
        return vol_open

    def residual(self, sigma=1.0):
        # return DataVol(self.vol != self.modal_smooth().vol)
        # return DataVol((np.abs(DataVol(self.vol)[:] - self.smooth(sigma=0.5).vol)*2))  #结果更稳定, 值稍微偏大
        # return DataVol(np.abs(DataVol(self.vol)[:] - self.smooth(sigma=sigma).vol))
        return DataVol(np.abs(DataVol(self.vol)[:] - self.median_smooth().vol))      #结果更尖锐, 可能在一些结构处丢失信息  成功率更高
    
    def scaled_residual(self, scale, sigma=1.0) -> DataVol:
        """层级残差"""
        res = self.rescale(scale).residual(sigma=sigma)
        return res.rescale(np.array(self.shape)/np.array(res.shape))
    
    def scaled_smooth(self, scale, sigma=1.0) -> DataVol:
        """层级平滑"""
        smt = self.rescale(scale).smooth(sigma)
        return smt.rescale(np.array(self.shape)/np.array(smt.shape))

    def scaled_median(self, scale) -> DataVol:
        med = self.rescale(scale).median_smooth()
        return med.rescale(np.array(self.shape)/np.array(med.shape))

    @property 
    def n_level(self):
        return np.log2(self.vxl_len)

    def multi_residual(self, sigma=1.0, n_level=5) -> list:
        """多级残差 返回有相同尺度的列表 更大层级应视作均匀背景"""
        #下采样后计算残差       
        return [self.scaled_residual(0.5**i, sigma=sigma) for i in range(min(self.n_level, n_level))]
    
    def multi_smooth(self, sigma=1.0, n_level=5) -> list:
        """多级残差 返回有相同尺度的列表 更大层级应视作均匀背景"""
        #下采样后计算残差       
        return [self.scaled_smooth(0.5**i, sigma) for i in range(min(self.n_level, n_level))]
    
    # def cut_stat(self, axis):
    #     """截面统计值 用于处理加载的体素网格数据"""
    #     avg = []
    #     std = []
    #     for cut in np.swapaxes(self.vol, axis, 0):
    #         cut[np.bitwise_or(cut == np.inf, cut == -np.inf)] = np.nan 
    #         #TODO 使用nanmean nanstd
    #         avg.append(np.average(cut[~np.isnan(cut)]))  
    #         std.append(np.std(cut[~np.isnan(cut)])) 
    #     return np.array(avg), np.array(std)

    @classmethod 
    def from_img(cls, fname, cuts=(slice(None), slice(None), slice(None)), process_func=lambda img: img) -> DataVol:
        from skimage import io
        from skimage.color import rgb2gray
        import os 

        if os.path.isfile(fname):
            if fname[-4:] in (".npy", ".txt"):
                img = np.load(fname)
            elif fname.split(".")[-1] in ("tif", "png", "jpg", "jpeg", "tiff", "bmp"): 
                img = io.imread(fname)
            else: 
                warnings.warn(f"unexpected file type {fname}, skipped")
                return cls(np.array([]))
            if len(img.shape) == 3 and (img.shape[2] in (3, 4)):  #多通道单张图片
                img = rgb2gray(img[:, :, :3])
            
            #图片处理函数
            img = process_func(img)

            if img.ndim == 2:  #单张图片
                return cls(img[cuts[0], cuts[1], None])
            elif img.ndim == 3: #三维图片
                return cls(img)
            else: 
                raise ValueError(f"vol from img failed: unsupported ndim {img.ndim}")

        elif os.path.isdir(fname):  #文件夹图片集
            names = os.listdir(fname)
            # names = [name for name in names if name[-4:] in (".jpg", ".png", ".bmp", ".tif", "jpeg", "tiff")]
            z1, z2, zs = cuts[2].start or 0, cuts[2].stop or len(names), cuts[2].step or 1
            img = [cls.from_img(f"{fname}/{names[z1+i*zs]}", cuts[:2], process_func=process_func)[:] for i in range((z2-z1)//zs)]  #处理每张图片
            img = [cut for cut in img if np.any(cut[:])]
            return cls(np.concatenate(img, 2))

        else:
            raise FileNotFoundError(f"{cls.__name__} from img failed: file {fname} not found")
    
    def threshold(self, ths) -> PhaseVol:
        """阈值化与处理"""
        if not isinstance(ths, (tuple, list, np.ndarray)): ths = [ths]

        vol = PhaseVol.ones(self.shape)
        for i, th in enumerate(ths):  #赋予属性值
            vol[np.where(self[:] >= th)] = i + 2  #越亮值越大
        return vol



class ObjVol(DataVol):
    """单相数组 0视作没有物体"""
    vol_dtype = np.bool_ 

    @auto_vol
    def __add__(self, value) -> ObjVol:
        return ObjVol(self.vol.__add__(value)) 
    
    @auto_vol
    def __iadd__(self, value) -> ObjVol:
        self.vol.__iadd__(value)
        return self 
    
    def a_shape(self, n_voxel=1.0, measure=fedt, pad=True) -> ObjVol:
        """获取rolling_ball算法计算的外边界"""
        if n_voxel <= 0: return self 
        if self.is_uni: return self  #空物体/均匀物体不计算

        n_pad = int(n_voxel - 0.5) if pad else 0 #减去0.5,保证2维问题是约束的 

        obj_vol = np.pad(self.vol, pad_width=n_pad, mode='constant', constant_values=False) 
        img_dist = measure(obj_vol==0)
        ball_outer = img_dist > n_voxel
        ball_dist = measure(ball_outer==0)
        a_shape_pad = ball_dist > n_voxel

        vol_key = tuple([slice(n_pad, i_len-n_pad) for i_len in a_shape_pad.shape])
        a_shape_vol = a_shape_pad[vol_key]
        return ObjVol(a_shape_vol)

    def inv_a_shape(self, n_voxel=1.0, measure=fedt, pad=True) -> ObjVol:
        #NOTE 不可在外部用inv计算, 将会导致属性混乱
        if n_voxel <= 0: return self 
        if self.is_uni: return self  #空物体/均匀物体不计算

        n_pad = int(n_voxel - 0.5) if pad else 0 #减去0.5,保证2维问题是约束的 

        vol = 1 - self.vol.copy()
        obj_vol = np.pad(vol, pad_width=n_pad, mode='constant', constant_values=False) 
        img_dist = measure(obj_vol==0)
        ball_outer = img_dist > n_voxel
        ball_dist = measure(ball_outer==0)
        a_shape_pad = ball_dist > n_voxel

        vol_key = tuple([slice(n_pad, i_len-n_pad) for i_len in a_shape_pad.shape])
        a_shape_vol = a_shape_pad[vol_key]
        return ObjVol(1 - a_shape_vol)

    @property
    def inv(self):
        """属性取反 不维持之前的属性"""
        idxs = np.arange(np.max(self.attrs) + 1)
        return self.attr_map(idxs[::-1])
        # return self.__class__(self.vol == 0)

    @property
    def attrs(self):
        """属性表 0表示空位"""
        return np.sort(np.unique(self.vol).astype(np.int32)) 
    
    def attr_map(self, map_arr: np.ndarray):
        return self.__class__(np.array(map_arr)[self[:]])

    @property 
    def is_uni(self) -> bool:
        return bool(np.all(self.vol) or np.all(1 - self.vol))
    
    @property
    def is_zero(self) -> bool:
        return bool(np.any(self.vol))
    
    def open(self, depth=1.0, measure=cdt):
        """开运算 先腐蚀后膨胀 移除桥"""
        if depth <= 0: return self.__class__(self)
        return self.erosion(depth, measure=measure).dilation(depth, measure=measure)
     
    def close(self, depth=1.0, measure=cdt):
        """闭运算 先膨胀后腐蚀 移除孔"""
        if depth <= 0: return self.__class__(self)
        return self.dilation(depth, measure=measure).erosion(depth, measure=measure)
    
    def erosion(self, depth=1.0, measure=cdt, connect=1):
        if depth <= 0: return self.__class__(self)
        vol = self.vol.copy()
        if measure in (edt, fedt):
            return ObjVol((measure(vol[:]>0) - depth) > 0) 
        elif measure == cdt:
            for i in range(int(depth + 0.5)):
                if connect == 1: vol = erosion(vol)
                elif connect == 2: vol = erosion(vol, footprint=np.ones([3,3,3]))
                else: raise ValueError(f"ersion failed with connectivity {connect}")
            return ObjVol(vol)
        else: 
            raise ValueError(f"{self.__class__.__name__} erosion failed: unknown measure")
    
    def dilation(self, depth=1.0, measure=cdt, connect=1):
        if depth <= 0: return self.__class__(self)
        vol = self.vol.copy()
        if measure in (edt, fedt):
            return ObjVol((measure((vol[:]<=0)) - depth) > 0) 
        elif measure == cdt:
            for i in range(int(depth + 0.5)):
                if connect == 1: vol = dilation(vol)
                elif connect == 2: vol = dilation(vol, footprint=np.ones([3,3,3]))
                else: raise ValueError(f"ersion failed with connectivity {connect}")
            return ObjVol(vol)
        else: 
            raise ValueError(f"{self.__class__.__name__} dilation failed: unknown measure")
    
    def rescale(self, scale: float) -> ObjVol:
        """均匀插值放缩"""
        return ObjVol((rescale(self.vol[:], scale) - 0.5) > 0) 
    
    def resize(self, shape) -> DataVol:
        """插值重新采样进行放缩"""
        if np.all(np.array(shape) == self.shape):
            return self.__class__(self.vol.copy())
        loc_pnts = (move_axes(np.mgrid[tuple([slice(None, i_len) for i_len in shape])]) + 0.5)* self.shape / shape
        return self.__class__((self.itp_query(loc_pnts/ np.max(self.shape)) - 0.5) > 0)  #-0.5强度中心化

    @property
    def vol_frac(self):
        volume = np.prod(self.shape)
        return np.count_nonzero(self.vol[:])/volume
    
    @property
    def surface(self):
        """外表面提取"""
        return ObjVol(np.bitwise_and(1 - self.erosion()[:], self[:])) 

    @property
    def out_surface(self):
        """外表面提取"""
        return ObjVol(np.bitwise_and(self.dilation()[:], 1 - self[:])) 

    @property
    def spec_surf_area(self, voxel_len=1.0):
        """比表面积 voxel_len给出网格边长"""
        return np.count_nonzero(self.surface[:]) / np.count_nonzero(self[:]) / voxel_len

    @property
    def surf_area(self, voxel_len=1.0):
        """比表面积 voxel_len给出网格边长"""
        return np.count_nonzero(self.surface[:]) * voxel_len**2 

    def fractal_ndim(self):
        """分型维数计算
        Exapmles:
            #分型维数(平滑)
            ndim, blocks, scales = ObjVol(vols[0][:] == 1).open_smooth(2).fractal_ndim()
            plt.plot(np.log(scales), np.log(blocks), marker="o")
            print(ndim)
        """
        from scipy import stats

        #1/2缩放统计体素
        n_scale = int(np.log2(np.max(self.shape))) - 3  #减去本身-1, 保证精度-2
        if n_scale < 1: 
            raise RuntimeError(f"fractal dim calculation failed: the scale {n_scale} < 1")
        fac = 2
        vol = self
        blocks = [np.sum(vol[:])]
        scales = [1]
        for i in range(n_scale):
            shape = np.array(np.array(vol.shape)/fac + 0.5, dtype=np.int32)  #+0.5保证插值成功
            vol = DataVol(vol.resize(shape)[:]>0)
            blocks.append(np.sum(vol[:]))
            scales.append(fac**(i+1))
        #插值
        return stats.linregress(np.log(scales), np.log(blocks)).slope, blocks, scales

    def tortuosity(self, start_vol: ObjVol, weight_vol=DataVol(np.array([])), mode="avg", max_prg=1000):
        """计算带有局部距离权重的空间曲折度分布,输入物体数组,指定传播轴和起始位置. 
        使用8邻接连通性,在最大传播次数内未达到的点将标记未传播点距离. 
        Args: 
            mode: "min" or "avg", 指定计算模式,min表示最近距离传播 avg表示调和平均距离传播,距离越近,传播阻力越小,传播越多
        Examples:
        vol = PhaseVol.zeros([5, 5, 1])
        vol[3, 0] = 1
        vol[3, 1] = 1
        vol[2, 1] = 1
        vol[1, 0] = 1
        vol[1, 1] = 1
        vol[2, 4] = 1

        start = PhaseVol.zeros([5, 5, 1])
        start[0, :, :] = 1
        vol[:, :, 0], vol.tortuosity(start)[:, :, 0]
        """
        vol = self[:].astype(np.bool_)
        weight_vol = np.array(weight_vol[:], dtype=np.float32)
        start_vol = np.array(start_vol[:], dtype=np.bool_)

        #起始点标记
        start_vol = np.bitwise_and(start_vol, ~vol)  #起始点不允许占用物体体素位置
        
        #当前传播方向的直线坐标 使用对start_vol的距离变换实现
        loc_vol: np.ndarray = fedt(start_vol==0)
        # n_prg = int(np.max(loc_vol) * max_prg)  #最大传播次数为最远距离的一定倍数

        #局部窗口
        wnd_shape = np.full(vol.ndim, 3) #连通窗口形状
        slide_shape = np.array(list(vol.shape) + [-1])
        wnd_axis = tuple(-1-np.arange(vol.ndim)) #窗口所在的轴
        prg_wnd_fp = np.ones(wnd_shape, dtype=np.int32)
        if vol.ndim == 3:
            dist_wnd_fp = np.array([[[3, 2, 3],[2, 1, 2],[3, 2, 3]],
                                    [[2, 1, 2],[1, 0, 1],[2, 1, 2]],
                                    [[3, 2, 3],[2, 1, 2],[3, 2, 3]]], dtype=np.int32)   #窗口距离
        elif vol.ndim == 2:
            dist_wnd_fp = np.array([[2, 1, 2],[1, 0, 1],[2, 1, 2]], dtype=np.int32)
        else:
            raise RuntimeError("tortuosity of the input object vol dimension is not defined") 
        dist_wnd = np.array([0, 1, 1.414, 1.732, 2])[dist_wnd_fp]
        flat_dist_wnd = dist_wnd.reshape(-1)

        #传播距离(边界填充)
        #未传播点的距离为Inf(包括物体所在体素)
        dist_vol_pool = np.full(np.array(vol.shape)+2, np.inf, dtype=np.float32)   #默认为未传播距离标记
        pool_key = tuple([slice(1, -1) for i in range(vol.ndim)])
        dist_vol = dist_vol_pool[pool_key]

        #传播迭代
        inc_vol = start_vol #增量
        cur_vol = start_vol #当前累积
        dist_vol[inc_vol] = 0  #分配起点

        #最近邻最小曲折度传播
        for prg_i in range(max_prg):
            #传播迭代
            slide_wnd = np.lib.stride_tricks.sliding_window_view(dist_vol_pool, wnd_shape)

            #窗口距离传播 每个点附近最近/平均距离
            if mode == "min":
                pre_dist_vol = np.min(slide_wnd, axis=wnd_axis)  #窗口内最近传播点距离
            elif mode == "avg":
                pre_dist_vol = np.sum(slide_wnd < np.inf, axis=wnd_axis) / np.sum(1 / slide_wnd, axis=wnd_axis)
                pre_dist_vol[np.isnan(pre_dist_vol)] = np.inf  #发生除0位置视作传播无限大

            #最近传播距离+局部相对距离 多个传播来源则选取最小值作为传播来源
            loc_dist_vol = flat_dist_wnd[np.argmin(slide_wnd.reshape(slide_shape) + flat_dist_wnd, axis=-1)]  
            
            #相对距离允许添加权重
            if weight_vol.size:
                dist_vol[inc_vol] = (pre_dist_vol + loc_dist_vol * weight_vol)[inc_vol]   #相对增加距离
            else:
                dist_vol[inc_vol] = (pre_dist_vol + loc_dist_vol)[inc_vol]   #相对增加距离

            #传播点更新
            pre_vol = cur_vol  #更新之前传播位置
            cur_vol = np.bitwise_and(~vol, dilation(pre_vol, footprint=prg_wnd_fp))  #当前所有传播位置 
            inc_vol = np.bitwise_and(cur_vol, ~pre_vol)  #当前新增位置
            
            #若无新增传播区域 结束  
            # if not np.any(inc_vol) and mode == "min": break  
            if not np.any(inc_vol): break  

        #计算曲折度 
        tort_vol = (dist_vol + 0.5)/(loc_vol + 0.5)
        return DataVol(tort_vol) 

    # @property 
    def label_vol(self, connect=1):
        """按照4邻接标记域 对角相连不视作相连. color_vol则默认按照8邻接计算邻接关系图, 避免对角相连相视作不相连.
        目的在于避免marching_cubes中产生多对角连通歧义. 
        Examples:
            vol=PhaseVol.ones([6,6,1])
            vol[:3,:3,0]=2; vol[-3:,-3:,0]=2
            vol = vol.label_vol().color_vol()
        """
        from skimage.measure import label
        return PhaseVol(label(self[:], connectivity=connect))

    def segment(self, min_dist=3) -> PhaseVol:
        """有属性部分从1开始的分割"""
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max

        dist_vol = fedt(self.vol[:])
        seg_cen = peak_local_max(dist_vol, min_dist, exclude_border=False)   #保留所有的峰 包括边界点

        marks = PhaseVol.zeros(dist_vol.shape)  #标记点 
        marks[tuple(seg_cen.T)] = np.arange(seg_cen.shape[0]) + 1 
        lab_vol = PhaseVol(watershed(-dist_vol, marks[:], mask=self.vol[:])) 
        return lab_vol
        
    def seg_center(self, min_dist=3) -> np.ndarray:
        from skimage.feature import peak_local_max

        dist_vol = fedt(self.vol[:])
        seg_cen = peak_local_max(dist_vol, min_dist, exclude_border=False)
        return seg_cen 

    def seg_erosion(self, depth=1.0, measure=fedt):
        if depth < 0.5 or self.is_uni: return self.__class__(self.vol)
        # dlt_vol = self.dilation(depth, measure=measure)
        ero_vol = self.segment().color_vol().erosion(depth + 0.5, measure=measure)
        return self.__class__(ero_vol.open(depth + 0.5, measure=measure))

    def seeds_shape(self, depth=1.0, measure=fedt):
        return self 

    # @property
    # def rad_dist(self):
    #     from skimage.feature import peak_local_max

    #     dist_vol = fedt(self.vol[:])
    #     cen = peak_local_max(dist_vol, 3)
    #     rads = dist_vol[tuple(cen.T)]
    #     return rads / self.vxl_len  #归一化

class PhaseVol(ObjVol):
    """多相数组 加法不叠加(覆盖) 0视作没有相"""
    vol_dtype = np.int32

    def __getstate__(self):
        state_dict = {"vol": self.vol}
        if self.__dict__.get("legend", False): 
           state_dict.update({"legend": self.legend})
        if self.__dict__.get("voxel_size", False): 
           state_dict.update({"voxel_size": self.voxel_size})
        return state_dict

    @auto_vol
    def __add__(self, value: PhaseVol) -> PhaseVol:
        """相属性数组相加 后者非零部分覆盖前者"""
        value = np.array(value, dtype=self.vol_dtype)
        phs_vol = PhaseVol.zeros(self.shape)
        phs_vol[self.vol.astype(np.bool_)] = self.vol[self.vol.astype(np.bool_)]
        phs_vol[value.astype(np.bool_)] = value[value.astype(np.bool_)]
        return phs_vol

    @auto_vol
    def __iadd__(self, value) -> PhaseVol:
        value = np.array(value, dtype=self.vol_dtype)
        self.vol[value.astype(np.bool_)] = value[value.astype(np.bool_)]
        return self 
    
    def open(self, depth=1.0, measure=cdt):
        """开运算 先腐蚀后膨胀 移除桥"""
        vol = PhaseVol.zeros(self.shape)
        for attr in self.attrs:
            if attr == 0: continue
            vol[ObjVol(self[:] == attr).open(depth, measure)] = attr  #作为索引
        return vol

    def close(self, depth=1.0, measure=cdt):
        """闭运算 先膨胀后腐蚀 移除孔 多属性运算累积"""
        vol = PhaseVol.zeros(self.shape)
        for attr in self.attrs:
            if attr == 0: continue
            vol[ObjVol(self[:] == attr).close(depth, measure)] = attr  #作为索引
        return vol
    
    @property
    def surface(self):
        vol = PhaseVol.zeros(self.shape)
        for attr in self.attrs:
            obj_vol = ObjVol(self[:] == attr).surface
            vol[obj_vol[:]] = attr  #作为索引
        return vol
    
    @property
    def surf_area(self):
        return {attr: ObjVol(self[:] == attr).surf_area for attr in self.attrs}
    
    @property
    def spec_surf_area(self):
        return {attr: ObjVol(self.vol == attr).spec_surf_area for attr in self.attrs}

    def intf_area(self, a0, a1):
        ift_cnt = np.count_nonzero(np.bitwise_and(ObjVol(self[:]==a0).out_surface[:], self[:]==a1))
        return ift_cnt
    
    def spec_intf_area(self, a0, a1):
        atr_cnt = np.count_nonzero(self[:]==a1)
        return self.intf_area(a0, a1) / atr_cnt
    
    def segment(self, min_dist=1):
        """分水岭分割 LaggureVoro覆盖 未分割区域标记0 分割区域序号1~n"""
        seg_vol = PhaseVol.zeros(self.vol.shape)  
        for attr in self.attrs:
            if attr == 0: continue
            attr_vol = ObjVol(self.vol == attr)  #参与分割的区域掩膜
            idx_vol = attr_vol.segment(min_dist) #分割结果 序号从1开始 0表示未分割区域
            seg_vol[attr_vol] += idx_vol[attr_vol] + seg_vol.max 
        return seg_vol

    
    # def remove_small(self, *args, **kwargs):
    #     from skimage.morphology import remove_small_objects
    #     return remove_small_objects(self[:], *args, **kwargs)


    def seg_center(self, min_dist=1):
        """分水岭分割 LaggureVoro覆盖的中心点 顺序与segment对应"""
        # return ObjVol(self.surface[:]==0).seg_center()
        seg_cen = [] 
        for attr in self.attrs:
            if attr == 0: continue
            attr_vol = ObjVol(self.vol == attr)
            seg_cen.append(attr_vol.seg_center(min_dist))
        return np.concatenate(seg_cen, axis=0)

    def seg_erosion(self, depth=1.0, measure=cdt):
        seg_vol = PhaseVol.zeros(self.vol.shape)  
        for attr in self.attrs:
            if attr == 0: continue 
            seg_vol[ObjVol(self.vol == attr).seg_erosion(depth, measure=measure)] = attr 
        return seg_vol 

    @property
    def rad_dist(self):
        return {attr: ObjVol(self.vol == attr).rad_dist for attr in self.attrs}
    
    def a_shape(self, nvxls=1.0, measure=fedt, pad=True) -> PhaseVol:
        """为属性体素网格计算外边界 n_vxls输入除0位置之外的修正体素"""
        if self.is_uni: return self.__class__(self.vol)   #空物体/均匀物体不计算
        attrs = [attr for attr in self.attrs if attr != 0]

        if isinstance(nvxls, (int, float, np.integer, np.floating)):
            nvxls = [nvxls for attr in attrs]
        if not np.any(nvxls): return self.__class__(self.vol)  #vxl为0跳过
    
        ashp_vol = PhaseVol.zeros(self.shape)   #逐个属性连续计算
        for attr, nvxl in zip(attrs, nvxls):
            ashp_vol[ObjVol(self[:] == attr).a_shape(nvxl, measure, pad)] = attr  #作为索引
        #NOTE 后续的小半径会覆盖之前的
        return ashp_vol

    def inv_a_shape(self, nvxls=1.0, measure=fedt, pad=True) -> PhaseVol:
        if self.is_uni: return self.__class__(self.vol)   #空物体/均匀物体不计算
        attrs = [attr for attr in self.attrs if attr != 0]

        if isinstance(nvxls, (int, float, np.integer, np.floating)):
            nvxls = [nvxls for attr in attrs]
        if not np.any(nvxls): return self.__class__(self.vol)  #vxl为0跳过
    
        ashp_vol = PhaseVol.zeros(self.shape)
        for attr, nvxl in zip(attrs, nvxls):
            ashp_vol[ObjVol(self[:] == attr).inv_a_shape(nvxl, measure, pad)] = attr  #作为索引
        #NOTE 不发生覆盖, 但会产生空属性位置
        return ashp_vol

    @property 
    def is_uni(self) -> bool:
        return (np.max(self.vol) - np.min(self.vol)) < self.zero_th
    
    @property
    def is_zero(self) -> bool:
        return np.max(self.vol) < self.zero_th

    @property 
    def vol_frac(self):
        vol_frac_dict = {} 
        for attr in self.attrs:
            if attr == 0: continue
            vol_frac_dict.update({attr: ObjVol(self.vol[:]==attr).vol_frac})
        return vol_frac_dict
    

    def residual(self, sigma=1.0):
        """对属性0进行同样的计算"""
        res = DataVol.zeros(self.shape)
        for attr in self.attrs:
            res += ObjVol(self.vol == attr).residual(sigma=sigma)
        return res 
    
    def multi_residual(self, sigma=1.0, n_level=5) -> list:
        """不在自身的rescale上计算层级残差 而是在自身obj上计算层级残差后求和"""
        res_list = []
        for i in range(min(self.n_level, n_level)):
            res = DataVol.zeros(self.shape)
            for attr in self.attrs:
                res += ObjVol(self.vol == attr).scaled_residual(0.5**i, sigma=sigma)
                if attr == 0: res *= 2.0  #边缘加强
            res_list.append(res)
        return res_list

    def rescale(self, scale: float) -> PhaseVol:
        """均匀插值放缩"""
        vol = PhaseVol(ObjVol.zeros(self.shape).rescale(scale))
        for attr in self.attrs:
            mask = ObjVol(self.vol == attr).rescale(scale)
            vol[mask[:]] = attr 
        return vol
    
    def erosion(self, depth=1, measure=cdt, connect=1):
        vol = PhaseVol.zeros(self.shape)
        for attr in self.attrs:
            if attr == 0: continue 
            vol[ObjVol(self.vol == attr).erosion(depth, measure=measure, connect=connect)] = attr 
        return vol
    
    def dilation(self, depth=1, measure=cdt, connect=1):
        """多相膨胀 存在冲突时 属性值大的覆盖属性值小的"""
        vol = PhaseVol.zeros(self.shape)
        for attr in self.attrs:
            if attr == 0: continue 
            vol[ObjVol(self.vol == attr).dilation(depth, measure=measure, connect=connect)] = attr 
        return vol

    def resize(self, shape):
        vol = PhaseVol.zeros(shape)
        for attr in self.attrs:
            if attr == 0: continue 
            vol[ObjVol(self.vol == attr).resize(shape)] = attr 
        return vol 

    def watershed(self) -> PhaseVol:
        """体素voronoi变换"""
        from skimage.segmentation import watershed

        #分水岭算法
        distance = fedt(self.vol[:]==0)
        voro_vol = watershed(distance, self.vol[:])
        return PhaseVol(voro_vol)

    
    def query_attr(self, loc, sigma=1.2, smooth=True):
        """点的属性插值,选取多相内最权重大属性作为点属性"""
        attr_weights = np.zeros(shape=(loc.shape[0], self.attrs.size))  #属性权重
        for i, attr in enumerate(self.attrs):  #从1开始查询
            if attr == 0: continue             
            weight_vol = DataVol(self[:]==attr)  
            if smooth:  #默认平滑 能显著平滑尖锐点
                weight_vol = weight_vol.smooth(sigma)
            attr_weights[:, i] = weight_vol.itp_query(loc)
        return self.attrs[np.argmax(attr_weights, axis=1)]

    @property 
    def connect_graph(self, connect=2):  #对角视作连通, 避免对角歧义
        """不同属性间的连通关系"""
        #NOTE 三维和二维的connect不同 三维connect=3才是方块形
        # from skimage.graph import RAG

        # rag = RAG(self.vol, connectivity=connect).adj 
        # graph = {key: tuple(rag[key]) for key in rag}
        # return graph 
        from scipy import ndimage as ndi
        import networkx as nx 

        labels = self.vol
        conn = ndi.generate_binary_structure(labels.ndim, connect)  #获取滤波掩膜
        eroded = ndi.grey_erosion(labels, footprint=conn)
        dilated = ndi.grey_dilation(labels, footprint=conn)
        boundaries0 = eroded != labels
        boundaries1 = dilated != labels
        labels_small = np.concatenate((eroded[boundaries0], labels[boundaries1]))
        labels_large = np.concatenate((labels[boundaries0], dilated[boundaries1]))

        pairs = np.concatenate([labels_small[:, None], labels_large[:, None]], axis=1)
        edges = remove_dul_points(pairs)[0]
        edges = np.concatenate([edges, edges[:, ::-1]])

        rag = nx.Graph()
        rag.add_edges_from(edges)
        graph = {key: tuple(rag[key]) for key in rag.adj}
        return graph
    
    # @property 
    def color_vol(self):
        """图着色"""
        cg = self.connect_graph
        if 0 in cg: cg.pop(0)  #属性0不上色 仍为0 
        cv_dict = color_nodes(cg) 
        attr_list = np.arange(np.max(tuple(cv_dict.keys())) + 1)
        color_list = np.array([cv_dict.get(attr, -1) for attr in attr_list]) + 1  #从0开始 颜色映射关系
        color_vol = PhaseVol(color_list[self[:]]) 
        return color_vol

    @classmethod
    def from_hsdat(self, shape, head_rows=1):
        """shape对应 HsDat(x,y,z) -> PhaseVol(z,y,x)"""
        skiprows = np.prod(np.array(shape) + 1) + head_rows
        cell_data = np.loadtxt("meshs/mesh.dat", skiprows=skiprows)[:, -1:]
        return PhaseVol(cell_data.reshape(shape)) 

def remove_dul_points(vertices):
    """移除重复点,输入整数点坐标
    Examples:
        pnts = np.array([[1,2,3],[4,2,1],[2,4,5],[3,4,5],[1,2,3]])
        new_pnts, pnts_idx, invs_idx = remove_dul_points(pnts)
    """
    vertices_flat = np.ravel(vertices.view([('', vertices.dtype)]*vertices.shape[1]))
    _, verts_idx, invs_idx, counts = np.unique(vertices_flat, return_index=True, return_inverse=True, return_counts=True)
    return vertices[verts_idx], verts_idx, invs_idx, counts

def color_nodes(graph):
    """图着色贪婪算法"""
    vertices = sorted((list(graph.keys())))
    colour_graph = {}

    for vertex in vertices:
        unused_colours = len(vertices) * [True]

        for neighbor in graph[vertex]:
            if neighbor in colour_graph:
                colour = colour_graph[neighbor]
                unused_colours[colour] = False
        for colour, unused in enumerate(unused_colours):
            if unused:
                colour_graph[vertex] = colour
                break

    return colour_graph


class DistVol(DataVol):
    vol_dtype = np.float64 


class ProbVol(DataVol):
    """概率场,不允许小于0"""
    vol_dtype = np.float64

    @auto_vol
    def __mul__(self, value) -> ProbVol:
        return ProbVol(self.vol.__mul__(value))

    @auto_vol
    def __imul__(self, value) -> ProbVol:
        self.vol.__imul__(value)
        return self 
    
    @auto_vol
    def __truediv__(self, value) -> ProbVol:
        return ProbVol(self.vol.__truediv__(value)) 

    @auto_vol
    def __itruediv__(self, value) -> ProbVol:
        self.vol.__itruediv__(value)
        return self 

    def normalize(self):
        """归一化,检查是否为0概率场,尽量少调用"""
        #应该始终保持概率场是归一化的 通过控制归一化中的权重分配大小
        vol_sum = self.sum
        if self.is_zero: 
            return self.vol
        elif self.is_norm: 
            return self.vol
        else: 
            self.vol = self.vol/vol_sum  #保存归一化结果
            return self.vol
    
    def sample(self, cnt: int) -> np.ndarray:
        """对概率场采样,返回点在域中的坐标"""
        self.normalize() #尽量只在概率采样前进行标准化
        list_pos = np.random.choice(self.vol.size, cnt, p=self.vol.reshape((-1,)))
        return vol_pos(list_pos, self.shape) / self.vol_len  
    
    # def local_prod(self, nvxl=3) -> ProbVol:
    #     """局部累计 按照给定的体素半径"""
    #     #后验概率: 假设为概率乘积
    #     # log_cum = DataVol(np.log2(self[:]+1e-8)).scaled_smooth(nvxl/3)
    #     log_smt = DataVol(np.log2(self[:]+1e-8)).rescale(nvxl/1)  #下采样rescale自带平滑
    #     log_cum = log_smt.rescale(np.array(self.shape)/np.array(log_smt.shape))
    #     return self.__class__(np.exp2(log_cum[:]))



#-------------------------------------------------------------------------------------#
# 其他函数
#-------------------------------------------------------------------------------------#
                
def vol_pos(list_pos, vol_shape):
    """将列表坐标转化为体素坐标
    Args:
        list_pos (np.array): 点的列表坐标
        vol_shape (list or tuple): 体素网格形状
    Retrun:
        np.ndarray: xyz方向坐标列表,不同点对应不同行
    """
    vol_pos = np.zeros(shape=(list_pos.size, 3), dtype=np.int32)
    vol_pos[:, 0] = list_pos//(vol_shape[1]*vol_shape[2])
    vol_pos[:, 1] = list_pos%(vol_shape[1]*vol_shape[2])//(vol_shape[2])
    vol_pos[:, 2] = list_pos%(vol_shape[2])
    return vol_pos 

def list_pos(vol_pos, vol_shape):
    """将体素坐标转化为列表坐标"""
    return  vol_pos[2]*vol_shape[0]*vol_shape[1] + vol_pos[1]*vol_shape[0] + vol_pos[0]

def move_axes(arr):
    """向量维度轮换 最外维度轮换到最内部"""
    arr = np.array(arr)
    rshp_key = tuple([slice(None) for _ in range(arr.ndim)] + [np.newaxis])
    new_shape = list(arr.shape[1:]) + [arr.shape[0]]
    new_arr = np.swapaxes(arr[rshp_key], 0, -1).reshape(new_shape)
    return new_arr

def s2_point_correlation(pnt1_vol: ObjVol, pnt2_vol: ObjVol, ndim, n_sample=100000, periods=[False, False, False], nbin=400):
    #按照相索引对 选取采样点
    dist = s2_pntcor_data(pnt1_vol, pnt2_vol, n_sample, periods)
    hists, edges = np.histogram(dist, bins=nbin)
    return edges[1:], hists/n_sample/edges[1:]**max(0, ndim-1)

def s2_pntcor_data(pnt1_vol: ObjVol, pnt2_vol: ObjVol, n_sample=100000, periods=[False, False, False]):
    #周期性边界填充
    npad1 = np.array(pnt1_vol.shape)/2 * periods
    npad2 = np.array(pnt2_vol.shape)/2 * periods
    if np.any(npad1):
        pnt1_pad = np.pad(pnt1_vol[:], np.array([npad1, npad1], dtype=np.int32).T, "wrap")  #周期性边界补充
        pnt2_pad = np.pad(pnt2_vol[:], np.array([npad2, npad2], dtype=np.int32).T, "wrap")  #周期性边界补充
    else: 
        pnt1_pad = pnt1_vol
        pnt2_pad = pnt2_vol

    loc1 = move_axes(np.array(np.where(pnt1_pad[:])/np.max(pnt1_vol.shape)))
    loc2 = move_axes(np.array(np.where(pnt2_pad[:])/np.max(pnt2_vol.shape)))

    n_sample = int(n_sample)
    idx1 = np.random.randint(0, loc1.shape[0], n_sample) 
    idx2 = np.random.randint(0, loc2.shape[0], n_sample) 

    rloc = loc1[idx1] - loc2[idx2]
    for i, period in enumerate(periods):  #应用周期性边界
        if period: rloc[:, i] = rloc[:, i] % 1.0
    dists = np.linalg.norm(rloc, axis=1)
    return dists

def c2_point_correlation(pnt1_vol: ObjVol, pnt2_vol: ObjVol, n_sample=100000, nbin=100):
    return  



@auto_vol
def calculate_3d_correlation_lengths(voxel_grid):
    """
    计算三维体素网格在x、y、z三个方向上的相关长度
    
    参数:
    voxel_grid: 三维numpy数组，表示体素网格
    
    返回:
    tuple: (corr_length_x, corr_length_y, corr_length_z) 三个方向的相关长度
    """
    from scipy.fft import fftn, ifftn, fftshift
    # 将输入转换为浮点数以提高计算精度
    data = voxel_grid.astype(np.float64)
    
    # 1. 零均值化
    data_mean = np.mean(data)
    data_zero_mean = data - data_mean
    
    # 2. 使用FFT计算自相关函数
    fft_data = fftn(data_zero_mean)
    power_spectrum = fft_data * np.conj(fft_data)
    acf = np.real(ifftn(power_spectrum))
    
    # 3. 归一化自相关函数
    acf_center = acf[0, 0, 0]  # 中心点值
    acf_normalized = acf / acf_center
    
    # 4. 将原点移到中心以便分析
    acf_shifted = fftshift(acf_normalized)
    
    # 5. 获取网格中心坐标
    center_z, center_y, center_x = np.array(acf_shifted.shape) // 2
    
    # 6. 提取通过中心点的三个方向的一维自相关函数
    # x方向 (保持y和z在中心)
    acf_x = acf_shifted[center_z, center_y, :]
    # y方向 (保持x和z在中心)
    acf_y = acf_shifted[center_z, :, center_x]
    # z方向 (保持x和y在中心)
    acf_z = acf_shifted[:, center_y, center_x]
    
    # 7. 计算每个方向的相关长度
    corr_length_x = find_correlation_length(acf_x, center_x)
    corr_length_y = find_correlation_length(acf_y, center_y)
    corr_length_z = find_correlation_length(acf_z, center_z)
    
    return corr_length_x, corr_length_y, corr_length_z

@auto_vol
def calculate_3d_autocorrelation(voxel_grid):
    """
    计算三维体素网格的自相关函数和相关长度
    
    参数:
    voxel_grid: 三维numpy数组，表示体素网格
    
    返回:
    tuple: (acf_x, acf_y, acf_z, corr_length_x, corr_length_y, corr_length_z)
           acf_x, acf_y, acf_z: 三个方向的一维自相关函数
           corr_length_x, corr_length_y, corr_length_z: 三个方向的相关长度
    """
    
    from scipy.fft import fftn, ifftn, fftshift

    # 将输入转换为浮点数以提高计算精度
    data = voxel_grid.astype(np.float64)
    
    # 1. 零均值化
    data_mean = np.mean(data)
    data_zero_mean = data - data_mean
    
    # 2. 使用FFT计算自相关函数
    fft_data = fftn(data_zero_mean)
    power_spectrum = fft_data * np.conj(fft_data)
    acf_3d = np.real(ifftn(power_spectrum))
    
    # 3. 归一化自相关函数
    acf_center = acf_3d[0, 0, 0]  # 中心点值
    acf_normalized = acf_3d / acf_center
    
    # 4. 将原点移到中心以便分析
    acf_shifted = fftshift(acf_normalized)
    
    # 5. 获取网格中心坐标
    nz, ny, nx = acf_shifted.shape
    center_z, center_y, center_x = nz // 2, ny // 2, nx // 2
    
    # 6. 提取通过中心点的三个方向的一维自相关函数
    # x方向 (保持y和z在中心)
    acf_x = acf_shifted[:, center_y, center_x]
    # y方向 (保持x和z在中心)
    acf_y = acf_shifted[center_z, :, center_x]
    # z方向 (保持x和y在中心)
    acf_z = acf_shifted[center_z, center_y, :]
    
    return acf_x, acf_y, acf_z

def find_correlation_length(acf_1d, center_index):
    """
    从一维自相关函数中找到相关长度
    
    参数:
    acf_1d: 一维自相关函数数组
    center_index: 中心点的索引
    
    返回:
    float: 相关长度
    """
    # 只考虑从中心点到右半部分（正方向）
    right_half = acf_1d[center_index:]
    
    # 找到第一个自相关值低于1/e的点
    threshold = 1 / np.e  # 约0.3679
    
    # 找到所有低于阈值的点
    below_threshold = np.where(right_half < threshold)[0]
    
    if len(below_threshold) == 0:
        # 如果没有点低于阈值，返回整个范围
        return len(right_half) - 1
    
    # 第一个低于阈值的点
    first_below = below_threshold[0]
    
    if first_below == 0:
        # 如果第一个点就低于阈值，返回0
        return 0.0
    
    # 线性插值以获取更精确的相关长度
    # 获取低于阈值点前的一个点（高于阈值）
    prev_index = first_below - 1
    prev_value = right_half[prev_index]
    curr_value = right_half[first_below]
    
    # 计算插值位置
    t = (threshold - prev_value) / (curr_value - prev_value)
    corr_length = prev_index + t
    
    return corr_length