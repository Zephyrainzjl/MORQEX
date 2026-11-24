from  __future__ import annotations
from typing import Callable

from morqex.microstructure.core.rasterization import Voxelizer 
from morqex.microstructure.core.voxel_fields import * 
from morqex.microstructure.core.distributions import * 

import time 

#-------------------------------------------------------------------------------------#
# 全局设置
#-------------------------------------------------------------------------------------#

np.set_printoptions(suppress=True, threshold=20, precision=8)
np.set_printoptions(legacy="1.25")

#-------------------------------------------------------------------------------------#
# 基类: 源节点 & 链接节点
#-------------------------------------------------------------------------------------#

#check在运行前检查依赖状态 访问自身值之前检查即可
def auto_check(func):
    """函数修饰器 在调用函数前检查依赖是否改变"""
    def auto_check_func(self: Link, *args, **kwargs):
        self.check_update()  #先检查引用是否改变 
        return func(self, *args, **kwargs)
    return auto_check_func

def auto_memo(func):
    """自动缓存函数调用结果, 根据key函数返回的值标识状态"""
    def mem_func(self: Link, *args, **kwargs):
        if not self.__dict__.get("__memo", False):
            self.__memo = {} 
        self.check_update() #若发生了状态改变 则更新缓存
        # args = [id(arg) if isinstance(arg, np.ndarray) else arg for arg in args]  # 处理不可hash对象
        # memo_key = (*self.run_key(), func, *args, *tuple(kwargs.values()))  #由自身状态,函数,函数参数决定 kw顺序会影响key
        arg_keys = tuple([arg.run_key() if isinstance(arg, Node) else arg for arg in args])
        kwarg_keys = tuple([arg.run_key() if isinstance(arg, Node) else arg for arg in tuple(kwargs.values())])
        memo_key = (*self.run_key(), func, *arg_keys, *kwarg_keys)
        if not self.__memo.get(memo_key, False): 
            ret = func(self, *args, **kwargs)
            self.__memo.update({memo_key: ret}) 
        return self.__memo[memo_key]
    return mem_func

def hex_id(obj):
    return hex(id(obj))[-Node.id_len:]


class OrderSet(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check()
        return 

    def check(self):
        order_set = [] 
        for item in self:
            if item in order_set:
                self.remove(item)  #已经记录过的元素
                continue
            order_set.append(item)
        return self 

    def add(self, item):
        if item not in self:
            self.append(item) 
    
    def update(self, item_set):
        for item in item_set:
            self.add(item)
        return self 
        
    def intersection(self, item_set):
        iset = OrderSet()
        for item in self:
            if item in item_set:
                iset.add(item)
        return iset

    def difference(self, item_set):
        dset = OrderSet()
        for item in self:
            if item not in item_set:
                dset.add(item)
        return dset
    
    def union(self, item_set):
        return self.update(item_set)
    
    def __add__(self, items):
        return self.update(items)

class Node():
    id_len = 4 
    def __init__(self, *in_args, **in_kwargs) -> None:
        """按照一定参数初始化 kwargs需要保持顺序 
        源节点的参数决定节点自身 若参数变化则key变化
        若包含array作为参数 则需要额外定义自身输入参数状态 如使用id(array)
        """
        self.__in_args = in_args 
        self.__in_kwargs = in_kwargs 
        self.update() 

    def dec_key(self) -> tuple:
        """节点标识数据 自身数据域 依赖的数据域"""
        return ()
    
    def run_key(self) -> tuple:  #声明时即为运行时状态 子类不重载
        return self.dec_key()
    
    def modified(self):
        return self._dec_key != self.dec_key()

    def update(self):
        """强制更新自身状态 source子类不重载"""
        self.__memo = {} 
        self._dec_key = self.dec_key()
    
    def check_update(self):
        """检查依赖状态 判断自身是否发生参数更新"""
        if self.modified(): self.update() 

    def __setstate__(self, state):
        """加载数据, 并更新其后继数据状态"""
        self.__dict__.update(state)
        self.update()
    
    def load_instance(self, instance):
        """从对象加载状态 保持自身地址"""
        self.__setstate__(instance.__getstate__())

    @property
    def cls_id_str(self):
        return f"{self.cls_str}-{self.id_str}"
    @property
    def cls_str(self):
        return f"{self.__class__.__name__}"
    @property
    def id_str(self):
        return f"{hex_id(self)}"
    
    def __str__(self):
        return f"{self.cls_str}()" 
    
    def __repr__(self):
        return self.__str__()

class Link(Node):
    """链接节点自身无参数 其状态由依赖节点的状态决定 其依赖的对象由key描述 其依赖对象的状态由dep_key描述"""
    #自身运行状态依赖自身输入的节点状态
    def __init__(self, *args, **kwargs) -> None:
        """链接 在依赖节点的key改变后自动更新"""
        self.__args = args 
        self.update() 

    def dec_key(self) -> tuple:  
        """标识自身当前依赖对象(自身参数) 返回值必须为tuple"""
        return self.nodes

    def run_key(self) -> tuple:
        """在声明节点未改变前提下 查询自身运行状态 自身参数+依赖配置参数"""
        return *tuple([dep.dec_key() for dep in self.args]), *self.dec_key()   #依赖节点 + 依赖节点参数状态 (是否更换了依赖节点(可选),依赖节点参数是否变化)
    
    def update(self):
        """更新自身状态 维持自身与输入节点的关系 Link子类重载"""
        Node.update(self) 
        self._run_key = self.run_key() 

    def modified(self):
        return self._run_key != self.run_key()

    def check_update(self):
        """检查依赖状态 判断自身是否发生参数更新"""
        Node.check_update(self) #检查声明状态是否改变
        if self.modified(): self.update() 


class ProcessBar():
    """进度条"""
    n_bar = 40
    bar_str = "="
    fill_str = " "

    def __init__(self):
        self.n_bat = 100
        self.cur_bat = 0 
        self.on = True 

    def print_process(self, end=""):
        bar_cnt = int(min((self.cur_bat + 1)/self.n_bat, 1) * self.n_bar) if self.n_bat else 0 
        time_str = f"{np.round(time.time() - self.start_time, 3)}s"
        proc_str = f"{self.cur_bat}/{self.n_bat}"
        self.print(f"\r{self.__class__.__name__}: [{self.bar_str*bar_cnt}{self.fill_str*(self.n_bar-bar_cnt)}]{proc_str} ({time_str})    ", end=end)
        return time_str, proc_str

    def start(self, n_bat: int):
        self.start_time = time.time()

        self.n_bat = n_bat
        self.cur_bat = 0 
        end = "\n" if self.cur_bat >= self.n_bat else ""
        self.print_process(end=end)

    def inc(self):
        self.cur_bat += 1
        end = "\n" if self.cur_bat >= self.n_bat else ""
        self.print_process(end=end)
    
    def end(self):
        return 

    def print(self, *args, **kwargs):
        if self.on: print(*args, **kwargs)
        

#-------------------------------------------------------------------------------------#
# 点相关特征函数
#-------------------------------------------------------------------------------------#



class FeatureFunc(Node):
    """从物体点计算最近邻特征场, 包括距离变换,取向变换,相对坐标变换"""
    n_feature = 1 
    cen_r = 1e-8
    #中心体素半径 对较小种子的表面判断会产生较大影响 取更小值允许更多的>0采样点,减少表面距离
    #主要影响填充图片过程: 松散/集中 取1e-8与LaguerreVoro一致

    def ftr_shape(self, obj_vol: ObjVol):
        return np.append(obj_vol.shape, self.n_feature) 

    def transform(self, obj_pnt: ObjectPnt) -> DistVol:
        """约定: 最内部维度包含不同特征, 默认返回四维数组"""
        #从物体网格计算特征场 
        ret_vol = DistVol(np.zeros(self.ftr_shape(obj_pnt.vol)))
        return ret_vol
    
    def tgt_transform(self, obj_pnt: SeedsPnt = None) -> ObjVol:
        """计算采样点"""
        return ObjVol.zeros(obj_pnt.vol.shape)
    
    def sur_dist(self, obj_pnt: SeedsPnt = None) -> float:
        """目标表面距离(体素)"""
        if isinstance(obj_pnt, type(None)):
            return self.cen_r
        else:
            return obj_pnt.nvxl + self.cen_r
    
    def copy(self):
        return self.__class__()

class NNFeatureFunc(FeatureFunc):
    """最近邻特征提取"""
    n_feature = 1

    def __init__(self, measure=cdt): 
        self.measure = measure
        self.update() 
    
    def dec_key(self) -> tuple:
        return self.measure, 

    def __str__(self):
        return f"{self.cls_str}(ms:{self.measure})"
    
    def __repr__(self):
        return f"{self.cls_str}(ms:{self.measure})"
    
    def copy(self):
        return self.__class__(self.measure)
    
class NNDst(NNFeatureFunc):
    """最近邻距离变换"""
    n_feature = 1

    #不在此处自动缓存 obj_vol通常在变动
    def transform(self, obj_pnt: ObjectPnt) -> DistVol:
        """约定: 最内部维度包含不同特征, 默认返回四维数组"""
        #从物体网格计算特征场 
        obj_vol = obj_pnt.vol[:]
        dist_vol = DistVol(self.measure(obj_vol==0, periods=obj_pnt.reg.periods).reshape(self.ftr_shape(obj_vol)))
        return dist_vol

    def tgt_transform(self, obj_pnt: SeedsPnt) -> ObjVol:
        """计算采样点 取种子中心点"""
        return ObjVol(obj_pnt.seeds.discrete(obj_pnt.reg, 0.0)) 



# class NNOrt(NNFeatureFunc):
#     """最近极坐标变换"""
#     n_feature = 3

#     @auto_vol
#     def transform(self, obj_pnt: ObjectPnt) -> DistVol:
#         obj_vol = obj_pnt.vol[:]
#         dist, ftr_vol = self.measure(obj_vol==0, periods=obj_pnt.reg.periods, return_indices=True, return_distances=True)
#         rloc_vol = ftr_vol - np.mgrid[tuple([slice(i_len) for i_len in obj_vol.shape])]

#         #TODO 性能优化 与 数据域约定
#         #坐标轴夹角余弦
#         rloc_vol = DistVol(move_axes(rloc_vol))
#         # dist 
#         return rloc_vol 


class NNLoc(NNFeatureFunc):
    """最近邻相对位置变换"""
    n_feature = 3 

    @auto_vol
    def transform(self, obj_pnt: ObjVol) -> DistVol:
        """约定: 最内部维度包含不同特征, 默认返回四维数组"""
        #从物体网格计算特征场 
        obj_vol = obj_pnt.vol[:]
        ftr_vol = self.measure(obj_vol==0, periods=obj_pnt.reg.periods, return_indices=True, return_distances=False)
        npad = np.array(obj_vol.shape)/2 * obj_pnt.reg.periods

        gird_key = [slice(None, ax_len) for ax_len in obj_vol.shape]
        rloc_vol = DistVol(move_axes(np.mgrid[gird_key]) - move_axes(ftr_vol) + npad - 0.5)
        return rloc_vol


class SymLoc(NNLoc):
    """对称相对位置, 小于0部分取绝对值"""
    @auto_vol
    def transform(self, obj_pnt: ObjVol) -> DistVol:
        rloc_vol = NNLoc.transform(self, obj_pnt)
        return DistVol(np.abs(rloc_vol[:]))


fastdt = NNDst()        #表面距离cdt
surdt = NNDst(fedt)     #表面距离edt  fedt速度快4倍左右
itrdt = NNDst(invfedt)  #内表面距离
clldt = NNDst(ndt)      #中心在区域外
clsdt = NNDst(invdt)    #中心封闭在区域内

locft = NNLoc(edt)      #相对坐标edt
slocft = SymLoc(edt)
# ortft = NNOrt()       #空间取向cdt


class Std2PntCor(FeatureFunc):
    """标准两点相关函数距离变换(同一相随机点距离,不归一化)"""
    n_feature = 40

    def __init__(self): 
        self.update() 

    def transform(self, obj_pnt1: ObjectPnt, obj_pnt2: ObjectPnt=None) -> DistVol:
        """空间中所有点到随机物体点的距离
        vol = ObjVol.zeros([120, 1, 120])
        vol[60, 0, 60] = 1 
        pnt = ObjectPnt(ImgObject(vol), reg)
        Std2PntCor().transform(pnt)
        """ 
        if obj_pnt2 == None: 
            obj_pnt2 = obj_pnt1
        obj_vol1 = obj_pnt1.vol[:].astype(np.bool_)
        obj_vol2 = obj_pnt2.vol[:].astype(np.bool_)
        if not np.any(obj_vol1) and np.any(obj_vol2):  #目标为空 返回零距离
            return DistVol.zeros(self.ftr_shape(obj_vol1))

        pnt_dist = s2_pntcor_data(obj_vol1, obj_vol2, self.n_feature*obj_vol1[:].size, obj_pnt1.reg.periods)
        return DistVol(pnt_dist.reshape(self.ftr_shape(obj_vol1)))
    
    def tgt_transform(self, obj_pnt: SeedsPnt = None) -> ObjVol:
        """计算采样点 取obj_obt占据体素"""
        return ObjVol.ones(obj_pnt.vol.shape) 

    def sur_dist(self, obj_pnt: SeedsPnt = None) -> float:
        """目标表面距离 和特征变换类型有关"""
        return 0.0 + self.cen_r

    
# class Cll2PntCor(Std2PntCor):
#     """考虑碰撞关系的两点相关函数"""

#     def transform(self, obj_pnt: ObjectPnt) -> DistVol:
#         """空间中所有点到随机物体点的距离 有源存在的位置赋特殊值""" 
#         obj_vol = obj_pnt.vol[:]
#         pcdist_vol = Std2PntCor.transform(self, obj_pnt)
#         pcdist_vol[ObjVol(obj_vol)] = np.max(obj_vol.shape)*(DistFunc.xmax-1e-4)  #在distfunc范围内的特殊值
#         return pcdist_vol


stdpcf = Std2PntCor()
# cllpcf = Cll2PntCor() 

#-------------------------------------------------------------------------------------#
# 概率密度模型 不在内部缓存数据
#-------------------------------------------------------------------------------------#


class DistFunc(Link): 
    """分布函数 其特征相对域定义"""
    xmin = 0.0
    xmax = 1.8
    nbin = same.nbin 
    model = same 
    wrn_th = 0.3
    n_ftr = 1 
    n_dist_sample = 1000

    def __init__(self, *params):  #定义分布模型
        self.params = params
        # self.dist = self.model(*self.params)

        # self.n_srs = 1
        self.reset_data() #每次拟合只添加 不覆盖
        self.update()

    def reset_data(self):
        self._data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32)

    def dec_key(self) -> tuple:
        return *tuple([id(param) for param in list(self.params)]), #在self.dist更新后将导致key变化

    def run_key(self) -> tuple:
        return *tuple([id(param) for param in list(self.params)]), id(self._data)   #通过传入params更换分布,而不是dist
    
    def update(self):
        if self._data.size:  #有缓存数据时进行数据更新
            #放弃distfunc范围外的数据 
            out_mark = self.invld_data_mask(self._data)
            data = self._data[~out_mark]  #抛弃不合要求的值

            out_prop = np.count_nonzero(out_mark)/out_mark.size
            if out_prop > self.wrn_th:
                warnings.warn(f"fitting {self.__str__()} warning: invalid data prop: {out_prop} reaches threshold {self.wrn_th}")
            
            # self.params = self.model.fit(data, self.n_srs)
            self.params = self.model.fit(data)
            #一次update后抛弃数据
            self.reset_data()

        #更新自身分布对象
        self.dist = self.model(*tuple(self.params))
        Link.update(self) 

    @auto_check
    def update_params(self, params):
        self.params = params
        self.dist = self.model(*params)
        Link.update(self) 

    @classmethod 
    def invld_data_mask(cls, data, tol=0.0): 
        return np.bitwise_or(data < cls.xmin - tol, data > cls.xmax + tol) 

    def __str__(self) -> str:
        if not np.size(self.params): 
            return f"{self.cls_str}(pm:{str(self.params)})"
        elif type(self.params[0]) in (np.integer, np.floating):
            return f"{self.cls_str}(pm:{np.round(self.params[:3], 3)})"
        else: 
            return f"{self.cls_str}(pm:{str(self.params)[:40]}..)"
    
    def __repr__(self) -> str:
        return self.__str__() 

    @auto_check
    def __getstate__(self):
        return {"params": self.params, "xmax": self.xmax, "xmin": self.xmin}
        
    def __setstate__(self, state):
        self.params = tuple(state["params"])
        self.reset_data()
        self.update()

    @classmethod 
    def from_pcdt(cls, dists: DistVol, ndim=1):
        data = dists[:].reshape(-1)
        if not np.any(data):
            raise ValueError(f"distfunc init from pc distance error: empty dists")
            
        step = same.x()[1] - same.x()[0]
        return cls(same.from_data(data).p / (same.x()+step)**max(0, ndim-1))
    
    @classmethod 
    def from_pcpnt(cls, pnt1: ObjectPnt, pnt2: ObjectPnt=None, ftr_func=stdpcf):
        return cls.from_pcdt(ftr_func.transform(pnt1, pnt2), pnt1.reg.ndim)

    @classmethod 
    def from_dist(cls, dist):
        return cls(*cls.model.fit(dist.rvs(cls.n_dist_sample)))

    @classmethod 
    def from_data(cls, data):
        return cls(*cls.model.fit(data))
    
    #@auto_check  #不检查 在所有数据添加完成后更新
    def fit(self, data: np.ndarray):  #拟合数据 获取分布参数
        """仅保存数据,data的id随之改变,在update内更新"""
        self._data = np.append(self._data, data.reshape(-1, self.n_ftr), axis=0)  #保持维度一致 
    
    @auto_check
    def sample(self, n_sample=1): #采样 获得数据实例
        """采样 返回特征值 不缓存结果"""
        return self.dist.rvs(n_sample) 
    
    @auto_check
    def density(self, x: np.ndarray): #输入多元变量X
        """返回某个特征出现的概率密度"""
        return self.dist.pdf(x).reshape(-1)
    #TODO density依赖r 

    @auto_check
    def cumdensity(self, x: np.ndarray): 
        """返回某个特征出现的分部函数"""
        # print(f"call cumdensity")
        return self.dist.cdf(x).reshape(-1)
        # return np.insert(self.dist.cdf(same.x()).reshape(-1)[1:], 0, 0.0)
    
    @auto_check
    def nndensity(self, x: np.ndarray):
        """根据变量最小值运算得到最近邻距离"""
        # print(f"call nndensity x:{x[:10]} max:{np.max(x)} min:{np.min(x)}")
        return self.density(x) * (1 - self.cumdensity(x)) 

    #NOTE 从最近邻计算全局是病态的 需要估计1-Fx 可以通过迭代近似

    @auto_check
    def glbdensity(self, x: np.ndarray):
        """迭代计算全局距离分布"""
        pdf = self.density(x)
        dns = pdf.copy()
        for i in range(12):
            cdf = np.cumsum(dns/np.sum(dns))
            dns =  pdf / (1 - cdf + 1e-2)
            dns[np.isnan(dns)] = 0
            dns[dns > 1e4] = 0
        return dns/np.sum(dns)

    # @auto_check
    # def density(self, x: np.ndarray, glb=False): #输入多元变量X
    #     """返回某个特征出现的概率密度"""
    #     if not glb: 
    #         return self.dist.pdf(x).reshape(-1)
    #     else: 
    #         dns = self.density(np.linspace(same.xmin, same.xmax, same.nbin))  #和same统一
    #         glb_dns = dns / (1 - np.cumsum(dns/np.sum(dns)) + 1.0/same.nbin)**3 
    #         return same(glb_dns / np.sum(glb_dns)).pdf(x).reshape(-1)

    # @auto_check
    # def norm_density(self, x: np.ndarray): #输入多元变量X
    #     """返回某个特征出现的概率密度 x为多维坐标,按照x采样结果进行归一化"""
    #     dns = self.dist.pdf(x).reshape(-1)
    #     return dns / np.sum(dns)

    # @auto_check
    # def density(self, x: np.ndarray, n_indp=1): 
    #     """返回某个特征出现的概率密度(利用n_indp还原min)"""
    #     if n_indp == 1: 
    #         return self.dist.pdf(x).reshape(-1) 
    #     else: 
    #         #NOTE min修正:远程更强 cdf修正:中程更强
    #         #min2修正def 
    #         dns = self.density(np.linspace(same.xmin, same.xmax, same.nbin))  #和same统一
    #         inv_cdf = 1 - np.cumsum(dns/np.sum(dns)) + 1.0/same.nbin  #需要对不同dns进行归一化 否则scipy.stat报错
    #         norm_dns = dns/n_indp * (inv_cdf)**((1-n_indp)/n_indp) 
    #         return same(norm_dns/np.sum(norm_dns)).pdf(x).reshape(-1) #创建新分布
            
    #         #cdf修正 
    #         # dns = self.density(np.linspace(same.xmin, same.xmax, same.nbin))  #和same统一
    #         # cdf = np.cumsum(dns/np.sum(dns)) + 1.0/same.nbin
    #         # return same(dns*cdf).pdf(x).reshape(-1) 

    @auto_check
    def density_map_r(self, R_range=(0.0, 1.0)):
        """Wicksell's problem 修正2维概率密度"""
        """Saltykov formula"""
        x = same.x()
        Ri = x
        r1 = x[:-1]
        r2 = x[1:]
        delta = (x[-1] - x[0])/x.size
        Pi = self.density(x)
        k = np.array([np.nansum(np.sqrt(np.clip(R**2 - r1**2, 0, None)) - np.sqrt(np.clip(R**2 - r2**2, 0, None)))/delta for R in Ri])
        Fi = Pi / (k + 1e-16)
        Fi[np.isclose(k, 0)] = 0
        Fi[:np.searchsorted(x, R_range[0])] = 0
        Fi[np.searchsorted(x, R_range[1]):] = 0
        Fi[np.isnan(Fi)] = 0
        return self.__class__.from_dist(DiscreteDist(Fi).dist)

    @auto_check
    def density_map_d(self):
        #NOTE 同一相内部用min, 只需要最近距离即可
        x = same.x()
        Pi = self.density(x)
        r = [min(r1, r2) for r1 in x for r2 in x]
        p = [p1 * p2 for p1 in Pi for p2 in Pi]
        # p[0] = self.dist.p[0]  # 0距离保持不变

        Fi = np.zeros(Pi.size+1)
        Fi_idx = np.searchsorted(x, r)
        for i, idx_p  in zip(Fi_idx, p): 
            Fi[i] += idx_p
        return self.__class__.from_dist(DiscreteDist(Fi).dist)

    @auto_check
    def copy(self):
        return self.__class__(*self.params)


def dist_error(dist1: DistFunc, dist2: DistFunc):
    """两个distfunc间的误差"""
    dns1 = dist1.density(same.x())
    dns2 = dist2.density(same.x())
    mean = (dns1 + dns2) / 2 + 1e-16

    if np.any(dns1) and np.any(dns2): #不为空密度  #JS_divergence
        return (stat.entropy(dns1, mean) + stat.entropy(dns2, mean))/2
        # return np.sqrt(np.sum((dns1/np.sum(dns1) - dns2/np.sum(dns2))**2)*same.nbin)
    else:
        warnings.warn("dist func error calc warning: empty density")
        return 1.0 

def data_error(data1: np.ndarray, data2: np.ndarray):
    dist1 = DiscreteDist.fit(data1) 
    dist2 = DiscreteDist.fit(data2)
    return dist_error(dist1, dist2)


class MulNormDist(DistFunc):
    xmax = 1.8
    xmin = -1.8
    model = stat.multivariate_normal()._dist
    n_ftr = 3

    def __init__(self, mean=np.zeros(3), cov=np.eye(3)):  #定义分布模型
        self.params = (np.array(mean), np.array(cov))
        self._data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32) #每次拟合只添加 不覆盖
        self.update()

    def dec_key(self) -> tuple:  #只根据params本身判断
        return *tuple([id(param) for param in self.params]), 

    def run_key(self) -> tuple:
        return *tuple([id(param) for param in self.params]), id(self._data)
    
    #TODO 高维cdf近似算法以提高速度
    # @auto_check
    # def nndensity(self, x: np.ndarray):
    #     return self.dist.pdf(x) * (1 - self.dist.cdf(x)) 


class NormDist(DistFunc):
    model = stat.norm().dist

    def __init__(self, loc=0.0, scale=1.0):  #定义分布模型
        self.params = (loc, scale)
        self._data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32) #每次拟合只添加 不覆盖
        self.update()

    def dec_key(self) -> tuple:  #只根据params本身判断
        return *tuple(self.params), 

    def run_key(self) -> tuple:
        return *tuple(self.params), id(self._data)


class UniformDist(DistFunc):
    model = stat.uniform().dist 

    def __init__(self, loc=0.0, scale=1.0):  #定义分布模型
        self.params = (loc, scale)
        self._data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32) #每次拟合只添加 不覆盖
        self.update()

    def dec_key(self) -> tuple:  #只根据params本身判断
        return *tuple(self.params), 

    def run_key(self) -> tuple:
        return *tuple(self.params), id(self._data)
    

class IndependDist(UniformDist):
    """独立分布"""
    xmin = -1.8
    xmax = 1.8
    def __init__(self, loc=-1.0, scale=2.0):  #定义分布模型
        self.params = (loc, scale)
        self._data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32) #每次拟合只添加 不覆盖
        self.update()

class ExponDist(DistFunc):
    model = stat.expon().dist 
    
    def __init__(self, loc=0.0, scale=1.0):  #定义分布模型
        self.params = (loc, scale)
        self._data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32) #每次拟合只添加 不覆盖
        self.update()

    def dec_key(self) -> tuple:
        return *tuple(self.params), 

    def run_key(self) -> tuple:
        return *tuple(self.params), id(self._data)
    
class BetaDist(DistFunc):
    model = stat.beta(2.0, 8.0).dist
    
    def __init__(self, a=2.0, b=8.0, loc=0.0, scale=1.0):  #定义分布模型
        self.params = (a, b, loc, scale)
        self._data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32)
        self.update()

    def dec_key(self) -> tuple:
        return *tuple(self.params), 

    def run_key(self) -> tuple:
        return *tuple(self.params), id(self._data)        


class DiscreteDist(DistFunc):
    model = same 
    
    # @classmethod 
    # def from_dist(cls, dist):
    #     return cls(*cls.model.from_dist(dist).args)

    # @classmethod 
    # def from_data(cls, data):
    #     return cls(*cls.model.from_data(data).args)
    
    @auto_check
    def density_map_r(self, R_range=(0.0, 1.0)):
        """Wicksell's problem 修正2维概率密度"""
        # from scipy.integrate import trapezoid
        
        # x = same.x()
        # f_r = self.density(x)  #半径为0位置不允许有概率密度

        # # cum_integral = np.trapezoid(f_r * x, x)
        # cum_integral = trapezoid(f_r * x, x)  #计算累积积分 用于后续反卷积
        # f_R = f_r / (2 * np.pi * x * (1 - cum_integral) + 1e-8) #反卷积公式计算3D颗粒真实半径概率密度
        # f_R /= np.trapz(f_R, x)  # 对密度函数进行归一化
        # return DiscreteDist(f_R)

        """Saltykov formula"""
        x = same.x()
        Ri = x
        r1 = x[:-1]
        r2 = x[1:]
        delta = (x[-1] - x[0])/x.size
        Pi = self.density(x)
        k = np.array([np.nansum(np.sqrt(np.clip(R**2 - r1**2, 0, None)) - np.sqrt(np.clip(R**2 - r2**2, 0, None)))/delta for R in Ri])
        Fi = Pi / (k + 1e-16)
        Fi[np.isclose(k, 0)] = 0
        Fi[:np.searchsorted(x, R_range[0])] = 0
        Fi[np.searchsorted(x, R_range[1]):] = 0
        Fi[np.isnan(Fi)] = 0
        return DiscreteDist(Fi)

        # x = same.x() + (same.xmax-same.xmin)/same.nbin
        # Pi = self.density(x)
        # Fi = np.zeros(x.size)
        # for j in range(Fi.size):
        #     Fi[j] = np.sum([Pi[j] / (x[j] * np.sqrt(x[j]**2 - x[i]**2)) for i in range(j)])
        # Fi[:np.searchsorted(x, R_range[0])] = 0
        # Fi[np.searchsorted(x, R_range[1]):] = 0
        # Fi[np.isnan(Fi)] = 0
        # return DiscreteDist(Fi)

    @auto_check
    def density_map_d(self):
        # x = same.x()
        # f_d = self.density(x)

        # # 计算3D距离概率密度函数
        # f_D = f_d * x **2
        
        # # 对结果进行归一化
        # f_D /= np.trapz(f_D, x)
        # return DiscreteDist(f_D)

        #NOTE 同一相内部用min, 只需要最近距离即可
        #NOTE 不同相应当用欧式距离
        # r = [np.sqrt((r1**2 + r2**2)/2) for r1 in self.dist.x() for r2 in self.dist.x()]
        r = [min(r1, r2) for r1 in self.dist.x() for r2 in self.dist.x()]
        # p = [(p1 * p2)**2 for p1 in self.dist.p for p2 in self.dist.p]
        p = [p1 * p2 for p1 in self.dist.p for p2 in self.dist.p]
        # p[0] = self.dist.p[0]  # 0距离保持不变
        pn = np.zeros(self.dist.p.size+1)
        pn_idx = np.searchsorted(self.dist.x(), r)
        for i, idx_p  in zip(pn_idx, p): 
            pn[i] += idx_p
        return DiscreteDist(pn[:-1])

    # @auto_check
    # def density_map_d(self):
    #     x = same.x()
    #     Pi = self.density(x)
    #     Ri = x
    #     r1 = x[:-1]
    #     r2 = x[1:]
    #     delta = (x[-1] - x[0])/x.size
    #     k = np.array([np.nansum(np.sqrt(np.clip(R**2 - r1**2, 0, None)) - np.sqrt(np.clip(R**2 - r2**2, 0, None)))/delta for R in Ri])
    #     Fi = Pi / (k + 1e-16)
    #     Fi[np.isclose(k, 0)] = 0
    #     return self.__class__.from_dist(DiscreteDist(Fi).dist)
        

    @auto_check
    def rescale(self, scale):
        """分布按照尺度缩放"""
        # rs_dist = self.from_data(self.sample(10000) * scale)
        # self.params = rs_dist.params
        return self.from_data(self.sample(10000) * scale)


class SmoothDist(DiscreteDist):
    model = smooth



class CompDist(DistFunc):
    """三向复合分布, 支持假设沿x,y,z正方向的各向异性"""

    def __init__(self, dfs = [UniformDist(-1.0, 2.0)]*3):  #定义分布模型
        self.dfs = dfs
        self.n_ftr = len(dfs)
        self._data = np.zeros((0, self.n_ftr))
        self.update()

    def __str__(self) -> str:
        return f"{self.cls_str}" + "(" + ' '.join([df.__str__() for df in self.dfs]) + ")"
    
    def __repr__(self) -> str:
        return self.__str__() 


    def update(self):
        for df in self.dfs: df.update()
        Link.update(self)

    def dec_key(self) -> tuple:
        return *tuple([df.dec_key() for df in self.dfs]), 

    def run_key(self) -> tuple:
        return *tuple([df.run_key() for df in self.dfs]), 
    
    @property
    @auto_check
    def params(self):
        return tuple([df.params for df in self.dfs])
    
    @params.setter
    def params(self, value):
        for i, df in enumerate(self.dfs): 
            df.params = value[i]

    @auto_check
    def update_params(self, params):
        for df, param in zip(self.dfs, params):
            df.params = param
            df.dist = df.model(*param)
            Link.update(df)

    @auto_check
    def __getstate__(self):
        return {"params": self.params, "dfs": self.dfs, "n_ftr": self.n_ftr}
        
    def __setstate__(self, state):
        self.dfs = list(state["dfs"])
        self.n_ftr = len(self.dfs)
        self.params = tuple(state["params"])
        self._data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32)
        self.update()

    @classmethod 
    def from_dist(cls, dist):
        return cls([dist] * cls.n_ftr)

    @classmethod 
    def from_data(cls, data):
        return cls([DiscreteDist.from_data(data[:, i:i+1]) for i in range(cls.n_ftr)])
    
    @classmethod 
    def from_pcdt(cls, dists: DistVol, ndim=1):
        return cls([DiscreteDist.from_pcdt(dists[:, i:i+1], ndim) for i in range(cls.n_ftr)]) 
    
    @classmethod 
    def from_pcpnt(cls, pnt1: ObjectPnt, pnt2: ObjectPnt=None, ftr_func=stdpcf):
        #TODO 各向异性两点相关
        raise NotImplemented()


    #@auto_check  #不检查 在所有数据添加完成后更新
    def fit(self, data: np.ndarray):  #拟合数据 获取分布参数
        """仅保存数据,data的id随之改变,在update内更新"""
        self._data = np.append(self._data, data.reshape(-1, self.n_ftr), axis=0)  #保持维度一致 

    @auto_check
    def sample(self, n_sample=1): #采样 获得数据实例
        """采样 返回特征值 不缓存结果"""
        return np.concatenate([df.sample(n_sample)[:, None] for df in self.dfs], axis=1)
    
    @auto_check
    def density(self, x: np.ndarray): #输入多元变量X
        """返回某个特征出现的概率密度"""
        return np.prod([df.density(x[:, i:i+1]) for i, df in enumerate(self.dfs)], axis=0)

    @auto_check
    def cumdensity(self, x: np.ndarray): 
        """返回某个特征出现的分部函数"""
        #将独立分布的分布函数相乘
        return np.prod([df.cumdensity(x[:, i:i+1]) for i, df in enumerate(self.dfs)], axis=0)

    @property
    def xmin(self):
        return min([df.xmin for df in self.dfs])
    @property
    def xmax(self):
        return max([df.xmax for df in self.dfs])
    @xmin.setter
    def xmin(self, value):
        for df in self.dfs:
            df.xmin = value
    @xmax.setter
    def xmax(self, value):
        for df in self.dfs:
            df.xmax = value

def __const_update(self):
    """不拟合数据的update, 但仍会根据params更新"""
    self.dist = self.model(*self.params)
    self.data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32)
    Link.update(self) 

ConstDistFunc = type("ConstDistFunc", (DistFunc, ), {"update": __const_update})
ConstNormDist = type("ConstNormDist", (NormDist, ConstDistFunc), {"update": __const_update})
ConstUniformDist = type("ConstUniformDist", (UniformDist, ConstDistFunc), {"update": __const_update})
ConstIndependDist = type("ConstIndependDist", (IndependDist, ConstDistFunc), {"update": __const_update})
ConstExponDist = type("ConstExponDist", (ExponDist, ConstDistFunc), {"update": __const_update})
ConstDiscreteDist = type("ConstDiscreteDist", (DiscreteDist, ConstDistFunc), {"update": __const_update})
ConstSmoothDist = type("ConstSmoothDist", (SmoothDist, ConstDistFunc), {"update": __const_update})
ConstMulNormDist = type("ConstMulNormDist", (MulNormDist, ConstDistFunc), {"update": __const_update})
ConstCompDist = type("ConstCompDist", (CompDist, ConstDistFunc), {"update": __const_update})

#预声明分布函数
cst = ConstUniformDist()
lin = ConstDiscreteDist()  #线性分布 优先选取远距离
lin.params = tuple([np.linspace(0.0, 1.0, lin.model.nbin)**2, ])
idp = ConstIndependDist() 

#-------------------------------------------------------------------------------------#
# 材料域
#-------------------------------------------------------------------------------------#

class Region(Node):  
    """材料域 允许周期性边界"""
    # min_nvxl = np.inf  #关闭自适应
    min_nvxl = 8     #NOTE 存在物体离散误差问题, 低分辨率导致物体误差较大. 且隐式修改离散分辨率不利于控制. 不启用
    vlz = Voxelizer(60, 3)

    def __init__(self, res: float=1.0, size: list[float]=[80.0, 80.0, 80.0], periods: list[bool] = [False, False, False]):
        """初始化域:
        Args: 
            res: 体素边长(分辨率)
            size: 域实际大小(边长大小)
        """
        if res > np.min(size): 
            warnings.warn(f"init region wraning: resolution {res} larger than minium size {np.min(size)}, auto fixed")
        # self.res = min(res, np.min(size))   #分辨率不允许大于size最小大小
        self.res = res 
        assert(np.size(size) == 3)
        assert(np.size(periods) == 3)
        self.size = list(np.clip(size, self.res, np.inf))  #分辨率不允许大于size最小大小
        self.periods = periods
        self.update()

    def dec_key(self) -> tuple:
        return self.res, tuple(self.size), self.ndim, tuple(self.periods)
    
    def __str__(self):
        return f"{self.cls_str}(rs:{self.res:.4} sz:{self.size} pd:{np.array(self.periods, dtype=np.int32)})"
    
    def __getstate__(self):
        return {"res": self.res, "size": self.size, "periods": self.periods, "ndim": self.ndim}

    def update(self):
        shape = (np.array(self.size) / self.res).astype(np.int32)
        self.res = self.size[0] / shape[0] 
        Node.update(self)

    @property
    def ndim(self) -> int:
        return np.count_nonzero((np.array(self.size)/self.res)>1.001)

    @property
    def len(self) -> float:
        return np.max(self.size)
    @property 
    def volume(self) -> float:
        return np.prod(self.size)
    @property
    def vxl_len(self):
        return np.max(self.shape)
    @property
    def vxl_unit(self):
        """体素单位长度"""
        return 1/self.vxl_len
    #@auto_memo  #必须用memo 保证后续访问同一向量 list输出缓存
    @property
    def shape(self) -> list:
        """"""
        return (np.array(self.size)/self.res).astype(np.int32).tolist()
        # return np.clip((np.array(self.size)/self.res), 1, None).astype(np.int32).tolist()
    @property
    def limits(self) -> tuple:
        uni_size = np.array(self.shape) / self.vxl_len
        return ((0, uni_size[0]), (0, uni_size[1]), (0, uni_size[2]))

    def rescale(self, fac: float) -> Region:
        # size = self.size
        # if self.ndim == 2: size[2] = self.res * fac
        return self.__class__(self.res * fac, self.size, self.periods)

    # def reshape(self, vxl_len: int) -> Region:
    #     return Region(self.len / vxl_len, self.size, self.periods)

    def adapt(self, rad) -> Region:
        """自适应域分辨率 同一个SCC内共用reg 需要所有源可随reg变化"""
        # if not seeds.cnt:
        #     warnings.warn(f"adapt reg failed: {seeds.__str__()} is emtpy")
        #     return self 
        min_vxl = np.min(rad) * self.vxl_len
        adp_res = self.res * min_vxl / self.min_nvxl
        if min_vxl > self.min_nvxl and adp_res <= np.min(self.size[:self.ndim]):
            return self.rescale(min_vxl / self.min_nvxl)
        else:
            return self 
    
    @auto_check
    def copy(self):
        return self.__class__(self.res, self.size, self.periods)


class MSPRegion(Region):
    @property
    def ndim(self) -> int:
        return 1

class Region2D(Region):  
    vlz = Voxelizer(60, 2)
    @property
    def ndim(self) -> int:
        return 2

    def __init__(self, res: float=1.0, size_2d: list[float]=[80.0, 80.0], periods_2d: list[bool] = [False, False]):
        """初始化域:
        Args: 
            res: 体素边长(分辨率)
            size: 域实际大小(边长大小)
        """
        self.res = res 
        self.size_2d = size_2d[:2]
        self.periods_2d = periods_2d[:2]
        self.update()

    @property 
    def size(self):
        return list(self.size_2d[:2]) + [self.res]
    
    @size.setter
    def size(self, value): 
        self.size_2d = list(value[:2]) 
        self.res = value[-1]

    @property
    def periods(self): 
        return list(self.periods_2d[:2]) + [False]

    @periods.setter
    def periods(self, value): 
        self.periods_2d = list(value[:2]) 

    def __str__(self):
        return f"{self.cls_str}(rs:{self.res:.4} sz:{self.size[:2]} pd:{np.array(self.periods, dtype=np.int32)[:2]})"

    @auto_check
    def copy(self):
        return self.__class__(self.res, self.size_2d, self.periods_2d)
    
#-------------------------------------------------------------------------------------#
# 批量划分器基类 节点内不存储数据
#-------------------------------------------------------------------------------------#

class Splitor(Node):
    def __init__(self, *args):
        """种子引用划分器"""
        super().__init__(*args)   

    
    
#-------------------------------------------------------------------------------------#
# 半径批量划分
#-------------------------------------------------------------------------------------#

def rths_mod(get_rths):
    def mod_get_rths(self, seeds, nbat, *args, **kwargs):
        if not seeds.cnt: 
            return np.concatenate([[np.inf], np.full(nbat-1, 0), [-np.inf]]).tolist() 
        ret = get_rths(seeds, nbat, *args, **kwargs)
        return list(ret) if not isinstance(ret, list) else ret 


class RadSplitor(Splitor):
    def __init__(self, *args):
        """配置默认半径批量分配器"""
        super().__init__(*args)
    
    def get_rths(self, seeds, n_batch, ndim=3):
        """均匀切分半径阈值,得到n_batch个批量,给出n_batch-1个阈值"""
        if not seeds.cnt: 
            return np.concatenate([[np.inf], np.full(n_batch-1, 0), [-np.inf]]).tolist() 
        linear_ths = np.linspace(np.min(seeds.r), np.max(seeds.r), n_batch+1)  #从最大最小值范围之外开始
        return list(np.concatenate([[np.inf], linear_ths[1:-1][::-1], [-np.inf]])) 
    

class MulOtsuRadSplitor(RadSplitor):
    """MultiOTSU半径批量分配器
    """
    def __init__(self, n_rad_bin = 200):
        """配置MultiOTSU半径批量分配器
        Args:
            n_rad_bin (int): 半径划分区间数目
        """
        from skimage.filters.thresholding import threshold_multiotsu

        self.n_rad_bin = n_rad_bin
        self.multi_otsu = threshold_multiotsu
        super().__init__(n_rad_bin)
        
    def dec_key(self) -> tuple:
        return self.n_rad_bin,
    
    def hist_split(self, hist: np.ndarray, n_batch: int) -> np.ndarray:
        """按照直方图给出划分 时间复杂度随批量数量增加很快 当批量数目大于4时将会在子区间上递归调用该算法进行划分"""
        #当批量数大于数据数时 会给出空批量
        hist_cnt = np.count_nonzero(hist)  #密度非0数目
        # if hist_cnt == 0: 
        if n_batch <= 1:
            return np.array([], dtype=np.int32)  #返回空索引表
        elif n_batch <= 3: #批量数目较小 且小于数据分布数目 直接调用
            if hist_cnt >= n_batch:
                return self.multi_otsu(classes=n_batch, hist=hist)
            #允许划分给出空批量 给出多个相同edge
            elif hist_cnt >= 2:  #数据分布数目小于需要划分的批量数 重复给出划分最小边界值 重复分配边界索引0
                edge_idxs = self.multi_otsu(classes=hist_cnt, hist=hist)
                return np.concatenate([np.repeat(0, n_batch - hist_cnt), edge_idxs]) 
            else: 
                #hist_cnt为1时 返回的edge_idx为空
                return np.repeat(0, n_batch - hist_cnt) 
        else:  #在分为两部分的子区间上递归调用
            if hist_cnt >= 2: #能够分为两部分时
                edge_idx = self.multi_otsu(classes=2, hist=hist)[0] + 1
                edge_idxs0 = self.hist_split(hist[:edge_idx], n_batch//2) #子区间调用结果
                edge_idxs1 = self.hist_split(hist[edge_idx:], n_batch - n_batch//2)
                return np.concatenate([edge_idxs0, [edge_idx], edge_idxs1 + edge_idx])  #返回全局索引
            else:  #不能分为两部分 返回全0索引
                return np.repeat(0, n_batch - hist_cnt) 

    def get_rths(self, seeds: Seeds, nbat: int, reg=Region(1, [120, 120, 120], [False, False, False])) -> np.ndarray:
        """在半径的体积加权后,按照多阈值OTSU算法给出当前种子半径阈值
        注意:当种子半径分布过于集中时,可能无法给出合理结果
        """
        if not seeds.cnt: 
            return np.concatenate([[np.inf], np.full(nbat-1, 0), [-np.inf]]).tolist() 
        r_range = np.max(seeds.r) - np.min(seeds.r)
        bin_range = r_range / self.n_rad_bin
        hist_range = (np.min(seeds.r) - bin_range, np.max(seeds.r) + bin_range)  #补充边界范围,从最大最小值之外开始
        hist, edges = np.histogram(seeds.r, self.n_rad_bin, weights=seeds.discrete_counts(reg), range=hist_range)   #按照体积加权 
        #当批量数目过多时 递归划分
        edge_idxs = self.hist_split(hist, nbat)  #从加权直方图划分 
        return np.concatenate([[np.inf], edges[edge_idxs][::-1], [-np.inf]]).tolist() 
    
    def __str__(self):
        return f"{self.cls_str}(nbin:{self.n_rad_bin})"

class UniRadSplitor(RadSplitor):
    """均匀分割种子数目,并且按照半径排序. 最后一个批量将补齐.
    当n_batch大于种子数目时,将补充种子最小半径到最后(重新排序前的最前).
    """
    def __init__(self):
        super().__init__()
    
    def get_rths(self, seeds:Seeds, n_batch: int, ndim=3):
        """均匀分割种子数目
        """
        if not seeds.cnt: 
            return np.concatenate([[np.inf], np.full(n_batch-1, 0), [-np.inf]]).tolist() 
        if n_batch > seeds.cnt:
            uni_bat_cnt = 1
            sorted_r = np.ones(n_batch) * np.min(seeds.r)
            sorted_r[-seeds.cnt:] = np.sort(seeds.r)  #有序部分
        else:
            uni_bat_cnt = seeds.cnt // n_batch
            sorted_r = np.sort(seeds.r)
        min_ths = sorted_r[np.arange(0, uni_bat_cnt*(n_batch-1)-1, uni_bat_cnt)]
        max_ths = sorted_r[np.arange(1, uni_bat_cnt*(n_batch-1), uni_bat_cnt)]
        uni_ths = (min_ths + max_ths)/2
        return np.concatenate([[np.inf], uni_ths[::-1], [-np.inf]]).tolist()  #范围从大到小
    
    
#-------------------------------------------------------------------------------------#
# 小批量划分
#-------------------------------------------------------------------------------------#

class RndSplitor(Splitor):
    def __init__(self, prop0=0.2, propn=0.04, prop_ini=0.0):
        """默认线性衰减"""
        self.prop0 = max(prop0, propn)
        self.propn = min(prop0, propn)
        self.prop_ini = prop_ini
        Splitor.__init__(self, prop0, propn)
    
    def dec_key(self) -> tuple:
        return self.prop0, self.propn, self.prop_ini
    
    def get_prop(self, seeds: Seeds, n_it: int, cur_it: int) -> float:
        """获取分割后的种子"""
        if n_it == 0: return max(self.prop0, self.prop_ini)
        if cur_it == 0: return max(self.prop0, self.prop_ini)
        return self.prop0*(1 - cur_it/n_it) + self.propn*cur_it/n_it
    
    def __str__(self):
        return f"{self.cls_str}(prop:{self.prop_ini}|{self.prop0}~{self.propn})"


class ExpSplitor(RndSplitor):
    def __init__(self, prop0=0.32, propn=0.04, prop_ini=0.0, fac=2.0):
        """指数衰减 默认第一次迭代比例为1.0(1.0影响随机初始化, 导致种子过于集中)"""
        self.fac = fac
        RndSplitor.__init__(self, prop0, propn, prop_ini)
    
    def get_prop(self, seeds: Seeds, n_it: int, cur_it: int) -> float:
        """获取分割后的种子
        Example:
            plt.plot([ExpRndSplitor().get_prop(Seeds(), 40, i) for i in range(40)]) 
        """
        if n_it == 0: return max(self.prop0, self.prop_ini)
        if cur_it == 0: return max(self.prop0, self.prop_ini)
        return (self.prop0 - self.propn) * (10 ** self.fac) ** (-cur_it/n_it) + self.propn
    

class PropSplitor(RndSplitor):
    def __init__(self, prop0=0.2, prop_ini=0.0):
        """固定比例划分 prop0和propn相等"""
        RndSplitor.__init__(self, prop0, prop0, prop_ini)

    def get_prop(self, seeds: Seeds, n_it: int, cur_it: int) -> float:
        """获取分割后的种子"""
        return self.prop0

    def __str__(self):
        return f"{self.cls_str}(prop:{self.prop_ini}|{self.prop0})"

#-------------------------------------------------------------------------------------#
# 物体 
#-------------------------------------------------------------------------------------#


class Object(Node):
    """物体采样点, 点相关图顶点基类"""
    # _r = np.inf 
    _ndim = 3
    # cen_r = 1.2 
    
    def __init__(self, mark=1):
        """空物体"""
        self.mark = mark
        self.update() 

    def dec_key(self) -> tuple:
        return self._ndim, 

    def discrete(self, reg: Region) -> PhaseVol:
        return PhaseVol.zeros(reg.shape) 
    
    # def discrete_center(self, reg: Region) -> PhaseVol:
    #     copy = self.copy()
    #     copy.r = np.full(np.size(copy.r), self.cen_r)/reg.vxl_len if np.size(copy.r) > 1 else self.cen_r
    #     return copy.discrete(reg)
    
    #种子:0->p^2(与到种子中心距离有关) 平面:2->常数 线条:1->p^1 立体图形:3->0
    @property
    def ndim(self) -> int:
        return self._ndim 

    @ndim.setter 
    def ndim(self, value): 
        self._ndim = value 

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, value) -> float:
        self._r = value 

    def __str__(self):
        return f"{self.cls_str} mark:{self.mark}" 
    
    def __repr__(self):
        return self.__str__()

    def copy(self):
        import copy 
        return copy.deepcopy(self) 

    @property 
    def subs(self):
        """基本物体 保持顺序"""
        return OrderSet([self])

    def __add__(self, obj2):
        return CompObject([self, obj2])
    
    @property 
    def cnt(self):
        return 1 

non_obj = Object(0)

#-------------------------------------------------------------------------------------#
# 种子集合 
#-------------------------------------------------------------------------------------#

    
class Seeds(Object):
    """种子集合模型,提供半径采样函数"""
    #数据域声明
    n_data = 10
    data_dict = {"x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5, "b": 6, "a": 7, "r": 8, "mark": 9, "loc": slice(0, 3), "ort": slice(3, 6)}
    data_str_tab = ["x", "y", "z", "vx", "vy", "vz", "b", "a", "r", "mark"]
    _ndim = 0 
    _r = 0 
    n_circle = 4.2

    def __init__(self, bdata: np.ndarray=np.array([[]]), idx: np.ndarray=np.array([])):
        if not isinstance(bdata, np.ndarray): bdata = np.array(bdata) 
        self.bdata = np.zeros((0, self.__class__.n_data)) if not bdata.size else bdata  #保证共用数据
        self.idx = np.array(idx, dtype=np.int32)
        self.update()

    @classmethod
    def from_info(cls, loc: np.ndarray, r=0.0, mark=1, ort=np.zeros(3), b=0.0, a=0.0):
        loc = np.array(loc)
        seeds = cls(np.zeros([loc.shape[0], cls.n_data]), np.arange(loc.shape[0]))
        seeds.loc = loc 
        seeds.r = r
        seeds.mark = mark 
        seeds.ort = ort 
        seeds.b = b
        seeds.a = a
        return seeds 
        
    @classmethod
    def from_msp_seeds(cls, msp_seeds):
        seeds = cls(np.zeros([len(msp_seeds), cls.n_data]), np.arange(len(msp_seeds)))
        for idx, seed in enumerate(msp_seeds.seeds):
            seeds.loc = seed.position
            seeds.r = seed.geometry.r
            seeds.mark = seed.phase + 1 #本程序中从1开始 
        return seeds

    @classmethod 
    def from_seeds(cls, value: Seeds):
        seeds = cls(np.zeros((value.cnt, cls.n_data)), np.arange(value.cnt))
        seeds.loc = value.loc 
        seeds.r = value.r 
        seeds.mark = value.mark
        seeds.a = value.a
        seeds.b = value.b 
        seeds.ort = value.ort
        return seeds 
        
    def __setstate__(self, state):
        try:
            bdata = np.array(state["bdata"])
        except KeyError:
            bdata = np.array(state["dat"])

        if bdata.shape[1] == 5:  #兼容老版本Seeds格式
            self.bdata = np.zeros([bdata.shape[0], self.n_data])
            self.bdata[:, :3] = bdata[:, :3]
            self.bdata[:, 8] = bdata[:, 3]
            self.bdata[:, 9] = bdata[:, 4]
        else:
            self.bdata = bdata

        self.idx = np.array(state["idx"], dtype=np.int32)
        self.mod_time()
        self.update()

    def update(self):
        self.mod_time()
        Node.update(self)
        
    def mod_time(self):
        self._mod_time = time.time() 

    def dec_key(self) -> tuple:
        return id(self.bdata), id(self.idx)
    
    def run_key(self) -> tuple:
        return id(self.bdata), id(self.idx), self._mod_time

    def __str__(self):
        return f"{self.cls_str}(adrs:{hex_id(self.bdata)} cnt:{self.cnt})"

    def __iter__(self):
        self.cur_idx = -1 
        return self 

    def __next__(self):
        self.cur_idx += 1 
        if self.cur_idx >= self.cnt:
            raise StopIteration
        return Seeds(self.bdata, self.idx[self.cur_idx:self.cur_idx+1])  #返回共用数据

    @auto_check  #检查更新 避免复制过期的引用
    def copy(self) -> Seeds:
        """返回单纯的种子复制 不包括引用关系"""
        return Seeds(np.copy(self.bdata), np.copy(self.idx))  #注意idx和data都需要复制

    @auto_check 
    def mark_map(self, map) -> Seeds:
        """属性映射"""
        seeds = self.copy()
        seeds.mark = np.array(map)[seeds.mark]
        return seeds 

    #半径阈值, 随机掩膜 决定 半径批量和随机批量
    #问题本质在于确定种子的坐标 Seeds顶点存储重建/提取的运行状态 
    @property
    def void(self):
        return np.ndarray((0, self.n_data))
    @property
    def void_seeds(self):
        return Seeds(self.bdata, np.array([]))
    @property
    @auto_check
    def data(self) -> np.ndarray:
        if self.idx.size and self.bdata.size: 
            return self.bdata[self.idx]
        else:
            return self.void  
    @property  #不缓存 及时更新 
    @auto_check
    def loc(self) -> np.ndarray:
        return self[:, "loc"]
    @property 
    @auto_check
    def x(self) -> np.ndarray:
        return self[:, "x"]
    @property 
    @auto_check
    def y(self) -> np.ndarray:
        return self[:, "y"]
    @property 
    @auto_check
    def z(self) -> np.ndarray:
        return self[:, "z"]
    @property 
    @auto_check
    def r(self) -> np.ndarray:
        """返回球形种子的半径, 椭球种子较短的轴"""
        #判断种子类型 可能复合
        seeds = Seeds(self.data, np.arange(self.cnt))
        rads = np.zeros(seeds.cnt)
        sphs = seeds.mask_ref(seeds[:, "r"]!=0)
        ells = seeds.mask_ref(seeds[:, "a"]!=0)
        
        #球形种子
        if sphs.cnt:
            rads[sphs.idx] = sphs[:, "r"]
        #椭球种子
        if ells.cnt:
            rads[ells.idx] = np.min([ells[:, "b"], ells[:, "a"]], axis=0)

        # 
        # if self.cnt and np.any(self[:, "r"]): 
        #     return self[:, "r"]
        # 
        # elif self.cnt and np.any(self.b) and np.any(self.a):
        #     return np.min([self.b, self.a], axis=0)
        # else:
        #     return 0.0 

        if rads.size:
            return rads
        else:
            return 0.0 

    @property
    @auto_check
    def r_median(self) -> float:
        if self.cnt and np.any(self.r):
            return np.median(self.r) 
        else:
            return 0.0 
    @property
    @auto_check
    def mark(self) -> np.ndarray:
        return self[:, "mark"].astype(np.int32)

    @property  #不缓存 及时更新 
    @auto_check
    def ort(self) -> np.ndarray:
        return self[:, "ort"]
    @property 
    @auto_check
    def vx(self) -> np.ndarray:
        return self[:, "vx"]
    @property 
    @auto_check
    def vy(self) -> np.ndarray:
        return self[:, "vy"]
    @property 
    @auto_check
    def vz(self) -> np.ndarray:
        return self[:, "vz"]
    @property 
    @auto_check
    def b(self) -> np.ndarray:
        return self[:, "b"]
    @property 
    @auto_check
    def a(self) -> np.ndarray:
        return self[:, "a"]

    @property 
    @auto_check
    def aabb(self):
        """边界盒子坐标:(种子,维度,最大最小坐标)"""
        min_pos = self.loc - self.r[:, None]
        max_pos = self.loc + self.r[:, None]
        aabb_pos = np.swapaxes(np.swapaxes(np.array([min_pos, max_pos]), 0, 2), 0, 1)
        return aabb_pos
    @property 
    @auto_check
    def cnt(self):
        if not self.bdata.size:
            return 0 
        return self.idx.size 
    @property
    @auto_check
    def mask(self):
        mask = np.zeros(self.bdata.shape[0] ,dtype=np.bool_)
        if self.idx.size:
            mask[self.idx] = 1 
        return mask 
    # @auto_check
    # def volume(self, reg: Region):
    #     return np.sum(4/3*np.pi * (self.r * reg.len)**3)
    
    @loc.setter
    def loc(self, value: np.ndarray):
        self[:, "loc"] = value
        self.mod_time()
    @x.setter
    def x(self, value: np.ndarray):
        self[:, "x"] = value
        self.mod_time()
    @y.setter
    def y(self, value: np.ndarray):
        self[:, "y"] = value
        self.mod_time()
    @z.setter
    def z(self, value: np.ndarray):
        self[:, "z"] = value
        self.mod_time()
    @r.setter
    def r(self, value):
        self[:, "r"] = value
        self.mod_time()
    @mark.setter
    def mark(self, value):
        self[:, "mark"] = value
        self.mod_time()
    @data.setter 
    def data(self, value):
        """注意seeds.data[:] = value将不会进行赋值"""
        self.bdata[self.idx] = value 
        self.mod_time()
    @ort.setter
    def ort(self, value: np.ndarray):
        self[:, "ort"] = value
        self.mod_time()
    @vx.setter
    def vx(self, value: np.ndarray):
        self[:, "vx"] = value
        self.mod_time()
    @vy.setter
    def vy(self, value: np.ndarray):
        self[:, "vy"] = value
        self.mod_time()
    @vz.setter
    def vz(self, value: np.ndarray):
        self[:, "vz"] = value
        self.mod_time()
    @b.setter
    def b(self, value: np.ndarray):
        self[:, "b"] = value
        self.mod_time()
    @a.setter
    def a(self, value: np.ndarray):
        self[:, "a"] = value
        self.mod_time()
    @auto_check
    def __getitem__(self, key) -> np.ndarray:
        """索引获取种子列表子集,返回相同底层数据,不同索引的种子.
        key的第一个维度对idx索引,第二个维度对data的列进行索引,返回复制.
        """
        if isinstance(key, (list, tuple)):
            if isinstance(key[0], Phase):
                idx = self.idx[self.mark==key[0].mark]  
            else: 
                idx = self.idx[key[0]] 
            if key[1] in self.data_dict.keys():
                key1 = self.data_dict[key[1]]  
            else:
                key1 = key[1]
            if not idx.size:  #idx为空 返回正确维度的空值
                return self.void[:, key1]
            return self.bdata[idx, key1] 
        #访问种子集合
        elif isinstance(key, (slice, np.ndarray)):
            return Seeds(self.bdata, self.idx[key])  #索引引用:切片/花式
        elif isinstance(key, int):
            return Seeds(self.bdata, self.idx[key:key+1])
        elif isinstance(key, Phase):
            return self.phase_ref(key)
        else: 
            warnings.warn(f"{self.class_id_str} getitem warning: undefined seeds index method")
            return self.bdata[key]
            
    @auto_check
    def __setitem__(self, key, value):
        """设置种子列表子集,允许输入键来赋值特定数据域
        Example:
            s1[0] = [0,1,2,1,0]
            s1[:,"r"] = 10
            s2[:,"phase"] = 1 
        """
        if not self.data.size:  #数据为空 跳过赋值
            return 
        #NOTE 在外部手动mod time setitem不会视作状态更新
        
        #部分索引引用数据域赋值
        elif isinstance(key, (list, tuple)):  
            if isinstance(key[0], Phase):
                return self.phase_ref(key[0]).__setitem__((slice(None), key[1]), value) #无自动更新的复制 
            
            # elif isinstance(key[0], (slice, np.ndarray)):
            idx = self.idx[key[0]] 
            key1 = self.data_dict.get(key[1], key[1]) #查询字典或直接使用key[1]
            if idx.size: self.bdata[idx, key1] = value   #若idx不为空 赋值
            
                
        #相索引赋值
        elif isinstance(key, Phase):
            return self.phase_ref(key).__setitem__(slice(None), value) 
        # elif isinstance(key, (slice, np.ndarray)):
        #     if self.idx[key].size: self._data[self.idx[key], :] = value   #全data赋值

        #种子间赋值 或行数据域赋值
        else:
            # warnings.warn(f"{self.class_id_str} setitem warning: undefined seeds index method")
            self.bdata[self.idx[key], :] = value.bdata[value.idx, :] if isinstance(value, Seeds) else value

    @auto_check 
    def __add__(self, seeds: Seeds) -> Seeds:
        """种子集合求和,重新分配底层数据,提供索引引用"""
        #支持空种子相加 共用_data对象
        if self.bdata.size == 0:
            self = seeds 
            return self 
        if id(self.bdata) != id(seeds.bdata):
            seeds.idx += self.bdata.shape[0]

            add_data = np.concatenate([self.bdata, seeds.bdata], axis=0)
            add_idx = np.concatenate([self.idx, seeds.idx])

            seeds.bdata = add_data 
            self.bdata = add_data 
            add_seeds = Seeds(add_data, add_idx)
        else:   #底层数据相同 直接相加
            add_seeds = Seeds(self.bdata, np.concatenate([self.idx, seeds.idx]))

        #域处理
        if "reg" in self.__dict__: 
            add_seeds.reg = self.__dict__["reg"]
        elif "reg" in seeds.__dict__: 
            add_seeds.reg = seeds.__dict__["reg"]

        return add_seeds 
        
    @auto_check
    def __iadd__(self, seeds: Seeds) -> Seeds:
        if self.bdata.size == 0:
            self = seeds 
            return self 
        elif id(self.bdata) != id(seeds.bdata):
            seeds.idx += self.bdata.shape[0]

            add_data = np.concatenate([self.bdata, seeds.bdata], axis=0)
            add_idx = np.concatenate([self.idx, seeds.idx])

            seeds.bdata = add_data 
            self.bdata = add_data 
            self.idx = add_idx 
        else: 
            self.idx = np.concatenate([self.idx, seeds.idx])

        #域处理
        if "reg" in seeds.__dict__ and ("reg" not in self.__dict__): 
            self.reg = seeds.__dict__["reg"]
        return self 
        
    @auto_check
    def __and__(self, seeds: Seeds) -> Seeds:
        """在两个种子集合共用底层数据前提下, 求其共用种子引用"""
        if id(self.bdata) == id(seeds.bdata): 
            return self.data_mask_ref(np.bitwise_and(self.mask, seeds.mask))
        else: 
            raise RuntimeError(f"{self.__str__()} and {seeds.__str__()} do not share the bdata")
        
    @auto_check
    def __or__(self, seeds: Seeds) -> Seeds:
        if id(self.bdata) == id(seeds.bdata): 
            return self.data_mask_ref(np.bitwise_or(self.mask, seeds.mask))
        else: 
            raise RuntimeError(f"{self.__str__()} and {seeds.__str__()} do not share the bdata")
    
    @auto_check
    def __invert__(self) -> Seeds:
        return self.data_mask_ref(np.bitwise_not(self.mask)) 
    
    @auto_check
    def __sub__(self, seeds: Seeds) -> Seeds:
        return self.__and__(seeds.__invert__())  #自身与seeds的反的交集

    @auto_check
    def __getstate__(self):
        return {"bdata": self.bdata, "idx": self.idx}
    
    def __move(self, reg: Region, delta=1.2):
        """位置移动 相对上限/绝对上限(按照种子半径决定补偿) """
        dirc = np.random.choice([1, -1], self.loc.shape)
        delta = np.random.exponential(self.r_median * delta, self.loc.shape)
        self.loc += delta * dirc  #种子标志同时更新
        self.loc %= (reg.size / reg.len) #限制坐标位置
        return self 
    
    @auto_check
    def move(self, objpnt: ObjectPnt, delta=1.2, natp=1000):
        """位置移动 返回不重叠的种子坐标"""
        for i in range(natp):
            move_seeds = self.copy().__move(objpnt.reg, delta)
            if not objpnt.vol.query(move_seeds.loc): break 
        self.loc = move_seeds.loc 
        return self 
            
    @auto_check
    def mask_ref(self, mask: np.ndarray) -> Seeds:
        """在当前索引基础上 创建索引引用种子(子集)"""
        return Seeds(self.bdata, self.idx[mask])  #mask和idx可同时为空

    @auto_check
    def idx_ref(self, idx: np.ndarray) -> Seeds:
        if self.idx.size: 
            return Seeds(self.bdata, self.idx[idx])
        else:
            return self.void_seeds

    @auto_check
    def cut_ref(self, axis: int, rpos: float) ->Seeds:
        mask = np.abs(self.loc[:, axis] - rpos) < self.r
        return self.mask_ref(mask)

    # @auto_check 
    # def remove_data_idx(self, data_idx: np.ndarray) -> Seeds: 
    #     """从现有索引中移除指定索引
    #     Examples:
    #     s1 = Seeds(np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]), np.array([0]))
    #     s2 = Seeds(s1._data, np.array([1]))
    #     s1 += s2 
    #     s1.remove_data_idx(np.array([2])).data
    #     """
    #     if not data_idx.size: 
    #         return self 
    #     mask = np.zeros(self._data.shape[0], dtype=np.bool_)  #最大索引范围
    #     mask[self.idx] = 1 #自身索引位置保留 
    #     mask[data_idx] = 0  #输入索引位置删除
    #     return self.data_mask_ref(mask)

    @auto_check
    def mask_hide(self, mask):
        """隐藏种子 坐标更新前不再作为源渲染"""
        #只能用于非周期域边界 
        #一个种子可能触发多次
        self.mask_ref(mask).loc -= 3.0  #注意: 将更新种子时间 依赖链将更新 
        # self.mask_ref(mask).r -= 1.0
        
    @auto_check
    def remove_hide(self) -> Seeds:
        """移除隐藏种子,返回种子引用"""
        seeds = self.mask_ref(np.all(self.loc >= 0.0, axis=1))  #采样得到的种子坐标>=0
        seeds = seeds.mask_ref(seeds.r > 0.0)
        return seeds

    @auto_check
    def data_mask_ref(self, data_mask: np.ndarray) -> Seeds:
        """创建引用 重新设置底层数据掩膜"""
        return Seeds(self.bdata, np.where(data_mask)[0])

    @auto_check
    def data_idx_ref(self, data_idx: np.ndarray) -> Seeds:
        """创建索引引用种子(子集) 重新设置索引"""
        return Seeds(self.bdata, data_idx)

    @auto_check
    def rad_ref(self, r_th: float) -> Seeds:
        return self.mask_ref(self.r > r_th)

    @auto_memo  #计算量较大 缓存结果 对相同的种子给出固定结果,radbat中阻止继续传播
    def get_rths(self, n_rad: int, rad_splitor: RadSplitor) -> list:
        return rad_splitor.get_rths(self, n_rad)

    @auto_check
    def itv_ref(self, r_itv: list[float]) -> Seeds:
        itv_mask = np.bitwise_and(self.r>min(r_itv), self.r<=max(r_itv))
        return self.mask_ref(itv_mask)

    @auto_check
    def rad_bat_ref(self, n_rad: int, cur_rad:int, rad_splitor: RadSplitor) -> Seeds:
        """ ref_seeds未改变时 在多个半径引用间切换实现(切换前检查ref_seeds)"""
        rths = self.get_rths(n_rad, rad_splitor)
        cur_rad = min(cur_rad, n_rad-1) #不超过rth上限
        return self.itv_ref([rths[cur_rad], rths[cur_rad+1]]) 
    
    @auto_check
    def cum_bat_ref(self, n_rad: int, cur_rad:int, rad_splitor: RadSplitor) -> Seeds:
        """ ref_seeds未改变时 在多个半径引用间切换实现(切换前检查ref_seeds)"""
        rths = self.get_rths(n_rad, rad_splitor)
        cur_rad = min(cur_rad, n_rad-1) #不超过rth上限
        return self.rad_ref(rths[cur_rad])  #取已经完成投入的种子 

    @auto_check
    def mark_ref(self, mark: int) -> Seeds:
        return self.mask_ref(self.mark == mark)

    @auto_check
    def phase_ref(self, phase: Phase) -> Seeds:
        if isinstance(phase, Phase): 
            seeds = Seeds() 
            for mark in phase.sub_marks:
                seeds += self.mark_ref(mark)
            return seeds 
        elif isinstance(phase, (int, np.integer)):  #输入int
            return self.mark_ref(phase)
        else: 
            warnings.warn(f"{self.__str__()} phase ref warning: unknown input type of phase")
            return self.void_seeds
    
    @auto_check
    def rand_ref(self, prop: float=0) -> Seeds: 
        """随机批量引用"""        
        #按照比例提取 允许选中0个种子
        if prop < 1e-16 or not self.cnt: 
            return self.void_seeds
        elif prop > 1 - 1e-16:
            new_idx = np.random.choice(self.idx, self.cnt, replace=False)
            return self.data_idx_ref(new_idx)
        else: 
            new_idx = np.random.choice(self.idx, min(int(prop*self.cnt)+1, self.cnt-1), replace=False)
            return self.data_idx_ref(new_idx)

        # #在有种子时 不为0和1时,至少提取一个,保留一个
        # elif mode == "solid": 
        #     if self.cnt:  
        #         new_idx = np.random.choice(self.idx, min(int(prop*self.cnt) + 1, self.cnt-1), replace=False)
        #         return self.data_idx_ref(new_idx)
        #     else: 
        #         return self.void_seeds

        # else:
        #     raise ValueError(f"{self.__str__()} rand ref failed: unknown mode {mode}")
    
    @auto_check
    def rnd_bat_ref(self, n_it: int, cur_it:int, rnd_splitor: RndSplitor) -> Seeds:
        return self.rand_ref(rnd_splitor.get_prop(self, n_it, cur_it))  #当前随机批量

    #TODO 调试 
    @auto_check
    def queue_ref(self, prop: float=0, mode="solid") -> Seeds: 
        """循环队列引用 不重复引用,直到所有种子完成一次引用"""
        if not self.__dict__.get("queue_i", False): self.queue_i = 1
        else: self.queue_i += 1
        
        if prop <= 0 and mode == "solid": 
            prop = 1e-16 

        if self.cnt and prop > 0: 
            idx_list = np.array_split(self.idx, min(int(1/prop), self.cnt))
            new_idx = idx_list[self.queue_i]
            return self.data_idx_ref(new_idx)
        else:
            return self.void_seeds 
    
    @auto_check
    def sort(self):
        """属性从大到小 半径从大到小排序 对idx操作"""
        r_sort = self.idx_ref(np.argsort(self.r, kind="stable"))  #按半径排序
        m_sort = r_sort.idx_ref(np.argsort(r_sort.mark, kind="stable")[::-1]) #按属性
        return m_sort
        
    @auto_check
    def que_bat_ref(self, n_it: int, cur_it:int, rnd_splitor: PropSplitor) -> Seeds:
        # rnd_splitor 应当提供相同的分割比例
        return self.queue_ref(rnd_splitor.get_prop(self, n_it, cur_it))  #当前随机批量

    @auto_check
    def discrete(self, reg: Region, rpos=1.0) -> PhaseVol: 
        seeds_vol = np.zeros(reg.shape, dtype=reg.vlz.vol_dtype)
        if not self.cnt: return PhaseVol(seeds_vol) 

        ells = self.mask_ref(np.bitwise_and(self.a != 0, self.a >= self.b))
        obla = self.mask_ref(np.bitwise_and(self.a != 0, self.a < self.b))
        sphs = self.mask_ref(self.a == 0)
        
        if sphs.cnt:
            int_sour_pos = (sphs.loc * reg.vxl_len).astype(np.int32) #注意种子内保存的为相对坐标
            int_sour_r = (sphs.r * rpos * reg.vxl_len).astype(np.int32) #半径为负则不渲染
            reg.vlz.discrete_volume(seeds_vol, int_sour_pos, int_sour_r, sphs.mark.astype(np.uint32), reg.periods)
        
        if obla.cnt:
            sub_ells, _ = oblate_ellipsoids_to_ellipsoids(obla)
            _, prod = closest_axes(sub_ells.a[:, None] * sub_ells.ort)
            circle_cnt = int(np.max(prod / reg.res * reg.len) * sub_ells.n_circle)
            locs, rads, axs, marks = ellipsoids_to_circles(sub_ells.loc, sub_ells.ort, sub_ells.b * rpos, sub_ells.a * rpos, sub_ells.mark, circle_cnt, reg.limits)

            int_sour_pos = (locs * reg.vxl_len).astype(np.int32) #注意种子内保存的为相对坐标
            int_sour_r = (rads * reg.vxl_len).astype(np.int32)
            Region2D.vlz.discrete_volume_axis(seeds_vol, int_sour_pos, int_sour_r, marks.astype(np.uint32), axs.astype(np.uint32), reg.periods)


        if ells.cnt:
            _, prod = closest_axes(ells.a[:, None] * ells.ort)
            circle_cnt = int(np.max(prod / reg.res * reg.len) * ells.n_circle)
            locs, rads, axs, marks = ellipsoids_to_circles(ells.loc, ells.ort, ells.b * rpos, ells.a * rpos, ells.mark, circle_cnt, reg.limits)

            int_sour_pos = (locs * reg.vxl_len).astype(np.int32) #注意种子内保存的为相对坐标
            int_sour_r = (rads * reg.vxl_len).astype(np.int32)
            Region2D.vlz.discrete_volume_axis(seeds_vol, int_sour_pos, int_sour_r, marks.astype(np.uint32), axs.astype(np.uint32), reg.periods)

        return PhaseVol(seeds_vol) 

    @auto_check
    def discrete_counts(self, reg: Region, rpos=1.0) -> PhaseVol: 
        """计算域内部种子体素数目 域内部设置周期性边界
        Examples: 
            loc = [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]
            reg = Region(1, [120, 120, 120])
            vcfg.reg = reg 
            vcfg.rpos = 0.5
            seeds = Seeds.from_info(loc, r=[0.0, 0.2], a=[0.1, 0.0], b=[0.4, 0.0], ort=[[1, 1, 1], [0, 0, 0]], mark=1)
            seeds.discrete_counts(reg), seeds.discrete(reg)
        """
        seeds_vol = np.zeros(reg.shape, dtype=reg.vlz.vol_dtype)
        if not self.cnt: return PhaseVol(seeds_vol) 
        
        #建立完全种子集和 用于索引访问
        seeds = Seeds(self.data, np.arange(self.cnt))  #重置局部索引,避免多级依赖导致索引问题
        ells = seeds.mask_ref(np.bitwise_and(seeds.a != 0, seeds.a >= seeds.b))
        obla = seeds.mask_ref(np.bitwise_and(seeds.a != 0, seeds.a < seeds.b))
        sphs = seeds.mask_ref(seeds.a == 0)
        vxl_cnts = np.zeros(seeds.cnt)

        if sphs.cnt:
            int_sour_pos = (sphs.loc * reg.vxl_len).astype(np.int32) #注意种子内保存的为相对坐标
            int_sour_r = (sphs.r * rpos * reg.vxl_len).astype(np.int32)
            vxl_cnts[sphs.idx] = reg.vlz.discrete_volume(seeds_vol, int_sour_pos, int_sour_r, sphs.mark.astype(np.uint32), reg.periods)

        if obla.cnt:
            sub_ells, inv_cnts = oblate_ellipsoids_to_ellipsoids(obla)
            cuts = np.cumsum([0] + inv_cnts)
            sub_vxl_cnts = sub_ells.discrete_counts(reg)
            vxl_cnts[obla.idx] = [np.sum(sub_vxl_cnts[cuts[i]:cuts[i+1]]) for i in range(cuts.size-1)]

        if ells.cnt:
            _, prod = closest_axes(ells.a[:, None] * ells.ort)  # 最近轴上的投影向量
            circle_cnt = int(np.max(prod / reg.res * reg.len) * ells.n_circle)
            locs, rads, axs, marks = ellipsoids_to_circles(ells.loc, ells.ort, ells.b * rpos, ells.a * rpos, ells.mark, circle_cnt, reg.limits)

            int_sour_pos = (locs * reg.vxl_len).astype(np.int32) #注意种子内保存的为相对坐标
            int_sour_r = (rads * reg.vxl_len).astype(np.int32)
            cnt = Region2D.vlz.discrete_volume_axis(seeds_vol, int_sour_pos, int_sour_r, marks.astype(np.uint32), axs.astype(np.uint32), reg.periods)

            #从mask重建统计
            # sub_cnt = np.zeros(ells.cnt * circle_cnt); sub_cnt[mask] = cnt
            # vxl_cnts[ells.idx] = np.sum(sub_cnt.reshape((-1, circle_cnt)), axis=1)
            vxl_cnts[ells.idx] = np.sum(cnt.reshape((-1, circle_cnt)), axis=1)
        return vxl_cnts

    @auto_check
    def discrete_dmn(self, reg: Region, ashp: AlphaShape) -> PhaseVol:
        seeds_vol = self.discrete(reg, ashp.rpos)
        # n_voxel = ashp.rfix * ashp.rpos * self.r_median * reg.vxl_len
        return ashp.a_shape(seeds_vol, self.r_median, reg)
        # return seeds_vol.a_shape(n_voxel, measure=ashp.measure)

    @auto_check
    def discrete_dmn_hrc(self, reg: Region, ashp: AlphaShape, n_hrc=4) -> PhaseVol: 
        """计算种子占据域 允许层级alphashape修正"""
        if n_hrc <= 1:
            return self.discrete_dmn(reg, ashp)

        seeds_vol = PhaseVol.zeros(reg.shape)
        rad_bat = RadBatRef(Seeds(self.bdata, self.idx), n_hrc, 0, ashp.splitor)
        for rad_bat.cur_bat in range(n_hrc):
            seeds_vol += rad_bat.discrete(reg, ashp.rpos)
            # n_voxel = ashp.rfix * ashp.rpos* rad_bat.r_median * reg.vxl_len
            seeds_vol = ashp.a_shape(seeds_vol, rad_bat.r_median, reg)
            # seeds_vol = seeds_vol.a_shape(n_voxel, measure=ashp.measure)  #大层级保留的情形下进行修正
        return PhaseVol(seeds_vol) 
    
    # @auto_check
    # def remove_overlap(self, rpos = 0.8, th = 0.01) -> Seeds:
    #     """移除在th精度(相对最长边)内重叠的种子,rad指定半径大小(考虑整个球)
    #     """
    #     seeds = self.sort()
    #     return seeds.idx_ref(seeds.check_overlap_center(rpos, th))
    
    # @auto_check
    # def remove_overlap_aabb(self, th = 0.2):
    #     from aabbtree import AABB
    #     from aabbtree import AABBTree 

    #     #要求:大的种子先加入
    #     seeds = self.sort()[::-1].copy()
    #     tree = AABBTree()
    #     nov_idxs = []  #保留的种子索引
    #     for seed_i, aabb in enumerate(seeds.aabb):    
    #         cur_aabb = AABB(aabb)
    #         cur_ov_idxs = tree.overlap_values(cur_aabb, method='BFS')  #当前aabb碰撞索引
            
    #         if len(cur_ov_idxs): 
    #             dists = np.linalg.norm(seeds.loc[seed_i] - seeds.loc[cur_ov_idxs], axis=1)
    #             rad_th = min(seeds.r[seed_i], np.min(seeds.r[cur_ov_idxs])) * th 
    #             if not np.all(dists - seeds.r[seed_i] - seeds.r[cur_ov_idxs] > -rad_th):  #现有种子包含了当前种子中心
    #                 continue
            
    #         tree.add(cur_aabb, seed_i)  #将当前种子加入树
    #         nov_idxs.append(seed_i)  #保留当前种子
        
    #     return seeds.idx_ref(np.array(nov_idxs))

    # @auto_check
    # def check_overlap_center(self, rel_rad = 1.0, th = 0.01) -> np.ndarray:
    #     reg = Region(th, np.array([1,1,1]))  #在默认region内进行检查
    #     ov_mask = self.discrete_seeds_vol_check(reg, rel_rad)[1].astype(np.bool_)
    #     nov_idxs = np.where(~ov_mask)[0]
    #     return nov_idxs

    # @auto_check
    # def remove_duplicate(self, seeds: Seeds) -> Seeds:
    #     loc_idxs = self.data_dict["loc"]
    #     data_dup_mark = np.sum(np.abs(self.bdata[:, loc_idxs] - seeds.bdata[:, loc_idxs]), axis=1) < 1e-8
    #     return self and self.data_mask_ref(~data_dup_mark)  #自身所有位置变化过的种子

    # @auto_check
    # def discrete_seeds_vol_check(self, reg, rpos=0.8) -> tuple:
    #     """离散种子体积,通过相对半径指定离散范围 返回当前种子重叠标志,即范围内是否包含其他种子"""
    #     seeds_vol = np.zeros(shape=reg.shape, dtype=vlz.vol_dtype)
    #     ovlp_mark = np.zeros(shape=self.cnt, dtype=vlz.data_dtype) 
    #     if self.cnt:
    #         uint_loc = (self.loc * reg.vxl_len).astype(np.uint32)
    #         uint_r = (self.r * rpos * reg.vxl_len).astype(np.uint32)
    #         ovlp_mark = vlz.discrete_volume_check(seeds_vol, uint_loc, uint_r, 1)
    #     else:
    #         print("discrete seeds vol failed: seeds is empty")
    #     return seeds_vol, ovlp_mark
    

    @auto_check 
    def search_seeds_vf(self, reg: Region, vol_frac: float) -> Seeds:
        """二分查找确定种子数目 优先保留靠前的种子"""
        cum_vfs = np.cumsum(self.discrete_counts(reg) / np.prod(reg.shape))
        n_seed = np.searchsorted(cum_vfs, vol_frac) 
        return self.__class__(np.copy(self.data[:n_seed, :]), np.arange(n_seed))
    
    @auto_check 
    def mark_map(self, map):
        seeds = self.copy()
        seeds.mark = np.array(map)[self.mark]
        return seeds
    
    attr_map = mark_map

#椭球渲染方法
def closest_axes(orientations):
    """
    计算多个方向向量与 x, y, z 轴方向最接近的坐标轴索引。
    
    Parameters:
    - orientations: (N, 3) 数组，表示 N 个方向向量。
    
    Returns:
    - closest_axes: (N,) 数组，每个元素为与对应 orientation 最接近的轴的索引 (x: 0, y: 1, z: 2)。
    """
    # 定义坐标轴的单位向量 (3, 3) 数组，每列表示一个轴方向
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T  # (3, 3
    # 计算 orientations 与每个坐标轴的点积 (N, 3)
    dot_products = np.abs(np.dot(orientations, axes))  # (N, 3)
    # 找到每个方向向量与最接近的轴，返回对应的索引 (N,)
    closest_axes = np.argmax(dot_products, axis=1)
    return closest_axes, dot_products


def ellipsoids_to_circles(centers, orientations, radiis, heights, marks, num_circles, limits):
    """
    将多个圆柱体种子分解为多个二维圆形种子，使用向量化操作。
    
    Parameters:
    - centers: (N, 3) 数组，每行代表一个圆柱体的中心坐标 (x, y, z)。
    - orientations: (N, 3) 数组，每行代表一个圆柱体的方向向量。
    - radii: (N,) 数组，每个圆柱体的底面半径。
    - heights: (N,) 数组，每个圆柱体的高度。
    - num_circles: 每个圆柱体用来近似的二维圆形种子数量。
    
    Returns:
    - circle_centers_radii: (N * num_circles, 4) 数组，包含所有二维圆形种子的坐标和半径，每行代表一个圆 (x, y, z, r)。
    """
    # 归一化方向向量
    orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)
    ax, _ = closest_axes(orientations)

    # 沿着圆柱体的高度方向生成圆心坐标，等距分布
    t_values = np.linspace(-0.5, 0.5, num_circles)  # [-0.5, 0.5] 范围内的均匀分布

    # 扩展 t_values 和 orientations 的维度以支持广播
    t_values = np.repeat(t_values[None, :], len(centers), axis=0)  # (N, num_circles)
    orientations = np.repeat(orientations[:, None, :], num_circles, axis=1)  # (N, num_circles, ndim)
    centers = np.repeat(centers[:, None, :], num_circles, axis=1)  # (N, num_circles, ndim)
    axs = np.repeat(ax, num_circles)
    marks = np.repeat(marks, num_circles)

    # 计算圆心坐标 
    height_offsets = t_values * heights[:, None] * 2 # (N, num_circles)
    circle_centers = centers + (height_offsets[:, :, None] * orientations)  # (N, nc, 3)
    # 扁平化圆心坐标
    circle_centers = circle_centers.reshape(-1, 3)  # (N * num_circles, 3)
    # mask = np.bitwise_and(np.all(circle_centers >= np.array(limits)[:, 0], axis=1), 
    #         np.all(circle_centers < np.array(limits)[:, 1], axis=1))
    # mask = np.ones(circle_centers.shape[0], dtype=np.bool_)

    # 计算每个位置的截面半径，基于椭圆曲线
    ellipse_factors = np.sqrt(1 - (t_values * 1.8)**2)  # (N, num_circles)
    circle_radii = (ellipse_factors * radiis[:, None]).reshape(-1)  # (N, num_circles)
    # return circle_centers[mask], circle_radii[mask], axs[mask], marks[mask], mask
    return circle_centers, circle_radii, axs, marks

def oblate_ellipsoids_to_ellipsoids(obla, num_elli=4):
    """扁椭球转化为椭球集合"""
    locs = obla.loc
    orts = obla.ort
    alens = obla.a
    blens = obla.b
    marks = obla.mark

    #节点数目
    n_ells = np.int32(blens / alens) * num_elli + 1
    sub_locs = []
    sub_rads = []
    sub_lens = []
    sub_orts = []
    sub_marks = []
    inv_cnts = []
    #内部每个椭球
    for si in range(n_ells.size):
        n = n_ells[si]
        a = alens[si]
        b = blens[si]
        loc = locs[si]
        ort = orts[si]
        mark = marks[si]
        
        #取向计算(垂直平面内任取两个垂直基向量)
        norm = np.cross(ort, [0, 0, 1])
        norm = norm if np.any(norm) else np.array([0, 1, 0])  #排除共面
        plane = np.cross(ort, norm)
        plane = plane / np.linalg.norm(plane)

        offsets = np.linspace(-b, b, n)

        #节点半径
        sub_locs.append([loc + offset*plane for offset in offsets])
        sub_rads.append(np.sqrt(np.clip([a**2 - (a**2)*(offset**2)/(b**2) for offset in offsets], 0, 1)))
        sub_lens.append(np.sqrt(np.clip([b**2 - offset**2 for offset in offsets], 0, 1)))

        sub_orts.append([norm] * n)  #默认沿z方向法平面
        sub_marks.append([mark] * n)
        inv_cnts.append(n)

    return Seeds.from_info(loc=np.concatenate(sub_locs), b=np.concatenate(sub_rads), a=np.concatenate(sub_lens), \
        ort=np.concatenate(sub_orts), mark=np.concatenate(sub_marks)), inv_cnts


#-------------------------------------------------------------------------------------#
# 相与复合相
#-------------------------------------------------------------------------------------#


class Phase(Node):
    """相: 同一相种子的宏观描述"""
    rmin = 0.4  #相对体素边长的最小种子半径
    n_pre_sample = 1000
    min_sample_seeds = 1
    max_sample_seeds = 10000000
    pre_sample_prop = 2.0 

    def __init__(self, r_dist=DistFunc(), v_frac: float=1.0, mark: int=1, name="") -> None:
        """初始化,其中r_dist为相对region.len的半径"""
        self.r_dist = r_dist 
        self.v_frac = v_frac 
        self.mark = mark 
        self.name = name 

        self.reset_data() #保存拟合所需种子

        if mark == 0: raise Warning("Phase init warning: mark is set as 0, phase is ignored")

        self.update()

    def dec_key(self) -> tuple:  #自身作为节点的声明时输入参数
        return self.r_dist, self.v_frac, self.mark, self._seeds

    def run_key(self) -> tuple:
        return *self.r_dist.run_key(), self.v_frac, self.mark, self._seeds  #调用依赖节点的run_key,检测依赖节点的状态变化 

    def __str__(self):
        if len(self.name):
            return f"{self.name}(vf:{np.round(self.v_frac, 3)})"
        else:
            return f"{self.cls_str}(vf:{np.round(self.v_frac, 3)} mk:{self.sub_marks_str})"
    
    @auto_check  #保存之前更新
    def __getstate__(self):
        return {"r_dist": self.r_dist, "v_frac": self.v_frac, "mark": self.mark, "name": self.name} #, "_seeds": self._seeds, "_reg": self._reg}
    
    def __setstate__(self, value):
        # print("phase loaded")
        self.__dict__.update(value)
        self.reset_data()
        return 

    def copy(self):
        return self.__class__(self.r_dist, self.v_frac, self.mark, self.name)

    @property
    def subs(self):
        return OrderSet([self])
    
    def reset_data(self):
        """清空缓存的拟合种子"""
        self._seeds = Seeds()  #保存拟合所需种子
        self._reg = None 
        self._n_fit = 0

    # @auto_memo  #可能在后续底层数据被更换  需要缓存,使得fit,sample访问相同的种子集合
    @auto_check
    def sample(self, reg: Region) -> Seeds:
        """采样,获得种子集合实例. 相对region的len定义半径"""
        #采样得到种子体积均值(相对值)
        pre_r = self.r_dist.sample(self.n_pre_sample) 
        reg = reg.adapt(pre_r)  #尽量避免超过范围
        pre_loc = reg.size/reg.len * np.random.rand(pre_r.size, 3) 
        pre_seeds = Seeds.from_info(loc=pre_loc, r=pre_r, mark=self.mark) 
        avg_vf = np.average(pre_seeds.discrete_counts(reg) / np.prod(reg.shape)) 

        #预计算种子数目
        n_pre_seeds = int(self.v_frac / avg_vf * self.pre_sample_prop)   #均为相对值 
        if n_pre_seeds < self.min_sample_seeds:
            raise RuntimeError(f"sample phase {self.__str__()} warn:seeds count reaches min value, check radius distribution")
        if n_pre_seeds > self.max_sample_seeds:
            warnings.warn(f"sample phase {self.__str__()} warn:seeds count reaches max value, auto clip")        
        seeds_r = self.r_dist.sample(min(n_pre_seeds, self.max_sample_seeds))   #返回相对半径

        #种子半径裁剪 保证最小值满足
        if np.min(seeds_r) < reg.vxl_unit * self.rmin:
            warnings.warn(f"sample phase {self.__str__()} warn:seeds radius reaches min value") 
        seeds_r = np.clip(seeds_r, reg.vxl_unit * self.rmin, None)
        
        #完成TODO 利用种子离散体素数目给出所需种子数目
        loc = reg.size/reg.len * np.random.rand(seeds_r.size, 3) 
        seeds = Seeds.from_info(loc=loc, r=seeds_r, mark=self.mark) 
        seeds = seeds.search_seeds_vf(reg, self.v_frac)  #不能采用sort,否则分布将偏移
        seeds.reg = reg
        return seeds

    # @auto_check
    def fit(self, seeds: Seeds, reg: Region):
        """从种子集合计算distfunc参数"""
        if seeds.cnt <= 0:
            warnings.warn(f"phase fit warning: {seeds.__str__()} is empty, skip")
            return 
        # seeds = seeds.phase_ref(self)
        # self.r_dist.fit(seeds.r)  #拟合相对半径分布 
        # self.v_frac = np.sum(seeds.discrete_counts(reg, 1.0)) / np.prod(reg.shape)  #只拟合边界内的种子体积, 对应周期性边界重建
        # self.mark = np.max(seeds.mark) if seeds.cnt else self.mark 
        self._seeds += seeds
        self._reg = reg
        self._n_fit += 1

    def update(self):
        # print("phase update called", self._seeds, self._reg, self._n_fit)
        if self._seeds.cnt and self._reg: 
            # raise RuntimeError()
            seeds = self._seeds
            reg = self._reg
            seeds = seeds.phase_ref(self)
            self.r_dist.fit(seeds.r)  #拟合相对半径分布 
            self.v_frac = np.sum(seeds.discrete_counts(reg, 1.0)) / np.prod(reg.shape) / self._n_fit  #只拟合边界内的种子体积, 对应周期性边界重建
            self.mark = np.max(seeds.mark) if seeds.cnt else self.mark 

            #一次update后抛弃数据
            self.reset_data()

        Node.update(self)  #对于输入的source节点 将其拆分为基本参数传入

    #不允许添加检查修饰器
    @property
    def sub_marks(self) -> tuple:
        return tuple([phs.mark for phs in self.subs]) 
    
    @sub_marks.setter
    def sub_marks(self, value: int):
        for phs in self.subs:
            phs.mark = value
        return 
    
    @property
    def sub_marks_str(self):
        marks_str = ""
        for mark in self.sub_marks:
            marks_str += f"{mark}-"
        return marks_str[:-1]

    def __add__(self, phs2):
        return CompPhase([self, phs2], name=f"{self.name}-{phs2.name}")


class AnisoPhase(Phase):
    """各向异性相"""
    def __init__(self, r_dist=DistFunc(), ab_dist=UniformDist(1.0, 0.0), ort_dist=CompDist(), v_frac: float=1.0, mark: int=1, name="") -> None:
        """初始化,其中r_dist为相对region.len的半径"""
        self.ort_dist = ort_dist 
        self.r_dist = r_dist
        self.ab_dist = ab_dist

        self.ab_dist.xmax = 101
        self.ort_dist.xmin = -same.xmax
        self.ort_dist.xmax = same.xmax

        self.v_frac = v_frac 
        self.mark = mark
        self.name = name
        if mark == 0: raise Warning("Phase init warning: mark is set as 0, phase is ignored") 
        self.reset_data() #保存拟合所需种子
        self.update() 

    def dec_key(self) -> tuple:  #自身作为节点的声明时输入参数
        return self.r_dist, self.ab_dist, self.v_frac, self.mark, self.ort_dist, self._seeds

    def run_key(self) -> tuple:
        # print(self)
        return *self.r_dist.run_key(), *self.ab_dist.run_key(), self.v_frac, self.mark, *self.ort_dist.run_key(), self._seeds  #调用依赖节点的run_key,检测依赖节点的状态变化 

    def copy(self):
        return self.__class__(self.r_dist, self.ab_dist, self.ort_dist, self.v_frac, self.mark, self.name)

    @auto_check
    def sample(self, reg: Region) -> Seeds:
        """采样,获得种子集合实例. 相对region的len定义半径"""
        #采样得到种子体积均值(相对值)
        pre_b = self.r_dist.sample(self.n_pre_sample) 
        pre_a = self.ab_dist.sample(self.n_pre_sample) * pre_b
        pre_ort = self.ort_dist.sample(self.n_pre_sample) 
        pre_loc = reg.size/reg.len * np.random.rand(pre_b.size, 3)
        pre_seeds = Seeds.from_info(loc=pre_loc, ort=pre_ort, b=pre_b, a=pre_a, mark=self.mark) 
        avg_vf = np.average(pre_seeds.discrete_counts(reg) / np.prod(reg.shape)) 

        #预计算种子数目
        n_pre_seeds = int(self.v_frac / avg_vf * self.pre_sample_prop)   #均为相对值 
        if n_pre_seeds < self.min_sample_seeds:
            raise RuntimeError(f"sample phase {self.__str__()} warn:seeds count reaches min value, check radius distribution")
        if n_pre_seeds > self.max_sample_seeds:
            warnings.warn(f"sample phase {self.__str__()} warn:seeds count reaches max value, auto clip")        
        seeds_b = self.r_dist.sample(min(n_pre_seeds, self.max_sample_seeds))   #返回相对半径
        seeds_a = self.ab_dist.sample(min(n_pre_seeds, self.max_sample_seeds)) * seeds_b
        seeds_ort = self.ort_dist.sample(min(n_pre_seeds, self.max_sample_seeds))  

        #种子半径裁剪 保证最小值满足
        if np.min(seeds_b) < reg.vxl_unit * self.rmin:
            warnings.warn(f"sample phase {self.__str__()} warn:seeds radius reaches min value") 
        seeds_b = np.clip(seeds_b, reg.vxl_unit * self.rmin, None)
        if np.min(seeds_a) < reg.vxl_unit * self.rmin:
            warnings.warn(f"sample phase {self.__str__()} warn:seeds radius reaches min value") 
        seeds_a = np.clip(seeds_a, reg.vxl_unit * self.rmin, None)
        
        #完成TODO 利用种子离散体素数目给出所需种子数目
        loc = reg.size/reg.len * np.random.rand(seeds_b.size, 3) 
        seeds = Seeds.from_info(loc=loc, ort=seeds_ort, b=seeds_b, a=seeds_a, mark=self.mark) 
        seeds = seeds.search_seeds_vf(reg, self.v_frac)  #不能采用sort,否则分布将偏移
        seeds.reg = reg 
        return seeds
    
    #TODO ab_dist 的范围归一化, 以指数形式定义
    # @auto_check
    # def fit(self, seeds: Seeds, reg: Region):
    #     """从种子集合计算distfunc参数"""
    #     if seeds.cnt <= 0:
    #         warnings.warn(f"phase fit warning: {seeds.__str__()} is empty, skip")
    #         return 
    #     seeds = seeds.phase_ref(self)
        
    #     self.ab_dist.xmax = np.inf
    #     self.ort_dist.xmin = -np.inf
    #     self.ort_dist.xmax = np.inf

    #     self.r_dist.fit(seeds.b)  
    #     self.ab_dist.fit(seeds.a / seeds.b)  #拟合相对半径分布 
    #     self.ort_dist.fit(seeds.ort)
    #     self.v_frac = np.sum(seeds.discrete_counts(reg, 1.0)) / np.prod(reg.shape)  #只拟合边界内的种子体积, 对应周期性边界重建
    #     self.mark = np.max(seeds.mark) if seeds.cnt else self.mark 
    #     seeds.reg = reg

    def update(self):
        if self._seeds.cnt and self._reg: 
            seeds = self._seeds.phase_ref(self)
            
            self.ab_dist.xmax = 101
            self.ort_dist.xmin = -np.inf
            self.ort_dist.xmax = np.inf

            self.r_dist.fit(seeds.b)  
            self.ab_dist.fit(seeds.a / seeds.b)  #拟合相对半径分布 
            self.ort_dist.fit(seeds.ort)
            self.v_frac = np.sum(seeds.discrete_counts(self._reg, 1.0)) / np.prod(self._reg.shape) / self._n_fit  #只拟合边界内的种子体积, 对应周期性边界重建
            self.mark = np.max(seeds.mark) if seeds.cnt else self.mark 
            #一次update后抛弃数据
            self.reset_data()

        Node.update(self)  #对于输入的source节点 将其拆分为基本参数传入

    @auto_check  #保存之前更新
    def __getstate__(self):
        return {"r_dist": self.r_dist, "v_frac": self.v_frac, "mark": self.mark, "name": self.name,
                "ort_dist": self.ort_dist, "ab_dist": self.ab_dist}


class ConstPhase(Phase):
    def fit(self, seeds: Seeds, reg: Region): return

class ConstAnisoPhase(AnisoPhase):
    def fit(self, seeds: Seeds, reg: Region): return  

class CompPhase(Phase, Link):  #继承Phase的方法 Link的状态表示
    def __init__(self, phss: list[Phase]=[], name=""):
        self.phss = phss 
        self.name = name
        Link.update(self)

    def dec_key(self) -> tuple:  #声明时依赖的键值 
        return *tuple(self.phss), 

    def run_key(self) -> tuple:  #运行时依赖键值的状态
        return *tuple([phs.run_key() for phs in self.phss]),

    def __str__(self):
        if len(self.name):
            return f"{self.name}(marks:{self.sub_marks_str})"
        else:
            return f"{self.cls_str}(marks:{self.sub_marks_str})"
    
    @auto_check
    def __getstate__(self):
        return {"phss": self.phss, "name": self.name}
    
    def copy(self):
        phss = [Phase(phs.r_dist, phs.v_frac, phs.mark, phs.name) for phs in self.subs]
        return self.__class__(phss, self.name) 
    
    @property
    def subs(self) -> set[Phase]:
        """依赖的子相"""
        subs = OrderSet()
        for phase in self.phss: 
            subs.update(phase.subs) #subs内包含基本相
        return subs  #保证访问子相前检查依赖是否更新

    # @auto_memo
    @auto_check
    def sample(self, reg: Region) -> Seeds: 
        seeds = Seeds() 
        for phase in self.subs:
            seeds += phase.sample(reg)
        seeds.reg = reg
        return seeds 
    
    # @auto_check
    def fit(self, seeds: Seeds, reg: Region): 
        for phase in self.subs:
            phase.fit(seeds.phase_ref(phase), reg)
    
    @auto_check
    def get_dict(self):
        return {sub.mark: sub.name for sub in self.subs}



#-------------------------------------------------------------------------------------#
# 相种子
#-------------------------------------------------------------------------------------#

class PCNodeOperation(Link):
    """顶点操作"""
    def __init__(self, phs: Phase, reg: Region, *args) -> None:
        self.phs = phs
        self.args = args 
        self.reg = reg  #相在一定域大小下 给出种子
        self.update()

    def dec_key(self) -> tuple:
        return self.phs, self.reg 
    
    def run_key(self) -> tuple:
        return *self.phs.run_key(), tuple(self.reg.shape) 
    
    def __str__(self):
        return f"{self.cls_str}(phs:{self.phs})"


class PCNodeSample(PCNodeOperation):
    def __init__(self, phs: Phase, reg: Region) -> None:
        """根据相和域采样得到种子集合"""
        PCNodeOperation.__init__(self, phs, reg)

    def update(self):  #update将在初始化时调用 建立自身和输入的关系
        seeds = self.phs.sample(self.reg).sort()
        # seeds = self.phs.sample(self.reg).
        self.seeds = seeds 
        Link.update(self)

class PCNodeFit(PCNodeOperation):  
    def __init__(self, phs: Phase, seeds: Seeds, reg: Region) -> None:
        """从种子集合与域设置相参数"""
        self.seeds = seeds 
        PCNodeOperation.__init__(self, phs, reg, seeds)

    def dec_key(self) -> tuple:
        return self.phs, self.reg, self.seeds 

    def run_key(self) -> tuple:
        return *self.phs.run_key(), *self.seeds.run_key(), tuple(self.reg.shape) 

    def update(self):  #update将在初始化时调用 建立自身和输入的关系
        self.phs.fit(self.seeds, self.reg)
        Link.update(self)


class PhaseSeeds(PCNodeOperation, Seeds):  #继承link的运行行为 和seeds的数据操作
    def __init__(self, phs: Phase, reg: Region) -> None:
        """将Phase和Region作为源 产生种子"""
        if not isinstance(phs, Phase):
            raise ValueError(f"init {self.__class__.name} failed: expect type {Phase} for parameter phs")
        self.phs = phs
        self.reg = reg  #相在一定域大小下 给出种子
        Seeds.__init__(self)

    def run_key(self) -> tuple:
        return *self.phs.run_key(), tuple(self.reg.shape), id(self.bdata), id(self.idx), self._mod_time  #只依赖region的大小 和分辨率无关
        #reg的size为自动求出函数 其id将更新

    def update(self):  #update将在初始化时调用 建立自身和输入的关系
        seeds = self.phs.sample(self.reg).phase_ref(self.phs)
        self.bdata = seeds.bdata 
        self.idx = seeds.idx 
        self.mod_time()
        Link.update(self)

#-------------------------------------------------------------------------------------#
# 种子引用
#-------------------------------------------------------------------------------------#

class RefSeeds(Seeds, Link):
    """按照一定规则自动同步更改的种子集合 避免外部管理分配
    每次访问数据前检查源是否改变 若改变 重新建立引用
    不捕获data内数据的变化(data数据变化 不会引起后续引用的变化)
    """
    def __init__(self, ref_seeds: RefSeeds, func: Callable, *args):
        self.ref_seeds = ref_seeds
        self.n_data = ref_seeds.n_data
        self.data_dict = ref_seeds.data_dict
        self.data_str_tab = ref_seeds.data_str_tab
        self.n_circle = ref_seeds.n_circle
        self.func = func
        self.args = args 
        Seeds.__init__(self)

    def mod_time(self):
        """底层数据更新 反向传播的时间更新"""
        self._mod_time = time.time() 
        self.ref_seeds.mod_time()  #反向传播时间更新
 
    def dec_key(self) -> tuple:
        return self.ref_seeds, self.func, *self.args 

    def run_key(self) -> tuple: 
        return *self.ref_seeds.run_key(), self.func, *self.args, self._mod_time  #只检查依赖种子的声明参数是否改变
        #引用种子的更新导致key变化   自身底层数据变化导致后续key变化

    def __str__(self):
        return f"{self.cls_str}({hex_id(self.ref_seeds)} cnt:{self.cnt})"
    
    def __repr__(self):
        return f"{self.__str__()} => {self.ref_seeds.__str__()}"
    
    def update(self):
        seeds = self.func(self.ref_seeds, *self.args)
        self.bdata = seeds.bdata
        self.idx = seeds.idx 
        self._mod_time = time.time() 
        Link.update(self) #缓存键值 

    @auto_check  #检查更新 避免复制过期的引用
    def copy(self) -> Seeds:
        """返回单纯的种子复制 不包括引用关系"""
        return self.ref_seeds.__class__(np.copy(self.bdata), np.copy(self.idx))  #注意idx和data都需要复制

    @property
    def base_ref(self) -> Seeds:
        seeds = self
        ref_seeds = seeds.__dict__.get("ref_seeds", seeds)  #获取引用 否则返回自身
        while ref_seeds != seeds: #无引用 seeds即为引用 则结束
            seeds = ref_seeds   #有引用 则将引用作为自身
            ref_seeds = seeds.__dict__.get("ref_seeds", seeds) 
        return seeds 
    
    # @auto_check
    # def discrete(self, *args):
    #     return self.ref_seeds.__class__.discrete(self, *args)

    # @auto_check
    # def discrete_counts(self, *args):
    #     return self.ref_seeds.__class__.discrete_counts(self, *args)

    # @auto_check
    # def discrete_dmn(self, *args):
    #     return self.ref_seeds.__class__.discrete_dmn(self, *args)

    # @auto_check
    # def discrete_dmn_hrc(self, *args):
    #     return self.ref_seeds.__class__.discrete_dmn_hrc(self, *args)

    # @auto_check
    # def mask_ref(self, *args):
    #     return self.ref_seeds.__class__.mask_ref(self, *args)

    # @auto_check
    # def idx_ref(self, *args):
    #     return self.ref_seeds.__class__.idx_ref(self, *args)
    
    # @auto_check
    # def data_mask_ref(self, *args):
    #     return self.ref_seeds.__class__.data_mask_ref(self, *args)

    # @auto_check
    # def data_idx_ref(self, *args):
    #     return self.ref_seeds.__class__.data_idx_ref(self, *args)
    
    # @auto_check
    # def phase_ref(self, *args):
    #     return self.ref_seeds.__class__.data_idx_ref(self, *args)

    # @auto_check
    # def __getitem__(self, *args):
    #     return self.ref_seeds.__class__.__getitem__(self, *args)

    # @auto_check
    # def __setitem__(self, *args):
    #     return self.ref_seeds.__class__.__setitem__(self, *args)
    # @auto_check
    # def __next__(self, *args):
    #     return self.ref_seeds.__class__.__next__(self, *args)

    # @property  #不缓存 及时更新 
    # @auto_check
    # def ort(self) -> np.ndarray:
    #     return self[:, "ort"]
    # @property 
    # @auto_check
    # def vx(self) -> np.ndarray:
    #     return self[:, "vx"]
    # @property 
    # @auto_check
    # def vy(self) -> np.ndarray:
    #     return self[:, "vy"]
    # @property 
    # @auto_check
    # def vz(self) -> np.ndarray:
    #     return self[:, "vz"]
    # @property 
    # @auto_check
    # def b(self) -> np.ndarray:
    #     return self[:, "b"]
    # @property 
    # @auto_check
    # def a(self) -> np.ndarray:
    #     return self[:, "a"]
    # @property 
    # @auto_check
    # def r(self) -> np.ndarray:
    #     return self.ref_seeds.__class__.r.fget(self)
    
    # @ort.setter
    # def ort(self, value: np.ndarray):
    #     self[:, "ort"] = value
    #     self.mod_time()
    # @vx.setter
    # def vx(self, value: np.ndarray):
    #     self[:, "vx"] = value
    #     self.mod_time()
    # @vy.setter
    # def vy(self, value: np.ndarray):
    #     self[:, "vy"] = value
    #     self.mod_time()
    # @vz.setter
    # def vz(self, value: np.ndarray):
    #     self[:, "vz"] = value
    #     self.mod_time()
    # @b.setter
    # def b(self, value: np.ndarray):
    #     self[:, "b"] = value
    #     self.mod_time()
    # @a.setter
    # def a(self, value: np.ndarray):
    #     self[:, "a"] = value
    #     self.mod_time()
    # @r.setter
    # def r(self, value: np.ndarray):
    #     return self.ref_seeds.__class__.r.fset(self, value)

    # @property
    # def void_seeds(self):
    #     return self.ref_seeds.__class__.void_seeds(self)

class SelfRef(RefSeeds):
    """创建种子集合的自身引用"""
    def __init__(self, ref_seeds: Seeds):
        self.ref_seeds = ref_seeds 
        Seeds.__init__(self, self.ref_seeds.bdata, self.ref_seeds.idx)
    
    def __str__(self):
        return self.ref_seeds.__str__()
    
    def __repr__(self):
        return self.ref_seeds.__repr__()
    
    def dec_key(self) -> tuple:
        return self.ref_seeds, 

    def run_key(self) -> tuple:
        """依赖自身引用参数和源种子数据"""
        return *self.ref_seeds.run_key(), self._mod_time

    def update(self):
        self.bdata = self.ref_seeds.bdata
        self.idx = self.ref_seeds.idx 
        self._mod_time = time.time() 
        Link.update(self) 

class PhaseRef(RefSeeds):
    """特定相的引用 支持复合相"""
    def __init__(self, ref_seeds: RefSeeds, phs: Phase):
        self.ref_seeds = ref_seeds 
        self.phs = phs 
        RefSeeds.__init__(self, ref_seeds, RefSeeds.phase_ref, phs)

    def __str__(self):
        return f"{self.cls_str}(phs:{self.phs} cnt:{self.cnt})"
    
    def dec_key(self) -> tuple:
        return self.ref_seeds, self.phs 

    def run_key(self) -> tuple: 
        return *self.ref_seeds.run_key(), *self.phs.run_key(), self._mod_time 
    
    def update(self):
        self.args = (self.phs, )
        RefSeeds.update(self) #缓存键值 


class RadItvRef(RefSeeds):
    """半径区间引用"""
    def __init__(self, ref_seeds: RefSeeds, r1: float, r2: float):
        self.ref_seeds = ref_seeds 
        self.r1 = r1 
        self.r2 = r2 
        RefSeeds.__init__(self, ref_seeds, RefSeeds.itv_ref, r1, r2)
    
    def __str__(self):
        return f"{self.cls_str}(r1:{self.r1} r2:{self.r2} cnt:{self.cnt})"
    
    def dec_key(self) -> tuple:
        return self.ref_seeds, self.r1, self.r2 

    def run_key(self) -> tuple: 
        return *self.ref_seeds.run_key(), self.r1, self.r2, self._mod_time 
    
    def update(self):
        self.args = (self.r1, self.r2)
        RefSeeds.update(self) #缓存键值 


class RadThRef(RadItvRef):
    """半径阈值引用"""
    def __init__(self, ref_seeds: RefSeeds, rth: float):
        self.ref_seeds = ref_seeds 
        self.rth = rth 
        RadItvRef.__init__(self, ref_seeds, rth, np.inf)

    def __str__(self):
        return f"{self.cls_str}(th:{self.rth} cnt:{self.cnt})"

    def dec_key(self) -> tuple:
        return self.ref_seeds, self.rth 

    def run_key(self) -> tuple: 
        return *self.ref_seeds.run_key(), self.rth, self._mod_time 
    
    def update(self):
        self.args = (self.rth, )
        RefSeeds.update(self) #缓存键值 

class RandRef(RefSeeds):
    """随机批量引用"""
    def __init__(self, ref_seeds: RefSeeds, prop: float):
        self.ref_seeds = ref_seeds
        self.prop = prop 
        RefSeeds.__init__(self, ref_seeds, RefSeeds.rand_ref, prop)

    def __str__(self):
        return f"{self.cls_str}(p:{self.prop} cnt:{self.cnt})"
    
    def dec_key(self) -> tuple:
        return self.ref_seeds, self.prop

    def run_key(self) -> tuple: 
        return *self.ref_seeds.run_key(), self.prop, self._mod_time 
    
    def update(self):
        self.args = (self.prop, )
        RefSeeds.update(self) #缓存键值 


class BatRef(RefSeeds):
    """批量引用基类 自动创建bat和curbat参数 在第一次使用时调用set_bat 更新状态"""
    def __init__(self, ref_seeds: RefSeeds, func: Callable,  n_bat: int=1, cur_bat: int=0, splitor=Splitor()):
        self.ref_seeds = ref_seeds 
        self.func = func 
        self.n_bat = n_bat
        self.cur_bat = cur_bat
        self.splitor = splitor
        RefSeeds.__init__(self, ref_seeds, func, n_bat, cur_bat, splitor)

    def dec_key(self) -> tuple:
        #在cur_bat改变时不更新
        return self.ref_seeds, self.func, self.n_bat, self.splitor

    def run_key(self) -> tuple:
        return *self.ref_seeds.run_key(), self.func, self.n_bat, self.cur_bat, self.splitor, self._mod_time

    def __str__(self):
        return f"{self.cls_str}(nbat:{self.n_bat} cnt:{self.cnt})"

    def update(self):
        self.args = (self.n_bat, self.cur_bat, self.splitor)
        RefSeeds.update(self) #缓存键值 
    

class RadBatRef(BatRef):
    """半径批量引用关系"""  
    #BUG (自身不允许嵌套 原因不明)
    def __init__(self, ref_seeds: RefSeeds, n_bat: int=1, cur_bat: int=0, splitor=MulOtsuRadSplitor()):
        BatRef.__init__(self, ref_seeds, RefSeeds.rad_bat_ref, n_bat, cur_bat, splitor)

    def mod_time(self):
        """底层数据更新 不进行引用更新传播 而是利用cur_bat在外部管理更新(半径批量间独立)"""
        self._mod_time = time.time() 

class MulRadBatRef(RadBatRef):
    """多半径批量引用 跨批量重建"""
    def __init__(self, ref_seeds: RefSeeds, n_bat: int=1, cur_bat: int=0, splitor=MulOtsuRadSplitor()):
        """定义跨批量引用函数,输入BatRef基类"""
        def ref_func(seeds, n_bat, cur_bat, splitor):
            if cur_bat < n_bat-1: 
                return seeds.rad_bat_ref(n_bat, cur_bat, splitor) + seeds.rad_bat_ref(n_bat, cur_bat+1, splitor)
            else: 
                return seeds.rad_bat_ref(n_bat, cur_bat, splitor)
        BatRef.__init__(self, ref_seeds, ref_func, n_bat, cur_bat, splitor)

class RndBatRef(BatRef):
    """随机批量引用关系"""
    def __init__(self, ref_seeds: RefSeeds, n_bat: int=1, cur_bat: int=0, splitor=RndSplitor()):
        BatRef.__init__(self, ref_seeds, RefSeeds.rnd_bat_ref, n_bat, cur_bat, splitor)

class QueBatRef(BatRef): 
    def __init__(self, ref_seeds: RefSeeds, n_bat: int=1, cur_bat: int=0, splitor=PropSplitor()):
        """循环队列引用"""
        BatRef.__init__(self, ref_seeds, RefSeeds.que_bat_ref, n_bat, cur_bat, splitor)

class RefOperator(RefSeeds):
    """引用操作符 
    """
    def __init__(self, seeds1: RefSeeds, seeds2: RefSeeds, func: Callable):
        #引用对象 本身不改变
        self.seeds1 = seeds1
        self.seeds2 = seeds2 
        #引用函数和参数
        self.func = func
        Seeds.__init__(self)
    
    def __repr__(self):
        return f"{self.cls_str} => s1:{self.seeds1.__str__()} s2:{self.seeds2.__str__()}"
        
    def __str__(self):
        return f"{self.cls_str}(s1:{self.seeds1.id_str} s2:{self.seeds2.id_str} cnt:{self.cnt})"

    def dec_key(self) -> tuple:
        return self.seeds1, self.seeds2, self.func

    def run_key(self) -> tuple:
        return *self.seeds1.run_key(), *self.seeds2.run_key(), self.func, self._mod_time

    def update(self):
        #重新建立引用
        seeds = self.func(self.seeds1, self.seeds2)
        #创建自身数据 更新refer 和 memo
        self.bdata = seeds.bdata
        self.idx = seeds.idx   #创建引用
        self._mod_time = time.time()  #引用创建时间
        Link.update(self) 

    def mod_time(self):
        """反向传播的时间更新"""
        self._mod_time = time.time() 
        self.seeds1.mod_time()  
        self.seeds2.mod_time()  #并集时间更新 (默认)
    

class AndRef(RefOperator):
    """交集引用"""
    def __init__(self, seeds1: RefSeeds, seeds2: RefSeeds):
        """seeds1的底层数据 在and中检查是否和seeds2一样
        seeds1, seeds2的索引变化将导致重新计算and 
        """
        RefOperator.__init__(self, seeds1, seeds2, RefSeeds.__and__) 

    def mod_time(self):
        """反向传播的时间更新"""
        self._mod_time = time.time() 
        self.seeds1.mod_time()  #交集时间更新

class OrRef(RefOperator):
    """并集引用"""
    def __init__(self, seeds1: RefSeeds, seeds2: RefSeeds):
        RefOperator.__init__(self, seeds1, seeds2, RefSeeds.__or__) 


class SubRef(RefOperator):
    def __init__(self, seeds1: RefSeeds, seeds2: RefSeeds):
        RefOperator.__init__(self, seeds1, seeds2, RefSeeds.__sub__)  #从1中减去2 

    def mod_time(self):
        """反向传播的时间更新"""
        self._mod_time = time.time() 
        self.seeds1.mod_time()  




#-------------------------------------------------------------------------------------#
# 边界物体
#-------------------------------------------------------------------------------------#

class Bound(Object):
    """边界物体,一般对应方向不设置周期性"""
    _ndim = 2 
    _r = 0.0 

    def __init__(self, axis: int=0, rpos=0.0, mark=1):
        """域边界
        Args:
            axis (int): 指定轴,012对应xyz
            rpos (int): 指定平面位置(指定轴方向的相对值)
        """
        self.axis = axis 
        self.rpos = rpos 
        Object.__init__(self, mark)

    def __str__(self):
        return f"{self.cls_str}(ax:{self.axis}[{self.rpos}] mark:{self.mark})"
    
    @auto_check
    def __getstate__(self):
        return {"axis": self.axis, "rpos": self.rpos, "mark": self.mark}

    def dec_key(self):
        return self.axis, tuple(self.rpos), self.mark

    @auto_memo
    def discrete(self, reg: Region) -> PhaseVol:
        vol = PhaseVol.zeros(reg.shape)
        uint_pos = min(int(self.rpos * reg.shape[self.axis]), reg.shape[self.axis]-1)
        key = tuple([uint_pos if self.axis==i else slice(None) for i in range(3)])
        vol[key] = self.mark
        return vol 
    
   
class FillBound(Bound):
    """填充边界场"""
    
    def __init__(self, axis: int=0, rpos: tuple[float]=(0.0, 0.5), mark=1):
        rpos = np.sort([pos % 1.0 if pos > 1.0 or pos < 0.0 else pos for pos in rpos]) 
        Bound.__init__(self, axis, list(rpos), mark)

    def dec_key(self):
        return self.axis, tuple(self.rpos), self.mark
    
    def __str__(self):
        return f"{self.cls_str}(ax:{self.axis}[{self.rpos[0]}~{self.rpos[1]}] mark:{self.mark})"
    
    @auto_memo
    def discrete(self, reg: Region) -> PhaseVol:
        vol = PhaseVol.zeros(reg.shape)
        uint_pos = tuple((np.array(self.rpos) * reg.shape[self.axis]).astype(np.int32)) 
        key = tuple([slice(*uint_pos) if self.axis==i else slice(None) for i in range(3)])
        vol[key] = self.mark
        return vol 
    
    @property
    def r(self) -> float:
        return (self.rpos[1] - self.rpos[0])/2 
    
    @r.setter 
    def r(self, value) -> float:
        self.rpos[0] = (self.rpos[1] + self.rpos[0])/2 - value
        self.rpos[1] = (self.rpos[1] + self.rpos[0])/2 + value 

    def copy(self):
        return self.__class__(self.axis, self.rpos, self.mark)


class Cylinder(Object):
    _ndim = 1 
    _r = 0.0 

    def __cly(self, x, y, z, ax, cx, cy, ch0, ch1, cr):
        ch0, ch1 = min(ch0, ch1), max(ch0, ch1)
        px, py = [loc for i,loc in enumerate([x,y,z]) if i != ax]
        pz = [x, y, z][ax]
        return ((px - cx)**2 + (py - cy)**2 - cr**2)<0 and pz > ch0 and pz < ch1 

    def __init__(self, axis: int=0, rpos: tuple[float]=(0.5, 0.5), r: float=0.0, h: tuple[float]=(0.0, 1.0), mark=1):
        """圆柱所在的轴 平面内坐标(域最长边相对值按012顺序) 半径(域最长边相对值)"""
        self.axis = axis 
        self.rpos = np.array(rpos) #另外两个维度的坐标 
        self.r = r 
        self.h = h 
        self.mark = mark 
        Node.__init__(self)

    def dec_key(self):
        return self.axis, tuple(self.rpos), self.r, tuple(self.h), self.mark
    
    def __str__(self):
        return f"{self.cls_str}(ax:{self.axis} ({self.rpos[0]},{self.rpos[1]}) r:{self.r} h:{self.h} mark:{self.mark})"
    
    @auto_memo
    def discrete(self, reg: Region) -> DataVol:
        vol = PhaseVol.zeros(reg.shape)
        
        rpos_len = tuple([reg.shape[ax_i] for ax_i in range(np.size(reg.shape)) if ax_i != self.axis])
        uint_cloc = tuple(rpos_len * np.array(self.rpos))
        uint_hloc = tuple(reg.shape[self.axis] * np.array(self.h))  #指定方向的相对值

        grid_key = tuple([slice(None, i_len) for i_len in vol.shape])
        mask = np.vectorize(self.__cly)(*np.mgrid[grid_key], self.axis, *uint_cloc, *uint_hloc, self.r * reg.vxl_len)
        vol[mask] = self.mark
        return vol

    def copy(self):
        return self.__class__(self.axis, self.rpos, self.r, self.h, self.mark)
    

class Cube(Object):
    def __init__(self, axis: int=0, rpos: tuple[float]=(0.5, 0.5, 0.5), h: tuple[float]=(0.1, 0.1, 0.1), mark=1):
        """圆柱所在的轴 平面内坐标(域最长边相对值按012顺序) 半径(域最长边相对值)"""
        self.axis = axis 
        self.rpos = np.array(rpos) #另外两个维度的坐标 
        self.h = h 
        self.mark = mark 
        Node.__init__(self)

    def dec_key(self):
        return self.axis, tuple(self.rpos), self.h, self.mark
    
    def __str__(self):
        return f"{self.cls_str}(ax:{self.axis} ({self.rpos[0]},{self.rpos[1]}) r:{self.r} h:{self.h} mark:{self.mark})"
    
    @auto_memo
    def discrete(self, reg: Region) -> DataVol:
        #TODO 
        return

    def copy(self):
        return self.__class__(self.axis, self.rpos, self.h, self.mark)

    @property
    def r(self) -> float:
        return np.min(self.h)
    
    @r.setter 
    def r(self, value) -> float:
        self.rpos[0] = (self.rpos[1] + self.rpos[0])/2 - value
        self.rpos[1] = (self.rpos[1] + self.rpos[0])/2 + value 
    

class ImgObject(Object, PhaseVol):
    """从三维数组建立的图像物体对象 用图像的voronoi覆盖剖分网格"""
    _ndim = 3 

    @auto_vol 
    def __init__(self, img = PhaseVol()): #, res = -1.0):
        """图像物体(单相物体)初始化"""
        self.img = PhaseVol(img)
        # self.res = 1/self.vol.vxl_len  #默认在单位长度图像
        Node.__init__(self)

    def dec_key(self) -> tuple:
        return id(self.img), #, self.res

    def run_key(self):
        return id(self.img.vol), 

    def __str__(self):
        return f"{self.cls_str}(img:{hex_id(self.img)} mk:{self.mark})"

    @auto_memo
    def discrete(self, reg: Region) -> PhaseVol:
        if not np.all(np.array(reg.shape) == self.img.shape):
            return self.img.resize(reg.shape)
        return self.img
    
    @property 
    def vol(self):
        return self.img.vol   #和phasevol一致, 可用phasevol函数

    @vol.setter
    def vol(self, value):
        self.img.vol = np.array(value, dtype=self.img.vol.dtype)  #和phasevol一致, 可用phasevol函数

    @property
    def inv(self) -> ImgObject:
        return ImgObject(self.img.inv)

    @property
    def size(self) -> np.ndarray:
        return np.array(self.img.shape)

    @property
    def r(self) -> float:
        return 1/self.img.vxl_len 

    # @r.setter 
    # def r(self, value) -> float:
    #     self.res = value 

    @property
    def mark(self) -> float:
        return self.img.max 

    @mark.setter 
    def mark(self, value) -> float:
        self.img[self.img[:]!=0] = value 

    def __getstate__(self):
        return {"img": self.img}
    
    def __setstate__(self, value):
        self.img = PhaseVol(np.array(value["img"][:]))


class FileImg(ImgObject):
    """从文件加载的图像对象"""
    _ndim = 3 

    @auto_vol 
    def __init__(self, fname="data/img.png", ths=[], n_smth=0, cuts=(slice(None), slice(None), slice(None)), process_func=lambda img: img):
        """图像物体(单相物体)初始化"""
        self.fname = fname 
        self.ths = ths 
        self.n_smth = n_smth
        self.process_func = process_func
        self.cuts = cuts
        self.update()

    def dec_key(self) -> tuple:
        return self.fname, tuple(self.ths), self.n_smth, self.cuts, self.process_func

    def run_key(self):
        return self.fname, tuple(self.ths), self.n_smth, self.cuts, self.process_func

    def __str__(self):
        return f"{self.cls_str}(f:{self.fname} ths:{self.ths})"
    
    @property 
    @auto_memo 
    def img(self):
        import os 
        phs_vol = PhaseVol()
        # if os.path.isfile(self.fname):
        phs_vol = DataVol.from_img(self.fname, self.cuts, process_func=self.process_func)
        # print(f"load image, range: {np.min(phs_vol[:])}~{np.max(phs_vol[:])}")
        if np.size(self.ths): phs_vol = phs_vol.threshold(self.ths)
        if self.n_smth and isinstance(phs_vol, PhaseVol): phs_vol = phs_vol.open_smooth(self.n_smth)
        return phs_vol 
    
    # @auto_vol 
    # def equals(self, value):
    #     return ImgObject(self.vol.__eq__(value))
        
    # @auto_vol 
    # def nequals(self, value):
    #     return ImgObject(self.vol.__ne__(value))
    

    # def __getstate__(self):
    #     return {"fname": self.fname, "ths": self.ths, "n_smth": self.n_smth, 
    #             "cuts": [[cut.start, cut.stop, cut.step] for cut in self.cuts]}
    
    # def __setstate__(self, value):
    #     self.__dict__.update(value)
    #     self.cuts = tuple([slice(cut.start, cut.stop, cut.step) for cut in value["cuts"]])


class CompObject(Object, Link):
    _r = 0.0 
    _ndim = 0 
    def __init__(self, objs: list[Object]=[]):
        """复合物体, 输入物体列表初始化"""
        self.objs = objs
        regs = [obj.__dict__["reg"] for obj in objs if "reg" in obj.__dict__]
        if len(regs):
            self.reg = regs[0]
        self.update()

    def dec_key(self):
        return *tuple([obj.dec_key() for obj in self.objs]),

    def run_key(self):
        return *tuple([obj.run_key() for obj in self.objs]),

    def discrete(self, reg: Region) -> PhaseVol:
        objs_vol = PhaseVol.zeros(reg.shape)
        for obj in self.objs:
            objs_vol += obj.discrete(reg)  #后者覆盖前者
        return objs_vol 
    
    def __str__(self):
        objs_str = "".join([f"{obj}-" for obj in self.objs])[:-1]
        return f"{self.cls_str}({objs_str})"

    @property
    def ndim(self) -> int:
        return max([obj.ndim for obj in self.objs])

    @property
    def r(self) -> float:
        return np.concatenate([np.reshape(obj.r, -1) for obj in self.objs]) 
    
    @r.setter  
    def r(self, value): 
        start = 0 
        value = np.reshape(value, -1)
        for obj in self.objs:
            end = start + np.reshape(obj.r, -1).size
            obj.r = value[start:end][0] if end-start==1 else value[start:end]
            start = end 

    @property
    def mark(self) -> float:
        return np.concatenate([np.reshape(obj.mark, -1) for obj in self.objs]) 

    @mark.setter  
    def mark(self, value): 
        """设置mark, obj可以是复合物体"""
        start = 0 
        value = np.reshape(value, -1)
        for obj in self.objs:
            end = start + np.reshape(obj.mark, -1).size
            obj.mark = value[start:end][0] if end-start==1 else value[start:end]
            start = end 

    @property
    def subs(self) -> set[Object]:
        subs = OrderSet([])
        for obj in self.objs:
            subs += obj.subs
        return subs
    
    # @property
    # def phs(self) -> set[Object]:
    #     phss = OrderSet([])
    #     for obj in self.subs:
    #         if isinstance(obj, PhaseObject):
    #             phss.add(obj.phs)
    #     return phss

    @auto_check
    def copy(self):
        return self.__class__([obj.copy() for obj in self.subs])
    


class CompImg(CompObject):
    @auto_vol 
    def __init__(self, vol: PhaseVol, res = 0.0):
        self.vol = PhaseVol(vol)
        CompObject.__init__(self, [ImgObject(self.vol[:]==attr, res) for attr in self.vol.attrs if attr != 0])
        self.mark = [attr for attr in self.vol.attrs if attr != 0]
        


#-------------------------------------------------------------------------------------#
# 域/采样点
#-------------------------------------------------------------------------------------#

class ObjectPnt(Link):
    cen_vxl = 0.0 #中心距离缩减 过小会导致体积分数膨胀,可能引起电解液填充问题

    def __init__(self, obj: Object, reg: Region, *args):
        """种子与其组成结构关系 支持层级渲染"""
        self.obj = obj 
        self.reg = reg 
        self.args = args 
        self.update()

    def __str__(self):
        return f"{self.cls_str}(obj:{self.obj})"
    
    def __repr__(self):
        return self.__str__()
    
    @auto_check
    def __getstate__(self):
        return {"obj": self.obj, "reg": self.reg}
    
    def dec_key(self) -> tuple:
        return self.obj, self.reg, *self.args

    def run_key(self) -> tuple:
        return *self.obj.run_key(), tuple(self.reg.shape), *self.args

    @property
    def ndim(self) -> int:
        return self.obj.ndim
    
    @property
    @auto_check 
    def pnt_r(self) -> float:
        return np.median(self.obj.r) 

    @property
    @auto_check 
    def nvxl(self) -> float:
        return max(self.pnt_r * self.reg.vxl_len - self.cen_vxl, self.cen_vxl)
    
    @auto_memo
    def ftr_trans(self, ftr_func: FeatureFunc):
        """缓存不同ftr的变换结果 srs变化时清空"""
        # print("srs trans")
        return ftr_func.transform(self)  

    def sur_ftr_vol(self, tgt: SeedsPnt = None, ftr_func: FeatureFunc = surdt):
        """到目标的表面距离场 tgt变化时需要重新计算sur_dist"""
        dist_vol = self.ftr_trans(ftr_func)     #表面体素距离(保持半径定义的一致性)
        return DistVol((dist_vol[:] - ftr_func.sur_dist(tgt)) / dist_vol.vxl_len)  #相对域的表面距离,注意和edgeopteration中保持一致
    
    def cen_ftr_vol(self, srs: SeedsPnt = None, ftr_func: FeatureFunc = surdt):
        """起点到目标中心距离场"""
        dist_vol = self.ftr_trans(ftr_func)     #表面体素距离
        return DistVol((dist_vol[:] + ftr_func.sur_dist(srs)) / dist_vol.vxl_len)  #假设tgt外部和srs内部对称, 放弃对称性假设
        
    # @auto_memo
    # def dense_func(self, tphs_pnt: SeedsPnt, ftr_func: FeatureFunc, dns_func: DistFunc):
    #     """用于缓存点相关函数 缓存目标相种子输入的结果 仅用于SA算法"""
    #     sur_dist_vol = self.sur_ftr_vol(tphs_pnt, ftr_func)
    #     srs_dns_func = DistFunc(dns_func.dist)  #源采样点样本建立分布函数
    #     srs_dns_func.fit(sur_dist_vol[ftr_func.tgt_transform(tphs_pnt)])  #计算tgt占据体素的点相关
    #     return srs_dns_func

    @property 
    @auto_memo 
    def vol(self) -> PhaseVol:  #随obj更新
        return self.obj.discrete(self.reg)

    @property 
    @auto_memo 
    def seeds(self) -> Seeds:  
        return LaguerreVoro.from_vol(self.vol).seeds 

    @property
    @auto_check 
    def obj_cnt(self) -> int: 
        return self.obj.cnt 



#-------------------------------------------------------------------------------------#
# a-shape参数 用于描述一个相构成的区域
#-------------------------------------------------------------------------------------#

class AlphaShape(Node):
    """种子占据域的形状 alpha-shape计算修正"""
    def __init__(self, rpos=1.0, rfix=0.6) -> None:
        self.rpos = rpos  #相对位置
        self.rfix = rfix  #rfix取0表示不修正该相
        self.measure = fedt 
        self.splitor = MulOtsuRadSplitor()
        self.update() 

    def dec_key(self) -> tuple:
        return self.rpos, self.rfix, 
    
    def __str__(self):
        return f"{self.cls_str}(ps:{self.rpos} fx:{self.rfix})"

    def a_shape(self, vol, r, reg):
        n_voxel = self.rfix * self.rpos * r * reg.vxl_len
        return PhaseVol.a_shape(vol, n_voxel, measure=self.measure)
    
    def __getstate__(self):
        return {"rpos": self.rpos, "rfix": self.rfix}
    
    def __setstate__(self, value):
        self.rpos = value["rpos"]
        self.rfix = value["rfix"]


class InvAShape(AlphaShape):
    def a_shape(self, vol, r, reg):
        n_voxel = self.rfix * self.rpos * r * reg.vxl_len
        return PhaseVol.inv_a_shape(vol, n_voxel, measure=self.measure)


class PolyShape(AlphaShape):
    """晶体间隙形状修正
    Example:
        reg = Region(1, [240, 1, 240])  #分辨率/域边长
        vcfg.reg = reg 
        seeds = Seeds.load("FCell_fit")
        vol = seeds.discrete(reg)
        PolyShape().a_shape(vol, seeds.r_median, reg)
    """
    def __init__(self, rpos=1.0, rfix=0.1) -> None:
        AlphaShape.__init__(self, rpos, rfix)

    def a_shape(self, vol, r, reg):
        n_voxel = self.rfix * self.rpos * r * reg.vxl_len
        return PhaseVol.seg_erosion(vol, n_voxel, measure=self.measure)
    

class VoroShape(AlphaShape):
    """晶体形态修正"""
    def __init__(self, phs=Phase(UniformDist(), 0.0, 10000)) -> None:
        self.rpos = 1.0 
        self.phs = phs
        self.rfix = 0.0
        self.seeds = None
        self.measure = fedt 
        self.splitor = MulOtsuRadSplitor()
        self.update() 

    def dec_key(self) -> tuple:
        return self.phs, 
    
    def __str__(self):
        return f"{self.cls_str}(p:{self.phs.__str__()})"

    def a_shape(self, vol, r, reg):
        if type(self.seeds) == type(None):
            return vol 
        elif self.seeds.cnt == 0:
            return vol 
        else: 
            seeds_vol = self.seeds.discrete(reg) + vol
            voro_vol = seeds_vol.watershed()
            voro_vol[voro_vol[:] == self.phs.mark] = 0
            return voro_vol  #保持vol的其他属性不变

    def __getstate__(self):
        return {"rpos": self.rpos, "rfix": self.rfix, "phs": self.phs}
    
    def __setstate__(self, value):
        self.rpos = value["rpos"]
        self.rfix = value["rfix"]
        self.phs = value["phs"]

class SeedsShape(AlphaShape):
    """种子外围形状 不进行修正"""
    def __init__(self, rpos=1.0) -> None:
        self.rpos = rpos  #相对位置
        AlphaShape.__init__(self, rpos, 0.0)

    def dec_key(self) -> tuple:
        return self.rpos, 
    
    def __repr__(self):
        return f"{self.cls_str}(pos:{self.rpos})"
    
    def a_shape(self, vol, r, reg):
        n_voxel = self.rfix * self.rpos * r * reg.vxl_len
        return PhaseVol.seeds_shape(vol, n_voxel, measure=self.measure)


class AnisoShape(AlphaShape):
    """各向异性形状,有边界限制"""


#-------------------------------------------------------------------------------------#
# 种子占据域 支持半径修正
#-------------------------------------------------------------------------------------#

class SeedsPnt(ObjectPnt):
    def __init__(self, seeds: Seeds, ashp: AlphaShape, reg: Region):
        """种子与其组成结构关系 支持层级渲染"""
        self._seeds = seeds 
        self.ashp = ashp 
        ObjectPnt.__init__(self, seeds, reg, ashp)  #种子作为obj

    def dec_key(self) -> tuple:
        return self._seeds, self.ashp, self.reg, 
    
    def run_key(self) -> tuple:
        return *self._seeds.run_key(), *self.ashp.run_key(), tuple(self.reg.shape)  #region内缓存shape值 在reg修改前为固定id
    
    def __str__(self):
        return f"{self.cls_str}(s:{self._seeds.__str__()} alp(rp:{self.ashp.rpos} rf:{self.ashp.rfix}))"

    @auto_check
    def __getstate__(self):
        return {"_seeds": self._seeds, "ashp": self.ashp, "reg": self.reg}
       
    @property
    def mark(self):
        return np.max(self.seeds.mark)

    @property
    def sub_marks(self):
        return np.unique(self.seeds.mark)

    @property
    @auto_check 
    def pnt_r(self):
        return self.obj.r_median * self.ashp.rpos

    @property 
    @auto_memo
    def vol(self):
        # return self.seeds.discrete_dmn(self.reg, self.ashp)  
        return self.seeds.discrete(self.reg, self.ashp.rpos)

    @classmethod
    def from_phs_obj(cls, phs_obj: PhaseObject, seeds: Seeds, reg: Region):
        if isinstance(phs_obj, ConstPhaseObject): 
            return ConstSeedsPnt(seeds, phs_obj.ashp, reg) 
        elif isinstance(phs_obj, PhaseObject): 
            return SeedsPnt(seeds, phs_obj.ashp, reg) 
        else: 
            raise NotImplementedError(f"Init seeds point fromm phase object failed: unkown phase object{phs_obj}")

    @property
    @auto_check 
    def obj_cnt(self) -> int: 
        return self.seeds.cnt
    
    @property
    def seeds(self) -> Seeds:
        return self._seeds   


class ConstSeedsPnt(SeedsPnt):
    """固定的种子点"""


class CumSeedsPnt(SeedsPnt):
    """累积种子采样点 缓存层级渲染结果"""
    def __init__(self, seeds: RefSeeds, rad_bat: RadBatRef, ashp: AlphaShape, reg: Region):
        """种子与其组成结构关系 支持层级渲染 认为当前输入种子集合已不再改变"""
        self._seeds = seeds  #当前批量,而非累计批量 活动对象
        self.rad_bat = rad_bat  #seeds引用/自身对应的累积批量划分
        self.ashp = ashp
        ObjectPnt.__init__(self, seeds, reg, ashp)

    def dec_key(self) -> tuple:
        return self._seeds, self.ashp, self.reg, self.rad_bat.n_bat #不随cur_bat变化
    
    def run_key(self) -> tuple:
        return self._seeds, *self.ashp.run_key(), tuple(self.reg.shape), self.rad_bat.n_bat  #region内缓存shape值 在reg修改前为固定id
    
    def __str__(self):
        return f"{self.cls_str}(s:{self._seeds.__str__()} bat:{self.rad_bat.__str__()} alp:{self.ashp.rpos})"

    @auto_check
    def __getstate__(self):
        return {"_seeds": self._seeds, "ashp": self.ashp, "reg": self.reg, "rad_bat": self.rad_bat}

    @auto_memo #根据cur_bat返回不同记录结果
    def get_bat_vol(self, cur_bat):
        """调用此函数时 对应seeds的cur_bat之上累积批量已不再改变"""
        #缓存当前批量索引
        save_bat = self.rad_bat.cur_bat

        #切换当前批量,进行渲染
        self.rad_bat.cur_bat = cur_bat #seeds随之改变
        vol = self._seeds.discrete_dmn(self.reg, self.ashp) 
        
        #还原当前批量
        self.rad_bat.cur_bat = save_bat
        return vol

    def get_bat_seeds(self, cur_bat):
        if cur_bat < self.rad_bat.n_bat and cur_bat >= 0: 
            save_bat = self.rad_bat.cur_bat
            self.rad_bat.cur_bat = cur_bat #存储的层级_seeds随之改变
            seeds = self._seeds[:]
            self.rad_bat.cur_bat = save_bat
            return seeds 
        else: 
            return Seeds(self.rad_bat.bdata)
    
    @auto_memo 
    def get_cum_vol(self, cur_bat):
        """按照输入cur_bat缓存/读取数据 本身不起作用"""
        vol = PhaseVol.zeros(self.reg.shape)   #vol从0开始计算
        for bat_i in range(min(self.rad_bat.cur_bat, cur_bat)):  #直到当前批量 cur_bat最大值不超过输入cur_bat
            vol += self.get_bat_vol(bat_i)  #为第0个半径批量时 累积采样点为空
        return vol
    
    @property 
    @auto_check  #进行更新检查 不缓存结果 cumbat特殊行为
    def vol(self) -> PhaseVol:
        """根据缓存的种子渲染结果 实时计算当前cur_bat时的数组 因为pnt本身不随cur_bat改变,此为vol的行为"""
        #获取直到cur_bat的所有体素网格
        #注意:vol在整个过程中不变
        return self.get_cum_vol(self.rad_bat.cur_bat)

    @property
    @auto_check
    def seeds(self) -> Seeds:
        seeds = Seeds(self.rad_bat.bdata)
        #NOTE 考虑维进入循环的情况 _data保持一致
        for bat_i in range(min(self.rad_bat.cur_bat, self.rad_bat.n_bat)):  #直到当前批量 cur_bat最大值不超过输入cur_bat
            seeds += self.get_bat_seeds(bat_i)  #为第0个半径批量时 累积采样点为空
        return seeds
    


class HrcSeedsPnt(SeedsPnt):
    """固定的层级渲染种子"""
    def __init__(self, seeds: RefSeeds, ashp: AlphaShape, reg: Region, n_hrc: int=1):
        """种子与其组成结构关系 构造完成的层级渲染"""
        self.n_hrc = n_hrc 
        SeedsPnt.__init__(self, seeds, ashp, reg)
    
    def dec_key(self) -> tuple:
        return self._seeds, self.ashp, self.reg, self.n_hrc #不随cur_bat变化
    
    def run_key(self) -> tuple:
        return self._seeds, *self.ashp.run_key(), tuple(self.reg.shape), self.n_hrc  #region内缓存shape值 在reg修改前为固定id

    @auto_check
    def __getstate__(self):
        return {"_seeds": self._seeds, "ashp": self.ashp, "reg": self.reg, "n_hrc": self.n_hrc}

    @property 
    @auto_memo
    def vol(self):
        if isinstance(self.ashp, VoroShape): 
            return self.seeds.discrete_dmn_hrc(self.reg, self.ashp, 1)  
        return self.seeds.discrete_dmn_hrc(self.reg, self.ashp, self.n_hrc)  

class SeedsPntOperator(SeedsPnt):
    """采样点操作 两个采样点为相同类型的Pnt"""
    def __init__(self, pnt1: ObjectPnt, pnt2: ObjectPnt, func: Callable):
        self.pnt1 = pnt1 
        self.pnt2 = pnt2 
        self.func = func 
        self.reg = pnt1.reg
        self.update()

    def __str__(self):
        return f"{self.cls_str}(p1:{self.pnt1} p2:{self.pnt2})"

    def dec_key(self) -> tuple:
        return self.pnt1, self.pnt2 
    
    def run_key(self) -> tuple:
        return *self.pnt1.run_key(), *self.pnt2.run_key() 

    @property
    def obj(self) -> CompObject:    
        return CompObject([self.pnt1.obj, self.pnt2.obj])

    @property
    @auto_memo  #捕获pnt1和pnt2变化 
    def vol(self) -> PhaseVol:
        return self.func(self.pnt1.vol, self.pnt2.vol) 

    @property
    @auto_check
    def ndim(self) -> int:
        """返回两个采样点更大的维度"""
        return max(self.pnt1.obj.ndim, self.pnt2.obj.ndim)

    @property 
    @auto_check
    def pnt_r(self) -> float:
        return max(self.pnt1.pnt_r, self.pnt2.pnt_r)

    @property
    @auto_check
    def seeds(self) -> Seeds:  #NOTE: or 和 __or__不同
        return self.pnt1.seeds.__or__(self.pnt2.seeds) #只保留不重复的种子集合 重叠种子体素重叠

    @property 
    @auto_check
    def obj_cnt(self) -> int:
        return max(self.pnt1.obj_cnt, self.pnt2.obj_cnt)

class AddSeedsPnt(SeedsPntOperator):
    def __init__(self, pnt1: ObjectPnt, pnt2: ObjectPnt):
        SeedsPntOperator.__init__(self, pnt1, pnt2, PhaseVol.__add__)
    
    @property
    @auto_memo  #捕获pnt1和pnt2变化 
    def vol(self) -> PhaseVol:
        return self.func(self.pnt1.vol, self.pnt2.vol) 

    @property 
    @auto_check
    def pnt_r(self) -> float:
        return min(self.pnt1.pnt_r, self.pnt2.pnt_r)

    @property 
    @auto_check
    def obj_cnt(self) -> int:
        return self.pnt1.obj_cnt + self.pnt2.obj_cnt




#-------------------------------------------------------------------------------------#
# 相采样点和相间关系 (用于声明API)
#-------------------------------------------------------------------------------------#

class PhaseObject(Object):
    """相的采样点 为相种子添加alphashape描述 用于点相关图定义声明顶点"""
    _r = 0.0 
    _ndim = 0 
    def __init__(self, phs = Phase(), ashp = AlphaShape()):
        """种子组成结构占据的域"""
        self.phs = phs 
        self.ashp = ashp 
        self.update()

    def dec_key(self) -> tuple:
        return self.phs, self.ashp 
    
    def run_key(self) -> tuple:
        return *self.phs.run_key(), *self.ashp.run_key()
    
    def __str__(self):
        return f"{self.cls_str}(phs:{self.phs} alp:{self.ashp.rpos})"
    
    def discrete(self, reg: Region) -> PhaseVol:
        """非实体,仅用于可视化"""
        return self.phs.sample(reg).discrete_dmn_hrc(reg, self.ashp, 4)
    
    @auto_check
    def __getstate__(self):
        return {"phs": self.phs, "ashp": self.ashp}
    

class ConstPhaseObject(PhaseObject):
    """固定的相物体 不在迭代中更新坐标"""

        
class ObjectRelation(Node):
    """相的采样点 为相种子添加alphashape描述 用于点相关图定义声明顶点(只与连通关系有关)"""
    def __init__(self, srs = Object(), tgt = PhaseObject(), ftr_func = FeatureFunc(), dns_func = DistFunc()):
        """种子组成结构占据的域 支持复合相种子声明为目标 在后续种子集合解析中划分"""
        self.srs = srs 
        self.tgt = tgt
        self.ftr_func = ftr_func 
        self.dns_func = dns_func
        Node.update(self) 

    def dec_key(self) -> tuple:
        return self.srs, self.tgt, self.ftr_func, self.dns_func
    
    def run_key(self) -> tuple:
        return *self.srs.run_key(), *self.tgt.run_key(), *self.ftr_func.run_key(), *self.dns_func.run_key()
    
    def __str__(self):
        return f"{self.cls_str}({self.srs} => {self.tgt})"

    @auto_check
    def __getstate__(self):
        return {"srs": self.srs, "tgt": self.tgt, "ftr_func": self.ftr_func, "dns_func": self.dns_func}

    # def __getstate__(self):
    #     return {"srs": self.srs, "tgt": self.tgt, "ftr_func": self.ftr_func, "dns_func": self.dns_func}
    
    def tgt_itsc(self, phss: set[Phase]) -> set:
        """目标相的交集"""
        phs_subs: set[Phase] = OrderSet()
        for phs in OrderSet(phss):
            phs_subs.update(phs.subs) 
        # return set(self.tgt.phs.sub_marks).intersection(set([phs.mark for phs in phs_subs]))
        return self.tgt.phs.subs.intersection(phs_subs)
    
    def srs_itsc(self, phss: set[Phase]) -> set:
        """源相的交集"""
        if not isinstance(self.srs, PhaseObject):
            return OrderSet()
        else: 
            phs_subs: set[Phase] = OrderSet()
            for phs in OrderSet(phss):
                phs_subs.update(phs.subs)
            return self.srs.phs.subs.intersection(phs_subs)

    def copy(self):
        return self.__class__(self.srs, self.tgt, self.ftr_func, self.dns_func)
        
    
class PhaseRelation(ObjectRelation):
    """相的采样点 为相种子添加alphashape描述 用于点相关图定义声明顶点(只与连通关系有关)"""
    def __init__(self, srs = Object(), tgt = PhaseObject(), ftr_func = FeatureFunc(), dns_func = DistFunc()):
        """种子组成结构占据的域 支持复合相种子声明为目标 在后续种子集合解析中划分"""
        self.srs = srs 
        self.tgt = tgt
        self.ftr_func = ftr_func 
        self.dns_func = dns_func
        Node.update(self) 
    
    def __str__(self):
        return f"{self.cls_str}({self.srs.phs} => {self.tgt.phs})"

    @property
    def inv(self):
        return self.__class__(self.tgt, self.srs, self.ftr_func, self.dns_func)



#-------------------------------------------------------------------------------------#
# 种子间关系 (用于重建)
#-------------------------------------------------------------------------------------#


class SeedsPntCor(Node):
    """种子间的点相关关系 输入seedspnt"""
    def __init__(self, srs: ObjectPnt, tgt: SeedsPnt, ftr_func: FeatureFunc, dns_func: DistFunc):
        """种子组成结构占据的域 支持复合相种子声明为目标 在后续种子集合解析中划分"""
        self.srs = srs 
        self.tgt = tgt
        self.ftr_func = ftr_func 
        self.dns_func = dns_func
        self.update() 

    def dec_key(self) -> tuple:
        return self.srs, self.tgt, self.ftr_func, self.dns_func
    
    def __str__(self):
        return f"{self.cls_str}({self.srs} => {self.tgt})"


#-------------------------------------------------------------------------------------#
# 点相关操作
#-------------------------------------------------------------------------------------#

class PCEdgeOperation(Link):
    """单次点相关操作"""
    
    def __init__(self, pcfs: set[SeedsPntCor], *args): 
        self.pcfs = OrderSet(pcfs)
        self.update()

    def __str__(self):
        return f"{self.cls_str}(pcfs:{self.pcfs})"

    def dec_key(self) -> tuple:
        return *tuple(self.pcfs), 

    def run_key(self) -> tuple:
        return *tuple([pcf.run_key() for pcf in self.pcfs]),

    
class PCEdgeSample(PCEdgeOperation):

    def __init__(self, pcfs: set[SeedsPntCor]): 
        """视作所有pcf有相同的目标 概率场独立相乘采样"""
        PCEdgeOperation.__init__(self, pcfs)

    def update(self):
        pcfs = list(self.pcfs)
        if not len(pcfs): Link.update(self); return 

        pcf = pcfs[0]
        if pcf.tgt.obj_cnt <= 0: Link.update(self); return  #空种子退出  不需要更新坐标的目标对象
        
        prob_vol = ProbVol.ones(pcf.srs.reg.shape) #初始化为1 
        for pcf in pcfs:  #无pcfs则直接返回均匀prob
            # print(f"pcf sample: {str(pcf)}")
            if not np.any(pcf.srs.vol[:]): continue   #空源物体跳过 采用均匀概率
            if not isinstance(pcf.ftr_func, NNFeatureFunc): raise ValueError(f"edge sample failed: unsupported feature {pcf.ftr_func}")

            #距离变换
            # print(f"dist trans")
            sur_dist_vol = pcf.srs.sur_ftr_vol(pcf.tgt, pcf.ftr_func) 
            sur_dists = sur_dist_vol[:].reshape(-1, pcf.ftr_func.n_feature)

            if not (pcf.ftr_func.n_feature == pcf.dns_func.n_ftr):
                raise ValueError(f"{pcf} sample error: unmatched features for density function{pcf.dns_func.n_ftr} and feature transfrom{pcf.ftr_func.n_feature}")

            #拟合时采用全局距离 则进行全局采样
            #dns_func保存的是全局点相关函数
            if  PCEdgeFit.glb_fit:
                #对源和目标相同采用全局采样
                #BUG 首次采样srs.vol为空
                #NOTE 物体和种子不可使用相同索引, 不同相应该使用不同索引, 否则物体可能会影响此处的判断
                if pcf.tgt.obj_cnt <= 0: Link.update(self); return 
                if isinstance(pcf.srs, SeedsPnt) and pcf.tgt.mark in pcf.srs.vol[:]: 
                # if isinstance(pcf.srs, PhaseObject) and pcf.tgt.phs.subs.intersection(pcf.srs.phs.subs):
                    # print(f"density")
                    prob_list = pcf.dns_func.density(sur_dists)  #内部相互作用边, 用全局点相关函数迭代
                #不同则计算最近邻距离
                else:  
                    # print(f"nn density")
                    prob_list = pcf.dns_func.nndensity(sur_dists)  #外部相互作用边, 用最近邻点相关函数
            else: 
                #只对源和目标相同时有全局采样 则直接采样
                # print(f"density")
                prob_list = pcf.dns_func.density(sur_dists)   #保存时区分了内外部作用边, 直接采样
            
            #存在一个零概率则退出 不进行采样
            # print("prob vol")
            cur_prob_vol = ProbVol(prob_list.reshape(sur_dist_vol.shape[:-1]))
            if cur_prob_vol.is_zero: break 

            #归一化缩放计算 按照估计的采样点数量归一化
            # print(f"calc normalization")
            uni_fac = max(pcf.srs.reg.ndim - pcf.srs.ndim - 1.0, 0.0)  #归一化指数
            if uni_fac > 0.0:
                cen_dist_vol = pcf.srs.cen_ftr_vol(pcf.tgt, surdt)     #NOTE 概率归一化, 采用欧拉中心距离
                uni_prob_vol = ProbVol((cen_dist_vol[:]**uni_fac).reshape(cur_prob_vol.shape))   #采样点数量归一化概率, 转化为概率数组(3维)
                #均匀概率 不需要归一化
                cur_prob_vol = cur_prob_vol/uni_prob_vol if not uni_prob_vol.is_uni else cur_prob_vol

            #独立相乘
            # print(f"prob multiply")
            # prob_vol *= pcf.ftr_func.local_prod(cur_prob_vol, pcf.tgt.nvxl)
            prob_vol *= cur_prob_vol

        #目标种子修改标志, 其修改标识将逐层向上传播 直到phaseref
        if not prob_vol.is_zero: #不为0概率则采样
            pcf.tgt.seeds.loc = prob_vol.sample(pcf.tgt.seeds.cnt)  #底层数据更新 需要反向传播更新时间

        Link.update(self)

class PCEdgeSampleInit(PCEdgeOperation):

    def __init__(self, pcfs: set[SeedsPntCor]): 
        """视作所有pcf有相同的目标 概率场独立相乘采样"""
        PCEdgeOperation.__init__(self, pcfs)

    def update(self):
        pcfs = list(self.pcfs)
        if not len(pcfs): Link.update(self); return 

        pcf = pcfs[0]
        if pcf.tgt.obj_cnt <= 0: Link.update(self); return  #空种子退出  不需要更新坐标的目标对象
        
        prob_vol = ProbVol.ones(pcf.srs.reg.shape) #初始化为1 
        for pcf in pcfs:  #无pcfs则直接返回均匀prob
            if not np.any(pcf.srs.vol[:]): continue   #空源物体跳过 采用均匀概率
            if not isinstance(pcf.ftr_func, NNFeatureFunc): raise ValueError(f"edge sample failed: unsupported feature {pcf.ftr_func}")

            #距离变换
            sur_dist_vol = pcf.srs.sur_ftr_vol(pcf.tgt, pcf.ftr_func) 
            sur_dists = sur_dist_vol[:].reshape(-1, pcf.ftr_func.n_feature)

            if not (pcf.ftr_func.n_feature == pcf.dns_func.n_ftr):
                raise ValueError(f"{pcf} sample error: unmatched features for density function{pcf.dns_func.n_ftr} and feature transfrom{pcf.ftr_func.n_feature}")

            #初始化采用均匀概率
            if pcf.tgt.obj_cnt <= 0: Link.update(self); return 
            prob_list = cst.density(sur_dists) 
            
            cur_prob_vol = ProbVol(prob_list.reshape(sur_dist_vol.shape[:-1]))
            prob_vol *= cur_prob_vol

        if not prob_vol.is_zero: #不为0概率则采样
            pcf.tgt.seeds.loc = prob_vol.sample(pcf.tgt.seeds.cnt)

        Link.update(self)


class PCEdgeFit(PCEdgeOperation):
    """边的拟合 采样的对称过程"""
    glb_fit = True  #true 采用全局点相关/false采用最近点相关 #是否始终存储最近邻距离 / 否则仅在srs tgt不同时存储
    def __init__(self, pcfs: set[SeedsPntCor]): 
        PCEdgeOperation.__init__(self, pcfs)

    def update(self):
        pcfs = list(self.pcfs)
        if not len(pcfs):
            Link.update(self)
            return 
        pcf = pcfs[0] 
        if pcf.tgt.obj_cnt <= 0:  #空目标种子退出
            Link.update(self)
            return

        for pcf in self.pcfs:
            if isinstance(pcf.dns_func, ConstDistFunc): continue  #不拟合的分布对象 跳过(更新pcf前)
            if not np.any(pcf.srs.vol[:]): continue   #空目标物体(更新pcf后)跳过
            if not isinstance(pcf.ftr_func, NNFeatureFunc): raise ValueError(f"edge fit failed: unsupported feature {pcf.ftr_func}")

            #内部边(tgt和srs均为同一种子相) 交换源和目标
            #相内部迭代需要采用全局距离分布
            #NOTE 当srs和tgt数目差距较大时, 需要设置更小的fit_assp随机划分比例, 否则接近最近距离
            if self.glb_fit:
                #统一采用全局距离
                # (srs, tgt) = (pcf.tgt, pcf.srs)  #交换, 统计全局
                srs = pcf.srs
                tgt = pcf.tgt
            else:
                #对源和目标相同相采用全局距离
                if pcf.tgt.obj_cnt <= 0: Link.update(self); return 
                if pcf.tgt.mark in pcf.srs.vol[:]:
                # if isinstance(pcf.srs, PhaseObject) and pcf.tgt.phs.subs.intersection(pcf.srs.phs.subs):
                    (srs, tgt) = (pcf.tgt, pcf.srs)
                else: 
                    (srs, tgt) = (pcf.srs, pcf.tgt)  #不交换, 统计最近邻
            
            #距离变换  #为较小种子补充半径 缓解表面重叠
            sur_dist_vol = srs.sur_ftr_vol(tgt, pcf.ftr_func) #相对域的表面距离
            # sur_dist_vol[:] += 2 * pcf.ftr_func.cen_r / sur_dist_vol.vxl_len #拟合时补充一定距离

            #距离采样 
            #TODO debug
            ftr_list = sur_dist_vol.query(tgt.seeds.loc) #必须和种子对应

            #pcf中输入的全局seeds的底层坐标将被修改
            pcf.dns_func.fit(ftr_list)

        Link.update(self)


class PCEdgeCheck(PCEdgeOperation):
    """边的检查 移除不合要求的种子"""
    def __init__(self, pcfs: set[SeedsPntCor]): 
        PCEdgeOperation.__init__(self, pcfs)

    def update(self):
        pcfs = list(self.pcfs)
        if not len(pcfs):
            # print("non relation return")
            Link.update(self)
            return 
        pcf = pcfs[0] 
        if pcf.tgt.obj_cnt <= 0:  #空目标种子退出
            # print("non object return")
            Link.update(self)
            return

        for pcf in self.pcfs:
            # print(f"pcf check: {pcf}")
            # if isinstance(pcf.dns_func, ConstDistFunc): continue  #不拟合的分布对象 跳过(更新pcf前)
            # if pcf.srs.obj_cnt <= 0: continue   #空源物体(更新pcf后)跳过
            if not np.any(pcf.srs.vol[:]): continue 
            if not isinstance(pcf.ftr_func, NNFeatureFunc): raise ValueError(f"edge fit failed: unsupported feature {pcf.ftr_func}")

            #距离变换  #为较小种子补充半径 缓解表面重叠
            # print(f"sur dt")
            sur_dist_vol = pcf.srs.sur_ftr_vol(pcf.tgt, pcf.ftr_func) #相对域的表面距离
            # sur_dist_vol[:] += 2 * pcf.ftr_func.cen_r / sur_dist_vol.vxl_len #拟合时补充一定距离

            #距离采样 
            # print(f"query")
            ftr_list = sur_dist_vol.query(pcf.tgt.seeds.loc) #必须和种子对应
            mask = pcf.dns_func.invld_data_mask(ftr_list, tol=pcf.tgt.seeds.r_median/8)[:, 0]

            #pcf中输入的全局seeds的底层坐标将被修改
            # print(f"remove {np.where(mask)[0][:10]}")
            pcf.tgt.seeds.mask_hide(mask)  #重叠种子不再作为源被渲染 也不作为目标参与距离分布提取
        Link.update(self)


class PCFuncFit(PCEdgeOperation):
    def __init__(self, pcfs: set[SeedsPntCor]): 
        PCEdgeOperation.__init__(self, pcfs)

    def update(self):
        pcfs = list(self.pcfs)
        if not len(pcfs):
            Link.update(self)
            return 
        pcf = pcfs[0] 
        if pcf.tgt.obj_cnt <= 0:  #空目标种子退出
            Link.update(self)
            return

        for pcf in self.pcfs:
            if isinstance(pcf.dns_func, ConstDistFunc): continue  #不拟合的分布对象 跳过(更新pcf前)
            if pcf.srs.obj_cnt <= 0: continue   #空目标物体(更新pcf后)跳过
            if not isinstance(pcf.ftr_func, Std2PntCor): raise ValueError(f"edge pcf fit failed: unsupported feature {pcf.ftr_func}")
            
            #内部边(tgt和srs均为同一种子相) 源和目标相同
            if pcf.tgt.obj_cnt <= 0: Link.update(self); return 
            if pcf.tgt.mark in pcf.srs.vol[:]:
            # if isinstance(pcf.srs, PhaseObject) and pcf.tgt.phs.subs.intersection(pcf.srs.phs.subs):
                (srs, tgt) = (pcf.tgt, pcf.tgt)  
            else:
                (srs, tgt) = (pcf.srs, pcf.tgt)

            #pcf中输入的全局seeds的底层坐标将被修改
            fit_params = DistFunc.from_pcpnt(srs, tgt, pcf.ftr_func).params
            pcf.dns_func.update_params(fit_params)

        Link.update(self)

class PCEdgeMove(PCEdgeOperation):
    """点相关图边的移动和接受检验"""
    def __init__(self, pcfs: set[SeedsPntCor], T: float=1.0, cur_error=None): 
        """视作所有pcf有相同的目标"""
        self.T = T  #当前迭代温度
        self.cur_error = cur_error
        PCEdgeOperation.__init__(self, pcfs)

    def update(self):
        pcfs = list(self.pcfs) 
        if not len(pcfs):
            Link.update(self)
            return 
        pcf = pcfs[0]
        if isinstance(pcf.tgt, ConstSeedsPnt) or pcf.tgt.seeds.cnt <= 0:  #空种子退出  不需要更新坐标的目标对象
            Link.update(self)
            return

        #移动
        cur_seeds = RandRef(pcf.tgt.seeds, 1e-8)
        mod_seeds = cur_seeds.copy()  #注意seeds坐标更新将导致mod标志变化 引用链将同时更新 因此在复制上检验点相关函数变化
        mod_seeds.move(pcf.srs)  #和srs无碰撞的移动 srs可以和tgt相同
        
        #首次更新当前误差
        if self.cur_error == None: 
            self.cur_error = 0 
            for pcf in pcfs:
                if pcf.srs.obj_cnt <= 0: continue  #空源物体边的误差视作0

                self.cur_error += dist_error(pcf.dns_func, DistFunc.from_pcpnt(pcf.srs, pcf.tgt)) / len(pcfs)

        #累积误差 当前/修改后
        mod_error = 0
        for pcf in pcfs:
            if pcf.srs.obj_cnt <= 0: continue  #空源物体边的误差视作0
            if not isinstance(pcf.ftr_func, Std2PntCor): raise ValueError(f"edge move failed: unsupported feature {pcf.ftr_func}")

            mod_tgt_pnt = SeedsPnt(mod_seeds, pcf.tgt.ashp, pcf.tgt.reg)
            sub_tgt_pnt = SeedsPnt(SubRef(pcf.tgt.seeds, cur_seeds), pcf.tgt.ashp, pcf.tgt.reg) 
            mod_pnt = AddSeedsPnt(sub_tgt_pnt, mod_tgt_pnt)

            mod_error += dist_error(pcf.dns_func, DistFunc.from_pcpnt(pcf.srs, mod_pnt)) / len(pcfs)


        #计算接受移动概率
        delta_error = mod_error - self.cur_error
        #NOTE 单个种子变动 delta_error在0.01数量级 但误差较大, 当n_feature取50时,误差在0.005左右
        prob = 1.0 if delta_error <= 0.0 else np.exp(-delta_error/(self.T*0.01))
        
        #调试打印
        # print(mod_error, "-", self.cur_error)
        # print("d: ", delta_error)
        # print("p: ", prob)

        #是否接受修改
        if np.random.random() <= prob:
            cur_seeds.loc = mod_seeds.loc 
            self.cur_error = mod_error  #更新当前误差
        Link.update(self)



#-------------------------------------------------------------------------------------#
# 重建/拟合假设
#-------------------------------------------------------------------------------------#

class PCGAssumption(Node):
    """点相关采样重建假设"""
    def __init__(self, n_hrc, n_it, rnd_splitor=ExpSplitor(), rad_splitor=MulOtsuRadSplitor()) -> None:
        self.n_hrc = n_hrc
        self.n_it = n_it 
        self.rnd_splitor = rnd_splitor
        self.rad_splitor = rad_splitor
        self.update() 

    def dec_key(self) -> tuple: 
        return self.n_hrc, self.n_it, self.rnd_splitor, self.rad_splitor
    
    def __str__(self):
        return f"{self.cls_str}(hrc:{self.n_hrc} it:{self.n_it} sp:{self.rnd_splitor})"
    
    def get_n_hrc(self, seeds):
        return max(self.n_hrc, 1)
    
    def get_n_it(self, seeds):
        return max(self.n_it, 1)


#预声明重建假设
nonit_assp = PCGAssumption(1, 1, PropSplitor(1.0), RadSplitor())  #不进行迭代
fill_assp = PCGAssumption(8, 15, ExpSplitor(1.0, 0.04))    #必须从1.0开始, 否则无法保证不同相之间的关系
fit_assp = PCGAssumption(1, 20, PropSplitor(0.90))         #对称的收敛比例, 保证提供空间中随机点(改为fit内对内部边实现)
# fit_assp = PCGAssumption(1, 60, PropSplitor(0.04))       #启用源和目标交换时
sample_assp = PCGAssumption(4, 30, ExpSplitor(0.64, 0.04))  
check_assp = PCGAssumption(4, 30, ExpSplitor(1.0, 0.12))   #从1开始检查更多种子, 和sample保持相同层级数目
fast_assp = PCGAssumption(3, 6, ExpSplitor(1.0, 0.12))    #预览假设
# stdpcf_assp = PCGAssumption(1, 120, RndSplitor(0.12, 0.04))  #标准两点相关,在一个半径批量内完成重建,不受尺度约束

class AnnealAssumption(PCGAssumption):
    """模拟退火重建假设""" 
    def __init__(self, n_hrc, r_it, rnd_splitor=ExpSplitor(), rad_splitor=MulOtsuRadSplitor(), T=ExpSplitor()) -> None:
        self.T = T #相对值 在迭代过程中随rnd_prop进一步变化
        self.r_it = r_it
        PCGAssumption.__init__(self, n_hrc, r_it, rnd_splitor, rad_splitor)

    def dec_key(self) -> tuple:
        return self.n_hrc, self.r_it, self.rnd_splitor, self.rad_splitor, self.T
    
    def __str__(self):
        return f"{self.cls_str}(hrc:{self.n_hrc} rit:{self.r_it} sp:{self.rnd_splitor} T:{self.T})"

    def get_n_it(self, seeds):
        return max(int(self.r_it * seeds.cnt), 1)


anneal_assp = AnnealAssumption(1, 10.0, PropSplitor(1.0), T=RndSplitor(0.4, 0.1))  #不进行rnd_bat划分 逐个种子判断是否优化

#-------------------------------------------------------------------------------------#
# (核心)材料对象 解释相和相间关系 实例化种子集合
#-------------------------------------------------------------------------------------#

class PhsRltSCC(Link, ProcessBar):
    """强连通分量重建 内部种子部分顺序"""
    # nvxl_th = 8 #np.inf  #最小种子的最大分辨体素数目 Inf关闭自适应

    def __init__(self, phss: list[Phase]=[], rlts: list[ObjectRelation]=[]):
        """内部不含复合相的点相关图强连通分量"""
        self.phss = OrderSet(phss)  #作为目标的相
        self.rlts = OrderSet(rlts)  #所有相作为目标参与的点相关
        ProcessBar.__init__(self)
        self.update()
        
    def dec_key(self) -> tuple:
        return *tuple([rlt for rlt in self.rlts]),
    
    def run_key(self) -> tuple:
        return *tuple([rlt.run_key() for rlt in self.rlts]), 

    def __str__(self):
        return f"{self.cls_str}(phs:{' '.join([phs.__str__() for phs in self.phss])})"
    
    @auto_check
    def __getstate__(self):
        return {"phss": self.phss, "rlts": self.rlts}

    def __setstate__(self, state):
        self.__dict__.update(state)
        # print("dict update")
        self.phss = sub_phss(self.phss)
        
        #按照标记重建相映射
        phs_map = {phs.mark: phs for phs in self.phss}
        for rlt in self.rlts:
            if isinstance(rlt.srs, PhaseObject):
                rlt_phss = [phs_map[mark] for mark in rlt.srs.phs.sub_marks]
                if len(rlt_phss) > 1:
                    rlt.srs.phs = CompPhase(rlt_phss)
                elif len(rlt_phss) == 1:
                    rlt.srs.phs = rlt_phss[0]
            if isinstance(rlt.tgt, PhaseObject):
                rlt_phss = [phs_map[mark] for mark in rlt.tgt.phs.sub_marks]
                if len(rlt_phss) > 1:
                    rlt.tgt.phs = CompPhase(rlt_phss)
                elif len(rlt_phss) == 1:
                    rlt.tgt.phs = rlt_phss[0]
        # print("rlt built")
        self.update()
            

    def update(self):
        #检查输入关系列表 只保留和目标相有关的关系
        #检查输入相 转化为基本相
        self.phss = OrderSet(self.phss) 
        self.rlts = OrderSet(self.rlts)
        self.vld_phss = sub_phss(self.phss)
        self.vld_rlts = tgt_itsc(self.vld_phss, self.rlts)
        # print("phss calculated")
        # self.rlts = idp_decomp(self.rlts)
        Link.update(self)


    def parse_phs_rlts(self, seeds: RefSeeds, reg: Region, assp: PCGAssumption, tgt_sub=True) -> tuple:
        """将相和物体间关系转化为 种子和物体采样点之间的关系"""      
        pr_dict = {phs: tgt_itsc(OrderSet([phs]), self.sub_rlts) for phs in self.vld_phss}  #建立相-指入关系字典  
        # obj_rads = np.concatenate([[np.min(rlt.srs.r) for rlt in pr_dict[phs] if isinstance(rlt.srs, Object)] for phs in self.vld_phss])

        #自适应域
        # rads = np.concatenate([seeds.phase_ref(CompPhase(list(self.vld_phss))).r, obj_rads])
        scc_reg = reg.adapt(seeds.phase_ref(CompPhase(list(self.vld_phss))).r)
            
        #声明种子引用
        if not seeds.cnt:
            raise RuntimeError(f"{self.__str__()} parse_phs_rlts failed: emtpy seeds input {seeds.__str__()}")
        rad_bat = RadBatRef(seeds, assp.get_n_hrc(seeds), 0, assp.rad_splitor) #材料内所有种子层级划分结果
        rnd_bat = RndBatRef(rad_bat, assp.get_n_it(seeds), 0, assp.rnd_splitor)

        #建立点相关关系
        #缓存定义的种子采样点 只对已完成重建分量内的种子有效
        pcfs = {phs: OrderSet() for phs in self.vld_phss}
        
        for phs in self.vld_phss: 
            #随机批量 
            tgt_seeds = PhaseRef(rnd_bat, phs)

            for rlt in pr_dict[phs]:
                if isinstance(rlt.tgt, ConstPhaseObject): 
                    tgt_pnt = ConstSeedsPnt(tgt_seeds, rlt.tgt.ashp, scc_reg) 
                elif isinstance(rlt.tgt, PhaseObject): 
                    tgt_pnt = SeedsPnt(tgt_seeds, rlt.tgt.ashp, scc_reg) 
                
                if isinstance(rlt.srs, PhaseObject): 
                    if isinstance(rlt.srs.ashp, VoroShape):  #同一个scc,且以完成重建批量才进行voro表面计算
                        rlt.srs.ashp.seeds = PhaseRef(seeds, rlt.srs.ashp.phs)   #将所有该相种子作为外部voro边界 跨不同相 跨ssc
                    if rlt.srs.phs.subs.intersection(self.vld_phss):  #当前分量内的相种子 (会在迭代中更新) 则参与半径和随机批量分配
                        #当前批量
                        cur_srs_seeds = PhaseRef(rad_bat, rlt.srs.phs) 
                        if tgt_sub: cur_srs_seeds = SubRef(cur_srs_seeds, tgt_seeds) #源不包含目标
                        cur_srs_pnt = SeedsPnt(cur_srs_seeds, rlt.srs.ashp, scc_reg)
                        #累积固定批量
                        cum_srs_seeds = PhaseRef(rad_bat, rlt.srs.phs)
                        cum_srs_pnt = CumSeedsPnt(cum_srs_seeds, rad_bat, rlt.srs.ashp, scc_reg)
                        srs_pnt = AddSeedsPnt(cur_srs_pnt, cum_srs_pnt)
                    else:  #已重建分量内的种子,不会发生修改 进行全部层级渲染后固定为图像物体
                        srs_seeds = PhaseRef(seeds, rlt.srs.phs)
                        srs_pnt = HrcSeedsPnt(srs_seeds, rlt.srs.ashp, scc_reg, max(int(assp.n_hrc/2), 1)) 
                #解决TODO 允许输入复合物体 
                elif isinstance(rlt.srs, Object): 
                    #物体作为点相关起点 始终不变
                    srs_pnt = ObjectPnt(rlt.srs, scc_reg)
                else:
                    raise RuntimeWarning("empty source object")
                    srs_pnt = ObjectPnt(non_obj, reg) #空物体处理

                #添加点相关边
                pcfs[phs].add(SeedsPntCor(srs_pnt ,tgt_pnt, rlt.ftr_func, rlt.dns_func))

        self.rad_bat, self.rnd_bat, self.pcfs, self.reg = rad_bat, rnd_bat, pcfs, scc_reg
        return rad_bat, rnd_bat, pcfs, scc_reg

    @auto_memo
    def sample(self, seeds: Seeds, reg: Region, assp=sample_assp, rnd=None, on=True) -> Seeds:
        if rnd: np.random.seed(rnd)
        self.on = on 

        #迭代是否启用: 判断是否进行层级直接采样重建
        assp = nonit_assp if (not self.is_cycle) else assp 

        #解析相间关系 转化为点相关图
        rad_bat, rnd_bat, pcfs, scc_reg = self.parse_phs_rlts(seeds, reg, assp)  #种子间点相关

        #声明采样关系 完成重建
        phs_str = " ".join([phs.__str__() for phs in self.vld_phss])
        task_str = {True: "Iterative Sample:", False: "Direct Sample:"}[self.is_cycle]
        if self.on: self.print(f"{task_str} phs:{phs_str} res:1/{scc_reg.vxl_len} pd:{scc_reg.periods}")
        self.start(rad_bat.n_bat * rnd_bat.n_bat)

        #迭代采样
        for rad_bat.cur_bat in range(rad_bat.n_bat):
            # for rnd_bat.cur_bat in range(0, 1):
            #     self.inc()
            #     for phs in self.vld_phss: 
            #         PCEdgeSampleInit(pcfs[phs])
            for rnd_bat.cur_bat in range(rnd_bat.n_bat):
                self.inc()
                for phs in self.vld_phss: 
                    PCEdgeSample(pcfs[phs])

        self.end()
        #更新种子标志
        seeds.reg = reg
        return seeds 

    
    @auto_memo
    def fit(self, seeds: Seeds, reg: Region, assp=fit_assp, rnd=None, on=True) -> Seeds:
        if rnd: np.random.seed(rnd)
        self.on = on 
        #提取时迭代启用 不论是否为环 

        #解析相间关系 转化为点相关图  #必须保证输入parse的seeds和输入fit的seeds相同
        rad_bat, rnd_bat, pcfs, scc_reg = self.parse_phs_rlts(seeds, reg, assp)

        #声明点相关拟合 完成表征
        phs_str = " ".join([phs.__str__() for phs in self.vld_phss])
        if self.on: self.print(f"Iterative Fit: phs:{phs_str} res:1/{scc_reg.vxl_len} pd:{scc_reg.periods}")
        self.start(rad_bat.n_bat * rnd_bat.n_bat)

        for rad_bat.cur_bat in range(rad_bat.n_bat):
            for rnd_bat.cur_bat in range(rnd_bat.n_bat):
                self.inc()
                for phs in self.vld_phss: 
                    PCEdgeFit(pcfs[phs])

        self.end()
        seeds.reg = reg
        return seeds
    
    @auto_memo
    def ann_fit(self, seeds: Seeds, reg: Region, assp=nonit_assp, rnd=None, on=True) -> Seeds:
        if rnd: np.random.seed(rnd)
        self.on = on 
        #提取时迭代启用 不论是否为环 

        #解析相间关系 转化为点相关图  #必须保证输入parse的seeds和输入fit的seeds相同
        rad_bat, rnd_bat, pcfs, scc_reg = self.parse_phs_rlts(seeds, reg, assp, tgt_sub=False)

        #声明点相关拟合 完成表征
        phs_str = " ".join([phs.__str__() for phs in self.vld_phss])
        if self.on: self.print(f"Anneal Fit: phs:{phs_str} res:1/{scc_reg.vxl_len} pd:{scc_reg.periods}")
        self.start(rad_bat.n_bat * rnd_bat.n_bat)

        for rad_bat.cur_bat in range(rad_bat.n_bat):
            for rnd_bat.cur_bat in range(rnd_bat.n_bat):
                self.inc()
                for phs in self.vld_phss: 
                    PCFuncFit(pcfs[phs])

        self.end()
        seeds.reg = reg
        return seeds

    @auto_memo
    def fill(self, seeds: Seeds, reg: Region, assp=fill_assp, rnd=None, on=True) -> Seeds:
        if rnd: np.random.seed(rnd)
        self.on = on 
        #提取时迭代启用 不论是否为环 

        #解析相间关系 转化为点相关图  #必须保证输入parse的seeds和输入fit的seeds相同
        rad_bat, rnd_bat, pcfs, scc_reg = self.parse_phs_rlts(seeds, reg, assp)
    
        #声明点相关拟合 完成表征
        phs_str = " ".join([phs.__str__() for phs in self.vld_phss])
        if self.on: self.print(f"Iterative Fill: phs:{phs_str} res:1/{scc_reg.vxl_len} pd:{scc_reg.periods}")
        self.start(rad_bat.n_bat * rnd_bat.n_bat)

        for rad_bat.cur_bat in range(rad_bat.n_bat):
            # print(f"radius bat: {rad_bat.cur_bat}")
            # print(f"edge sample")
            # for rnd_bat.cur_bat in range(0, 1):
            #     self.inc()
            #     for phs in self.vld_phss: 
            #         PCEdgeSampleInit(pcfs[phs])
            for rnd_bat.cur_bat in range(rnd_bat.n_bat):
                # print(f"rand bat: {rnd_bat.cur_bat}")
                self.inc()
                for phs in self.vld_phss: 
                    # print(f"phs: {phs}")
                    PCEdgeSample(pcfs[phs])
            # print(f"edge check")
            for rnd_bat.cur_bat in range(rnd_bat.n_bat):  #每个半径批量后进行对称的重叠性检查
                # print(f"rand bat: {rnd_bat.cur_bat}")
                for phs in self.vld_phss: 
                    # print(f"phs: {phs}")
                    PCEdgeCheck(pcfs[phs])

        self.end()
        seeds = seeds.remove_hide() #只返回有效种子的引用
        seeds.reg = reg
        return seeds

    @auto_memo
    def check(self, seeds: Seeds, reg: Region, assp=check_assp, rnd=None, on=True) -> Seeds:
        if rnd: np.random.seed(rnd)
        self.on = on 
        #提取时迭代启用 不论是否为环 

        #解析相间关系 转化为点相关图  #必须保证输入parse的seeds和输入fit的seeds相同
        rad_bat, rnd_bat, pcfs, scc_reg = self.parse_phs_rlts(seeds, reg, assp)

        #声明点相关拟合 完成表征
        phs_str = " ".join([phs.__str__() for phs in self.vld_phss])
        if self.on: self.print(f"Iterative Check: phs:{phs_str} res:1/{scc_reg.vxl_len} pd:{scc_reg.periods}")
        self.start(rad_bat.n_bat * rnd_bat.n_bat)

        for rad_bat.cur_bat in range(rad_bat.n_bat):
            for rnd_bat.cur_bat in range(rnd_bat.n_bat):  #每个半径批量后进行对称的重叠性检查
                self.inc()
                for phs in self.vld_phss: 
                    PCEdgeCheck(pcfs[phs])
        
        self.end()
        seeds = seeds.remove_hide() #只返回有效种子的引用
        seeds.reg = reg
        return seeds


    @auto_memo
    def anneal(self, seeds: Seeds, reg: Region, assp=anneal_assp, rnd=None, on=True) -> Seeds:
        if rnd: np.random.seed(rnd)
        self.on = on 

        #解析相间关系 转化为点相关图
        rad_bat, rnd_bat, pcfs, scc_reg = self.parse_phs_rlts(seeds, reg, assp, tgt_sub=False)
    
        #声明点相关拟合 完成表征
        phs_str = " ".join([phs.__str__() for phs in self.vld_phss])
        if self.on: self.print(f"Simulated Annealing: phs:{phs_str} res:1/{scc_reg.vxl_len} pd:{scc_reg.periods}")
        self.start(rad_bat.n_bat * rnd_bat.n_bat)
        
        cur_error = None 
        for rad_bat.cur_bat in range(rad_bat.n_bat):
            for rnd_bat.cur_bat in range(rnd_bat.n_bat):
                self.inc()
                for phs in self.vld_phss: 
                    move_T = anneal_assp.T.get_prop(rnd_bat, rnd_bat.n_bat, rnd_bat.cur_bat)
                    cur_error = PCEdgeMove(pcfs[phs], move_T, cur_error).cur_error 

        self.end()
        seeds.reg = reg
        return seeds

    # def adapt_reg(self, seeds, reg: Region) -> Region:
    #     """自适应域分辨率 同一个SCC内共用reg 需要所有源可随reg变化"""
    #     if not seeds.cnt:
    #         warnings.warn(f"adapt reg failed: emtpy seeds: {seeds.__str__()}")
    #         return reg 
    #     min_vxl = np.min(seeds.r) * reg.vxl_len
    #     if min_vxl > self.nvxl_th:
    #         return reg.rescale(min_vxl / self.nvxl_th)
    #     else:
    #         return reg 

    @property
    def sub_rlts(self) -> set:
        """按照目标相分解关系, 获得重建相间关系"""
        return sub_comp_rlts(self.vld_rlts)

    @property
    def is_cycle(self) -> bool:
        if len(self.vld_phss) == 1 and len(srs_itsc(self.vld_phss, self.vld_rlts)):  #数目为1 且自相交为环
            return True 
        elif len(self.vld_phss) > 1:    #连通分量内相数目大于1 则存在环 
            return True 
        return False 

    @property
    def pcg(self) -> dict[Phase, dict[PhaseRelation, Phase]]:
        """相之间构成的点相关图 参数化物体不参与 所有节点为待重建节点 复合相转化为基本相"""
        return get_pcg(self.vld_rlts)

    @property
    def sccs_dict(self) -> dict[tuple[Phase], set[ObjectRelation]]:
        return get_sccs(self.pcg, self.sub_rlts)
    
    @property
    def sccs_nodes(self) -> list[PhsRltSCC]:
        return get_sccs_nodes(self.pcg)[::-1]
    
    @auto_check
    def copy(self):
        scc = self.__class__()
        scc.__setstate__({
            "phss": [phs.copy() for phs in self.phss], 
            "rlts": [rlt.copy() for rlt in self.rlts]}
            )
        return scc
    
    @auto_check
    def get_rlt(self, *mark):
        # sub_rlts = [rlt for rlt in self.pcg.values()]
        for phs in self.pcg.keys():
            for rlt in self.pcg[phs].keys():
                if "phs" in rlt.srs.__dict__: 
                    srs_mark = rlt.srs.phs.mark
                else:  #物体对象
                    srs_mark = rlt.srs.mark
                if mark == (srs_mark, rlt.tgt.phs.mark):
                    return rlt
        raise ValueError(f"get rlt failed: mark {mark} does not exist")
    
    @auto_check
    def get_phs(self, mark):
        # sub_rlts = [rlt for rlt in self.pcg.values()]
        for phs in self.pcg.keys():
            if phs.mark == mark: 
                return phs 
        raise ValueError(f"get phs failed: mark {mark} does not exist")


class PhsRltGraph(PhsRltSCC):
    """相间关系图 解析连通分量 转化相关系定义"""
    def __init__(self, phss: list[Phase]=[], rlts: list[ObjectRelation]=[]):
        self.phss = OrderSet(phss) 
        self.rlts = OrderSet(rlts)
        self.update()

    def dec_key(self) -> tuple:
        return *tuple([rlt for rlt in self.rlts]),
    
    def run_key(self) -> tuple:
        return *tuple([rlt.run_key() for rlt in self.rlts]), #以rlt作为key 而非内部保存的sccs

    def update(self):
        self.phss = OrderSet(self.phss) 
        self.rlts = OrderSet(self.rlts)
        self.vld_phss = sub_phss(self.phss)
        self.vld_rlts = tgt_itsc(self.vld_phss, self.rlts)  #目标相应输入
        # self.rlts = idp_decomp(self.rlts)
        self.sccs = [PhsRltSCC(list(scc_phss), list(self.sccs_dict[scc_phss])) for scc_phss in self.sccs_nodes]    #scc在更新时产生
        Link.update(self)

    @auto_memo
    def sample(self, reg: Region, assp=sample_assp, seeds=None, rnd=None, on=True): 
        """采样重建 所有种子将被保留""" 
        if on: print(f"PCG Sample | {assp}")

        if rnd: np.random.seed(rnd) 

        #顶点采样
        if not isinstance(seeds, Seeds):
            seeds = PCNodeSample(CompPhase(list(self.vld_phss)), reg).seeds  #先建立整个材料的共用种子

        #声明连通分量 输入局部相间关系 完成重建
        for scc in self.sccs: scc.sample(seeds, reg, assp, on=on)   #未修改的scc可直接跳过
        # if on: print("\n")
        seeds.reg = reg
        return seeds

    
    @auto_memo
    def fit(self, seeds: Seeds, reg: Region, assp=fit_assp, rnd=None, on=True): 
        """声明连通分量 输入先验相间关系 拟合"""
        if on: print(f"PCG Fit | {assp}")

        if rnd: np.random.seed(rnd) 

        #边拟合 只保留有效边密度函数拟合种子
        for scc in self.sccs: scc.fit(seeds, reg, assp, on=on)  

        #顶点拟合 (后进行 避免相更新造成后续变化)
        PCNodeFit(CompPhase(list(self.vld_phss)), seeds, reg)  #只保留有效的种子参与拟合
        # if on: print("\n")
        seeds.reg = reg
        return seeds

    
    @auto_memo
    def ann_fit(self, seeds: Seeds, reg: Region, assp=nonit_assp, rnd=None, on=True): 
        if on: print(f"Annealing Fit | {assp}")

        if rnd: np.random.seed(rnd) 

        #边拟合 只保留有效边密度函数拟合种子
        for scc in self.sccs: scc.ann_fit(seeds, reg, assp, on=on)  

        #顶点拟合 (后进行 避免相更新造成后续变化)
        # print("scc fit end")
        PCNodeFit(CompPhase(list(self.vld_phss)), seeds, reg)  #只保留有效的种子参与拟合
        # print("pcnode fit end")
        # if on: print("\n")
        seeds.reg = reg
        return seeds


    @auto_memo
    def fill(self, reg: Region, assp=fill_assp, seeds=None, rnd=None, on=True): 
        """声明连通分量 输入先验相间关系 拟合"""
        if on: print(f"PCG Fill | {assp}")

        if rnd: np.random.seed(rnd) 

        #顶点采样
        if not isinstance(seeds, Seeds):
            seeds = PCNodeSample(CompPhase(list(self.vld_phss)), reg).seeds  #先建立整个材料的共用种子

        #边拟合 只保留有效边密度函数拟合种子
        for scc in self.sccs: scc.fill(seeds, reg, assp, on=on)  
        # if on: print("\n")
        seeds = seeds.remove_hide()
        seeds.reg = reg
        return seeds
        
    @auto_memo
    def check(self, seeds:Seeds, reg: Region, assp=check_assp, rnd=None, on=True): 
        """声明连通分量 输入先验相间关系 拟合"""
        if on: print(f"PCG Check | {assp}")

        if rnd: np.random.seed(rnd) 

        #边拟合 只保留有效边密度函数拟合种子
        for scc in self.sccs: scc.check(seeds, reg, assp, on=on)  
        # if on: print("\n")
        seeds = seeds.remove_hide()
        seeds.reg = reg
        return seeds

    @auto_memo
    def anneal(self, reg: Region, assp: AnnealAssumption = anneal_assp, seeds=None, rnd=None, on=True): 
        """声明连通分量 模拟退火重建"""
        if on: print(f"Annealing Rec| {assp}")

        if rnd: np.random.seed(rnd) 

        if not isinstance(seeds, Seeds):
            seeds = PCNodeSample(CompPhase(list(self.vld_phss)), reg).seeds  #顶点采样

        for scc in self.sccs: scc.anneal(seeds, reg, assp, on=on)
        # if on: print("\n")
        seeds.reg = reg
        return seeds

    

    




#-------------------------------------------------------------------------------------#
# 辅助函数
#-------------------------------------------------------------------------------------#


def tgt_itsc(phss: set[Phase], rlts: set[ObjectRelation]) -> set:
    return OrderSet([rlt for rlt in rlts if rlt.tgt_itsc(phss)]) 

def idp_decomp(rlts: set[ObjectRelation]) -> set:
    #区分自依赖和外依赖关系
    idp_rlts = OrderSet()
    for rlt in rlts:  #目标相关的关系
        if isinstance(rlt.srs, PhaseObject) and isinstance(rlt.tgt, PhaseObject):
            for tgt_phs in rlt.tgt.phs.subs:
                inner_phs = rlt.srs.phs.subs.intersection([tgt_phs])
                outer_phs = rlt.srs.phs.subs.difference([tgt_phs])
                inner_phs = inner_phs[0] if len(inner_phs) == 1 else CompPhase(inner_phs)
                outer_phs = outer_phs[0] if len(outer_phs) == 1 else CompPhase(outer_phs)

                tgt_pobj = PhaseObject(tgt_phs, rlt.tgt.ashp)  #需要分解
                if inner_phs.subs: idp_rlts.append(rlt.__class__(PhaseObject(inner_phs, rlt.srs.ashp), tgt_pobj, rlt.ftr_func, rlt.dns_func))
                if outer_phs.subs: idp_rlts.append(rlt.__class__(PhaseObject(outer_phs, rlt.srs.ashp), tgt_pobj, rlt.ftr_func, rlt.dns_func))
        else:
            idp_rlts.append(rlt)
    return idp_rlts

def srs_itsc(phss: set[Phase], rlts: set[ObjectRelation]) -> set:
    return OrderSet([rlt for rlt in rlts if rlt.srs_itsc(phss)]) 


def sub_phss(phss: set[Phase]) -> set[Phase]:
    """分解复合相"""
    subs = OrderSet()
    for phs in phss: 
        subs.update(phs.subs)
    return subs 

def sub_comp_rlts(rlts: set[ObjectRelation], sub_srs = False) -> set[ObjectRelation]:
    """
    分解复合物体参与的关系, 分解复合相作为目标参与的关系  当计算依赖关系时启用sub_srs, 当计算重建关系时关闭. 
    Examples:
        bnd = CompObject([sep, clt, PhaseObject(LFP, sur)])
        rlts = [ObjectRelation(bnd, PhaseObject(LFP, sur), nnsurdt, cll), ]
        sub_comp_rlts(rlts, True)
    """
    rlts = OrderSet(rlts)
    sub_rlts = OrderSet()

    #解析源中的VoroShape, 增加虚拟依赖关系
    add_rlts = OrderSet()
    if sub_srs:  #只在解析依赖时启用
        for rlt in rlts: 
            if isinstance(rlt.srs, PhaseObject):
                if isinstance(rlt.srs.ashp, VoroShape):
                    add_rlts.add(rlt.__class__(rlt.srs.__class__(rlt.srs.ashp.phs, SeedsShape(0.0)), rlt.tgt, clldt, idp))

            elif isinstance(rlt.srs, CompObject) and np.any([isinstance(obj, PhaseObject) for obj in rlt.srs.subs]):
                #相物体作为源
                srs_phss = [obj for obj in rlt.srs.subs if isinstance(obj, Object) and isinstance(obj, PhaseObject)]

                #源子物体存在相
                if len(srs_phss):
                    for srs_phs in srs_phss:  #分不同源的相添加
                        if isinstance(srs_phs.ashp, VoroShape):
                            add_rlts.add(rlt.__class__(srs_phs.__class__(srs_phs.ashp.phs, SeedsShape(0.0)), rlt.tgt, clldt, idp)) 
        #增加虚拟依赖
        rlts = rlts.union(add_rlts)

    for rlt in rlts: #对相间关系进一步划分
        for tgt_phs in rlt.tgt.phs.subs:  #复合相有多个subs, 必须划分
            if sub_srs and isinstance(rlt.srs, PhaseObject): #对源进行分解 如果源是复合相 
                for srs_phs in rlt.srs.phs.subs:
                    sub_rlts.add(rlt.__class__(rlt.srs.__class__(srs_phs, rlt.srs.ashp), rlt.tgt.__class__(tgt_phs, rlt.tgt.ashp), rlt.ftr_func, rlt.dns_func))  #拆分的多个关系共用dns_func 在调用fit时 共用数据  
            
            #排除复合物体中的相 
            elif sub_srs and isinstance(rlt.srs, CompObject) and np.any([isinstance(obj, PhaseObject) for obj in rlt.srs.subs]):

                #相物体作为源
                srs_phss = [obj for obj in rlt.srs.subs if isinstance(obj, Object) and isinstance(obj, PhaseObject)]
                #物体作为源
                srs_objs = [obj for obj in rlt.srs.subs if isinstance(obj, Object) and not isinstance(obj, PhaseObject)]

                #源只有物体
                if len(srs_objs):
                    sub_rlts.add(rlt.__class__(CompObject(srs_objs), rlt.tgt.__class__(tgt_phs, rlt.tgt.ashp), rlt.ftr_func, rlt.dns_func))
                #源子物体存在相
                if len(srs_phss):
                    for srs_phs in srs_phss:  #分不同源的相添加
                        sub_rlts.add(rlt.__class__(srs_phs, rlt.tgt.__class__(tgt_phs, rlt.tgt.ashp), rlt.ftr_func, rlt.dns_func)) 

            else:  #将源视作纯复合物体
                sub_rlts.add(rlt.__class__(rlt.srs, rlt.tgt.__class__(tgt_phs, rlt.tgt.ashp), rlt.ftr_func, rlt.dns_func)) 
     #关系不覆盖 同时成立
    return sub_rlts


def get_pcg(rlts: set[ObjectRelation]) -> dict[Phase, dict[ObjectRelation, Phase]]:
    """获取点相关图 在基本相间关系定义基础上"""
    pc_rlts: set[PhaseRelation] = sub_comp_rlts(rlts, True)  #获取基本相间的关系
    tgt_nodes = OrderSet([rlt.tgt.phs for rlt in pc_rlts])
    srs_nodes = OrderSet([rlt.srs.phs if isinstance(rlt.srs, PhaseObject) else rlt.srs for rlt in pc_rlts]) 

    pcg = {} 
    for srs in tgt_nodes.union(srs_nodes):
        pcg.update({srs: {}})
        if isinstance(srs, Phase):
            for rlt in tgt_itsc(OrderSet([srs]), pc_rlts):
                if isinstance(rlt.srs, PhaseObject):
                    pcg[srs].update({rlt: rlt.srs.phs})
                else:  #物体作为源
                    pcg[srs].update({rlt: rlt.srs})
    return pcg 



def get_sccs(pcg, sub_rlts) -> dict[tuple[Phase], set[ObjectRelation]]:
    """强连通分量字典 返回字典: 分量相元组->相互作用 此时包含物体关系"""  
    return {tuple(scc) : tgt_itsc(OrderSet(scc), sub_rlts) for scc in get_sccs_nodes(pcg)}

def get_sccs_nodes(pcg) -> list[tuple[Phase]]:
    from . import GraphParse

    sccs = GraphParse.get_sccs({phs: OrderSet(pcg[phs].values()) for phs in pcg.keys()})
    phs_sccs = [tuple([phs for phs in scc if isinstance(phs, Phase)]) for scc in sccs]
    phs_sccs = [scc for scc in phs_sccs if len(scc)]
    return phs_sccs 



#-------------------------------------------------------------------------------------#
# Voronoi物体
#-------------------------------------------------------------------------------------#

class Voronoi(Object):
    rpos = 0.0 
    rmin = 0.6
    drsub = 0.4 #离散时缩减
    n_hrc = 6

    def __init__(self, seeds = Seeds(), id = -1): #, pobjs=[]): 
        self.seeds = seeds
        if "reg" in seeds.__dict__: self.reg = seeds.reg
        self.id = id
        # self.pobjs = pobjs
        self.update()

    def dec_key(self) -> tuple:
        return self.seeds, self.id, #, *tuple([pobj.dec_key() for pobj in self.pobjs]), 

    def __str__(self):
        if self.id >= 0:
            return f"{self.cls_str}({self.seeds}={self.id})"
        else: 
            return f"{self.cls_str}({self.seeds})"
    
    @auto_memo 
    def discrete(self, reg: Region) -> PhaseVol:
        """离散到域"""
        if self.seeds.cnt <= 0: 
            warnings.warn("voronoi discrete failed, empty seeds")
            return PhaseVol.zeros(reg.shape)
        
        seeds = self.seeds.copy()  #影响数据域的操作 在复制上进行
        max_mark = np.max(seeds.mark)
        seeds.mark_ref(0).mark = max_mark + 1  #将0赋值为新的mark进行离散
        # base_nvxl = max((np.min(seeds.r) * reg.vxl_len) / 2, 1)
        # mark_vxl = [np.median(seeds.mark_ref(mark).r) * reg.vxl_len for mark in np.unique(seeds.mark) if mark != 0]
        # ashp_vxl = [2/(1/nvxl + 1/base_nvxl) + self.rmin for nvxl in mark_vxl]  #不同尺度相在最后采用不同修正

        # seeds = seeds.itv_ref([self.rmin/reg.vxl_len, 1.0])
        seeds.r = np.clip(seeds.r - self.drsub/reg.vxl_len, self.rmin/reg.vxl_len, 1.0) #半径缩减后离散 
        seeds = seeds.sort()

        # seeds_vol = PhaseVol.zeros(reg.shape)
        # if len(self.pobjs): 
        #     for pobj in self.pobjs: seeds_vol += seeds[pobj.phs].discrete_dmn_hrc(reg, pobj.ashp, self.n_hrc) #大半径先离散
        # else: 
        seeds_vol = seeds.discrete(reg)
        voro_vol = seeds_vol.watershed()   #分水岭 
        voro_vol[voro_vol[:] == max_mark + 1] = 0  #恢复0属性
        if self.id < 0:
            return voro_vol
        else: 
            return PhaseVol(voro_vol[:] == self.id)
        
    def attr_map(self, map):
        return self.__class__(self.seeds.attr_map(map))

    @property 
    def r(self):
        return self.seeds.r * self.rpos 

    @r.setter
    def r(self, value):
        self.seeds.r = value 

    @property
    def mark(self):
        return self.seeds.mark 

    @mark.setter 
    def mark(self, value):
        self.seeds.mark = value 


class LaguerreVoro(Voronoi):
    rpos = 1.0 
    rmin = 0.6   #从体素初始化的最小分辨半径
    near_rpos = 0.6  #过大会导致区域连在一起

    @classmethod
    def from_vol(cls, vol, ndim=None, min_vxl=3) -> LaguerreVoro:
        """求Voronoi覆盖 转化为种子表示 对属性不为0的部分生效
        Examples:
            #清理图像 按大小分割
            voro = LaguerreVoro.from_vol(prt_vol, 2)
            voro.seeds.mark_ref(2).itv_ref([0.0, 0.05]).mark = 1
            LaguerreVoro(voro.seeds)
        """ 
        from skimage.measure import regionprops
        
        seg_vol = vol.segment(min_vxl)  #多相 序号从1开始 跳过0属性,0表示未分割区域
        seg_cen = vol.seg_center(min_vxl)

        regs = regionprops(seg_vol[:])  #只对1以上序号域统计 0表示空位 跳过
        nreg = len(regs) #reg质心不为voro的中心 采用local_peak提取的中心

        seeds = Seeds(np.zeros((nreg, Seeds.n_data)), np.arange(nreg))
        seeds.loc = seg_cen / vol.vxl_len 

        ndim = Region(1, vol.shape).ndim if not ndim else ndim
        if ndim == 3:
            seeds.r = (np.array([reg.area for reg in regs]) / (4/3*np.pi))**(1/3) / vol.vxl_len
        elif ndim == 2:
            seeds.r = (np.array([reg.area for reg in regs]) / (np.pi))**(1/2) / vol.vxl_len
        else: 
            raise ValueError("voro from vol failed: unsupport dimension")
        seeds.mark = vol.query_attr(seeds.loc)
        # seeds = seeds.mask_ref(seeds.r > cls.rmin/vol.vxl_len)
        voro = cls(seeds)
        voro.seg_vol = seg_vol
        return voro

    def resample(self, reg) -> LaguerreVoro:
        """种子Voro离散后重新计算Voronoi覆盖"""
        return self.from_vol(self.discrete(reg))

    def voro_reg(self, reg: Region) -> PhaseVol:
        from skimage.measure import regionprops

        lab_seeds = self.seeds.copy() 
        lab_seeds.r *= self.near_rpos
        lab_seeds.a *= self.near_rpos
        lab_seeds.b *= self.near_rpos
        lab_vol = lab_seeds.discrete(reg).label_vol()

        regs = regionprops(lab_vol[:])
        locs = np.array([reg.centroid for reg in regs]) / reg.vxl_len
        
        voro_vol = self.discrete(reg)
        marks = voro_vol.query(locs)
        for i in range(len(regs)): regs[i].label = marks[i]

        seeds = self.seeds.copy()
        seeds.mark = lab_vol.query(lab_seeds.loc)
        seeds.r = np.clip(seeds.r - self.drsub/reg.vxl_len, self.rmin/reg.vxl_len, 1.0) #半径缩减后离散 
        seeds = seeds.sort()
        reg_vol = seeds.discrete(reg).watershed()
        return regs, reg_vol
    

def filter_region(vol, itv_th, ndim=None, nvxl=3):
    """按照局部域大小过滤体素网格"""
    seeds = LaguerreVoro.from_vol(vol, ndim=ndim, min_vxl=nvxl).seeds
    seeds.itv_ref([0, itv_th[0]]).mark = 0
    seeds.itv_ref([itv_th[1], 1]).mark = 0
    return LaguerreVoro(seeds).discrete(Region(1, vol.shape))


class PolyVoro(LaguerreVoro):
    near_rpos = 0.6
        
    @auto_memo 
    def discrete(self, reg: Region, rnd_mark=True) -> PhaseVol:
        """离散到域"""
        seeds = self.seeds.copy() 

        lab_seeds = self.seeds.copy() 
        lab_seeds.r *= self.near_rpos
        lab_seeds.a *= self.near_rpos
        lab_seeds.b *= self.near_rpos
        lab_vol = lab_seeds.discrete(reg).label_vol()
        seeds.mark = lab_vol.query(lab_seeds.loc)

        if rnd_mark:
            marks = np.arange(seeds.mark.min(), seeds.mark.max()+1); np.random.shuffle(marks)
            mark_map = np.insert(marks, 0, 0)
            seeds.mark = mark_map[seeds.mark]

        seeds.r = np.clip(seeds.r - self.drsub/reg.vxl_len, self.rmin/reg.vxl_len, 1.0) #半径缩减后离散 
        seeds = seeds.sort()
        seeds_vol = seeds.discrete(reg)
        return seeds_vol.watershed()   #分水岭 



class Generator(CompObject):
    """物体生成器
    物体组合 Shape应用 表面平滑 孔隙分配"""
    def __init__(self, objs):
        CompObject.__init__(self, objs)