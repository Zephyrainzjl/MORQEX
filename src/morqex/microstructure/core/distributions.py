import numpy as np 
from scipy import stats as stat
import warnings 

class DiscreteDnsFunc():
    """离散分布"""
    nbin = 600
    xmin = 0.0
    xmax = 1.8
    
    def __getstate__(self, state):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

class smooth(DiscreteDnsFunc):
    """数据平滑分布"""
    nsmth = 10
    ndec = 6

    def __init__(self, p=[]):
        """输入采样点概率密度初始化, 默认为均匀分布
        Examples: 
            sm = smooth(*smooth.fit(nmc.sample(reg).r ))
            plt.plot(sm.pdf(np.linspace(0.01,0.5)))
        """
        from scipy.stats import rv_discrete 

        if (not np.any(p)) or (not np.any(np.clip(p, 0, None))):
            #自动初始化数据, 均匀
            # warnings.warn(f"init {self.__class__.__name__} warn: invalid input p, auto fixed")
            p = np.full(self.nbin, 1/self.nbin)

        if  np.shape(self.x()) != np.shape(p):
            warnings.warn(f"init {self.__class__.__name__} warn: invalid size of input p, auto fixed")
            p = p[:self.x().size]

        if np.any(np.isnan(p)): 
            warnings.warn(f"init {self.__class__.__name__} warn: nan exist in p, auto fixed")
            p[np.isnan(p)] = 0 
        
        if np.min(p) < 0:
            warnings.warn(f"init {self.__class__.__name__} warn: p for {self.__class__} less than zero, auto fixed")
            p = np.clip(p, 0, None)

        if not np.allclose(np.sum(p), 1):
            # warnings.warn(f"init {self.__class__.__name__} warn: the sum of p for {self.__class__} is not 1, auto fixed")
            p = p / np.sum(p)

        self.rv_dist = rv_discrete(values=(self.x(), p))  #拓展scipy.stats.rv_discrete实例
        self.p = p 
            
    @classmethod
    def fit(cls, x_tab):
        """
        类方法,从距离统计拟合分布, 返回值为自身初始化参数
        Examples:
            args = rd.dist.fit(np.array([0.1,0.2,0.3,-1,0.3,0.1,0.112,0.32,0.5,0.51]))
            rd1 = rd.dist(*args)
        """
        if not x_tab.size:
            return ([],)
            
        #统计频率  (保证输出在范围内)
        hist, _ = np.histogram(x_tab.reshape(-1), bins=cls.nbin+1, range=(cls.xmin, cls.xmax))    #nbin+1,保证重建-拟合对齐
        smooth_hist = np.convolve(hist[:-1], np.full((cls.nsmth,) , 1/cls.nsmth) , "same")
        dns = smooth_hist / np.sum(smooth_hist)
        return (dns, )
        # else: 
        #     inv_cdf = 1 - np.cumsum(dns) + 1.0/cls.nbin
        #     norm_dns = dns/n_srs * (inv_cdf)**((1-n_srs)/n_srs) 
        #     return (norm_dns / np.sum(norm_dns), )

    @classmethod
    def from_itv(cls, x1=-np.inf, x2=np.inf):
        """从指定区间内的均匀分布初始化"""
        p = np.full(cls.nbin, 1/cls.nbin) 
        p[~np.bitwise_and(x1 < cls.x(), cls.x() < x2)] = 0
        if np.any(p):
            return cls(p / np.sum(p)) 
        else: 
            raise ValueError(f"init {cls.__name__} from interval failed: no sample point in {x1, x2}")

    @classmethod
    def from_dist(cls, dist):
        """从scipy连续分布初始化"""
        p = dist.pdf(cls.x())
        return cls(p / np.sum(p)) 

    @classmethod
    def from_data(cls, data):
        """从采样数据初始化"""
        p = cls().fit(data)[0] 
        return cls(p / np.sum(p)) 

    @classmethod
    def x(cls):
        """预定义采样点"""
        return np.round(np.linspace(cls.xmin, cls.xmax, cls.nbin+1)[:-1], cls.ndec)  #放弃一定精度 保持值固定

    @classmethod
    def pad_x(cls):
        return np.round(np.linspace(cls.xmin, cls.xmax, cls.nbin+1) - (cls.xmax - cls.xmin)/cls.nbin, cls.ndec)

    @property
    def args(self) -> tuple:
        """初始化参数"""
        return self.p, 
    
    @property
    def dist(self):
        """分布类型"""
        return self.__class__

    def rvs(self, n: int=1):
        """采样,输入样本数目"""
        return self.rv_dist.rvs(size = n)

    def pdf(self, kx):
        """概率密度函数 通过数组查找实现连续采样 无上边界"""
        idx = np.searchsorted(self.x(), kx)
        return self.rv_dist.pmf(self.pad_x()[idx]).reshape(-1)

    def cdf(self, kx):
        """概率分布函数"""
        idx = np.searchsorted(self.x(), kx)
        return self.rv_dist.cdf(self.pad_x()[idx]).reshape(-1)


class same(smooth):
    """和数据保持完全相同的分布"""
    nbin = 800
    nsmth = 1
