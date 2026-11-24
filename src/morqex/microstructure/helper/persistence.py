from morqex.microstructure.core.particle_graph import * 
import os 
import pickle 
import json


__functypes = (type(sum), type(np), type(np.sum), type(DataVol.sum), type(DataVol.ones), type(edt))
__numertypes = (np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, np.complex64, float, int, complex)


#-------------------------------------------------------------------------------------#
# 对象保存设置
#-------------------------------------------------------------------------------------#

class Config():
    def load(self, load_cfg):
        self.__dict__.update(load_cfg.__dict__)
        
class SaveConfig(Config):
    def __init__(self) -> None:
        self.dir = "data"
        self.fmt = "pkl" #json #jcsv
        self.fmts = ["pkl", "json", "jcsv"]
        self.vol_fmt = "npy"
        self.vol_fmts = ["npy", "txt", "tiff", "png"]

scfg = SaveConfig()


def jsonlize(obj):
    """转换为json序列对象"""
    if isinstance(obj, type(None)):
        return obj 
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return float(obj)
    elif isinstance(obj, (complex, np.complex64)):
        return complex(obj)
    elif isinstance(obj, (list, tuple)) and not len(obj):
        return obj 
    elif isinstance(obj, (list, tuple)) and type(obj[0]) == np.ndarray: 
        return [jsonlize(cont) for cont in obj if not isinstance(cont, __functypes)]
    elif isinstance(obj, (list, tuple)) and np.array(obj).dtype in __numertypes:
        return obj
    elif isinstance(obj, (list, tuple)):
        return [jsonlize(cont) for cont in obj if not isinstance(cont, __functypes)]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        json_dict = {} 
        for key in obj.keys():
            if not isinstance(obj[key], __functypes) and not (type(key) == str and key[:1]=="_"):
                json_dict.update({key: jsonlize(obj[key])})
        return json_dict
    elif isinstance(obj, object):# and not isinstance(obj, __functypes):
        try: obj_dict = jsonlize(obj.__getstate__()) 
        except AttributeError: obj_dict = jsonlize(obj.__dict__)
        
        cls_name = obj.__class__.__name__
        obj_dict.update({"__cls_name__": cls_name}) 
        return obj_dict 

def jcsvlize(obj, name="-"):
    """转换为json-csv对象"""
    if isinstance(obj, type(None)):
        return obj 
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return float(obj)
    elif isinstance(obj, (complex, np.complex64)):
        return complex(obj)
    elif isinstance(obj, (list, tuple)) and not len(obj):
        return obj 
    elif isinstance(obj, (list, tuple)) and type(obj[0]) == np.ndarray: 
        return [jcsvlize(cont, f"{name}{i}") for i, cont in enumerate(obj) if not isinstance(cont, __functypes)]
    elif isinstance(obj, (list, tuple)) and np.array(obj).dtype in __numertypes:
        return obj
    elif isinstance(obj, (list, tuple)):
        return [jcsvlize(cont, f"{name}{i}") for i, cont in enumerate(obj) if not isinstance(cont, __functypes)]
    elif isinstance(obj, np.ndarray):
        if obj.size: 
            np.savetxt(f"{scfg.dir}/{name}.csv", obj)
            return f"{name}.csv"
        else: 
            return []
    elif isinstance(obj, type):
        return obj.__name__
    elif isinstance(obj, dict):
        json_dict = {} 
        for key in obj.keys():
            if not isinstance(obj[key], __functypes) and not (type(key) == str and key[:1]=="_"):
                json_dict.update({key: jcsvlize(obj[key], f"{name}-{key}")})
        return json_dict
    elif isinstance(obj, object) and not isinstance(obj, __functypes):
        cls_name = obj.__class__.__name__
        try: obj_dict = jcsvlize(obj.__getstate__(), name)
        except AttributeError: obj_dict = jcsvlize(obj.__dict__, name)
        
        obj_dict.update({"__cls_name__": cls_name}) 
        return obj_dict 

def objectlize(js):
    """转换为python对象"""
    if isinstance(js, type(None)):
        return js 
    elif isinstance(js, (int, float, complex)):
        return js 
    elif isinstance(js, str):
        return js 
    elif isinstance(js, (list, tuple)) and np.array(js).dtype in __numertypes:
        return list(js)
    elif isinstance(js, (list, tuple)):
        return [objectlize(cont) for cont in js]
    elif isinstance(js, dict) and "__cls_name__" in js.keys():  #实例
        for key in js.keys(): 
            if key[:2] != "__": js[key] = objectlize(js[key])
        exec(f"obj = {js['__cls_name__']}()")
        exec(f"try: obj.__setstate__(js) \nexcept AttributeError: obj.__dict__.update(js)")
        return locals()["obj"]
    elif isinstance(js, dict):  #字典容器
        return {key: objectlize(js[key]) for key in js.keys()}
    else: 
        return js 



def auto_load(cls, path_str: str):
    """根据id字符串加载,返回实例"""
    #自定义名称
    path_list = path_str.split("/")
    fname_str = path_list[-1]
    name_list = fname_str.split(".")
    fmt = name_list[-1] if len(name_list) > 1 and name_list[-1] in scfg.fmts else scfg.fmt
    name = name_list[0]
    path = "/".join(path_list[:-1]) if len(path_list)>1 else scfg.dir
    path_name = f"{path}/{cls.__name__}-{name}.{fmt}"

    if fmt == "pkl":
        with open(path_name, "rb") as in_file:
            # obj = cls().__setstate__(pickle.load(in_file).__dict__)
            obj = pickle.load(in_file)
    elif fmt in ("json", "jcsv"):
        with open(path_name, "r") as in_file:
            obj = objectlize(json.load(in_file))
    else:
        raise NotImplementedError(f"load object failed with format {fmt}")
    return obj 


def auto_save(self, path_str=""):
    """默认保存为类名+id字符串"""
    #自定义名称
    path_list = path_str.split("/")
    fname_str = path_list[-1]
    name_list = fname_str.split(".")
    fmt = name_list[-1] if len(name_list) > 1 and name_list[-1] in scfg.fmts else scfg.fmt
    name = name_list[0]
    path = "/".join(path_list[:-1]) if len(path_list)>1 else scfg.dir
    path_name = f"{path}/{self.__class__.__name__}-{name}.{fmt}"

    #保存数据
    if not os.path.isdir(path):
        os.mkdir(path)
    if fmt == "pkl":
        with open(path_name, "wb") as out_file:
            pickle.dump(self, out_file)
            print(f"{path_name} saved.")
    elif fmt == "json":
        with open(path_name, "w") as out_file:
            json.dump(jsonlize(self), out_file, indent=True)
            print(f"{path_name} saved.")
    elif fmt == "jcsv":
        with open(path_name, "w") as out_file:
            json.dump(jcsvlize(self, name), out_file, indent=True)
            print(f"{scfg.dir}/{self.__class__.__name__}-{name}.{fmt} saved.")
    else:
        raise NotImplementedError(f"save object failed with format {fmt}")
    return 


def save_vol(self, path_str):
    #自定义名称
    path_list = path_str.split("/")
    fname_str = path_list[-1]
    name_list = fname_str.split(".")
    fmt = name_list[-1] if len(name_list) > 1 and name_list[-1] in scfg.vol_fmts else scfg.vol_fmt
    name = name_list[0]
    path = "/".join(path_list[:-1]) if len(path_list)>1 else scfg.dir
    path_name = f"{path}/{self.__class__.__name__}-{name}.{fmt}"

    if fmt == "npy":
        np.save(path_name, self.vol[:])
    elif fmt == "txt":
        np.savetxt(path_name, self.vol[:])
    else:
        from skimage import io
        io.imsave(path_name, self.vol[:])

    print(f"{scfg.dir}/{self.__class__.__name__}-{name}.{fmt} saved.")
    return 

@classmethod
def load_vol(cls, path_str, *args):
    path_list = path_str.split("/")
    fname_str = path_list[-1]
    name_list = fname_str.split(".")
    fmt = name_list[-1] if len(name_list) > 1 and name_list[-1] in scfg.vol_fmts else scfg.vol_fmt
    name = name_list[0]
    path = "/".join(path_list[:-1]) if len(path_list)>1 else scfg.dir
    path_name = f"{path}/{cls.__name__}-{name}.{fmt}"

    if fmt == "npy":
        return cls(np.load(path_name))
    elif fmt == "txt":
        return cls(np.loadtxt(path_name))
    else:
        return cls.from_img(path_name, *args)


for __cls in (Seeds, DistFunc, Phase, ObjectRelation, PhsRltSCC, Object):
    #不建议保存Link对象 其依赖在读取后很可能发生更新而改变了底层数据
    __cls.save = auto_check(auto_save)
    __cls.load = classmethod(auto_load)

for __cls in (DataVol, ):
    __cls.save = auto_save
    __cls.load = classmethod(auto_load)
    __cls.save_vol = save_vol
    __cls.load_vol = load_vol
