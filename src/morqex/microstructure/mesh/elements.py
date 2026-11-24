from morqex.microstructure.core.voxel_fields import *  
import os
import pickle



#数值精度控制
def set_float_precision(value_arr, precision=12, zero_th=1e-32):
    value_arr[np.where(np.abs(value_arr) < zero_th)] = 0
    value_arr = np.round(value_arr, precision)
    return value_arr


def hex_id(obj):
    return hex(id(obj))[-Mesh.id_len:]


class Mesh():
    mesh_dir = "meshes"
    id_len = 4
    cell_type = "unknow"
    
    #格式声明 
    mesh_fmt = "vtk"

    #当前网格类型支持的导出格式
    meshio_fmts = ['abaqus', 'ansys', 'avsucd', 'cgns', 'dolfin-xml', 'exodus', 'flac3d', 'gmsh', 'gmsh22', 
            'h5m', 'hmf', 'mdpa', 'med', 'medit', 'nastran', 'nas', 'netgen', 'neuroglancer', 'obj', 'off', 
            'permas', 'ply', 'stl', 'su2', 'svg', 'tecplot', 'tetgen', 'ugrid', 'vtk', 'vtk42', 
            'vtk51', 'vtu', 'wkt', 'xdmf', 'msh']
    comsol_fmts = ["mph", "mphtxt"]
    hscae_fmts = ["hsdat", "dat"]
    fmt_list = meshio_fmts + comsol_fmts + hscae_fmts

    #缺陷检查
    min_vol_rth = 1e-2

    def __init__(self, verts=np.array([]), cells=np.array([]), vert_data=np.array([]), cell_data=np.array([])): #, vdat_dict={}, cdat_dict={}):
        self.verts = verts
        self.cells = cells 
        self.vert_data = vert_data 
        self.cell_data = cell_data
        # self.vdat_dict = vdat_dict
        # self.cdat_dict = cdat_dict

        self.cell_mark = np.array([])  #边界单元标记(和cell_data相同结构,不同序号表示不同边界)
        self.face_mark = np.array([])

    @classmethod
    def parse_name(self, path_str=""):
        #自定义名称
        path_list = path_str.split("/")
        fname_str = path_list[-1]
        name_list = fname_str.split(".")
        fmt = name_list[-1] if len(name_list) > 1 and name_list[-1] in self.fmt_list else self.mesh_fmt
        name = name_list[0]
        path = "/".join(path_list[:-1]) if len(path_list)>1 else self.mesh_dir
        path_name = f"{path}/{self.__class__.__name__}-{name}.{fmt}"
        return path, name, fmt

    def save_mesh(self, path_str="", on=True):
        path, name, fmt = self.parse_name(path_str)

        if not os.path.isdir(path):
            os.mkdir(path)

        #检查内容
        if not(self.n_cell and self.n_vert):
            raise RuntimeError("save mesh failed: cells and verts is empty")
        
        #精度控制 
        self.verts = set_float_precision(self.verts, 12)

        #使用meshio保存
        if fmt in self.meshio_fmts:
            self.write_meshio(path, name, fmt, on=on)
        #comsol网格文件
        elif fmt in self.comsol_fmts:
            self.write_mph(path, name, "mphtxt", on=on)
        elif fmt in self.hscae_fmts: 
            self.write_hsdat(path, name, "hsdat", on=on)
            self.write_bounds(path, name, "hsbnd", on=on)
        else:
            raise NotImplementedError(f"save mesh failed: {fmt} is not support")
        
        if on: print(self)
        return 
    
    @classmethod
    def load_mesh(cls, name_str="", on=True):
        path, name, fmt = cls.parse_name(name_str)
        if fmt in cls.comsol_fmts:
            #TODO 
            # return cls.read_mphtxt(path, name, fmt)
            raise NotImplementedError()
        elif fmt in cls.meshio_fmts: 
            return cls.read_meshio(path, name, fmt, on=on)

    @classmethod
    def read_meshio(cls, path="", name="meshio_mesh", fmt="vtk", on=True):
        import meshio
        io_mesh = meshio.read(f"{path}/{cls.__name__}-{name}.{fmt}")

        cell_type = list(io_mesh.cells_dict.keys())[0]  #默认一种单元
        verts = io_mesh.points
        cells = io_mesh.cells_dict[cell_type]
        
        #属性赋值
        cell_data = np.ones(cells.shape[0], dtype=np.float32)
        vert_data = np.ones(verts.shape[0], dtype=np.float32)
        cdat_dict = {}
        vdat_dict = {} 
        for cell_data_str in io_mesh.cell_data.keys():
            cdat = io_mesh.cell_data[cell_data_str][0].reshape(-1)
            cdat_dict.update({cell_data_str: cdat})
            cell_data = cdat  #默认存储一种属性
        for point_data_str in io_mesh.point_data.keys():
            vdat = io_mesh.point_data[point_data_str].reshape(-1)
            vdat_dict.update({point_data_str: vdat})
            vert_data = vdat 

        if on: print(f"meshio readed => {path}/{cls.__name__}-{name}.{fmt}")
        return cls(verts, cells, vert_data, cell_data), vdat_dict, cdat_dict
        
    @classmethod
    def read_gmsh(cls, path="", name="meshio_mesh", fmt="vtk"):
        """调用meshio.gmsh读取gmsh文件"""
        #TODO 

    def __str__(self) -> str:
        if self.unique_attrs.size > 10:
            return f"\n{self.cls_str}(n_cell:{self.n_cell} n_vert:{self.n_vert} attrs:...)"
        vf = self.vol_frac
        for attr in vf:
            vf[attr] = np.round(vf[attr], 3)
        return f"\n{self.cls_str}(n_cell:{self.n_cell} n_vert:{self.n_vert} attrs:{vf})"

    def __repr__(self):
        return self.__str__() 

    def __getstate__(self):
        return {"verts": self.verts, "cells": self.cells, "vert_data": self.vert_data, "cell_data": self.cell_data}
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.verts = np.array(state["verts"])
        self.cells = np.array(state["cells"], dtype=np.int32)
        self.vert_data = np.array(state["verts"])
        self.cell_data = np.array(state["cell_data"])

    @property
    def n_cell(self):
        return np.shape(self.cells)[0]
    @property
    def n_vert(self):
        return np.shape(self.verts)[0]
    @property
    def n_attr(self):
        return self.unique_attrs.size 
    @property
    def unique_attrs(self):
        return np.unique(self.cell_data)
    @property
    def cell_centers(self):
        if self.n_cell > 0:
            return np.average(self.verts[self.cells], axis=1)
        else: 
            return np.array([])
    @property 
    def vol_frac(self):
        cvol = self.cell_volume
        mesh_vol = np.sum(cvol) 
        vol_frac_dict = {} 
        for attr in np.unique(self.cell_data):
            attr_vf = np.sum(cvol[self.cell_data == attr]) 
            vol_frac_dict.update({attr: attr_vf / mesh_vol})
        return vol_frac_dict
    @property
    def spec_surf_area(self):
        """比表面积"""
        ssa_dict = {}
        for attr in self.unique_attrs:
            sub_mesh = self.mask_mesh(self.cell_data == attr)
            ssa_dict.update({attr : sub_mesh.surf_area / sub_mesh.reg_volume}) 
        return ssa_dict

    @property 
    def cell_volume(self):
        return np.ones(self.cells.shape[0])  #默认网格内为均匀体积
    @property
    def reg_volume(self):
        # return np.prod(np.max(self.verts, axis=0) - np.min(self.verts, axis=0))
        return np.sum(self.cell_volume) 
    @property
    def cls_id_str(self):
        return f"{self.cls_str}-{self.id_str}"
    @property
    def cls_str(self):
        return f"{self.__class__.__name__}"
    @property
    def id_str(self):
        return f"{hex_id(self)}"

    def find(self, mark):
        return np.where(self.cell_data==mark)[0].astype(np.int32)

    def mask_mesh(self, cell_mask):
        """从当前单元网格掩膜构造新的网格"""
        #单元删除
        mask_cells = self.cells[cell_mask]  #筛选当前相的单元
        mask_vert_idx = np.unique(mask_cells)  #当前相使用的顶点序号

        #顶点删除
        new_verts = self.verts[mask_vert_idx]  #使用的顶点坐标
        new_vert_data = self.vert_data[mask_vert_idx] if self.vert_data.size else self.vert_data
        
        #重建顶点索引引用
        vert_idx_tab = np.zeros(np.max(mask_cells) + 1, dtype=np.uint32) #顶点索引映射表
        vert_idx_tab[mask_vert_idx] = np.arange(mask_vert_idx.size) #新建顶点索引
        new_cells = vert_idx_tab[mask_cells]
        new_cell_data = self.cell_data[cell_mask] if self.cell_data.size else self.cell_data   #映射到新的单元属性
    
        return self.__class__(new_verts, new_cells, new_vert_data, new_cell_data)

    def grid_sample(self, n_intp=(50,50,50), mode="vert"):
        from scipy.interpolate import griddata
        
        if isinstance(n_intp, int):
            n_intp = [n_intp] * 3 

        #采样属性源
        if mode == "vert":
            points = self.verts 
            data = self.vert_data 
        elif mode == "cell":
            points = self.cell_centers 
            data = self.cell_data 

        #网格采样点 
        point_range = np.array([np.min(points, 0), np.max(points, 0)]) 
        x_pnts = np.linspace(point_range[0, 0], point_range[1, 0], n_intp[0])
        y_pnts = np.linspace(point_range[0, 1], point_range[1, 1], n_intp[1])
        z_pnts = np.linspace(point_range[0, 2], point_range[1, 2], n_intp[2])
        grid_pnts = np.array(np.meshgrid(x_pnts, y_pnts, z_pnts)).reshape((3,-1)).T
        
        #网格插值
        grid_value = np.swapaxes(griddata(points, data, grid_pnts).reshape(n_intp[0], n_intp[1], n_intp[2]), 0, 2)
        return DataVol(grid_value) 


    def write_meshio(self, path="", name="gmsh_mesh", fmt="vtk", on=True):
        """调用meshio写文件"""
        import meshio

        #一致性
        verts = self.verts
        cell_type = self.cell_type
        cells = self.cells

        #默认单元属性
        vert_data = np.ones(verts.shape[0], dtype=np.float32) if not self.vert_data.size else self.vert_data
        cell_data = np.ones(cells.shape[0], dtype=np.float32) if not self.cell_data.size else self.cell_data

        #vtk浮点数精度
        vert_data = vert_data.astype(np.float32) if fmt == "vtk" else vert_data
        cell_data = cell_data.astype(np.float32) if fmt == "vtk" else cell_data 

        #写出文件
        if fmt == "xdmf":  #不写顶点属性
            io_mesh = meshio.Mesh(verts, [(cell_type, cells)], cell_data = {"cell_phase": [cell_data]})    
        else:
            io_mesh = meshio.Mesh(verts, [(cell_type, cells)], {"pnt_phase": vert_data}, {"cell_phase": [cell_data]})
        io_mesh.write(f"{path}/{self.cls_str}-{name}.{fmt}")
        if on: print(f"meshio writed => {path}/{self.cls_str}-{name}.{fmt}")
        return 

    def write_gmsh(self, path="", name="meshio_mesh", fmt="msh", binary=False, on=True):
        """调用meshio.gmsh写.msh文件"""
        import meshio

        cell_dim = [(attr, self.ndim) for attr in self.unique_attrs if attr != 0]
        cell_name = [f"attr{attr}" for attr in self.unique_attrs if attr != 0]
        field_data = {name: dim for name, dim in zip(cell_dim, cell_name)}
        # bnd_dim
        # bnd_name
        io_mesh = meshio.Mesh(points=self.verts, 
            cells=
            [("tetra", self.cells), 
            ("triangle", self.bnds)],  #TODO 三角表面
            cell_data={
            "gmsh:physical": [self.cell_data, self.bnd_data], 
            "gmsh:geometrical": [self.cell_data, self.bnd_data],}, 
            field_data = {
            "cathode": (1, self.ndim), 
            "anode": (2, self.ndim), 
            "electrolyte": (3, self.ndim), 
            "seperator": (4, self.ndim), 
            "ctd_reaction_sur": (1, self.ndim - 1), 
            "and_reaction_sur": (2, self.ndim - 1), 
            "pos_current_sur": (3, self.ndim - 1), 
            "neg_current_sur": (4, self.ndim - 1), }
            )
        meshio.gmsh.write(f"{path}/{self.cls_str}-{name}.{fmt}", io_mesh, binary=binary, fmt_version="2.2")
        if on: print(f"gmsh writed => {path}/{self.cls_str}-{name}.{fmt}")
        return 

    def write_mph(self, path="", name="mph_mesh", fmt="mphtxt", on=True): #, info=True):
        """导出comsol支持的mph格式(三角形, 四边形, 四面体, 六面体)"""
        # if self.cell_type not in 
        cell_type = {"tetra": "tet", "hexahedron": "hex", "triangle" : "tri", "quadrilateral" : "quad"}.get(self.cell_type, self.cell_type)
        verts = self.verts
        cells = self.cells 
        attrs = self.cell_data.astype(np.int32)
        if 0 in attrs:
            attrs[attrs==0] = np.max(attrs) + 1
            warnings.warn("write_mph warning: domain idx 0 is not supported in mph format, auto fixed")

        attr_list = np.sort(np.unique(attrs)).astype(np.int32) #整数属性
        attr_strs = [f"attr_{attr}" for attr in attr_list]

        n_vert = self.n_vert 
        n_cell = self.n_cell
        n_vert_cell = cells[0].size 
        n_dim = verts[0].size 
        n_ent = self.n_cell
        n_phase = self.n_attr 
        n_type = n_phase + 1 # 1对应Mesh实体 其余为selection

        min_vert_idx = np.min(self.cells)

        #comsol顶点顺序修正, 默认为vtk顺序
        if cell_type == "hex": cells = cells[:, (0, 1, 3, 2, 4, 5, 7, 6)] 

        fstr = "# mph mesh generated by PntCorGraph(YSY)\n"
        fstr += "# Major & minor version\n"
        fstr += "0 1 \n"
        fstr += "\n"

        fstr += "# number of tags\n"
        fstr +=  f"{n_type} \n"  #标签数目 
        fstr += "# Tags\n"
        fstr += f"{len(name)} {name} \n" #网格标签  #标签字符数(对象名称) 标签字符串
        
        fstr += "".join(f"{len(attr_str)} {attr_str} \n" for attr_str in attr_strs)  #属性标签
        fstr += "\n"

        fstr += "# number of types\n" 
        fstr += f"{n_type} \n"
        fstr += "# Types\n"
        fstr += "".join(["3 obj \n" for _ in range(n_type)])  #物体数目
        fstr += "\n"
        type_cnt = 0

        #网格物体 
        type_cnt += 1 
        fstr += f"# --------- Object {type_cnt} ----------\n"
        fstr += "0 0 1 \n"  #固定值: 物体序列头
        fstr += "4 Mesh #class\n"  #物体的类: 可序列化的Mesh或Selection
        fstr += "4 # version\n" 
        fstr += f"{n_dim} # sdim\n" #物体维度
        fstr += "\n"

        #顶点坐标头数据 
        fstr += "# num of verts \n"
        fstr += f"{n_vert}\n"
        fstr += "# start vert idx \n"
        fstr += f"{min_vert_idx}\n"
        fstr += "\n"

        fstr += "# Mesh vertex coordinates\n"
        if n_dim == 2:
            fstr += "".join([f"{vert[0]} {vert[1]}\n" for vert in verts]) 
        elif n_dim == 3:
            fstr += "".join([f"{vert[0]} {vert[1]} {vert[2]}\n" for vert in verts]) 
        else:
            raise NotImplementedError(f"vert dimension {n_dim} is not supported")
        fstr += "\n"

        #元素列表 (几何顶点,边)
        fstr += "# number of element types\n"
        fstr += "1\n"  #单元类型数目 
        fstr += "\n"
        elem_type_cnt = 0 

        #第一类元素列表
        elem_type_cnt += 1 
        fstr += f"# type {elem_type_cnt}\n"
        fstr += f"{len(cell_type)} {cell_type}\n" #声明单元类型

        fstr += f"# number of vertice per element\n"
        fstr += f"{n_vert_cell}\n"  #每个元素顶点数目 
        fstr += "\n"
        fstr += f"# number of elements\n"
        fstr += f"{n_cell}\n"
        fstr += f"# elements \n"
        if n_vert_cell == 3:
            fstr += "".join([f"{cell[0]} {cell[1]} {cell[2]}\n" for cell in cells])
        elif n_vert_cell == 4:
            fstr += "".join([f"{cell[0]} {cell[1]} {cell[2]} {cell[3]}\n" for cell in cells])
        elif n_vert_cell == 8:
            fstr += "".join([f"{cell[0]} {cell[1]} {cell[2]} {cell[3]} {cell[4]} {cell[5]} {cell[6]} {cell[7]}\n" for cell in cells])
        fstr += "\n"

        fstr += "# number of entities\n"
        fstr += f"{n_ent}\n"  #几何实体(和cell对应)序号
        fstr += "# entity indices\n"
        fstr += "".join([f"{attr}\n" for attr in attrs])
        fstr += "\n"

        #选择对象 
        for attr_str, attr in zip(attr_strs, attr_list):
            type_cnt += 1 
            fstr += f"# --------- Object {type_cnt} ----------\n"
            fstr += "0 0 1 \n"  #固定值: 物体序列头
            fstr += "9 Selection # class\n"  #物体的类: 可序列化的Mesh或Selection
            fstr += "0 # version\n"   #注意selection版本不同 
            fstr += f"{len(attr_str)} {attr_str}\n"  #label
            fstr += f"{len(name)} {name}\n"  #引用网格的tag
            fstr += f"{n_dim} # sdim\n" #物体维度
            
            #指定实体
            fstr += "# number of entities\n"
            fstr += "1\n"
            fstr += "# entities\n"
            fstr += f"{attr}\n"
            fstr += "\n"

        with open(f"{path}/{self.cls_str}-{name}.{fmt}", "w") as out_file:
            out_file.write(fstr)

        if on: print(f"mphtxt writed => {path}/{self.cls_str}-{name}.{fmt}")

        # if info:
        #     self.save_mph_info(name_str)

    # def save(self, name_str=""):
    #     """默认保存为类名+id字符串"""
    #     #自定义名称
    #     name_str = self.id_str if not len(name_str) else name_str 

    #     #保存数据
    #     if not os.path.isdir(self.mesh_dir):
    #         os.mkdir(self.mesh_dir)
    #     with open(f"{self.mesh_dir}/{self.cls_str}-{name_str}.pkl", "wb") as out_file:
    #         pickle.dump(self, out_file)
    #         print(f"{self.mesh_dir}/{self.cls_str}-{name_str}.pkl saved.")
    #     return 

    # @classmethod
    # def load(cls, name_str):
    #     """根据id字符串加载,返回实例"""
    #     with open(f"{cls.mesh_dir}/{cls.__name__}-{name_str}.pkl", "rb") as in_file:
    #         obj = pickle.load(in_file)
    #     return obj 

    # def get_vol(self, shape=(50,50,50)) -> DataVol:
    #     """插值查询属性值"""
    #     return DataVol(query_value(self.cell_centers, self.cell_data, shape)) 

    # def save_config(self, name_str=""):
    #     name_str = self.id_str if not len(name_str) else name_str 
        
    #     head_str = "# mesh generated by PntCorGraph(YSY)\n"
    #     config_str = f"n_cell:{self.n_cell} \nn_vert:{self.n_vert} \nvolume:{self.reg_volume} \nvol_frac:{self.vol_frac}"

    #     fname = f"{self.mesh_dir}/{self.cls_str}-{name_str}.config"
    #     with open(fname, "w") as cfg_file:
    #         cfg_file.write(head_str + config_str)

    # def save_mph_info(mesh, name_str: str):
    #     """保存网格信息 mph参数"""
    #     name_str = mesh.id_str if not len(name_str) else name_str 
        
    #     head_str = "# mesh info generated by PntCorGraph(YSY)\n"
    #     info_str =  f"edge_rth \t0.06 \tedge thickness relative threshold\n"

    #     thickness = (np.max(mesh.verts, axis=0) - np.min(mesh.verts, axis=0))[0]
    #     info_str += f"reg_thickness \t{thickness}[m] \telectrode thickness\n"

    #     info_str += f"acm_vf \t{mesh.vol_frac[1]} \tvolume fraction of ACM\n"
    #     info_str += f"reg_volume \t{mesh.reg_volume}[m^3] \tvolume of represent region\n"
        
    #     info_str += f"acm_volume \tacm_vf * reg_volume \tvolume of ACM\n"

    #     info_str += f"cell_cnt \t{mesh.n_cell} \tnumber of cells\n"
    #     info_str += f"vert_cnt \t{mesh.n_vert} \tnumber of verts\n"

    #     fname = f"{mesh.mesh_dir}/{mesh.cls_str}-{name_str}-info.txt"
    #     with open(fname, "w") as info_file:
    #         info_file.write(head_str + info_str)
        
    def attr_map(self, map: list[int]):
        self.cell_data = np.array(map)[self.cell_data]

    @auto_vol 
    def vol_query(self, vol, stable=False):
        """顶点插值采样 网格采样 返回datavol"""
        verts = self.verts[:, :self.ndim] / np.max(self.verts)
        gird_key = [slice(None, ax_len) for ax_len in vol.shape]
        rloc_vol = DistVol((move_axes(np.mgrid[gird_key]) + 0.5) / np.max(vol.shape))
        value = self.query(rloc_vol.reshape(-1, 3)[:, :verts.shape[1]], stable=stable)
        return DataVol(value.reshape(vol.shape))

    @auto_vol 
    def query(self, loc, stable=False):
        """顶点插值采样"""
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        Interpolator = NearestNDInterpolator if stable else LinearNDInterpolator

        verts = self.verts[:, :self.ndim] / np.max(self.verts)

        vitp = Interpolator(verts, self.vert_data)
        value = vitp(loc[:, :self.ndim])
        return value
    
    @property 
    def ndim(self):
        """空间维度 默认verts按顺序填充维度"""
        if not np.any(self.verts[:, 1]): 
            return 1
        elif not np.any(self.verts[:, 2]): 
            return 2
        else: 
            return 3

class LineMesh(Mesh):
    cell_type = "line"

    @classmethod
    @auto_vol 
    def from_vol(cls, vol: PhaseVol):
        """从体素网格(三维, (n, 1, 1))初始化的线段网格"""
        n_cell = vol.shape[0]
        n_vert = n_cell + 1
        verts = np.vstack([np.linspace(0.0, 1.0, n_vert), np.zeros(n_vert),  np.zeros(n_vert)]).T
        cells = np.vstack([np.arange(0, n_cell), np.arange(1, n_cell+1)]).T
        cell_data = vol[:, 0, 0]
        return cls(verts, cells, cell_data=cell_data)



class TriMesh(Mesh):
    cell_type = "triangle"

    @property
    def surf_area(self):
        return np.sum(self.cell_volume)
    
    @property 
    def h(self):
        """"""

    @property
    def cell_volume(self):
        """网格表面积计算"""
        if self.verts.shape[1] == 2:
            verts = np.concatenate([self.verts, np.full((self.verts.shape[0], 1), 0.0)], axis=1)
        else:
            verts = self.verts

        triangles = self.cells 
        tri_verts = verts[triangles]  # 形状变为 (n_triangles, 3, 3)
        
        # 提取三个顶点 A, B, C
        A = tri_verts[:, 0, :]
        B = tri_verts[:, 1, :]
        C = tri_verts[:, 2, :]
        
        # 计算两个边向量
        AB = B - A
        AC = C - A
        
        # 计算叉积 (AB × AC)
        cross = np.cross(AB, AC)
        
        # 计算叉积的模长（向量长度）
        cross_magnitude = np.linalg.norm(cross, axis=1)
        
        # 三角形面积是叉积模长的一半
        areas = cross_magnitude / 2.0
        return areas

    @classmethod
    def volres_weight(cls, phs_vol: PhaseVol, smp_verts, sigma=1.0, n_level=np.inf, wfac=2.0):
        """残差权重计算  中值+平滑/高斯"""
        n_level = min(int(np.log2(phs_vol.vxl_len)-2.0), n_level) 
        rsds = phs_vol.multi_residual(sigma, n_level=n_level)    #NOTE median中值平均, 可能出现全为空的结果(原因是此时恰好局部为规则形状)
        rwts = np.sum([(rsd[:]/2**i)**wfac for i, rsd in enumerate(rsds)], axis=0)
        rwts = DataVol(rwts).multi_smooth(sigma, n_level=n_level)
        wgt_vol = np.sum([rwt[:] for rwt in rwts], axis=0)
        wgt_vol = DataVol(wgt_vol/np.sum([(1/2**i)**wfac for i in range(n_level)]))
        # wgt_vol = DataVol(rwts/np.sum([(1/2**i)**wfac for i in range(n_level)]))
        wtgs = wgt_vol.query(smp_verts/phs_vol.vxl_len)
        return wtgs
    

    @classmethod
    def mcvrt_weight(cls, phs_vol: PhaseVol, smp_verts, sigma=1.0, n_level=np.inf, wfac=2.0):
        """残差权重计算  中值+平滑/高斯"""
        #缩放到体素空间
        phs_vol = PhaseVol(phs_vol)

        #体素网格离散权重
        wgt_vol = DataVol.zeros(np.array(phs_vol.shape)+1)
        n_level = min(int(np.log2(phs_vol.vxl_len)-2.0), n_level) 

        wgts = np.ones(smp_verts.shape[0])
        for i in range(1, n_level):  #从1开始, 跳过第一层属性查询, 改为从1初始化
            scale = 2**i
            shape = np.array(np.array(phs_vol.shape)/scale+1, np.int32)
            wgt_vol = DataVol.zeros(shape)
            
            #两个位置loc, 避免舍入误差影响
            locs = np.array(smp_verts/scale, dtype=np.int32).T
            wgt_vol[locs[0], locs[1], locs[2]] += (1/scale)**(wfac/sigma)/2

            locs = np.array((smp_verts+0.5)/scale, dtype=np.int32).T
            wgt_vol[locs[0], locs[1], locs[2]] += (1/scale)**(wfac/sigma)/2

            wgts += wgt_vol.query(smp_verts/phs_vol.vxl_len)
        return wgts

    @classmethod
    @auto_vol 
    def from_vol(cls, vol: PhaseVol, vsub=0.1, sigma=1.0, n_level=np.inf, wfac=2.0):
        """从体素网格初始化的四面体网格"""
        #相网格和上色网格
        vol = PhaseVol(vol)
        npad = min(int(np.log2(PhaseVol(vol).vxl_len)-2.0), n_level)
        if npad < 1: raise RuntimeError(f"trimesh init failed: input vol is too small {PhaseVol(vol).vxl_len}, length should >= 8")
        phs_vol = PhaseVol(np.pad(vol[:], npad))
        if phs_vol.is_zero: raise ValueError("trimesh init failed: invalid input vol") #空输入

        #连通性优化
        # clr_vol = phs_vol.label_vol().color_vol()
        clr_vol = phs_vol.color_vol()

        #多相表面提取
        mc_verts, mc_faces, _, _ = attr_marching_cubes(clr_vol, method="lorensen")  #lorensen不连接对角
        mc_verts += 0.5           #边界坐标从0.0开始, 体素中心0.5, 1.5...

        #计算权重(用原始体素网格坐标, 避免舍入误差)
        wgts = cls.volres_weight(clr_vol, mc_verts, sigma=sigma, n_level=n_level, wfac=wfac) 
        # wgts = cls.mcvrt_weight(clr_vol, mc_verts, sigma=sigma, n_level=n_level, wfac=wfac)

        #多相表面提取四边形表面  由于ACVD是三角面聚类效果不好
        # mc_verts, mc_faces = extract_surface_triangles(clr_vol)   #attr_surface_extract
        # mc_verts /= phs_vol.vxl_len   #转化为相对坐标

        #归一化
        mc_verts /= phs_vol.vxl_len   #转化为相对坐标

        #表面优化
        nvrt = int(np.sum(wgts) * vsub)
        cvd_verts, cvd_faces = sur_optimize(mc_verts, mc_faces, nvrt, wgts)
        #修复退化单元
        cvd_verts, cvd_faces, ivld_tets = remove_degenerate(cvd_verts, cvd_faces, quality_th=0.2)
        # cvd_verts, cvd_faces, _ = remove_degenerate(cvd_verts, cvd_faces, quality_th=0.05)

        #查询属性
        face_attr = phs_vol.query_attr(np.average(cvd_verts[cvd_faces], axis=1), smooth=phs_vol.attrs.size<8)

        #还原填充
        cvd_verts -= npad / phs_vol.vxl_len
        cvd_verts /= (phs_vol.vxl_len - npad *2) / phs_vol.vxl_len
        if ivld_tets.size: TetMesh(cvd_verts, ivld_tets, cell_data=np.arange(ivld_tets.shape[0])+1).save_mesh("debug.vtk")

        #移除退化单元
        quality_th = 0.01
        mesh = cls(cvd_verts, cvd_faces, cell_data=face_attr)
        quality = mesh.skewness_quality
        #删除
        if np.any(quality < quality_th):
            warnings.warn(f"TriMesh from_vol warn: {np.where(quality < quality_th)[0].size} low qualtiy cells detected, auto fix")
            mask = quality >= quality_th
            return mesh.mask_mesh(mask)
        return mesh


    @classmethod
    @auto_vol 
    def from_vol2d(cls, vol: PhaseVol, vsub=0.1, sigma=1.0, n_level=np.inf, wfac=1.0):
        """从体素网格初始化, 平面三角网格"""
        #相网格和上色网格
        if vol.ndim == 2: vol = vol[:, :, None]
        vol = PhaseVol(vol)
        npad = min(int(np.log2(PhaseVol(vol).vxl_len)-2.0), n_level)
        vol = np.repeat(vol[:, :, 0:1], npad*2, axis=2)

        phs_vol = PhaseVol(np.pad(vol[:], npad))
        if phs_vol.is_zero: raise ValueError("trimesh init failed: invalid input vol") #空输入
        
        #连通性优化
        # clr_vol = phs_vol.label_vol().color_vol()
        clr_vol = phs_vol

        #多相表面提取
        mc_verts, mc_faces, _, _ = attr_marching_cubes(clr_vol, method="lorensen") 
        #截取z方向面
        tm = TriMesh(mc_verts, mc_faces).mask_mesh(np.average(mc_verts[mc_faces], axis=1)[:, 2] < npad+0.1)
        tm.verts[:,2][tm.verts[:,2] > npad-0.5+0.1] += 1.0  #加强边界(仅限二维)

        mc_verts, mc_faces = tm.verts, tm.cells
        mc_verts += 0.5           #边界坐标从0.0开始, 体素中心0.5, 1.5...
        mc_verts /= phs_vol.vxl_len   #转化为相对坐标

        #计算权重
        wgt_vol = cls.volres_weight(clr_vol, sigma=sigma, n_level=n_level, wfac=wfac)

        #表面优化
        wgts = wgt_vol.itp_query(mc_verts)  #属性在体素中心, 查询坐标处的最大分量属性值
        nvrt = int(np.sum(wgts) * vsub)
        cvd_verts, cvd_faces = sur_optimize(mc_verts, mc_faces, nvrt, wgts)
        
        #查询属性
        face_attr = phs_vol.query_attr(np.average(cvd_verts[cvd_faces], axis=1), smooth=phs_vol.attrs.size<8)

        #还原填充
        cvd_verts -= npad / phs_vol.vxl_len
        cvd_verts /= (phs_vol.vxl_len - npad *2) / phs_vol.vxl_len
        cvd_verts = cvd_verts[:, :2]
        return cls(cvd_verts[:, :2], cvd_faces, cell_data=face_attr)

    @property
    def skewness_quality(self):
        """
        向量化计算三角形网格的偏度
        输入: tris - (N, 3, 3)数组，N个三角形的顶点坐标
        输出: (N,)数组，每个三角形的偏度值(0~1，1表示等边三角形)
        """
        tris = self.verts[self.cells]  # (N, 3, 3)数组
        
        N = tris.shape[0]
        
        # 计算三角形的三个边向量
        # 边1: 从顶点0到顶点1
        v01 = tris[:, 1] - tris[:, 0]  # (N, 3)
        # 边2: 从顶点0到顶点2
        v02 = tris[:, 2] - tris[:, 0]  # (N, 3)
        # 边3: 从顶点1到顶点2
        v12 = tris[:, 2] - tris[:, 1]  # (N, 3)
        
        # 计算三个内角(使用余弦定理)
        # 计算各边的长度
        a = np.linalg.norm(v12, axis=1)  # 边v12的长度，对边为顶点0
        b = np.linalg.norm(v02, axis=1)  # 边v02的长度，对边为顶点1
        c = np.linalg.norm(v01, axis=1)  # 边v01的长度，对边为顶点2
        
        # 避免除零错误
        eps = 1e-12
        a_safe = np.maximum(a, eps)
        b_safe = np.maximum(b, eps)
        c_safe = np.maximum(c, eps)
        
        # 计算各个角度的余弦值
        cos_A = (b_safe**2 + c_safe**2 - a_safe**2) / (2 * b_safe * c_safe)
        cos_B = (a_safe**2 + c_safe**2 - b_safe**2) / (2 * a_safe * c_safe)
        cos_C = (a_safe**2 + b_safe**2 - c_safe**2) / (2 * a_safe * b_safe)
        
        # 限制余弦值在有效范围内并计算角度(度)
        clipped_cos_A = np.clip(cos_A, -1.0, 1.0)
        clipped_cos_B = np.clip(cos_B, -1.0, 1.0)
        clipped_cos_C = np.clip(cos_C, -1.0, 1.0)
        
        angle_A = np.degrees(np.arccos(clipped_cos_A))  # (N,)
        angle_B = np.degrees(np.arccos(clipped_cos_B))  # (N,)
        angle_C = np.degrees(np.arccos(clipped_cos_C))  # (N,)
        
        # 组合所有角度
        angles = np.column_stack([angle_A, angle_B, angle_C])  # (N, 3)
        
        # 方法1: 基于最小和最大角度与理想角度的偏差
        min_angles = np.min(angles, axis=1)  # (N,)
        max_angles = np.max(angles, axis=1)  # (N,)
        
        # 理想三角形内角 (等边三角形)
        ideal_angle = 60.0  # 等边三角形的内角
        
        # 计算偏度 - 基于角度偏差
        # 对于三角形，偏度通常定义为: max((理想角-最小角)/理想角, (最大角-理想角)/(180-理想角))
        skewness_low = (ideal_angle - min_angles) / ideal_angle
        skewness_high = (max_angles - ideal_angle) / (180 - ideal_angle)
        
        # 取两者中的较大值作为偏度
        skewness = np.maximum(skewness_low, skewness_high)
        
        # 限制在[0, 1]范围内
        skewness = np.maximum(skewness, 0)
        skewness = np.minimum(skewness, 1.0)
        
        # 返回质量因子 (1 - 偏度)，值越高表示质量越好
        return 1 - skewness
    

# def remove_degenerate(tri_verts, tri_faces): #, epsilon=1e-6):
#     """
#     移除有重叠部分的三角面 按照边连通性进行修复 同时会填补结构中的孔洞
#     examples: 
#         tri_verts = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 2, 0], [1, 2, 0]], dtype=np.float32)
#         tri_faces = np.array([[0, 1, 2], [1, 2, 3], [1, 4, 5], [1, 3, 4], [1, 3, 5], [3, 4, 5]], dtype=np.int32)
#         remove_degenerate(tri_verts, tri_faces)
#     """
#     from core.cyfunc.cy_geometry import find_tetrahedrons, fill_holes_bound

#     #填充孔洞, 然后查找四面体
#     # fill_faces = fill_holes(tri_faces)
#     # tri_faces = np.concatenate([tri_faces, fill_faces], axis=0)  #对tetgen成功率影响不大
#     tets, tets_faces = find_tetrahedrons(tri_faces.astype(np.int32))

#     #部分四面体网格 筛选退化四面体
#     # ivld_mask = TetMesh(tri_verts, tets).cell_volume < epsilon
#     quality_th = 0.01
#     ivld_mask = TetMesh(tri_verts, tets).skewness_quality < quality_th
#     if not np.any(ivld_mask): return tri_faces #无退化单元 返回

#     #退化四面体
#     ivld_tets = tets[ivld_mask]

#     #通过连通性修复面
#     del_face = fix_tetrahedrons(tri_faces, ivld_tets)

#     #合并保留面和修复面
#     _, idx = remove_points(np.sort(tri_faces, axis=1), np.sort(del_face, axis=1))
#     new_faces = tri_faces[idx]

#     #填补三个边围成的孔洞
#     fill_faces = fill_holes_bound(tri_faces)   #改善边界和属性查询
#     return np.concatenate([new_faces, fill_faces], axis=0) 


def remove_degenerate(tri_verts, tri_faces, quality_th=0.01):
    """
    移除有重叠部分的三角面 按照对角长度选择面进行修复
    examples: 
        verts = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0.5, 0], [1.5, 0.5, 0]])
        faces = np.array([[0, 1, 2], [0, 3, 2], [0, 1, 3], [1, 2, 3], [1, 2, 4], 
                        [2, 3, 6], [2, 6, 5], [3, 6, 5], [2, 3, 5]])
        tets = np.array([[0, 1, 2, 3], [2, 3, 6, 5]])
        remove_degenerate(verts, faces)
    """
    from morqex.microstructure.core.cyfunc.cy_geometry import find_tetrahedrons, fill_holes, fill_holes_bound

    #修补边
    fill_faces = fill_holes(tri_faces)
    ftri_faces = np.concatenate([tri_faces, fill_faces], axis=0)  #对tetgen成功率影响不大

    #在修补后的网格上查找四面体
    tets, _ = find_tetrahedrons(ftri_faces.astype(np.int32))

    #部分四面体网格 筛选退化四面体
    ivld_mask = TetMesh(tri_verts, tets).skewness_quality < quality_th
    # mesh = TetMesh(tri_verts, tets).mask_mesh(ivld_mask)
    # mesh.cell_data = np.arange(ivld_mask.size) + 1
    # mesh.save_mesh("debug.vtk")

    #无退化单元 返回
    if not np.any(ivld_mask): 
        return tri_verts, tri_faces, np.ndarray(shape=(0, 4), dtype=np.int32) 
    
    #退化四面体
    ivld_tets = tets[ivld_mask]

    #在填补后的网格上查询, 从中删除tet中不一致面
    # new_faces = fix_tetrahedrons_len(tri_verts, ivld_tets)
    new_verts, new_faces = fix_tetrahedrons(tri_verts, ftri_faces, ivld_tets)   
    
    #填补三个边围成的孔洞  有可能引入退化单元?
    fill_faces = fill_holes_bound(new_faces)
    new_faces = np.concatenate([new_faces, fill_faces], axis=0)    #改善边界和属性查询
    return new_verts, new_faces, ivld_tets


def remove_entities(entities, to_remove):
    """
    从实体列表中移除指定的实体（忽略顶点顺序），完全向量化实现。
    支持点、线、三角面、四面体等任意维度的实体。
    参数:
        entities (np.ndarray): 形状为 (N, k) 的数组，表示原始实体列表，
        to_remove (np.ndarray): 形状为 (M, k) 的数组，表示要移除的实体列表。
    返回:
        np.ndarray: 移除后的实体列表，形状为 (K, k)，K <= N。
    """
    entities = np.array(entities, dtype=np.int32)
    to_remove = np.array(to_remove, dtype=np.int32)
    if entities.ndim == 1: 
        entities = entities.reshape(-1, 1)
    if to_remove.ndim == 1: 
        to_remove = to_remove.reshape(-1, 1)
    
    if entities.shape[0] == 0 or to_remove.shape[0] == 0:
        return entities.copy()

    # 对每个实体的顶点排序（标准化顺序）
    sorted_ents = np.sort(entities, axis=1)
    sorted_remove = np.sort(to_remove, axis=1)
    
    # 获取实体维度
    k = entities.shape[1]

    # 将每行转换为连续的字节表示
    ents_view = np.ascontiguousarray(sorted_ents).view(f'V{sorted_ents.dtype.itemsize * k}').reshape(-1)
    remove_view = np.ascontiguousarray(sorted_remove).view(f'V{sorted_remove.dtype.itemsize * k}').reshape(-1)
    
    # 使用np.isin找出哪些实体在to_remove中
    mask = ~np.isin(ents_view, remove_view)

    # 返回保留的原始实体（保持原始顶点顺序）
    return entities[mask].copy()


# def remove_dengerate_delauany(verts, cells):
#     """只保留和delauany相交的面"""
#     from scipy.spatial import Delaunay
#     dly = Delaunay(verts)
#     dly_faces = extract_faces_tet(dly.simplices)
#     dly_faces = dly_faces[remove_dul_points(np.sort(dly_faces, axis=1))[1]]
#     all_faces = np.concatenate([cells, dly_faces], axis=0)
#     all_faces, _, _, cnts = remove_dul_points(np.sort(all_faces, axis=1))
#     return all_faces[cnts==2]


# def remove_unbound(faces):
#     """根据边的邻接性移除重叠面"""
#     from core.cyfunc.cy_geometry import compute_edge_adjacency

#     edge_face_map = compute_edge_adjacency(faces)
#     ivd_faces = [edge_face_map[edge][0] for edge in edge_face_map if len(edge_face_map[edge]) == 1]
#     return np.delete(faces, ivd_faces, axis=0)

# def fix_tetrahedrons_len(verts, tets):
#     """按照对角长度修复四面体"""
#     #BUG 不一定是长方形, 可能是梯形, 最长边不是对角线. 
#     #假设为凸四边形, 用最长边和短边构成新三角形
#     edge_len = edge_length(verts, tets)

#     #修复三角面的局部顶点映射 根据edge_len最长边选择保留面
#     fix_map = np.array([[[0, 1, 2], [0, 3, 1]], [[0, 2, 1], [0, 3, 2]], [[0, 3, 2], [0, 1 ,3]]])
#     fix_faces = np.concatenate([tets[i, fix_map[np.argmax(edge_len[i, :3])]] for i in range(tets.shape[0])], axis=0)
#     return fix_faces

def fix_tetrahedrons(verts, faces, tets):
    """将平面四面体划分转化为一致的三角面
    测试交叉面移除
    examples: 
        verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], 
                    [1, 1, 1], [0, 1, -1], [1, 0, -1]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [3, 0, 1], [1, 3, 2], [0, 4, 2], [3, 5, 1], [3, 1, 6]])
        tets = np.array([[0, 1, 2, 3]])
        fix_tetrahedrons(verts, faces, tets)
    """
    from scipy.spatial import ConvexHull
    from morqex.microstructure.core.cyfunc.cy_geometry import compute_edge_adjacency

    #前置计算 边连通性和长度
    ef_map = compute_edge_adjacency(faces.astype(np.int32))

    #对每个退化四面体
    del_faces = np.ndarray(shape=(0, 3), dtype=np.int32)
    add_faces = np.ndarray(shape=(0, 3), dtype=np.int32)
    tets_verts = verts[tets]
    for i in range(tets.shape[0]):
        tet = tets[i]
        tet_verts = tets_verts[i]

        #投影计算二维包
        verts_2d, tet_norm = _cal_2d_projection(tet_verts)
        cov2d = ConvexHull(verts_2d)
        vertices = tet[cov2d.vertices]  #局部编号->全局顶点编号

        #平面四边形
        if vertices.size == 4: 
            #convex局部对角线
            e1 = tuple(np.sort([vertices[0], vertices[2]]))
            e2 = tuple(np.sort([vertices[1], vertices[3]]))

            #连通性解析
            cnnt_e1 = ef_map[e1]
            cnnt_e2 = ef_map[e2]
            cnnt_cnt1 = np.sum(cnnt_e1 != -1)
            cnnt_cnt2 = np.sum(cnnt_e2 != -1)


            #分类进行面移除
            del_face = []
            add_face = []
            if cnnt_cnt1 >= 3 and cnnt_cnt2 >= 3: 
                #不修复交叉问题
                #NOTE 修复方法不完善 可能引起边界面丢失
                continue
                # warnings.warn(f"fix_tet failed: unexpected crossing faces at {np.average(tet_verts, axis=0)}")
                e1_verts = faces[cnnt_e1[cnnt_e1>=0]]
                e2_verts = faces[cnnt_e2[cnnt_e2>=0]]
                out_verts = remove_entities(np.append(e1_verts, e2_verts).ravel(), tet).ravel()

                #选择正负最远的两个连接四面体表面
                signs = np.dot(verts[out_verts], tet_norm)  #v方向表示
                
                #移除所有相关三角面
                del_face += np.append(e1_verts, e2_verts, axis=0).tolist()

                #存在正向最大点
                pos_idx = np.argmax(signs)
                if signs[pos_idx] > 0: 
                    pos_v = out_verts[pos_idx]
                    #连接四棱锥外表面
                    add_face += [[pos_v, e1[0], e2[0]], [pos_v, e1[0], e2[1]], 
                                [pos_v, e1[1], e2[0]], [pos_v, e1[1], e2[1]]]
                    add_face += [[pos_v, e1[0], e1[1]]]

                #存在负向最大点
                neg_idx = np.argmin(signs)
                neg_v = out_verts[neg_idx]
                if signs[neg_idx] < 0:
                    #连接四棱锥外表面
                    add_face += [[neg_v, e1[0], e2[0]], [neg_v, e1[0], e2[1]], 
                                [neg_v, e1[1], e2[0]], [neg_v, e1[1], e2[1]]]
                    add_face += [[neg_v, e1[0], e1[1]]]

                #补充对角三角面, 默认补充e1
                add_face += [[e1[0], e1[1], e2[0]], [e1[0], e1[1], e2[1]]]
                if np.any(add_face) and np.any(del_face): 
                    del_face = remove_entities(np.array(del_face), np.array(add_face)).tolist()

            elif cnnt_cnt1 >= 3:
                del_face += [[e2[0], e2[1], e1[0]], [e2[0], e2[1], e1[1]]]  #删除对角线2, 及其相邻两个面
            elif cnnt_cnt2 >= 3:  #边存在外部连通
                del_face += [[e1[0], e1[1], e2[0]], [e1[0], e1[1], e2[1]]]  #删除对角线1
            else: 
                del_face += [[e1[0], e1[1], e2[0]], [e1[0], e1[1], e2[1]]]

            if np.any(del_face): del_faces = np.append(del_faces, del_face, axis=0)
            if np.any(add_face): add_faces = np.append(add_faces, add_face, axis=0)

        #平面三角形
        elif vertices.size == 3: 
            del_face = [vertices]
            del_faces = np.append(del_faces, del_face, axis=0)

        #直线或点
        elif cov2d.vertices.size <= 2:
            warnings.warn("fix_tet failed: unexpected linear/point degenerate tet")
        else: 
            warnings.warn("fix_tet failed: unexpected degenerate tet")

    #添加/移除面 删除重复面
    new_faces = remove_entities(faces, del_faces)
    new_faces = np.append(new_faces, add_faces, axis=0)
    new_faces = new_faces[remove_dul_points(np.sort(new_faces, axis=1))[1]]
    new_verts = verts 
    return new_verts, new_faces

# def fix_tetrahedrons(faces, tets):
#     """按照边邻接数目修复四面体"""
#     #NOTE 在存在内表面的网格中, 一条边可以连接超过两个三角面. 不完全适用所有情况
#     from core.cyfunc.cy_geometry import compute_edge_adjacency

#     ef_map = compute_edge_adjacency(faces)
#     tets = np.sort(tets, axis=1)

#     #统计连通性
#     tet_cnnts = []
#     cnnt_cnts = []
#     for i in range(tets.shape[0]): 
#         cnnt = np.array([
#         ef_map[(tets[i, 0], tets[i, 1])], ef_map[(tets[i, 1], tets[i, 2])],  #从边查询邻接面
#         ef_map[(tets[i, 2], tets[i, 3])], ef_map[(tets[i, 0], tets[i, 2])], 
#         ef_map[(tets[i, 0], tets[i, 3])], ef_map[(tets[i, 1], tets[i, 3])]])  #面邻接三角形个数
#         tet_cnnts.append(cnnt)
#         cnnt_cnts.append(np.sum(cnnt != -1, axis=1))

#     #面处理
#     del_face = np.array([], dtype=np.int32)
#     for cnnt, cnnt_cnt in zip(tet_cnnts, cnnt_cnts):
#         n_inner = np.bincount(cnnt_cnt)[2]  #只有内部连接的边数目
#         #双内切平面 存在交叉面
#         if n_inner == 0:
#             continue
        
#         #内切平面
#         elif n_inner == 1: 
#             del_face = np.append(del_face, cnnt[cnnt_cnt==2].ravel())

#         #平面四边形
#         elif n_inner == 2: 
#             del_face = np.append(del_face, cnnt[cnnt_cnt==2][0].ravel())  #只移除一个对角

#         #一个点凸出的四面体 
#         elif n_inner == 3: 
#             del_face = np.append(del_face, cnnt[cnnt_cnt==2].ravel())
        
#         #fill hole填补为三角面接触 
#         elif n_inner == 4: 
#             del_face = np.append(del_face, cnnt[cnnt_cnt==2].ravel())
        
#         #一条边相连的孤立四面体
#         elif n_inner == 5: 
#             del_face = np.append(del_face, cnnt[cnnt_cnt==2].ravel())
        
#         #孤立退化四面体
#         elif n_inner == 6: 
#             del_face = np.append(del_face, cnnt[cnnt_cnt==2].ravel())

#     return faces[np.unique(del_face[del_face>-1])]


# def worker(vsub, vol, sigma=1.0, n_level=5):
#     """单个进程中执行 TetMesh.from_vol(vol, vsub)"""
#     try:
#         # 调用 TetMesh.from_vol(vol, vsub)
#         result = TetMesh.from_vol(vol, vsub, sigma=sigma, n_level=n_level)
#         # print(f"successed with vsub={vsub}")
#         return (vsub, result)  # 返回 vsub 和对应的 result
#     except RuntimeError as e:
#         # print(f"failed with vsub={vsub}")
#         return (vsub, str(e))  # 处理可能的异常并返回错误信息


class TetMesh(Mesh):
    cell_type = "tetra"

    # @classmethod
    # @auto_vol 
    # def from_vol_auto(cls, vol, vsubs=np.linspace(0.06, 0.1, 10, endpoint=False), sigma=1.0, n_level=np.inf, nmp=None):
    #     """并行地对 vsubs 列表中的每个 vsub 调用 TetMesh.from_vol"""
    #     import multiprocessing as mp
    #     if not nmp: nmp = np.size(vsubs)

    #     with mp.Pool(processes=nmp) as pool:
    #         # 使用 pool.map 并行执行每个 vsub 的测试
    #         results = pool.starmap(worker, [(vsub, vol, sigma, n_level) for vsub in vsubs])
        
    #     for result in results[::-1]: 
    #         if isinstance(result[1], cls): 
    #             print(f"{cls.__name__}.from_vol success with {result[0]}")
    #             return result[1]
        
    #     raise RuntimeError(f"{cls.__name__}.from_vol failed with {vsubs}")
    #     # return results

    @classmethod
    @auto_vol 
    def from_sur(cls, sur_mesh, vol=PhaseVol.ones((3, 3, 3))):
        """从表面网格剖分四面体网格
        examples: 
            smesh = TriMesh.from_vol(cell_vol, vsub=0.1)
            tverts, tcells = sur_tetgen(smesh.verts, smesh.cells)
            vert_attr = cell_vol.query_attr(tverts, smooth=cell_vol.attrs.size<8)
            cell_attr = vol.query_attr(np.average(tverts[tcells], axis=1), smooth=cell_vol.attrs.size<8)
            #np.savetxt("test.txt", np.sum(tcells < smesh.verts.shape[0], axis=1), fmt="%d")  #拥有边界点数量
            mask = np.sum(tcells < smesh.verts.shape[0], axis=1) == 3
        """
        from morqex.microstructure.core.cyfunc.cy_geometry import partition_domains

        #四面体化
        verts, cells, sur_cells = sur_tetgen(sur_mesh.verts, sur_mesh.cells, nobisect=True)  #不允许额外插入面
        #属性查询
        cell_attr = vol.query_attr(np.average(verts[cells], axis=1), smooth=vol.attrs.size<8)
        vert_attr = vol.query_attr(verts)

        #表面顶点属性为-1 
        vert_attr[sur_cells.ravel()] = -1 
        #查找不为-1的属性作为单元属性(假设非边界四面体的顶点属性相同)
        vcell_attr = np.max(vert_attr[cells], axis=1)
        cell_attr[vcell_attr >= 0] = vcell_attr[vcell_attr >= 0]

        #检查退化单元
        tet_mesh = cls(verts, cells, cell_data=cell_attr)
        quality = tet_mesh.skewness_quality
        quality_th = 0.01
        if np.any(quality < quality_th):
            #删除
            warnings.warn(f"TetMesh from_sur warn: {np.where(quality < quality_th)[0].size} low qualtiy cells detected, auto fix")
            mask = quality >= quality_th
            cells = cells[mask]
            cell_attr = cell_attr[mask]

        #按域分配属性
        # sur_cells = np.append(sur_cells, fill_holes(sur_cells), axis=0)
        domains = partition_domains(verts.astype(np.float32), cells, sur_cells)  #域划分
        # print([len(domain) for domain in domains])
        domain_data = [np_mode(cell_attr[np.array(domain)]) for domain in domains]  #域属性
        for domain, data in zip(domains, domain_data): cell_attr[np.array(domain)] = data  #分配单元属性
        tet_mesh = cls(verts, cells, cell_data=cell_attr)

        #检查属性分配
        mvf = tet_mesh.vol_frac
        tvf = np.sum([vol.vol_frac[attr] for attr in vol.vol_frac.keys()])   #vol的vf总体值包括0, 计算总体积排除0占的体积分数
        vvf = {attr: vol.vol_frac[attr]/tvf for attr in vol.vol_frac.keys()}  
        error = [vvf[attr] - mvf.get(attr, 0.0) for attr in vvf.keys()]

        #检查体积分数变化
        if np.sum(np.abs(error)) > 0.1 or np.any(np.array([mvf.get(attr, 0.0) for attr in vvf.keys()]) == 0):
            warnings.warn(f"TetMesh from_sur warn: volume fraction changed: {vol.vol_frac} -> {tet_mesh.vol_frac}")
        return tet_mesh

# face_adjs, tet_adjs = compute_face_adjacency(cells)
# smooth_tets = np.array([face_adjs[tuple(face)] 
#                         for face in np.sort(sur_cells, axis=1) if tuple(face) in face_adjs]).ravel()
# smooth_tets = smooth_tets[smooth_tets>=0]
# n_it = 1
# for i in range(n_it):
#     nbh_tets = tet_adjs[:, 1:][smooth_tets]
#     cell_attr_query = np.append(cell_attr, [-1])
#     nbh_attrs = cell_attr_query[nbh_tets]

#     modal_attr = mode_per_row(nbh_attrs)
#     cell_attr[smooth_tets][modal_attr >= 0] = modal_attr[modal_attr >= 0]
# TetMesh(verts, cells, cell_data=cell_attr).save_mesh("attr_test.vtk")

    @classmethod
    @auto_vol 
    def from_vol(cls, vol, vsub=0.1, sigma=1.0, n_level=np.inf, wfac=2.0):
        """从体素网格剖分四面体网格
        Args: 
            vol(PhaseVol): 多相体素网格
            vsub(float): 取0~1,表示 单位面积(球表面)的顶点数,1.0表示全部保留
            sigma(float): 权重平滑半径 1.0表示3x3范围内平滑
            n_level(int): 多级平滑的最大层级数
            prop(bool): 是否进行属性传播
        """
        #表面提取
        vol = PhaseVol(vol)
        sur_mesh = TriMesh.from_vol(vol, vsub=vsub, sigma=sigma, n_level=n_level, wfac=wfac)
        return TetMesh.from_sur(sur_mesh, vol)

    @property 
    def cell_volume(self):
        if not (self.n_cell and self.n_vert): return super().cell_volume
        verts = self.verts
        cells = self.cells 

        p1 = verts[cells[:, 0]]; p2 = verts[cells[:, 1]]; p3 = verts[cells[:, 2]]; p4 = verts[cells[:, 3]]
        x1 = p1[:, 0]; y1 = p1[:, 1]; z1 = p1[:, 2]; x2 = p2[:, 0]; y2 = p2[:, 1]; z2 = p2[:, 2]
        x3 = p3[:, 0]; y3 = p3[:, 1]; z3 = p3[:, 2]; x4 = p4[:, 0]; y4 = p4[:, 1]; z4 = p4[:, 2]

        A1 = x2 * (y3 * z4 - y4 * z3) - x3 * (y2 * z4 - y4 * z2) + x4 * (y2 * z3 - y3 * z2)
        A2 = x1 * (y3 * z4 - y4 * z3) - x3 * (y1 * z4 - y4 * z1) + x4 * (y1 * z3 - y3 * z1)
        A3 = x1 * (y2 * z4 - y4 * z2) - x2 * (y1 * z4 - y4 * z1) + x4 * (y1 * z2 - y2 * z1)
        A4 = x1 * (y2 * z3 - y3 * z2) - x2 * (y1 * z3 - y3 * z1) + x3 * (y1 * z2 - y2 * z1)
        volume = (A1 - A2 + A3 - A4) / 6
        return np.abs(volume)  #修复可能存在的顺序错误
        # return volume  #不修复 网格修正依赖体积正负值 

    @property
    def circum_radius(self):
        """
        计算四面体网格中每个四面体的外接圆半径
        输入: tets - (N, 4, 3)数组，N个四面体的顶点坐标
        输出: (N,)数组，每个四面体的外接圆半径
        """
        tets = self.verts[self.cells]

        # 提取四面体的四个顶点 (A, B, C, D)
        A = tets[:, 0, :]
        B = tets[:, 1, :]
        C = tets[:, 2, :]
        D = tets[:, 3, :]
        
        # 计算向量
        AB = B - A
        AC = C - A
        AD = D - A
        
        # 计算向量点积
        ab_ab = np.sum(AB * AB, axis=1)
        ac_ac = np.sum(AC * AC, axis=1)
        ad_ad = np.sum(AD * AD, axis=1)
        ab_ac = np.sum(AB * AC, axis=1)
        ab_ad = np.sum(AB * AD, axis=1)
        ac_ad = np.sum(AC * AD, axis=1)
        
        # 避免除以零 (退化四面体)
        volume = self.cell_volume
        epsilon = 1e-16
        volume = np.where(volume < epsilon, epsilon, volume)
        
        # 计算外接圆半径
        numerator = (
            ab_ab * (ac_ac * ad_ad - ac_ad**2) +
            ab_ac * (ab_ad * ac_ad - ac_ac * ab_ad) +
            ab_ad * (ab_ac * ac_ad - ad_ad * ab_ac)
        )
                
        denominator = volume**2
        rad = np.sqrt(np.abs(numerator) / denominator) / 2
        #BUG 
        return rad 
    
    @property
    def skewness_quality(self):
        """
        向量化计算四面体网格的偏度
        输入: tets - (N, 4, 3)数组，N个四面体的顶点坐标
        输出: (N,)数组，每个四面体的偏度值(0~1)
        """
        # 定义四面体的四个面 (每个面由三个顶点索引)
        faces = np.array([
            [0, 1, 2],  # 面0
            [1, 0, 3],  # 面1
            [0, 2, 3],  # 面2
            [1, 3, 2]   # 面3
        ])
        tets = self.verts[self.cells]
        
        N = tets.shape[0]
        
        # 计算所有面的法向量 (N, 4, 3)
        # 提取每个面的三个顶点 (N, 4, 3, 3) -> 然后重塑为 (4*N, 3, 3)
        face_verts = tets[:, faces].reshape(-1, 3, 3)
        
        # 计算每个面的两个边向量
        v1 = face_verts[:, 1] - face_verts[:, 0]  # (4*N, 3)
        v2 = face_verts[:, 2] - face_verts[:, 0]  # (4*N, 3)
        
        # 计算法向量 (叉积)
        normals = np.cross(v1, v2)  # (4*N, 3)
        
        # 计算法向量模长并归一化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms)
        
        # 重塑为 (N, 4, 3)
        normals = normals.reshape(N, 4, 3)
        
        # 定义6条边对应的面索引对
        edge_pairs = np.array([
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3], [2, 3]
        ])
        
        # 计算所有面对的法向量点积 (N, 6)
        n1 = normals[:, edge_pairs[:, 0]]  # (N, 6, 3)
        n2 = normals[:, edge_pairs[:, 1]]  # (N, 6, 3)
        dots = np.einsum('nij,nij->ni', n1, n2)  # (N, 6)
        
        # 确保点积在有效范围内并计算角度(度)
        clipped_dots = np.clip(dots, -1.0, 1.0)
        angles = np.degrees(np.arccos(clipped_dots))  # (N, 6)
        
        # 找到每个四面体的最小和最大角度
        min_angles = np.min(angles, axis=1)  # (N,)
        max_angles = np.max(angles, axis=1)  # (N,)
        
        # 理想四面体二面角 (arccos(1/3) ≈ 70.5288°)
        ideal_angle = np.degrees(np.arccos(1/3))
        
        # 计算偏度
        skewness_low = (ideal_angle - min_angles) / ideal_angle
        skewness_high = (max_angles - ideal_angle) / (180 - ideal_angle)
        
        # 组合并限制在[0, 1]范围内
        skewness = np.maximum(skewness_low, skewness_high)
        skewness = np.maximum(skewness, 0)  # 确保非负
        skewness = np.minimum(skewness, 1.0)  # 上限为1
        return 1 - skewness

    @property 
    def h(self):
        """
        计算四面体网格中所有四面体的最大高度
        
        参数:
        verts : numpy.ndarray, 形状为 (N, 3) 的数组, 表示N个三维顶点坐标
        cells : numpy.ndarray, 形状为 (M, 4) 的数组, 表示M个四面体的顶点索引
        
        返回:
        float: 整个网格中的最大高度
        """
        # 通过索引生成四面体顶点坐标 (形状: [M, 4, 3])
        verts = self.verts 
        cells = self.cells 
        tetras = verts[cells]
        
        # 提取四个顶点 A, B, C, D
        A = tetras[:, 0, :]
        B = tetras[:, 1, :]
        C = tetras[:, 2, :]
        D = tetras[:, 3, :]
        
        # 计算棱边向量
        AB = B - A
        AC = C - A
        AD = D - A
        BC = C - B
        BD = D - B
        
        # 计算各面叉乘
        cross_ABC = np.cross(AB, AC)
        cross_ABD = np.cross(AB, AD)
        cross_ACD = np.cross(AC, AD)
        cross_BCD = np.cross(BC, BD)
        
        # 计算各面面积
        S_ABC = 0.5 * np.linalg.norm(cross_ABC, axis=1)
        S_ABD = 0.5 * np.linalg.norm(cross_ABD, axis=1)
        S_ACD = 0.5 * np.linalg.norm(cross_ACD, axis=1)
        S_BCD = 0.5 * np.linalg.norm(cross_BCD, axis=1)
        
        # 计算体积 (通过标量三重积)
        triple_product = np.sum(AB * np.cross(AC, AD), axis=1)
        V = np.abs(triple_product) / 6.0
        
        # 计算四个高度
        hD = (3 * V) / S_ABC
        hC = (3 * V) / S_ABD
        hB = (3 * V) / S_ACD
        hA = (3 * V) / S_BCD
        
        # 找出每个四面体的最大高度
        heights = np.stack([hA, hB, hC, hD], axis=1)
        max_heights = np.max(heights, axis=1)
        
        # 返回全局最大高度
        return np.max(max_heights)

    @property
    def surf_mesh(self):
        """求表面网格,界面属性重新映射,表面重新旋转"""
        #按照属性划分不同子网格
        #只参与一个单元的顶点 为表面顶点 
        #属于同一个单元的顶点构成表面 计算表面积
        # 初始化一个字典，用于存储每个四面体的表面和相邻四面体的映射关系
        face_cell_dict = {}

        # 检查每个四面体的表面
        for i, cell in enumerate(self.cells):
            for j in range(cell.size):
                # 构建表面，这里我们将四个顶点的索引排序来确保一致性
                face = tuple(np.sort(cell[np.arange(cell.size) != j])) #面顶点索引
                
                # 如果表面不存在于字典中，创建一个新的键值对
                if face not in face_cell_dict:
                    face_cell_dict[face] = []
                
                # 将当前四面体的索引添加到表面的相邻四面体列表中
                face_cell_dict[face].append(i)

        # 输出表面-相邻单元表
        surfaces = [] 
        for face, adj_cells in face_cell_dict.items():
            if len(adj_cells) <= 1: 
                surfaces.append(face) 
        surfaces = np.array(surfaces)
        return TriMesh(self.verts, surfaces, self.vert_data, np.full(surfaces.shape[0], self.cell_data[0]))

    @property
    def surf_area(self):
        """四面体网格表面积计算"""
        return self.surf_mesh.surf_area


def attr_marching_cubes(attr_vol: PhaseVol, method="lorensen"):
    """多属性marching_cubes算法
    Marching cubes algorithm to find surfaces in 3d volumetric data.
    In contrast with Lorensen et al. approach [2], Lewiner et al. algorithm is faster, resolves ambiguities, 
    and guarantees topologically correct results. Therefore, this algorithm generally a better choice.
    """
    from skimage.measure import marching_cubes
    # attr_vol = PhaseVol(np.pad(attr_vol[:], 1))

    #声明数组
    attr_vol = PhaseVol(attr_vol)
    verts = np.ndarray((0,3), dtype=np.float32)
    faces = np.ndarray((0,3), dtype=np.int32)
    norms = np.ndarray((0,3), dtype=np.float32)
    face_attrs = np.array([], dtype=attr_vol.vol_dtype)
    
    #计算多相marching_cubes
    for attr in attr_vol.attrs:
        if attr == 0: continue 
        ret = marching_cubes(attr_vol[:]==attr, level=0.5, allow_degenerate=False, method=method)
        faces = np.append(faces, ret[1] + verts.shape[0], axis=0)
        verts = np.append(verts, ret[0], axis=0)
        norms = np.append(norms, ret[2], axis=0)
        face_attrs = np.append(face_attrs, np.full(ret[1].shape[0], attr))
    
    # #表面平滑
    # if sur_smooth:
    #     for i in range(3):
    #         verts[:, i][verts[:, i] < verts[:, i].min()+1e-3] += 0.5
    #         verts[:, i][verts[:, i] > verts[:, i].max()-1e-3] -= 0.5

    #重建不重叠顶点网格
    int_verts = (verts * 2).astype(np.int32)
    _, verts_idx, invs_idx, _ = remove_dul_points(int_verts) #新不重复顶点, 新顶点在原顶点内的索引, 原顶点的新索引
    ridx_faces = invs_idx[faces] #重建面的顶点索引 
    new_verts = verts[verts_idx]
    new_norms = norms[verts_idx]
    
    #删除重叠面 
    _, faces_idx, _, _ = remove_dul_points(np.sort(ridx_faces, axis=1))
    new_faces = ridx_faces[faces_idx]
    new_attrs = face_attrs[faces_idx]
    return new_verts, new_faces, new_norms, new_attrs  #还原顶点坐标



# ---------------------------------------------------------
# 主函数：提取多相体素网格表面
# ---------------------------------------------------------
def extract_surface_triangles(label):
    """
    label: (Nx,Ny,Nz) 多相体素标签，0 表示空位
    返回： vertices (M,3), faces (K,3)
    """
    # pad 以便处理边界
    padded = np.pad(label[:], 1, mode="constant", constant_values=0)

    # 6 个方向的邻居
    nbr = {
        '+x': padded[2:, 1:-1, 1:-1],
        '-x': padded[:-2, 1:-1, 1:-1],
        '+y': padded[1:-1, 2:, 1:-1],
        '-y': padded[1:-1, :-2, 1:-1],
        '+z': padded[1:-1, 1:-1, 2:],
        '-z': padded[1:-1, 1:-1, :-2],
    }

    # 当前体素
    center = padded[1:-1, 1:-1, 1:-1]

    # ---------------------------------------------------------
    # 条件：相 != 邻居相，且 (center != 0 或 neighbor != 0)
    # ---------------------------------------------------------
    interfaces = {}
    for key in nbr:
        A = center
        B = nbr[key]

        # 条件：两相不同 且 不是 0-外部
        valid = (A != B) & ((A != 0) | (B != 0))

        interfaces[key] = np.where(valid)

    # ---------------------------------------------------------
    # 顶点坐标生成（向量化）
    # ---------------------------------------------------------
    # 所有可能面的四个顶点偏移（cube faces）
    # 每个面由固定四个顶点组成，后续再三角化
    face_vertex_offsets = {
        '+x': np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]]),
        '-x': np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0]]),

        '+y': np.array([[0,1,0],[0,1,1],[1,1,1],[1,1,0]]),
        '-y': np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]]),

        '+z': np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
        '-z': np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]]),
    }

    vertices_list = []
    faces_list = []
    vertex_map = dict()
    vid = 0

    # ---------------------------------------------------------
    # 函数：获取或分配顶点 ID
    # ---------------------------------------------------------
    def get_vid(p):
        nonlocal vid
        p_tuple = tuple(p)
        if p_tuple not in vertex_map:
            vertex_map[p_tuple] = vid
            vertices_list.append(p)
            vid += 1
        return vertex_map[p_tuple]

    # ---------------------------------------------------------
    # 遍历 6 个方向的界面 (仍保持向量化，不遍历体素)
    # ---------------------------------------------------------
    for key, (ix, iy, iz) in interfaces.items():
        if ix.size == 0:
            continue

        # 选取此方向的面 4 顶点坐标（世界坐标）
        offs = face_vertex_offsets[key]  # (4,3)

        # 体素原点
        cell_origins = np.stack([ix, iy, iz], axis=1)  # (N,3)

        # 所有面 4 顶点 = cell + offsets
        pts = cell_origins[:, None, :] + offs[None, :, :]  # (N,4,3)

        # 为每个面 4 顶点分配 vertex id
        face_ids = np.zeros((len(pts), 4), dtype=int)
        for j in range(4):
            for i in range(len(pts)):
                face_ids[i, j] = get_vid(pts[i, j])

        # ---------------------------------------------------------
        # 每个 quad 分割成两个 triangle
        # ---------------------------------------------------------
        f1 = face_ids[:, [0,1,2]]
        f2 = face_ids[:, [0,2,3]]

        faces_list.append(f1)
        faces_list.append(f2)

    # ---------------------------------------------------------
    # 输出
    # ---------------------------------------------------------
    vertices = np.array(vertices_list, dtype=np.float32)
    faces = np.vstack(faces_list)
    verts = np.ascontiguousarray(vertices)
    faces = np.ascontiguousarray(faces)

    #重建不重叠顶点网格
    int_verts = (verts * 2).astype(np.int32)
    _, verts_idx, invs_idx, _ = remove_dul_points(int_verts) #新不重复顶点, 新顶点在原顶点内的索引, 原顶点的新索引
    ridx_faces = invs_idx[faces] #重建面的顶点索引 
    new_verts = verts[verts_idx]
    
    #删除重叠面 
    _, faces_idx, _, _ = remove_dul_points(np.sort(ridx_faces, axis=1))
    new_faces = ridx_faces[faces_idx]
    return new_verts, new_faces


# def attr_surface_extract(attr_vol):
#     #声明数组
#     attr_vol = PhaseVol(attr_vol)
#     verts = np.ndarray((0,3), dtype=np.float32)
#     faces = np.ndarray((0,3), dtype=np.int32)
#     face_attrs = np.array([], dtype=attr_vol.vol_dtype)
    
#     #计算多相marching_cubes
#     for attr in attr_vol.attrs:
#         if attr == 0: continue 
#         ret = surface_extract(attr_vol[:]==attr)
#         faces = np.append(faces, ret[1] + verts.shape[0], axis=0)
#         verts = np.append(verts, ret[0], axis=0)
#         face_attrs = np.append(face_attrs, np.full(ret[1].shape[0], attr))

#     #重建不重叠顶点网格
#     int_verts = (verts * 2).astype(np.int32)
#     _, verts_idx, invs_idx, _ = remove_dul_points(int_verts) #新不重复顶点, 新顶点在原顶点内的索引, 原顶点的新索引
#     ridx_faces = invs_idx[faces] #重建面的顶点索引 
#     new_verts = verts[verts_idx]
    
#     #删除重叠面 
#     _, faces_idx, _, _ = remove_dul_points(np.sort(ridx_faces, axis=1))
#     new_faces = ridx_faces[faces_idx]
#     new_attrs = face_attrs[faces_idx]
#     return new_verts, new_faces, new_attrs  #还原顶点坐标

# def surface_extract(vol):
#     direction = np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)])  #非结构化 
#     hex_loc = np.array(np.where(vol[:] != 0)).T     #六面体位置表 -1表示没有单元的位置 

#     #计算所有使用的顶点坐标 
#     pnts_loc = hex_loc.repeat(8, axis=0) + direction[np.newaxis, :, :].repeat(hex_loc.shape[0], axis=0).reshape((-1,3))
#     cells = np.arange(pnts_loc.shape[0]).reshape((-1,8))
#     faces = extract_faces_hex(cells)

#     #移除重复 得到共顶点网格
#     new_pnts, pnts_idx, invs_idx, _ = remove_dul_points(pnts_loc)
#     new_pnts = new_pnts.astype(np.float64)
#     ridx_faces = invs_idx[faces] #重建面的顶点索引 

#     #删除重叠面
#     _, faces_idx, _, face_cnt = remove_dul_points(np.sort(ridx_faces, axis=1))
#     new_faces = ridx_faces[faces_idx[face_cnt==1]]
#     pnts, quad_faces = remove_unuse_points(new_pnts, new_faces)
#     tri_faces = extract_faces_quad(quad_faces)
#     return pnts, tri_faces


# def attr_marching_cubes2d(attr_vol: PhaseVol):
#     """多属性marching_cubes算法(2d)"""
#     #BUG 面错位
#     #NOTE 在两相顶点交汇出出现面的重叠
#     from skimage.measure import marching_cubes
#     attr_vol = PhaseVol(attr_vol)

#     #声明数组
#     verts = np.ndarray((0,3), dtype=np.float32)
#     faces = np.ndarray((0,3), dtype=np.int32)
#     face_attrs = np.array([], dtype=attr_vol.vol_dtype)
    
#     #计算多相marching_cubes
#     for attr in attr_vol.attrs:
#         if attr == 0: continue 
#         ret = marching_cubes(attr_vol[:]==attr, allow_degenerate=False)
#         faces = np.append(faces, ret[1] + verts.shape[0], axis=0)
#         verts = np.append(verts, ret[0], axis=0)
#         face_attrs = np.append(face_attrs, np.full(ret[1].shape[0], attr))
    
#     #重建不重叠顶点网格
#     mask = verts[:, 2] < np.max(verts[:, 2])
#     new_verts = verts[mask]
#     new_verts[:, 2] = np.min(new_verts[:, 2])

#     invs_idx = np.full(verts.shape[0], -1, dtype=np.int32)
#     invs_idx[mask] = np.arange(new_verts.shape[0])

#     face_mask = np.all(mask[faces], axis=1)
#     new_faces = faces[face_mask] 
#     new_attrs = face_attrs[face_mask]
#     new_faces = invs_idx[new_faces]
#     return new_verts, new_faces, np.repeat([0, 0, 1.0], new_verts.size, axis=0), new_attrs  #还原顶点坐标


# def merge_vertices(vertices, faces, face_mask, epsilon):
#     face_mask = face_mask.astype(bool)
#     selected_faces = faces[face_mask]
    
#     if len(selected_faces) == 0:
#         return vertices.copy(), faces.copy()
    
#     # 获取选中面的所有顶点
#     v_indices_selected = np.unique(selected_faces.ravel())
#     if len(v_indices_selected) == 0:
#         return vertices.copy(), faces.copy()
    
#     # 生成边集合（自动去重）
#     edges = selected_faces[:, [[0,1], [1,2], [2,0]]].reshape(-1, 2)
#     sorted_edges = np.sort(edges, axis=1)
#     unique_edges = np.unique(sorted_edges, axis=0)
    
#     # 过滤仅包含选中顶点的边
#     mask = np.isin(unique_edges[:, 0], v_indices_selected) & \
#            np.isin(unique_edges[:, 1], v_indices_selected)
#     valid_edges = unique_edges[mask]
    
#     # 建立全局到本地顶点索引的映射
#     v_global_to_local = {v: i for i, v in enumerate(v_indices_selected)}
#     V = vertices[v_indices_selected]
    
#     # 转换边到本地索引并计算距离
#     merge_pairs = []
#     epsilon_sq = epsilon ** 2
#     for (v1, v2) in valid_edges:
#         i, j = v_global_to_local[v1], v_global_to_local[v2]
#         if np.sum((V[i] - V[j])**2) < epsilon_sq:
#             merge_pairs.append((i, j) if i < j else (j, i))
    
#     # 并查集合并
#     parent = np.arange(len(V))
#     def find(u):
#         while parent[u] != u:
#             parent[u] = parent[parent[u]]
#             u = parent[u]
#         return u
    
#     for i, j in merge_pairs:
#         root_i = find(i)
#         root_j = find(j)
#         if root_i != root_j:
#             parent[root_j] = root_i if root_i < root_j else root_j
    
#     # 压缩路径
#     for i in range(len(parent)):
#         find(i)
    
#     # 生成新顶点
#     unique_roots = np.unique(parent)
#     root_centers = np.array([V[parent == r].mean(0) for r in unique_roots])
    
#     # 合并未选择的顶点和新顶点
#     unselected_mask = ~np.isin(np.arange(len(vertices)), v_indices_selected)
#     new_vertices = np.vstack([vertices[unselected_mask], root_centers])
    
#     # 构建顶点映射表
#     global_to_new = {}
#     new_idx = np.sum(unselected_mask)
#     for i, v in enumerate(v_indices_selected):
#         global_to_new[v] = new_idx + np.where(unique_roots == parent[i])[0][0]
    
#     # 更新面索引
#     new_faces = np.copy(faces)
#     for i in range(faces.shape[0]):
#         for j in range(3):
#             v = faces[i,j]
#             if v in global_to_new:
#                 new_faces[i,j] = global_to_new[v]
#             else:
#                 new_faces[i,j] = np.where(unselected_mask)[0].searchsorted(v)
    
#     return new_vertices.astype(np.float32), new_faces.astype(np.int32)


# def dual_contour(vol, verts, norms):
#     """表面网格重采样 Voronoi近似"""
#     #TODO 通过求marching_cubes对偶得到结果
#     from scipy.spatial import Voronoi
#     # 计算顶点位置 合并空间顶点
#     points = np.concatenate([verts + norms*0.5, verts - norms*0.5], axis=0)  #顶点沿法相复制
#     voro = Voronoi(points)
#     ndim = voro.points.shape[1]

#     #顶点-边表
#     vr_list = [[] for _ in range(voro.vertices.shape[0]+1)]  #额外留出存放无边界点
#     for rdg_i, vert_idxs in enumerate(voro.ridge_vertices):  #边-顶点表 一条边可以有奇数个顶点
#         ridge_verts = voro.vertices[np.array(vert_idxs)]
#         if (np.max(ridge_verts, axis=0) - np.min(ridge_verts, axis=0)).max() < 2: #边长限制
#             for vert_i in vert_idxs:    
#                 vr_list[vert_i].append(rdg_i)

#     #顶点掩膜
#     v_mask = []
#     for vrt_rdg in vr_list[:-1]:
#         if np.any(vrt_rdg):
#             vrt_pnt = voro.ridge_points[np.array(vrt_rdg)] #边-中心表  -> 顶点-中心表 
#             in_mark = vrt_pnt < verts.shape[0]
#             v_mask.append(~np.all(in_mark) and np.any(in_mark))  #不全为内部/外部 
#         else:
#             v_mask.append(False) 
#     v_mask = np.array(v_mask, dtype=np.bool_)
#     locs = (voro.vertices).astype(np.int32)[v_mask]    #所在单元的索引
#     vertices = voro.vertices[v_mask]

#     #网格内坐标合并
#     vert_vol = np.zeros(np.append(np.array(vol.shape)-1, np.array(vol.shape).size))
#     cnt_vol = np.zeros(np.append(np.array(vol.shape)-1, 1))
#     for loc, vert in zip(locs, vertices):
#         vert_vol[tuple(loc)] += vert   #顶点坐标求和
#         cnt_vol[tuple(loc)] += 1  #单元内顶点数目
#     vert_vol /= cnt_vol

#     #坐标网格表面提取
#     data_axis = tuple(np.arange(ndim, ndim*2+1, dtype=np.int32))
#     #xy平面
#     xy_view = np.lib.stride_tricks.sliding_window_view(vert_vol, (2,2,1,1))
#     xy_view = xy_view.reshape(xy_view.shape[:-1]) #删去数据维度
#     xyf_mask = np.all(~np.isnan(xy_view), data_axis)  #面匹配索引
#     xy_faces = xy_view[xyf_mask]  #xy方向面的匹配点坐标 (平面数, 存储的xyz坐标, 匹配面形状)
#     xy_faces = np.swapaxes(xy_faces[:, :, (0, 1, 1, 0), (0, 0, 1, 1), (0, 0, 0, 0)], 1, 2)  #面顶点排序 + 维度转化-> (平面数, 面顶点, 顶点坐标)
    
#     #yz平面
#     yz_view = np.lib.stride_tricks.sliding_window_view(vert_vol, (1,2,2,1))
#     yz_view = yz_view.reshape(yz_view.shape[:-1]) #删去数据维度
#     yzf_mask = np.all(~np.isnan(yz_view), data_axis)  #面匹配索引
#     yz_faces = yz_view[yzf_mask]  #xy方向面的匹配点坐标 (平面数, 存储的xyz坐标, 匹配面形状)
#     yz_faces = np.swapaxes(yz_faces[:, :, (0, 0, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1)], 1, 2)
    
#     #zx平面
#     zx_view = np.lib.stride_tricks.sliding_window_view(vert_vol, (2,1,2,1))
#     zx_view = zx_view.reshape(zx_view.shape[:-1]) #删去数据维度
#     zxf_mask = np.all(~np.isnan(zx_view), data_axis)  #面匹配索引
#     zx_faces = zx_view[zxf_mask]  #xy方向面的匹配点坐标 (平面数, 存储的xyz坐标, 匹配面形状)
#     zx_faces = np.swapaxes(zx_faces[:, :, (0, 0, 1, 1), (0, 0, 0, 0), (0, 1, 1, 0)], 1, 2) 

#     #移除xyz方向扫描的重复顶点,合并得到网格
#     pnts_loc = np.concatenate([xy_faces, yz_faces, zx_faces], axis=0).reshape(-1, ndim)
#     cells = np.arange(pnts_loc.shape[0]).reshape(-1, 4)

#     new_pnts, _, invs_idx, _ = remove_dul_points(pnts_loc)
#     new_pnts = new_pnts.astype(np.float64)
#     new_cells = invs_idx[cells]
#     faces = np.concatenate([new_cells[:, (0,1,2)], new_cells[:, (0,2,3)]], axis=0)  #建立三角网格
#     return np.ascontiguousarray(new_pnts), np.ascontiguousarray(faces)
    

def sur_optimize(verts, faces, nvrt, weights=np.array([])):
    """加权表面粗化
    example: 
        #聚簇可视化
        from morqex.microstructure.mesh import pyacvd

        #多相表面提取
        npad = min(int(np.log2(PhaseVol(tess_vol).vxl_len)-2.0), np.inf)
        clr_vol = PhaseVol(np.pad(tess_vol[:], npad)).label_vol().color_vol()
        mc_verts, mc_faces, _, _ = attr_marching_cubes(clr_vol, method="lorensen")  #lorensen不连接对角
        mc_verts += 0.5           #边界坐标从0.0开始, 体素中心0.5, 1.5...
        mc_verts /= clr_vol.vxl_len   #转化为相对坐标

        #计算权重
        wgt_vol = TriMesh.volres_weight(clr_vol, sigma=1.0, n_level=np.inf, wfac=2.0)  #TODO 使用color_vol计算权重

        #表面优化
        wgts = wgt_vol.itp_query(mc_verts)  #属性在体素中心, 查询坐标处的最大分量属性值
        nvrt = int(np.sum(wgts) * 0.2)
        clus_mesh = pyacvd.Clustering(mc_verts, mc_faces, wgts)
        clus_mesh.cluster(int(nvrt))  #设置顶点数目
        cvd_verts, cvd_faces = clus_mesh.create_mesh()
        rnd = np.arange(clus_mesh.clusters.max()+1)
        np.random.shuffle(rnd)
        TriMesh(clus_mesh.points, clus_mesh.faces, vert_data=rnd[clus_mesh.clusters]).save_mesh("test.vtk")
        TriMesh(cvd_verts, cvd_faces).save_mesh("test1.vtk")
        TriMesh(mc_verts, mc_faces).save_mesh("test2.vtk")
    """
    from morqex.microstructure.mesh import pyacvd
    
    if nvrt < 4 or nvrt > verts.shape[0]:
        warnings.warn(f"nvrt should falls in 4~{verts.shape[0]}, having {nvrt}, auto fixed")
        nvrt = max(4, nvrt)
        nvrt = min(verts.shape[0]-1, nvrt)

    clus_mesh = pyacvd.Clustering(verts, faces, weights)
    clus_mesh.cluster(int(nvrt))  #设置顶点数目
    cvd_verts, cvd_faces = clus_mesh.create_mesh()
    return cvd_verts, cvd_faces


def sur_tetgen(verts, faces, order=1, mindihedral=15, minratio=2.0, nobisect=True):
    """四面体化"""
    from morqex.microstructure.mesh import tetgen 

    try: 
        sur_faces = faces
        tgen = tetgen.TetGen(verts, sur_faces)
        tet_verts, tet_cells = tgen.tetrahedralize(order=order, mindihedral=mindihedral, 
        minratio=minratio, nobisect=nobisect) 
    
    except:
        #移除skip_faces后重试
        invld_faces = np.loadtxt("_skipped.face", dtype=np.int32, skiprows=1).reshape(-1, 5)[:, 1:4]
        warnings.warn(f"Tetgen warning: {invld_faces.shape[0]} invalid faces detected, trying again")

        sur_faces = remove_entities(sur_faces, invld_faces)
        tgen = tetgen.TetGen(verts, sur_faces)
        tet_verts, tet_cells = tgen.tetrahedralize(order=order, mindihedral=mindihedral, 
        minratio=minratio, nobisect=nobisect) 

        TriMesh(verts, invld_faces).save_mesh("tetgen_debug.vtk")

    if os.path.isfile("_skipped.face"): os.remove("_skipped.face")
    if os.path.isfile("_skipped.node"): os.remove("_skipped.node")
    return tet_verts, tet_cells, sur_faces

# def compute_tetrahedron_centers(tetrahedra, vertices):
#     """
#     计算每个四面体的中心坐标
#     :param tetrahedra: 四面体顶点索引, shape=(N_tetrahedra, 4)
#     :param vertices: 顶点坐标, shape=(N_vertices, 3)
#     :return: 四面体的中心坐标, shape=(N_tetrahedra, 3)
#     """
#     # 获取四面体四个顶点的坐标
#     tetra_vertices = vertices[tetrahedra]  # shape = (N_tetrahedra, 4, 3)
#     # 计算四面体中心，取四个顶点坐标的平均值
#     tetra_centers = np.mean(tetra_vertices, axis=1)
#     return tetra_centers

# def propagate_attributes_to_tetrahedra(tetra_centers, centers, center_attributes):
#     """
#     通过距离传播属性到每个四面体单元
#     :param tetra_centers: 四面体单元中心坐标, shape=(N_tetrahedra, 3)
#     :param centers: 区域中心坐标, shape=(N_regions, 3)
#     :param center_attributes: 区域中心的属性, shape=(N_regions,)
#     :return: 每个四面体单元的属性, shape=(N_tetrahedra,)
#     """
#     # 计算每个四面体中心到所有区域中心的距离，得到距离矩阵，shape=(N_tetrahedra, N_regions)
#     distances = np.linalg.norm(tetra_centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1)
    
#     # 找到每个四面体单元最近的区域中心
#     closest_region_idx = np.argmin(distances, axis=1)
    
#     # 根据最近区域的索引赋值属性
#     tetra_attributes = center_attributes[closest_region_idx]
#     return tetra_attributes

# def attr_prop(cells, edges, cell_attr):
#     """相邻单元属性传播, edges指定阻断传播的边界 要求tetgen不分割边界
#     Example:
#         verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
#         cells = np.array([[0,1,3,4],[1,2,3,6],[1,3,4,6]]) 
#         edges = np.array([[1,3,6]])
#         cell_attr = np.array([1,1,2])
#     """
#     #NOTE cvd的结果可能不封闭,导致属性传播出现错误
#     # cell_attr = attr_prop(cells, sur_mesh.cells, cell_attr)
#     import networkx as nx 

#     #建立单元-相邻单元表
#     n_cell = cells.shape[0]
#     n_face = n_cell * 4 
#     n_edge = edges.shape[0] 

#     cfidx = np.array([[i for i in range(4) if i != j] for j in range(4)])
#     cell_faces = np.sort(cells[:, cfidx].reshape(-1, 3), axis=1)
#     edge_faces = np.sort(edges, axis=1)
#     _, idx, inv, _ = remove_dul_points(np.concatenate([edge_faces, cell_faces], axis=0))  #引入cvd表面

#     #idx[inv] #单元面指向重复面的索引(可能是自身,可能是共用面,边界面只有自身一个)
#     #NOTE 可能存在一些cvd_face(edges)在网格cells中删除,作为独立的不发生重复的边界
#     uni_fidx = idx[inv] - n_edge
#     face_nbh = uni_fidx[-n_face:]//4 #单元面指向相邻单元
#     face_slf = np.repeat(np.arange(n_cell), 4) #单元面指向自身单元 单元-相邻单元
#     nbh_cell = np.concatenate([face_slf[-n_face:, None], face_nbh[-n_face:, None]], axis=1)

#     #检查边界面 指向cvd的重复面(边界面)移除 
#     nbh_fidx = nbh_cell[(uni_fidx >= 0)[-n_face:]]  
#     # #建立单元-相邻单元表
#     # n_cell = cells.shape[0]
#     # n_edge = edges.shape[0] 
#     # fncell = face_ncell(cells)

#     # #检查边界面 指向cvd的重复面(边界面)移除 
#     # cfidx = np.array([[i for i in range(4) if i != j] for j in range(4)])
#     # cell_faces = np.sort(cells[:, cfidx].reshape(-1, 3), axis=1)
#     # edge_faces = np.sort(edges, axis=1)
#     # _, idx, inv = remove_dul_points(np.concatenate([edge_faces, cell_faces], 0))
#     # nbh_fidx = fncell[((idx[inv] - n_edge) >= 0)[-n_cell*4:]]

#     #使用图在连通分量内传播属性
#     #NOTE reg_cell中的索引是否和cell对应, 可能存在完全独立,全边界的单个cell没有加入reg_cell
#     prop_cell_attr = cell_attr.copy()
#     cell_graph = nx.Graph()
#     cell_graph.add_edges_from(nbh_fidx)
#     for reg in nx.connected_components(cell_graph):
#         reg_cell = np.array(list(reg))
#         uni, cnt = np.unique(cell_attr[reg_cell], return_counts=True)
#         reg_attr = uni[np.argmax(cnt)]
#         prop_cell_attr[reg_cell] = reg_attr
        
#     #返回更新的属性表
#     return prop_cell_attr 
    
def remove_dul_points(vertices):
    """移除重复点,输入整数点坐标
    Examples:
        pnts = np.array([[1,2,3],[4,2,1],[2,4,5],[3,4,5],[1,2,3]])
        new_pnts, pnts_idx, invs_idx, _ = remove_dul_points(pnts)
    """
    vertices = np.array(vertices)
    vertices_flat = np.ravel(vertices.view([('', vertices.dtype)]*vertices.shape[1]))
    _, verts_idx, invs_idx, counts = np.unique(vertices_flat, return_index=True, return_inverse=True, return_counts=True)
    return vertices[verts_idx], verts_idx, invs_idx, counts

# def remove_points(vertices, to_remove):
#     """
#     从顶点列表中移除指定的顶点集合（支持任意维度）
    
#     参数：
#         vertices   : 原始顶点数组，形状为(M, n) 
#         to_remove  : 需要移除的顶点数组，形状为(K, n)
    
#     返回：
#         过滤后的顶点数组，形状为(P, n)
#     """
#     vertices = np.asarray(vertices)
#     to_remove = np.asarray(to_remove)

#     # 处理空输入
#     if vertices.size == 0:
#         return np.empty((0, vertices.shape[1]), dtype=vertices.dtype)
#     if to_remove.size == 0:
#         return vertices.copy()
    
#     # 验证维度一致性
#     if vertices.shape[1] != to_remove.shape[1]:
#         raise ValueError(f"维度不匹配: vertices是{vertices.shape[1]}D，to_remove是{to_remove.shape[1]}D")

#     # 动态创建结构化dtype
#     n_dim = vertices.shape[1]
#     dtype = np.dtype([(f'f{i}', vertices.dtype) for i in range(n_dim)])

#     # 转换为连续的结构化数组
#     v_cont = np.ascontiguousarray(vertices).view(dtype).reshape(-1)
#     r_cont = np.ascontiguousarray(to_remove).view(dtype).reshape(-1)

#     # 生成过滤掩膜
#     mask = np.isin(v_cont, r_cont)
    
#     # 保留未匹配的顶点
#     return vertices[~mask], np.where(~mask)

def remove_unuse_points(vertices, faces):
    # 获取所有被面引用的顶点索引
    old_indices = faces.ravel()
    unique_old_indices = np.unique(old_indices)
    
    # 创建新的顶点数组
    new_vertices = vertices[unique_old_indices]
    
    # 构建旧索引到新索引的映射
    index_map = np.full(vertices.shape[0], -1, dtype=int)
    index_map[unique_old_indices] = np.arange(len(unique_old_indices))
    
    # 重新映射面中的顶点索引
    new_faces = index_map[faces]
    return new_vertices, new_faces

def remove_dul_points_xyz(vertices):
    inv_vertices = np.ascontiguousarray(vertices[:, ::-1])
    inv_unique_verts, verts_idx, invs_idx, counts = remove_dul_points(inv_vertices)
    return inv_unique_verts[:, ::-1], verts_idx, invs_idx, counts

def face_ncell(cells):
    """建立单元-相邻单元表"""
    n_cell = cells.shape[0]

    cfidx = np.array([[i for i in range(4) if i != j] for j in range(4)])
    cell_faces = np.sort(cells[:, cfidx].reshape(-1, 3), axis=1)
    _, idx, inv, _ = remove_dul_points(cell_faces) 

    #idx[inv] #单元面指向重复面的索引(可能是自身,可能是共用面,边界面只有自身一个)
    uni_fidx = idx[inv] 
    face_nbh = uni_fidx // 4 #单元面指向相邻单元,或自身,或边界
    face_slf = np.repeat(np.arange(n_cell), 4) #单元面指向自身单元 单元-相邻单元
    nbh_cell = np.concatenate([face_slf[:, None], face_nbh[:, None]], axis=1)
    return nbh_cell 

def _cal_2d_projection(points):
    """投影退化四面体到二维平面"""
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]

    #计算投影基向量
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # 单位化
    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    #坐标映射
    point_2d = [(np.dot(points[i]-points[0], u), 
                 np.dot(points[i]-points[0], v)) for i in range(points.shape[0])]
    return np.array(point_2d), normal
    
def edge_length(verts, tets):
    """
    计算单元的边长
    :param verts: 顶点坐标
    :param tets: 单元连接关系
    :return: 边长
    """
    edges = np.zeros((tets.shape[0], 6))
    for i, pair in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
        edges[:, i] = np.linalg.norm(verts[tets[:, pair[0]], :] - verts[tets[:, pair[1]], :], axis=1)
    return edges


class QuadMesh(Mesh):
    cell_type = "quad"

    @classmethod
    def from_vol_2d(cls, attr_vol: PhaseVol):
        vol = attr_vol[:]
        direction = np.array([(0,0,0),(1,0,0),(1,1,0),(0,0,0),(1,1,0),(0,1,0)])
        tri_loc = np.array(np.where(vol[:] != 0)).T     #六面体位置表 -1表示没有单元的位置 

        #计算所有使用的顶点坐标 
        pnts_loc = tri_loc.repeat(6, axis=0) + direction[np.newaxis, :, :].repeat(tri_loc.shape[0], axis=0).reshape((-1,3))
        cells = np.arange(pnts_loc.shape[0]).reshape((-1,3))

        #移除重复 得到共顶点网格
        new_pnts, _, invs_idx, _ = remove_dul_points(pnts_loc)
        new_pnts = new_pnts.astype(np.float64)
        new_pnts /= np.max(vol.shape)  #归一化坐标
        new_cells = invs_idx[cells]

        #六面体属性表查询
        cell_attr = vol[:][tri_loc.T[0].repeat(2), tri_loc.T[1].repeat(2), tri_loc.T[2].repeat(2)]
        return cls(new_pnts, new_cells, np.repeat([[0, 0, 1.0]], new_pnts.shape(0), axis=0), cell_attr)  #还原顶点坐标

    @classmethod
    def from_vol(cls, vol: PhaseVol):
        #网格重新编号 注意结构化和非结构化vtk格式不同
        direction = np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)])  #非结构化 
        hex_loc = np.array(np.where(vol[:] != 0)).T     #六面体位置表 -1表示没有单元的位置 

        #计算所有使用的顶点坐标 
        pnts_loc = hex_loc.repeat(8, axis=0) + direction[np.newaxis, :, :].repeat(hex_loc.shape[0], axis=0).reshape((-1,3))
        cells = np.arange(pnts_loc.shape[0]).reshape((-1,8))
        faces = extract_faces_hex(cells)

        #移除重复 得到共顶点网格
        new_pnts, _, invs_idx, _ = remove_dul_points(pnts_loc)
        new_pnts = new_pnts.astype(np.float64)
        ridx_faces = invs_idx[faces] #重建面的顶点索引 

        #删除重叠面
        _, faces_idx, _, face_cnt = remove_dul_points(np.sort(ridx_faces, axis=1))
        new_faces = ridx_faces[faces_idx[face_cnt==1]]
        return cls(remove_unuse_points(new_pnts, new_faces))

    
class HexMesh(Mesh):
    cell_type = "hexahedron"

    def __init__(self, verts, cells, vert_attr, cell_attr):
        super().__init__(verts, cells, vert_attr, cell_attr)

    @classmethod
    def from_vol(cls, vol: PhaseVol):
        """将体素网格转化为六面体网格"""
        #网格重新编号 注意结构化和非结构化vtk格式不同
        # direction = ((0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1))  #结构化 
        direction = np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)])  #非结构化 
        hex_loc = np.array(np.where(vol[:] != 0)).T     #六面体位置表 -1表示没有单元的位置 

        #计算所有使用的顶点坐标 
        pnts_loc = hex_loc.repeat(8, axis=0) + direction[np.newaxis, :, :].repeat(hex_loc.shape[0], axis=0).reshape((-1,3))
        cells = np.arange(pnts_loc.shape[0]).reshape((-1,8))

        #移除重复 得到共顶点网格
        new_pnts, pnts_idx, invs_idx, _ = remove_dul_points(pnts_loc)
        new_pnts = new_pnts.astype(np.float64)
        new_pnts /= np.max(vol.shape)  #归一化坐标
        new_cells = invs_idx[cells]

        #六面体属性表查询
        cell_attr = vol[:][hex_loc.T[0], hex_loc.T[1], hex_loc.T[2]]  
        vert_attr = np.zeros(new_pnts.shape[0], vol[:].dtype) 
        return cls(new_pnts, new_cells, vert_attr, cell_attr)
    
    @property 
    def cell_volume(self):
        if not self.n_cell and self.n_vert: return 0 
        return np.sum(np.abs(self.verts[self.cells[:, 0]] - self.verts[self.cells[:, 1]]), axis=1)**3 

    def vol_from_tet(cell_verts, cells, cell_attrs, voxel_l, stride=1):
        """将四面体网格转化为体素网格"""
        #TODO 加速和debug
        #确定边长
        x_len = np.max(cell_verts[:,0]) - np.min(cell_verts[:,0])
        y_len = np.max(cell_verts[:,1]) - np.min(cell_verts[:,1])
        z_len = np.max(cell_verts[:,2]) - np.min(cell_verts[:,2])
        zero_th = voxel_l*0.01  #0范围阈值

        #创建体素网格 
        x_size = int(x_len/(voxel_l*stride) + 0.5) #超过中心则需要考虑该位置 
        y_size = int(y_len/(voxel_l*stride) + 0.5) 
        z_size = int(z_len/(voxel_l*stride) + 0.5) 

        #属性体素网格
        vol_attr = np.zeros((x_size, y_size, z_size), dtype=np.int32) - 1
        for cell_i, cell in enumerate(cells):
            verts_pos = np.array(cell_verts[cell])  #四个顶点的坐标 
            pack_hex = np.zeros((2, cell_verts.shape[1]))
            for i in range(cell_verts.shape[1]):
                pack_hex[0,i] = np.min(verts_pos[:,i]) 
                pack_hex[1,i] = np.max(verts_pos[:,i]) #顶点的最小封装长方体
            pack_idx = np.array(pack_hex/voxel_l + 0.5).astype(np.uint32) #封装长方体对角的体素坐标 直接取整(包含0.5则体素中心在内)
            #每个单元默认下限在区域内,上限在区域外 若上下相等则不计算当前单元 
            #从下界到上界 区域内所有点
            grid = np.meshgrid(range(pack_idx[0,0],pack_idx[1,0]),
                            range(pack_idx[0,1],pack_idx[1,1]),
                            range(pack_idx[0,2],pack_idx[1,2]))  #x,y,z对应坐标
            if not grid[0].size:  #当前四面体不包含任何体素中心 
                continue 
            pnts = (np.array([grid[0].reshape(-1,), grid[1].reshape(-1,), grid[2].reshape(-1,)]).T + 0.5)*voxel_l  #转化为体素中心坐标
            #计算区域所有点是否在所有面内 
            #对四个顶点计算相对位置向量 
            vert_vec0 = verts_pos[0] - pnts
            vert_vec1 = verts_pos[1] - pnts
            vert_vec2 = verts_pos[2] - pnts
            vert_vec3 = verts_pos[3] - pnts
            #顶点是否在面内的表
            pnts_in_face031 = np.sum(np.cross(vert_vec0, vert_vec3)*vert_vec1, axis=1) <= zero_th  #为了处理在面内的情况 
            pnts_in_face120 = np.sum(np.cross(vert_vec1, vert_vec2)*vert_vec0, axis=1) <= zero_th
            pnts_in_face213 = np.sum(np.cross(vert_vec2, vert_vec1)*vert_vec3, axis=1) <= zero_th
            pnts_in_face302 = np.sum(np.cross(vert_vec3, vert_vec0)*vert_vec2, axis=1) <= zero_th
            #求交 要求四次结果全部为真 则在四面体内 
            pts_in_faces0 = np.bitwise_and(pnts_in_face120, pnts_in_face213)
            pts_in_faces1 = np.bitwise_and(pnts_in_face302, pnts_in_face031)
            #再次求交 并且将结果还原为对应的点
            pts_in_hex = np.bitwise_and(pts_in_faces0, pts_in_faces1)
            #在体素内的点进行属性赋值 
            pnts_idx = (pnts/voxel_l).astype(np.uint32)[np.where(pts_in_hex == True)]
            vol_attr[tuple(pnts_idx.T)] = cell_attrs[cell_i]
        return vol_attr

class HsHexMesh(HexMesh):
    cell_type = "hexahedron"
    suffix = "dat"

    def __init__(self, verts, cells, vert_data, cell_data, face_mark=np.array([]), cell_mark=np.array([])):
        self.face_mark = face_mark 
        self.cell_mark = cell_mark
        super().__init__(verts, cells, vert_data, cell_data)

    @classmethod
    def from_vol(cls, vol: PhaseVol):
        """将体素网格转化为六面体网格"""
        #网格重新编号 注意结构化和非结构化vtk格式不同
        # direction = ((0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1))  #结构化 
        direction = np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)])  #非结构化 
        hex_loc = np.array(where_xyz(vol[:] != 0)).T     #六面体位置表 -1表示没有单元的位置 

        #计算所有使用的顶点坐标 
        pnts_loc = hex_loc.repeat(8, axis=0) + direction[np.newaxis, :, :].repeat(hex_loc.shape[0], axis=0).reshape((-1,3))
        cells = np.arange(pnts_loc.shape[0]).reshape((-1,8))
        faces = extract_faces_hex(cells)

        #移除重复 得到共顶点网格
        new_pnts, pnts_idx, invs_idx, _ = remove_dul_points_xyz(pnts_loc)
        new_pnts = new_pnts.astype(np.float64)
        new_pnts /= np.max(vol.shape)  #归一化坐标
        new_cells = invs_idx[cells]

        #六面体属性表查询
        cell_attr = vol[:][hex_loc.T[0], hex_loc.T[1], hex_loc.T[2]]
        vert_attr = np.zeros(new_pnts.shape[0], vol[:].dtype) 
        return cls(new_pnts, new_cells, vert_attr, cell_attr)
    
    @property 
    def cell_volume(self):
        if not self.n_cell and self.n_vert: return 0 
        return np.sum(np.abs(self.verts[self.cells[:, 0]] - self.verts[self.cells[:, 1]]), axis=1)**3 

    def add_axis_bound(self, axis, mark=(0, 0), cell_relative_th = 0.6, face_relative_th = 0.1):
        """为体积网格添加坐标轴指定的边界,mark指示上下界的标记,0表示不为边界 要求mark的值大于0. 默认单元顺序zyx. 
        Examples:
            vol = PhaseVol.ones((3,3,3))
            hmesh = HsHexMesh.from_vol(vol)
            hmesh.add_axis_bound(2, (1, 2))

            hmesh.verts
            np.savetxt("tmp.txt", hmesh.verts)
            hmesh.faces[:5]
        """
        if self.n_cell <= 0:
            raise RuntimeError("add axis bound failed: mesh of mesh is empty")
        #交换xz:
        # axis = 2 - axis

        #计算边界坐标阈值 
        cell_centers = self.cell_centers
        voxel_l = 2 * np.min(cell_centers)  #最小坐标
        mesh_l = np.max(cell_centers, axis=0) + voxel_l/2

        cell_low_bound = voxel_l * cell_relative_th
        cell_up_bound = mesh_l[axis] - voxel_l * cell_relative_th
        #标记单元
        self.cell_mark = np.zeros(self.n_cell, dtype=np.int32)
        self.cell_mark[cell_centers[:, axis] < cell_low_bound] = mark[0]
        self.cell_mark[cell_centers[:, axis] > cell_up_bound] = mark[1]
        
        #计算边界坐标阈值(对于面)
        face_low_bound = voxel_l * face_relative_th
        face_up_bound = mesh_l[axis] - voxel_l * face_relative_th
        #标记表面
        faces = self.faces 
        face_centers = self.face_centers
        self.face_mark = np.zeros(faces.shape[0], dtype=np.int32)
        self.face_mark[face_centers[:, axis] < face_low_bound] = mark[0]
        self.face_mark[face_centers[:, axis] > face_up_bound] = mark[1]

    @classmethod
    def read_hsdat(cls, path="", name="hsdat_mesh", fmt="hsdat", on=True):
        #读取有效行
        str_list = file_str_list(f"{path}/{cls.__name__}-{name}.{fmt}")

        #解析头
        n_vert = int(str_list[0][1].replace(",",""))
        n_cell = int(str_list[0][3].replace(",",""))
        
        #解析顶点和单元
        vert_list = str_list[1:1+n_vert]
        cell_list = str_list[1+n_vert:1+n_vert+n_cell]

        verts_info = np.array(vert_list).astype(np.float32)
        verts = verts_info[:, :-1]
        vert_data = verts_info[:, -1]
        cells_info = np.array(cell_list).astype(np.int32)
        cells = cells_info[:, :-1]
        cell_data = cells_info[:, -1]

        if on: print(f"hsdat readed => {path}/{cls.__name__}-{name}.{fmt}")
        return cls(verts, cells, vert_data, cell_data)

    @property 
    def vol(self) -> PhaseVol:
        #计算域形状
        vxl_size = np.array([np.unique(self.verts[:,i]).size for i in range(3)], dtype=np.int32) - 1
        # return PhaseVol(self.cell_data.reshape(vxl_size)).swapaxes(0, 2)
        return PhaseVol(self.cell_data.reshape(vxl_size))

    def write_hsdat(self, path="", name="hsdat_mesh", fmt="hsdat", on=True):
        #一致性转化
        cells = self.cells + 1 #顶点序号从1开始
        verts = self.verts
        cell_attrs = self.cell_data.astype(np.int32)

        n_vert = self.n_vert
        vert_idxs = np.arange(n_vert) + 1
        n_cell = self.n_cell

        # fstr = "# hsdat struct hex mesh generated by PntCorGraph(YSY)\n"
        fstr = "" 
        # fstr += "# vert head:\n"
        fstr += f"vertex={n_vert:>8d}, cuboid={n_cell:>8d}\n"
        # fstr += "# vert data:\n"
        fstr_list = [f"{vert[0]:.6e} {vert[1]:.6e} {vert[2]:.6e} {vert_i:>8d}\n" 
                for vert_i, vert in zip(vert_idxs, verts)]
        fstr += "".join(fstr_list)
        
        # fstr += "# cell data:\n"
        fstr_list = [f"{cell[0]:>10d} {cell[1]:>10d} {cell[2]:>10d} {cell[3]:>10d} {cell[4]:>10d} {cell[5]:>10d} {cell[6]:>10d} {cell[7]:>10d} {attr:>10d}\n" 
                for cell, attr in zip(cells, cell_attrs)]
        fstr += "".join(fstr_list)

        with open(f"{path}/{self.cls_str}-{name}.{fmt}", "w") as out_file:
            out_file.write(fstr)

        if on: print(f"hsdat writed => {path}/{self.cls_str}-{name}.{fmt}")
        return 
        

    def write_bounds(self, path="", name="hsdat_bnd", fmt="hsbnd", on=True):
        """
        如果标记了边界则进行边界文件的写出
        序号按照np.where计算(zyx)
        """
        if not np.any(self.cell_mark): #未定义边界时自动跳过
            print("write bound failed: bound_cell_mark is empty")
            return 

        for mark in np.unique(self.cell_mark):
            if mark <= 0: continue 
            #边界单元索引  (如果标记面和单元不一致 将出错)
            bound_cell_i = np.where(self.cell_mark == mark)[0] 
            bound_face_ri = np.where(self.face_mark == mark)[0] - bound_cell_i * 6   #面的相对索引
            bound_cell_i += 1  #从1开始
            #写文件
            # fstr = "# hsdat bound cells generated by PntCorGraph(YSY)\n"
            fstr = ""
            fstr += "\n".join([f"{cell_i:>10d}{face_ri:>10d}" for cell_i, face_ri in zip(bound_cell_i, bound_face_ri)])

            with open(f"{path}/{self.cls_str}-{name}.{fmt}", "w") as out_file:
                out_file.write(fstr) 

        if on: print(f"hsbnd writed => {path}/{self.cls_str}-{name}.{fmt}")
        return 

    @property
    def faces(self):
        return extract_faces_hex(self.cells)
    
    @property
    def n_face(self):
        return self.faces.shape[0]
    
    @property
    def face_centers(self):
        if self.n_face > 0:
            return np.average(self.verts[self.faces], axis=1)
        else:
            return np.ndarray([0, 4])

def where_xyz(vol):
    """按xyz方向顺序的where"""
    loc = np.where(np.swapaxes(vol[:], 0, 2))
    return (loc[2], loc[1], loc[0])

def np_mode(arr):
    """针对非负整数的优化版本"""
    counts = np.bincount(arr)
    return np.argmax(counts)

# def extrac_edges(cells):
#     """提取边界"""

# def compute_face_adjacency(cells):
#     cell_faces = extract_faces_tet(cells)
#     new_faces, _, invs_idx, _ = remove_dul_points(np.sort(cell_faces, axis=1))

#     face_cell_dict = {tuple(new_faces[i]): np.array([-1, -1]) for i in range(new_faces.shape[0])}
#     for face_i in range(invs_idx.size):
#         key = tuple(new_faces[invs_idx[face_i]])
#         if face_cell_dict[key][0] < 0:
#             face_cell_dict[key][0] = face_i//4
#         else: 
#             face_cell_dict[key][1] = face_i//4
#     return face_cell_dict


def extract_faces_tet(cells):
    """从单元提取表面,顺序对应"""
    faces = np.array([
        cells[:, (0,1,2)],  
        cells[:, (3,0,2)],  
        cells[:, (1,3,2)],  
        cells[:, (3,1,0)],
        ])
    return faces.swapaxes(0,1).reshape(cells.shape[0]*4, 3)

def extract_tets_hex(cells):
    """从单元提取四面体"""
    tets = np.array([
        cells[:, (0,1,4,3)],
        cells[:, (5,6,4,1)],
        cells[:, (7,4,6,3)],
        cells[:, (2,3,6,1)],
        cells[:, (3,1,4,6)],
        ])
    return tets.swapaxes(0,1).reshape(cells.shape[0]*5, 4)

def extract_faces_hex(cells):
    """从单元提取表面,顺序对应(有冗余,临时)"""
    if cells[0].size != 8:
        raise ValueError(f"cells vert number should be 8, having {cells[0].size}")
    faces = np.array([
        cells[:, (4,7,3,0)],  
        cells[:, (1,5,4,0)],  
        cells[:, (0,3,2,1)],  
        cells[:, (5,6,7,4)],
        cells[:, (3,7,6,2)],
        cells[:, (2,6,5,1)],
        ])
    faces = faces.swapaxes(0,1).reshape(cells.shape[0]*6, 4)
    return faces 

def extract_faces_quad(cells):
    """"""
    if cells[0].size != 4:
        raise ValueError(f"cells vert number should be 4, having {cells[0].size}")
    faces = np.array([
        cells[:, (0, 1, 2)],
        cells[:, (0, 2, 3)],
        ])
    faces = faces.swapaxes(0,1).reshape(cells.shape[0]*2, 3)
    return faces 

def query_value(points, data, shape=(50,50,50)):
    """获取网格数据"""
    from scipy.interpolate import griddata
    #网格采样点 
    point_range = np.array([np.min(points, axis=0), np.max(points, axis=0)]) 
    x_pnts = np.linspace(point_range[0, 0], point_range[1, 0], shape[0])
    y_pnts = np.linspace(point_range[0, 1], point_range[1, 1], shape[1])
    z_pnts = np.linspace(point_range[0, 2], point_range[1, 2], shape[2])
    grid_pnts = np.array(np.meshgrid(x_pnts, y_pnts, z_pnts)).reshape((3,-1)).T[:,::-1]
    #网格插值
    grid_value = np.swapaxes(griddata(points, data, grid_pnts).reshape(shape), 0, 2)
    return grid_value

def subdivide_mesh(vertices, faces):
    """
    对三角网格进行一次细分，每个三角形细分为4个小三角形
    
    参数：
        vertices : np.float32数组，形状为(N, 3)
        faces    : np.int32数组，形状为(M, 3)
        
    返回：
        new_vertices : 细分后的顶点数组
        new_faces    : 细分后的三角形面数组
    """
    # 初始化数据结构
    edge_map = {}          # 记录边到新顶点的映射
    new_vertices = vertices.tolist()  # 保留原始顶点
    new_faces = []

    # 遍历所有原始三角形
    for face in faces:
        v0, v1, v2 = face
        
        # 获取或创建每条边的新顶点
        edge_keys = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        
        edge_verts = []
        for key in edge_keys:
            if key not in edge_map:
                # 计算中点并添加到顶点列表
                mid = (vertices[key[0]] + vertices[key[1]]) / 2
                new_vertices.append(mid)
                edge_map[key] = len(new_vertices) - 1  # 记录新顶点索引
            edge_verts.append(edge_map[key])
            
        e0, e1, e2 = edge_verts
        
        # 生成四个新的三角形
        new_faces.extend([
            [v0, e0, e2],
            [e0, v1, e1],
            [e2, e1, v2],
            [e0, e1, e2]
        ])

    # 转换为numpy数组并返回
    return np.array(new_vertices, dtype=np.float32), np.array(new_faces, dtype=np.int32)

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

def vert_face_normals(vert_normals, faces):
    """从顶点计算面法向量"""
    return np.average(vert_normals[faces], 1)

def orient_tetrahedrons(tetrahedrons, vertices):
    """
    调整四面体顶点顺序使法向一致向外
    参数：
        tetrahedrons: 四面体顶点索引数组 (N,4)
        vertices: 顶点坐标数组 (M,3)
    返回：
        调整后的四面体数组，满足右手法则的正体积
    """
    oriented = tetrahedrons.copy()
    
    for i in range(len(oriented)):
        v0, v1, v2, v3 = oriented[i]
        
        # 计算带符号体积
        vec1 = vertices[v1] - vertices[v0]
        vec2 = vertices[v2] - vertices[v0]
        vec3 = vertices[v3] - vertices[v0]
        det = np.dot(np.cross(vec1, vec2), vec3)
        
        # 调整顶点顺序使行列式为正
        if det < 0:
            oriented[i, 1], oriented[i, 2] = oriented[i, 2], oriented[i, 1]
    
    return oriented
    
def calc_voro(points, limits, dispersion, weights):
    """计算voroni图 返回顶点坐标列表,多边形顶点索引列表
    """
    import pyvoro

    voro_regions = pyvoro.compute_voronoi(points, limits, dispersion, weights)
    return voro_extract(voro_regions)

def voro_extract(voro_regions):
    """从voro域提取全局顶点坐标和多边形索引表"""
    #删除重复点 
    #创建新的全局索引
    vertices = []
    verts_conn = []
    local_kverts_conn = {}

    for region_i, region_data in enumerate(voro_regions):
        pts_local = region_data['vertices']

        for face_data in region_data['faces']:
            adj_cell = face_data['adjacent_cell']
            simplex_local = face_data['vertices']
            for kp_local in simplex_local:
                key = (region_i, kp_local)
                if key in local_kverts_conn:
                    kverts_global = local_kverts_conn[key]

                else:
                    kverts_global = len(vertices)
                    pt = pts_local[kp_local]
                    vertices.append(pt)
                    verts_conn.append({region_i: kp_local})
                    local_kverts_conn[key] = kverts_global

                if (adj_cell >= 0) and (adj_cell < len(voro_regions)):
                    conn_info = verts_conn[kverts_global]
                    if adj_cell not in conn_info:
                        adj_cell_data = voro_regions[adj_cell]
                        adj_pts_local = np.array(adj_cell_data['vertices'])
                        rel_pos = adj_pts_local - vertices[kverts_global]
                        sq_dist = np.sum(rel_pos * rel_pos, axis=-1)
                        adj_kv_local = np.argmin(sq_dist)

                        adj_key = (adj_cell, adj_kv_local)
                        local_kverts_conn[adj_key] = kverts_global
                        verts_conn[kverts_global][adj_cell] = adj_kv_local

    #创建面列表和域列表
    faces = []
    regions_faces = [[] for region in voro_regions]
    for region_i, region_data in enumerate(voro_regions):
        for face_data in region_data['faces']:
            n_adj_region = face_data['adjacent_cell']
            if n_adj_region >= len(voro_regions):
                n_adj_region = -1

            if n_adj_region < region_i:
                neighbor_pair = (n_adj_region, region_i)
                s_lcl = face_data['vertices']
                s_glbl = [local_kverts_conn[(region_i, kp)] for kp in s_lcl]
                if n_adj_region < 0:
                    pts_f = [vertices[kp] for kp in s_glbl]
                    if not _is_outward(pts_f, n_adj_region):
                        s_glbl.reverse()

                face_num = len(faces)
                faces.append(s_glbl)

                for f_cell_num in neighbor_pair:
                    if (f_cell_num >= 0) and (f_cell_num < len(voro_regions)):
                        regions_faces[f_cell_num].append(face_num)

    #从区域的面索引建立区域的顶点索引 
    regions = []
    for region in regions_faces:
        face_vertices_idx = []
        for face_idx in region:
            face_vertices_idx += faces[face_idx]
        regions.append(np.unique(face_vertices_idx))
    return np.array(vertices), regions, faces, regions_faces


def _is_outward(pt_list, voropp_face_num):
    n_dim = len(pt_list[0])

    voropp_sgn = 1-2*(voropp_face_num % 2)
    voropp_axis = int((-voropp_face_num - 1)/2)
    face_vec = voropp_sgn * np.eye(n_dim)[voropp_axis]

    if n_dim == 2:
        pt1 = pt_list[0]
        pt2 = pt_list[1]
        rel_pos = np.array(pt2) - np.array(pt1)
        n_vec = np.array([-rel_pos[1], rel_pos[0]])
    elif n_dim == 3:
        pt1 = pt_list[0]
        pt2 = pt_list[1]
        pt3 = pt_list[2]
        r1 = np.array(pt2) - np.array(pt1)
        r2 = np.array(pt3) - np.array(pt1)
        n_vec = np.cross(r1, r2)
    else:
        raise ValueError('Function does not support {}D.'.format(n_dim))

    n_u = n_vec / np.linalg.norm(n_vec)
    return np.dot(n_u, face_vec) > 0

def _write_poly(verts, faces, region_centers=[], region_attrs=[], fname=""):
    n_vert = verts.shape[0]
    n_face = len(faces) 
    n_region = np.shape(region_centers)[0]
    n_dim = 3 

    #写文件
    fstr = "#poly mesh generated by PntCorGraph(YSY)\n"
    fstr += "#nodes head:\n"
    fstr += f"{n_vert} {n_dim} 0 0\n"
    if n_vert > 0:
        fstr += "#nodes data:\n"
        for vert_i, vert in enumerate(verts):  #无顶点属性 
            fstr += f"{vert_i + 1} {vert[0]} {vert[1]} {vert[2]}\n"
    fstr += "\n"

    fstr += "#faces head:\n"
    fstr += f"{n_face} 0\n"
    if n_face > 0:
        fstr += "#faces data:\n"
        for face_i, face in enumerate(faces):
            fstr += "1 \n"  #当前面包含一个多边形
            fstr += f"{np.shape(face)[0]} " + " ".join(map(str, face)) + "\n"
    fstr += "\n"

    fstr += "#holes head:\n"
    fstr += "0\n"  #暂无孔洞的定义
    fstr += "\n"

    fstr += "#regions head:\n"
    fstr += f"{n_region}\n"  #暂不限制区域大小
    if n_region > 0:
        fstr += "#regions data:\n"
        for region_i, (region, attr) in enumerate(zip(region_centers, region_attrs)):
            fstr += f"{region_i + 1} {region[0]} {region[1]} {region[2]} {attr}\n"
    fstr += "\n"

    #写poly文件
    with open(f"{fname}", "w") as out_file:
        out_file.write(fstr)




def file_str_list(fname, comm_start="#"):
    with open(fname, "r") as file:
        fstr = file.read()

    str_list = []
    for line in fstr.split("\n"):
        llist = line.split()
        if len(llist) <= 0:
            continue 
        if llist[0] == comm_start:  #去除注释
            continue 
        str_list.append(llist) 

    return str_list

  
def read_node(fname):
    """读取节点坐标文件"""
    #获取文件字符串列表
    str_list = file_str_list(fname)
    
    #读取头
    str_head = str_list[0]
    n_pnt = int(str_head[0])
    n_dim = int(str_head[1]) 

    #读取点坐标
    str_data = str_list[1:1+n_pnt]
    arr_data = np.array(str_data).astype(np.float32)
    verts = arr_data[:, 1:1+n_dim]
    # vert_data = arr_data[:, 1+n_dim]
    vert_data = np.zeros(n_pnt, dtype=np.int32)
    return verts, vert_data


def read_ele(fname):
    """读取单元文件"""
    #获取文件字符串列表
    str_list = file_str_list(fname)

    #读取头
    str_head = str_list[0]
    n_ele = int(str_head[0])
    n_idx = int(str_head[1])

    #读取顶点索引 
    str_data = str_list[1:1+n_ele]
    arr_data = np.array(str_data).astype(np.int64)
    cells = arr_data[:, 1:1+n_idx] - 1 #保持从0开始
    cell_data = arr_data[:, 1+n_idx]
    return cells, cell_data


def read_face(fname):
    """读取面(包含邻接关系)文件"""
    str_list = file_str_list(fname)

    #读取头
    str_head = str_list[0]
    n_face = int(str_head[0])
    n_idx = 3

    #读取顶点索引 
    str_data = str_list[1:1+n_face]
    arr_data = np.array(str_data).astype(np.int64)
    faces = arr_data[:, 1:1+n_idx] - 1

    #读取邻接单元索引
    fncells = arr_data[:, 2+n_idx:2+n_idx+2] - 1
    return faces, fncells


def read_neigh(fname):
    """读取邻接单元文件"""
    str_list = file_str_list(fname)

    #读取头
    str_head = str_list[0]
    n_ele = int(str_head[0])
    n_idx = int(str_head[1])

    #读取单元索引 
    str_data = str_list[1:1+n_ele]
    arr_data = np.array(str_data).astype(np.int64)
    neighs = arr_data[:, 1:1+n_idx] - 1
    return neighs 


def read_tetgen(fname, idx=1):
    """读取tetgen文件,返回顶点,面,单元,邻接关系"""  
    verts, vert_data = read_node(f"{fname}.{idx}.node")
    faces, fncells = read_face(f"{fname}.{idx}.face")
    cells, cell_data = read_ele(f"{fname}.{idx}.ele")
    return verts, cells, vert_data, cell_data, faces, fncells 


class VoronoiMesh(Mesh):
    cell_type = "tetra"

    def __init__(self, points, reg, rads=[], attrs=[]):
        """初始化,建立Laguerre-Voronoi多面体网格,进行Delaunay三角剖分
        出现voro++错误的可能原因(跳过一些种子的划分): 
        1.种子重复 2.一个种子包含了另一个种子的中心
        解决方案:
        1.渲染检测,移除重叠种子 2.种子半径缩减(同时缩减结果不变)
        
        tetgen错误的可能原因:
        1.种子半径差距过大,voronoi划分后出现过小的区域
        
        Args:
            pos (np.ndarray): 二维向量,种子中心坐标
            domain (Domain): 划分的边界限制,需要和seeds的domain大小对应,网格密度不需要对应 
            r (np.array): 种子半径,对应Laguerre-Voronoi的半径 
            attrs (np.array): 划分单元的属性 
            seeds_type (np.array): 取值0,1,2,对应"void","crystal","amorphous".进行不同的划分单元边界处理
        Examples:
            pos = np.array([[1,1,1],[1,1,2],[1,2,1],[2,1,1],[2,2,1],[2,1,2],[1,2,2],[2,2,2]])
            domain = Domain([3,3,3], 0.1)
            rads = np.full(8, 0.2)
            attrs = np.array([1,1,2,2,3,3,2,2])
            vm = VoronoiMesh(pos, domain, rads, attrs)

            points = (np.mgrid[:3,:3,:3].reshape((3, -1)).T + 1) * 0.25
            r = np.full(27, 0.1)
            attrs = np.array([0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
            vm = VoronoiMesh(points, elec_dmn, r, attrs)
        """
        super().__init__()
        if not np.any(rads):
            rads = np.ones(points.shape[0])
        if not np.any(attrs):
            attrs = np.ones(points.shape[0])
        
        self.attrs = attrs  #中心点的属性
        self.points = points
        self.rads = rads
        self.reg = reg 

        #调用pyvoro计算voroni图 不允许使用周期性边界
        self.vertices, self.regions, self.faces, self.region_faces = calc_voro(points, reg.limits, 1.0, rads)
        #self.calc_tetra_tetgen()

    @classmethod
    def from_seeds(cls, seeds, reg):
        """从种子初始化"""
        return cls(seeds.loc, reg, seeds.r, seeds.mark)

    def __repr__(self):
        return super().__repr__() + f"\nn_region: {self.n_region} \nn_vertice: {self.n_vertice}"
    
    @property
    def n_region(self):
        return len(self.regions)
    @property
    def n_point(self):
        return self.points.shape[0]
    @property
    def n_vertice(self):
        return self.vertices.shape[0]

    #外部定义函数
    def write_poly(voro_mesh, fname=""):
        """写voronoi线性多面体网格文件(tetgen的poly)"""
        #处理定义一致性转换
        verts = voro_mesh.vertices
        faces = [[vidx + 1 for vidx in face] for face in voro_mesh.faces]  #面的顶点索引从1开始
        #不直接使用顶点坐标(因为可能在域外部) 而是voro++输出的域的重心  region_centers = voro_mesh.points   
        region_centers = [np.average(voro_mesh.vertices[region], axis=0) for region in voro_mesh.regions]

        attrs_tab = np.arange(np.max(voro_mesh.attrs) + 1, dtype=np.int32) + 1  #避免属性为0 
        region_attrs = attrs_tab[voro_mesh.attrs.astype(np.int32)] #保证为整数属性
        return _write_poly(verts, faces, region_centers, region_attrs, fname)
        

    def calc_tetra_tetgen(voro_mesh, min_angle=14):
        delete_fmt_list = ["ele", "face", "node", "edge", "neigh"]
        del_cmd = "del"  #rm #根据平台变换
        fdir = "tmp/"   #tmp/

        #将voro多边形写为poly网格
        poly_fname = fdir + voro_mesh.cls_id_str + ".poly"
        voro_mesh.write_poly(poly_fname)

        #调用命令行对网格进行划分
        #面邻接关系输出
        tetgen_ret = os.system(f"tetgen.exe -pq2.0/{min_angle} -A -nn {poly_fname}")
        #vtk输出
        # tetgen_ret = os.system(f"tetgen.exe -pqA -k {poly_fname}")
        # if tetgen_ret != 0:
        #     print(f"calc_tri_mesh_tetgen failed: tetgen run error for file:{poly_fname}")
        voro_mesh.verts, voro_mesh.cells, voro_mesh.vert_data, voro_mesh.cell_data, voro_mesh.cell_faces, voro_mesh.fncells = read_tetgen(fdir + voro_mesh.cls_id_str)
        
        #读取划分结果(vtk)
        # voro_mesh.from_vtk(fdir + voro_mesh.class_id_str + ".1.vtk")  #此处1为边界标记序号
        
        #清除中间文件
        for fmt in delete_fmt_list:
            clean_ret = os.system(f"{del_cmd} {fdir}*.{fmt}")
            if clean_ret != 0:
                print(f"clean tetgen file failed: {del_cmd}")
        
        if voro_mesh.n_cell and voro_mesh.n_vert:
            return voro_mesh.cell_verts, voro_mesh.cells, voro_mesh.vert_data, voro_mesh.cell_data


# import pyvoro

# reg = Region(0.8, [100, 100, 75])
# seeds = seeds.remove_overlap_aabb()
# voro_regions = pyvoro.compute_voronoi(seeds.loc, reg.limits, 1.0, seeds.r)

# verts = []
# faces = []
# verts_cnt = 0
# for cell in voro_regions:
#     verts.append(cell["vertices"]) 
#     for face in cell["faces"]:
#         vert0 = np.repeat(face["vertices"][0], np.size(face["vertices"]) - 2).reshape(-1, 1)
#         vert1 = np.repeat(face["vertices"][1:], 2)[1:-1].reshape(-1, 2)
#         faces.append(np.concatenate([vert0, vert1], 1) + verts_cnt - 1) 
#     verts_cnt += len(cell["vertices"])

# verts = np.concatenate(verts, 0)
# faces = np.concatenate(faces, 0)

# verts, verts_idx, invs_idx = remove_dul_points(verts) #新不重复顶点, 新顶点在原顶点内的索引, 原顶点的新索引
# faces = invs_idx[faces] #重建面的顶点索引 

# # _, faces_idx, _ = remove_dul_points(np.sort(faces, axis=1))
# # faces = faces[faces_idx]
# TriMesh(verts, faces).save_mesh("test.vtk")