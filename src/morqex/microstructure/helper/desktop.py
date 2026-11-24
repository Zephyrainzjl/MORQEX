from morqex.microstructure.core.particle_graph import *
from morqex.microstructure.mesh.elements import *

import tkinter as tk 
from tkinter import ttk
from tkinter import filedialog

import threading

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import io
import sys 
import os 
import pickle 
import json

window = tk.Tk()
window.title('PCG Generator (YSY)')
window.geometry('1080x720')
window.current_scale = [1.0]
# window.tk.call("tk", "scaling", 1.5) 

#-------------------------------------------------------------------------------------#
# 打印设置
#-------------------------------------------------------------------------------------#

np.set_printoptions(legacy="1.25")

def float_repr(x):
    # 控制打印精度，例如保留 3 位小数
    return f"{x:.4f}"

def custom_display_hook(value):
    if value is None:
        return
    if isinstance(value, float):
        print(float_repr(value))
    else:
        print(custom_format(value))

def custom_format(obj):
    if isinstance(obj, float):
        return float_repr(obj)
    if isinstance(obj, (list, tuple)):
        inner = ", ".join(custom_format(x) for x in obj)
        if isinstance(obj, list):
            return "[" + inner + "]"
        else:
            return "(" + inner + ")"
    return repr(obj)

sys.displayhook = custom_display_hook

#-------------------------------------------------------------------------------------#
# PCG UI会话
#-------------------------------------------------------------------------------------#

class TKUI():
    """"""


class RedirectText(io.TextIOBase):
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, output): self.text_widget.insert(tk.END, output)
    def flush(self): pass

class RedirectBox(io.TextIOBase):
    def __init__(self, master, title="massage"):
        self.master = master
        self.title = title

    def write(self, output): 
        box = tk.Toplevel(self.master)
        box.title(self.title)
        txt = tk.Text(box, width=50, font=("Consolas", 10))
        txt.insert(tk.END, output)
        txt.pack() 
        box.mainloop()

    def flush(self): pass



def set_scaling(scale):
    """动态调整 UI 缩放 + 窗口大小"""
    # 1. UI 元素缩放
    window.tk.call("tk", "scaling", scale)

    # 2. 获取当前窗口大小
    window.update_idletasks()   # 刷新以获得真实尺寸
    w = window.winfo_width()
    h = window.winfo_height()

    # 3. 按比例调整窗口大小
    w2 = int(w * scale / window.current_scale[0])
    h2 = int(h * scale / window.current_scale[0])

    # 4. 更新 geometry
    window.geometry(f"{w2}x{h2}")

    # 保存当前缩放值
    window.current_scale[0] = scale




def exec_in_main_thread(func, *args, **kwargs):
    """在 Tk 主线程执行（适用于 UI 更新）"""
    window.after(0, lambda: func(*args, **kwargs))

        
class Context():
    def __init__(self, env=None):
        if not env: env = locals()
        self.env = env 
    
        #状态栏
        self.state_panel = tk.PanedWindow(window, orient=tk.HORIZONTAL, relief="raised")
        self.state_panel.pack(fill=tk.X, side=tk.BOTTOM)
        self.view_cfg = ViewConfig()
        self.save_cfg = SaveConfig()
        self.view_reg = ViewRegion()
        self.init_state()

        #可视化栏
        self.view_panel = tk.PanedWindow(window, orient=tk.VERTICAL, relief="sunken")
        self.view_panel.pack(fill=tk.Y, side=tk.RIGHT)     
        self.tkvar_view_pan = tk.StringVar(window, value="view_panel")
        self.tklab_view_pan = tk.Label(self.view_panel, textvariable=self.tkvar_view_pan, width=45)
        self.tklab_view_pan.pack(side=tk.TOP)
        self.view_slot = {}

        #命令栏
        self.cmd_panel = tk.PanedWindow(window, orient=tk.VERTICAL, relief="sunken")
        self.cmd_panel.pack(fill=tk.Y, side=tk.LEFT)
        self.cmd_panel_head = tk.PanedWindow(self.cmd_panel)
        self.cmd_panel_head.pack(side=tk.TOP)
        self.tklab_cmd_pan = tk.Label(self.cmd_panel_head, text="cmd_panel", width=42)
        self.tklab_cmd_pan.pack(side=tk.LEFT)
        self.tklab_cmd_pan = tk.Button(self.cmd_panel_head, width=3, text="↵", command=self.exec_scr_cmd)
        self.tklab_cmd_pan.pack(side=tk.LEFT)
        
        #脚本框
        self.tktxt_scr = tk.Text(self.cmd_panel, wrap=tk.WORD, width=45, height=22, font=("Consolas", 10))
        self.tktxt_scr.pack(side=tk.TOP)

        #添加对象命令
        self.cmd_panel_rows = []
        self.cmd_panel_rows.append(tk.PanedWindow(self.cmd_panel))
        self.cmd_panel_rows[-1].pack(side=tk.TOP)

        #单行命令
        self.cmd_panel_rows.append(tk.PanedWindow(self.cmd_panel))
        self.cmd_panel_rows[-1].pack(side=tk.TOP)
        self.tkvar_cmd = tk.StringVar(self.cmd_panel, value="")
        self.tkety_cmd = tk.Entry(self.cmd_panel_rows[-1], width=40, textvariable=self.tkvar_cmd, validate="none", font=("Consolas", 10))
        # self.tkety_cmd.bind("<Enter>", self.exec_cmd)
        self.tkety_cmd.pack(side=tk.LEFT)
        
        #执行单行命令
        self.tkbtn_cmd = tk.Button(self.cmd_panel_rows[-1], width=3, text="↵", command=self.exec_cmd, height=1)
        self.tkbtn_cmd.pack(side=tk.LEFT)

        #输出框
        self.out_panel = tk.PanedWindow(self.cmd_panel, width=45, relief="sunken")
        self.out_panel.pack(side=tk.TOP) 
        self.tklab_out = tk.Label(self.out_panel, text="out_panel", width=45)
        self.tklab_out.pack(side=tk.TOP)

        self.tktxt_out = tk.Text(self.out_panel, wrap=tk.WORD, width=45, height=18, font=("Consolas", 10))
        self.tktxt_out.pack(side=tk.TOP)
        sys.stdout = RedirectText(self.tktxt_out)
        #报错框
        # sys.stderr = RedirectText(self.tktxt_out)
        # sys.stderr = RedirectBox(self.cmd_panel, "Error")

        #对象面板
        self.obj_panel = tk.PanedWindow(window, orient=tk.VERTICAL, relief="sunken")
        self.obj_panel.pack(fill=tk.Y, side=tk.TOP)
        self.obj_panel_head = tk.PanedWindow(self.obj_panel)
        self.obj_panel_head.pack(side=tk.TOP)

        self.tklab_obj_panel = tk.Label(self.obj_panel_head, text="obj_panel", width=42)
        self.tklab_obj_panel.pack(side=tk.LEFT)
        self.tkbtn_obj_panel = tk.Button(self.obj_panel_head, text="↻", command=self.relist_obj, width=3)
        self.tkbtn_obj_panel.pack(side=tk.LEFT)
        # self.tkbtn_sort_obj = tk.Button(self.obj_panel_head, text="↓", command=self.move_obj, width=3)
        # self.tkbtn_sort_obj.pack(side=tk.LEFT)

        #对象列表
        self.tklst_obj = tk.Listbox(self.obj_panel, width=45, height=20)
        self.tklst_obj.pack(side=tk.TOP)
        self.tklst_obj.bind("<Double-Button-1>", self.view_select)
        self.obj_slot = {} 

        #对象操作
        self.obj_panel_rows = []
        self.obj_panel_rows.append(tk.PanedWindow(self.obj_panel))
        self.obj_panel_rows[-1].pack(side=tk.TOP)

        self.tkbtn_objdel = tk.Button(self.obj_panel_rows[-1], text="×", width=3, command=self.del_select)
        self.tkbtn_objdel.pack(side=tk.LEFT)
        self.tkbtn_objcopy = tk.Button(self.obj_panel_rows[-1], text="cp", width=3, command=self.copy_select)
        self.tkbtn_objcopy.pack(side=tk.LEFT)
        self.tkbtn_objname = tk.Button(self.obj_panel_rows[-1], text="rn", width=3, command=self.rename_select)
        self.tkbtn_objname.pack(side=tk.LEFT)
        self.tkbtn_objsave = tk.Button(self.obj_panel_rows[-1], text="↓", width=3, command=self.save_select)
        self.tkbtn_objsave.pack(side=tk.LEFT)
        self.tkbtn_objload = tk.Button(self.obj_panel_rows[-1], text="↑", width=3, command=self.load_select)
        self.tkbtn_objload.pack(side=tk.LEFT)
        
        #外部对象列表
        self.file_panel = tk.PanedWindow(self.obj_panel, relief="sunken")
        self.file_panel.pack(side=tk.TOP)
        self.file_panel_head = tk.PanedWindow(self.file_panel)
        self.file_panel_head.pack(side=tk.TOP)
        self.tklab_file_panel = tk.Label(self.file_panel_head, text="file_panel", width=39)
        self.tklab_file_panel.pack(side=tk.LEFT)
        self.tkbtn_file_panel = tk.Button(self.file_panel_head, text="↻", command=self.relist_file, width=3)
        self.tkbtn_file_panel.pack(side=tk.LEFT)
        self.tkbtn_file_dir = tk.Button(self.file_panel_head, text="...", command=self.save_cfg.askdir, width=3)
        self.tkbtn_file_dir.pack(side=tk.LEFT)

        self.tklst_file = tk.Listbox(self.file_panel, width=45, height=12)
        self.tklst_file.pack(side=tk.TOP)
        # self.tklst_file.bind("<Double-Button-1>", self.view_file_select)
        self.file_slot = {} 

        self.relist_obj()
        self.relist_file()

        #菜单栏
        self.menu_panel = tk.Menu(window)
        self.menus = []

        #项目菜单
        self.out_path = None 
        self.menus.append(tk.Menu(self.menu_panel, tearoff=0))
        self.menus[-1].add_command(label="new", command=self.new_proj)
        # self.menus[-1].add_command(label="close", command=self.close_proj)
        self.menus[-1].add_command(label="load", command=self.load_proj)
        self.menus[-1].add_command(label="save", command=self.save_proj)
        self.menus[-1].add_command(label="save as", command=self.save_as_proj)
        self.menu_panel.add_cascade(label="Project", menu=self.menus[-1])
        self.menu_panel.bind("<Control-s>", self.save_proj)

        #脚本菜单
        self.menus.append(tk.Menu(self.menu_panel, tearoff=0))
        self.menus[-1].add_command(label="load", command=self.load_scr)
        self.menus[-1].add_command(label="save", command=self.save_scr)
        self.menu_panel.add_cascade(label="Script", menu=self.menus[-1])

        #实例菜单
        self.menus.append(tk.Menu(self.menu_panel, tearoff=0))
        self.menus[-1].add_command(label="load", command=self.load_select)
        self.menus[-1].add_command(label="save", command=self.save_select)
        self.menu_panel.add_cascade(label="Instance", menu=self.menus[-1])

        #视图菜单
        self.menus.append(tk.Menu(self.menu_panel, tearoff=0))
        scales = {"100%": 1.0, "125%": 1.25, "150%": 1.5, "200%": 2.0}
        for name, value in scales.items():
           self.menus[-1].add_command(label=name, command=lambda v=value: set_scaling(v))
        self.menu_panel.add_cascade(label="View", menu=self.menus[-1])
            
        #添加对象菜单
        for base_cls in base_clss:
            self.menus.append(tk.Menu(self.menu_panel, tearoff=0)) 
            obj_clss = [cls for cls in TKUI.__subclasses__() if issubclass(cls, base_cls)]
            for cls in obj_clss:
                self.menus[-1].add_command(label=cls.__name__, command=exec_str_ret(f"lambda: add_{cls.__name__}()"))
            self.menu_panel.add_cascade(label=base_cls.__name__, menu=self.menus[-1])


    def init_state(self):
        self.view_cfg.tkvar_view.set("-")
        self.save_cfg.tkvar_view.set("-")
        self.view_reg.tkvar_view.set("-")
        self.view_cfg.__repr__(master=self.state_panel)
        self.save_cfg.__repr__(master=self.state_panel)
        self.view_reg.__repr__(master=self.state_panel)

    def get_name(self, obj):
        for name in self.env:
            if id(self.env[name]) == id(obj):
                return name 
        return None
    
    def capture_obj(self):
        """捕获当前对话中的UI对象"""
        self.obj_slot = {} 
        for name in self.env:
            if isinstance(self.env[name], TKUI):
                self.obj_slot.update({name: self.env[name]})
        
    def relist_obj(self, event="<Enter>"):
        self.capture_obj() 
        self.sort_obj()

        self.tklst_obj.delete(0, tk.END)
        for name in self.obj_slot:
            self.tklst_obj.insert(tk.END, f"{name}: {self.obj_slot[name]}")
    
    def sort_obj(self):
        sort_name = sorted(self.obj_slot, key=lambda x: clss_order[self.obj_slot[x].__class__])
        self.obj_slot = {name: self.obj_slot[name] for name in sort_name}

    # def move_obj(self):
    #     names = self.get_select_obj_names()
    #     for name in names: 
    #         pass 
    #     return 

    def capture_file(self):
        """捕获当前的外部文件"""
        self.file_slot = {} 
        if not os.path.isdir(self.save_cfg.dir): 
            os.mkdir(self.save_cfg.dir)
        fnames = os.listdir(self.save_cfg.dir)
        for fname in fnames:
            name = fname.split("-")[-1].split(".")[0]
            cls_name = fname.split("-")[0]
            obj_cls_names = [cls.__name__ for cls in obj_clss]
            if fname.split(".")[-1] == self.save_cfg.fmt and cls_name in obj_cls_names:
                self.file_slot.update({name: fname})

    def relist_file(self, event="<Enter>"):
        self.capture_file() 
        self.tklst_file.delete(0, tk.END)
        for name in self.file_slot:
            self.tklst_file.insert(tk.END, f"{name}: {self.file_slot[name]}")

    def view_select(self, event="<Enter>"):
        self.view_obj(self.get_select_obj_names())

    # def view_file_select(self, event="<Enter>")
    #     self.load_select(event)

    def del_select(self, event="<Enter>"):
        names = self.get_select_obj_names()
        for name in names: 
            self.exec_cmd(cmd=f"del {name}")
        self.relist_obj()

        names = self.get_select_file_names()
        for name in names: 
            self.exec_cmd(cmd=f"os.remove('{self.save_cfg.dir}/{self.file_slot[name]}')")
        self.relist_file()

    def copy_select(self, event="<Enter>"):
        names = self.get_select_obj_names()
        for name in names:
            self.exec_cmd(cmd=f"{name}_copy = {name}.copy()")
        self.relist_obj()

    def rename_select(self, event="<Enter>"):
        if not np.size(self.get_select_obj_names()): 
            return 

        name = self.get_select_obj_names()[0]
        new_name = self.tkvar_cmd.get() #从单行命令获取
        if new_name:
            self.exec_cmd(cmd=f"{new_name} = {name}")
            self.exec_cmd(cmd=f"del {name}")
        else:
            warnings.warn(f"rename warn: no name provided for {name}")
        self.relist_obj()
        
    def save_select(self, event="<Enter"):
        names = self.get_select_obj_names()
        for name in names:
            self.exec_cmd(cmd=f"{name}.save()")
        self.relist_file()

    def load_select(self, event="<Enter"):
        names = self.get_select_file_names()
        for name in names:
            cls_str = self.file_slot[name].split("-")[0] 
            self.exec_cmd(cmd=f"{name} = {cls_str}.load('{name}')")
        self.relist_file()

    # def save_obj(self, event="<Enter"):
    #     names = self.get_select_obj_names()
    #     for name in names:
    #         self.exec_cmd(cmd=f"{name}.save()")
    #     self.relist_file()

    # def load_obj(self, event="<Enter"):
    #     names = self.get_select_file_names()
    #     for name in names:
    #         cls_str = self.file_slot[name].split("-")[0] 
    #         self.exec_cmd(cmd=f"{name} = {cls_str}.load('{name}')")
    #     self.relist_file()

    def get_select_obj_names(self):
        select_idx = self.tklst_obj.curselection()
        if np.size(select_idx) and np.size(list(self.obj_slot.keys())): 
            obj_strs = np.array(list(self.obj_slot.keys()))[np.array(select_idx)]
            return obj_strs
        else:
            return [] 
    
    def get_select_file_names(self):
        select_idx = self.tklst_file.curselection()
        if np.size(select_idx) and np.size(list(self.file_slot.keys())): 
            file_strs = np.array(list(self.file_slot.keys()))[np.array(select_idx)]
            return file_strs
        else:
            return [] 

    def view_obj(self, names):
        """显示对象控件"""
        #注销显示
        for name in self.view_slot:
            if isinstance(self.view_slot[name], TKUI): 
                self.view_slot[name].__unrepr__()
        
        #注册显示
        self.tkvar_view_pan.set(f"view_panel: {names}")
        self.view_slot = {}
        for name in names:
            if isinstance(self.obj_slot[name], TKUI):
                self.view_slot.update({name: self.obj_slot[name]})
                self.obj_slot[name].__repr__(master=self.view_panel)

    def exec_scr_cmd(self, event="<Enter>"):
        """执行脚本框"""
        scr_str = self.tktxt_scr.get("0.0", tk.END)
        if scr_str:
            exec(scr_str, self.env)
            self.relist_obj()
            self.tktxt_scr.insert(tk.END, f"\n# script run\n")

    def save_scr(self, event="<Enter>"):
        """保存脚本"""
        scr_str = self.tktxt_scr.get("0.0", tk.END)
        out_path = filedialog.asksaveasfilename(defaultextension=".py", 
            filetypes=[("Text Files", "*.txt"), ("Python Files", "*.py")])
        if not out_path: return 
        with open(out_path, "w") as f:
            f.write(scr_str)

    def load_scr(self, event="<Enter>"):
        """导入脚本"""
        in_path = filedialog.askopenfilename(defaultextension=".py", 
            filetypes=[("Text Files", "*.txt"), ("Python Files", "*.py"), ("All Files", "*.*")])
        if not in_path: return 
        with open(in_path, "r") as f:
            scr_str = f.read()
        self.tktxt_scr.insert(tk.END, scr_str)

    # def exec_cmd(self, event="<Enter>", cmd=""):
    #     """执行命令行"""
    #     if cmd:
    #         cmd_str = cmd 
    #     else: 
    #         cmd_str = self.tkvar_cmd.get()
    #         self.tkvar_cmd.set("")

    #     #写入已执行命令
    #     if cmd_str: self.tktxt_scr.insert(tk.END, f"{cmd_str}\n")
    #     #执行当前命令
    #     exec(cmd_str, self.env)

    #     #检查对象更新
    #     self.relist_obj()


    def exec_cmd(self, event="<Enter>", cmd=""):
        """外层接口：在新线程中执行命令，避免阻塞 UI"""
        threading.Thread(
            target=self._exec_cmd_internal,
            args=(event, cmd),
            daemon=True
        ).start()

    def _exec_cmd_internal(self, event, cmd):
        """原本的 exec_cmd 内容（放在线程中执行）"""

        # 读取命令要在线程里执行没有问题
        if cmd:
            cmd_str = cmd
        else:
            cmd_str = self.tkvar_cmd.get()
            # 清空输入框 -> 改到主线程执行
            exec_in_main_thread(self.tkvar_cmd.set, "")

        # 写入执行记录 -> 必须在主线程更新 UI
        if cmd_str:
            exec_in_main_thread(self.tktxt_scr.insert, tk.END, f"{cmd_str}\n")

        # -------------------------------
        #      在子线程执行命令逻辑
        # -------------------------------
        # exec执行 Python 代码不会影响线程安全
        exec(cmd_str, self.env)

        # 执行后刷新对象列表 -> 通常有 UI 操作，所以放主线程
        exec_in_main_thread(self.relist_obj)

    def __getstate__(self):
        return {"save_cfg": self.save_cfg, "view_cfg": self.view_cfg, "view_reg": self.view_reg,
            "scr_text": self.tktxt_scr.get("0.0", tk.END), "tktxt_out": self.tktxt_out.get("0.0", tk.END), 
            "obj_slot": self.obj_slot, "view_slot": self.view_slot, "out_path": self.out_path}

    def __setstate__(self, state):
        self.out_path = state["out_path"]
        self.save_cfg.__unrepr__()
        self.view_cfg.__unrepr__()
        self.view_reg.__unrepr__()

        self.save_cfg = state["save_cfg"]
        self.view_cfg = state["view_cfg"]
        self.view_reg = state["view_reg"]
        self.init_state()

        self.tktxt_scr.insert(tk.END, state["scr_text"])
        self.tktxt_out.insert(tk.END, state["tktxt_out"])
        for name in state["obj_slot"]:
            self.obj_slot.update({name: state["obj_slot"][name]})
            self.env.update({name: state["obj_slot"][name]})
        for name in state["view_slot"]:
            self.view_slot.update({name: state["view_slot"][name]})

    def new_proj(self):
        for name in self.obj_slot:
            self.exec_cmd(cmd=f"del {name}")
        self.obj_slot = {} 
        for name in self.view_slot:
            self.view_slot[name].__unrepr__()
        self.view_slot = {} 
        self.file_slot = {} 
        self.out_path = None 

        self.save_cfg.__unrepr__()
        self.view_cfg.__unrepr__()
        self.view_reg.__unrepr__()

        self.save_cfg = SaveConfig()
        self.view_cfg = ViewConfig()
        self.view_reg = ViewRegion()
        self.init_state()

        self.tktxt_scr.delete("1.0", tk.END)
        self.tktxt_out.delete("1.0", tk.END)
        self.relist_obj()
        self.relist_file()

        #不执行 容易覆盖
        # self.exec_cmd(cmd="reg = TKRegion()")
        # self.exec_cmd(cmd="uni = TKDistFunc(0.0, 1.0)")

    def save_proj_path(self, out_path):
        if not out_path: return 
        proj_dict = self.__getstate__()

        #在备份文件中保存
        if out_path[-4:] == ".pkl":
            with open(out_path+".backup", "wb") as out_file:
                pickle.dump(proj_dict, out_file)
                print(f"{out_path} saved.")
        elif out_path[-5:] == ".json":
            with open(out_path+".backup", "w") as out_file:
                json.dump(jsonlize(proj_dict), out_file, indent=True)
                print(f"{out_path} saved.")

        #save成功删除备份
        if os.path.isfile(out_path): os.remove(out_path)
        os.rename(out_path+".backup", out_path)
        # os.remove(out_path+".backup")

    def save_as_proj(self):
        out_path = filedialog.asksaveasfilename(defaultextension=".json", 
            filetypes=[("Json Files", "*.json"), ("Pickle Files", "*.pkl")])
        self.out_path = out_path
        self.save_proj_path(out_path)
        
    def save_proj(self):
        if self.out_path: 
            out_path = self.out_path
        else: 
            out_path = filedialog.asksaveasfilename(defaultextension=".json", 
            filetypes=[("Json Files", "*.json"), ("Pickle Files", "*.pkl")])
            self.out_path = out_path
        self.save_proj_path(out_path)


    def load_proj(self):
        in_path = filedialog.askopenfilename(defaultextension="*.json", 
            filetypes=[("Json Files", "*.json"), ("Pickle Files", "*.pkl"), ("All Files", "*.*")])
        if not in_path: return 
        if in_path[-4:] == ".pkl":
            with open(in_path, "rb") as in_file:
                proj_dict = pickle.load(in_file)
                print(f"{in_path} loaded.")
        else: 
            with open(in_path, "r") as in_file:
                proj_dict = objectlize(json.load(in_file))
                print(f"{in_path} loaded.")
        self.__setstate__(proj_dict) 
        self.relist_obj()
        self.relist_file()

#-------------------------------------------------------------------------------------#
# 程序UI扩展
#-------------------------------------------------------------------------------------#

def tkinit_dec(initfunc, vars={}, pltfuncs=[], pltsize=(3, 3), *args, **kwargs):
    """TKinter面板
    Args:
        master: 主控件
        ivars: 从面板输入变量,通过property直接引用
        ovars: 输出到面板的变量,每次update更新
    """
    def tkinit(self, *args, **kwargs):
        self.vars = vars
        self.pltfuncs = pltfuncs 
        self.pltsize = pltsize
        
        self.panel = None 

        self.tkvar_str = tk.StringVar(window)
        self.tkvar_view = tk.StringVar(window, value="+")

        for key in self.vars:
            self.__dict__[f"tkvar_{key}"] = tk.StringVar(self.panel, value=str(self.vars[key]))

        #自动布局
        initfunc(self, *args, **kwargs)

    return tkinit 


def tkregister(self, master):
    #首次建立
    if not self.panel: 
        #面板
        self.panel = tk.PanedWindow(master)
        self.tkpan_rows = [tk.PanedWindow(self.panel)]

        #摘要字符 展开/隐藏控件
        self.tklab_str = tk.Label(self.tkpan_rows[0], textvariable=self.tkvar_str, wraplength=280)  
        self.tkbtn_view = tk.Button(self.tkpan_rows[0], textvariable=self.tkvar_view, command=self.layout, height=1)
        self.tkbtn_redr = tk.Button(self.tkpan_rows[0], text="↻", command=self.redraw, height=1)
        self.tkvar_view.set(self.view)

        #可视化控件
        if self.pltfuncs:
            self.plt_fig = Figure(figsize=self.pltsize)
            self.plt_axis = self.plt_fig.add_subplot()
            self.plt_axis.margins(0, 0)
            self.tkpan_rows.append(tk.PanedWindow(self.panel))
            self.plt_cvas = FigureCanvasTkAgg(self.plt_fig, master=self.tkpan_rows[-1])
        
        #变量控件
        for key in self.vars:
            self.tkpan_rows.append(tk.PanedWindow(self.panel))
            self.__dict__[f"tklab_{key}"] = tk.Label(self.tkpan_rows[-1], text=f"{key}")
            self.__dict__[f"tkety_{key}"] = tk.Entry(self.tkpan_rows[-1], textvariable=self.__dict__[f"tkvar_{key}"])
            #绘图回调
            self.__dict__[f"tkety_{key}"].bind("<Enter>", self.redraw)
        
        #子项行(倒序使用)
        for prec in self.precs:
            self.tkpan_rows.append(tk.PanedWindow(self.panel))
            

def tklayout(self, event="<Enter>", side=tk.TOP):
    self.__str__()

    #展开
    if self.tkvar_view.get() == "+":
        self.panel.pack(side=side)

        [pan.pack(side=tk.TOP) for pan in self.tkpan_rows]
        self.tklab_str.pack(side=tk.LEFT)
        self.tkbtn_view.pack(side=tk.LEFT)

        if self.pltfuncs: 
            self.tkbtn_redr.pack(side=tk.LEFT)
            self.plt_cvas.get_tk_widget().pack(side=tk.TOP)

        for key in self.vars:
            self.__dict__[f"tklab_{key}"].pack(side=tk.LEFT)
            self.__dict__[f"tkety_{key}"].pack(side=tk.LEFT)
    
        self.tkvar_view.set("-")

    #折叠
    elif self.tkvar_view.get() == "-":
        self.panel.pack(side=side)

        self.tkpan_rows[0].pack(side = tk.TOP)
        self.tklab_str.pack(side=tk.LEFT)
        self.tkbtn_view.pack(side=tk.LEFT)
        if self.pltfuncs: self.tkbtn_redr.pack(side=tk.LEFT)

        for pan in self.tkpan_rows[1:]:
            pan.pack_forget()
            
        self.tkvar_view.set("+")
        

def tklayout_left(self, event="<Enter>"):
    tklayout(self, event=event, side=tk.LEFT)


def tkrepr_dec(reprfunc):
    def tkrepr(self, master):
        """注册可视化面板"""
        #显示控件
        self.regist(master=master)
        self.layout()
        self.redraw()
        
        #子项注册
        for i, prec in enumerate(self.precs[::-1], 1):  #从后向前绘制,顺序一致
            prec.regist(self.tkpan_rows[-i]) 
            prec.layout()
            prec.redraw()

        reprfunc(self)
    return tkrepr

def tkunrepr(self):
    for prec in self.precs:
        tkunrepr(prec) 

    if self.panel:
        self.panel.destroy()
        self.panel = None 
    return tkunrepr

def tkstr_dec(strfunc):
    """声明复合变量"""
    def tkstr(self):
        self.tkvar_str.set(strfunc(self))
        return self.tkvar_str.get()
    return tkstr

def tkredraw(self, event="<Enter>"):
    #字符更新
    self.__str__()

    #绘制更新
    if self.pltfuncs and self.plt_cvas.get_tk_widget().winfo_ismapped():
        self.plt_axis.cla()
        for pltfunc in self.pltfuncs:
            pltfunc(self, self.plt_axis, reg=context.view_reg)
        self.plt_cvas.draw()

    #对后继项的重绘
    # [succ.redraw() for succ in self.succs]
    return 



#-------------------------------------------------------------------------------------#
# 绘制函数
#-------------------------------------------------------------------------------------#


def tkprocess(self, end=""):
    if not self.on: return 
    time_str = f"{np.round(time.time() - self.start_time, 3)}s"
    proc_str = f"{self.cur_bat}/{self.n_bat}"
    return time_str, proc_str

def tkstart(self, n_bat):
    if not self.on: return 
    self.start_time = time.time()

    self.n_bat = n_bat
    self.cur_bat = 0 
    end = "\n" if self.cur_bat >= self.n_bat else ""
    time_str, proc_str = self.print_process(end=end)

    self.tkbar = ttk.Progressbar(context.state_panel, orient=tk.HORIZONTAL, length=200, 
            mode="determinate", maximum=n_bat)
    self.tkbar.pack(side=tk.LEFT)
    # self.tkbtn_bar = tk.Button(context.state_panel, text="×", command=self.end_proc)
    # self.tkbtn_bar.pack(side=tk.LEFT)
    self.tkvar_bar = tk.StringVar(window, value=f"{proc_str} {time_str}")
    self.tklab_bar = tk.Label(context.state_panel, textvariable=self.tkvar_bar)
    self.tklab_bar.pack(side=tk.LEFT)

def tkinc(self):
    if not self.on: return 
    self.cur_bat += 1
    end = "\n" if self.cur_bat >= self.n_bat else ""
    time_str, proc_str = self.print_process(end=end)
    
    self.tkbar["value"] = self.cur_bat
    self.tkvar_bar.set(f"{proc_str} {time_str}") 
    window.update() #刷新界面

def tkend(self):
    if not self.on: return 
    self.end_time = time.time()
    time_str, proc_str = self.print_process()
    print(f"{proc_str} {time_str}")

    self.tkbar.pack_forget()
    self.tklab_bar.pack_forget()
    self.tkbar.destroy()
    self.tklab_bar.destroy()

ProcessBar.print_process = tkprocess
ProcessBar.start = tkstart
ProcessBar.inc = tkinc
ProcessBar.end = tkend


def plt_vol(ax, vol, cb=False, legend={}):
    """绘制分布密度 支持多维分布"""
    if not np.any(vol): return 

    mask = np.array(vol.shape)==1
    axis = np.arange(vol.ndim)[mask][0] if np.any(mask) else context.view_cfg.axis

    #计算截面
    cut_axis_len = vol.shape[axis]
    cut_idx = min(int(cut_axis_len * context.view_cfg.rpos), cut_axis_len-1)
    key = [slice(None) if ax != axis else cut_idx for ax in range(3)]
    cut = np.copy(vol[tuple(key)]) 

    #轴方向
    sort_cut = cut[::-1] if axis == 1 else cut.T[::-1]

    #处理极端值
    if np.max(vol) < np.inf and np.min(vol) > -np.inf:
        ax.imshow(sort_cut, vmin=np.min(vol), vmax=np.max(vol))
    else: 
        ax.imshow(sort_cut)

    #自动轴标注
    ax.set_xticks([])
    ax.set_yticks([])
    ax_lab = {1: ("z", "x", "y"), 0: ("y", "z", "x"), 2: ("x", "y", "z")}[axis]
    ax.set_xlabel(ax_lab[0], loc="right", fontsize=8)
    ax.set_ylabel(ax_lab[1], loc="top", fontsize=8)
    ax.set_title(f"{ax_lab[1]}-{ax_lab[0]} {ax_lab[2]}[{cut_idx}] {vol.shape}", fontsize=8)

    if cb: ax.rcParams['font.size'] = 8; ax.colorbar()
    if legend: plt_legend(vol, legend)
    return 


def plt_legend(ax, vol, legend: dict={}):
    if not np.any(vol): return 
    if np.unique(vol).size > 10:
        warnings.warn(f"having {np.unique(vol).size} > 10 attrs, legend plot passed")
        return 

    #绘制图例
    marks = np.unique(vol).astype(np.int32)
    if marks.max()-marks.min() == 0:
        colors = plt.get_cmap('viridis')([0.0])
    else: 
        colors = plt.get_cmap('viridis')((marks - marks.min())/(marks.max()-marks.min()))  #按照数值范围映射

    #自动补充图例
    if not legend: legend = {mark :f"Phs{mark}" for mark in marks}
    
    for mark, color in zip(marks, colors):
        if mark == 0: continue
        if mark not in legend: 
            warnings.warn(f"plt legend warn: mark {mark} has no legend")
            continue 
        ax.scatter([],[], s=32, label=f"{legend[mark]}", color=color)
    
    ax.legend(scatterpoints=1, fontsize=6, facecolor="white", edgecolor="none", framealpha=1, frameon=True)


def plt_view_vol(obj, ax, reg):
    """绘制相 采用缓存"""
    plt_vol(ax, obj.view_vol(reg)[:], cb=False)
    plt_legend(ax, obj.view_vol(reg)[:])


def plt_dist(dist, ax, reg=None, nbin=1000, smth=1):
    """绘制分布密度 支持多维分布"""
    rng = (dist.xmax - dist.xmin)
    x0 = np.linspace(dist.xmin-0.1*rng, dist.xmax+0.1*rng, nbin)

    x = np.repeat(x0.reshape(-1, 1), dist.n_ftr, axis=1)
    p = np.convolve(dist.density(x),  np.full(smth, 1/smth), mode="same")
    
    #计算有效响应部分
    response_mask = p > np.max(p)*0.02
    if not np.any(response_mask):
        warnings.warn("dist_func_plt failed: empty sample density")
        ax.plot(x0, np.full(x0.size, 0))
        return 
    
    #绘制有效部分
    xmin = np.min(x0[response_mask]) - 0.02
    xmax = np.max(x0[response_mask]) + 0.02
    ax.set_xlim([xmin, xmax])
    ax.tick_params(labelsize=8)
    ax.plot(x0, p) 
    return 

def plt_compdist(compdist, ax, reg=None, nbin=600, smth=1):
    for df in compdist.dfs:
        plt_dist(df, ax, reg=reg, nbin=nbin, smth=smth)


#-------------------------------------------------------------------------------------#
# PCG UI封装
#-------------------------------------------------------------------------------------#

class TKUI():
    """UI封装基类"""
    view = "+"
    __unrepr__ = tkunrepr
    regist = tkregister
    redraw = tkredraw
    layout = tklayout

    @property 
    def precs(self): return []
    @property 
    def succs(self): return []

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()
        for key in state:
            exec(f"self.{key} = state[key]")
        self.update()
    
    def update(self):
        return 
    
    def check_update(self):
        return 
    
    def modified(self):
        return True


def exec_str_ret(vstr, env=None):
    """执行字符串 返回结果"""
    # print(vstr)
    # print(type(vstr))
    if not env: env = locals()
    try: 
        exec(f"_ = {vstr}")
    except NameError: return None 
    return env["_"]


def dist_init(self, *params):  #定义分布模型
    if len(params): self.params = params
    self._data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32) #每次拟合只添加 不覆盖
    self.update()

cls_vars={"params": (0.02, 0.02), "stat_model": "uniform"}

def dist_str(self) -> str:
    if not np.size(self.params): 
        return f"{self.stat_model}(pm:{str(self.params)})"
    elif type(self.params[0]) in (np.integer, np.floating):
        return f"{self.stat_model}(pm:{np.round(self.params[:3], 3)})"
    else: 
        return f"{self.stat_model}(pm:{str(self.params)[:40]}..)"

class TKDistFunc(DistFunc, TKUI):
    __init__ = tkinit_dec(dist_init, vars=cls_vars, pltfuncs=[plt_dist], pltsize=(3, 1.5))
    __repr__ = tkrepr_dec(auto_check(DistFunc.__repr__))
    __str__ = tkstr_dec(auto_check(dist_str))
    run_key = lambda self: (tuple(self.params), self.stat_model)
    for key in cls_vars:
        if key == "stat_model": continue  #字符串不解析
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    def dec_key(self) -> tuple:  #只根据params本身判断
        return *tuple(self.params), 

    def run_key(self) -> tuple:
        return *tuple(self.params), id(self._data)
    
    @auto_check
    def __getstate__(self):
        return {"params": self.params, "stat_model": self.stat_model}
    
    def __setstate__(self, state):
        TKUI.__setstate__(self, state)
        self.params = tuple(state["params"])
        self.stat_model = state["stat_model"]

    @property
    def stat_model(self):
        return self.tkvar_stat_model.get()
    
    @stat_model.setter 
    def stat_model(self, value):
        return self.tkvar_stat_model.set(value)

    @property
    def model(self):
        if self.stat_model == "uniform":
            return UniformDist.model
        elif self.stat_model == "expon":
            return ExponDist.model
        elif self.stat_model == "beta":
            return BetaDist.model
        elif self.stat_model == "norm":
            return NormDist.model
        

uni = TKDistFunc(0.0, 1.0)


def init_compdist(self, dfs=[TKDistFunc(-1.0, 2.0), TKDistFunc(-1.0, 2.0), TKDistFunc(-1.0, 2.0)]):
    self.dfs = dfs
    self.n_ftr = len(dfs)
    self._data = np.zeros((0, self.n_ftr))
    self.xmin = -1.0
    self.xmax = 1.0
    self.update()

cls_vars = {"df0_ref": None, "df1_ref": None, "df2_ref": None, }

class TKCompDist(CompDist, TKUI):
    """"""
    __init__ = tkinit_dec(init_compdist, vars=cls_vars, pltfuncs=[plt_compdist], pltsize=(3, 1.5))
    __repr__ = tkrepr_dec(CompDist.__repr__)
    __str__ = tkstr_dec(CompDist.__str__)

    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    @property
    def precs(self):
        return [self.df0, self.df1, self.df2]
    
    @property
    def df0(self):
        return self.df0_ref or self._df0 
    @df0.setter 
    def df0(self, value):
        self._df0 = value 
    @property
    def df1(self):
        return self.df1_ref or self._df1
    @df1.setter 
    def df1(self, value):
        self._df1 = value 
    @property
    def df2(self):
        return self.df2_ref or self._df2
    @df2.setter 
    def df2(self, value):
        self._df2 = value 
    @property
    def dfs(self):
        return [self.df0, self.df1, self.df2]
    @dfs.setter
    def dfs(self, value):
        self.df0 = value[0]
        self.df1 = value[1]
        self.df2 = value[2]

    @auto_check
    def __getstate__(self):
        return {"params": self.params, "n_ftr": self.n_ftr, "dfs": self.dfs, 
                "df0": self.df0, "df1": self.df1, "df2": self.df2, 
                "df0_ref": self.tkvar_df0_ref.get(), "df1_ref": self.tkvar_df1_ref.get(), "df2_ref": self.tkvar_df2_ref.get()}
    
    def __setstate__(self, state):
        TKUI.__setstate__(self, state)
        self.dfs = list(state["dfs"])
        self.n_ftr = len(self.dfs)
        self.params = tuple(state["params"])
        self._data = np.ndarray(shape=(0, self.n_ftr), dtype=np.float32)
        self.update()


cls_vars={"res": 1.0, "size": [80.0, 80.0, 80.0], "periods": [False, False, False]}

class TKRegion(Region, TKUI):
    vlz = Voxelizer(50, 3)

    __init__ = tkinit_dec(Region.__init__, vars=cls_vars)
    __repr__ = tkrepr_dec(Region.__repr__)
    __str__ = tkstr_dec(Region.__str__)
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")
    
    @auto_check
    def __getstate__(self):
        return {"res": self.tkvar_res.get(), "size": self.tkvar_size.get(), "periods": self.tkvar_periods.get()}
    __setstate__ = TKUI.__setstate__

reg = TKRegion()


#NOTE region2d只允许传入前两个维度, 第三个维度根据分辨率确定

cls_vars={"res": 1.0, "size_2d": [80.0, 80.0], "periods_2d": [False, False]}

class TKRegion2D(Region2D, TKUI):
    vlz = Voxelizer(50, 2)

    __init__ = tkinit_dec(Region2D.__init__, vars=cls_vars)
    __repr__ = tkrepr_dec(Region2D.__repr__)
    __str__ = tkstr_dec(Region2D.__str__)
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")
    
    @auto_check
    def __getstate__(self):
        return {"res": self.tkvar_res.get(), "size_2d": self.tkvar_size_2d.get(), "periods_2d": self.tkvar_periods_2d.get()}
    __setstate__ = TKUI.__setstate__



def phs_init(self, r_dist=TKDistFunc(), v_frac: float=1.0, mark: int=1, name=""):
    Phase.__init__(self, r_dist, v_frac, mark, name)

cls_vars={"v_frac": 1.0, "mark": 1, "r_dist_ref": None}

class TKPhase(Phase, TKUI):
    __init__ = tkinit_dec(phs_init, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(auto_check(Phase.__repr__))
    __str__ = tkstr_dec(auto_check(Phase.__str__))
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    @property
    def r_dist(self):
        return self.r_dist_ref or self._r_dist 
    @r_dist.setter 
    def r_dist(self, value):
        self._r_dist = value 

    @auto_memo
    def view_vol(self, reg):
        return self.sample(reg).discrete(reg)

    @property
    def precs(self):
        return [self.r_dist]
    
    @auto_check
    def __getstate__(self):
        return {"v_frac": self.tkvar_v_frac.get(), "mark": self.tkvar_mark.get(), "r_dist": self.r_dist,
            "r_dist_ref": self.tkvar_r_dist_ref.get()}
    __setstate__ = TKUI.__setstate__



def aniphs_init(self, r_dist=TKDistFunc(), ab_dist=TKDistFunc(1.0, 0.0), 
                ort_dist=TKCompDist([TKDistFunc(-1.0, 2.0), TKDistFunc(-1.0, 2.0), TKDistFunc(-1.0, 2.0)]), 
                v_frac: float=1.0, mark: int=1, name=""):
    AnisoPhase.__init__(self, r_dist, ab_dist, ort_dist, v_frac, mark, name)

cls_vars={"v_frac": 1.0, "mark": 1, "r_dist_ref": None, "ab_dist_ref": None, "ort_dist_ref": None}

class TKAniPhase(AnisoPhase, TKUI):
    __init__ = tkinit_dec(aniphs_init, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(AnisoPhase.__repr__)
    __str__ = tkstr_dec(AnisoPhase.__str__)
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    @property
    def r_dist(self):
        return self.r_dist_ref or self._r_dist 
    @r_dist.setter 
    def r_dist(self, value):
        self._r_dist = value 
    @property
    def ab_dist(self):
        return self.ab_dist_ref or self._ab_dist 
    @ab_dist.setter 
    def ab_dist(self, value):
        self._ab_dist = value 
    @property
    def ort_dist(self):
        return self.ort_dist_ref or self._ort_dist 
    @ort_dist.setter 
    def ort_dist(self, value):
        self._ort_dist = value 

    @auto_memo
    def view_vol(self, reg):
        return self.sample(reg).discrete(reg)

    @property
    def precs(self):
        return [self.r_dist, self.ab_dist]
    
    @auto_check
    def __getstate__(self):
        return {"v_frac": self.tkvar_v_frac.get(), "mark": self.tkvar_mark.get(), 
                "r_dist": self.r_dist, "ab_dist": self.ab_dist, "ort_dist": self.ort_dist, 
                "r_dist_ref": self.tkvar_r_dist_ref.get(), "ab_dist_ref": self.tkvar_ab_dist_ref.get(), "ort_dist_ref": self.tkvar_ort_dist_ref.get()}
    __setstate__ = TKUI.__setstate__


cls_vars = {"phss_ref": []}

class TKCompPhase(CompPhase, TKUI):
    __init__ = tkinit_dec(CompPhase.__init__, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(CompPhase.__repr__)
    __str__ = tkstr_dec(CompPhase.__str__)
    # view_vol = TKPhase.view_vol 

    # def regist(self, master):
    #     tkregister(self, master)
    #     self.tkpan_rows.append(tk.PanedWindow(self.panel))
    #     self.tklab_comp = tk.Label(self.tkpan_rows[-1], text="phss")
    #     self.tklst_comp = tk.Listbox(self.tkpan_rows[-1], width=45, height=4)

    # def layout(self):
    #     tklayout(self)
    #     self.tklab_comp.pack()
    #     self.tklst_comp.pack()

    # def redraw(self, event="<Enter>"):
    #     tkredraw(self)
    #     self.tklst_comp.delete(0, tk.END)
    #     for phs in self.phss: 
    #         self.tklst_comp.insert(tk.END, phs.__str__())
    
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    @property
    def phss(self):
        return self.phss_ref or self._phss
    @phss.setter 
    def phss(self, value):
        self._phss = value 

    @auto_check
    def __getstate__(self):
        return {"phss": self.phss, "phss_ref": self.tkvar_phss_ref.get()}
    __setstate__ = TKUI.__setstate__

    @auto_memo
    def view_vol(self, reg):
        return self.sample(reg).discrete(reg)


cls_vars={"rpos": 1.0, "rfix": 0.0, "shape_model": "seeds"}

class TKAlphaShape(AlphaShape, TKUI):
    __init__ = tkinit_dec(AlphaShape.__init__, vars=cls_vars)
    __repr__ = tkrepr_dec(AlphaShape.__repr__)
    __str__ = tkstr_dec(AlphaShape.__str__)
    run_key = lambda self: (self.rpos, self.rfix, self.shape_model)
    for key in cls_vars:
        if key == "shape_model": continue 
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")
    
    @auto_check
    def __getstate__(self):
        return {"rpos": self.tkvar_rpos.get(), "rfix": self.tkvar_rfix.get(), "shape_model": self.shape_model}
    __setstate__ = TKUI.__setstate__

    def a_shape(self, *args, **kwargs):
        if self.shape_model == "alpha":
            return AlphaShape.a_shape(self, *args, **kwargs)
        elif self.shape_model == "seeds":
            return SeedsShape.a_shape(self, *args, **kwargs)
        elif self.shape_model == "inva":
            return InvAShape.a_shape(self, *args, **kwargs)
        elif self.shape_model == "poly":
            return PolyShape.a_shape(self, *args, **kwargs)

    @property
    def shape_model(self):
        return self.tkvar_shape_model.get()
    
    @shape_model.setter 
    def shape_model(self, value):
        return self.tkvar_shape_model.set(value)


@auto_memo 
def view_vol(self, reg):
    return self.discrete(reg)

Object.view_vol = view_vol


def pobj_init(self, phs=None, ashp=None):
    if not phs: phs = TKPhase()
    if not ashp: ashp = TKAlphaShape()
    PhaseObject.__init__(self, phs, ashp)
    
cls_vars={"phs_ref": None, "ashp_ref": None}

class TKPhasePnt(PhaseObject, TKUI):
    __init__ = tkinit_dec(pobj_init, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(PhaseObject.__repr__)
    __str__ = tkstr_dec(PhaseObject.__str__)
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    @property 
    def precs(self): return [self.phs, self.ashp]
    
    @property
    def phs(self):
        return self.phs_ref or self._phs
    @phs.setter 
    def phs(self, value):
        self._phs = value 

    @property
    def ashp(self):
        return self.ashp_ref or self._ashp
    @ashp.setter 
    def ashp(self, value):
        self._ashp = value 

    @auto_check
    def __getstate__(self):
        return {"phs": self.phs, "ashp": self.ashp, 
            "phs_ref": self.tkvar_phs_ref.get(), "ashp_ref": self.tkvar_ashp_ref.get()}
    __setstate__ = TKUI.__setstate__


cls_vars = {"axis": 0, "rpos": (0.0, 0.5), "mark": 1}

class TKFillBound(FillBound, TKUI):
    __init__ = tkinit_dec(Node.update, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(FillBound.__repr__)
    __str__ = tkstr_dec(FillBound.__str__)
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")
    
    def __getstate__(self):
        return {"axis": self.tkvar_axis.get(), "rpos": self.tkvar_rpos.get(), "mark": self.tkvar_mark.get()}
    __setstate__ = TKUI.__setstate__
    

cls_vars = {"axis": 0, "rpos": (0.5, 0.5), "r": 0.1, "h": (0.0, 1.0), "mark": 1}

class TKCylinder(Cylinder, TKUI):
    __init__ = tkinit_dec(Node.update, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(Cylinder.__repr__)
    __str__ = tkstr_dec(Cylinder.__str__)
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    def __getstate__(self):
        return {"axis": self.tkvar_axis.get(), "rpos": self.tkvar_rpos.get(), "r": self.tkvar_r.get(), "h": self.tkvar_h.get(), "mark": self.tkvar_mark.get()}
    __setstate__ = TKUI.__setstate__



cls_vars = {"fname": "data/img.png", "ths": [], "n_smth": 1, 
            "cuts": "(slice(None), slice(None), slice(None))", "attr": 1}

class TKFileImg(FileImg, TKUI):
    __init__ = tkinit_dec(FileImg.__init__, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(FileImg.__repr__)
    __str__ = tkstr_dec(FileImg.__str__)

    for key in cls_vars:
        if key == "fname": continue
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")
        
    def regist(self, master):
        tkregister(self, master)
        self.tkvar_vf = tk.StringVar(window)
        self.tkpan_rows.append(tk.PanedWindow(self.panel))
        self.tklab_vf = tk.Label(self.tkpan_rows[-1], textvariable=self.tkvar_vf)

        self.tkpan_rows.append(tk.PanedWindow(self.panel))
        self.tkbtn_rec = tk.Button(self.tkpan_rows[-1], text="recording", width=20, command=self.recording)

    def layout(self):
        tklayout(self)
        self.tklab_vf.pack(side = tk.TOP)
        self.tkbtn_rec.pack(side = tk.TOP)

    def redraw(self, event="<Enter>"):
        tkredraw(self)
        if isinstance(self.img, ObjVol):
            vf = self.img.vol_frac
            for attr in vf:
                vf[attr] = np.round(vf[attr], 3)
            self.tkvar_vf.set(f"vf: {str(vf)}")

    @property
    def fname(self):
        return self.tkvar_fname.get()

    @fname.setter
    def fname(self, value):
        return self.tkvar_fname.set(value)

    def __getstate__(self):
        return {"fname": self.fname, "ths": self.tkvar_ths.get(), "n_smth": self.tkvar_n_smth.get(), 
                "cuts": str(self.cuts)}
    __setstate__ = TKUI.__setstate__

    @auto_check
    def recording(self):
        if not self.__dict__.get("operating", False):
            self.operating = True
            img_name = context.get_name(self)
            context.exec_cmd(cmd=f"{img_name}_phs{self.attr} = TKImage(TKPhaseVol({img_name}.img[:]=={self.attr}))")
            print(f"{img_name} recording => {img_name}_phs{self.attr}")
            self.operating = False 
        return 

    @auto_check
    def view_vol(self, reg):
        return self.img 
    
    # @auto_vol 
    # def __eq__(self, value):
    #     return TKImage(self.vol.__eq__(value))
        
    # @auto_vol 
    # def __ne__(self, value):
    #     return TKImage(self.vol.__ne__(value))


@auto_vol 
def imgobj_init(self, img = PhaseVol.zeros([10, 10, 10])):
    """图像物体(单相物体)初始化"""
    self.img = TKPhaseVol(img)
    Node.__init__(self)
    
cls_vars = {"img_ref": None}

class TKImage(ImgObject, TKUI):
    __init__ = tkinit_dec(imgobj_init, vars=cls_vars, pltfuncs=[])
    __repr__ = tkrepr_dec(ImgObject.__repr__)
    __str__ = tkstr_dec(ImgObject.__str__)

    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    def regist(self, master):
        tkregister(self, master)
        self.tkvar_vf = tk.StringVar(window)
        self.tkpan_rows.append(tk.PanedWindow(self.panel))
        self.tklab_vf = tk.Label(self.tkpan_rows[-1], textvariable=self.tkvar_vf)

    def layout(self):
        tklayout(self)
        self.tklab_vf.pack(side = tk.TOP)

    def redraw(self, event="<Enter>"):
        tkredraw(self)
        if isinstance(self.img, ObjVol):
            vf = self.img.vol_frac
            for attr in vf:
                vf[attr] = np.round(vf[attr], 3)
            self.tkvar_vf.set(f"vf: {str(vf)}")

    @property
    def precs(self):
        return [self.img]

    def __getstate__(self):
        return {"img": self.img, "img_ref": self.tkvar_img_ref.get()}
    def __setstate__(self, state):
        TKUI.__setstate__(self, state)
        self.img = TKPhaseVol(state["img"])
        self.tkvar_img_ref.set(state["img_ref"])

    @auto_check
    def view_vol(self, reg):
        return self.vol 

    @property
    def img(self):
        return self.img_ref or self._img

    @img.setter 
    def img(self, value):
        self._img = value



def compobj_init(self, objs=[]):
    self.objs = objs 
    self.reg = TKRegion()
    ProcessBar.__init__(self)
    self.update()

cls_vars = {"objs_ref": [], "reg_ref":None, "depth": 0}

class TKCompObject(CompObject, ProcessBar, TKUI):
    __init__ = tkinit_dec(compobj_init, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(CompObject.__repr__)
    __str__ = tkstr_dec(CompObject.__str__)

    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    # def regist(self, master):
    #     tkregister(self, master)
    #     self.tkpan_rows.append(tk.PanedWindow(self.panel))
    #     self.tklab_comp = tk.Label(self.tkpan_rows[-1], text="objs")
    #     self.tklst_comp = tk.Listbox(self.tkpan_rows[-1], width=45, height=4)

    # def layout(self):
    #     tklayout(self)
    #     self.tklab_comp.pack(side=tk.TOP)
    #     self.tklst_comp.pack(side=tk.TOP)

    # def redraw(self, event="<Enter>"):
    #     tkredraw(self)
    #     self.tklst_comp.delete(0, tk.END)
    #     for obj in self.objs: 
    #         self.tklst_comp.insert(tk.END, obj.__str__())

    def regist(self, master):
        tkregister(self, master)
        self.tkpan_rows.append(tk.PanedWindow(self.panel))
        self.tkbtn_msh = tk.Button(self.tkpan_rows[-1], text="discreting", width=20, command=self.discreting)

    def layout(self):
        tklayout(self)
        self.tkbtn_msh.pack(side = tk.TOP)

    def redraw(self, event="<Enter>"):
        tkredraw(self)

    @auto_check 
    def discreting(self):
        if not self.__dict__.get("operating", False):
            self.operating = True
            # self.start(2)
            obj_name = context.get_name(self)
            context.exec_cmd(cmd=f"{obj_name}_vol = TKPhaseVol.from_vol(\
                                {obj_name}.discrete({obj_name}.reg)\
                                .open_smooth({self.depth}))")
            # self.inc()
            print(f"{obj_name} discreting => {obj_name}_vol")
            # self.end()
            self.operating = False 
        return  

    def dec_key(self):
        return *tuple([obj.dec_key() for obj in self.objs]), self.reg.dec_key()

    def run_key(self):
        return *tuple([obj.run_key() for obj in self.objs]), self.reg.run_key()

    @property 
    def precs(self): 
        return [self.reg]

    @auto_memo
    def view_vol(self, reg):
        return self.discrete(reg)

    @property
    def objs(self):
        return self.objs_ref or self._obj
    @objs.setter
    def objs(self, value):
        self._obj = value 

    @property
    def reg(self):
        return self.reg_ref or self._reg
    @reg.setter
    def reg(self, value):
        self._reg = value 

    @auto_check
    def __getstate__(self):
        return {"objs": self.objs, "objs_ref": self.tkvar_objs_ref.get(), 
            "reg": self.reg, "reg_ref": self.tkvar_reg_ref.get()}
    __setstate__ = TKUI.__setstate__



#-------------------------------------------------------------------------------------#
# 网格对象
#-------------------------------------------------------------------------------------#

#NOTE 需要先通过默认初始化函数创建对象, 再对其进行赋值
#NOTE python默认用repr代替str, 若没有str定义的类, 将会提前调用repr进行UI注册导致错误

def tetmesh_init(self, *args, **kwargs):
    TetMesh.__init__(self, *args, **kwargs)
    ProcessBar.__init__(self)
    self.update()

cls_vars = {"dir": "meshes", "wfmt": "vtk"}

class TKTetMesh(TetMesh, ProcessBar, TKUI):
    """四面体网格"""
    __init__ = tkinit_dec(tetmesh_init, vars=cls_vars)
    __repr__ = tkrepr_dec(TetMesh.__repr__)
    __str__ = tkstr_dec(TetMesh.__str__)
    
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    def regist(self, master):
        tkregister(self, master)
        self.tkvar_vf = tk.StringVar(window)

        self.tkpan_rows.append(tk.PanedWindow(self.panel))
        self.tkbtn_dir = tk.Button(self.tkpan_rows[1], text="...", width=3, command=self.askdir)

        self.tkpan_rows.append(tk.PanedWindow(self.panel))
        self.tklab_vf = tk.Label(self.tkpan_rows[-1], textvariable=self.tkvar_vf)

        self.tkpan_rows.append(tk.PanedWindow(self.panel))
        self.tkbtn_msh = tk.Button(self.tkpan_rows[-1], text="write", width=20, command=self.writing)

    def layout(self):
        tklayout(self)
        self.tklab_vf.pack(side = tk.TOP)
        self.tkbtn_msh.pack(side = tk.TOP)
        self.tkbtn_dir.pack(side = tk.RIGHT)

    def redraw(self, event="<Enter>"):
        tkredraw(self)
        vf = self.vol_frac
        for attr in vf:
            vf[attr] = np.round(vf[attr], 3)
        self.tkvar_vf.set(f"vf: {str(vf)}")

    @classmethod
    def from_mesh(cls, value):
        tetmesh = cls()
        tetmesh.verts = value.verts
        tetmesh.cells = value.cells
        tetmesh.vert_data = value.vert_data
        tetmesh.cell_data = value.cell_data
        print(f"meshing completed")
        return tetmesh
    
    def writing(self):
        if not self.__dict__.get("operating", False):
            self.operating = True
            # self.start(2)
            name = context.get_name(self)
            mesh_fmt = self.tkvar_wfmt.get()
            self.mesh_dir = self.tkvar_dir.get()
            
            context.exec_cmd(cmd=f"{name}.save_mesh('{self.mesh_dir}/{name}.{mesh_fmt}')")
            # self.inc()
            print(f"{name} writed => '{self.mesh_dir}/{name}.{mesh_fmt}'")
            # self.end()
            self.operating = False 
        return  
    
    def askdir(self):
        dir = filedialog.askdirectory()
        if dir: 
            self.tkvar_dir.set(dir)
            context.relist_file()

    def update(self):
        return 

def phsvol_init(self, *args, **kwargs):
    PhaseVol.__init__(self, *args, **kwargs)
    ProcessBar.__init__(self)
    self.update()

cls_vars = {"depth": 2, "vsub": 0.1, "sigma": 1.0}

class TKPhaseVol(PhaseVol, ProcessBar, TKUI):
    """体素网格"""
    __init__ = tkinit_dec(phsvol_init, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(PhaseVol.__repr__)
    __str__ = tkstr_dec(PhaseVol.__str__)

    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    def view_vol(self, reg):
        return PhaseVol(self.vol)
    
    @classmethod
    def from_vol(cls, value):
        vol = cls()
        vol.vol = PhaseVol(value).vol
        print("discreting completed")
        return vol

    def regist(self, master):
        tkregister(self, master)
        self.tkvar_vf = tk.StringVar(window)

        self.tkpan_rows.append(tk.PanedWindow(self.panel))
        self.tkbtn_msh = tk.Button(self.tkpan_rows[-1], text="meshing", width=20, command=self.meshing)

        self.tkpan_rows.append(tk.PanedWindow(self.panel))
        self.tklab_vf = tk.Label(self.tkpan_rows[-1], textvariable=self.tkvar_vf)

    def layout(self):
        tklayout(self)
        self.tkbtn_msh.pack(side = tk.TOP)
        self.tklab_vf.pack(side = tk.TOP)

    def redraw(self, event="<Enter>"):
        vf = self.vol_frac
        for attr in vf:
            vf[attr] = np.round(vf[attr], 3)
        self.tkvar_vf.set(f"vf: {str(vf)}")
        tkredraw(self)

    @auto_check 
    def meshing(self):
        if not self.__dict__.get("operating", False):
            self.operating = True
            obj_name = context.get_name(self)
            context.exec_cmd(cmd=f"{obj_name}_mesh = TKTetMesh.from_mesh(\
                                TetMesh.from_vol({obj_name}\
                                .open_smooth({self.depth}), \
                                vsub={self.vsub}, sigma={self.sigma}))")
            print(f"{obj_name} meshing => {obj_name}_mesh")
            self.operating = False 
        return  

    @auto_check
    def __getstate__(self):
        return {"vol": self.vol, "depth": self.tkvar_depth.get(), "vsub": self.tkvar_vsub.get(), 
                "sigma": self.tkvar_sigma.get()}
    
    def __setstate__(self, state):
        """更新 其中vol更新为numpy向量"""
        TKUI.__setstate__(self, state)
        self.vol = np.array(state["vol"], dtype=self.vol_dtype)


def rlt_init(self, srs = None, tgt = None, ftr_func = surdt, dns_func = None):
    if not srs: srs = TKPhasePnt()
    if not tgt: tgt = TKPhasePnt()
    if not dns_func: dns_func = TKDistFunc()
    ObjectRelation.__init__(self, srs, tgt, ftr_func, dns_func)

cls_vars = {"srs_ref": None, "tgt_ref": None, "dns_func_ref": None, "ftr_func_ref": "None"}

class TKRelation(ObjectRelation, TKUI):
    __init__ = tkinit_dec(rlt_init, vars=cls_vars)
    __repr__ = tkrepr_dec(auto_check(ObjectRelation.__repr__))
    __str__ = tkstr_dec(auto_check(ObjectRelation.__str__))
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    @property 
    def precs(self): 
        return [self.dns_func]

    @property
    def srs(self):
        return self.srs_ref or self._srs
    @srs.setter 
    def srs(self, value):
        self._srs = value 

    @property
    def tgt(self):
        return self.tgt_ref or self._tgt
    @tgt.setter 
    def tgt(self, value):
        self._tgt = value 

    @property
    def ftr_func(self):
        return self.ftr_func_ref or self._ftr_func
    @ftr_func.setter 
    def ftr_func(self, value):
        self._ftr_func = value 
    
    @property
    def dns_func(self):
        return self.dns_func_ref or self._dns_func
    @dns_func.setter 
    def dns_func(self, value):
        self._dns_func = value 

    def __getstate__(self):
        return {"srs": self.srs, "tgt": self.tgt, "ftr_func": self.ftr_func, "dns_func": self.dns_func, 
            "srs_ref": self.tkvar_srs_ref.get(), "tgt_ref": self.tkvar_tgt_ref.get(), "dns_func_ref": self.tkvar_dns_func_ref.get(), "ftr_func_ref": self.tkvar_ftr_func_ref.get()}
    __setstate__ = TKUI.__setstate__



def init_pcg(self, phss: list[TKPhase]=[], rlts: list[TKRelation]=[]):
    self.phss = OrderSet(phss) 
    self.rlts = OrderSet(rlts)
    self.reg = TKRegion()
    ProcessBar.__init__(self)
    self.update()

cls_vars = {"phss_ref": [], "rlts_ref": [], "reg_ref": None, "seeds": None}

class TKPCG(PhsRltGraph, TKUI):
    __init__ = tkinit_dec(init_pcg, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(PhsRltGraph.__repr__)
    __str__ = tkstr_dec(PhsRltGraph.__str__)
    
    def dec_key(self) -> tuple:
        return *tuple([rlt for rlt in self.rlts]), self.reg.dec_key()
    
    def run_key(self) -> tuple:
        return *tuple([rlt.run_key() for rlt in self.rlts]), self.reg.run_key()   #以rlt作为key 而非内部保存的sccs

    def regist(self, master):
        tkregister(self, master)
        self.tkpan_rows.append(tk.PanedWindow(self.panel))
        self.tkbtn_smp = tk.Button(self.tkpan_rows[-1], text="sample", width=15, command=self.sample_seeds)
        self.tkbtn_fit = tk.Button(self.tkpan_rows[-1], text="fit", width=15, command=self.fit_seeds)
        self.tkbtn_fill = tk.Button(self.tkpan_rows[-1], text="fill", width=15, command=self.fill_seeds)
        self.tkpan_rows.append(tk.PanedWindow(self.panel))
    def layout(self):
        tklayout(self)
        self.tkbtn_smp.pack(side=tk.LEFT)
        self.tkbtn_fit.pack(side=tk.LEFT)
        self.tkbtn_fill.pack(side=tk.LEFT)

    def redraw(self, event="<Enter>"):
        tkredraw(self)

    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    @auto_memo
    def view_vol(self, reg):
        # surdt.measure = cdt
        vol = self.sample(reg, fast_assp, on=False).discrete(reg)
        # surdt.measure = edt
        return vol

    @property
    def phss(self):
        return self.phss_ref or self._phss
    @phss.setter 
    def phss(self, value):
        self._phss = value 

    @property
    def rlts(self):
        return self.rlts_ref or self._rlts
    @rlts.setter 
    def rlts(self, value):
        self._rlts = value 

    @property 
    def precs(self): 
        return [self.reg]
    
    @property
    def reg(self):
        return self.reg_ref or self._reg
    @reg.setter
    def reg(self, value):
        self._reg = value 

    @auto_check 
    def sample_seeds(self):
        if not self.__dict__.get("operating", False):
            self.operating = True
            pcg_name = context.get_name(self)
            context.exec_cmd(cmd=f"{pcg_name}_seeds = TKSeeds.from_seeds({pcg_name}.sample({pcg_name}.reg, on=True))")
            context.exec_cmd(cmd=f"{pcg_name}.seeds = '{pcg_name}_seeds'")
            print(f"{pcg_name} sampled => {pcg_name}_seeds")
            self.operating = False 
        return  

    @auto_check 
    def fit_seeds(self):
        if not self.__dict__.get("operating", False):
            self.operating = True
            pcg_name = context.get_name(self)
            context.exec_cmd(cmd=f"{pcg_name}_seeds = TKSeeds.from_seeds({pcg_name}.fit({pcg_name}.seeds, {pcg_name}.reg, on=True))")
            # context.exec_cmd(cmd=f"{pcg_name}.seeds = '{pcg_name}_seeds'")
            print(f"{pcg_name} fitted <= {pcg_name}_seeds")
            self.operating = False 
        return  

    @auto_check 
    def fill_seeds(self):
        if not self.__dict__.get("operating", False):
            self.operating = True
            pcg_name = context.get_name(self)
            context.exec_cmd(cmd=f"{pcg_name}_seeds = TKSeeds.from_seeds({pcg_name}.fill({pcg_name}.reg, on=True))")
            context.exec_cmd(cmd=f"{pcg_name}.seeds = '{pcg_name}_seeds'")
            print(f"{pcg_name} filled => {pcg_name}_seeds")
            self.operating = False 
        return  

    @auto_check
    def __getstate__(self):
        return {"phss": self.phss, "rlts": self.rlts, "reg_ref": self.tkvar_reg_ref.get(), "seeds": self.tkvar_seeds.get(), 
            "phss_ref": self.tkvar_phss_ref.get(), "rlts_ref": self.tkvar_rlts_ref.get()}
    __setstate__ = TKUI.__setstate__


class TKSeeds(Seeds, TKUI):
    __init__ = tkinit_dec(Seeds.__init__, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(Seeds.__repr__)
    __str__ = tkstr_dec(Seeds.__str__)

    # @property 
    # def reg(self):
    #     max_loc = np.array([np.max(self.loc[:, i], 0) for i in range(3)])
    #     return TKRegion(np.max(max_loc) / 100, list(max_loc))

    @auto_memo
    def view_vol(self, reg):
        return self.discrete(reg)

    @auto_check
    def __getstate__(self):
        return {"bdata": self.bdata, "idx": self.idx}
    
    def __setstate__(self, state):
        TKUI.__setstate__(self, state)
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



cls_vars = {"seeds_ref": None}

def voro_init(self, seeds=None):
    if not seeds: seeds = TKSeeds()
    self.seeds = seeds 
    self.update() 
    
class TKVoro(LaguerreVoro, TKUI):
    __init__ = tkinit_dec(voro_init, vars=cls_vars, pltfuncs=[plt_view_vol])
    __repr__ = tkrepr_dec(LaguerreVoro.__repr__)
    __str__ = tkstr_dec(LaguerreVoro.__str__)

    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    @auto_memo
    def view_vol(self, reg):
        return self.discrete(reg)
    # @property
    # def precs(self):
    #     return [self.seeds]

    @property 
    def reg(self):
        return self.seeds.reg 
    @property
    def seeds(self):
        return self.seeds_ref or self._seeds
    @seeds.setter 
    def seeds(self, value):
        self._seeds = value

    @auto_check
    def __getstate__(self):
        return {"seeds": self.seeds, "seeds_ref": self.tkvar_seeds_ref.get()}
    __setstate__ = TKUI.__setstate__


#-------------------------------------------------------------------------------------#
# 默认可视化配置
#-------------------------------------------------------------------------------------#


def view_str(self):
    return f"View(ax{self.axis}:[{self.rpos}])"

def view_cfg_init(self, rpos=0.5, axis=2):
    self.rpos = rpos
    self.axis = axis
    self.update()

cls_vars={"rpos": 0.5, "axis": 2}

class ViewConfig(Node, TKUI):
    view = "-"
    __init__ = tkinit_dec(view_cfg_init, vars=cls_vars)
    __repr__ = tkrepr_dec(Node.__repr__)
    __str__ = tkstr_dec(view_str)
    
    layout = tklayout_left  #从左至右 状态栏
    dec_key = lambda self: (self.rpos, self.axis)
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")
    
    @auto_check
    def __getstate__(self):
        return {"rpos": self.tkvar_rpos.get(), "axis": self.tkvar_axis.get()}
    __setstate__ = TKUI.__setstate__



def save_str(self):
    return f"IO({self.dir}/*.{self.fmt})"

def save_cfg_init(self, dir='proj', fmt='json'):
    self.dir = dir
    self.fmt = fmt
    self.update()

cls_vars={"dir": "proj", "fmt": "json"}

class SaveConfig(Node, TKUI):
    view = "-"
    __init__ = tkinit_dec(save_cfg_init, vars=cls_vars)
    __repr__ = tkrepr_dec(Node.__repr__)
    __str__ = tkstr_dec(save_str)

    def regist(self, master):
        tkregister(self, master)
        self.tkbtn_save = tk.Button(self.tkpan_rows[1], text="...", width=3, command=self.askdir)

    def layout(self):
        tklayout_left(self)
        self.tkbtn_save.pack(side=tk.RIGHT)

    def redraw(self, event="<Enter>"):
        tkredraw(self)

    dec_key = lambda self: (self.dir, self.fmt)
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn self.tkvar_{key}.get()")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")

    @auto_check
    def __getstate__(self):
        return {"dir": self.dir, "fmt": self.fmt}
    __setstate__ = TKUI.__setstate__

    def askdir(self):
        dir = filedialog.askdirectory()
        if dir: 
            self.tkvar_dir.set(dir)
            context.relist_file()



def view_reg_init(self, res=1.0, size=[64.0, 64.0, 64.0], periods=[False, False, False]):
    Region.__init__(self, res, size, periods)

cls_vars={"res": 1.0, "size": [64.0, 64.0, 64.0], "periods": [False, False, False]}

class ViewRegion(TKRegion):
    vlz = Voxelizer(50, 3)
    view = "-"
    __init__ = tkinit_dec(view_reg_init, vars=cls_vars)
    layout = tklayout_left
    for key in cls_vars:
        exec(f"@property\ndef {key}(self):\n\treturn exec_str_ret(self.tkvar_{key}.get())")
        exec(f"@{key}.setter\ndef {key}(self, value):\n\treturn self.tkvar_{key}.set(str(value))")



#-------------------------------------------------------------------------------------#
# 保存/加载
#-------------------------------------------------------------------------------------#


__functypes = (type(sum), type(np), type(np.sum), type(DataVol.sum), type(DataVol.ones), type(edt))
__numertypes = (np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, np.complex64, float, int, complex)
__io_fmts = ["pkl", "json", "jcsv"]

def aload(cls, path_str):
    """根据id字符串加载,返回实例"""
    if cls == Context: return 
    fmt = context.save_cfg.fmt 
    dir = context.save_cfg.dir

    #自定义名称
    path_list = path_str.split("/")
    fname_str = path_list[-1]
    name_list = fname_str.split(".")
    fmt = name_list[-1] if len(name_list) > 1 and name_list[-1] in __io_fmts else fmt
    name = name_list[0]
    path = "/".join(path_list[:-1]) if len(path_list)>1 else dir
    path_name = f"{path}/{cls.__name__}-{name}.{fmt}"

    if fmt == "pkl":
        with open(path_name, "rb") as in_file:
            # obj = cls().__setstate__(pickle.load(in_file).__dict__)
            obj = pickle.load(in_file)
    elif fmt in ("json"):
        with open(path_name, "r") as in_file:
            obj = objectlize(json.load(in_file))
    else:
        raise NotImplementedError(f"load object failed with format {fmt}")
    
    context.env.update({name: obj})
    return obj 

def asave(self, path_str=""):
    """默认保存为类名+id字符串"""
    path_str = context.get_name(self)
    dir = context.save_cfg.dir
    fmt = context.save_cfg.fmt 

    #自定义名称
    path_list = path_str.split("/")
    fname_str = path_list[-1]
    name_list = fname_str.split(".")
    fmt = name_list[-1] if len(name_list) > 1 and name_list[-1] in __io_fmts else fmt
    name = name_list[0]
    path = "/".join(path_list[:-1]) if len(path_list)>1 else dir
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
    else:
        raise NotImplementedError(f"save object failed with format {fmt}")
    return 


for __cls in (Object, DistFunc, Phase, ObjectRelation, PhsRltSCC):
    #不建议保存Link对象 其依赖在读取后很可能发生更新而改变了底层数据
    __cls.save = auto_check(asave)
    __cls.load = classmethod(aload)

for __cls in (DataVol, ):
    __cls.save = asave
    __cls.load = classmethod(aload)
    

#-------------------------------------------------------------------------------------#
# 对象保存设置
#-------------------------------------------------------------------------------------#

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



# #文件兼容性
# TKRegion = type("TKRegion", (Region, ), {})
# TKDistFunc = type("TKDistFunc", (UniformDist, ), {})
# TKPhase = type("TKPhase", (Phase, ), {})
# TKImage = type("TKImage", (ImgObject, ), {})
# TKFileImg = type("TKFileImg", (FileImg, ), {})
# TKFillBound = type("TKFillBound", (FillBound, ), {})
# TKCylinder = type("TKCylinder", (Cylinder, ), {})
# TKCompObject = type("TKCompObject", (CompObject, ), {})
# TKAlphaShape = type("TKAlphaShape", (AlphaShape, ), {})
# TKPhasePnt = type("TKPhasePnt", (PhaseObject, ), {})
# TKRelation = type("TKRelation", (ObjectRelation, ), {})
# TKPCG = type("TKPCG", (PhsRltGraph, ), {})
# TKSeeds = type("TKSeeds", (Seeds, ), {})
# TKVoro = type("TKVoro", (LaguerreVoro, ), {})
# tk_classes = ()

for __cls in (TKDistFunc, TKCompDist, TKPhase, TKPhasePnt, TKRelation, TKCompObject, TKFileImg, 
              TKRegion, TKPCG, TKSeeds, TKPhaseVol, TKTetMesh):
    #不建议保存Link对象 其依赖在读取后很可能发生更新而改变了底层数据
    __cls.save = asave
    __cls.load = classmethod(aload)


#-------------------------------------------------------------------------------------#
# 启动会话
#-------------------------------------------------------------------------------------#

#注册菜单函数
base_clss = (DataVol, Region, DistFunc, Phase, Object, ObjectRelation, PhsRltGraph)
obj_clss = [cls for cls in TKUI.__subclasses__()]
clss_order = {cls: i for i, cls in enumerate(obj_clss, 1)}

context = Context(locals())

for base_cls in base_clss:
    base_obj_clss = [cls for cls in TKUI.__subclasses__() if issubclass(cls, base_cls)]
    for cls in base_obj_clss:
        exec(f"def add_{cls.__name__}(): context.tkvar_cmd.set('{context.tkvar_cmd.get()} = {cls.__name__}()')")


window.config(menu=context.menu_panel)
window.mainloop()


## 其他
"""
import tkinter as tk
from tkinter import filedialog, messagebox


def browse_input_file():
    filename = filedialog.askopenfilename(title="Select Input File")
    input_file_var.set(filename)


def browse_output_file():
    filename = filedialog.asksaveasfilename(title="Select Output File", defaultextension=".mesh")
    output_file_var.set(filename)


def start_meshing():
    input_file = input_file_var.get()
    output_file = output_file_var.get()
    resolution = resolution_var.get()
    reduction = reduction_var.get()
    smooth_radius = smooth_radius_var.get()
    min_angle = min_angle_var.get()
    refinement = refinement_var.get()

    if not input_file or not output_file:
        messagebox.showerror("Error", "Please select both input and output files.")
        return

    try:
        resolution = float(resolution)
        reduction = float(reduction)
        smooth_radius = float(smooth_radius)
        min_angle = float(min_angle)
        refinement = int(refinement)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for the parameters.")
        return

    # Here you would call your meshing function, passing these parameters
    # meshing_function(input_file, output_file, resolution, reduction, smooth_radius, min_angle, refinement)

    messagebox.showinfo("Success", "Meshing completed successfully!")


# Create the main window
root = tk.Tk()
root.title("Mesh Generation")

# Create StringVars to hold file paths and parameters
input_file_var = tk.StringVar()
output_file_var = tk.StringVar()
resolution_var = tk.StringVar(value="1.0")
reduction_var = tk.StringVar(value="0.5")
smooth_radius_var = tk.StringVar(value="2.0")
min_angle_var = tk.StringVar(value="20.0")
refinement_var = tk.StringVar(value="2")

# Create and place widgets
tk.Label(root, text="Input File:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=input_file_var, width=40).grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse...", command=browse_input_file).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Output File:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=output_file_var, width=40).grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Browse...", command=browse_output_file).grid(row=1, column=2, padx=10, pady=5)

tk.Label(root, text="Voxel Grid Resolution:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=resolution_var, width=10).grid(row=2, column=1, padx=10, pady=5, sticky="w")

tk.Label(root, text="Vertex Reduction:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=reduction_var, width=10).grid(row=3, column=1, padx=10, pady=5, sticky="w")

tk.Label(root, text="Smoothing Radius:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=smooth_radius_var, width=10).grid(row=4, column=1, padx=10, pady=5, sticky="w")

tk.Label(root, text="Minimum Face Angle:").grid(row=5, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=min_angle_var, width=10).grid(row=5, column=1, padx=10, pady=5, sticky="w")

tk.Label(root, text="Refinement Level:").grid(row=6, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=refinement_var, width=10).grid(row=6, column=1, padx=10, pady=5, sticky="w")

tk.Button(root, text="Start Meshing", command=start_meshing).grid(row=7, column=1, pady=20)

# Start the Tkinter main loop
root.mainloop()
"""



"""
import tkinter as tk
from tkinter import filedialog, messagebox


def browse_output_file():
    filename = filedialog.asksaveasfilename(title="Select Export File", defaultextension=".txt")
    output_file_var.set(filename)


def start_generation():
    domain_size_x = domain_size_x_var.get()
    domain_size_y = domain_size_y_var.get()
    domain_size_z = domain_size_z_var.get()
    resolution = resolution_var.get()
    fiber_diameter = fiber_diameter_var.get()
    orientation_alpha = orientation_alpha_var.get()
    orientation_beta = orientation_beta_var.get()
    orientation_gamma = orientation_gamma_var.get()
    output_file = output_file_var.get()

    if not output_file:
        messagebox.showerror("Error", "Please select an export file path.")
        return

    try:
        domain_size_x = float(domain_size_x)
        domain_size_y = float(domain_size_y)
        domain_size_z = float(domain_size_z)
        resolution = float(resolution)
        fiber_diameter = float(fiber_diameter)
        orientation_alpha = float(orientation_alpha)
        orientation_beta = float(orientation_beta)
        orientation_gamma = float(orientation_gamma)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for the parameters.")
        return

    # Here you would call your fiber generation function, passing these parameters
    # generate_fibers(domain_size, resolution, fiber_diameter, orientation_distribution, output_file)

    messagebox.showinfo("Success", "Carbon fiber electrode generation completed!")


# Create the main window
root = tk.Tk()
root.title("Carbon Fiber Electrode Generation")

# Create StringVars to hold parameters and file paths
domain_size_x_var = tk.StringVar(value="100.0")
domain_size_y_var = tk.StringVar(value="100.0")
domain_size_z_var = tk.StringVar(value="100.0")
resolution_var = tk.StringVar(value="1.0")
fiber_diameter_var = tk.StringVar(value="7.0")
orientation_alpha_var = tk.StringVar(value="0.0")
orientation_beta_var = tk.StringVar(value="0.0")
orientation_gamma_var = tk.StringVar(value="0.0")
output_file_var = tk.StringVar()

# Create and place widgets
tk.Label(root, text="Domain Size (X, Y, Z):").grid(row=0, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=domain_size_x_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")
tk.Entry(root, textvariable=domain_size_y_var, width=10).grid(row=0, column=2, padx=5, pady=5, sticky="w")
tk.Entry(root, textvariable=domain_size_z_var, width=10).grid(row=0, column=3, padx=5, pady=5, sticky="w")

tk.Label(root, text="Domain Resolution:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=resolution_var, width=10).grid(row=1, column=1, padx=10, pady=5, sticky="w")

tk.Label(root, text="Fiber Diameter:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=fiber_diameter_var, width=10).grid(row=2, column=1, padx=10, pady=5, sticky="w")

tk.Label(root, text="Orientation Distribution (α, β, γ):").grid(row=3, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=orientation_alpha_var, width=10).grid(row=3, column=1, padx=5, pady=5, sticky="w")
tk.Entry(root, textvariable=orientation_beta_var, width=10).grid(row=3, column=2, padx=5, pady=5, sticky="w")
tk.Entry(root, textvariable=orientation_gamma_var, width=10).grid(row=3, column=3, padx=5, pady=5, sticky="w")

tk.Label(root, text="Export File Path:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=output_file_var, width=40).grid(row=4, column=1, columnspan=2, padx=10, pady=5)
tk.Button(root, text="Browse...", command=browse_output_file).grid(row=4, column=3, padx=10, pady=5)

tk.Button(root, text="Start Generation", command=start_generation).grid(row=5, column=1, columnspan=2, pady=20)

# Start the Tkinter main loop
root.mainloop()

"""