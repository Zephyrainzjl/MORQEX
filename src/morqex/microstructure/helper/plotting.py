import os 
from morqex.microstructure.core.particle_graph import * 
import matplotlib.pyplot as plt 


#-------------------------------------------------------------------------------------#
# 可视化配置
#-------------------------------------------------------------------------------------#

class Config():
    def load(self, load_cfg):
        self.__dict__.update(load_cfg.__dict__)

class ViewConfig(Config):
    def __init__(self) -> None:
        self.on = True 
        self.rpos = 0.5
        self.rdep = 0.4
        self.axis = 1 
        self.font_size = 13

        self.img_size = (5, 5)
        self.img_save = False 
        self.img_dir = "images"
        self.img_fmt = "png" #"svg" 

        self.hist_bin = 400
        self.hist_smooth = 4 

        self.legend = True 
        self.legend_size = 1.0
        self.legend_void = True  

        self.reg = Region(0.01, np.array([1,1,1]))

vcfg = ViewConfig()


def plt_vol(vol, legend={}, cb=False, vrange=(), cmap="viridis"): #, zero=False):
    """绘制离散空间体积"""
    vol = vol[:]
    if not np.any(vol):
        warnings.warn("plt_vol failed: empty vol")
        return 

    #特征维度拆散
    if vol.ndim == 4:
        for i in range(vol.shape[3]):
            plt_vol(vol[:, :, :, i])
        return 

    #自动适应二维数据
    mask = np.array(vol.shape)==1
    axis = np.arange(vol.ndim)[mask][0] if np.any(mask) else vcfg.axis

    #计算截面
    cut_axis_len = vol.shape[axis]
    cut_idx = min(int(cut_axis_len * vcfg.rpos), cut_axis_len-1)
    key = [slice(None) if ax != axis else cut_idx for ax in range(3)]
    cut = np.copy(vol[tuple(key)]) 

    #绘制声明
    # plt.clf()
    plt.figure(figsize=vcfg.img_size)
    
    #轴方向
    sort_cut = cut[::-1] if axis == 1 else cut.T[::-1]
    # if zero: sort_cut[0,0] = 0
    #处理极端值
    if vrange:
        plt.imshow(sort_cut, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
    elif np.max(vol) < np.inf and np.min(vol) > -np.inf:
        plt.imshow(sort_cut, vmin=np.min(vol), vmax=np.max(vol), cmap=cmap)
    else: 
        vol[vol == np.inf] = np.nan
        vol[vol == -np.inf] = np.nan
        plt.imshow(sort_cut, vmin=np.nanmin(vol), vmax=np.nanmax(vol), cmap=cmap)

    
    plt.xticks([])
    plt.yticks([])
    ax_lab = {1: ("z", "x", "y"), 0: ("y", "z", "x"), 2: ("x", "y", "z")}[axis]
    plt.xlabel(ax_lab[0], loc="right", fontsize=vcfg.font_size)
    plt.ylabel(ax_lab[1], loc="top", fontsize=vcfg.font_size)
    plt.title(f"{ax_lab[1]}-{ax_lab[0]} {ax_lab[2]}[{cut_idx}] {vol.shape}", fontsize=vcfg.font_size)

    if cb: plt_cb()
    if legend: plt_legend(vol, legend, cmap=cmap)
    

def plt_vol_sem(phases_vol, phase_scale=12, depth_scale=0.2, cmap="viridis"):
    """绘制离散空间体积,渲染时重新赋予相序号"""
    phases_vol = phases_vol[:]
    if not np.any(phases_vol):
        print("plot seeds vol sem failed: seeds render vol empty")
    phases_vol = phases_vol * phase_scale
    phase_cnt = 0
    if not isinstance(phases_vol, type(None)) and np.any(phases_vol):
        phases = np.unique(phases_vol)
        sem_vol = [] 
        for phase in phases: 
            if phase <= 0:
                continue
            phase_cnt += 1
            vol = np.zeros_like(phases_vol)
            vol[np.where(phases_vol == phase)] = phase_cnt * phase_scale  #新序号
            cut_idx0 = int((vol.shape[vcfg.axis]-1)*vcfg.rpos)
            cut_idx1 = int((vol.shape[vcfg.axis]-1)*(vcfg.rpos+vcfg.rdep))
            idx0 = min(cut_idx0, cut_idx1)
            idx1 = max(cut_idx0, cut_idx1)
            key = [slice(None) if axis != vcfg.axis else slice(idx0, idx1) for axis in range(3)]
            cut_vol = vol[tuple(key)]
            depth_decay = np.cumsum(np.full(cut_vol.shape, depth_scale), axis=vcfg.axis)
            if np.any(sem_vol):
                sem_vol += (cut_vol - depth_decay)  
            else: 
                sem_vol = (cut_vol - depth_decay) 

        #提取第一个非0值
        cut_axis = vcfg.axis
        p1_axis = (vcfg.axis + 1)%3
        p2_axis = (vcfg.axis + 2)%3
        sem_cut = np.zeros_like(np.max(sem_vol, axis=cut_axis)) 
        for j in range(sem_vol.shape[p1_axis]):
            for k in range(sem_vol.shape[p2_axis]):
                key = [0, 0, 0]
                key[cut_axis] = slice(None)
                key[p1_axis] = j 
                key[p2_axis] = k 
                non_zero_idx = np.where(sem_vol[tuple(key)] > 0)[0]
                if np.any(non_zero_idx):
                    value = sem_vol[tuple(key)][non_zero_idx[0]] 
                    key.remove(slice(None))
                    sem_cut[tuple(key)] = value 

        # plt.clf()
        plt.figure(figsize=vcfg.img_size)
        plt.xticks([])
        plt.yticks([])
        ax_lab = {1: ("z", "x", "y"), 0: ("y", "z", "x"), 2: ("x", "y", "z")}[vcfg.axis]
        plt.xlabel(ax_lab[0], loc="right", fontsize=vcfg.font_size)
        plt.ylabel(ax_lab[1], loc="top", fontsize=vcfg.font_size)
        plt.title(f"{ax_lab[1]}-{ax_lab[0]} {ax_lab[2]}[{cut_idx0}~{cut_idx1}] {vol.shape}", fontsize=vcfg.font_size)
        plt.imshow(sem_cut, cmap=cmap) #, cmap=ListedColormap(colors))
        # plt.colorbar()
    else:
        print("plot vol failed: the vol is empty")
    return 


def plt_cb():
    plt.rcParams['font.size'] = vcfg.font_size
    plt.colorbar()


def plt_legend(vol, legend: dict={}, cmap="viridis", *args, **kwargs):
    vol = vol[:]
    if np.unique(vol).size > 10:
        warnings.warn(f"having {np.unique(vol).size} > 10 attrs, legend plot passed")
        return 

    if not np.any(vol): return 
    if not vcfg.legend: return 

    #计算截面
    cut_axis_len = vol.shape[vcfg.axis]
    cut_idx = min(int(cut_axis_len * vcfg.rpos), cut_axis_len-1)
    key = [slice(None) if axis != vcfg.axis else cut_idx for axis in range(3)]
    cut = vol[tuple(key)]

    #绘制图例
    marks = np.unique(vol).astype(np.int32)
    if marks.max()-marks.min() == 0:
        colors = plt.get_cmap(cmap)([1.0])
    else: 
        colors = plt.get_cmap(cmap)((marks - marks.min())/(marks.max()-marks.min()))  #按照数值范围映射

    #自动补充图例
    if not legend:
        legend = {mark :f"Phase{mark}" for mark in marks}
    
    for mark, color in zip(marks, colors):
        if mark == 0 and not vcfg.legend_void: continue
        if mark not in legend: 
            warnings.warn(f"plt legend warn: mark {mark} has no legend")
            continue 
        plt.scatter([],[], s=100*vcfg.legend_size, label=f"{legend[mark]}", color=color)
    
    plt.legend(scatterpoints=1, fontsize=vcfg.font_size-1, facecolor="white", edgecolor="none", framealpha=1, frameon=True, *args, **kwargs)



def plt_data(data, nbin=50, density=True, weights=[], cum=False, *args, **kwargs):
    """"分布数据的绘制"""
    # plt.clf()
    # plt.figure(figsize=(vcfg.img_size[0]*1.2, vcfg.img_size[1]))
    if not np.any(weights):
        weights = np.ones(data.shape) 

    hists, edges = np.histogram(data, bins=nbin, density=density, weights=weights)
    # smooth_hist = np.convolve(hists, np.full((vcfg.hist_smooth,), 1/vcfg.hist_smooth), "same")
    if cum==False: plt.plot(edges[1:], hists/np.sum(hists), *args, **kwargs)
    else: plt.plot(edges[1:], np.cumsum(hists/np.sum(hists)), *args, **kwargs)
    plt.xticks(fontsize=vcfg.font_size)
    plt.yticks(fontsize=vcfg.font_size)
    plt.legend(fontsize=vcfg.font_size+1)
    plt.xlabel("Relative Radius", fontsize=vcfg.font_size+2)
    plt.ylabel("Density", fontsize=vcfg.font_size+2)
    # plt.title("Radius Distribution", fontsize=vcfg.font_size+2)
    # plt.show()

def plt_dns(mats, rlt_mark, x, colors, methods, marks, styles, smth=1, loc="best", every=3):
    # plt.clf()
    plt.figure(figsize=vcfg.img_size)
    mtd_cls = {mtd: clr for mtd, clr in zip(methods, colors)}
    for mat, mtd, mark, style in zip(mats, methods, marks, styles):
        if isinstance(rlt_mark, (list, tuple)): 
            rlt = mat.get_rlt(*rlt_mark)
            # if rlt_mark[0] == rlt_mark[1]: dns = rlt.dns_func.nn_dist.density(x)
            # else: dns = rlt.dns_func.density(x)
            dns = rlt.dns_func.density(x)
            dns = dns/np.sum(dns)
            label = "Distance"
            print(f"{mtd} error:", dist_error(mats[0].get_rlt(*rlt_mark).dns_func, mat.get_rlt(*rlt_mark).dns_func))
        elif isinstance(rlt_mark, (int, np.integer)):
            phs = mat.get_phs(rlt_mark)
            dns = phs.r_dist.density(x)
            dns = dns/np.sum(dns)
            label = "Radius"
            print(f"{mtd} error:", dist_error(mats[0].get_phs(rlt_mark).r_dist, mat.get_phs(rlt_mark).r_dist))
        
        smt_dns = np.convolve(dns, np.full(smth, 1/smth), "same")
        plt.plot(x, smt_dns, color=mtd_cls[mtd], label=mtd, marker=mark, linestyle=style, markevery=every)

        # plt.scatter(x[mask], smt_dns[mask])
        # dist_func_plt(rlt.dns_func, smth, color=mtd_cls[mtd], label=mtd)
    plt.legend(loc=loc, fontsize=vcfg.font_size+1)
    plt.xticks(fontsize=vcfg.font_size)
    plt.yticks(fontsize=vcfg.font_size)
    plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y', useMathText=True)
    plt.gca().yaxis.get_offset_text().set_fontsize(vcfg.font_size)
    plt.xlabel(f"Relative {label}", fontsize=vcfg.font_size+2)
    plt.ylabel("Desity", fontsize=vcfg.font_size+2)
    # plt.show()


def plt_bar(hs, xs, labs, clrs, loc=2, per=True, sci=False, **kwargs):
    """绘制多级条形图"""
    width = 1 / (np.size(labs) + 2)

    for i, x in enumerate(xs):
        for j, h in enumerate(hs[i]):
            if i == 0:
                plt.bar(i + width * j, height=h, width=width, color=clrs[j], label=labs[j], **kwargs)
            else:
                plt.bar(i + width * j, height=h, width=width, color=clrs[j], **kwargs)
            if per: plt.text(i + width * j, h, f"{h*100:.1f}%", ha='center',va='bottom', fontsize=vcfg.font_size) 
            elif not per and not sci: plt.text(i + width * j, h, f"{h:.2f}", ha='center',va='bottom', fontsize=vcfg.font_size) 
            else: plt.text(i + width * j, h, f"{h:.3e}", ha='center',va='bottom', fontsize=vcfg.font_size) 

        hs0 = np.array(hs[0])
        print(f"{x} error:", (np.array(hs[i])-hs0)/hs0)
    if isinstance(xs[0], str):
        plt.xticks([i + (len(hs[i])-1)*width/2 for i in range(len(xs))], xs, fontsize=vcfg.font_size)
    else: 
        plt.xticks(xs, fontsize=vcfg.font_size)
    plt.yticks(fontsize=vcfg.font_size)
    plt.legend(loc=loc, fontsize=vcfg.font_size+1) 

def plt_bar_stack(hs, xs, labs, clrs, loc=2, per=True):
    """绘制多级条形图"""
    width = 1 / (np.size(labs) + 2)

    for i, x in enumerate(xs):
        for j, h in enumerate(hs[i]):
            h0 = sum(hs[i][:j])
            if i == 0:
                plt.barh(i*0.6 + width/2, height=width, left=h0, width=h, color=clrs[j], label=labs[j])
            else:
                plt.barh(i*0.6 + width/2, height=width, left=h0, width=h, color=clrs[j])
            if per: plt.text(h0+h-0.02, i*0.6 + width, f"{h*100:.1f}%", ha='center',va='bottom', fontsize=vcfg.font_size) 
            else: plt.text(h0+h-0.02, i*0.6 + width, f"{h:.2f}", ha='center',va='bottom', fontsize=vcfg.font_size) 

        hs0 = np.array(hs[0])
        print(f"{x} error:", (np.array(hs[i])-hs0)/hs0)
    if isinstance(xs[0], str):
        plt.yticks([i*0.6 + (len(hs[i])-1)*width/2 for i in range(len(xs))], xs, fontsize=vcfg.font_size)
    else: 
        plt.yticks(xs, fontsize=vcfg.font_size)
    plt.xticks(fontsize=vcfg.font_size)
    plt.legend(loc=loc, fontsize=vcfg.font_size + 1)

def poly(x, y, xi, reg=1):
    p = np.polyfit(x, y, reg)
    return np.polyval(p, xi)

def plt_img_save(fname):
    if not os.path.isdir(vcfg.img_dir):
        os.mkdir(vcfg.img_dir)
    plt.savefig(f"{vcfg.img_dir}/{fname}.{vcfg.img_fmt}", format=vcfg.img_fmt)

def adjust_pts_order(pts_2ds):

    ''' sort rectangle points by counterclockwise '''

    cen_x, cen_y = np.mean(pts_2ds, axis=0)
    #refer_line = np.array([10,0]) 
    d2s = []
    for i in range(len(pts_2ds)):

        o_x = pts_2ds[i][0] - cen_x
        o_y = pts_2ds[i][1] - cen_y
        atan2 = np.arctan2(o_y, o_x)
        if atan2 < 0:
            atan2 += np.pi * 2
        d2s.append([pts_2ds[i], atan2])

    d2s = sorted(d2s, key=lambda x:x[1])
    order_2ds = np.array([x[0] for x in d2s])

    return order_2ds


def plt_circle2d(seeds, figure, axes, colors=None, fill=True):
    from matplotlib.patches import Circle

    # seeds = seeds.remove_overlap_aabb(0.4)[::-1]
    if colors == None:
        marks = seeds.mark
        colors = plt.get_cmap('viridis')((marks - marks.min())/(marks.max()-marks.min()))  #按照数值范围映射

    shown = [] 
    for loc, r, mark, color in zip(seeds.loc[:, (0, 1)], seeds.r, seeds.mark, colors): 
        if not mark: continue
        if mark not in shown:
            axes.add_patch(Circle(loc, r, color=color, fill=fill, label=f"Phase{mark}")) 
            shown.append(mark)
        else: 
            axes.add_patch(Circle(loc, r, color=color, fill=fill))

    figure.set_figwidth(vcfg.img_size[1])
    figure.set_figheight(vcfg.img_size[1])
    axes.set_xticks([])
    axes.set_yticks([])
    axes.legend(loc = 1, facecolor="white", framealpha=1, frameon=True, fontsize=vcfg.font_size+1)

def plt_voro2d(seeds, figure, axes, colors=None, fill=True):
    from tess import Container
    from matplotlib.patches import Polygon 

    seeds = seeds.remove_overlap_aabb(0.8)[::-1]
    cells = Container(seeds.loc, limits=(1,1,1), periodic=False, radii=seeds.r)[::-1]
    if colors == None:
        marks = seeds.mark
        colors = plt.get_cmap('viridis')((marks - marks.min())/(marks.max()-marks.min()))  #按照数值范围映射

    shown = [] 
    for cell, mark, color in zip(cells, seeds.mark, colors):
        if not mark: continue
        verts = np.array(cell.vertices())
        sort_verts = adjust_pts_order(verts[verts[:, 1]<0.1][:, (0, 1)]) 
        if mark not in shown:
            axes.add_patch(Polygon(sort_verts, color=color, fill=fill, label=f"Phase{mark}")) 
            shown.append(mark)
        else: 
            axes.add_patch(Polygon(sort_verts, color=color, fill=fill)) 

    figure.set_figwidth(vcfg.img_size[1])
    figure.set_figheight(vcfg.img_size[1]) 
    axes.set_xticks([])
    axes.set_yticks([])
    axes.legend(loc = 1, facecolor="white", framealpha=1, frameon=True, fontsize=vcfg.font_size+1)


def stereographic_projection(v):
    """
    v: array of shape (N,3)
    return: projected x,y coordinates on pole figure
    """
    v = v / np.linalg.norm(v, axis=1)[:, None]   # normalize
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    # Only upper hemisphere (z >= 0)
    mask = z >= 0
    x, y, z = x[mask], y[mask], z[mask]

    # Stereographic projection
    denom = 1 + z
    X = x / denom
    Y = y / denom
    return X, Y


def plot_pole_figure(vectors):
    # Project directions
    X, Y = stereographic_projection(vectors)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # Draw the main circle (pole figure boundary)
    circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='none', linewidth=1.2)
    ax.add_patch(circle)

    # Draw radial grid (every 10°)
    for angle in np.arange(10, 90, 10):
        r = np.tan(np.radians(angle/2))  # stereographic radius
        circ = plt.Circle((0, 0), r, edgecolor='gray', facecolor='none', linestyle='--', linewidth=0.6)
        ax.add_patch(circ)

    # Draw azimuth lines
    for az in np.arange(0, 360, 30):
        rad = np.linspace(0, 1, 200)
        theta = np.radians(az)
        ax.plot(rad*np.cos(theta), rad*np.sin(theta), color='gray', linewidth=0.6, linestyle='--')

    # Scatter plot of poles
    ax.scatter(X, Y, s=40, color='blue', alpha=0.4)

    # Labels
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pole Figure (Stereographic Projection)", fontsize=14)

    plt.show()


# ==========================================================
# Normalize vectors
# ==========================================================
def normalize(v):
    return v / np.linalg.norm(v, axis=1)[:, None]


# ==========================================================
# Reduce direction into cubic fundamental region for IPF sector
# Region:  0 <= φ <= 45°,  0 <= θ <= 54.7°
# where θ is angle to [001], φ is angle to [100] within (001) plane
# ==========================================================
def cubic_ipf_sector_reduce(v):
    """
    Map arbitrary direction to cubic IPF sector
    Using rules:
        - Take absolute value (no polarity)
        - Sort axes so z >= y >= x
    Equivalent to standard O_h symmetry reduction
    """
    v = np.abs(v)
    out = []
    for x, y, z in v:
        # Sort so that z is largest component → closest to [001]
        comps = np.array([x, y, z])
        idx = np.argsort(comps)  # ascending
        comps = comps[idx]
        out.append(comps)  # x <= y <= z   ->  in IPF sector
    return np.array(out)

def plot_pole_figure_kde(vectors, grid_res=400):
    from scipy.stats import gaussian_kde
        # --- Projection ---
    X, Y = stereographic_projection(vectors)
    xy = np.vstack([X, Y])

    # --- KDE ---
    kde = gaussian_kde(xy)

    # --- Grid ---
    grid_x = np.linspace(-1, 1, grid_res)
    grid_y = np.linspace(-1, 1, grid_res)
    Xgrid, Ygrid = np.meshgrid(grid_x, grid_y)

    # Circle mask
    R = np.sqrt(Xgrid**2 + Ygrid**2)
    inside = R <= 1

    # Compute KDE only inside circle
    Z = np.full_like(Xgrid, np.nan)     # outside = NaN → will stay white
    pts = np.vstack([Xgrid[inside], Ygrid[inside]])
    Z[inside] = kde(pts)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # White background
    ax.set_facecolor("white")

    # Density image (only inside circle)
    im = ax.imshow(
        Z,
        extent=[-1, 1, -1, 1],
        origin="lower",
        cmap="jet"
    )

    # Draw outline circle
    circle = plt.Circle((0, 0), 1, edgecolor="black", facecolor="none", linewidth=1.2)
    ax.add_patch(circle)

    # Radial grids
    for angle in np.arange(10, 90, 10):
        r = np.tan(np.radians(angle / 2))
        circ = plt.Circle((0, 0), r, edgecolor='gray', facecolor='none',
                           linestyle='--', linewidth=0.6)
        ax.add_patch(circ)

    # Azimuth grids
    for az in np.arange(0, 360, 30):
        rad = np.linspace(0, 1, 200)
        theta = np.radians(az)
        ax.plot(rad*np.cos(theta), rad*np.sin(theta),
                 color='gray', linewidth=0.6, linestyle='--')

    # Colorbar
    plt.colorbar(im, ax=ax, label="Density")

    # Clean axes
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pole Figure KDE Density (White Outside)", fontsize=14)

    plt.show()

# ==========================================================
# Project cubic IPF sector (扇形反极图)
# Standard mapping:
#   r = arccos(z)          (angle from [001])
#   phi = arctan(y/x)
#
# For cubic IPF sector: 0 <= phi <= 45°,  0 <= r <= arccos(1/sqrt(3))
# ==========================================================
def ipf_sector_projection(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    # polar angle from [001]
    r = np.arccos(z)

    # azimuth angle in (100)-(110) plane
    phi = np.arctan2(y, x)

    # scale r to unit radius (r_max = angle([001],[111]) )
    r_max = np.arccos(1 / np.sqrt(3))
    R = r / r_max

    # Cartesian plot coordinates
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    return X, Y


# ==========================================================
# Plot Fan-shaped IPF
# ==========================================================
def plot_ipf_sector(vectors, title="Cubic Inverse Pole Figure (Fan-shaped)"):
    v = normalize(vectors)
    v_red = cubic_ipf_sector_reduce(v)
    X, Y = ipf_sector_projection(v_red)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # Draw the fan boundary  (sector)
    # 0 to r_max arc (sector radius = 1)
    phi = np.linspace(0, np.pi/4, 200)
    ax.plot(np.cos(phi), np.sin(phi), 'k-', lw=1.5)

    # Two straight edges:
    ax.plot([0, 1], [0, 0], 'k-', lw=1.5)       # [001] - [101]
    ax.plot([0, np.sqrt(2)/2], [0, np.sqrt(2)/2], 'k-', lw=1.5) # [001] - [111]

    # Scatter points
    ax.scatter(X, Y, s=25, color='blue', edgecolor=None, alpha=0.4)

    # Labels at key directions
    ax.text(0, 0, "[001]", fontsize=12, ha='right', va='top')
    ax.text(1, 0, "[101]", fontsize=12, ha='left', va='top')
    ax.text(np.sqrt(2)/2, np.sqrt(2)/2, "[111]", fontsize=12, ha='left', va='bottom')

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14)

    plt.show()


#-------------------------------------------------------------------------------------#
# 注册绘制函数
#-------------------------------------------------------------------------------------#

data_vol_repr = DataVol.__repr__
phs_vol_repr = ObjVol.__repr__
obj_repr = Object.__repr__ 
phs_repr = PhaseObject.__repr__
obj_pnt_repr = ObjectPnt.__repr__
pnt_opt_repr = SeedsPntOperator.__repr__
seeds_pnt_repr = SeedsPnt.__repr__ 
dist_func_repr = DistFunc.__repr__
seeds_repr = Seeds.__repr__
re_repr = RndSplitor.__repr__ 

#绘制函数可能引起update

def data_vol_plt(data_vol: DataVol) -> str:
    if vcfg.on:
        plt_vol(data_vol)
        plt_cb()
        if vcfg.img_save:
            plt_img_save(f"{data_vol.cls_id_str}")
    return data_vol_repr(data_vol)

def phs_vol_plt(phs_vol: ObjVol) -> str:
    if vcfg.on:
        #维持有无0时的颜色映射相同
        plt_vol(phs_vol)
        plt_legend(phs_vol)
        if vcfg.img_save:
            plt_img_save(f"{phs_vol.cls_id_str}")
    return phs_vol_repr(phs_vol)

def obj_plt(obj: Object) -> str:
    if vcfg.on:
        if not obj.__dict__.get("reg", False): 
            vol = obj.discrete(vcfg.reg)
        else: 
            vol = obj.discrete(obj.reg)
        vol = obj.discrete(vcfg.reg)
        plt_vol(vol)
        plt_legend(vol)
        if vcfg.img_save:
            plt_img_save(f"{obj.cls_id_str}")
    return obj_repr(obj)

# def phs_plt(phsobj: PhaseObject) -> str:
#     if vcfg.on:
#         vol = phsobj.discrete(vcfg.reg)
#         #BUG 渲染结果和外部调用discrete不同
#         plt_vol(vol)
#         plt_legend(vol)
#         plt_img_save(f"{phsobj.cls_id_str}")
#     return phs_repr(phsobj)

def obj_pnt_plt(obj_pnt: ObjectPnt) -> str:
    if vcfg.on:
        plt_vol(obj_pnt.vol)
        plt_legend(obj_pnt.vol)
        if vcfg.img_save:
            plt_img_save(f"{obj_pnt.cls_id_str}")
    return obj_pnt_repr(obj_pnt)

def seeds_pnt_plt(obj_pnt: SeedsPnt) -> str:
    if vcfg.on:
        plt_vol(obj_pnt.vol)
        plt_legend(obj_pnt.vol)
        if vcfg.img_save:
            plt_img_save(f"{obj_pnt.cls_id_str}")
    return seeds_pnt_repr(obj_pnt)

def pnt_opt_plt(obj_pnt: SeedsPntOperator) -> str:
    if vcfg.on:
        plt_vol(obj_pnt.vol)
        plt_legend(obj_pnt.vol)
        if vcfg.img_save:
            plt_img_save(f"{obj_pnt.cls_id_str}")
    return pnt_opt_repr(obj_pnt)

def dist_func_plt(df, smth=1, *args, **kwargs):
    """绘制分布密度 支持多维分布"""
    if vcfg.on:
        plt.figure(figsize=vcfg.img_size)
        rng = (DistFunc.xmax - DistFunc.xmin)
        x0 = np.linspace(DistFunc.xmin-0.1*rng, DistFunc.xmax+0.1*rng, vcfg.hist_bin)

        x = np.repeat(x0.reshape(-1, 1), df.n_ftr, axis=1)
        p = np.convolve(df.density(x),  np.full(smth, 1/smth))
        
        if not np.any(p):
            warnings.wran("dist_func_plt failed: empty sample density")
            return dist_func_repr(df)
        #截取有效响应部分
        response_x = np.where(p > np.max(p)*0.02)[0]
        if np.any(response_x):
            pad_size = int(x0.size*0.015)
            cut_key = slice(max(np.min(response_x)-pad_size, 0), np.max(response_x)+pad_size)
            plt.plot(x0[cut_key], p[cut_key], *args, **kwargs)
            if vcfg.img_save:
                plt_img_save(f"{df.cls_id_str}")
        else: 
            plt.plot(np.array([-1.0, 1.0]), np.zeros(2), *args, **kwargs)
    return dist_func_repr(df)

def seeds_plt(seeds):
    if vcfg.on:
        if not seeds.cnt:
            warnings.warn(f"plot seeds failed: {seeds} is emtpy")
            return seeds_repr(seeds)
        if not seeds.__dict__.get("reg", False): 
            seeds_vol = seeds.discrete(vcfg.reg)
        else: 
            seeds_vol = seeds.discrete(seeds.reg)
        plt_vol(seeds_vol)
        plt_legend(seeds_vol)
        if vcfg.img_save:
            plt_img_save(f"{seeds.cls_id_str}")
    return seeds_repr(seeds)

def rnd_sp_plt(rp):
    if vcfg.on:
        nbin = 100 
        plt.plot([rp.get_prop(Seeds(), nbin, i) for i in range(nbin)])
        if vcfg.img_save:
            plt_img_save(f"{rp.cls_id_str}")
    return re_repr(rp)

DataVol.__repr__ = data_vol_plt
ObjVol.__repr__ = phs_vol_plt
Object.__repr__ = obj_plt 
# PhaseObject.__repr__ = phs_plt
ObjectPnt.__repr__ = obj_pnt_plt
SeedsPntOperator.__repr__ = pnt_opt_plt
SeedsPnt.__repr__ = seeds_pnt_plt
DistFunc.__repr__ = dist_func_plt
Seeds.__repr__ = seeds_plt 
RndSplitor.__repr__ = rnd_sp_plt


def hist_plt(data, nbin=40):
    hists, edges = np.histogram(data, bins=nbin)
    plt.plot(edges[:-1], hists)

def error_plt(avg, std, alpha=0.5, x=[], label=None):
    x = np.linspace(0, 1, avg.size) if not np.any(x) else x 
    plt.plot(x, avg, label=label)
    plt.fill_between(x, avg-std, avg+std, alpha=alpha)

def curve_error(x1, y1, x2, y2):
    from scipy import interpolate
    f = interpolate.interp1d(x1, y1, fill_value="extrapolate") 
    return np.sqrt(np.sum(((y2 - f(x2)) / y1)**2/x2.size)) 
