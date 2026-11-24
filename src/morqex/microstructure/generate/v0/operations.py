import yaml
import os
from torch import nn
import torch
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys


def load_config(config_path):
    """
    读取 YAML 配置文件并返回参数 dict，
    自动检查配置结构是否完整。
    参数
    ----
    config_path : str
        YAML 配置文件的路径
    返回
    ----
    config : dict
        结构化的 SliceGAN 参数
    """

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ----------- 基本结构检查 ----------- #
    required_top_keys = ["project", "data", "latent", "network", "generation"]

    for k in required_top_keys:
        if k not in config:
            raise ValueError(f"Config missing top-level section: '{k}'")

    # project
    if "name" not in config["project"] or "dir" not in config["project"]:
        raise ValueError("Project config must contain 'name' and 'dir'")

    # data
    required_data = ["image_type", "img_channels", "data_type",
                     "data_path", "img_size", "scale_factor"]
    for k in required_data:
        if k not in config["data"]:
            raise ValueError(f"Data section missing: '{k}'")

    # latent
    if "z_channels" not in config["latent"]:
        raise ValueError("latent must contain 'z_channels'")

    # network
    required_net = ["lays", "laysd", "dk", "ds", "df", "dp",
                    "gk", "gs", "gf", "gp"]
    for k in required_net:
        if k not in config["network"]:
            raise ValueError(f"Network section missing: '{k}'")

    # generation
    if "lf" not in config["generation"] or "periodic" not in config["generation"]:
        raise ValueError("generation must contain 'lf' and 'periodic'")

    return config


def mkdr(proj,proj_dir,Training):
    """
    When training, creates a new project directory or overwrites an existing directory according to user input. When testing, returns the full project path
    :param proj: project name
    :param proj_dir: project directory
    :param Training: whether new training run or testing image
    :return: full project path
    """
    pth = proj_dir + '/' + proj
    if Training:
        try:
            os.mkdir(pth)
            return pth + '/' + proj
        except FileExistsError:
            print('Directory', pth, 'already exists. Enter new project name or hit enter to overwrite')
            new = input()
            if new == '':
                return pth + '/' + proj
            else:
                pth = mkdr(new, proj_dir, Training)
                return pth
        except FileNotFoundError:
            print('The specifified project directory ' + proj_dir + ' does not exist. Please change to a directory that does exist and again')
            sys.exit()
    else:
        return pth + '/' + proj


def weights_init(m):
    """
    初始化训练权重
    m: 要初始化的卷积
    return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda,nc):
    """
    计算一批真实数据和生成数据的梯度惩罚
    netD: 判别器网络
    real_data: 真实数据
    fake_data: 生成数据
    batch_size: 批量大小
    l: 图像尺寸
    device: 设备
    gp_lambda: GP 的学习参数
    nc: 通道数
    return: 梯度惩罚
    """
    alpha = torch.rand(batch_size, 1, device = device)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)

    # create interpolate dataset
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    #pass interpolates through netD
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device = device),
                              create_graph=True, only_inputs=True)[0]
    # extract the grads and calculate gp
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def show_training_time(steps, time, start, i, epoch_num, num_epochs):
    """
    根据已用时间和训练轮数估算剩余时间
    steps:
    time: 当前时间
    start: 开始时间
    i: 当前轮次的迭代
    epoch_num: 当前的伦次
    param num_epochs: 总轮数
    """
    elap = time - start
    progress = epoch_num * steps + i + 1
    rem = num_epochs * steps - progress
    seconds = rem / progress * elap
    hrs = int(seconds / 3600)
    mins = int((seconds / 3600 % 1) * 60)
    print('[%d/%d-%d/%d]\tETA: %d hrs %d mins'
          % (epoch_num, num_epochs, i, steps,
             hrs, mins))

def plot_transfer(img,img_type):
    """
    将一张热成像图像转换回灰度图
    img: 输入图像
    imtype: 图像类型
    return: 可绘制的图像，形式与训练数据相同
    """

    try:
        img = img.detach().cpu()
    except:
        pass
    if img_type == 'colour':
        return np.int_(255 * (np.swapaxes(img[0], 0, -1)))
    if img_type == 'grayscale':
        return 255*img[0][0]
    else:
        nphase = img.shape[1]
        return 255*torch.argmax(img, 1)/(nphase-1)

# def training_plotter(img,num_axis,img_type,img_pth):
#     """
#     创建一个包含 3*num_axis 子图的图，显示训练过程中沿三个轴的切片
#     img: 原始输入图像
#     num_axis: 每个方向要获取的切片数量
#     img_type: 图像类型
#     pth: 保存图像的位置
#     """
#     if  img_type == 'colour':
#         img = plot_transfer(img,img_type)
#     else:
#         img = plot_transfer(img,img_type)[0]
#     fig, axs = plt.subplots(num_axis, 3)
#     if img_type == 'colour':
#         for j in range(num_axis):
#             axs[j, 0].imshow(img[j, :, :, :], vmin = 0, vmax = 255)
#             axs[j, 1].imshow(img[:, j, :, :],  vmin = 0, vmax = 255)
#             axs[j, 2].imshow(img[:, :, j, :],  vmin = 0, vmax = 255)
#     elif img_type == 'grayscale':
#         for j in range(num_axis):
#             axs[j, 0].imshow(img[j, :, :], cmap = 'gray')
#             axs[j, 1].imshow(img[:, j, :], cmap = 'gray')
#             axs[j, 2].imshow(img[:, :, j], cmap = 'gray')
#     else:
#         for j in range(num_axis):
#             axs[j, 0].imshow(img[j, :, :])
#             axs[j, 1].imshow(img[:, j, :])
#             axs[j, 2].imshow(img[:, :, j])
#     plt.savefig(img_pth + 'train_fig.png')
#     plt.close()

def training_plotter(img, num_axis, img_type, img_pth):
    """
    创建一个包含 3*num_axis 子图的图，显示训练过程中沿三个轴的切片
    img: 原始输入图像
    num_axis: 每个方向要获取的切片数量
    img_type: 图像类型
    img_pth: 保存图像的位置
    """

    # =========================
    # 论文风格设置
    # =========================
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # =========================
    # 数据预处理
    # =========================
    if img_type == 'colour':
        img = plot_transfer(img, img_type)
    else:
        img = plot_transfer(img, img_type)[0]

    # 防止 num_axis = 1 时 axs 不是二维数组
    fig, axs = plt.subplots(
        num_axis, 3,
        figsize=(9, 3.0 * num_axis),
        constrained_layout=False
    )

    if num_axis == 1:
        axs = np.expand_dims(axs, axis=0)

    # =========================
    # 设置列标题
    # =========================
    col_titles = ['Slice along X', 'Slice along Y', 'Slice along Z']
    for i in range(3):
        axs[0, i].set_title(col_titles[i],pad=10, fontweight='bold')

    # =========================
    # 绘图
    # =========================
    for j in range(num_axis):
        if img_type == 'colour':
            axs[j, 0].imshow(img[j, :, :, :], vmin=0, vmax=255)
            axs[j, 1].imshow(img[:, j, :, :], vmin=0, vmax=255)
            axs[j, 2].imshow(img[:, :, j, :], vmin=0, vmax=255)

        elif img_type == 'grayscale':
            axs[j, 0].imshow(img[j, :, :], cmap='gray')
            axs[j, 1].imshow(img[:, j, :], cmap='gray')
            axs[j, 2].imshow(img[:, :, j], cmap='gray')

        else:
            axs[j, 0].imshow(img[j, :, :])
            axs[j, 1].imshow(img[:, j, :])
            axs[j, 2].imshow(img[:, :, j])

        # 去掉坐标轴刻度，科研图更简洁
        for k in range(3):
            axs[j, k].set_xticks([])
            axs[j, k].set_yticks([])
            axs[j, k].spines['top'].set_visible(False)
            axs[j, k].spines['right'].set_visible(False)
            axs[j, k].spines['bottom'].set_visible(False)
            axs[j, k].spines['left'].set_visible(False)

        # 每一行左侧标明切片编号
        axs[j, 0].set_ylabel(
            f'Index {j}',
            rotation=90,
            fontsize=18,
            labelpad=12,
            va='center',
            fontweight='bold'
        )

    # =========================
    # 调整布局
    # =========================
    plt.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.93,
        bottom=0.04,
        wspace=0.05,
        hspace=0.12
    )

    # =========================
    # 保存
    # =========================
    save_path = os.path.join(img_pth, 'train_fig.png') if os.path.isdir(img_pth) else img_pth + 'train_fig.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()

# def dinamic_plot(data,labels,pth,name):
#     """
#     适用于各种不同图表的简单作图工具,进行训练的过程的动态绘图
#     data: 数据数组列表
#     labels: 图表标签列表
#     pth: 保存图表的位置
#     name: 图表名称
#     return:
#     """
#     for datum,lbl in zip(data,labels):
#         plt.plot(datum, label = lbl)
#     plt.legend()
#     plt.savefig(pth + '_' + name)
#     plt.close()

def dinamic_plot(data, labels, pth, name):
    """
    适用于各种不同图表的简单作图工具，进行训练过程的动态绘图
    data: 数据数组列表
    labels: 图表标签列表
    pth: 保存图表的位置
    name: 图表名称
    return:
    """

    # =========================
    # 科研论文风格全局设置
    # =========================
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # =========================
    # 创建画布
    # =========================
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)
    # =========================
    # 绘图
    # =========================
    for datum, lbl in zip(data, labels):
        ax.plot(
            datum,
            label=lbl,
            linewidth=1.5,
        )

    # =========================
    # 坐标轴美化
    # =========================
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.tick_params(direction="in", length=5, width=1.1)

    # 只保留下边框和左边框，更适合论文
    ax.spines["top"].set_linewidth(1.2)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # 图例
    ax.legend(frameon=False)

    # 可选浅网格，科研图里常用
    # ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    # 自动紧凑布局
    plt.tight_layout()

    # =========================
    # 保存
    # =========================
    save_path = pth + '_' + name
    plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

def generate_img(pth, img_type, netG, nz = 64, lf = 4, periodic=False):
    """
    为已训练或正在训练的生成器保存测试卷
    pth: 保存图像的位置，同时也是查找生成器的位置g_
    img_type: 图像类型
    netG: 已加载的生成器类
    nz: 潜在 z 维度
    lf: 长度因子
    periodic: 轴1到n的周期性列表
    """
    netG.load_state_dict(torch.load(pth + '_Gen.pt'))
    netG.eval()
    netG.cuda()
    noise = torch.randn(1, nz, lf, lf, lf).cuda()
    if periodic:
        if periodic[0]:
            noise[:, :, :2] = noise[:, :, -2:]
        if periodic[1]:
            noise[:, :, :, :2] = noise[:, :, :, -2:]
        if periodic[2]:
            noise[:, :, :, :, :2] = noise[:, :, :, :, -2:]
    with torch.no_grad():
        raw = netG(noise)
    print('Postprocessing')
    gb = plot_transfer(raw,img_type)[0]
    if periodic:
        if periodic[0]:
            gb = gb[:-1]
        if periodic[1]:
            gb = gb[:,:-1]
        if periodic[2]:
            gb = gb[:,:,:-1]
    tif = np.int_(gb)
    tifffile.imwrite(pth + '.tif', tif)

    return tif, raw, netG

def generate_img_checkpoint(pth, img_type, netG, nz = 64, lf = 4, periodic=False, device='cuda'):
    """
    为已训练或正在训练的生成器保存测试卷
    pth: 保存图像的位置，同时也是查找生成器的位置g_
    img_type: 图像类型
    netG: 已加载的生成器类
    nz: 潜在 z 维度
    lf: 长度因子
    periodic: 轴1到n的周期性列表
    """
    ckpt_path = pth + '_checkpoint.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(checkpoint['netG'])
    netG.eval()
    netG.cuda()
    noise = torch.randn(1, nz, lf, lf, lf).cuda()
    if periodic:
        if periodic[0]:
            noise[:, :, :2] = noise[:, :, -2:]
        if periodic[1]:
            noise[:, :, :, :2] = noise[:, :, :, -2:]
        if periodic[2]:
            noise[:, :, :, :, :2] = noise[:, :, :, :, -2:]
    with torch.no_grad():
        raw = netG(noise)
    print('Postprocessing')
    gb = plot_transfer(raw,img_type)[0]
    print(gb.shape)
    if periodic:
        print(1)
        if periodic[0]:
            gb = gb[:-1]
        if periodic[1]:
            gb = gb[:,:-1]
        if periodic[2]:
            gb = gb[:,:,:-1]
    tif = np.int_(gb)
    print(tif.shape)
    tifffile.imwrite(pth + '.tif', tif)

    return tif, raw, netG

def generate_img_cond(pth, img_type, netG, phi, nz=64, lf=4, periodic=False):
    """
    为已训练或正在训练的生成器保存测试卷（条件：孔隙率 phi）

    pth: 保存图像的位置，同时也是查找生成器的位置 pth+'_Gen.pt'
    img_type: 图像类型
    netG: 已加载的生成器实例（注意是实例，不是类）
    phi: 目标孔隙率（float 或 tensor），推荐范围 (0,1)
    nz: 潜在 z 维度（对应你的 z_channels）
    lf: 长度因子（对应 lz=4）
    periodic: 轴1到n的周期性列表，例如 [True, False, True]
    """
    # 1) load & eval & cuda（保持你原逻辑）
    netG.load_state_dict(torch.load(pth + '_Gen.pt'))
    netG.eval()
    netG.cuda()

    # 2) 生成噪声（保持你原逻辑）
    noise = torch.randn(1, nz, lf, lf, lf).cuda()
    if periodic:
        if periodic[0]:
            noise[:, :, :2] = noise[:, :, -2:]
        if periodic[1]:
            noise[:, :, :, :2] = noise[:, :, :, -2:]
        if periodic[2]:
            noise[:, :, :, :, :2] = noise[:, :, :, :, -2:]

    # ✅ 3) 设置孔隙率条件（最小新增）
    # 统一为 [1,1]，避免 batch mismatch
    if not torch.is_tensor(phi):
        phi_t = torch.tensor([[float(phi)]], device=noise.device, dtype=torch.float32)
    else:
        phi_t = phi.to(device=noise.device, dtype=torch.float32)
        if phi_t.dim() == 0:
            phi_t = phi_t.view(1, 1)
        elif phi_t.dim() == 1:
            phi_t = phi_t.view(-1, 1)
        if phi_t.shape[0] != 1:
            # 这里是生成单个样本，强制用第一项
            phi_t = phi_t[:1]

    # 兼容 DataParallel
    if hasattr(netG, "module"):
        netG.module.set_phi(phi_t)
    else:
        netG.set_phi(phi_t)

    # 4) 生成（保持你原逻辑）
    with torch.no_grad():
        raw = netG(noise)

    print('Postprocessing')
    gb = plot_transfer(raw, img_type)[0]

    # 5) periodic 裁切（保持你原逻辑）
    if periodic:
        if periodic[0]:
            gb = gb[:-1]
        if periodic[1]:
            gb = gb[:, :-1]
        if periodic[2]:
            gb = gb[:, :, :-1]

    tif = np.int_(gb)
    tifffile.imwrite(pth + '.tif', tif)

    return tif, raw, netG
