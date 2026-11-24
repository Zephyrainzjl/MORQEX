import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveUpsample3D(nn.Module):
    """
    3D自适应上采样模块（基于光流/特征预测的思想）
    优势：1. 无棋盘格伪影 2. 自适应学习细节 3. 可灵活调整上采样倍数
    输入: [N, C, D, H, W]
    输出: [N, C, D*s, H*s, W*s]
    """
    def __init__(self, in_channels, out_channels, upscale_factor, kernel_size=3):
        super().__init__()
        self.s = upscale_factor
        self.out_channels = out_channels

        # 1. 特征提取与偏移预测（预测每个位置的采样偏移）
        self.offset_predictor = nn.Sequential(
            nn.Conv3d(in_channels, in_channels * 2, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm3d(in_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels * 2, 3 * self.s * self.s * self.s, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        )
        
        # 2. 特征转换（保持通道数匹配）
        self.feature_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 3. 初始化偏移预测器的最后一层（初始为0，保证初始是普通上采样）
        nn.init.constant_(self.offset_predictor[-1].weight, 0.)
        nn.init.constant_(self.offset_predictor[-1].bias, 0.)

    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # 步骤1：预测偏移量 [b, 3*s³, d, h, w]
        offset = self.offset_predictor(x)
        
        # 步骤2：生成采样网格
        # 生成原始坐标网格
        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, d, device=x.device),
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_w, grid_h, grid_d], dim=0)  # [3, d, h, w]
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)      # [b, 3, d, h, w]
        
        # 调整偏移量尺度并加到原始网格上
        offset = offset.view(b, 3, self.s**3, d, h, w)      # [b, 3, s³, d, h, w]
        offset = offset / (max(d, h, w) / 2)  # 归一化偏移量
        grid = grid.unsqueeze(2) + offset     # [b, 3, s³, d, h, w]
        
        # 重塑网格用于采样
        grid = grid.permute(0, 3, 4, 5, 2, 1)  # [b, d, h, w, s³, 3]
        grid = grid.reshape(b, d, h, w*self.s**3, 3)
        
        # 步骤3：特征转换
        x = self.feature_conv(x)  # [b, out_c, d, h, w]
        
        # 步骤4：网格采样（核心：自适应采样）
        x = F.grid_sample(
            x, grid,
            mode='bilinear',      # 3D bilinear 采样
            padding_mode='zeros',
            align_corners=True
        )  # [b, out_c, d, h, w*s³]
        
        # 步骤5：重塑到目标尺寸
        x = x.view(b, self.out_channels, d, h, w, self.s, self.s, self.s)
        x = x.permute(0, 1, 2, 5, 3, 6, 4, 7)
        x = x.contiguous().view(b, self.out_channels, d*self.s, h*self.s, w*self.s)

        return x

class PixelShuffle3D(nn.Module):
    """
    3D 版 Pixel Shuffle (Voxel Shuffle)。
    将通道维度的数据重排到空间维度 (D, H, W)。
    输入: [N, C * r^3, D, H, W]
    输出: [N, C, D * r, H * r, W * r]
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        r = self.upscale_factor
        # 检查通道数是否足够
        if channels % (r ** 3) != 0:
            raise ValueError(f"Input channels {channels} must be divisible by r^3 ({r**3})")
        out_channels = channels // (r ** 3)
        # 1. Reshape: 分解通道
        x = x.view(batch_size, out_channels, r, r, r, in_depth, in_height, in_width)
        # 2. Permute: 重新排列维度
        # [N, C_out, r, r, r, D, H, W] -> [N, C_out, D, r, H, r, W, r]
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        # 3. View: 合并维度生成最终的高分辨率图
        x = x.contiguous().view(batch_size, out_channels, in_depth * r, in_height * r, in_width * r)
        return x

# --- 3D Squeeze-and-Excitation (通道注意力) ---
class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        # 3D 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 两个全连接层(用Conv3d实现1x1卷积)建立通道间的依赖关系
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1, padding=0, bias=False),
            nn.SiLU(inplace=True),  # 使用 SiLU 代替 ReLU
            nn.Conv3d(channel // reduction, channel, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        # 将学到的通道权重乘回原特征
        return x * y.expand_as(x)

def upsample_net(pth, Training, imtype, d_kernel, d_stride, d_feature_channels, d_padding,
                 g_scale, g_feature_channels):

    # ... (参数保存/加载代码保持不变) ...
    params = [d_kernel, d_stride, d_feature_channels, d_padding, g_scale, g_feature_channels]
    if Training:
        with open(pth + '_params.data', 'wb') as filehandle:
            pickle.dump(params, filehandle)
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            d_kernel, d_stride, d_feature_channels, d_padding, g_scale, g_feature_channels = pickle.load(filehandle)

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            # ---------- Layer0: PixelShuffle3D (pix), 4->8 ----------
            s = 2
            in_ch0  = g_feature_channels[0]   # 32
            out_ch0 = g_feature_channels[1]   # 1024
            exp_ch0 = out_ch0 * (s ** 3)

            self.pix = nn.Sequential(
                nn.Conv3d(in_ch0, exp_ch0, kernel_size=3, stride=1, padding=1, bias=False),
                PixelShuffle3D(upscale_factor=s),   # 4 -> 8
                nn.BatchNorm3d(out_ch0),
                nn.ReLU(inplace=True),
            )

            # ---------- pix 后：卷积裁剪一次 (valid conv), 8->6 ----------
            # 用 k=3,p=0 的 Conv3d：空间维度减 2
            # 通道保持 1024 不变，避免影响后续通道链
            self.crop_conv = nn.Sequential(
                nn.Conv3d(out_ch0, out_ch0, kernel_size=3, stride=1, padding=0, bias=False),  # 8 -> 6
                nn.BatchNorm3d(out_ch0),
                nn.ReLU(inplace=True),
            )

            # ---------- 4 x deconv, paddings = [2,2,2,3], 6->10->18->34->64 ----------
            paddings = [2, 2, 2, 3]

            chs = [
                (g_feature_channels[1], g_feature_channels[2]),  # 1024 -> 512
                (g_feature_channels[2], g_feature_channels[3]),  # 512  -> 128
                (g_feature_channels[3], g_feature_channels[4]),  # 128  -> 32
                (g_feature_channels[4], g_feature_channels[4]),  # 32   -> 32
            ]

            self.deconvs = nn.ModuleList()
            self.debns   = nn.ModuleList()
            for (in_ch, out_ch), p in zip(chs, paddings):
                self.deconvs.append(nn.ConvTranspose3d(
                    in_ch, out_ch, kernel_size=4, stride=2, padding=p, bias=False
                ))
                self.debns.append(nn.BatchNorm3d(out_ch))

            # ---------- 最后 Upsample：不变形状（平滑） ----------
            self.up = nn.Upsample(scale_factor=1, mode="trilinear", align_corners=False)

            # ---------- 输出映射：32->img_channels，保持 64 不变 ----------
            self.rcconv = nn.Conv3d(g_feature_channels[4], g_feature_channels[-1],
                                    kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            # pix
            x = self.pix(x)          # [B,1024,8,8,8]

            # pix 后裁剪一次
            x = self.crop_conv(x)    # [B,1024,6,6,6]

            # 4层反卷积
            for conv, bn in zip(self.deconvs, self.debns):
                x = F.relu_(bn(conv(x)))  # 6->10->18->34->64

            # 最后 Upsample（不变形状）
            x = self.up(x)  # 仍是 64

            out = torch.softmax(self.rcconv(x), dim=1)  # [B,3,64,64,64]
            return out





    # --- 判别器 (保持谱归一化改进) ---
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(d_kernel, d_stride, d_padding)):
                self.convs.append(
                    nn.utils.spectral_norm(
                        nn.Conv2d(d_feature_channels[lay], d_feature_channels[lay + 1], k, s, p, bias=False)
                    ))

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x

    return Discriminator, Generator

def upsample_net_conv(pth, Training, imtype, d_kernel, d_stride, d_feature_channels, d_padding,
                 g_scale, g_feature_channels):
    """
    complex_net
    """

    gk = [4] * len(g_scale)
    gp = [2] * (len(g_scale) - 1) + [3]

    params = [d_kernel, d_stride, d_feature_channels, d_padding, gk, g_scale, g_feature_channels, gp]

    if Training:
        with open(pth + '_params.data', 'wb') as filehandle:
            pickle.dump(params, filehandle)
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            d_kernel, d_stride, d_feature_channels, d_padding, gk, g_scale, g_feature_channels, gp = pickle.load(filehandle)

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.rcconv = nn.Conv3d(g_feature_channels[-2], g_feature_channels[-1], 3, 1, 0)

            for lay, (k, s, p) in enumerate(zip(gk, g_scale, gp)):
                self.convs.append(
                    nn.ConvTranspose3d(
                        g_feature_channels[lay],
                        g_feature_channels[lay + 1],
                        k, s, p,
                        bias=False
                    )
                )
                self.bns.append(nn.BatchNorm3d(g_feature_channels[lay + 1]))

        def forward(self, x):
            for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
                x = F.relu_(bn(conv(x)))

            size = (
                (x.shape[2] - 1) * 2,
                (x.shape[3] - 1) * 2,
                (x.shape[4] - 1) * 2,
            )
            up = F.interpolate(x, size=size, mode='trilinear', align_corners=False)
            out = torch.softmax(self.rcconv(up), dim=1)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(d_kernel, d_stride, d_padding)):
                self.convs.append(
                    nn.Conv2d(
                        d_feature_channels[lay],
                        d_feature_channels[lay + 1],
                        k, s, p,
                        bias=False
                    )
                )

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x

    return Discriminator, Generator

dk=[4,4,4,4,4,4]
ds=[2,2,2,2,2,2]
df=[3,64,128,256,128,1]
dp=[1,1,1,1,0]

gk=[4,4,4,4]
gs=[2,2,2,2,2]
gf=[32,1024,512,128,32,3]
gp=[2,2,2,2,3]

img_size = 64
z_channels = 32
img_channels = 3


# ======================
# 实例化网络
# ======================
D, G = upsample_net(
    pth="test_project",
    Training=True,
    imtype="nphase",
    d_kernel=dk, d_stride=ds, d_feature_channels=df, d_padding=dp,
    g_scale=gs, g_feature_channels=gf
)

G = G()
D = D()

# ======================
# 测试生成器
# ======================
z = torch.randn((8, 32, 5, 5, 5))
print("Generator input:", z.shape)

outG = G(z)
print("Generator output:", outG.shape)  # 期望: (1, 3, 64, 64, 64)

# ======================
# 取中间切片 → 判别器
# ======================
mid = outG.shape[2] // 2
slice2d = outG[:, :, mid, :, :]    # (1, 3, 64, 64)
print("Discriminator input:", slice2d.shape)

outD = D(slice2d)
print("Discriminator output:", outD.shape)