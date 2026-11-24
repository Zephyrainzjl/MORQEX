import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

# === 基础组件 ===
class ResBlock3D(nn.Module):
    """3D 残差块，用于生成器"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.BatchNorm3d(channels)
        )
    def forward(self, x):
        return x + self.conv(x)

class ResBlock2D(nn.Module):
    """2D 残差块，用于判别器"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1))
        )
    def forward(self, x):
        return x + self.conv(x)

# === 生成器 (3D) ===
class Generator(nn.Module):
    def __init__(self, z_dim=64, dim=64, out_channels=1):
        super().__init__()
        # 初始：从噪声 z 变成 4x4x4 的小方块
        self.linear = nn.Linear(z_dim, 4 * 4 * 4 * dim * 4)
        self.dim = dim
        
        self.blocks = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose3d(dim*4, dim*2, 4, 2, 1),
            nn.BatchNorm3d(dim*2),
            nn.LeakyReLU(0.2),
            ResBlock3D(dim*2),
            
            # 8 -> 16
            nn.ConvTranspose3d(dim*2, dim, 4, 2, 1),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU(0.2),
            ResBlock3D(dim),
            
            # 16 -> 32
            nn.ConvTranspose3d(dim, dim // 2, 4, 2, 1),
            nn.BatchNorm3d(dim // 2),
            nn.LeakyReLU(0.2),
            
            # 32 -> 64
            nn.ConvTranspose3d(dim // 2, out_channels, 4, 2, 1),
            nn.Tanh() # 输出范围 [-1, 1]
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, self.dim * 4, 4, 4, 4)
        return self.blocks(x)

# === 判别器 (2D) ===
class Discriminator(nn.Module):
    def __init__(self, dim=64, in_channels=1):
        super().__init__()
        # 接收 64x64 的切片
        self.net = nn.Sequential(
            # 64 -> 32
            spectral_norm(nn.Conv2d(in_channels, dim, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            # 32 -> 16
            spectral_norm(nn.Conv2d(dim, dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            ResBlock2D(dim * 2),
            
            # 16 -> 8
            spectral_norm(nn.Conv2d(dim * 2, dim * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            # 8 -> 4
            spectral_norm(nn.Conv2d(dim * 4, dim * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            # 4 -> 1 (Logits)
            nn.Conv2d(dim * 8, 1, 4, 1, 0) 
        )

    def forward(self, x):
        return self.net(x).view(-1)