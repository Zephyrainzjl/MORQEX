import os
import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入自定义模块
from .network import Generator, Discriminator
from .utils import compute_gradient_penalty, post_process_output, save_checkpoint

# === 参数设置 ===
CONFIG = {
    'image_path': 'data/training_slice.png', # 输入图片路径
    'img_size': 64,       # 生成立方体大小 64x64x64
    'z_dim': 64,          # 噪声维度
    'batch_size': 8,     # 显存不够可调小
    'epochs': 2000,       # 训练轮数 (SliceGAN收敛较慢，建议多跑)
    'lr': 1e-4,           # 学习率
    'lambda_gp': 10,      # 梯度惩罚权重
    'n_critic': 5,        # 判别器训练次数 per 生成器步
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def create_dummy_data_if_not_exists():
    """如果没有训练图片，生成一张模拟的多孔结构图"""
    if not os.path.exists('data'): os.makedirs('data')
    if not os.path.exists(CONFIG['image_path']):
        print("未找到训练图片，正在生成模拟数据...")
        from scipy.ndimage import gaussian_filter
        # 生成类似海绵的结构
        noise = np.random.randn(256, 256)
        noise = gaussian_filter(noise, sigma=3)
        img = (noise > 0).astype(np.uint8) * 255 # 0 和 255
        Image.fromarray(img).resize((64,64)).save(CONFIG['image_path'])
        print(f"模拟数据已保存至 {CONFIG['image_path']}")

def load_data():
    img = Image.open(CONFIG['image_path']).convert('L') # 转灰度
    img = img.resize((CONFIG['img_size'], CONFIG['img_size']))
    arr = np.array(img)
    # 归一化到 [-1, 1] 用于 GAN
    arr = (arr.astype(np.float32) - 127.5) / 127.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64)

def main_process():
    os.makedirs('output', exist_ok=True)
    create_dummy_data_if_not_exists()
    
    device = CONFIG['device']
    
    # 1. 准备数据
    real_img = load_data().to(device)
    # 我们通过重复来模拟 Batch (因为只有一张图)
    
    # 2. 初始化网络
    netG = Generator(z_dim=CONFIG['z_dim'], dim=64).to(device)
    netD = Discriminator(dim=64).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.9))
    
    print(f"开始训练 on {device}...")
    
    # 3. 训练循环
    pbar = tqdm(range(CONFIG['epochs']))
    for epoch in pbar:
        
        # === 训练判别器 (Critic) ===
        for _ in range(CONFIG['n_critic']):
            netD.zero_grad()
            
            # (A) 真实样本
            # 对真实图片加一点微小的噪声可以增加稳定性
            real_data = real_img.repeat(CONFIG['batch_size'], 1, 1, 1)
            real_validity = netD(real_data)
            
            # (B) 生成样本
            z = torch.randn(CONFIG['batch_size'], CONFIG['z_dim'], device=device)
            fake_vol = netG(z).detach() # 此时不更新 G
            
            # (C) SliceGAN 核心：切片提取
            # 从生成的 3D 体积中随机切取 X, Y, Z 三个方向的切片
            # fake_vol shape: (B, 1, D, H, W)
            
            perm1 = fake_vol.permute(0, 1, 3, 2, 4).reshape(-1, 1, CONFIG['img_size'], CONFIG['img_size']) # XZ plane
            perm2 = fake_vol.permute(0, 1, 4, 2, 3).reshape(-1, 1, CONFIG['img_size'], CONFIG['img_size']) # YZ plane
            perm3 = fake_vol.reshape(-1, 1, CONFIG['img_size'], CONFIG['img_size']) # XY plane
            
            # 为了效率，我们随机选一些切片，而不是全部
            # 这里简单起见，我们将三个视角的切片拼起来，然后随机抽 batch_size 个
            all_slices = torch.cat([perm1, perm2, perm3], dim=0)
            indices = torch.randperm(all_slices.size(0))[:CONFIG['batch_size']]
            fake_slices = all_slices[indices]
            
            fake_validity = netD(fake_slices)
            
            # (D) Loss & Backprop (WGAN Loss)
            # Loss = D(fake) - D(real) + gradient_penalty
            gp = compute_gradient_penalty(netD, real_data, fake_slices, device)
            d_loss = fake_validity.mean() - real_validity.mean() + CONFIG['lambda_gp'] * gp
            
            d_loss.backward()
            optimizerD.step()
            
        # === 训练生成器 (Generator) ===
        netG.zero_grad()
        
        z = torch.randn(CONFIG['batch_size'], CONFIG['z_dim'], device=device)
        fake_vol = netG(z)
        
        # 同样的切片逻辑，但这次我们需要梯度回传给 G
        perm1 = fake_vol.permute(0, 1, 3, 2, 4).reshape(-1, 1, CONFIG['img_size'], CONFIG['img_size'])
        perm2 = fake_vol.permute(0, 1, 4, 2, 3).reshape(-1, 1, CONFIG['img_size'], CONFIG['img_size'])
        perm3 = fake_vol.reshape(-1, 1, CONFIG['img_size'], CONFIG['img_size'])
        
        all_slices = torch.cat([perm1, perm2, perm3], dim=0)
        # 训练 G 时，我们希望 D 把所有切片都判别为真
        # 为了显存，可以只判别一部分，但最好判别大部分以保证各向同性
        # 这里我们判别所有切片的一个随机子集 (3 * batch_size)
        indices = torch.randperm(all_slices.size(0))[:CONFIG['batch_size']*3]
        selected_slices = all_slices[indices]
        
        fake_validity = netD(selected_slices)
        g_loss = -fake_validity.mean()
        
        g_loss.backward()
        optimizerG.step()
        
        # === 打印与保存 ===
        pbar.set_description(f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        
        if epoch % 100 == 0:
            # 推理并保存结果
            with torch.no_grad():
                sample_z = torch.randn(1, CONFIG['z_dim'], device=device)
                generated_vol = netG(sample_z)[0, 0] # (D, H, W)
                
                # 后处理为整数结构 (物理相)
                # 读取原始整数图用于计算比例
                ref_img_int = np.array(Image.open(CONFIG['image_path']).convert('L').resize((64,64)))
                final_vol = post_process_output(generated_vol, ref_img_int)
                
                # 保存中心切片图
                plt.figure()
                plt.imshow(final_vol[32], cmap='gray')
                plt.title(f"Epoch {epoch}")
                plt.savefig(f"output/epoch_{epoch}.png")
                plt.close()
                
                # 保存 Numpy 数组
                np.save(f"output/vol_{epoch}.npy", final_vol)

    print("训练完成！结果已保存在 output 文件夹。")

# if __name__ == "__main__":
#     main()