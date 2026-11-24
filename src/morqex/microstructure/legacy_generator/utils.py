import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """计算 WGAN-GP 的梯度惩罚"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def post_process_output(generated_vol, reference_img):
    """
    后处理：将生成的[-1, 1]连续体积映射回参考图像的整数分布 (相分布)
    """
    # 1. 转为 Numpy
    if isinstance(generated_vol, torch.Tensor):
        generated_vol = generated_vol.detach().cpu().numpy()
    if isinstance(reference_img, torch.Tensor):
        reference_img = reference_img.detach().cpu().numpy()
        
    vol_flat = generated_vol.flatten()
    ref_flat = reference_img.flatten()
    
    # 2. 获取参考图的唯一值（相）及其比例
    unique_vals, counts = np.unique(ref_flat, return_counts=True)
    ratios = counts / counts.sum()
    
    # 3. 对生成体积进行分位数阈值分割 (Thresholding based on Quantiles)
    # 比如参考图有 30% 孔隙(0) 和 70% 固体(1)
    # 我们就找生成体积中数值最小的 30% 设为 0，其余设为 1
    
    output_vol = np.zeros_like(generated_vol, dtype=np.uint8)
    
    current_ratio = 0
    # 对 generated_vol 进行排序以确定阈值
    sorted_vol = np.sort(vol_flat)
    
    for val, ratio in zip(unique_vals, ratios):
        # 找到分界线的数值
        limit_idx = int((current_ratio + ratio) * len(vol_flat))
        limit_idx = min(limit_idx, len(vol_flat) - 1)
        threshold = sorted_vol[limit_idx]
        
        if current_ratio == 0:
            mask = generated_vol <= threshold
        else:
            prev_limit_idx = int(current_ratio * len(vol_flat))
            prev_threshold = sorted_vol[prev_limit_idx]
            mask = (generated_vol > prev_threshold) & (generated_vol <= threshold)
            
        output_vol[mask] = int(val)
        current_ratio += ratio
        
    return output_vol

def save_checkpoint(netG, netD, epoch, path="checkpoint.pth"):
    torch.save({
        'G': netG.state_dict(),
        'D': netD.state_dict(),
        'epoch': epoch
    }, path)