from .operations import *
from .synthesis import *
from .settings import ConfigProcessor
import numpy as np
import os
import tifffile as tif
from .restoration import *

def generator(config_path,phi):
    raise RuntimeError('The historical conditional model file was absent. Use morqex.microstructure.generate.sample(checkpoint, fractions=...) with a trained v1 diffusion checkpoint.')
    configs=ConfigProcessor(config_path)
    project_path=configs.cfg.output_path.out_pth
    pth=configs.cfg.output_path.out_pth
    image_type=configs.cfg.data.image_type
    z_channels = configs.cfg.model_params.z_channels
    d_kernel=configs.cfg.model_params.d_kernel
    d_stride = configs.cfg.model_params.d_stride
    d_feature_channels = configs.cfg.model_params.d_feature_channels
    d_padding = configs.cfg.model_params.d_padding
    g_scale = configs.cfg.model_params.g_scale
    g_feature_channels = configs.cfg.model_params.g_feature_channels
    _,netG = upsample_net(pth, True, image_type ,d_kernel,d_stride,d_feature_channels,d_padding,g_scale,g_feature_channels)
    lf=configs.cfg.generation.lf
    periodic=configs.cfg.generation.periodic
    img, raw, netG = generate_img_cond(project_path, image_type, netG(), phi ,z_channels, lf, periodic)

def save_3d_tif_to_2d_slices(
    input_tif_path: str,
    output_folder: str,
    slice_prefix: str = "slice",
    check_dim_order: bool = False
) -> None:
    """
    将三维TIF文件分割为二维切片并保存到指定文件夹

    参数:
        input_tif_path: 输入三维TIF文件的路径
        output_folder: 输出二维切片的文件夹路径
        slice_prefix: 切片文件名前缀（默认：slice）
        check_dim_order: 是否检查维度顺序（默认True，要求shape为(z,y,x)）
    
    异常:
        FileNotFoundError: 输入文件不存在
        ValueError: 输入文件不是三维数组或维度顺序不符合预期
        RuntimeError: 读取/写入文件失败
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_tif_path):
        raise FileNotFoundError(f"输入文件不存在: {input_tif_path}")
    
    # 2. 创建输出文件夹（不存在则创建）
    os.makedirs(output_folder, exist_ok=True)
    
    # 3. 读取三维TIF文件
    try:
        img_3d = tif.imread(input_tif_path)
    except Exception as e:
        raise RuntimeError(f"读取TIF文件失败: {str(e)}")
    
    # 4. 验证维度是否为3
    if img_3d.ndim != 3:
        raise ValueError(f"输入文件不是三维TIF，当前维度: {img_3d.ndim} (预期: 3)")
    
    # 5. 可选：检查维度顺序（确保z轴为第一维）
    if check_dim_order:
        # 通常三维TIF的维度顺序为 (z, y, x)，若为 (x, y, z) 则提示并转换
        # 可根据实际数据调整判断逻辑
        z_dim, y_dim, x_dim = img_3d.shape
        if z_dim > y_dim and z_dim > x_dim:  # 假设z轴长度通常小于x/y（可选判断）
            raise ValueError(
                f"维度顺序可能异常！当前shape: {img_3d.shape} (预期: (z,y,x))\n"
                "若确认维度顺序为(x,y,z)，请将check_dim_order设为False并手动调整：img_3d = np.transpose(img_3d, (2,0,1))"
            )
    
    # 6. 遍历并保存每个切片
    num_slices = img_3d.shape[0]
    num_digits = len(str(num_slices - 1))  # 计算补零位数（如1000个切片补4位）
    
    for slice_idx in range(num_slices):
        # 提取当前切片
        slice_2d = img_3d[slice_idx, :, :].astype(np.uint8)
        
        # 构建切片文件名（补零保证排序，如slice_0001.tif）
        slice_filename = f"{slice_prefix}_{slice_idx:0{num_digits}d}.tif"
        slice_save_path = os.path.join(output_folder, slice_filename)
        
        # 保存切片
        try:
            # 保留原始数据类型，不压缩
            tif.imwrite(
                slice_save_path,
                slice_2d,
                dtype=slice_2d.dtype,
                compression=None
            )
        except Exception as e:
            raise RuntimeError(f"保存切片 {slice_idx} 失败: {str(e)}")
    
    print(f"处理完成！共生成 {num_slices} 个二维切片，保存至：\n{output_folder}")

def save_3d_tif_to_2d_slices_denoise(
    input_tif_path: str,
    output_folder: str,
    slice_prefix: str = "slice",
    check_dim_order: bool = False
) -> None:
    """
    将三维TIF文件分割为二维切片并保存到指定文件夹

    参数:
        input_tif_path: 输入三维TIF文件的路径
        output_folder: 输出二维切片的文件夹路径
        slice_prefix: 切片文件名前缀（默认：slice）
        check_dim_order: 是否检查维度顺序（默认True，要求shape为(z,y,x)）
    
    异常:
        FileNotFoundError: 输入文件不存在
        ValueError: 输入文件不是三维数组或维度顺序不符合预期
        RuntimeError: 读取/写入文件失败
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_tif_path):
        raise FileNotFoundError(f"输入文件不存在: {input_tif_path}")
    
    # 2. 创建输出文件夹（不存在则创建）
    os.makedirs(output_folder, exist_ok=True)
    
    # 3. 读取三维TIF文件
    try:
        img_3d = tif.imread(input_tif_path)
    except Exception as e:
        raise RuntimeError(f"读取TIF文件失败: {str(e)}")
    
    # 4. 验证维度是否为3
    if img_3d.ndim != 3:
        raise ValueError(f"输入文件不是三维TIF，当前维度: {img_3d.ndim} (预期: 3)")
    
    # 5. 可选：检查维度顺序（确保z轴为第一维）
    if check_dim_order:
        # 通常三维TIF的维度顺序为 (z, y, x)，若为 (x, y, z) 则提示并转换
        # 可根据实际数据调整判断逻辑
        z_dim, y_dim, x_dim = img_3d.shape
        if z_dim > y_dim and z_dim > x_dim:  # 假设z轴长度通常小于x/y（可选判断）
            raise ValueError(
                f"维度顺序可能异常！当前shape: {img_3d.shape} (预期: (z,y,x))\n"
                "若确认维度顺序为(x,y,z)，请将check_dim_order设为False并手动调整：img_3d = np.transpose(img_3d, (2,0,1))"
            )
    
    # 6. 遍历并保存每个切片
    num_slices = img_3d.shape[0]
    num_digits = len(str(num_slices - 1))  # 计算补零位数（如1000个切片补4位）
    
    for slice_idx in range(num_slices):
        # 提取当前切片
        slice_2d = img_3d[slice_idx, :, :]
        slice_2d = snap_to_closest(final_advanced_pipeline(slice_2d)).astype(np.uint8)
        # 构建切片文件名（补零保证排序，如slice_0001.tif）
        slice_filename = f"{slice_prefix}_{slice_idx:0{num_digits}d}.tif"
        slice_save_path = os.path.join(output_folder, slice_filename)
        
        # 保存切片
        try:
            # 保留原始数据类型，不压缩
            tif.imwrite(
                slice_save_path,
                slice_2d,
                dtype=slice_2d.dtype,
                compression=None
            )
        except Exception as e:
            raise RuntimeError(f"保存切片 {slice_idx} 失败: {str(e)}")
    
    print(f"处理完成！共生成 {num_slices} 个二维切片，保存至：\n{output_folder}")
    

def denoise_3D_tif( input_tif_path: str,
    output_folder: 'output_2',
    slice_prefix: str = "slice",
    check_dim_order: bool = False,
    save = False):
    """
    将三维TIF文件分割为二维切片并保存到指定文件夹

    参数:
        input_tif_path: 输入三维TIF文件的路径
        output_folder: 输出二维切片的文件夹路径
        slice_prefix: 切片文件名前缀（默认：slice）
        check_dim_order: 是否检查维度顺序（默认True，要求shape为(z,y,x)）
    异常:
        FileNotFoundError: 输入文件不存在
        ValueError: 输入文件不是三维数组或维度顺序不符合预期
        RuntimeError: 读取/写入文件失败
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_tif_path):
        raise FileNotFoundError(f"输入文件不存在: {input_tif_path}")
    # 3. 读取三维TIF文件
    try:
        img_3d = tif.imread(input_tif_path)
    except Exception as e:
        raise RuntimeError(f"读取TIF文件失败: {str(e)}")
    
    # 4. 验证维度是否为3
    if img_3d.ndim != 3:
        raise ValueError(f"输入文件不是三维TIF,当前维度: {img_3d.ndim} (预期: 3)")
    img_3d=img_3d[1:,:,:]
    # 5. 可选：检查维度顺序（确保z轴为第一维）
    if check_dim_order:
        # 通常三维TIF的维度顺序为 (z, y, x)，若为 (x, y, z) 则提示并转换
        # 可根据实际数据调整判断逻辑
        z_dim, y_dim, x_dim = img_3d.shape
        if z_dim > y_dim and z_dim > x_dim:  # 假设z轴长度通常小于x/y（可选判断）
            raise ValueError(
                f"维度顺序可能异常!当前shape: {img_3d.shape} (预期: (z,y,x))\n"
                "若确认维度顺序为(x,y,z),请将check_dim_order设为False并手动调整:img_3d = np.transpose(img_3d, (2,0,1))"
            )
    
    # 6. 遍历并保存每个切片
    num_slices = img_3d.shape[0]
    num_digits = len(str(num_slices - 1))  # 计算补零位数（如1000个切片补4位）
    
    for slice_idx in range(num_slices):
        # 提取当前切片
        img_3d[slice_idx, :, :] = img_3d[slice_idx, :, :].astype(np.uint8)
        img_3d[slice_idx, :, :] = snap_to_closest(final_advanced_pipeline(img_3d[slice_idx, :, :])).astype(np.uint8)
        img_3d[:, slice_idx, :] = snap_to_closest(final_advanced_pipeline(img_3d[:, slice_idx, :])).astype(np.uint8)
        img_3d[:, :, slice_idx] = snap_to_closest(final_advanced_pipeline(img_3d[:, :, slice_idx])).astype(np.uint8)
        # 保存切片
        # 2. 创建输出文件夹（不存在则创建）
    if save == True:
        os.makedirs(output_folder, exist_ok=True)
        tiff_save_path = os.path.join(output_folder,'test_tiff')
        try:
            # 保留原始数据类型，不压缩
            tif.imwrite(
                tiff_save_path,
                img_3d,
                dtype=img_3d.dtype,
                compression=None
            )
        except Exception as e:
            raise RuntimeError(f"切片 {slice_idx} 失败: {str(e)}")
        print(f"处理完成")
    return img_3d

import numpy as np
import tifffile
from scipy import ndimage
import os
import time

def smooth_voxel_domains_iterative(input_data, kernel_size=3, iterations=5, save_path=None):
    """
    对三维体素进行迭代式区域平滑。
    通过多次执行多数投票，可以有效消除顽固的孤立点和微小的噪点团簇。

    参数:
        input_data (str or np.ndarray): 输入的tif路径或三维数组。
        kernel_size (int): 邻域大小，默认为3 (即 3x3x3)。
        iterations (int): 迭代次数。建议 3 到 10 次。次数越多，平滑越彻底（孤立点越少）。
        save_path (str): 保存路径。

    返回:
        np.ndarray: 平滑后的数组。
    """
    
    # 1. 读取数据
    if isinstance(input_data, str):
        print(f"读取文件: {input_data}")
        data = tifffile.imread(input_data)
    else:
        data = input_data.copy()

    if data.ndim != 3:
        raise ValueError("数据必须是三维的")

    # 获取所有标签
    unique_labels = np.unique(data)
    print(f"包含标签: {unique_labels}")
    print(f"平滑窗口: {kernel_size}x{kernel_size}x{kernel_size}, 迭代次数: {iterations}")

    # 定义卷积核 (3x3x3 全1)
    kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.uint8)
    
    # 当前处理的数据
    current_data = data
    
    # 2. 开始迭代循环
    for i in range(iterations):
        print(f"--- 正在执行第 {i+1} / {iterations} 次迭代 ---")
        
        # 记录本次迭代是否有变化，如果没变化可以提前结束
        changes_count = 0
        
        # 初始化：用于记录每个位置最高得票数 和 对应的赢家标签
        # 初始设为 -1 表示还没有标签占据
        max_votes = np.zeros(current_data.shape, dtype=np.int32) - 1
        winner_labels = np.zeros(current_data.shape, dtype=current_data.dtype)

        # 针对每个标签进行卷积计算
        for label in unique_labels:
            # 制作二值掩膜：当前标签为1，其他为0
            mask = (current_data == label).astype(np.uint8)
            
            # 计算周围有多少个邻居是这个标签 (包括自己)
            # 使用 constant 模式处理边界，防止边界出现奇怪的值
            votes = ndimage.convolve(mask, kernel, mode='constant', cval=0)
            
            # 竞争机制：
            # 如果当前标签的票数 > 目前记录的最大票数，则更新赢家
            # 注意：这里会自然解决 Tie (平票) 问题，后遍历到的标签如果票数相同不会替换，
            # 但由于我们是迭代的，下一轮可能会打破平衡。
            better_mask = votes > max_votes
            
            max_votes[better_mask] = votes[better_mask]
            winner_labels[better_mask] = label
        
        # 统计变化率
        diff = current_data != winner_labels
        changes_count = np.sum(diff)
        print(f"    -> 本轮修正了 {changes_count} 个体素")
        
        # 更新数据用于下一轮
        current_data = winner_labels
        
        # 如果已经没有变化，提前退出
        if changes_count == 0:
            print("    -> 数据已收敛，停止迭代。")
            break

    # 3. 输出
    if save_path:
        print(f"保存结果至: {save_path}")
        tifffile.imwrite(save_path, current_data, compression='zlib')
        
    return current_data

# ==========================================
# 强力去噪扩展版 (可选)
# 如果上面的迭代还不够，可以使用下面的函数彻底清除小于特定体积的孤立块
# 需要安装: pip install scikit-image
# ==========================================
def remove_small_islands(data, min_size=27):
    """
    强力模式：直接移除小于 min_size (例如27个像素) 的所有连通域。
    这能保证绝对没有“单独点”。
    """
    from skimage import morphology
    
    print(f"正在执行强力去噪 (移除小于 {min_size} 像素的孤立区域)...")
    cleaned_data = data.copy()
    
    # 获取所有标签
    unique_labels = np.unique(data)
    
    for label in unique_labels:
        if label == 0: continue # 假设0是背景，如果不希望背景被填补，跳过
        
        # 提取该标签的二值图
        binary_img = (data == label)
        
        # 移除小物体
        # connectivity=1 (6邻域) 或 2 (18邻域) 或 3 (26邻域)
        # 这里用 26邻域 保证斜角接触也算连接
        cleaned_binary = morphology.remove_small_objects(binary_img, min_size=min_size, connectivity=3)
        
        # 将被移除的地方标记为“待填补”（这里简单地让它变回背景，或者你需要后续再平滑一次填补空洞）
        # 注意：skimage这个函数直接把小的删了变成False。
        
        # 更新主数据：
        # 1. 先把原数据中该标签的位置清空（如果它被判定为噪点）
        # 逻辑：原图是label 且 cleaned是False -> 说明是噪点
        noise_mask = (binary_img) & (~cleaned_binary)
        
        # 这里有个问题：删掉的点填什么？
        # 简单策略：先填成众数或背景，然后再次调用上面的 smooth_voxel_domains_iterative 填补空洞
        # 这里我们暂时填成 0 (或者你可以填成 nearby 的值，但比较复杂)
        cleaned_data[noise_mask] = 0 
        
    return cleaned_data

# ==========================================
# 调用脚本
# ==========================================
