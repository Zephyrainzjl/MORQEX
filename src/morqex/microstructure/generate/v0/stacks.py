import tifffile as tif
import numpy as np
import os


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
    
    print(f"✅ 处理完成！共生成 {num_slices} 个二维切片，保存至：\n{output_folder}")


# ------------------- 示例用法 -------------------
# if __name__ == "__main__":
#     # 替换为你的实际路径
#     INPUT_3D_TIF = r"E:\data\3d_image.tif"    # 输入三维TIF路径
#     OUTPUT_FOLDER = r"E:\data\2d_slices"      # 输出文件夹路径

#     try:
#         save_3d_tif_to_2d_slices(
#             input_tif_path=INPUT_3D_TIF,
#             output_folder=OUTPUT_FOLDER,
#             slice_prefix="cell_slice"  # 自定义切片前缀
#         )
#     except Exception as e:
#         print(f"❌ 处理失败：{str(e)}")