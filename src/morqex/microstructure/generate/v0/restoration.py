from scipy import stats
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma
from sklearn.cluster import KMeans
from skimage.filters import unsharp_mask
from sklearn.cluster import KMeans




def advanced_denoising_pipeline(img):
    # 1. 读取图像并转换为灰度
    # 如果是彩色图读取为灰度，如果是RGBA则忽略Alpha
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Image not found.")
        return

    # 归一化到 [0, 1] 方便处理
    img_float = img.astype(np.float64) / 255.0

    # -----------------------------------------------------------
    # 步骤 1: 全变分去噪 (Total Variation Denoising) - TV-Chambolle
    # -----------------------------------------------------------
    # 这是针对分段常数图像（色块图）最先进的传统算法之一。
    # weight 参数控制平滑程度：
    #   - 太小：去噪不干净
    #   - 太大：会吃掉细小的细节（small blobs）
    # 针对你提供的图像，0.1 到 0.2 之间通常是保留细节的最佳平衡点。
    denoised_tv = denoise_tv_chambolle(img_float, weight=0.3, max_num_iter=200)

    # -----------------------------------------------------------
    # 步骤 2: 颜色量化/重分割 (K-Means Refinement)
    # -----------------------------------------------------------
    # 这一步是保留“锐利细节”的关键。
    # 原始图像看起来主要由 3 种颜色组成（黑、灰、白）。
    # TV去噪后边缘会有灰度渐变，我们用 K-Means 把它强行拉回 3 种颜色。
    
    # 准备数据给 KMeans (N, 1)
    h, w = denoised_tv.shape
    pixel_values = denoised_tv.reshape(-1, 1)
    
    # 设定聚类中心为3 (黑、灰、白)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_
    
    # 用聚类中心的值重构图像
    quantized_img = centers[labels].reshape(h, w)
    
    # -----------------------------------------------------------
    # 可视化结果
    # -----------------------------------------------------------
    # plt.figure(figsize=(15, 6))

    # # 原始图
    # plt.subplot(1, 3, 1)
    # plt.imshow(img, cmap='gray')
    # plt.title('Original Noisy Image')
    # plt.axis('off')

    # # TV 去噪结果 (平滑但保留边缘)
    # plt.subplot(1, 3, 2)
    # plt.imshow(denoised_tv, cmap='gray')
    # plt.title('Step 1: TV Denoising (Edge Preserving)')
    # plt.axis('off')

    # # 最终结果 (去噪 + 锐化重构)
    # plt.subplot(1, 3, 3)
    # plt.imshow(quantized_img, cmap='gray')
    # plt.title('Step 2: Re-Quantized (Final Result)')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    return quantized_img

def sota_detail_preserving_pipeline(img):
    # 1. 读取图像
    if img is None:
        print("Error: Image not found.")
        return

    img_float = img.astype(np.float64) / 255.0

    # -----------------------------------------------------------
    # 步骤 1: 智能噪声估计
    # -----------------------------------------------------------
    # 自动评估图像的噪声强度 sigma，避免盲目调节参数
    sigma_est = np.mean(estimate_sigma(img_float))
    print(f"Estimated Noise Sigma: {sigma_est:.4f}")

    # -----------------------------------------------------------
    # 步骤 2: Non-Local Means (非局部均值去噪) - 最关键步骤
    # -----------------------------------------------------------
    # patch_size: 搜索补丁的大小，越小越能保留微小细节 (建议 5-7)
    # patch_distance: 搜索半径 (建议 6-11)
    # h: 平滑强度。h 越高越平滑，h 越低细节越多。
    #    这里设定为 0.8 * sigma，这是一个非常保守的值，专门为了保留细节。
    # fast_mode=False: 关闭快速模式，使用更精确的计算，虽然慢但质量最高。
    
    denoised_nlm = denoise_nl_means(
        img_float, 
        h=1.2 * sigma_est, 
        sigma=sigma_est,
        fast_mode=False, 
        patch_size=5, 
        patch_distance=6
    )

    # -----------------------------------------------------------
    # 步骤 3: 细节回补 (Unsharp Masking)
    # -----------------------------------------------------------
    # NLM 虽然好，但难免会让边缘微微软化。
    # 这里我们用“反锐化掩模”把高频细节（边缘、微小颗粒）重新“提”出来。
    # radius: 锐化半径，越小越针对微细结构
    # amount: 锐化强度
    sharpened = unsharp_mask(denoised_nlm, radius=1.0, amount=1.0)

    # -----------------------------------------------------------
    # 步骤 4: 软聚类 (K-Means Refinement)
    # -----------------------------------------------------------
    h, w = sharpened.shape
    pixel_values = sharpened.reshape(-1, 1)
    
    # 强制聚类为3色 (黑、灰、白)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_
    
    # 排序中心点，保证颜色顺序一致（黑->灰->白）
    sorted_centers = np.sort(centers, axis=0)
    
    # 这里做一个映射，确保原来的黑还是黑，白还是白
    # (这一步是为了防止聚类后颜色翻转，虽然只是视觉问题)
    # 简单的方法是直接用聚类中心重构
    quantized_img = centers[labels].reshape(h, w)

    # -----------------------------------------------------------
    # 可视化对比
    # -----------------------------------------------------------
    # plt.figure(figsize=(18, 6))

    # 原始图
    # plt.subplot(1, 4, 1)
    # plt.imshow(img, cmap='gray')
    # plt.title('Original')
    # plt.axis('off')

    # # NLM 结果
    # plt.subplot(1, 4, 2)
    # plt.imshow(denoised_nlm, cmap='gray')
    # plt.title('Step 1: NL-Means (Detail Preserved)')
    # plt.axis('off')
    
    # # 锐化结果
    # plt.subplot(1, 4, 3)
    # plt.imshow(sharpened, cmap='gray')
    # plt.title('Step 2: Detail Enhanced')
    # plt.axis('off')

    # # 最终结果
    # plt.subplot(1, 4, 4)
    # plt.imshow(quantized_img, cmap='gray')
    # plt.title('Final: Segmented & Clean')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()
    
    return quantized_img

def remove_small_regions_strict(img_labels, min_size=4):
    """
    严格的拓扑过滤器：
    1. 识别所有面积 < min_size 的连通域。
    2. 并不简单删除，而是将其合并到包围它的最大邻居颜色中。
    """
    # 确保输入是整数标签图
    refined_labels = img_labels.copy().astype(np.int32)
    height, width = refined_labels.shape

    # 1. 计算连通组件 (只看上下左右4邻域，符合最严格的孤立定义)
    # connectivity=4: 仅上下左右相连算连通，对角线不算
    num_labels, labels_map, values, centroids = cv2.connectedComponentsWithStats(
        refined_labels.astype(np.uint8), connectivity=4
    )

    # values[:, 4] 是面积 (CC_STAT_AREA)
    # 找到所有面积小于 min_size 的区域的 label ID (忽略背景0，如果0也是有效颜色需谨慎)
    # 这里我们遍历所有 label，因为 label ID 只是编号，不是颜色值
    small_area_indices = np.where(values[:, cv2.CC_STAT_AREA] < min_size)[0]
    
    # 如果没有小区域，直接返回
    if len(small_area_indices) == 0:
        return refined_labels

    # 准备 3x3 的膨胀核，用于寻找邻居
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for label_idx in small_area_indices:
        # 跳过整个画布大小的背景（极少见）
        if values[label_idx, cv2.CC_STAT_AREA] >= (height * width):
            continue
            
        # 生成当前小区域的 Mask
        current_mask = (labels_map == label_idx).astype(np.uint8)
        
        # 如果 Mask 为空（可能在上一步迭代中被处理了），跳过
        if np.sum(current_mask) == 0:
            continue

        # --- 寻找邻居颜色 ---
        # 1. 膨胀当前区域
        dilated = cv2.dilate(current_mask, kernel, iterations=1)
        
        # 2. 环形边界 = 膨胀区域 - 原始区域
        boundary_mask = dilated - current_mask
        
        # 3. 获取边界上的颜色值 (从原始标签图中取)
        neighbor_colors = refined_labels[boundary_mask > 0]
        
        if len(neighbor_colors) > 0:
            # 找到出现次数最多的邻居颜色 (Mode)
            mode_res = stats.mode(neighbor_colors, keepdims=True)
            major_neighbor_color = mode_res.mode[0]
            
            # 将当前小区域的颜色 同化 为邻居颜色
            refined_labels[current_mask > 0] = major_neighbor_color
            
    return refined_labels

def final_advanced_pipeline(img):
    # 1. 读取图像
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return

    # === 【修复点】 ===
    # 确保图像一定是 uint8 类型，避免 OpenCV 报错
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    # -----------------------------------------------------------
    # 步骤 1: 预去噪 (仅去除单像素极值噪点，不破坏结构)
    # -----------------------------------------------------------
    # 使用 medianBlur 之前确保是 uint8
    img_smooth = cv2.medianBlur(img, 3)

    # -----------------------------------------------------------
    # 步骤 2: K-Means 聚类 (强制三色分段)
    # -----------------------------------------------------------
    h, w = img_smooth.shape
    pixel_values = img_smooth.reshape(-1, 1)
    
    # 聚类为3类
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_  # 形状 (3, 1)
    
    # 将中心点排序: 0->黑, 1->灰, 2->白
    # 这样我们可以用 0,1,2 这个整数标签来处理逻辑，最后再映射回颜色
    sorted_idx = np.argsort(centers.flatten())
    map_to_ordered = np.zeros(3, dtype=np.int32)
    map_to_ordered[sorted_idx] = [0, 1, 2] # 将 kmeans 的随机 label 映射为有序 label
    
    ordered_labels = map_to_ordered[kmeans_labels].reshape(h, w)
    
    # -----------------------------------------------------------
    # 步骤 3: 严格的拓扑过滤 (Min Pixel = 4)
    # -----------------------------------------------------------
    # print("正在执行拓扑去噪 (去除 < 4 像素的孤立斑点)...")
    
    # 运行第一遍
    cleaned_labels = remove_small_regions_strict(ordered_labels, min_size=4)
    
    # 运行第二遍 (因为第一遍填补后可能会产生新的微小边界，两遍通常是完美的)
    cleaned_labels = remove_small_regions_strict(cleaned_labels, min_size=4)

    # -----------------------------------------------------------
    # 步骤 4: 重建最终图像
    # -----------------------------------------------------------
    # 找回原本的灰度值 (黑、灰、白)
    sorted_centers = np.sort(centers.flatten())
    final_img = sorted_centers[cleaned_labels].astype(np.uint8)

    # -----------------------------------------------------------
    # 可视化结果
    # -----------------------------------------------------------
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(final_img, cmap='gray')
    # plt.title('Final Result (Min Isolated Size=4)')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    return final_img

def snap_to_closest(arr):
    """
    输入: 任意维度的 numpy 数组或列表
    输出: 同样形状的数组，元素变为 [0, 63, 127, 191, 255] 中最近的一个
    """
    # 1. 确保输入是 numpy 数组
    arr = np.array(arr)
    
    # 2. 定义目标值
    targets = np.array([0, 63, 127, 191, 255])
    
    # 3. 计算距离矩阵 (Broadcasting)
    # arr[..., None] 将输入增加一个维度，变为 (N, 1)
    # targets 是 (5,)
    # 结果 diff 是 (N, 5)，表示每个像素到5个目标值的绝对距离
    diff = np.abs(arr[..., np.newaxis] - targets)
    
    # 4. 找到最小距离的索引 (argmin)
    min_indices = np.argmin(diff, axis=-1)
    
    # 5. 根据索引映射回目标值
    return targets[min_indices]

