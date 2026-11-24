import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

from PIL import Image
import tifffile


def batch_crop_two_slice_folders(
    input_dir1: str,
    input_dir2: str,
    output_dir1: str,
    output_dir2: str,
    crop_size: Optional[Tuple[int, int]] = None,
    use_smaller_common_size: bool = True,
    max_slices: Optional[int] = None,
    crop_mode: str = "center",
    allowed_exts: Optional[Tuple[str, ...]] = None,
    sort_key=None,
    keep_original_names: bool = True,
    output_format: Optional[str] = None,
    tif_compression: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    对两个文件夹中的切片图像进行批量裁剪，并输出到两个新文件夹。

    参数
    ----------
    input_dir1, input_dir2 : str
        两个输入文件夹路径。每个文件夹中是一组二维切片图像。

    output_dir1, output_dir2 : str
        两个输出文件夹路径。

    crop_size : tuple[int, int] or None
        指定裁剪尺寸 (height, width)。
        如果为 None，则在 use_smaller_common_size=True 时自动取两组中可共同裁剪的最小尺寸。

    use_smaller_common_size : bool
        是否自动裁剪到两组图像中较小的公共尺寸。
        - True: 自动取所有待处理图像中最小的高和最小的宽
        - False: 必须显式提供 crop_size

    max_slices : int or None
        输出切片数量。
        - None: 自动使用两个文件夹中较少的那个数量
        - 指定整数: 实际输出数量 = min(max_slices, 两组较少数量)

    crop_mode : str
        裁剪模式，目前支持:
        - "center": 中心裁剪

    allowed_exts : tuple[str, ...] or None
        允许的图像扩展名。默认为常见格式。

    sort_key : callable or None
        文件排序函数。默认按文件名排序。

    keep_original_names : bool
        是否保留原始文件名。
        - True: 输出文件名沿用原名
        - False: 输出为 slice_0000.xxx 形式

    output_format : str or None
        强制输出格式，例如:
        - "tif"
        - "png"
        - None: 保持原始扩展名

    tif_compression : str or None
        tif 保存压缩方式，例如:
        - None
        - "zlib"
        - "deflate"

    verbose : bool
        是否打印处理信息。
    """

    if allowed_exts is None:
        allowed_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

    input_dir1 = Path(input_dir1)
    input_dir2 = Path(input_dir2)
    output_dir1 = Path(output_dir1)
    output_dir2 = Path(output_dir2)

    output_dir1.mkdir(parents=True, exist_ok=True)
    output_dir2.mkdir(parents=True, exist_ok=True)

    if not input_dir1.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir1}")
    if not input_dir2.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir2}")

    def list_image_files(folder: Path) -> List[Path]:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts]
        if sort_key is None:
            files = sorted(files, key=lambda x: x.name)
        else:
            files = sorted(files, key=sort_key)
        return files

    def read_image(path: Path) -> np.ndarray:
        ext = path.suffix.lower()
        if ext in (".tif", ".tiff"):
            arr = tifffile.imread(str(path))
        else:
            with Image.open(path) as img:
                arr = np.array(img)
        return arr

    def save_image(path: Path, arr: np.ndarray) -> None:
        ext = path.suffix.lower()
        if ext in (".tif", ".tiff"):
            tifffile.imwrite(str(path), arr, compression=tif_compression)
        else:
            if arr.dtype == bool:
                arr = arr.astype(np.uint8) * 255

            if arr.dtype not in (np.uint8, np.uint16):
                arr_to_save = arr
                if np.issubdtype(arr.dtype, np.floating):
                    # 尝试安全转换到 uint8，仅用于普通图片格式
                    arr_min = arr.min()
                    arr_max = arr.max()
                    if arr_max > arr_min:
                        arr_to_save = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
                    else:
                        arr_to_save = np.zeros_like(arr, dtype=np.uint8)
                else:
                    arr_to_save = arr.astype(np.uint8)
            else:
                arr_to_save = arr

            Image.fromarray(arr_to_save).save(str(path))

    def get_hw(arr: np.ndarray) -> Tuple[int, int]:
        if arr.ndim < 2:
            raise ValueError(f"图像维度异常，至少应为2维，实际 shape={arr.shape}")
        return arr.shape[0], arr.shape[1]

    def center_crop(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = get_hw(arr)
        if target_h > h or target_w > w:
            raise ValueError(
                f"目标裁剪尺寸 ({target_h}, {target_w}) 大于图像尺寸 ({h}, {w})"
            )
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        if arr.ndim == 2:
            return arr[top:top + target_h, left:left + target_w]
        else:
            return arr[top:top + target_h, left:left + target_w, ...]

    def crop_array(arr: np.ndarray, target_h: int, target_w: int, mode: str) -> np.ndarray:
        if mode == "center":
            return center_crop(arr, target_h, target_w)
        else:
            raise ValueError(f"暂不支持的 crop_mode: {mode}")

    files1 = list_image_files(input_dir1)
    files2 = list_image_files(input_dir2)

    if len(files1) == 0:
        raise ValueError(f"文件夹中未找到支持的图像文件: {input_dir1}")
    if len(files2) == 0:
        raise ValueError(f"文件夹中未找到支持的图像文件: {input_dir2}")

    common_n = min(len(files1), len(files2))
    if max_slices is None:
        out_n = common_n
    else:
        if max_slices <= 0:
            raise ValueError("max_slices 必须大于 0")
        out_n = min(max_slices, common_n)

    files1 = files1[:out_n]
    files2 = files2[:out_n]

    if verbose:
        print(f"文件夹1图像数: {len(list_image_files(input_dir1))}")
        print(f"文件夹2图像数: {len(list_image_files(input_dir2))}")
        print(f"实际输出切片数: {out_n}")

    # 决定裁剪尺寸
    if crop_size is not None:
        target_h, target_w = crop_size
        if target_h <= 0 or target_w <= 0:
            raise ValueError("crop_size 必须为正整数")
    else:
        if not use_smaller_common_size:
            raise ValueError("当 crop_size=None 时，use_smaller_common_size 必须为 True")

        min_h = float("inf")
        min_w = float("inf")

        for p in files1 + files2:
            arr = read_image(p)
            h, w = get_hw(arr)
            min_h = min(min_h, h)
            min_w = min(min_w, w)

        target_h, target_w = int(min_h), int(min_w)

    if verbose:
        print(f"裁剪尺寸: (H={target_h}, W={target_w})")

    # 批量处理
    for idx, (p1, p2) in enumerate(zip(files1, files2)):
        arr1 = read_image(p1)
        arr2 = read_image(p2)

        h1, w1 = get_hw(arr1)
        h2, w2 = get_hw(arr2)

        if target_h > h1 or target_w > w1:
            raise ValueError(
                f"文件夹1中的图像 {p1.name} 尺寸为 ({h1}, {w1})，小于目标裁剪尺寸 ({target_h}, {target_w})"
            )
        if target_h > h2 or target_w > w2:
            raise ValueError(
                f"文件夹2中的图像 {p2.name} 尺寸为 ({h2}, {w2})，小于目标裁剪尺寸 ({target_h}, {target_w})"
            )

        cropped1 = crop_array(arr1, target_h, target_w, crop_mode)
        cropped2 = crop_array(arr2, target_h, target_w, crop_mode)

        if keep_original_names:
            name1 = p1.stem
            name2 = p2.stem
        else:
            name1 = f"slice_{idx:04d}"
            name2 = f"slice_{idx:04d}"

        ext1 = output_format.lower() if output_format is not None else p1.suffix.lower().lstrip(".")
        ext2 = output_format.lower() if output_format is not None else p2.suffix.lower().lstrip(".")

        out_path1 = output_dir1 / f"{name1}.{ext1}"
        out_path2 = output_dir2 / f"{name2}.{ext2}"

        save_image(out_path1, cropped1)
        save_image(out_path2, cropped2)

        if verbose:
            print(f"[{idx + 1}/{out_n}] 保存: {out_path1.name} | {out_path2.name}")

    if verbose:
        print("处理完成。")
        
