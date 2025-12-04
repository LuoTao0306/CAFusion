import os
import cv2
import numpy as np
from tqdm import tqdm


def ensure_dir(dir_path):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def merge_yuv_channels(gray_img_path, color_img_path, output_path=None, resize=True):
    """
    将灰度图像的Y通道与彩色图像的UV通道合并

    参数:
    gray_img_path: 灰度图像路径
    color_img_path: 彩色图像路径
    output_path: 输出图像路径，若为None则返回合并后的图像
    resize: 是否调整图像大小以匹配
    """
    # 读取图像
    gray_img = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(color_img_path, cv2.IMREAD_COLOR)

    if gray_img is None:
        raise ValueError(f"无法读取灰度图像: {gray_img_path}")
    if color_img is None:
        raise ValueError(f"无法读取彩色图像: {color_img_path}")

    # 调整图像大小（如果需要）
    if resize:
        gray_h, gray_w = gray_img.shape[:2]
        color_img = cv2.resize(color_img, (gray_w, gray_h))

    # 将彩色图像转换为YUV颜色空间
    yuv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_img)

    # 用灰度图像替换Y通道
    merged_yuv = cv2.merge([gray_img, u, v])

    # 转回BGR颜色空间
    merged_bgr = cv2.cvtColor(merged_yuv, cv2.COLOR_YUV2BGR)

    # 保存或返回结果
    if output_path:
        cv2.imwrite(output_path, merged_bgr)
        return output_path
    else:
        return merged_bgr


def process_image_folders(gray_dir, color_dir, output_dir, resize=True, suffix="_merged"):
    """
    处理两个文件夹中的图像，将灰度图像的Y通道与彩色图像的UV通道合并

    参数:
    gray_dir: 灰度图像文件夹路径
    color_dir: 彩色图像文件夹路径
    output_dir: 输出图像文件夹路径
    resize: 是否调整图像大小以匹配
    suffix: 输出文件名后缀
    """
    # 确保输出目录存在
    ensure_dir(output_dir)

    # 获取灰度图像文件列表
    gray_files = {os.path.splitext(f)[0]: f for f in os.listdir(gray_dir)
                  if os.path.isfile(os.path.join(gray_dir, f))}

    # 获取彩色图像文件列表
    color_files = {os.path.splitext(f)[0]: f for f in os.listdir(color_dir)
                   if os.path.isfile(os.path.join(color_dir, f))}

    # 找出匹配的文件名
    common_names = set(gray_files.keys()) & set(color_files.keys())

    if not common_names:
        print("警告: 灰度图像文件夹和彩色图像文件夹中没有找到匹配的文件名!")
        return

    print(f"找到 {len(common_names)} 对匹配的图像")

    # 处理每对图像
    for name in tqdm(common_names, desc="处理图像"):
        gray_path = os.path.join(gray_dir, gray_files[name])
        color_path = os.path.join(color_dir, color_files[name])

        # 生成输出文件名
        base_name, ext = os.path.splitext(gray_files[name])
        output_name = f"{base_name}{suffix}{ext}"
        output_path = os.path.join(output_dir, output_name)

        try:
            # 合并图像
            merge_yuv_channels(gray_path, color_path, output_path, resize)
        except Exception as e:
            print(f"处理图像 {name} 时出错: {e}")


if __name__ == "__main__":
    # 直接在代码中设置参数
    GRAY_DIR =  r"D:\coding\pycharm_projects\paper3\CAFusion\fusion_out\Y\SPECT"  # 灰度图像文件夹路径
    # GRAY_DIR = r"D:\Dataset\fused_img_dataset\med\DWT\PET_MRI"  # 灰度图像文件夹路径
    # GRAY_DIR = r"D:\coding\pycharm_projects\paper3\11_5\fusion_out\test\path_204"
    # GRAY_DIR = r"D:\coding\pycharm_projects\paper3\11_5\fusion_out\test\gfp"
    # GRAY_DIR = r"D:\coding\pycharm_projects\paper3\11_5\fusion_out\gfp_pet_pc_mri\path_14"
    COLOR_DIR = r"D:\Dataset\MY_DATASET(2025-2-27)\SPECT-MRI\test_data\SPECT-MRI\SPECT"  # 彩色图像文件夹路径
    # COLOR_DIR = r"D:\Dataset\MY_DATASET(2025-2-27)\PET-MRI\Test_Data\PET-MRI\PET"
    # COLOR_DIR = r"D:\Dataset\GFP-PC\test1\GFP"
    # COLOR_DIR = r"D:\Dataset\MSRS-main\MSRS-main\test\vi"

    OUTPUT_DIR = r"D:\coding\pycharm_projects\paper3\CAFusion\fusion_out\RGB\SPECT"  # 输出图像文件夹路径
    RESIZE = True  # 是否调整图像大小以匹配
    SUFFIX = "" # 输出文件名后缀

    # 处理图像文件夹
    process_image_folders(
        gray_dir=GRAY_DIR,
        color_dir=COLOR_DIR,
        output_dir=OUTPUT_DIR,
        resize=RESIZE,
        suffix=SUFFIX
    )