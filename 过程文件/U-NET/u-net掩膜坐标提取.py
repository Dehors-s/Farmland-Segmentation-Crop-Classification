import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_contours_from_mask(mask_path, min_area=100, epsilon_factor=0.001):
    """
    从掩膜图像中提取轮廓坐标
    
    Args:
        mask_path: 掩膜图像路径
        min_area: 忽略小于此面积的轮廓
        epsilon_factor: 轮廓近似系数 (越小越精细，越大越平滑)
        
    Returns:
        image_shape: (height, width)
        polygons: 轮廓点列表 [[[x, y], ...], ...]
    """
    # 读取掩膜 (灰度模式)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩膜: {mask_path}")
        
    h, w = mask.shape
    
    # 二值化 (确保是0和255)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    # cv2.RETR_EXTERNAL: 只检测外轮廓
    # cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线段
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        # 轮廓近似 (减少点数，使更平滑)
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 转换为列表格式 [[x, y], [x, y], ...]
        # approx shape is (N, 1, 2) -> (N, 2)
        points = approx.reshape(-1, 2).tolist()
        
        # 只有多于2个点的才能构成多边形
        if len(points) > 2:
            polygons.append(points)
            
    return (h, w), polygons

def save_to_json(polygons, image_path, output_json_path, image_shape):
    """
    保存轮廓到LabelMe格式的JSON文件
    """
    data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image_shape[0],
        "imageWidth": image_shape[1]
    }
    
    for poly in polygons:
        shape = {
            "label": "farmland",
            "points": poly,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        data["shapes"].append(shape)
        
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def visualize_contours(image_path, polygons, output_vis_path):
    """
    在原图上可视化提取的轮廓
    """
    # 读取原图
    # 如果原图不存在，创建一个黑色背景
    if os.path.exists(image_path):
        # 处理中文路径
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"警告: 无法读取原图 {image_path}，使用黑色背景")
            img = np.zeros((512, 512, 3), dtype=np.uint8) # 默认大小，后面会被resize或覆盖
    else:
        print(f"警告: 原图不存在 {image_path}，使用黑色背景")
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    # 绘制轮廓
    vis_img = img.copy()
    
    for poly in polygons:
        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # 绘制轮廓线 (绿色, 宽度2)
        cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
        # 填充半透明颜色 (红色)
        # overlay = vis_img.copy()
        # cv2.fillPoly(overlay, [pts], (0, 0, 255))
        # cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)

    # 保存可视化结果
    # cv2.imencode 支持中文路径
    cv2.imencode('.jpg', vis_img)[1].tofile(str(output_vis_path))

def main():
    """
    主函数：集成参数配置
    """
    # ==================== 参数配置区域 ====================
    
    # 1. 掩膜目录 (推理输出的 predictions_v7 目录)
    default_mask_dir = r"./test"
    
    # 2. 原图目录 (用于可视化背景，如果找不到会用黑色背景)
    # 如果推理结果在 predictions_v7，通常原图在 data/test 之类的地方
    # 这里需要用户指定，或者我们尝试从掩膜文件名反推
    default_image_dir = r"D:\Work space\DeepLearning\farm\U-NET\img.png"
    
    # 3. 输出目录 (保存json和可视化图)
    default_output_dir = r"./predictions_v7_json"
    
    # 4. 掩膜后缀 (匹配哪种掩膜文件)
    # 选项: _mask.png (最终), _seg_mask.png (仅分割), _boundary.png (仅边界)
    default_mask_suffix = "_mask.png"
    
    # 5. 最小轮廓面积 (过滤噪点)
    default_min_area = 50
    
    # ====================================================

    parser = argparse.ArgumentParser(description="U-Net 掩膜坐标提取工具")
    parser.add_argument('--mask_dir', type=str, default=default_mask_dir, help='掩膜文件所在目录')
    parser.add_argument('--image_dir', type=str, default=default_image_dir, help='原始图像所在目录 (可选, 用于可视化)')
    parser.add_argument('--output_dir', type=str, default=default_output_dir, help='结果输出目录')
    parser.add_argument('--suffix', type=str, default=default_mask_suffix, help='要处理的掩膜文件后缀')
    
    args = parser.parse_args()
    
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None
    
    if not mask_dir.exists():
        print(f"错误: 掩膜目录不存在: {mask_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualization"
    json_dir = output_dir / "json"
    vis_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)
    
    # 查找所有掩膜文件
    mask_files = list(mask_dir.glob(f"*{args.suffix}"))
    print(f"在 {mask_dir} 中找到 {len(mask_files)} 个掩膜文件 (后缀: {args.suffix})")
    
    success_count = 0
    
    for mask_file in tqdm(mask_files, desc="处理中"):
        try:
            # 1. 提取轮廓
            image_shape, polygons = extract_contours_from_mask(mask_file, min_area=default_min_area)
            
            # 推断原图文件名 (假设掩膜文件名是 name_mask.png -> name.png)
            # 需要根据后缀长度截断
            stem = mask_file.name.replace(args.suffix, "")
            # 尝试常见的图片扩展名
            image_name = f"{stem}.png" # 默认假设
            
            # 如果提供了原图目录，尝试寻找真实的原图
            original_image_path = image_name # 默认用于JSON记录
            real_image_path = None # 用于读取
            
            if image_dir and image_dir.exists():
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    probe_path = image_dir / f"{stem}{ext}"
                    if probe_path.exists():
                        real_image_path = probe_path
                        image_name = probe_path.name
                        original_image_path = image_name
                        break
            
            # 2. 保存 JSON
            json_path = json_dir / f"{stem}.json"
            save_to_json(polygons, original_image_path, json_path, image_shape)
            
            # 3. 可视化
            vis_path = vis_dir / f"{stem}_vis.jpg"
            # 如果找到了真实原图，用它；否则尝试用掩膜本身或黑色背景
            vis_source = real_image_path if real_image_path else mask_file
            visualize_contours(vis_source, polygons, vis_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"处理 {mask_file.name} 时出错: {e}")
            
    print(f"\n处理完成!")
    print(f"成功: {success_count}/{len(mask_files)}")
    print(f"JSON文件保存在: {json_dir}")
    print(f"可视化图保存在: {vis_dir}")

if __name__ == "__main__":
    main()
