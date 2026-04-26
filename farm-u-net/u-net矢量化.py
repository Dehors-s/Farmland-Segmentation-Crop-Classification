# u-net矢量化.py
# 集成功能：模型推理 -> 掩膜生成 -> 轮廓提取(矢量化) -> JSON保存
# 结合了 u-net分割推理--cbamV7.py 和 u-net掩膜坐标提取.py 的功能

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from tqdm import tqdm
import json
import warnings
from itertools import product

warnings.filterwarnings('ignore')


# ============================================================================
# 1. 模型定义 (必须与训练代码 u-net--CBAMV7.py 完全一致)
# ============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class CBAMDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_cbam=True, dropout_rate=0.2):
        super(CBAMDecoderBlock, self).__init__()
        self.use_cbam = use_cbam
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        if use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        if self.use_cbam:
            x = self.cbam(x)

        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        return x


class CBAMUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, use_cbam=True, dropout_rate=0.2):
        super(CBAMUNet, self).__init__()
        import torchvision.models as models

        base_model = models.resnet34(weights=None)

        self.encoder1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.encoder2 = base_model.layer1
        self.encoder3 = base_model.layer2
        self.encoder4 = base_model.layer3
        self.encoder5 = base_model.layer4

        encoder_channels = [64, 64, 128, 256, 512]
        decoder_channels = [256, 128, 64, 32]

        self.bridge = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = CBAMDecoderBlock(decoder_channels[0], encoder_channels[-2], decoder_channels[1], use_cbam, dropout_rate)
        self.decoder3 = CBAMDecoderBlock(decoder_channels[1], encoder_channels[-3], decoder_channels[2], use_cbam, dropout_rate)
        self.decoder2 = CBAMDecoderBlock(decoder_channels[2], encoder_channels[-4], decoder_channels[3], use_cbam, dropout_rate)
        self.decoder1 = CBAMDecoderBlock(decoder_channels[3], encoder_channels[-5], decoder_channels[3], use_cbam, dropout_rate)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        b = self.bridge(e5)
        d4 = self.decoder4(b, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        return d1


class MultiTaskUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, use_cbam=True, dropout_rate=0.2):
        super(MultiTaskUNet, self).__init__()

        self.base = CBAMUNet(encoder_name, encoder_weights, in_channels, classes, use_cbam, dropout_rate)

        self.boundary_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        self.distance_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.seg_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(16 + 1 + 1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(16, classes, kernel_size=1)
        )

    def forward(self, x):
        d1 = self.base(x)

        boundary_logit = self.boundary_head(d1)
        distance_logit = self.distance_head(d1)

        seg_feat = self.seg_up(d1)

        if boundary_logit.shape[2:] != x.shape[2:]:
            boundary_logit = F.interpolate(boundary_logit, size=x.shape[2:], mode='bilinear', align_corners=True)
            distance_logit = F.interpolate(distance_logit, size=x.shape[2:], mode='bilinear', align_corners=True)
            seg_feat = F.interpolate(seg_feat, size=x.shape[2:], mode='bilinear', align_corners=True)

        boundary_pred = torch.sigmoid(boundary_logit)
        distance_pred = distance_logit
        cat_feat = torch.cat([seg_feat, boundary_pred, distance_pred], dim=1)
        seg_logit = self.fusion(cat_feat)

        return seg_logit, boundary_logit, distance_logit


# ============================================================================
# 2. 预测器类
# ============================================================================
class Predictor:
    def __init__(self, model_path, device='cuda', dropout_rate=0.2, encoder_name='resnet34'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.model = MultiTaskUNet(
            encoder_name=encoder_name,
            encoder_weights=None,
            use_cbam=True,
            dropout_rate=dropout_rate
        ).to(self.device)

        self.load_weights(model_path)

        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])

    def load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件未找到: {path}")

        print(f"加载模型权重: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            best_iou = checkpoint.get('best_iou', 'Unknown')
            epoch = checkpoint.get('epoch', 'Unknown')
            print(f"训练轮数: {epoch}")
            print(f"模型最佳 IoU: {best_iou}")
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess(self, image_path):
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        original_shape = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        img_tensor = augmented['image'].unsqueeze(0).to(self.device)

        return img_tensor, img, original_shape

    def predict(self, image_path, threshold=0.5, boundary_threshold=0.1):
        img_tensor, original_img, original_shape = self.preprocess(image_path)

        with torch.no_grad():
            seg_logits, boundary_logits, distance_logits = self.model(img_tensor)

            seg_prob = torch.sigmoid(seg_logits)
            boundary_prob = torch.sigmoid(boundary_logits)
            distance_prob = distance_logits

        seg_mask = seg_prob.squeeze().cpu().numpy()
        boundary_mask = boundary_prob.squeeze().cpu().numpy()
        distance_map = distance_prob.squeeze().cpu().numpy()

        seg_mask = cv2.resize(seg_mask, (original_shape[1], original_shape[0]))
        boundary_mask = cv2.resize(boundary_mask, (original_shape[1], original_shape[0]))
        distance_map = cv2.resize(distance_map, (original_shape[1], original_shape[0]))

        binary_mask = (seg_mask > threshold).astype(np.uint8) * 255
        binary_boundary = (boundary_mask > boundary_threshold).astype(np.uint8) * 255

        return binary_mask, binary_boundary, original_img, seg_mask, boundary_mask, distance_map

    def process_and_save(self, save_dir, filename, binary_mask, boundary_mask, original_img, 
                         prob_mask=None, prob_boundary=None, distance_map=None,
                         save_overlay=True, save_prob=False, boundary_erode_iter=0):
        """
        处理掩膜(去除边界)并保存图片结果
        返回处理后的最终掩膜(final_mask)供后续矢量化使用
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(filename).stem

        # 保存分割掩膜
        cv2.imencode('.png', binary_mask)[1].tofile(str(save_dir / f"{stem}_seg_mask.png"))
        
        # 保存边界掩膜
        cv2.imencode('.png', boundary_mask)[1].tofile(str(save_dir / f"{stem}_boundary.png"))

        processed_boundary = boundary_mask.copy()
        if boundary_erode_iter and boundary_erode_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            processed_boundary = cv2.erode(processed_boundary, kernel, iterations=boundary_erode_iter)

        # 生成最终掩膜：从分割结果中减去边界
        final_mask = binary_mask.copy()
        final_mask[processed_boundary > 0] = 0
        
        # 保存最终掩膜
        cv2.imencode('.png', final_mask)[1].tofile(str(save_dir / f"{stem}_mask.png"))

        if save_overlay:
            overlay = original_img.copy()
            red_mask = np.zeros_like(original_img)
            red_mask[binary_mask == 255] = [255, 0, 0]
            green_mask = np.zeros_like(original_img)
            green_mask[processed_boundary > 0] = [0, 255, 0]

            mask_combined = cv2.addWeighted(red_mask, 1.0, green_mask, 1.0, 0)
            overlay = cv2.addWeighted(overlay, 0.7, mask_combined, 0.3, 0)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imencode('.jpg', overlay_bgr)[1].tofile(str(save_dir / f"{stem}_overlay.jpg"))

        if save_prob and distance_map is not None:
            distance_uint8 = (distance_map * 255).astype(np.uint8)
            cv2.imencode('.png', distance_uint8)[1].tofile(str(save_dir / f"{stem}_distance.png"))
            if prob_mask is not None:
                cv2.imencode('.png', (prob_mask * 255).astype(np.uint8))[1].tofile(str(save_dir / f"{stem}_prob_mask.png"))
            if prob_boundary is not None:
                cv2.imencode('.png', (prob_boundary * 255).astype(np.uint8))[1].tofile(str(save_dir / f"{stem}_prob_boundary.png"))

        return final_mask


# ============================================================================
# 3. 矢量化工具函数
# ============================================================================
def extract_contours_from_mask_array(mask, min_area=100, epsilon_factor=0.001,
                                     morph_kernel=0, morph_iter=1):
    """
    从掩膜数组(numpy)中提取轮廓坐标
    """
    h, w = mask.shape
    
    # 确保是二值
    if mask.max() > 1:
         _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = (mask * 255).astype(np.uint8)
    
    # 形态学平滑（闭运算填孔 + 开运算去噪）
    if morph_kernel and morph_kernel >= 3:
        if morph_kernel % 2 == 0:
            morph_kernel += 1
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        # 轮廓近似
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        points = approx.reshape(-1, 2).tolist()
        
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

def visualize_contours(image_img, polygons, output_vis_path):
    """
    在原图(numpy array)上可视化提取的轮廓
    """
    # 确保是BGR格式以便cv2保存
    if image_img.shape[2] == 3:
        # 假设输入是RGB (因为Predictor返回的是RGB)
        vis_img = cv2.cvtColor(image_img, cv2.COLOR_RGB2BGR)
    else:
        vis_img = image_img.copy()

    for poly in polygons:
        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)

    cv2.imencode('.jpg', vis_img)[1].tofile(str(output_vis_path))


def parse_sweep_values(raw_value, cast_func):
    """
    将逗号分隔参数解析为列表，例如 "0.12,0.15,0.18"
    """
    if raw_value is None:
        return []

    values = []
    for item in str(raw_value).split(','):
        item = item.strip()
        if not item:
            continue
        values.append(cast_func(item))
    return values


def build_run_name(config_index, seg_th, bound_th, boundary_erode_iter, morph_kernel, morph_iter, epsilon):
    return (
        f"run_{config_index:02d}"
        f"_seg{seg_th:.3f}"
        f"_bound{bound_th:.3f}"
        f"_erode{boundary_erode_iter}"
        f"_mk{morph_kernel}"
        f"_mi{morph_iter}"
        f"_eps{epsilon:.4f}"
    )


def run_single_config(files, predictor, output_dir, seg_th, bound_th, min_area,
                      epsilon, morph_kernel, morph_iter, boundary_erode_iter,
                      save_overlay, save_prob):
    mask_dir = output_dir / "masks"
    json_dir = output_dir / "json"
    vis_dir = output_dir / "visualization"

    mask_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for f in tqdm(files, desc=output_dir.name, leave=False):
        try:
            bin_mask, bin_boundary, orig_img, prob_mask, prob_boundary, dist_map = predictor.predict(
                f, seg_th, bound_th
            )

            final_mask = predictor.process_and_save(
                mask_dir, f.name, bin_mask, bin_boundary, orig_img,
                prob_mask, prob_boundary, dist_map,
                save_overlay=save_overlay,
                save_prob=save_prob,
                boundary_erode_iter=boundary_erode_iter
            )

            img_shape, polygons = extract_contours_from_mask_array(
                final_mask,
                min_area=min_area,
                epsilon_factor=epsilon,
                morph_kernel=morph_kernel,
                morph_iter=morph_iter
            )

            json_path = json_dir / f"{f.stem}.json"
            save_to_json(polygons, f.name, json_path, img_shape)

            vis_path = vis_dir / f"{f.stem}_vis.jpg"
            visualize_contours(orig_img, polygons, vis_path)

            success_count += 1

        except Exception as e:
            print(f"处理 {f.name} 失败: {e}")
            import traceback
            traceback.print_exc()

    return success_count


# ============================================================================
# 4. 主函数 (集成推理与矢量化)
# ============================================================================
def main():
    # ==================== 参数配置区域 ====================
    # 1. 模型路径
    default_model = r"D:\Work space\DeepLearning\farm\results_v7_ultimate\best_model.pth" 
    
    # 2. 输入图像路径
    default_input = r"D:\Work space\DeepLearning\farm\U-NET\3.png" 
    
    # 3. 输出基础目录
    default_output = r"./predictions_v7_vectorized"
    
    # 4. 阈值设置
    default_threshold = 0.5          # 统一阈值
    default_seg_threshold = None     # 分割阈值 (None则使用统一阈值)
    default_boundary_threshold = None # 边界阈值 (None则使用统一阈值 * 0.2)
    
    # 5. 矢量化设置
    default_min_area = 50            # 最小轮廓面积
    default_epsilon = 0.001          # 轮廓简化系数
    default_morph_kernel = 0         # 形态学平滑核大小 (0 表示关闭)
    default_morph_iter = 1           # 形态学迭代次数
    default_boundary_erode_iter = 0  # 扣边界前先腐蚀边界，减小向田内收缩
    
    # 6. 功能开关
    default_save_prob = False        # 保存概率图
    default_save_overlay = True      # 保存推理叠加图
    # ====================================================

    parser = argparse.ArgumentParser(description="U-Net 全流程：推理 -> 掩膜 -> 矢量化(JSON)")
    parser.add_argument('--model', type=str, default=default_model, help='模型权重路径 (.pth)')
    parser.add_argument('--input', type=str, default=default_input, help='输入图片或文件夹路径')
    parser.add_argument('--output', type=str, default=default_output, help='输出保存路径')
    
    # 推理参数
    parser.add_argument('--threshold', type=float, default=default_threshold, help='统一阈值')
    parser.add_argument('--seg_threshold', type=float, default=default_seg_threshold, help='分割阈值')
    parser.add_argument('--boundary_threshold', type=float, default=default_boundary_threshold, help='边界阈值')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--save_prob', action='store_true', default=default_save_prob, help='保存概率图')
    parser.add_argument('--boundary_erode_iter', type=int, default=default_boundary_erode_iter,
                        help='扣边界前先腐蚀边界的迭代次数(建议0~2)')
    
    # 矢量化参数
    parser.add_argument('--min_area', type=float, default=default_min_area, help='矢量化最小面积过滤')
    parser.add_argument('--epsilon', type=float, default=default_epsilon, help='轮廓简化系数(建议0.001~0.015)')
    parser.add_argument('--morph_kernel', type=int, default=default_morph_kernel, help='轮廓提取前形态学平滑核大小(0表示关闭)')
    parser.add_argument('--morph_iter', type=int, default=default_morph_iter, help='形态学平滑迭代次数')
    parser.add_argument('--sweep', action='store_true', help='自动批量导出多组参数结果')
    parser.add_argument('--sweep_boundary_thresholds', type=str,
                        default='0.10,0.12,0.15', help='批量边界阈值列表，逗号分隔')
    parser.add_argument('--sweep_boundary_erode_iters', type=str,
                        default='0,1', help='批量边界腐蚀次数列表，逗号分隔')
    parser.add_argument('--sweep_morph_kernels', type=str,
                        default='0,3', help='批量形态学核大小列表，逗号分隔')
    parser.add_argument('--sweep_epsilons', type=str,
                        default='0.004,0.006', help='批量轮廓简化系数列表，逗号分隔')
    parser.add_argument('--encoder_name', type=str, default='resnet50', help='骨干网络类型')
    
    args = parser.parse_args()

    # 参数校验
    if not args.model or not os.path.exists(args.model):
        print("错误: 请指定有效的模型路径 (--model)")
        # 为了方便用户直接修改代码运行，如果命令行没给，尝试检查default_model
        if default_model and os.path.exists(default_model):
            args.model = default_model
            print(f"使用代码内配置的模型: {args.model}")
        else:
            return

    if not args.input or not os.path.exists(args.input):
        print("错误: 请指定有效的输入路径 (--input)")
        if default_input and os.path.exists(default_input):
            args.input = default_input
            print(f"使用代码内配置的输入: {args.input}")
        else:
            return

    # 阈值处理
    threshold = args.threshold
    seg_th = args.seg_threshold if args.seg_threshold is not None else threshold
    bound_th = args.boundary_threshold if args.boundary_threshold is not None else threshold * 0.2

    # 初始化预测器
    predictor = Predictor(args.model, dropout_rate=args.dropout, encoder_name=args.encoder_name)

    # 准备文件列表
    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob('*.[jp][pn]g'))

    # 准备输出目录
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始处理 {len(files)} 张图片...")

    if args.sweep:
        sweep_boundary_thresholds = parse_sweep_values(args.sweep_boundary_thresholds, float)
        sweep_boundary_erode_iters = parse_sweep_values(args.sweep_boundary_erode_iters, int)
        sweep_morph_kernels = parse_sweep_values(args.sweep_morph_kernels, int)
        sweep_epsilons = parse_sweep_values(args.sweep_epsilons, float)

        configs = list(product(
            sweep_boundary_thresholds,
            sweep_boundary_erode_iters,
            sweep_morph_kernels,
            sweep_epsilons,
        ))

        print(f"批量模式: 共 {len(configs)} 组参数")
        print(f"输出基础目录: {out_dir}")

        total_success = 0
        for config_index, (cfg_bound_th, cfg_erode_iter, cfg_morph_kernel, cfg_epsilon) in enumerate(configs, start=1):
            run_name = build_run_name(
                config_index,
                seg_th,
                cfg_bound_th,
                cfg_erode_iter,
                cfg_morph_kernel,
                args.morph_iter,
                cfg_epsilon,
            )
            run_output_dir = out_dir / run_name

            print(f"\n[{config_index}/{len(configs)}] {run_name}")
            success_count = run_single_config(
                files=files,
                predictor=predictor,
                output_dir=run_output_dir,
                seg_th=seg_th,
                bound_th=cfg_bound_th,
                min_area=args.min_area,
                epsilon=cfg_epsilon,
                morph_kernel=cfg_morph_kernel,
                morph_iter=args.morph_iter,
                boundary_erode_iter=cfg_erode_iter,
                save_overlay=default_save_overlay,
                save_prob=args.save_prob,
            )
            print(f"完成: {success_count}/{len(files)} -> {run_output_dir}")
            total_success += success_count

        print(f"\n全部完成！总成功处理: {total_success}/{len(files) * len(configs)}")
        print(f"输出目录结构: {out_dir}")
        print("  └─ run_xx_*/     (每组参数一个子目录，内含 masks/json/visualization)")
    else:
        print(f"阈值配置: Seg={seg_th}, Boundary={bound_th}")
        print(f"边界处理: boundary_erode_iter={args.boundary_erode_iter}")

        success_count = run_single_config(
            files=files,
            predictor=predictor,
            output_dir=out_dir,
            seg_th=seg_th,
            bound_th=bound_th,
            min_area=args.min_area,
            epsilon=args.epsilon,
            morph_kernel=args.morph_kernel,
            morph_iter=args.morph_iter,
            boundary_erode_iter=args.boundary_erode_iter,
            save_overlay=default_save_overlay,
            save_prob=args.save_prob,
        )

        print(f"\n全部完成！成功处理: {success_count}/{len(files)}")
        print(f"输出目录结构: {out_dir}")
        print(f"  ├─ masks/         (推理生成的掩膜图片)")
        print(f"  ├─ json/          (矢量化JSON文件)")
        print(f"  └─ visualization/ (矢量化效果展示)")


if __name__ == "__main__":
    main()
