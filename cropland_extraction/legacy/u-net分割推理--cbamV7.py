# u-net分割推理--cbamV7.py
# 多任务推理脚本（分割 + 边界 + 距离）
# 基于u-net--CBAMV7.py训练代码和u-net分割推理--cbamV6模板

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
import warnings

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
# 2. 预测器类 (V7优化版)
# ============================================================================
class Predictor:
    def __init__(self, model_path, device='cuda', dropout_rate=0.2):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.model = MultiTaskUNet(
            encoder_name='resnet34',
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

    def save_result(self, save_dir, filename, binary_mask, boundary_mask, original_img,
                    prob_mask=None, prob_boundary=None, distance_map=None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(filename).stem

        print(f"  保存耕地掩膜: {binary_mask.shape}, 范围: [{binary_mask.min()}, {binary_mask.max()}]")
        seg_ratio = (binary_mask == 255).sum() / binary_mask.size
        print(f"    耕地占比: {seg_ratio:.2%}")
        cv2.imencode('.png', binary_mask)[1].tofile(str(save_dir / f"{stem}_seg_mask.png"))

        print(f"  保存边界掩膜: {boundary_mask.shape}, 范围: [{boundary_mask.min()}, {boundary_mask.max()}]")
        boundary_ratio = (boundary_mask > 0).sum() / boundary_mask.size
        print(f"    边界占比: {boundary_ratio:.2%}")
        cv2.imencode('.png', boundary_mask)[1].tofile(str(save_dir / f"{stem}_boundary.png"))
        final_mask = binary_mask.copy()

        # 只要是边界像素(>0)，就把对应的耕地像素置为背景(0)
        # 您可以根据边界的粗细调整这里，如果边界太细切不开，可以用 cv2.dilate 加粗 boundary_mask 后再减
        final_mask[boundary_mask > 0] = 0
        print(f"  保存最终掩膜: {final_mask.shape}, 范围: [{final_mask.min()}, {final_mask.max()}]")
        final_ratio = (final_mask > 0).sum() / final_mask.size
        print(f"    最终占比: {final_ratio:.2%}")
        cv2.imencode('.png', final_mask)[1].tofile(str(save_dir / f"{stem}_mask.png"))

        if distance_map is not None:
            distance_uint8 = (distance_map * 255).astype(np.uint8)
            print(f"  保存距离图: {distance_uint8.shape}, 范围: [{distance_uint8.min()}, {distance_uint8.max()}]")
            cv2.imencode('.png', distance_uint8)[1].tofile(str(save_dir / f"{stem}_distance.png"))

        overlay = original_img.copy()

        red_mask = np.zeros_like(original_img)
        red_mask[binary_mask == 255] = [255, 0, 0]

        green_mask = np.zeros_like(original_img)
        green_mask[boundary_mask > 0] = [0, 255, 0]

        mask_combined = cv2.addWeighted(red_mask, 1.0, green_mask, 1.0, 0)
        overlay = cv2.addWeighted(overlay, 0.7, mask_combined, 0.3, 0)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imencode('.jpg', overlay_bgr)[1].tofile(str(save_dir / f"{stem}_overlay.jpg"))

        if prob_mask is not None:
            prob_mask_uint8 = (prob_mask * 255).astype(np.uint8)
            cv2.imencode('.png', prob_mask_uint8)[1].tofile(str(save_dir / f"{stem}_prob_mask.png"))

        if prob_boundary is not None:
            prob_boundary_uint8 = (prob_boundary * 255).astype(np.uint8)
            cv2.imencode('.png', prob_boundary_uint8)[1].tofile(str(save_dir / f"{stem}_prob_boundary.png"))


# ============================================================================
# 3. 主函数
# ============================================================================
def main():
    # ============================================================================
    # 参数说明
    # ============================================================================
    # 基础参数
    # --model: 模型权重路径 (.pth) [必需]
    # --input: 输入图片或文件夹路径 [必需]
    # --output: 输出保存路径 [默认: ./predictions_v7]
    # --dropout: Dropout率 (必须与训练时一致) [默认: 0.2]
    # --save_prob: 是否保存概率图 [默认: False]
    #
    # 阈值参数 (优先级: 单独阈值 > 统一阈值)
    # ============================================================================
    # 方式1: 使用统一阈值 (推荐，最简单)
    # --threshold 0.5
    #   → 分割阈值 = 0.5, 边界阈值 = 0.1 (自动计算)
    #
    # 方式2: 使用统一阈值，单独调整边界
    # --threshold 0.5 --boundary_threshold 0.05
    #   → 分割阈值 = 0.5, 边界阈值 = 0.05
    #
    # 方式3: 分别指定分割和边界阈值 (最灵活)
    # --threshold 0.5 --seg_threshold 0.6 --boundary_threshold 0.1
    #   → 分割阈值 = 0.6, 边界阈值 = 0.1
    #
    # 阈值推荐
    # ============================================================================
    # 默认值 (推荐): --threshold 0.5
    #   分割阈值 = 0.5, 边界阈值 = 0.1 (自动计算)
    #
    # 边界太细: --threshold 0.5 --boundary_threshold 0.05
    #   降低边界阈值，让边界更明显
    #
    # 边界太粗: --threshold 0.5 --boundary_threshold 0.2
    #   提高边界阈值，让边界更粗
    #
    # 耕地太多: --threshold 0.6 --seg_threshold 0.6 --boundary_threshold 0.12
    #   提高分割阈值，减少耕地区域
    #
    # 耕地太少: --threshold 0.4 --seg_threshold 0.4 --boundary_threshold 0.08
    #   降低分割阈值，增加耕地区域
    # ============================================================================

    parser = argparse.ArgumentParser(description="Multi Task UNet V7 推理脚本")
    parser.add_argument('--model', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--input', type=str, required=True, help='输入图片或文件夹路径')
    parser.add_argument('--output', type=str, default='./predictions_v7', help='输出保存路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='统一阈值（分割和边界都使用此阈值）')
    parser.add_argument('--seg_threshold', type=float, default=None, help='分割二值化阈值（覆盖统一阈值）')
    parser.add_argument('--boundary_threshold', type=float, default=None, help='边界二值化阈值（覆盖统一阈值）')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率 (必须与训练时一致)')
    parser.add_argument('--save_prob', action='store_true', help='保存概率图')
    args = parser.parse_args()

    threshold = args.threshold
    seg_threshold = args.seg_threshold if args.seg_threshold is not None else threshold
    boundary_threshold = args.boundary_threshold if args.boundary_threshold is not None else threshold * 0.2

    predictor = Predictor(
        args.model,
        dropout_rate=args.dropout
    )

    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob('*.[jp][pn]g'))

    print(f"共找到 {len(files)} 张图片，开始推理...")
    print(f"\n========== 阈值配置 ==========")
    print(f"统一阈值: {threshold}")
    print(f"分割阈值: {seg_threshold}")
    print(f"边界阈值: {boundary_threshold}")
    print(f"===============================\n")

    for f in tqdm(files):
        try:
            mask, boundary, orig, prob_mask, prob_boundary, distance_map = predictor.predict(
                f,
                seg_threshold,
                boundary_threshold
            )
            predictor.save_result(
                args.output,
                f.name,
                mask,
                boundary,
                orig,
                prob_mask if args.save_prob else None,
                prob_boundary if args.save_prob else None,
                distance_map if args.save_prob else None
            )
        except Exception as e:
            print(f"处理 {f.name} 失败: {e}")

    print(f"\n推理完成！")
    print(f"结果保存在: {args.output}")
    print(f"\n输出文件说明:")
    print(f"  - *_seg_mask.png: 耕地二值掩膜 (阈值={seg_threshold})")
    print(f"  - *_boundary.png: 边界二值掩膜 (阈值={boundary_threshold})")
    print(f"  - *_mask.png: 真正的分割掩膜 (耕地掩膜 + 边界掩膜叠加)")
    print(f"  - *_overlay.jpg: 可视化叠加图 (红色=耕地，绿色=边界，绿色只在耕地区域内显示)")
    if args.save_prob:
        print(f"  - *_distance.png: 距离图 (像素到边界的欧氏距离)")
        print(f"  - *_prob_mask.png: 耕地概率图")
        print(f"  - *_prob_boundary.png: 边界概率图")


if __name__ == '__main__':
    main()
    # python u-net分割推理--cbamV7.py --model ./results_v7_ultimate\best_model.pth --input ./U-NET\img.png --output ./test --save_prob
