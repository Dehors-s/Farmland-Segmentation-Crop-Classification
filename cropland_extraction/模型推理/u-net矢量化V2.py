# u-net矢量化V2.py
# 针对 V8 多任务模型 (seg + boundary + distance) 优化
# 功能：多光谱推理 -> 滑窗拼接 -> 边界+距离引导地块分离 -> Shapefile 输出

import os
import warnings
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================================
# 模型定义 (与 u-net--CBAMV8.py 完全一致)
# ============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        hidden = max(8, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class SpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        hidden = max(4, in_channels // reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.proj(self.pool(x))
        return x * w


class CBAMDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_cbam=True, dropout_rate=0.2):
        super().__init__()
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
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        if self.use_cbam:
            x = self.cbam(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        return x


class CBAMUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=4, classes=1,
                 use_cbam=True, use_spectral_attention=True, dropout_rate=0.2):
        super().__init__()
        import torchvision.models as models

        if encoder_name == "resnet34":
            base_model = models.resnet34(weights="IMAGENET1K_V1" if encoder_weights else None)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == "resnet50":
            base_model = models.resnet50(weights="IMAGENET1K_V1" if encoder_weights else None)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            base_model = models.resnet18(weights="IMAGENET1K_V1" if encoder_weights else None)
            encoder_channels = [64, 64, 128, 256, 512]

        if in_channels != 3:
            old_conv1 = base_model.conv1
            new_conv1 = nn.Conv2d(
                in_channels, old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                padding=old_conv1.padding, bias=False,
            )
            nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
            with torch.no_grad():
                if in_channels > 3:
                    new_conv1.weight[:, :3] = old_conv1.weight
                    mean_weight = old_conv1.weight.mean(dim=1, keepdim=True)
                    for idx in range(3, in_channels):
                        new_conv1.weight[:, idx:idx + 1] = mean_weight
                else:
                    new_conv1.weight[:] = old_conv1.weight[:, :in_channels]
            base_model.conv1 = new_conv1

        self.input_spectral_attention = SpectralAttention(in_channels) if use_spectral_attention else nn.Identity()
        self.encoder1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.encoder2 = base_model.layer1
        self.encoder3 = base_model.layer2
        self.encoder4 = base_model.layer3
        self.encoder5 = base_model.layer4

        decoder_channels = [256, 128, 64, 32]
        self.bridge = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = CBAMDecoderBlock(decoder_channels[0], encoder_channels[-2], decoder_channels[1], use_cbam, dropout_rate)
        self.decoder3 = CBAMDecoderBlock(decoder_channels[1], encoder_channels[-3], decoder_channels[2], use_cbam, dropout_rate)
        self.decoder2 = CBAMDecoderBlock(decoder_channels[2], encoder_channels[-4], decoder_channels[3], use_cbam, dropout_rate)
        self.decoder1 = CBAMDecoderBlock(decoder_channels[3], encoder_channels[-5], decoder_channels[3], use_cbam, dropout_rate)

    def forward(self, x):
        x = self.input_spectral_attention(x)
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
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=4, classes=1,
                 use_cbam=True, use_spectral_attention=True, dropout_rate=0.2):
        super().__init__()
        self.base = CBAMUNet(encoder_name, encoder_weights, in_channels, classes,
                             use_cbam, use_spectral_attention, dropout_rate)

        self.boundary_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.distance_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.seg_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(16 + 1 + 1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(16, classes, kernel_size=1),
        )

    def forward(self, x):
        d1 = self.base(x)
        boundary_logit = self.boundary_head(d1)
        distance_logit = self.distance_head(d1)
        seg_feat = self.seg_up(d1)
        if boundary_logit.shape[2:] != x.shape[2:]:
            boundary_logit = F.interpolate(boundary_logit, size=x.shape[2:], mode="bilinear", align_corners=True)
            distance_logit = F.interpolate(distance_logit, size=x.shape[2:], mode="bilinear", align_corners=True)
            seg_feat = F.interpolate(seg_feat, size=x.shape[2:], mode="bilinear", align_corners=True)
        boundary_pred = torch.sigmoid(boundary_logit)
        distance_pred = distance_logit
        cat_feat = torch.cat([seg_feat, boundary_pred, distance_pred], dim=1)
        seg_logit = self.fusion(cat_feat)
        return seg_logit, boundary_logit, distance_logit


# ============================================================================
# 推断引擎
# ============================================================================
class InferenceEngine:
    def __init__(self, model_path, device="cuda", encoder_name="resnet50", in_channels=4,
                 dropout_rate=0.2, tile_size=512, tile_overlap=128):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tile_size = tile_size
        self.stride = tile_size - tile_overlap
        self.in_channels = in_channels
        print(f"设备: {self.device}")
        print(f"滑窗: tile={tile_size} stride={self.stride} (overlap={tile_overlap})")

        self.model = MultiTaskUNet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            use_cbam=True,
            use_spectral_attention=True,
            dropout_rate=dropout_rate,
        ).to(self.device)
        self._load_weights(model_path)

    def _load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件未找到: {path}")
        print(f"加载模型: {path}")
        ckpt = torch.load(path, map_location=self.device)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            print(f"  训练轮数: {ckpt.get('epoch', '?')}, 最佳IoU: {ckpt.get('best_iou', '?'):.4f}" if isinstance(ckpt.get('best_iou'), float) else f"  训练轮数: {ckpt.get('epoch', '?')}")
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _read_image(self, path, in_channels=None):
        in_channels = in_channels or self.in_channels
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in [".tif", ".tiff"]:
            import rasterio
            with rasterio.open(path) as ds:
                img = ds.read().astype(np.float32)  # C,H,W
                crs = ds.crs
                transform = ds.transform
            img = np.transpose(img, (1, 2, 0))  # H,W,C
        else:
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            crs = None
            transform = None

        if img.shape[2] < in_channels:
            pad_ch = in_channels - img.shape[2]
            img = np.concatenate([img, np.zeros((img.shape[0], img.shape[1], pad_ch), dtype=img.dtype)], axis=2)
        elif img.shape[2] > in_channels:
            img = img[:, :, :in_channels]
        return img, crs, transform

    @staticmethod
    def _normalize_multispectral(img):
        img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        max_val = float(np.max(img)) if img.size else 1.0
        if max_val > 2000:
            img = img / 10000.0
        elif max_val > 1.5:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        return img

    def _generate_positions(self, start, stop):
        if stop <= start:
            return []
        if stop - start <= self.tile_size:
            return [start]
        positions = list(range(start, stop - self.tile_size + 1, self.stride))
        if not positions:
            positions = [start]
        edge_start = stop - self.tile_size
        if positions[-1] != edge_start:
            positions.append(edge_start)
        return sorted(set(positions))

    def predict_tiled(self, image_path, seg_threshold=0.5, boundary_threshold=0.1, use_tta=False):
        """滑窗推理大图，Gaussian 权重融合重叠区域。"""
        img_full, crs, src_transform = self._read_image(image_path)
        img_full = self._normalize_multispectral(img_full)
        h_full, w_full = img_full.shape[:2]

        if h_full <= self.tile_size and w_full <= self.tile_size:
            return self._predict_single(img_full, seg_threshold, boundary_threshold, crs, src_transform, use_tta)

        row_positions = self._generate_positions(0, h_full)
        col_positions = self._generate_positions(0, w_full)

        seg_accum = np.zeros((h_full, w_full), dtype=np.float32)
        bdy_accum = np.zeros((h_full, w_full), dtype=np.float32)
        dist_accum = np.zeros((h_full, w_full), dtype=np.float32)
        weight_accum = np.zeros((h_full, w_full), dtype=np.float32)

        gauss_x = cv2.getGaussianKernel(self.tile_size, self.tile_size / 4.0)
        gauss_y = cv2.getGaussianKernel(self.tile_size, self.tile_size / 4.0)
        gauss_2d = gauss_x @ gauss_y.T

        self.model.eval()
        for row in tqdm(row_positions, desc="滑窗推理 (行)", leave=False):
            for col in col_positions:
                r_end = min(row + self.tile_size, h_full)
                c_end = min(col + self.tile_size, w_full)
                patch = np.zeros((self.tile_size, self.tile_size, img_full.shape[2]), dtype=np.float32)
                patch_h = r_end - row
                patch_w = c_end - col
                patch[:patch_h, :patch_w] = img_full[row:r_end, col:c_end]

                tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
                if use_tta:
                    seg, bdy, dist = _tta_predict_single(self.model, tensor, self.device)
                else:
                    with torch.no_grad():
                        seg_logit, bdy_logit, dist_logit = self.model(tensor)
                        seg = torch.sigmoid(seg_logit).squeeze().cpu().numpy()
                        bdy = torch.sigmoid(bdy_logit).squeeze().cpu().numpy()
                        dist = (dist_logit if isinstance(dist_logit, np.ndarray) else dist_logit.squeeze().cpu().numpy())

                w = gauss_2d[:patch_h, :patch_w]
                seg_accum[row:r_end, col:c_end] += seg[:patch_h, :patch_w] * w
                bdy_accum[row:r_end, col:c_end] += bdy[:patch_h, :patch_w] * w
                dist_accum[row:r_end, col:c_end] += dist[:patch_h, :patch_w] * w
                weight_accum[row:r_end, col:c_end] += w

        eps = 1e-8
        seg_map = seg_accum / (weight_accum + eps)
        bdy_map = bdy_accum / (weight_accum + eps)
        dist_map = dist_accum / (weight_accum + eps)

        bin_mask = (seg_map > seg_threshold).astype(np.uint8) * 255
        bin_boundary = (bdy_map > boundary_threshold).astype(np.uint8) * 255
        return bin_mask, bin_boundary, seg_map, bdy_map, dist_map, img_full, crs, src_transform

    def _predict_single(self, img, seg_threshold, boundary_threshold, crs, src_transform, use_tta=False):
        """小图直接推理，不做滑窗。"""
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if use_tta:
            seg_map, bdy_map, dist_map = _tta_predict_single(self.model, tensor, self.device)
        else:
            with torch.no_grad():
                seg_logit, bdy_logit, dist_logit = self.model(tensor)
                seg_map = torch.sigmoid(seg_logit).squeeze().cpu().numpy()
                bdy_map = torch.sigmoid(bdy_logit).squeeze().cpu().numpy()
                dist_map = dist_logit.squeeze().cpu().numpy()

        bin_mask = (seg_map > seg_threshold).astype(np.uint8) * 255
        bin_boundary = (bdy_map > boundary_threshold).astype(np.uint8) * 255
        return bin_mask, bin_boundary, seg_map, bdy_map, dist_map, img, crs, src_transform


# ============================================================================
# DenseCRF 后处理 (边界平滑)
# ============================================================================
def apply_dense_crf(image_rgb, prob_map, crf_iterations=5, sxy_gaussian=(3, 3), compat_gaussian=3,
                     sxy_bilateral=(80, 80), srgb_bilateral=(13, 13, 13), compat_bilateral=10):
    """
    对概率图施加全连接 CRF 后处理，平滑边界同时保持边缘。
    参考: Krahenbuhl & Koltun, NIPS 2011
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("  [CRF] pydensecrf 未安装，跳过 CRF (pip install pydensecrf)")
        return prob_map

    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb[:, :, :3] * 255).astype(np.uint8)
    elif image_rgb.shape[2] > 3:
        image_rgb = image_rgb[:, :, :3].astype(np.uint8)
    image_rgb = np.ascontiguousarray(image_rgb)

    if prob_map.ndim == 2:
        prob_bg = 1.0 - prob_map
        prob_map = np.stack([prob_bg, prob_map], axis=-1)

    prob_map = prob_map.astype(np.float32)
    prob_map = np.clip(prob_map, 1e-8, 1 - 1e-8)
    prob_map = np.ascontiguousarray(prob_map)
    unary = unary_from_softmax(prob_map.transpose(2, 0, 1))
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(image_rgb.shape[1], image_rgb.shape[0], 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=image_rgb, compat=compat_bilateral)

    Q = d.inference(crf_iterations)
    refined = np.array(Q).reshape((2, image_rgb.shape[0], image_rgb.shape[1]))
    refined = refined[1]  # foreground probability
    return refined


def apply_simple_crf_fallback(prob_map, seg_result, boundary_result, iterations=3):
    """
    pydensecrf 不可用时的简单备选: 用预测的 boundary map 做引导滤波。
    """
    if boundary_result is None:
        return prob_map
    bdy = boundary_result.astype(np.float32)
    if bdy.max() <= 1.0:
        bdy = bdy * 255.0

    refined = prob_map.copy()
    for _ in range(iterations):
        refined = cv2.bilateralFilter(refined.astype(np.float32), 9, 75, 75)
        refined = cv2.GaussianBlur(refined, (3, 3), 0)
        refined = cv2.addWeighted(refined, 0.7, prob_map, 0.3, 0)
        refined[bdy > 128] = refined[bdy > 128] * 0.5
    return np.clip(refined, 0, 1)


# ============================================================================
# TTA (Test-Time Augmentation) — 增强边界稳定性
# ============================================================================
def _tta_predict_single(model, tensor, device):
    """对单张图做 TTA (flip + rot90) 预测并平均概率。"""
    tta_transforms = [
        lambda t: t,
        lambda t: torch.flip(t, [-1]),
        lambda t: torch.flip(t, [-2]),
        lambda t: torch.rot90(t, 1, [-2, -1]),
    ]
    tta_inverses = [
        lambda s, b, d: (s, b, d),
        lambda s, b, d: (np.flip(s, -1), np.flip(b, -1), np.flip(d, -1) if d is not None else None),
        lambda s, b, d: (np.flip(s, -2), np.flip(b, -2), np.flip(d, -2) if d is not None else None),
        lambda s, b, d: (np.rot90(s, -1, (-2, -1)), np.rot90(b, -1, (-2, -1)), np.rot90(d, -1, (-2, -1)) if d is not None else None),
    ]

    seg_sum, bdy_sum, dist_sum = None, None, None
    for t_fn, inv_fn in zip(tta_transforms, tta_inverses):
        t_tensor = t_fn(tensor)
        with torch.no_grad():
            seg_logit, bdy_logit, dist_logit = model(t_tensor)
            seg = torch.sigmoid(seg_logit).squeeze().cpu().numpy()
            bdy = torch.sigmoid(bdy_logit).squeeze().cpu().numpy()
            dist = dist_logit.squeeze().cpu().numpy()
        seg, bdy, dist = inv_fn(seg, bdy, dist)
        if seg_sum is None:
            seg_sum = np.zeros_like(seg)
            bdy_sum = np.zeros_like(bdy)
            dist_sum = np.zeros_like(dist)
        seg_sum += seg
        bdy_sum += bdy
        dist_sum += dist

    return seg_sum / 4.0, bdy_sum / 4.0, dist_sum / 4.0


# ============================================================================
# 多边形后处理 — 顶点简化 + B-spline 平滑
# ============================================================================
def smooth_polygon(points, sigma=2.0):
    """对多边形顶点做高斯平滑，使边界更流畅。"""
    if len(points) < 5:
        return points
    pts = np.array(points, dtype=np.float32)
    closed = np.vstack([pts, pts[0]])
    from scipy.ndimage import gaussian_filter1d
    smooth_x = gaussian_filter1d(closed[:, 0], sigma, mode='wrap')
    smooth_y = gaussian_filter1d(closed[:, 1], sigma, mode='wrap')
    return np.column_stack([smooth_x[:-1], smooth_y[:-1]]).tolist()


def regularize_polygons(polygons, epsilon_factor=0.002, smooth_sigma=1.5, min_area=50):
    """
    对提取的多边形做两步后处理：
      1. Douglas-Peucker 简化到更少的顶点
      2. 高斯平滑顶点坐标
    """
    result = []
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        peri = cv2.arcLength(pts.reshape((-1, 1, 2)), True)
        epsilon = epsilon_factor * peri
        approx = cv2.approxPolyDP(pts.reshape((-1, 1, 2)), epsilon, True)
        approx = approx.reshape(-1, 2).tolist()
        if len(approx) < 3:
            continue
        if smooth_sigma > 0:
            approx = smooth_polygon(approx, sigma=smooth_sigma)
        if len(approx) >= 3:
            result.append(approx)
    return result


# ============================================================================
# 地块分离后处理 (核心优化)
# ============================================================================
def separate_fields_watershed(binary_mask, boundary_mask, distance_map, min_distance=3):
    """
    使用预测的 distance map + boundary mask 进行分水岭分离。
    策略：
      1. 从 seg mask 中减去 boundary，得到分开的田块核心
      2. 在核心区域上做距离变换，找局部极大值作为种子
      3. 分水岭膨胀回原始 seg mask 边界
    """
    seg = binary_mask.astype(np.uint8)
    if seg.max() <= 1:
        seg = (seg * 255).astype(np.uint8)

    bdy = boundary_mask.astype(np.uint8)
    if bdy.max() <= 1:
        bdy[:, :] = (bdy * 255).astype(np.uint8)

    # 从分割结果中减去边界线
    minus_bdy = seg.copy()
    minus_bdy[bdy > 127] = 0

    # 开运算清理碎片
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    minus_bdy = cv2.morphologyEx(minus_bdy, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # 距离变换找种子
    dist = cv2.distanceTransform(minus_bdy, cv2.DIST_L2, 5)
    dist_norm = dist / (dist.max() + 1e-8) if dist.max() > 0 else dist

    # 结合预测的 distance_map (从模型输出) 增强种子检测
    if distance_map is not None:
        dm = cv2.resize(distance_map, (seg.shape[1], seg.shape[0])) if distance_map.shape != seg.shape else distance_map
        dm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-8)
        dist_norm = 0.7 * dist_norm + 0.3 * dm

    # 找局部极大值作为种子 (距离 >= min_distance 且为局部最大)
    from scipy import ndimage
    local_max = ndimage.maximum_filter(dist_norm, size=min_distance * 2 + 1)
    sure_fg = (dist_norm == local_max) & (dist_norm > 0.05) & (minus_bdy > 0)
    sure_fg = sure_fg.astype(np.uint8)

    # 标记连通域
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[minus_bdy == 0] = 0

    # 分水岭
    seg_3ch = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(seg_3ch, markers)

    # 提取实例 mask
    instance_mask = np.zeros_like(seg, dtype=np.int32)
    for label in range(2, markers.max() + 1):
        instance_mask[markers == label] = label - 1

    return instance_mask


def separate_fields_simple(binary_mask, boundary_mask):
    """仅用边界减法 + 连通域分析分离地块。"""
    seg = binary_mask.astype(np.uint8)
    if seg.max() <= 1:
        seg = (seg * 255).astype(np.uint8)
    bdy = boundary_mask.astype(np.uint8)
    if bdy.max() <= 1:
        bdy = (bdy * 255).astype(np.uint8)

    minus_bdy = seg.copy()
    minus_bdy[bdy > 127] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    minus_bdy = cv2.morphologyEx(minus_bdy, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(minus_bdy // 255)
    return labels


# ============================================================================
# 矢量化
# ============================================================================
def instance_to_contours(instance_mask, min_area=50, epsilon_factor=0.001):
    """从实例 mask 提取每块田的轮廓。"""
    h, w = instance_mask.shape
    polygons = []
    for label in range(1, instance_mask.max() + 1):
        single = (instance_mask == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            points = approx.reshape(-1, 2).tolist()
            if len(points) > 2:
                polygons.append(points)
    return (h, w), polygons





def save_shapefile(polygons, image_shape, crs_wkt, geo_transform, output_path, source_epsg=None):
    import os
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon as ShpPoly
        from rasterio.crs import CRS
    except ImportError:
        print("  WARNING: geopandas/shapely not found. pip install geopandas")
        return

    # Apply geo transform to pixel coords
    shp_polys = []
    for poly in polygons:
        pts = []
        for px, py in poly:
            if geo_transform is not None:
                x, y = geo_transform * (px, py)
            else:
                x, y = px, py
            pts.append((x, y))
        if len(pts) > 2:
            shp_polys.append(ShpPoly(pts))

    if source_epsg:
        crs = CRS.from_epsg(source_epsg)
    elif crs_wkt:
        crs = CRS.from_wkt(crs_wkt)
        if crs and not crs.is_valid:
            crs = None
    else:
        crs = None

    gdf = gpd.GeoDataFrame({'id': list(range(len(shp_polys))), 'label': ['farmland']*len(shp_polys)},
                           geometry=shp_polys, crs=crs.to_wkt() if crs else None)
    gdf.to_file(str(output_path), encoding='utf-8')
    print(f"  Shapefile saved: {output_path} ({len(gdf)} features)")


def save_visualization(image, polygons, output_path):
    """在原图上绘制多边形边界。"""
    vis = image.copy()
    if vis.max() <= 1.0:
        vis = (vis * 255).astype(np.uint8)
    if vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    else:
        vis = cv2.cvtColor(vis[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)

    for poly in polygons:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", vis)[1].tofile(str(output_path))


# ============================================================================
# 主入口
# ============================================================================
def main():
    import argparse, json

    parser = argparse.ArgumentParser(description="U-Net V8 矢量化 V2 — boundary 引导地块分离")
    parser.add_argument("--model", type=str, required=True, help="V8 .pth 模型")
    parser.add_argument("--input", type=str, required=True, help="输入图像或目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--encoder_name", type=str, default="resnet50")
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=128)
    parser.add_argument("--seg_threshold", type=float, default=0.5)
    parser.add_argument("--boundary_threshold", type=float, default=0.15)
    parser.add_argument("--min_area", type=float, default=50)
    parser.add_argument("--poly_epsilon", type=float, default=0.001)
    parser.add_argument("--poly_smooth_sigma", type=float, default=0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--use_crf", action="store_true", help="启用 DenseCRF 边缘精修")
    parser.add_argument("--use_watershed", action="store_true", help="启用分水岭分离 (更精细)")
    parser.add_argument("--no_split", action="store_true", help="不分割相邻地块 (合并输出)")
    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--source_epsg", type=int, default=None, help="源坐标系的 EPSG 代号，用于将投影坐标转为 WGS84。适用于 CRS 元数据损坏的 GeoTIFF。")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    engine = InferenceEngine(
        model_path=args.model,
        device=args.device,
        encoder_name=args.encoder_name,
        in_channels=args.in_channels,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )

    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = (sorted(input_path.glob("*.[jp][pn]g")) + sorted(input_path.glob("*.tif")) + sorted(input_path.glob("*.tiff")))
    else:
        raise FileNotFoundError(f"输入路径不存在: {args.input}")

    out_dir = Path(args.output)
    shp_dir = out_dir / "shapefile"
    vis_dir = out_dir / "visualization"
    for d in [shp_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    flags = []
    if args.tta: flags.append("TTA")
    if args.use_crf: flags.append("CRF")
    if args.use_watershed: flags.append("Watershed")
    if args.no_split: flags.append("NoSplit")
    flag_str = " | ".join(flags) if flags else "默认(boundary分割)"
    print(f"\n处理 {len(files)} 张图片 | {flag_str}")
    total_polygons = 0

    for f in tqdm(files, desc="推理+矢量化"):
        try:
            _, _, seg_prob, bdy_prob, dist_map, img_full, crs, gt = engine.predict_tiled(
                str(f), args.seg_threshold, args.boundary_threshold, use_tta=args.tta
            )
        except Exception as e:
            print(f"  [跳过] {f.name}: {e}")
            continue

        stem = f.stem

        # --- CRF ---
        if args.use_crf:
            seg_prob = apply_dense_crf(img_full, seg_prob)

        # --- 二值化 ---
        seg_bin = (seg_prob > args.seg_threshold).astype(np.uint8) * 255
        bdy_bin = (bdy_prob > args.boundary_threshold).astype(np.uint8) * 255

        # --- 形态学清理 ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        if not args.no_split:
            # 腐蚀边界线（防止吃掉太多前景）
            bdy_eroded = cv2.erode(bdy_bin, np.ones((3, 3), np.uint8), iterations=1)

            seg_minus_bdy = seg_bin.copy()
            seg_minus_bdy[bdy_eroded > 127] = 0
            seg_minus_bdy = cv2.morphologyEx(seg_minus_bdy, cv2.MORPH_CLOSE, kernel, iterations=1)
            seg_minus_bdy = cv2.morphologyEx(seg_minus_bdy, cv2.MORPH_OPEN, kernel, iterations=1)

            if args.use_watershed:
                from scipy import ndimage
                dist = cv2.distanceTransform(seg_minus_bdy, cv2.DIST_L2, 5)
                if dist.max() > 0:
                    local_max = ndimage.maximum_filter(dist, size=7)
                    sure_fg = ((dist == local_max) & (dist > 5.0)).astype(np.uint8)
                    _, markers = cv2.connectedComponents(sure_fg)
                    markers = markers + 1
                    markers[seg_minus_bdy == 0] = 0
                    markers = cv2.watershed(cv2.cvtColor(seg_bin, cv2.COLOR_GRAY2BGR), markers)
                    seg_to_contour = np.zeros_like(seg_bin)
                    for lbl in range(2, markers.max() + 1):
                        seg_to_contour[markers == lbl] = 255

                    # Fallback: 如果分水岭没有产生任何实例，退回连通域
                    if seg_to_contour.max() == 0:
                        _, labels = cv2.connectedComponents(seg_minus_bdy // 255)
                        seg_to_contour = np.zeros_like(seg_bin)
                        for lbl in range(1, labels.max() + 1):
                            seg_to_contour[labels == lbl] = 255
                else:
                    seg_to_contour = seg_minus_bdy
            else:
                # 连通域分离
                num_labels, labels = cv2.connectedComponents(seg_minus_bdy // 255, connectivity=8)
                seg_to_contour = np.zeros_like(seg_bin)
                for lbl in range(1, labels.max() + 1):
                    seg_to_contour[labels == lbl] = 255
        else:
            # 不分割：直接用 seg mask
            seg_bin = cv2.morphologyEx(seg_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
            seg_bin = cv2.morphologyEx(seg_bin, cv2.MORPH_OPEN, kernel, iterations=1)
            seg_to_contour = seg_bin

        # --- 提取轮廓 ---
        contours, _ = cv2.findContours(seg_to_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < args.min_area:
                continue
            epsilon = args.poly_epsilon * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            pts = approx.reshape(-1, 2).tolist()
            if len(pts) > 2:
                polygons.append(pts)

        # --- 多边形平滑 ---
        if args.poly_smooth_sigma > 0:
            polygons = regularize_polygons(polygons, args.poly_epsilon, args.poly_smooth_sigma, args.min_area)

        # --- 保存 Shapefile ---
        from rasterio.transform import Affine
        gt_tuple = tuple(gt) if gt is not None else None
        crs_str = crs.to_wkt() if crs is not None else None
        af = Affine(*gt_tuple) if gt is not None else None
        save_shapefile(polygons, img_full.shape[:2], crs_str, af, shp_dir / f"{stem}.shp", args.source_epsg)

        # --- 可视化 ---
        save_visualization(img_full, polygons, vis_dir / f"{stem}_vis.jpg")

        if args.save_intermediate:
            mask_dir = out_dir / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".png", seg_bin)[1].tofile(str(mask_dir / f"{stem}_seg.png"))
            cv2.imencode(".png", bdy_bin)[1].tofile(str(mask_dir / f"{stem}_bdy.png"))

        total_polygons += len(polygons)

    print(f"\n完成! 总计 {len(files)} 张图, {total_polygons} 个地块")
    print(f"Shapefile: {shp_dir}")
    print(f"可视化: {vis_dir}")


if __name__ == "__main__":
    main()
