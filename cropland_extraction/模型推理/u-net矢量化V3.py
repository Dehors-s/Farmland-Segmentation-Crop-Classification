# u-net矢量化V3.py
# 针对 V9 多任务模型 (decoder0 + deep fusion) 优化
# 功能：多光谱推理 -> 滑窗拼接 -> 边界+距离引导地块分离 -> Shapefile 输出
# 改进：全分辨率解码级、逐通道百分位归一化、深层融合头

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

# scipy (延迟导入, 用于多边形平滑)
from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d


# ============================================================================
# 模型定义 (与 u-net--CBAMV9.py 一致)
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


# ============================================================================
# V9 CBAMUNet: 含 decoder0 全分辨率解码级
# ============================================================================
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

        # V9: 分离 stem (256x256) 和 encoder1 (pool → 128x128)
        self.stem = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)
        self.encoder1 = nn.Sequential(base_model.maxpool)
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

        # V9 decoder0: 256x256 → 512x512 + stem 跳连接
        self.decoder0_up = nn.ConvTranspose2d(
            decoder_channels[3], decoder_channels[3],
            kernel_size=4, stride=2, padding=1,
        )
        self.decoder0_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3] + encoder_channels[0], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.input_spectral_attention(x)
        x_stem = self.stem(x)       # 64ch, 256x256
        e1 = self.encoder1(x_stem)  # 64ch, 128x128
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        b = self.bridge(e5)
        d4 = self.decoder4(b, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)  # 32ch, 256x256

        d0_up = self.decoder0_up(d1)            # 32ch, 512x512
        stem_up = F.interpolate(x_stem, scale_factor=2, mode="bilinear", align_corners=True)  # 64ch, 512x512
        d0 = torch.cat([d0_up, stem_up], dim=1) # 96ch, 512x512
        d0 = self.decoder0_conv(d0)             # 32ch, 512x512
        return d0


# ============================================================================
# V9 MultiTaskUNet: 简化头 + 深层融合
# ============================================================================
class MultiTaskUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=4, classes=1,
                 use_cbam=True, use_spectral_attention=True, dropout_rate=0.2):
        super().__init__()
        self.base = CBAMUNet(encoder_name, encoder_weights, in_channels, classes,
                             use_cbam, use_spectral_attention, dropout_rate)

        # V9: 简化头，去掉 ConvTranspose2d（输入已为 512x512）
        self.boundary_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.distance_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.seg_up = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # V9: 深层融合头 (4层 conv vs V8 的单层)
        self.fusion = nn.Sequential(
            nn.Conv2d(16 + 1 + 1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, classes, kernel_size=1),
        )

    def forward(self, x):
        d0 = self.base(x)                       # 32ch, 512x512
        boundary_logit = self.boundary_head(d0) # 1ch, 512x512
        distance_logit = self.distance_head(d0) # 1ch, 512x512
        seg_feat = self.seg_up(d0)              # 16ch, 512x512
        boundary_pred = torch.sigmoid(boundary_logit)
        distance_pred = distance_logit
        cat_feat = torch.cat([seg_feat, boundary_pred, distance_pred], dim=1)
        seg_logit = self.fusion(cat_feat)       # 1ch, 512x512
        return seg_logit, boundary_logit, distance_logit


# ============================================================================
# 推断引擎
# ============================================================================
class InferenceEngine:
    def __init__(self, model_path, device="cuda", encoder_name="resnet50", in_channels=4,
                 dropout_rate=0.2, tile_size=512, tile_overlap=128,
                 norm_mode="percentile"):
        """
        Args:
            norm_mode: "percentile"（逐通道百分位归一化）或 "legacy"（全局 max）
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tile_size = tile_size
        self.stride = tile_size - tile_overlap
        self.in_channels = in_channels
        self.norm_mode = norm_mode
        print(f"设备: {self.device}")
        print(f"滑窗: tile={tile_size} stride={self.stride} (overlap={tile_overlap})")
        print(f"归一化: {norm_mode}")

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
            epoch = ckpt.get("epoch", "?")
            best_iou = ckpt.get("best_iou", None)
            if best_iou is not None:
                print(f"  训练轮数: {epoch}, 最佳IoU: {best_iou:.4f}")
            else:
                print(f"  训练轮数: {epoch}")
            best_th = ckpt.get("best_threshold", None)
            if best_th is not None:
                print(f"  最优阈值: {best_th:.2f} (IoU: {ckpt.get('best_threshold_iou', '?'):.4f})")
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _normalize_legacy(img):
        """全局 max 归一化（兼容 V8/V7 模型）"""
        img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        max_val = float(np.max(img)) if img.size else 1.0
        if max_val > 2000:
            img = img / 10000.0
        elif max_val > 1.5:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def _normalize_percentile(img, lower_pct=2, upper_pct=98):
        """逐通道百分位归一化（与 V9 训练一致）"""
        img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            lo, hi = np.percentile(channel, [lower_pct, upper_pct])
            if hi > lo:
                img[:, :, c] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
            else:
                mx = float(channel.max()) if channel.size else 1.0
                img[:, :, c] = np.clip(channel / max(mx, 1e-8), 0.0, 1.0)
        return img.astype(np.float32)

    def _read_image(self, path, in_channels=None):
        in_channels = in_channels or self.in_channels
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in [".tif", ".tiff"]:
            import rasterio
            with rasterio.open(path) as ds:
                img = ds.read().astype(np.float32)
                crs = ds.crs
                transform = ds.transform
            img = np.transpose(img, (1, 2, 0))
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

        # V3: 支持 percentiile 归一化
        if self.norm_mode == "percentile":
            img = self._normalize_percentile(img)
        else:
            img = self._normalize_legacy(img)
        return img, crs, transform

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
                        dist = dist_logit.squeeze().cpu().numpy()

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
    return refined[1]


def apply_simple_crf_fallback(prob_map, seg_result, boundary_result, iterations=3):
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
# TTA
# ============================================================================
def _tta_predict_single(model, tensor, device):
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
        lambda s, b, d: (np.rot90(s, -1, (-2, -1)), np.rot90(b, -1, (-2, -1)),
                         np.rot90(d, -1, (-2, -1)) if d is not None else None),
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
# 多边形后处理
# ============================================================================
def smooth_polygon(points, sigma=2.0):
    if len(points) < 5:
        return points
    pts = np.array(points, dtype=np.float32)
    closed = np.vstack([pts, pts[0]])
    smooth_x = _gaussian_filter1d(closed[:, 0], sigma, mode='wrap')
    smooth_y = _gaussian_filter1d(closed[:, 1], sigma, mode='wrap')
    return np.column_stack([smooth_x[:-1], smooth_y[:-1]]).tolist()


def regularize_polygons(polygons, epsilon_factor=0.002, smooth_sigma=1.5, min_area=50):
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
# 地块分离后处理
# ============================================================================
def separate_fields_watershed(binary_mask, boundary_mask, distance_map, min_distance=3):
    seg = binary_mask.astype(np.uint8)
    if seg.max() <= 1:
        seg = (seg * 255).astype(np.uint8)

    bdy = boundary_mask.astype(np.uint8)
    if bdy.max() <= 1:
        bdy[:, :] = (bdy * 255).astype(np.uint8)

    minus_bdy = seg.copy()
    minus_bdy[bdy > 127] = 0

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    minus_bdy = cv2.morphologyEx(minus_bdy, cv2.MORPH_OPEN, kernel_small, iterations=1)

    dist = cv2.distanceTransform(minus_bdy, cv2.DIST_L2, 5)
    dist_norm = dist / (dist.max() + 1e-8) if dist.max() > 0 else dist

    if distance_map is not None:
        dm = cv2.resize(distance_map, (seg.shape[1], seg.shape[0])) if distance_map.shape != seg.shape else distance_map
        dm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-8)
        dist_norm = 0.7 * dist_norm + 0.3 * dm

    # 用 OpenCV dilate 替代 scipy.ndimage.maximum_filter (快 10-100x)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_distance * 2 + 1, min_distance * 2 + 1))
    local_max = cv2.dilate(dist_norm, kernel)
    sure_fg = (dist_norm == local_max) & (dist_norm > 0.05) & (minus_bdy > 0)
    sure_fg = sure_fg.astype(np.uint8)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[minus_bdy == 0] = 0

    seg_3ch = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(seg_3ch, markers)

    instance_mask = np.zeros_like(seg, dtype=np.int32)
    for label in range(2, markers.max() + 1):
        instance_mask[markers == label] = label - 1
    return instance_mask


def separate_fields_simple(binary_mask, boundary_mask):
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
            pts = approx.reshape(-1, 2).tolist()
            if len(pts) >= 3:
                polygons.append(pts)
    return polygons


def contours_to_polygons(contours, min_area=50, epsilon_factor=0.001):
    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2).tolist()
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


# ============================================================================
# Shapefile 导出 (ArcGIS 兼容)
# ============================================================================
def save_polygons_to_shp(polygons, output_shp, geo_transform=None, src_crs=None, output_epsg=None):
    """保存为 Shapefile，将像素坐标转换为地理坐标。"""
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon as ShpPoly
        from rasterio.crs import CRS
    except ImportError:
        print("  [SHP] geopandas/shapely/rasterio 未安装，无法保存 shapefile")
        return

    # 像素坐标 → 地理坐标
    geom_list = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        try:
            pts = []
            for px, py in poly:
                if geo_transform is not None:
                    x, y = geo_transform * (px, py)
                else:
                    x, y = px, py
                pts.append((x, y))
            geom_list.append(ShpPoly(pts))
        except Exception as e:
            print(f"  [SHP] 多边形创建失败: {e}")

    if not geom_list:
        print("  [SHP] 没有有效多边形，不生成 shapefile")
        return

    # CRS 优先级: output_epsg > src_crs (from GeoTIFF)
    if output_epsg:
        crs = CRS.from_epsg(output_epsg)
    elif src_crs:
        crs = CRS.from_wkt(src_crs) if isinstance(src_crs, str) else src_crs
        if hasattr(crs, 'is_valid') and not crs.is_valid:
            crs = None
    else:
        crs = None

    gdf = gpd.GeoDataFrame(
        {"id": list(range(len(geom_list))), "label": ["farmland"] * len(geom_list)},
        geometry=geom_list, crs=crs.to_wkt() if crs else None,
    )
    gdf.to_file(output_shp, encoding="utf-8")
    print(f"  [SHP] 已保存 {len(geom_list)} 个地块 → {output_shp}")


# ============================================================================
# 主流程
# ============================================================================
def run_inference_pipeline(
    model_path, input_path, output_dir,
    seg_threshold=0.5, boundary_threshold=0.1,
    encoder_name="resnet50", in_channels=4,
    tile_size=512, tile_overlap=128,
    use_tta=False, use_crf=False, use_watershed=True,
    min_area=50, epsilon_factor=0.002,
    smooth_sigma=1.5, save_shp=False, save_vis=False,
    norm_mode="percentile", source_epsg=None,
):
    """完整的推理管线 (V2 经典流程 + V9 模型)。"""
    import geopandas as gpd
    from shapely.geometry import Polygon

    os.makedirs(output_dir, exist_ok=True)

    engine = InferenceEngine(
        model_path=model_path,
        encoder_name=encoder_name,
        in_channels=in_channels,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        norm_mode=norm_mode,
    )

    input_path = Path(input_path)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        image_paths = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in exts])

    print(f"\n共找到 {len(image_paths)} 张待推理图像")

    vis_output = os.path.join(output_dir, "visualization")
    shp_output = os.path.join(output_dir, "shapefile")
    os.makedirs(vis_output, exist_ok=True)
    os.makedirs(shp_output, exist_ok=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    all_count = 0

    for img_idx, img_path in enumerate(image_paths):
        stem = img_path.stem
        print(f"\n[{img_idx + 1}/{len(image_paths)}] {img_path.name}")

        try:
            bin_mask, bin_boundary, seg_map, bdy_map, dist_map, img_orig, crs, transform = \
                engine.predict_tiled(str(img_path), seg_threshold, boundary_threshold, use_tta)
        except Exception as e:
            print(f"  [ERROR] 推理失败: {e}")
            continue

        # ---------- V2 经典后处理流程 ----------
        seg_bin = bin_mask.astype(np.uint8)

        if bin_boundary is not None:
            bdy_bin = bin_boundary.astype(np.uint8)
            bdy_eroded = cv2.erode(bdy_bin, np.ones((3, 3), np.uint8), iterations=1)
            seg_minus_bdy = seg_bin.copy()
            seg_minus_bdy[bdy_eroded > 127] = 0
            seg_minus_bdy = cv2.morphologyEx(seg_minus_bdy, cv2.MORPH_CLOSE, kernel, iterations=1)
            seg_minus_bdy = cv2.morphologyEx(seg_minus_bdy, cv2.MORPH_OPEN, kernel, iterations=1)

            if use_watershed:
                dist = cv2.distanceTransform(seg_minus_bdy, cv2.DIST_L2, 5)
                if dist.max() > 0:
                    # 用 OpenCV dilate 替代 scipy.ndimage.maximum_filter (快 10-100x)
                    kernel_w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    local_max = cv2.dilate(dist, kernel_w)
                    sure_fg = ((dist == local_max) & (dist > 5.0)).astype(np.uint8)
                    _, markers = cv2.connectedComponents(sure_fg)
                    markers = markers + 1
                    markers[seg_minus_bdy == 0] = 0
                    markers = cv2.watershed(cv2.cvtColor(seg_bin, cv2.COLOR_GRAY2BGR), markers)
                    seg_to_contour = np.zeros_like(seg_bin)
                    for lbl in range(2, markers.max() + 1):
                        seg_to_contour[markers == lbl] = 255
                    if seg_to_contour.max() == 0:
                        _, labels = cv2.connectedComponents(seg_minus_bdy // 255)
                        seg_to_contour = np.zeros_like(seg_bin)
                        for lbl in range(1, labels.max() + 1):
                            seg_to_contour[labels == lbl] = 255
                else:
                    seg_to_contour = seg_minus_bdy
            else:
                num_labels, labels = cv2.connectedComponents(seg_minus_bdy // 255, connectivity=8)
                seg_to_contour = np.zeros_like(seg_bin)
                for lbl in range(1, labels.max() + 1):
                    seg_to_contour[labels == lbl] = 255
        else:
            seg_bin = cv2.morphologyEx(seg_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
            seg_bin = cv2.morphologyEx(seg_bin, cv2.MORPH_OPEN, kernel, iterations=1)
            seg_to_contour = seg_bin

        # 提取轮廓
        contours, _ = cv2.findContours(seg_to_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            pts = approx.reshape(-1, 2).tolist()
            if len(pts) >= 3:
                if smooth_sigma > 0:
                    pts = smooth_polygon(pts, sigma=smooth_sigma)
                if len(pts) >= 3:
                    polygons.append(pts)

        print(f"  [结果] {len(polygons)} 个地块")
        all_count += len(polygons)

        if save_shp and polygons:
            try:
                from rasterio.transform import Affine
                af = Affine(*transform) if transform is not None else None
                save_polygons_to_shp(
                    polygons, os.path.join(shp_output, f"{stem}.shp"),
                    geo_transform=af, src_crs=crs, output_epsg=source_epsg,
                )
            except Exception as e:
                print(f"  [SHP] 保存失败: {e}")

        if save_vis:
            vis = img_orig[:, :, :3].copy() if img_orig.shape[2] >= 3 else np.stack([img_orig[:, :, 0]] * 3, axis=2)
            if vis.max() <= 1.0:
                vis = (vis * 255).astype(np.uint8)
            overlay = vis.copy()
            for poly in polygons:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
            cv2.addWeighted(vis, 0.7, overlay, 0.3, 0, overlay)
            vis_path = os.path.join(vis_output, f"{stem}_vis.jpg")
            cv2.imencode('.jpg', overlay)[1].tofile(vis_path)
            print(f"  [vis] {vis_path}")

    print(f"\n{'=' * 60}")
    print(f"推理完成! 总计 {all_count} 个地块")
    if save_shp:
        print(f"  Shapefile: {shp_output}")
    if save_vis:
        print(f"  可视化: {vis_output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="U-Net V9 推理矢量化 (decoder0 + deep fusion)")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径 (.pth)")
    parser.add_argument("--input", type=str, required=True, help="输入图像或文件夹路径")
    parser.add_argument("--output", type=str, default="./predictions_v9", help="输出目录")
    parser.add_argument("--encoder_name", type=str, default="resnet50", help="骨干网络 (需与模型一致)")
    parser.add_argument("--in_channels", type=int, default=4, help="输入通道数")
    parser.add_argument("--seg_threshold", type=float, default=0.5, help="分割阈值")
    parser.add_argument("--boundary_threshold", type=float, default=0.15, help="边界阈值")
    parser.add_argument("--tile_size", type=int, default=512, help="滑窗大小")
    parser.add_argument("--tile_overlap", type=int, default=128, help="滑窗重叠")
    parser.add_argument("--min_area", type=int, default=50, help="最小地块面积 (像素)")
    parser.add_argument("--epsilon", type=float, default=0.002, help="多边形简化系数")
    parser.add_argument("--smooth_sigma", type=float, default=1.5, help="多边形平滑系数")
    parser.add_argument("--norm_mode", type=str, default="percentile", choices=["percentile", "legacy"],
                        help="归一化模式 (percentile=V9默认, legacy=V8兼容)")
    parser.add_argument("--tta", action="store_true", help="启用测试时增强")
    parser.add_argument("--crf", action="store_true", help="启用 DenseCRF 后处理")
    parser.add_argument("--no_watershed", action="store_true", help="禁用分水岭分离")
    parser.add_argument("--save_shp", action="store_true", help="保存 Shapefile (自动转换像素→地理坐标)")
    parser.add_argument("--save_vis", action="store_true", help="保存可视化结果")
    parser.add_argument("--source_epsg", type=int, default=None,
                        help="源坐标系 EPSG 编码 (默认从 GeoTIFF 读取)")

    args = parser.parse_args()

    flags = []
    if args.tta: flags.append("TTA")
    if args.crf: flags.append("CRF")
    if args.save_shp: flags.append("SHP")
    if flags:
        print(f"启用: {', '.join(flags)}")

    run_inference_pipeline(
        model_path=args.model,
        input_path=args.input,
        output_dir=args.output,
        seg_threshold=args.seg_threshold,
        boundary_threshold=args.boundary_threshold,
        encoder_name=args.encoder_name,
        in_channels=args.in_channels,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        use_tta=args.tta,
        use_crf=args.crf,
        use_watershed=not args.no_watershed,
        min_area=args.min_area,
        epsilon_factor=args.epsilon,
        smooth_sigma=args.smooth_sigma,
        save_shp=args.save_shp,
        save_vis=args.save_vis,
        norm_mode=args.norm_mode,
        source_epsg=args.source_epsg,
    )
