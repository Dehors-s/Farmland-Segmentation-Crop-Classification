# u-net矢量化-baseline.py
# 基线模型推理 + 矢量化 (DeepLab v3+ / D-LinkNet / HRNet-W48)
# 功能: 单图/批量推理 → 掩膜导出 → 轮廓矢量化(SHP/JSON)

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
# 模型定义 (与训练脚本一致)
# ============================================================================

# --- DeepLab v3+ ---
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations=(6, 12, 18)):
        super().__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3x3_d6 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=dilations[0], dilation=dilations[0], bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3x3_d12 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3x3_d18 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(0.1))

    def forward(self, x):
        size = x.shape[2:]
        return self.fuse(torch.cat([
            self.conv1x1(x), self.conv3x3_d6(x), self.conv3x3_d12(x), self.conv3x3_d18(x),
            F.interpolate(self.image_pool(x), size=size, mode="bilinear", align_corners=False)
        ], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, encoder_name="resnet50"):
        super().__init__()
        import torchvision.models as models
        base = models.resnet50(weights=None)
        enc_ch = [64, 256, 512, 1024, 2048]
        old_c1 = base.conv1
        new_c1 = nn.Conv2d(in_channels, old_c1.out_channels, kernel_size=old_c1.kernel_size, stride=old_c1.stride, padding=old_c1.padding, bias=old_c1.bias is not None)
        self.conv1 = nn.Sequential(new_c1, base.bn1, base.relu)
        self.maxpool = base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.aspp = ASPP(enc_ch[-1], out_channels=256)
        self.low_conv = nn.Sequential(nn.Conv2d(enc_ch[1], 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.seg_head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        ins = x.shape[2:]
        x = self.conv1(x)
        x = self.maxpool(x)       # 1/4
        f1 = self.layer1(x)       # 1/4
        f2 = self.layer2(f1)      # 1/8
        f3 = self.layer3(f2)      # 1/16
        f4 = self.layer4(f3)      # 1/32
        aspp_out = self.aspp(f4)
        aspp_up = F.interpolate(aspp_out, size=f1.shape[2:], mode="bilinear", align_corners=False)
        dec = self.decoder(torch.cat([aspp_up, self.low_conv(f1)], dim=1))
        out = F.interpolate(dec, size=ins, mode="bilinear", align_corners=False)
        out = self.seg_head(out)
        return out


# --- D-LinkNet ---
class DLinkNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, encoder_name="resnet34"):
        super().__init__()
        import torchvision.models as models
        if encoder_name == "resnet34":
            base = models.resnet34(weights=None); enc_ch = [64, 64, 128, 256, 512]
        else:
            base = models.resnet50(weights=None); enc_ch = [64, 256, 512, 1024, 2048]
        old_c1 = base.conv1
        new_c1 = nn.Conv2d(in_channels, old_c1.out_channels, kernel_size=old_c1.kernel_size, stride=old_c1.stride, padding=old_c1.padding, bias=old_c1.bias is not None)
        self.enc_conv1 = nn.Sequential(new_c1, base.bn1, base.relu, base.maxpool)
        self.enc_layer1, self.enc_layer2, self.enc_layer3, self.enc_layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        cc = enc_ch[-1]
        self.dilated_conv1 = nn.Conv2d(cc, cc, 3, padding=1, dilation=1, bias=False)
        self.dilated_conv2 = nn.Conv2d(cc, cc, 3, padding=2, dilation=2, bias=False)
        self.dilated_conv3 = nn.Conv2d(cc, cc, 3, padding=4, dilation=4, bias=False)
        self.dilated_conv4 = nn.Conv2d(cc, cc, 3, padding=8, dilation=8, bias=False)
        self.center_bn = nn.BatchNorm2d(cc); self.center_relu = nn.ReLU(inplace=True)
        self.center_up = nn.ConvTranspose2d(cc, enc_ch[3], 2, stride=2)
        self.dec4 = self._conv_block(enc_ch[3] + enc_ch[3], enc_ch[3])
        self.dec_up3 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], 2, stride=2)
        self.dec3 = self._conv_block(enc_ch[2] + enc_ch[2], enc_ch[2])
        self.dec_up2 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], 2, stride=2)
        self.dec2 = self._conv_block(enc_ch[1] + enc_ch[1], enc_ch[1])
        self.dec1 = self._conv_block(enc_ch[1] + enc_ch[1], enc_ch[1])
        self.final = nn.Sequential(nn.Conv2d(enc_ch[1], 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, num_classes, 1))

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.enc_conv1(x)         # 1/4
        e1 = self.enc_layer1(x)       # 1/4
        e2 = self.enc_layer2(e1)      # 1/8
        e3 = self.enc_layer3(e2)      # 1/16
        e4 = self.enc_layer4(e3)      # 1/32
        c = self.dilated_conv1(e4)
        c = self.dilated_conv2(c)
        c = self.dilated_conv3(c)
        c = self.dilated_conv4(c)
        c = self.center_relu(self.center_bn(c))
        d4 = self.dec4(torch.cat([self.center_up(c), e3], dim=1))
        d3 = self.dec3(torch.cat([self.dec_up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.dec_up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        out = self.final(d1)
        out = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=False)
        return out


# --- HRNet + FPN ---
class HRNetSegmentation(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, width=48):
        super().__init__()
        import timm
        self.backbone = timm.create_model(f"hrnet_w{width}", pretrained=False, features_only=True, out_indices=[0, 1, 2, 3, 4])
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)
        self.backbone.conv1 = new_conv
        self.backbone.eval()
        with torch.no_grad():
            feat_ch = [f.shape[1] for f in self.backbone(torch.zeros(1, in_channels, 224, 224))]
        self.backbone.train()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, 128, 1) for ch in feat_ch])
        self.fpn_conv = nn.Sequential(nn.Conv2d(128 * len(feat_ch), 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.seg_head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        ins = x.shape[2:]
        feats = self.backbone(x)
        ts = feats[0].shape[2:]
        up = []
        for f, c in zip(feats, self.lateral_convs):
            z = c(f)
            if z.shape[2:] != ts:
                z = F.interpolate(z, size=ts, mode="bilinear", align_corners=False)
            up.append(z)
        fused = self.fpn_conv(torch.cat(up, dim=1))
        out = F.interpolate(fused, size=ins, mode="bilinear", align_corners=False)
        out = self.seg_head(out)
        return out


# ============================================================================
# 模型工厂
# ============================================================================

MODEL_REGISTRY = {
    "deeplabv3plus": (DeepLabV3Plus, {"encoder_name": "resnet50"}),
    "dlinknet":       (DLinkNet, {"encoder_name": "resnet34"}),
    "hrnet":          (HRNetSegmentation, {"width": 48}),
}


# ============================================================================
# 归一化 (与训练脚本一致)
# ============================================================================

def normalize_multispectral(img):
    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(np.max(img)) if img.size else 1.0
    if mx > 2000: img /= 10000.0
    elif mx > 1.5: img /= 255.0
    return np.clip(img, 0.0, 1.0)


def normalize_percentile(img, lo_pct=2, hi_pct=98):
    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = np.percentile(ch, [lo_pct, hi_pct])
        if hi > lo:
            img[:, :, c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        else:
            mx = float(ch.max()) if ch.size else 1.0
            img[:, :, c] = np.clip(ch / max(mx, 1e-8), 0.0, 1.0)
    return img.astype(np.float32)


# ============================================================================
# 推理引擎
# ============================================================================

class BaselineInferencer:
    def __init__(self, model_path, model_type="deeplabv3plus", in_channels=4,
                 device="cuda", norm_mode="legacy"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.in_channels = in_channels
        self.norm_mode = norm_mode

        # Build model
        model_cls, extra_kwargs = MODEL_REGISTRY[model_type]
        self.model = model_cls(in_channels=in_channels, num_classes=1, **extra_kwargs)
        self.model.to(self.device)

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=self.device)
        sd = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()

        # Print info
        best_iou = ckpt.get("best_iou", "?")
        params = sum(p.numel() for p in self.model.parameters())
        print(f"  模型: {model_type}")
        print(f"  参数: {params/1e6:.1f}M")
        print(f"  Best IoU: {best_iou}")
        print(f"  设备: {self.device}")
        print(f"  归一化: {norm_mode}")

    def _read_geo(self, image_path):
        """读取 GeoTIFF 的地理参考信息。"""
        import rasterio
        with rasterio.open(image_path) as ds:
            return ds.transform, ds.crs
        return None, None

    def _read_image(self, image_path):
        """读取图像并归一化。返回 (H,W,C) float32 [0,1]。"""
        sfx = Path(image_path).suffix.lower()
        if sfx in [".tif", ".tiff"]:
            import rasterio
            with rasterio.open(image_path) as ds:
                img = ds.read().astype(np.float32)
            img = np.transpose(img, (1, 2, 0))
        else:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"无法读取: {image_path}")
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # 适配通道数
        if img.shape[2] < self.in_channels:
            pad = self.in_channels - img.shape[2]
            img = np.concatenate([img, np.zeros((*img.shape[:2], pad), dtype=img.dtype)], axis=2)
        elif img.shape[2] > self.in_channels:
            img = img[:, :, :self.in_channels]

        if self.norm_mode == "percentile":
            return normalize_percentile(img)
        return normalize_multispectral(img)

    @torch.no_grad()
    def predict(self, image_path, tile_size=512, overlap=128, threshold=0.5, use_tta=False):
        """对图像进行分割预测，支持大图滑窗。返回二值掩膜 (H,W)。"""
        img = self._read_image(image_path)
        h, w = img.shape[:2]

        if h <= tile_size and w <= tile_size:
            # 小图直接预测
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            pred = torch.sigmoid(logits).squeeze().cpu().numpy()
        else:
            # 大图滑窗预测
            pred = np.zeros((h, w), dtype=np.float32)
            weight = np.zeros((h, w), dtype=np.float32)
            transform = A.Compose([A.Resize(tile_size, tile_size), ToTensorV2()])

            step = tile_size - overlap
            for y in range(0, h, step):
                for x in range(0, w, step):
                    y1, y2 = y, min(y + tile_size, h)
                    x1, x2 = x, min(x + tile_size, w)
                    patch = img[y1:y2, x1:x2]
                    # pad if needed
                    if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                        pad_h = tile_size - patch.shape[0]
                        pad_w = tile_size - patch.shape[1]
                        patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

                    aug = transform(image=patch)
                    tensor = aug["image"].unsqueeze(0).to(self.device)

                    logits = self.model(tensor)
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                    prob = prob[:min(tile_size, h - y1), :min(tile_size, w - x1)]

                    pred[y1:y1 + prob.shape[0], x1:x1 + prob.shape[1]] += prob
                    weight[y1:y1 + prob.shape[0], x1:x1 + prob.shape[1]] += 1

            pred = np.divide(pred, weight, where=weight > 0)

            # TTA: horizontal flip ensemble
            if use_tta:
                pred_flip = np.zeros((h, w), dtype=np.float32)
                weight_flip = np.zeros((h, w), dtype=np.float32)
                for y in range(0, h, step):
                    for x in range(0, w, step):
                        y1, y2 = y, min(y + tile_size, h)
                        x1, x2 = x, min(x + tile_size, w)
                        patch = img[y1:y2, x1:x2]
                        if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                            pad_h = tile_size - patch.shape[0]; pad_w = tile_size - patch.shape[1]
                            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                        patch_flip = np.fliplr(patch).copy()
                        aug = transform(image=patch_flip)
                        tensor = aug["image"].unsqueeze(0).to(self.device)
                        logits = self.model(tensor)
                        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                        prob = np.fliplr(prob)[:min(tile_size, h - y1), :min(tile_size, w - x1)]
                        pred_flip[y1:y1 + prob.shape[0], x1:x1 + prob.shape[1]] += prob
                        weight_flip[y1:y1 + prob.shape[0], x1:x1 + prob.shape[1]] += 1
                pred_flip = np.divide(pred_flip, weight_flip, where=weight_flip > 0)
                pred = (pred + pred_flip) / 2

        binary = (pred > threshold).astype(np.uint8)
        return binary, pred

    def predict_file(self, image_path, output_dir, threshold=0.5, tile_size=512,
                     overlap=128, use_tta=False, save_shp=False, min_area=50,
                     poly_epsilon=0.001, source_epsg=None, use_watershed=False):
        """对单张图像预测并保存结果。"""
        name = Path(image_path).stem
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        mask, prob = self.predict(image_path, tile_size, overlap, threshold, use_tta)

        # 分水岭后处理（分离黏连地块）
        if use_watershed:
            seg_orig = mask.copy()
            try:
                seg_ws = (mask * 255).astype(np.uint8)
                dist = cv2.distanceTransform(seg_ws, cv2.DIST_L2, 5)
                dist_norm = dist / (dist.max() + 1e-8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                local_max = cv2.dilate(dist_norm, kernel)
                sure_fg = (dist_norm == local_max) & (dist_norm > 0.1) & (seg_ws > 0)
                _, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
                if markers.max() < 2:
                    raise ValueError(f"仅找到 {markers.max()} 个种子点")
                markers = markers + 1
                markers[seg_ws == 0] = 0
                markers = cv2.watershed(
                    cv2.cvtColor((prob * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                    markers
                )
                mask = np.zeros_like(mask)
                for label in range(2, markers.max() + 1):
                    mask[markers == label] = 1
                n_fields = markers.max() - 1
                print(f"  分水岭: {n_fields} 个地块")
            except Exception as e:
                mask = seg_orig
                print(f"  分水岭: 跳过 ({e})")

        # 保存掩膜
        mask_path = Path(output_dir) / f"{name}_mask.png"
        cv2.imwrite(str(mask_path), mask * 255)
        print(f"  掩膜: {mask_path}")

        # 保存概率图
        prob_path = Path(output_dir) / f"{name}_prob.png"
        cv2.imwrite(str(prob_path), (prob * 255).astype(np.uint8))
        print(f"  概率图: {prob_path}")

        # 矢量化（带地理参考）
        if save_shp:
            transform, crs = self._read_geo(image_path)
            if crs and source_epsg is None:
                source_epsg = crs.to_epsg()
            self._vectorize(mask, name, output_dir, min_area, poly_epsilon,
                            source_epsg, transform)
            print(f"  矢量: {output_dir}/{name}.shp")

        return mask

    def _vectorize(self, mask, name, output_dir, min_area=50, poly_epsilon=0.001,
                    source_epsg=None, transform=None):
        """将掩膜转为 SHP 文件（像素坐标 → 地理坐标）。"""
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon, MultiPolygon
            from shapely import simplify as shapely_simplify
            from rasterio.transform import xy as rio_xy
        except ImportError:
            print("  [WARN] geopandas/shapely 未安装，跳过矢量化")
            return

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            epsilon = poly_epsilon * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                if transform:
                    pts = [rio_xy(transform, float(p[0][1]), float(p[0][0]))
                           for p in approx]
                else:
                    pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
                poly = Polygon(pts)
                if poly.is_valid:
                    poly = shapely_simplify(poly, epsilon, preserve_topology=True)
                    polygons.append(poly)

        if not polygons:
            print(f"  [WARN] {name}: 未生成有效多边形")
            return

        crs = f"EPSG:{source_epsg}" if source_epsg else "EPSG:4326"
        gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=crs)
        shp_path = Path(output_dir) / f"{name}.shp"
        gdf.to_file(shp_path, encoding="utf-8")


# ============================================================================
# 主入口
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="基线模型推理 + 矢量化")
    parser.add_argument("--model", required=True, help=".pth 模型权重路径")
    parser.add_argument("--model_type", default="deeplabv3plus",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="模型类型 (默认: deeplabv3plus)")
    parser.add_argument("--input", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--norm_mode", default="legacy", choices=["legacy", "percentile"])
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true", help="测试时增强 (水平翻转集成)")
    parser.add_argument("--save_shp", action="store_true", help="保存为 SHP 矢量")
    parser.add_argument("--min_area", type=float, default=50, help="最小多边形面积 (像素)")
    parser.add_argument("--poly_epsilon", type=float, default=0.001)
    parser.add_argument("--source_epsg", type=int, default=None, help="源坐标系 EPSG 编码")
    parser.add_argument("--use_watershed", action="store_true", help="分水岭后处理（分离黏连地块，慢）")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # 初始化推理器
    inferencer = BaselineInferencer(
        model_path=args.model,
        model_type=args.model_type,
        in_channels=args.in_channels,
        device=args.device,
        norm_mode=args.norm_mode,
    )

    # 处理输入
    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.tif")) + sorted(input_path.glob("*.tiff")) \
              + sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
        print(f"找到 {len(files)} 个图像文件")

    # 批量推理
    for f in tqdm(files, desc="推理进度"):
            inferencer.predict_file(
                str(f), args.output,
                threshold=args.threshold,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
                use_tta=args.tta,
                save_shp=args.save_shp,
                min_area=args.min_area,
                poly_epsilon=args.poly_epsilon,
                source_epsg=args.source_epsg,
                use_watershed=args.use_watershed,
            )

    print(f"\n完成! 结果保存至: {args.output}")


if __name__ == "__main__":
    main()
