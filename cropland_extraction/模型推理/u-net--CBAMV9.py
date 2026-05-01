# u-net--CBAMV9.py
# V8改进版：全分辨率解码级 + 深层融合头 + 鲁棒增强 + 自适应归一化
# 改进要点：
#   1. 新增 decoder0：利用 stem 跳连接在 512×512 分辨率做可学习运算
#   2. 简化多任务头：去掉 ConvTranspose2d，在原生分辨率直接推理
#   3. 加深融合头：从单层 3×3 改为 4 层卷积
#   4. RandomScale 0.5-2.0x 增强
#   5. 逐通道百分位归一化
#   6. 最优阈值搜索
#   7. SWA (Stochastic Weight Averaging)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
import random
import copy
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 注意力模块 ====================

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
    """轻量光谱注意力：对多通道输入进行通道重标定。"""

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


# ==================== V9 改进：CBAMUNet + decoder0 ====================

class CBAMUNet(nn.Module):
    """
    V9: 在 V8 基础上新增 decoder0 全分辨率解码级。
    - stem = conv1+bn1+relu（保留 256×256 分辨率）
    - encoder1 只做 maxpool（降到 128×128）
    - decoder0 将 d1(256×256) 升采样到 512×512，融合 stem 跳连接
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1,
        use_cbam=True,
        use_spectral_attention=True,
        dropout_rate=0.2,
    ):
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

        # 多通道 conv1 适配
        if in_channels != 3:
            old_conv1 = base_model.conv1
            new_conv1 = nn.Conv2d(
                in_channels,
                old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=False,
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

        # 【V9】分离 stem（256×256）和 encoder1（pool → 128×128）
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

        # 【V9】decoder0：256×256 → 512×512，融合 stem 跳连接
        self.decoder0_up = nn.ConvTranspose2d(
            decoder_channels[3], decoder_channels[3],
            kernel_size=4, stride=2, padding=1
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
        x_stem = self.stem(x)       # 64ch, 256×256（stem 保留高分辨率）
        e1 = self.encoder1(x_stem)  # 64ch, 128×128（pool 后）
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        b = self.bridge(e5)
        d4 = self.decoder4(b, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)  # 32ch, 256×256

        # decoder0：d1(256×256) → 512×512 + stem 跳连接
        d0_up = self.decoder0_up(d1)          # 32ch, 512×512
        stem_up = F.interpolate(x_stem, scale_factor=2, mode="bilinear", align_corners=True)  # 64ch, 512×512
        d0 = torch.cat([d0_up, stem_up], dim=1)  # 96ch, 512×512
        d0 = self.decoder0_conv(d0)            # 32ch, 512×512
        return d0


# ==================== V9 改进：简化多任务头 + 深层融合 ====================

class MultiTaskUNet(nn.Module):
    """
    V9: 简化多任务头（移除 ConvTranspose2d，输入已为 512×512）
    融合头加深为 4 层卷积。
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1,
        use_cbam=True,
        use_spectral_attention=True,
        dropout_rate=0.2,
    ):
        super().__init__()

        self.base = CBAMUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            use_cbam=use_cbam,
            use_spectral_attention=use_spectral_attention,
            dropout_rate=dropout_rate,
        )

        # 【V9】边界头：去掉 ConvTranspose2d，直接在 512×512 运算
        self.boundary_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # 【V9】距离头：同样去掉 ConvTranspose2d
        self.distance_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 【V9】分割特征提取
        self.seg_up = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 【V9 改进】深层融合头：单层 → 4 层
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
        d0 = self.base(x)                      # 32ch, 512×512
        boundary_logit = self.boundary_head(d0)  # 1ch, 512×512
        distance_logit = self.distance_head(d0)  # 1ch, 512×512
        seg_feat = self.seg_up(d0)               # 16ch, 512×512

        boundary_pred = torch.sigmoid(boundary_logit)
        distance_pred = distance_logit
        cat_feat = torch.cat([seg_feat, boundary_pred, distance_pred], dim=1)
        seg_logit = self.fusion(cat_feat)         # 1ch, 512×512
        return seg_logit, boundary_logit, distance_logit


# ==================== 数据集 ====================

class FarmlandDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, img_size=512,
                 in_channels=4, norm_mode="percentile", debug=False):
        """
        Args:
            norm_mode: "percentile"（逐通道百分位归一化）或 "legacy"（全局 max）
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.in_channels = in_channels
        self.norm_mode = norm_mode
        self.debug = debug

        self.img_dir = self.root_dir / split / "img"
        self.mask_dir = self.root_dir / split / "lbl"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.img_dir}")

        self.image_files = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]]
        )
        print(f"找到 {len(self.image_files)} 个{split}图像")

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def _normalize_legacy(img):
        """V8 原始全局 max 归一化（保留兼容）"""
        img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        max_val = float(np.max(img)) if img.size else 1.0
        if max_val > 2000:
            img = img / 10000.0
        elif max_val > 1.5:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def _normalize_percentile(img, lower_pct=2, upper_pct=98):
        """【V9】逐通道百分位归一化，对多光谱更鲁棒"""
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

    def _read_image(self, img_path):
        suffix = img_path.suffix.lower()
        if suffix in [".tif", ".tiff"]:
            with rasterio.open(img_path) as ds:
                arr = ds.read().astype(np.float32)
            arr = np.transpose(arr, (1, 2, 0))
        else:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"无法读取图像: {img_path}")
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        if arr.shape[2] < self.in_channels:
            pad_ch = self.in_channels - arr.shape[2]
            arr = np.concatenate([arr, np.zeros((arr.shape[0], arr.shape[1], pad_ch), dtype=arr.dtype)], axis=2)
        elif arr.shape[2] > self.in_channels:
            arr = arr[:, :, :self.in_channels]

        if self.norm_mode == "percentile":
            return self._normalize_percentile(arr)
        return self._normalize_legacy(arr)

    def _read_mask(self, img_path):
        mask_path = self.mask_dir / f"{img_path.stem}.tif"
        if not mask_path.exists():
            mask_path_png = self.mask_dir / f"{img_path.stem}.png"
            mask_path = mask_path_png if mask_path_png.exists() else mask_path

        if not mask_path.exists():
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        if mask_path.suffix.lower() in [".tif", ".tiff"]:
            with rasterio.open(mask_path) as ds:
                mask = ds.read(1)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        return (mask > 0).astype(np.uint8)

    @staticmethod
    def generate_distance_map(mask):
        mask_uint8 = (mask * 255).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        max_dist = dist_transform.max()
        dist_normalized = dist_transform / max_dist if max_dist > 0 else dist_transform
        return dist_normalized.astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = self._read_image(img_path)
        mask = self._read_mask(img_path)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        distance_map = self.generate_distance_map(mask)

        if self.transform:
            augmented = self.transform(image=img, mask=mask, distance_map=distance_map)
            img = augmented["image"]
            mask = augmented["mask"]
            distance_map = augmented["distance_map"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float()
            distance_map = torch.from_numpy(distance_map).float()

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        if not isinstance(distance_map, torch.Tensor):
            distance_map = torch.from_numpy(distance_map).float()

        return img, mask.float(), distance_map


# ==================== V9 增强：RandomScale + 自适应归一化 ====================

def get_transforms(img_size, phase="train", use_color_aug=True):
    targets = {"distance_map": "mask"}

    if phase == "train":
        aug_list = [
            # 【V9】RandomScale 0.5-2.0x：PRUE 论文验证的关键增强
            A.RandomScale(scale_limit=(-0.5, 1.0), p=0.5),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.ChannelShuffle(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ]
        return A.Compose(aug_list, additional_targets=targets)

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ],
        additional_targets=targets,
    )


# ==================== 损失函数 ====================

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
        return dice_loss + bce_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class LogCoshDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        return torch.log(torch.cosh(1 - dice))


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        tp = (inputs_flat * targets_flat).sum()
        fp = (inputs_flat * (1 - targets_flat)).sum()
        fn = ((1 - inputs_flat) * targets_flat).sum()
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    def __init__(self, loss_type="log_cosh_dice", dice_weight=0.5, aux_weight=0.5,
                 tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight

        if loss_type == "log_cosh_dice":
            self.primary = LogCoshDiceLoss()
            self.aux = FocalLoss()
        elif loss_type == "tversky":
            self.primary = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
            self.aux = FocalLoss()
        elif loss_type == "dice_bce":
            self.primary = DiceBCELoss()
            self.aux = FocalLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, inputs, targets):
        return self.dice_weight * self.primary(inputs, targets) + self.aux_weight * self.aux(inputs, targets)


# ==================== V9 训练器：含 SWA + 阈值搜索 ====================

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(">>> 使用多任务模型 V9 (Decoder0 + DeepFusion + RandomScale)")
        self.model = MultiTaskUNet(
            encoder_name=config["encoder_name"],
            encoder_weights=config["encoder_weights"],
            in_channels=config["in_channels"],
            use_cbam=config["use_cbam"],
            use_spectral_attention=config["use_spectral_attention"],
            dropout_rate=config.get("dropout_rate", 0.2),
        ).to(self.device)

        loss_type = config.get("loss_type", "log_cosh_dice")
        self.criterion_seg = CombinedLoss(
            loss_type=loss_type,
            dice_weight=config.get("seg_dice_weight", 0.5),
            aux_weight=config.get("seg_focal_weight", 0.5),
            tversky_alpha=config.get("tversky_alpha", 0.3),
            tversky_beta=config.get("tversky_beta", 0.7),
        )
        self.criterion_bdy = DiceBCELoss()
        self.criterion_dist = nn.L1Loss()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        self.warmup_epochs = config.get("warmup_epochs", 2)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=self.warmup_epochs
        )
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=config["min_lr"],
        )
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.scheduler = main_scheduler
        self._warmup_done = False

        norm_mode = config.get("norm_mode", "percentile")
        train_ds = FarmlandDataset(
            config["data_root"],
            "train",
            get_transforms(config["img_size"], "train", use_color_aug=True),
            config["img_size"],
            in_channels=config["in_channels"],
            norm_mode=norm_mode,
            debug=config.get("debug_mode", False),
        )
        val_ds = FarmlandDataset(
            config["data_root"],
            "val",
            get_transforms(config["img_size"], "val", use_color_aug=False),
            config["img_size"],
            in_channels=config["in_channels"],
            norm_mode=norm_mode,
            debug=False,
        )

        num_workers = config.get("num_workers", 0)
        self.train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=config["batch_size"], shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = GradScaler() if config["use_amp"] else None

        self.start_epoch = 0
        self.best_iou = 0.0
        self.best_bdy_iou = 0.0
        self.train_losses = []
        self.val_ious = []
        self.val_bdy_ious = []
        self.boundary_losses = []
        self.distance_losses = []

        # 【V9】SWA 支持
        self.swa_start = config.get("swa_start", 0)  # 0 = 不启用
        self.swa_model = None
        self.swa_count = 0

        resume_path = config.get("resume_path")
        if resume_path and Path(resume_path).exists():
            self._load_checkpoint(resume_path)

        self.min_boundary_weight = config.get("min_boundary_weight", 0.1)

        print(f"设备: {self.device}")
        print(f"训练样本: {len(train_ds)}")
        print(f"验证样本: {len(val_ds)}")
        print(f"输入通道数: {config['in_channels']}")
        print(f"损失函数: {loss_type}")
        print(f"归一化模式: {norm_mode}")
        print(f"SWA: 自 epoch {self.swa_start} 启用" if self.swa_start > 0 else "SWA: 未启用")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")

    def _load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt.get("scheduler_state_dict", {}))
        self.start_epoch = ckpt.get("epoch", 0)
        self.best_iou = ckpt.get("best_iou", 0.0)
        self.best_bdy_iou = ckpt.get("best_bdy_iou", 0.0)
        self.train_losses = ckpt.get("train_losses", [])
        self.val_ious = ckpt.get("val_ious", [])
        self.val_bdy_ious = ckpt.get("val_bdy_ious", [])
        self.boundary_losses = ckpt.get("boundary_losses", [])
        self.distance_losses = ckpt.get("distance_losses", [])
        print(f">>> 从 {ckpt_path} 恢复训练 (epoch {self.start_epoch}, best_iou={self.best_iou:.4f})")

    @staticmethod
    def get_boundary_targets(masks):
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        kernel_size = 3
        padding = kernel_size // 2
        eroded = -F.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=padding)
        dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=padding)
        return dilated - eroded

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        seg_loss_sum = 0.0
        bdy_loss_sum = 0.0
        dist_loss_sum = 0.0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")

        for batch_idx, (imgs, masks, distance_maps) in enumerate(pbar):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).float()
            distance_maps = distance_maps.to(self.device).float()

            if self.config["use_amp"]:
                with autocast():
                    seg_logits, bdy_logits, dist_logits = self.model(imgs)
                    bdy_targets = self.get_boundary_targets(masks)

                    if masks.dim() == 3:
                        masks = masks.unsqueeze(1)
                    if distance_maps.dim() == 3:
                        distance_maps = distance_maps.unsqueeze(1)

                    loss_seg = self.criterion_seg(seg_logits, masks)
                    loss_bdy = self.criterion_bdy(bdy_logits, bdy_targets)
                    loss_dist = self.criterion_dist(dist_logits, distance_maps)

                    dynamic_boundary_weight = self.config["boundary_weight"]
                    if epoch > 10:
                        dynamic_boundary_weight *= 0.5
                    if epoch > 20:
                        dynamic_boundary_weight *= 0.5
                    dynamic_boundary_weight = max(dynamic_boundary_weight, self.min_boundary_weight)

                    loss = loss_seg + dynamic_boundary_weight * loss_bdy + 0.25 * loss_dist
                    loss = loss / self.config["gradient_accumulation_steps"]

                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.config["gradient_accumulation_steps"] == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                seg_logits, bdy_logits, dist_logits = self.model(imgs)
                bdy_targets = self.get_boundary_targets(masks)

                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                if distance_maps.dim() == 3:
                    distance_maps = distance_maps.unsqueeze(1)

                loss_seg = self.criterion_seg(seg_logits, masks)
                loss_bdy = self.criterion_bdy(bdy_logits, bdy_targets)
                loss_dist = self.criterion_dist(dist_logits, distance_maps)

                dynamic_boundary_weight = self.config["boundary_weight"]
                if epoch > 10:
                    dynamic_boundary_weight *= 0.5
                if epoch > 20:
                    dynamic_boundary_weight *= 0.5
                dynamic_boundary_weight = max(dynamic_boundary_weight, self.min_boundary_weight)

                loss = loss_seg + dynamic_boundary_weight * loss_bdy + 0.25 * loss_dist
                loss = loss / self.config["gradient_accumulation_steps"]
                loss.backward()

                if (batch_idx + 1) % self.config["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            epoch_loss += loss.item() * self.config["gradient_accumulation_steps"]
            seg_loss_sum += loss_seg.item()
            bdy_loss_sum += loss_bdy.item()
            dist_loss_sum += loss_dist.item()

            pbar.set_postfix_str(
                f"L_seg: {loss_seg.item():.3f} | L_bdy: {loss_bdy.item():.3f} | L_dist: {loss_dist.item():.3f}"
            )

        avg_loss = epoch_loss / len(self.train_loader)
        avg_seg_loss = seg_loss_sum / len(self.train_loader)
        avg_bdy_loss = bdy_loss_sum / len(self.train_loader)
        avg_dist_loss = dist_loss_sum / len(self.train_loader)

        self.train_losses.append(avg_loss)
        self.boundary_losses.append(avg_bdy_loss)
        self.distance_losses.append(avg_dist_loss)

        return avg_loss, avg_seg_loss, avg_bdy_loss, avg_dist_loss

    @torch.no_grad()
    def validate(self, threshold=0.5):
        """验证并计算 IoU，支持指定阈值"""
        self.model.eval()
        total_iou = 0.0
        total_bdy_iou = 0.0
        for imgs, masks, _ in tqdm(self.val_loader, desc="Validating"):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            seg_logits, bdy_logits, _ = self.model(imgs)

            pred = (torch.sigmoid(seg_logits) > threshold).long()
            if pred.shape[1] == 1:
                pred = pred.squeeze(1)

            intersection = (pred & masks.long()).float().sum((1, 2))
            union = (pred | masks.long()).float().sum((1, 2))
            iou = (intersection + 1e-6) / (union + 1e-6)
            total_iou += iou.mean().item()

            bdy_targets = self.get_boundary_targets(masks)
            bdy_pred = (torch.sigmoid(bdy_logits) > 0.5).long()
            if bdy_pred.shape[1] == 1:
                bdy_pred = bdy_pred.squeeze(1)
            if bdy_targets.dim() > bdy_pred.dim():
                bdy_targets = bdy_targets.squeeze(1)
            bdy_intersection = (bdy_pred & bdy_targets.long()).float().sum((1, 2))
            bdy_union = (bdy_pred | bdy_targets.long()).float().sum((1, 2))
            bdy_iou = (bdy_intersection + 1e-6) / (bdy_union + 1e-6)
            total_bdy_iou += bdy_iou.mean().item()

        return total_iou / len(self.val_loader), total_bdy_iou / len(self.val_loader)

    # 【V9】最优阈值搜索
    def find_best_threshold(self, thresholds=None):
        """在验证集上扫描阈值，返回最佳阈值和对应 IoU"""
        if thresholds is None:
            thresholds = np.arange(0.25, 0.81, 0.05)
        print(f"\n{'=' * 60}")
        print("扫描最优分割阈值...")
        print(f"{'=' * 60}")

        best_th = 0.5
        best_iou = 0.0
        results = []
        for th in thresholds:
            iou, _ = self.validate(threshold=float(th))
            results.append((th, iou))
            print(f"  阈值 {th:.2f} → IoU {iou:.4f}")
            if iou > best_iou:
                best_iou = iou
                best_th = th

        print(f"\n✅ 最佳阈值: {best_th:.2f} (IoU: {best_iou:.4f})")
        print(f"{'=' * 60}")
        return float(best_th), best_iou

    def save_model(self, filename, model=None, extra=None):
        model = model or self.model
        model_path = self.output_dir / filename
        state = {
            "epoch": self.start_epoch + len(self.train_losses),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.main_scheduler.state_dict(),
            "best_iou": self.best_iou,
            "best_bdy_iou": self.best_bdy_iou,
            "train_losses": self.train_losses,
            "val_ious": self.val_ious,
            "val_bdy_ious": self.val_bdy_ious,
            "boundary_losses": self.boundary_losses,
            "distance_losses": self.distance_losses,
            "config": self.config,
        }
        if extra:
            state.update(extra)
        torch.save(state, model_path)
        print(f"模型已保存到: {model_path}")

    # 【V9】SWA update
    def _swa_update(self):
        if self.swa_model is None:
            self.swa_model = copy.deepcopy(self.model)
            self.swa_count = 1
        else:
            # 指数移动平均
            decay = 1.0 / (self.swa_count + 1)
            for swa_param, param in zip(self.swa_model.parameters(), self.model.parameters()):
                swa_param.data.copy_(swa_param.data * (1 - decay) + param.data * decay)
            self.swa_count += 1

    def plot_training_curves(self):
        plt.figure(figsize=(30, 5))

        plt.subplot(1, 5, 1)
        plt.plot(self.train_losses, label="Total Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 5, 2)
        plt.plot(self.val_ious, label="Val IoU", linewidth=2, color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title("Validation IoU")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 5, 3)
        if self.val_bdy_ious:
            plt.plot(self.val_bdy_ious, label="Val Boundary IoU", linewidth=2, color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("Boundary IoU")
        plt.title("Boundary IoU")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 5, 4)
        plt.plot(self.boundary_losses, label="Boundary Loss", linewidth=2, color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Boundary Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 5, 5)
        plt.plot(self.distance_losses, label="Distance Loss", linewidth=2, color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Distance Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

    def run(self):
        print("开始训练V9 (Decoder0 + DeepFusion + RandomScale)...")
        patience_counter = 0
        early_stopping_patience = self.config.get("early_stopping_patience", 8)

        for epoch in range(self.start_epoch + 1, self.config["epochs"] + 1):
            print("\n" + "=" * 70)
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print("=" * 70)

            train_loss, seg_loss, bdy_loss, dist_loss = self.train_one_epoch(epoch)
            val_iou, val_bdy_iou = self.validate()
            self.val_ious.append(val_iou)
            self.val_bdy_ious.append(val_bdy_iou)

            # 【V9】SWA 更新
            if self.swa_start > 0 and epoch >= self.swa_start:
                self._swa_update()
                # 用 SWA 模型评估
                orig_model = self.model
                self.model = self.swa_model
                swa_iou, swa_bdy = self.validate()
                self.model = orig_model
                print(f"  SWA IoU: {swa_iou:.4f}")
                # 如果 SWA 更好，保存
                if swa_iou > self.best_iou:
                    self.best_iou = swa_iou
                    self.save_model("best_model_swa.pth", model=self.swa_model)

            if epoch <= self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                if not self._warmup_done:
                    self._warmup_done = True
                self.main_scheduler.step(val_iou)

            print(f"训练损失: {train_loss:.4f} (分割: {seg_loss:.4f}, 边界: {bdy_loss:.4f}, 距离: {dist_loss:.4f})")
            print(f"验证IoU: {val_iou:.4f} | 边界IoU: {val_bdy_iou:.4f}")

            if val_iou > self.best_iou:
                self.best_iou = val_iou
                patience_counter = 0
                self.save_model("best_model.pth")
                print(f"✅ 新的最佳模型! IoU: {val_iou:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ 未提升，耐心计数: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"\n🛑 早停触发！连续{early_stopping_patience}轮未提升")
                    break

            if val_bdy_iou > self.best_bdy_iou:
                self.best_bdy_iou = val_bdy_iou
                self.save_model("best_boundary_model.pth")

            if epoch % self.config.get("save_interval", 5) == 0:
                self.save_model(f"checkpoint_epoch_{epoch}.pth")
            if epoch % self.config.get("plot_interval", 2) == 0:
                self.plot_training_curves()

        self.save_model("final_model.pth")

        # 【V9】训练结束后寻找最佳阈值
        if self.config.get("auto_find_threshold", True):
            best_th, best_th_iou = self.find_best_threshold()
            self.save_model("final_model.pth", extra={"best_threshold": best_th, "best_threshold_iou": best_th_iou})

        print("\n" + "=" * 70)
        print(f"训练完成! 最佳验证IoU: {self.best_iou:.4f} | 最佳边界IoU: {self.best_bdy_iou:.4f}")
        print("=" * 70)


if __name__ == "__main__":
    import argparse

    set_seed(42)
    parser = argparse.ArgumentParser(description="U-Net Training V9 (Decoder0 + DeepFusion + Robust Aug)")
    parser.add_argument("--data_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出保存目录")
    parser.add_argument("--encoder_name", type=str, default="resnet50", help="骨干网络")
    parser.add_argument("--in_channels", type=int, default=4, help="输入光谱通道数")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Num workers")

    parser.add_argument("--loss_type", type=str, default="log_cosh_dice",
                        choices=["log_cosh_dice", "dice_bce", "tversky"],
                        help="损失函数类型 (PRUE推荐log_cosh_dice)")
    parser.add_argument("--norm_mode", type=str, default="percentile",
                        choices=["percentile", "legacy"],
                        help="归一化模式: percentile=逐通道百分位, legacy=全局max")
    parser.add_argument("--swa_start", type=int, default=30,
                        help="SWA 起始 epoch (0=不启用)")
    parser.add_argument("--auto_threshold", action="store_true", default=True,
                        help="训练结束后自动搜索最优阈值")
    parser.add_argument("--no_auto_threshold", action="store_false", dest="auto_threshold")
    parser.add_argument("--resume", type=str, default=None, help="断点续训 checkpoint 路径")

    args = parser.parse_args()

    config = {
        "data_root": args.data_root,
        "output_dir": args.output_dir,
        "encoder_name": args.encoder_name,
        "encoder_weights": "imagenet",
        "in_channels": args.in_channels,
        "use_cbam": True,
        "use_spectral_attention": True,
        "img_size": 512,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "min_lr": 1e-7,
        "weight_decay": 1e-4,
        "loss_type": args.loss_type,
        "seg_dice_weight": 0.5,
        "seg_focal_weight": 0.5,
        "tversky_alpha": 0.3,
        "tversky_beta": 0.7,
        "boundary_weight": 1.0,
        "num_workers": args.num_workers,
        "save_interval": 5,
        "plot_interval": 2,
        "use_amp": True,
        "gradient_accumulation_steps": 1,
        "warmup_epochs": 2,
        "early_stopping_patience": 8,
        "dropout_rate": 0.2,
        "debug_mode": False,
        "min_boundary_weight": 0.3,
        "norm_mode": args.norm_mode,
        "swa_start": args.swa_start,
        "auto_find_threshold": args.auto_threshold,
        "resume_path": args.resume,
    }

    trainer = Trainer(config)
    trainer.run()
