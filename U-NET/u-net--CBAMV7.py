# u-net--CBAMV7.py
# 终极优化版本：整合V4和V5的所有优点 + u-net.py的掩膜加载逻辑 + 距离变换回归任务
# 针对RTX 4090D 24GB显存优化

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import random
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')


# ==================== 1. 设置随机种子 (保证可复现) ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 2. CBAM注意力模块 ====================
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


# ==================== 3. 优化的解码器模块 (带Dropout) ====================
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


# ==================== 4. CBAM U-Net (基础模型) ====================
class CBAMUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, use_cbam=True, dropout_rate=0.2):
        super(CBAMUNet, self).__init__()

        import torchvision.models as models
        if encoder_name == 'resnet34':
            base_model = models.resnet34(weights='IMAGENET1K_V1' if encoder_weights else None)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            base_model = models.resnet50(weights='IMAGENET1K_V1' if encoder_weights else None)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            base_model = models.resnet18(weights='IMAGENET1K_V1' if encoder_weights else None)
            encoder_channels = [64, 64, 128, 256, 512]

        self.encoder1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.encoder2 = base_model.layer1
        self.encoder3 = base_model.layer2
        self.encoder4 = base_model.layer3
        self.encoder5 = base_model.layer4

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


# ==================== 5. 多任务U-Net (Field-Net优化版) ====================
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


# ==================== 6. 数据集与增强 (采用u-net.py的掩膜加载逻辑) ====================
class FarmlandDataset(Dataset):
    """
    耕地分割数据集类
    采用u-net.py的智能掩膜加载逻辑
    """

    def __init__(self, root_dir, split='train', transform=None, img_size=512, debug=False):
        """
        初始化数据集

        参数:
        - root_dir: 数据集根目录
        - split: 数据集分割 (train/val/test)
        - transform: 数据增强变换
        - img_size: 图像尺寸
        - debug: 是否开启调试模式
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.debug = debug

        # 根据数据集结构设置路径
        if split == 'test':
            # 测试集可能有不同的区域子目录
            self.image_files = []
            test_regions = [
                'Huang-Huai-Hai Plain', 'Loess Plateau', 'Northeast China Plain',
                'Northern Arid and Semi-arid Region', 'Sichuan Basin',
                'South China Areas', 'Yangtze River Middle and Lower Reaches Plain',
                'Yungui Plateau'
            ]

            for region in test_regions:
                region_path = self.root_dir / 'test' / region
                if region_path.exists():
                    png_files = list(region_path.glob('*.png'))
                    self.image_files.extend(png_files)

            # 如果没有找到，尝试在test目录下直接查找
            if len(self.image_files) == 0:
                test_dir = self.root_dir / 'test'
                self.image_files = list(test_dir.glob('*.png'))

            # 由于测试集可能没有标签，设置mask_dir为None
            self.mask_dir = None
        else:
            # 训练集和验证集
            self.img_dir = self.root_dir / split / 'img'
            self.mask_dir = self.root_dir / split / 'lbl'

            # 获取所有图像文件
            self.image_files = list(self.img_dir.glob('*.png'))

            if not self.img_dir.exists():
                print(f"警告: 图像目录不存在: {self.img_dir}")

            if not self.mask_dir.exists():
                print(f"警告: 掩膜目录不存在: {self.mask_dir}")

        print(f"找到 {len(self.image_files)} 个{split}图像")

    def __len__(self):
        return len(self.image_files)

    def load_mask(self, mask_path):
        """
        【u-net.py逻辑】智能掩膜加载
        自动判断并反转掩膜，彻底解决标签反转问题
        """
        if not mask_path.exists():
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取掩膜: {mask_path}")
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # 只在调试模式下打印掩膜信息
        if self.debug:
            print(f"\n调试掩膜: {mask_path}")
            print(f"掩膜形状: {mask.shape}")
            print(f"掩膜唯一值: {np.unique(mask)}")
            print(f"掩膜值范围: [{mask.min()}, {mask.max()}]")
            print(f"掩膜数据类型: {mask.dtype}")

        # 根据直方图选择合适的二值化方法
        unique_values = np.unique(mask)

        if self.debug:
            print(f"唯一像素值: {unique_values}")

        # 如果只有两个值，直接使用阈值
        if len(unique_values) == 2:
            if self.debug:
                print(f"二值图像，唯一值: {unique_values}")

            # 通常耕地是0或255，选择较小的值作为耕地
            if 0 in unique_values and 255 in unique_values:
                mask_binary = (mask == 0).astype(np.uint8)  # 假设黑色是耕地
                if self.debug:
                    print(f"选择二值化: 黑色(0)为耕地，白色(255)为背景")
            else:
                # 使用Otsu自适应阈值
                _, mask_binary = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if self.debug:
                    print(f"使用Otsu阈值")
        else:
            # 使用Otsu自适应阈值
            _, mask_binary = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if self.debug:
                print(f"使用Otsu自适应阈值")

        if self.debug:
            print(f"二值化后耕地比例: {mask_binary.sum() / mask_binary.size:.4f}")

        return mask_binary

    def generate_distance_map(self, mask):
        """
        【Field-Net逻辑】生成距离变换标签
        计算每个耕地像素到最近背景像素的欧氏距离
        """
        mask_uint8 = (mask * 255).astype(np.uint8)

        dist_transform = cv2.distanceTransform(
            mask_uint8,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE
        )

        max_dist = dist_transform.max()
        if max_dist > 0:
            dist_normalized = dist_transform / max_dist
        else:
            dist_normalized = dist_transform

        return dist_normalized.astype(np.float32)

    def __getitem__(self, idx):
        try:
            img_path = self.image_files[idx]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 无法读取图像 {img_path}，跳过")
                return self.__getitem__((idx + 1) % len(self))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 加载掩膜
            if self.split == 'test':
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                mask_filename = img_path.name
                mask_path = self.mask_dir / mask_filename
                mask = self.load_mask(mask_path)

            # 调整尺寸
            img = cv2.resize(img, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            # 生成距离标签
            distance_map = self.generate_distance_map(mask)

            # 应用数据增强
            if self.transform:
                # 【修正】直接以参数形式传入 distance_map
                augmented = self.transform(
                    image=img,
                    mask=mask,
                    distance_map=distance_map  # <--- 对应 get_transforms 里的 key
                )
                img = augmented['image']
                mask = augmented['mask']
                distance_map = augmented['distance_map']
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                mask = torch.from_numpy(mask).float()
                distance_map = torch.from_numpy(distance_map).float()

            # 确保是Tensor
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            if not isinstance(distance_map, torch.Tensor):
                distance_map = torch.from_numpy(distance_map).float()

            return img, mask.float(), distance_map

        except Exception as e:
            print(f"处理图像 {self.image_files[idx]} 时出错: {e}")
            return self.__getitem__((idx + 1) % len(self))


def get_transforms(img_size, phase='train'):
    """
    【修正版】正确配置 additional_targets
    """
    # 1. 定义额外的目标：告诉库 distance_map 是一个掩膜(mask)
    # 这样它就会跟随 image 进行旋转、翻转，但不会被色彩增强影响
    targets = {'distance_map': 'mask'}

    if phase == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(p=1.0),
            ], p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets=targets)  # <--- 必须加这个参数！
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets=targets)  # <--- 必须加这个参数！


# ==================== 7. 损失函数 (优化版) ====================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        return dice_loss + bce_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_bce = DiceBCELoss()
        self.focal = FocalLoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice_bce(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


# ==================== 8. 训练器 (V4+V5终极整合) ====================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f">>> 使用多任务模型 (Multi Task + CBAM + Distance) - V7终极优化版")
        self.model = MultiTaskUNet(
            encoder_name=config['encoder_name'],
            encoder_weights=config['encoder_weights'],
            use_cbam=config['use_cbam'],
            dropout_rate=config.get('dropout_rate', 0.2)
        ).to(self.device)

        self.criterion_seg = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
        self.criterion_bdy = DiceBCELoss()
        self.criterion_dist = nn.L1Loss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=config['min_lr']
        )

        self.warmup_epochs = config.get('warmup_epochs', 2)
        self.current_epoch = 0

        train_ds = FarmlandDataset(
            config['data_root'],
            'train',
            get_transforms(config['img_size'], 'train'),
            config['img_size'],
            debug=config.get('debug_mode', False)
        )
        val_ds = FarmlandDataset(
            config['data_root'],
            'val',
            get_transforms(config['img_size'], 'val'),
            config['img_size'],
            debug=False
        )

        num_workers = config.get('num_workers', 0)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.best_iou = 0.0
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = GradScaler() if config['use_amp'] else None

        self.train_losses = []
        self.val_ious = []
        self.boundary_losses = []
        self.distance_losses = []  # <--- 必须加上这一行！

        print(f"设备: {self.device}")
        print(f"训练样本: {len(train_ds)}")
        print(f"验证样本: {len(val_ds)}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"混合精度训练: {config['use_amp']}")
        print(f"梯度累积步数: {config['gradient_accumulation_steps']}")
        print(f"Dropout率: {config.get('dropout_rate', 0.2)}")

        # 【V5特性】断点续训逻辑
        if 'resume_path' in config and config['resume_path'] is not None:
            checkpoint_path = config['resume_path']
            if os.path.exists(checkpoint_path):
                print(f"🔄 正在加载预训练模型: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'best_iou' in checkpoint:
                    self.best_iou = checkpoint['best_iou']
                    print(f"   继承最佳 IoU: {self.best_iou:.4f}")
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'train_losses' in checkpoint:
                    self.train_losses = checkpoint['train_losses']
                if 'val_ious' in checkpoint:
                    self.val_ious = checkpoint['val_ious']
                print("✅ 模型权重加载成功，继续训练！")
            else:
                print(f"⚠️  检查点文件不存在: {checkpoint_path}")

    def get_boundary_targets(self, masks):
        """
        【V4优化】生成边界，使用较小的Kernel防止小地块消失
        """
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        kernel_size = 3
        padding = kernel_size // 2

        eroded = -F.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=padding)
        dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=padding)

        boundary = dilated - eroded
        return boundary

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        seg_loss_sum = 0
        bdy_loss_sum = 0
        dist_loss_sum = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}')
        for batch_idx, (imgs, masks, distance_maps) in enumerate(pbar):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).float()
            distance_maps = distance_maps.to(self.device).float()

            if self.config['use_amp']:
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

                    dynamic_boundary_weight = self.config['boundary_weight']
                    if epoch > 5:
                        dynamic_boundary_weight *= 0.5
                    if epoch > 10:
                        dynamic_boundary_weight *= 0.2

                    loss = loss_seg + dynamic_boundary_weight * loss_bdy + 0.25 * loss_dist
                    loss = loss / self.config['gradient_accumulation_steps']

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
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

                dynamic_boundary_weight = self.config['boundary_weight']
                if epoch > 5:
                    dynamic_boundary_weight *= 0.5
                if epoch > 10:
                    dynamic_boundary_weight *= 0.2

                loss = loss_seg + dynamic_boundary_weight * loss_bdy + 0.25 * loss_dist
                loss = loss / self.config['gradient_accumulation_steps']

                loss.backward()

                if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            epoch_loss += loss.item() * self.config['gradient_accumulation_steps']
            seg_loss_sum += loss_seg.item()
            bdy_loss_sum += loss_bdy.item()
            dist_loss_sum += loss_dist.item()

            pbar.set_postfix_str(f"L_seg: {loss_seg.item():.3f} | L_bdy: {loss_bdy.item():.3f} | L_dist: {loss_dist.item():.3f} | L_total: {loss.item() * self.config['gradient_accumulation_steps']:.3f}")

        avg_loss = epoch_loss / len(self.train_loader)
        avg_seg_loss = seg_loss_sum / len(self.train_loader)
        avg_bdy_loss = bdy_loss_sum / len(self.train_loader)
        avg_dist_loss = dist_loss_sum / len(self.train_loader)

        self.train_losses.append(avg_loss)
        self.boundary_losses.append(avg_bdy_loss)
        self.distance_losses.append(avg_dist_loss)

        return avg_loss, avg_seg_loss, avg_bdy_loss, avg_dist_loss

    def validate(self):
        self.model.eval()
        total_iou = 0

        with torch.no_grad():
            for imgs, masks, _ in tqdm(self.val_loader, desc='Validating'):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                seg_logits, _, _ = self.model(imgs)

                pred = (torch.sigmoid(seg_logits) > 0.5).long()
                if pred.shape[1] == 1:
                    pred = pred.squeeze(1)

                intersection = (pred & masks.long()).float().sum((1, 2))
                union = (pred | masks.long()).float().sum((1, 2))
                iou = (intersection + 1e-6) / (union + 1e-6)
                total_iou += iou.mean().item()

        return total_iou / len(self.val_loader)

    def save_model(self, filename):
        model_path = self.output_dir / filename
        torch.save({
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': self.best_iou,
            'train_losses': self.train_losses,
            'val_ious': self.val_ious,
            'boundary_losses': self.boundary_losses,
            'distance_losses': self.distance_losses,
            'config': self.config
        }, model_path)
        print(f"模型已保存到: {model_path}")

    def plot_training_curves(self):
        plt.figure(figsize=(24, 5))

        plt.subplot(1, 4, 1)
        plt.plot(self.train_losses, label='Total Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 2)
        plt.plot(self.val_ious, label='Validation IoU', linewidth=2, color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('Validation IoU')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 3)
        plt.plot(self.boundary_losses, label='Boundary Loss', linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Boundary Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 4)
        plt.plot(self.distance_losses, label='Distance Loss', linewidth=2, color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Distance Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        print("开始训练V7终极优化版...")

        self.patience_counter = 0
        self.early_stopping_patience = self.config.get('early_stopping_patience', 8)

        for epoch in range(1, self.config['epochs'] + 1):
            self.current_epoch = epoch
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"{'=' * 70}")

            train_loss, seg_loss, bdy_loss, dist_loss = self.train_one_epoch(epoch)
            val_iou = self.validate()
            self.val_ious.append(val_iou)

            if epoch <= self.warmup_epochs:
                warmup_lr = self.config['lr'] * (epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"Warmup阶段，学习率: {warmup_lr:.6f}")
            else:
                self.scheduler.step(val_iou)

            print(f"训练损失: {train_loss:.4f} (分割: {seg_loss:.4f}, 边界: {bdy_loss:.4f}, 距离: {dist_loss:.4f})")
            print(f"验证IoU: {val_iou:.4f}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self.patience_counter = 0
                self.save_model('best_model.pth')
                print(f"✅ 新的最佳模型! IoU: {val_iou:.4f}")
            else:
                self.patience_counter += 1
                print(f"⏳ 未提升，耐心计数: {self.patience_counter}/{self.early_stopping_patience}")

                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n🛑 早停触发！连续{self.early_stopping_patience}轮未提升")
                    break

            if epoch % self.config.get('save_interval', 5) == 0:
                self.save_model(f'checkpoint_epoch_{epoch}.pth')

            if epoch % self.config.get('plot_interval', 2) == 0:
                self.plot_training_curves()

        self.save_model('final_model.pth')

        print(f"\n{'=' * 70}")
        print(f"训练完成!")
        print(f"最佳验证IoU: {self.best_iou:.4f}")
        print(f"{'=' * 70}")


if __name__ == '__main__':
    set_seed(42)

    config = {
        'data_root': 'FarmSeg-VL',
        'output_dir': './results_v7_ultimate',
        'encoder_name': 'resnet34',
        'encoder_weights': 'imagenet',
        'use_cbam': True,
        'img_size': 512,
        'batch_size': 16,
        'epochs': 50,
        'lr': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'boundary_weight': 0.5,
        'num_workers': 4,
        'save_interval': 5,
        'plot_interval': 2,
        'use_amp': True,
        'gradient_accumulation_steps': 1,
        'warmup_epochs': 2,
        'early_stopping_patience': 8,
        'dropout_rate': 0.2,
        'debug_mode': False,
        'resume_path': None,
    }

    trainer = Trainer(config)
    trainer.run()
