# u-net--baseline-HRNet.py
# 论文基线: HRNet-W48 (高分辨率特征保留 + 轻量FPN解码器)
# RTX 4090 优化版 (batch=24, workers=8, epoch=150)
# 使用 timm 的 HRNet 骨干网络

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TIMM_HF_HUB_DISABLE"] = "1"  # 跳过 huggingface hub 检查

import warnings
import random
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


# ============================================================================
# 模型: HRNet + 轻量FPN解码器
# ============================================================================

class HRNetSegmentation(nn.Module):
    """
    HRNet-W{width} + 多尺度特征融合解码器

    使用 timm 的 HRNet 作为骨干 (features_only=True)，
    从4个分辨率阶段提取特征，用FPN风格融合后上采样到原图尺寸。
    """
    def __init__(self, in_channels=4, num_classes=1, width=48):
        super().__init__()
        import timm

        model_name = f"hrnet_w{width}"
        # HRNet 预训练权重通过 HF Hub 加载
        # 需要 huggingface_hub: pip install huggingface_hub
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=[0, 1, 2, 3, 4],
        )

        # 适配输入通道数
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            if in_channels > 3:
                m = old_conv.weight.mean(dim=1, keepdim=True)
                for ch in range(3, in_channels):
                    new_conv.weight[:, ch:ch+1] = m
        self.backbone.conv1 = new_conv

        # 用 forward pass 确定各阶段输出通道数
        # 注意: conv1 已被改为 in_channels 通道, dummy 需匹配
        self.backbone.eval()
        dummy = torch.zeros(1, in_channels, 224, 224)
        with torch.no_grad():
            feats = self.backbone(dummy)
        feat_channels = [f.shape[1] for f in feats]
        self.backbone.train()
        print(f"  HRNet-W{width} 特征通道: {feat_channels}")

        # 轻量 FPN 解码器: 1x1 lateral conv 将各层映射到统一通道
        self.lateral_convs = nn.ModuleList()
        for ch in feat_channels:
            self.lateral_convs.append(nn.Conv2d(ch, 128, kernel_size=1))

        # 融合卷积
        total_channels = 128 * len(feat_channels)
        self.fpn_conv = nn.Sequential(
            nn.Conv2d(total_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        ins = x.shape[2:]
        feats = self.backbone(x)  # 多尺度特征列表

        # 1x1 lateral conv + 上采样到同一尺寸
        target_size = feats[0].shape[2:]
        upsampled = []
        for feat, conv in zip(feats, self.lateral_convs):
            f = conv(feat)
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            upsampled.append(f)

        # 拼接 + 融合
        fused = self.fpn_conv(torch.cat(upsampled, dim=1))

        # 上采样到原图 + 预测
        out = F.interpolate(fused, size=ins, mode="bilinear", align_corners=False)
        out = self.seg_head(out)
        return out


# ============================================================================
# 损失函数
# ============================================================================

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dw = dice_weight
        self.bw = bce_weight

    def forward(self, inputs, targets, smooth=1):
        probs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        inter = (probs * targets).sum()
        dice = 1 - (2.0 * inter + smooth) / (probs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy_with_logits(inputs.view(-1), targets, reduction="mean")
        return self.dw * dice + self.bw * bce


# ============================================================================
# 数据集 (与 V8 完全相同)
# ============================================================================

class FarmlandDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, img_size=512,
                 in_channels=4, norm_mode="legacy", debug=False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.in_channels = in_channels
        self.norm_mode = norm_mode
        self.img_dir = self.root_dir / split / "img"
        self.mask_dir = self.root_dir / split / "lbl"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.img_dir}")
        self.image_files = sorted(
            [p for p in self.img_dir.iterdir()
             if p.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]]
        )
        print(f"找到 {len(self.image_files)} 个{split}图像")

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def _normalize_multispectral(img):
        img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mx = float(np.max(img)) if img.size else 1.0
        if mx > 2000:
            img /= 10000.0
        elif mx > 1.5:
            img /= 255.0
        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def _normalize_percentile(img, lo_pct=2, hi_pct=98):
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

    def _read_image(self, img_path):
        sfx = img_path.suffix.lower()
        if sfx in [".tif", ".tiff"]:
            with rasterio.open(img_path) as ds:
                arr = ds.read().astype(np.float32)
            arr = np.transpose(arr, (1, 2, 0))
        else:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"无法读取: {img_path}")
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        if arr.shape[2] < self.in_channels:
            pad = self.in_channels - arr.shape[2]
            arr = np.concatenate([arr, np.zeros((*arr.shape[:2], pad), dtype=arr.dtype)], axis=2)
        elif arr.shape[2] > self.in_channels:
            arr = arr[:, :, :self.in_channels]
        if self.norm_mode == "percentile":
            return self._normalize_percentile(arr)
        return self._normalize_multispectral(arr)

    def _read_mask(self, img_path):
        mp = self.mask_dir / f"{img_path.stem}.png"
        if not mp.exists():
            mp2 = self.mask_dir / f"{img_path.stem}.tif"
            mp = mp2 if mp2.exists() else mp
        if not mp.exists():
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        if mp.suffix.lower() in [".tif", ".tiff"]:
            with rasterio.open(mp) as ds:
                mask = ds.read(1)
        else:
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        return (mask > 0).astype(np.uint8)

    def __getitem__(self, idx):
        ip = self.image_files[idx]
        img = self._read_image(ip)
        mask = self._read_mask(ip)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        return img, mask.float()


def get_transforms(img_size, phase="train"):
    if phase == "train":
        aug = [
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ChannelShuffle(p=0.3), A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Resize(img_size, img_size), ToTensorV2(),
        ]
    else:
        aug = [A.Resize(img_size, img_size), ToTensorV2()]
    return A.Compose(aug)


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HRNetSegmentation(
            in_channels=config["in_channels"], num_classes=1,
            width=config["width"],
        ).to(self.device)

        self.criterion = DiceBCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
        )

        self.warmup_epochs = config.get("warmup_epochs", 3)
        ws = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=self.warmup_epochs)
        ms = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True, min_lr=config["min_lr"],
        )
        self.warmup_scheduler = ws
        self.main_scheduler = ms
        self.scheduler = ms
        self._warmup_done = False

        nm = config.get("norm_mode", "legacy")
        train_ds = FarmlandDataset(config["data_root"], "train",
            get_transforms(config["img_size"], "train"), config["img_size"],
            in_channels=config["in_channels"], norm_mode=nm)
        val_ds = FarmlandDataset(config["data_root"], "val",
            get_transforms(config["img_size"], "val"), config["img_size"],
            in_channels=config["in_channels"], norm_mode=nm)

        nw = config.get("num_workers", 0)
        self.train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
            shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
            shuffle=False, num_workers=nw, pin_memory=True)

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = GradScaler() if config["use_amp"] else None

        # 断点续训
        self.start_epoch = 0
        self.best_iou = 0.0
        self.train_losses = []
        self.val_ious = []
        resume_path = config.get("resume_path")
        if resume_path and Path(resume_path).exists():
            self._load_checkpoint(resume_path)

        print(f"\n{'='*60}")
        print(f"HRNet-W{config['width']} + FPN Decoder")
        print(f"{'='*60}")
        print(f"设备: {self.device}")
        print(f"训练样本: {len(train_ds)}")
        print(f"验证样本: {len(val_ds)}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")

    def _load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if not self.config.get("reset_optimizer", False):
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0)
        self.best_iou = ckpt.get("best_iou", 0.0)
        self.train_losses = ckpt.get("train_losses", [])
        self.val_ious = ckpt.get("val_ious", [])
        print(f">>> 从 {ckpt_path} 恢复训练 (epoch {self.start_epoch}, best_iou={self.best_iou:.4f})")

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).float().unsqueeze(1)
            if self.config["use_amp"]:
                with autocast():
                    logits = self.model(imgs)
                    loss = self.criterion(logits, masks) / self.config.get("gradient_accumulation_steps", 1)
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.config.get("gradient_accumulation_steps", 1) == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(imgs)
                loss = self.criterion(logits, masks) / self.config.get("gradient_accumulation_steps", 1)
                loss.backward()
                if (batch_idx + 1) % self.config.get("gradient_accumulation_steps", 1) == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            epoch_loss += loss.item() * self.config.get("gradient_accumulation_steps", 1)
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        avg = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg)
        return avg

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_iou = 0.0
        for imgs, masks in tqdm(self.val_loader, desc="Validating"):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            logits = self.model(imgs)
            pred = (torch.sigmoid(logits) > 0.5).long()
            if pred.shape[1] == 1:
                pred = pred.squeeze(1)
            masks_f = masks.squeeze(1) if masks.shape[1] == 1 else masks
            inter = (pred & masks_f.long()).float().sum((1, 2))
            union = (pred | masks_f.long()).float().sum((1, 2))
            iou = (inter + 1e-6) / (union + 1e-6)
            total_iou += iou.mean().item()
        avg = total_iou / len(self.val_loader)
        self.val_ious.append(avg)
        return avg

    def save_model(self, fn, extra=None):
        path = self.output_dir / fn
        st = {"epoch": self.start_epoch + len(self.train_losses),
              "model_state_dict": self.model.state_dict(),
              "optimizer_state_dict": self.optimizer.state_dict(),
              "scheduler_state_dict": self.scheduler.state_dict(),
              "best_iou": self.best_iou, "train_losses": self.train_losses,
              "val_ious": self.val_ious, "config": self.config}
        if extra:
            st.update(extra)
        torch.save(st, path)
        print(f">>> 模型已保存: {path}")

    def plot_curves(self):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ep = list(range(1, len(self.train_losses) + 1))
        ax[0].plot(ep, self.train_losses, "b-", label="Train Loss")
        ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
        ax[0].set_title("Training Loss"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
        ax[1].plot(ep, self.val_ious, "r-", label="Val IoU")
        ax[1].axhline(y=self.best_iou, color="g", ls="--", alpha=0.5, label=f"Best IoU={self.best_iou:.4f}")
        ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("IoU")
        ax[1].set_title("Validation IoU"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
        plt.suptitle(f"HRNet-W{self.config['width']} + FPN", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f">>> 训练曲线已保存")

    def run(self):
        epochs = self.config["epochs"]
        patience = self.config.get("early_stopping_patience", 12)
        pc = 0
        for ep in range(self.start_epoch + 1, epochs + 1):
            if not self._warmup_done:
                self.warmup_scheduler.step()
                if ep >= self.warmup_epochs:
                    self._warmup_done = True
                    self.scheduler = self.main_scheduler
            train_loss = self.train_one_epoch(ep)
            val_iou = self.validate()
            if self._warmup_done:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_iou)
                else:
                    self.scheduler.step()
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self.save_model("best_model.pth")
                pc = 0
            else:
                pc += 1
            clr = self.optimizer.param_groups[0]["lr"]
            print(f"  Epoch {ep:3d}/{epochs} | Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | "
                  f"Best: {self.best_iou:.4f} | LR: {clr:.2e} | {pc}/{patience}")
            if ep % self.config.get("save_interval", 10) == 0:
                self.save_model(f"checkpoint_epoch_{ep}.pth")
            if pc >= patience:
                print(f"\n>>> 早期停止! {patience} 轮无提升")
                break
        self.save_model("final_model.pth")
        self.plot_curves()
        print(f"\n{'='*60}")
        print(f"训练完成! HRNet-W{self.config['width']} + FPN")
        print(f"最佳验证 IoU: {self.best_iou:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HRNet-W48 基线模型训练")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=48, choices=[18, 32, 40, 48, 64],
                        help="HRNet 宽度 (默认: 48)")
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.add_argument("--norm_mode", default="legacy", choices=["legacy", "percentile"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="断点续训 checkpoint 路径")
    parser.add_argument("--reset_optimizer", action="store_true", help="仅加载模型权重，重置优化器")
    args = parser.parse_args()

    set_seed(args.seed)
    config = {k: v for k, v in vars(args).items()}
    config["max_grad_norm"] = 1.0
    config["resume_path"] = config.pop("resume", None)
    Trainer(config).run()
