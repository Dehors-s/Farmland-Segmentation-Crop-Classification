# u-net--baseline-FieldNet.py
# Field-Net (U-Net + spatial attention + multi-task) PyTorch reproduction

import os
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.pop("OMP_NUM_THREADS", None)
os.environ.pop("OMP_THREAD_LIMIT", None)
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return F.relu(out, inplace=True)


class SpatialAttentionGate(nn.Module):
    def __init__(self, x_ch, g_ch):
        super().__init__()
        inter_ch = max(1, x_ch // 2)
        self.theta_x = nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)

    def forward(self, x_l, x_h):
        # x_l: low-level (encoder), x_h: high-level (decoder), same spatial size
        theta = self.theta_x(x_l)
        phi = self.phi_g(x_h)
        f = F.relu(theta + phi, inplace=True)
        psi = torch.sigmoid(self.psi(f))
        return x_l * psi


class FieldNet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # Encoder (论文原文 ~11.5M)
        self.enc1 = ResidualBlock(in_channels, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)
        self.enc4 = ResidualBlock(128, 256)
        self.enc5 = ResidualBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.att4 = SpatialAttentionGate(256, 512)
        self.att3 = SpatialAttentionGate(128, 256)
        self.att2 = SpatialAttentionGate(64, 128)
        self.att1 = SpatialAttentionGate(32, 64)

        self.dec4 = ResidualBlock(256 + 512, 256)
        self.dec3 = ResidualBlock(128 + 256, 128)
        self.dec2 = ResidualBlock(64 + 128, 64)
        self.dec1 = ResidualBlock(32 + 64, 32)

        # Heads (mask / boundary / distance)
        self.head_mask = nn.Conv2d(32, 1, kernel_size=1)
        self.head_boundary = nn.Conv2d(32, 1, kernel_size=1)
        self.head_distance = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # 512 -> 32ch
        e2 = self.enc2(self.pool(e1))  # 256 -> 64ch
        e3 = self.enc3(self.pool(e2))  # 128 -> 128ch
        e4 = self.enc4(self.pool(e3))  # 64  -> 256ch
        e5 = self.enc5(self.pool(e4))  # 32  -> 512ch

        # Decoder
        d = self.up(e5)                # 64
        a = self.att4(e4, d)
        d = self.dec4(torch.cat([d, a], dim=1))  # 64 -> 256ch

        d = self.up(d)                 # 128
        a = self.att3(e3, d)
        d = self.dec3(torch.cat([d, a], dim=1))  # 128 -> 128ch

        d = self.up(d)                 # 256
        a = self.att2(e2, d)
        d = self.dec2(torch.cat([d, a], dim=1))  # 256 -> 64ch

        d = self.up(d)                 # 512
        a = self.att1(e1, d)
        d = self.dec1(torch.cat([d, a], dim=1))  # 512 -> 32ch

        # Heads
        mask = self.head_mask(d)
        bdy = self.head_boundary(d)
        dist = self.head_distance(d)
        return mask, bdy, dist


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        inter = (probs * targets).sum()
        return 1 - (2 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class FieldNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.l1 = nn.L1Loss()

    def forward(self, mask_logits, bdy_logits, dist_pred, mask_gt, bdy_gt, dist_gt):
        loss_mask = self.dice(mask_logits, mask_gt)
        loss_bdy = self.dice(bdy_logits, bdy_gt)
        loss_dist = self.l1(dist_pred, dist_gt)
        return loss_mask + loss_bdy + loss_dist


class FarmlandDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, img_size=256,
                 in_channels=4, norm_mode="legacy"):
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
        # 预计算 bdy/dist 并缓存
        self.cache_dir = self.root_dir / split / "_bdy_dist_cache"
        if split == "train":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for ip in tqdm(self.image_files, desc=f"预计算 {split} bdy/dist"):
                cf = self.cache_dir / f"{ip.stem}.npz"
                if cf.exists():
                    continue
                mask = self._read_mask(ip)
                mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                bdy, dist = self._build_boundary_and_distance(mask)
                np.savez_compressed(cf, bdy=bdy, dist=dist)
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

    @staticmethod
    def _build_boundary_and_distance(mask, width=4):
        # distance to boundary inside foreground
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
        bdy = ((dist > 0) & (dist <= width)).astype(np.uint8)
        # normalize distance to [0,1]
        if dist.max() > 0:
            dist = dist / dist.max()
        return bdy, dist

    def __getitem__(self, idx):
        ip = self.image_files[idx]
        img = self._read_image(ip)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = self._read_mask(ip)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        # 加载预计算的 bdy/dist
        cf = self.cache_dir / f"{ip.stem}.npz"
        if cf.exists():
            c = np.load(cf)
            bdy, dist = c["bdy"], c["dist"]
        else:
            bdy, dist = self._build_boundary_and_distance(mask)
        if self.transform:
            aug = self.transform(image=img, mask=mask, bdy=bdy, dist=dist)
            img = aug["image"]
            mask = aug["mask"]
            bdy = aug["bdy"]
            dist = aug["dist"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        if not isinstance(bdy, torch.Tensor):
            bdy = torch.from_numpy(bdy).float().unsqueeze(0)
        if not isinstance(dist, torch.Tensor):
            dist = torch.from_numpy(dist).float().unsqueeze(0)
        return img, mask.float(), bdy, dist


def get_transforms(img_size, phase="train"):
    if phase == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ChannelShuffle(p=0.3), A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Resize(img_size, img_size), ToTensorV2(),
        ], additional_targets={"bdy": "mask", "dist": "mask"})
    return A.Compose([A.Resize(img_size, img_size), ToTensorV2()],
                     additional_targets={"bdy": "mask", "dist": "mask"})


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FieldNet(in_channels=config["in_channels"]).to(self.device)
        self.criterion = FieldNetLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=0.0)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True, min_lr=config["min_lr"],
        )

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

        self.start_epoch = 0
        self.best_iou = 0.0
        self.train_losses = []
        self.val_ious = []

        resume_path = config.get("resume_path")
        if resume_path and Path(resume_path).exists():
            self._load_checkpoint(resume_path)

        print(f"\n{'='*60}")
        print(f"Field-Net (U-Net + Spatial Attention + Multi-task)")
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
        gas = self.config.get("gradient_accumulation_steps", 1)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        for batch_idx, (imgs, masks, bdy, dist) in enumerate(pbar):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            bdy = bdy.to(self.device).float()
            dist = dist.to(self.device).float()
            step = batch_idx + 1
            if self.config["use_amp"]:
                with autocast():
                    m_pred, b_pred, d_pred = self.model(imgs)
                    loss = self.criterion(m_pred, b_pred, d_pred, masks, bdy, dist) / gas
                self.scaler.scale(loss).backward()
                if step % gas == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                m_pred, b_pred, d_pred = self.model(imgs)
                loss = self.criterion(m_pred, b_pred, d_pred, masks, bdy, dist) / gas
                loss.backward()
                if step % gas == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            epoch_loss += loss.item() * gas
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        avg = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg)
        return avg

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_iou = 0.0
        for imgs, masks, _, _ in tqdm(self.val_loader, desc="Validating"):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            m_pred, _, _ = self.model(imgs)
            pred = (torch.sigmoid(m_pred) > 0.5).long()
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

    def save_model(self, fn):
        path = self.output_dir / fn
        st = {"epoch": self.start_epoch + len(self.train_losses),
              "model_state_dict": self.model.state_dict(),
              "optimizer_state_dict": self.optimizer.state_dict(),
              "scheduler_state_dict": self.scheduler.state_dict(),
              "best_iou": self.best_iou, "train_losses": self.train_losses,
              "val_ious": self.val_ious, "config": self.config}
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
        plt.suptitle("Field-Net", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f">>> 训练曲线已保存")

    def run(self):
        epochs = self.config["epochs"]
        patience = self.config.get("early_stopping_patience", 12)
        pc = 0
        for ep in range(self.start_epoch + 1, epochs + 1):
            train_loss = self.train_one_epoch(ep)
            val_iou = self.validate()
            self.scheduler.step(val_iou)
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
        print(f"训练完成! Field-Net")
        print(f"最佳验证 IoU: {self.best_iou:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Field-Net 基线模型训练")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--num_workers", type=int, default=8)
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
    config["resume_path"] = config.pop("resume", None)
    Trainer(config).run()
