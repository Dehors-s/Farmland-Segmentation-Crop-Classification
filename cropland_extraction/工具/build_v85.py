"""Build V8.5 from V8_4090: add V9 augmentations while keeping V8 architecture."""
import re

src = r"D:\Work space\DeepLearning\farm\cropland_extraction\模型推理\u-net--CBAMV8_4090.py"
dst = r"D:\Work space\DeepLearning\farm\cropland_extraction\模型推理\u-net--CBAMV8.5.py"

with open(src, "r", encoding="utf-8") as f:
    code = f.read()

# === 1. Header ===
code = code.replace(
    "# u-net--CBAMV8_4090.py",
    "# u-net--CBAMV8.5.py"
)
code = code.replace(
    "# RTX 4090 优化版",
    "# V8.5: V8 proven architecture + V9 augmentations (percentile norm + RandomScale + threshold search)\n# RTX 4090 优化版"
)

# === 2. Add _normalize_percentile method ===
insert_method = '''
    @staticmethod
    def _normalize_percentile(img, lower_pct=2, upper_pct=98):
        """V8.5: Per-channel percentile normalization (from V9)."""
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
'''
# Insert after _normalize_multispectral method
marker = "return img\n\n"
# Find the _normalize_multispectral return
idx = code.find("return img\n\n    @staticmethod\n    def generate_distance_map")
if idx == -1:
    # Try alternative marker
    idx = code.find("return img\n\n    def generate_distance_map")
if idx == -1:
    # Another try
    idx = code.find("return img\n\n    @staticmethod\n    def generate_distance_map(mask):")
if idx > 0:
    code = code[:idx] + insert_method + code[idx:]
    print("1. Added _normalize_percentile method")
else:
    print("WARNING: Could not find insertion point for _normalize_percentile")

# === 3. Add norm_mode to FarmlandDataset.__init__ ===
code = code.replace(
    'def __init__(self, root_dir, split="train", transform=None, img_size=512, in_channels=4, debug=False):',
    'def __init__(self, root_dir, split="train", transform=None, img_size=512, in_channels=4, norm_mode="percentile", debug=False):'
)
code = code.replace(
    'self.debug = debug\n',
    'self.norm_mode = norm_mode\n        self.debug = debug\n'
)
print("2. Added norm_mode to FarmlandDataset")

# === 4. Update _read_image to use norm_mode ===
old_return = "return self._normalize_multispectral(arr)"
new_return = '        if self.norm_mode == "percentile":\n            return self._normalize_percentile(arr)\n        return self._normalize_multispectral(arr)'
code = code.replace(old_return, new_return)
print("3. Updated _read_image for norm_mode")

# === 5. Add RandomScale to train transforms ===
old_aug = "            A.HorizontalFlip(p=0.5),"
new_aug = "            A.RandomScale(scale_limit=(-0.5, 1.0), p=0.5),\n            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT),\n            A.RandomCrop(img_size, img_size),\n            A.HorizontalFlip(p=0.5),"
code = code.replace(old_aug, new_aug)
print("4. Added RandomScale to augmentation")

# === 6. Add validate_with_threshold and find_best_threshold before save_model ===
old_save_def = "    def save_model(self, filename):"
new_threshold_methods = '''    @torch.no_grad()
    def validate_with_threshold(self, threshold=0.5):
        """V8.5: Validate at custom threshold."""
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

    def find_best_threshold(self, thresholds=None):
        """V8.5: Sweep thresholds on validation set."""
        if thresholds is None:
            thresholds = np.arange(0.25, 0.81, 0.05)
        print(f"\\n{'=' * 60}")
        print("V8.5: Scanning optimal threshold...")
        print(f"{'=' * 60}")
        best_th = 0.5
        best_iou = 0.0
        for th in thresholds:
            iou, _ = self.validate_with_threshold(threshold=float(th))
            print(f"  Threshold {th:.2f} -> IoU {iou:.4f}")
            if iou > best_iou:
                best_iou = iou
                best_th = th
        print(f"\\nBest threshold: {best_th:.2f} (IoU: {best_iou:.4f})")
        print(f"{'=' * 60}")
        return float(best_th), best_iou

    def save_model(self, filename):'''
code = code.replace(old_save_def, new_threshold_methods)
print("5. Added threshold search methods")

# === 7. Add threshold search at end of run() ===
old_final = '''        self.save_model("final_model.pth")
        print("\\n" + "=" * 70)'''
new_final = '''        self.save_model("final_model.pth")

        # V8.5: Auto threshold search
        if self.config.get("auto_find_threshold", True):
            best_th, best_th_iou = self.find_best_threshold()

        print("\\n" + "=" * 70)'''
code = code.replace(old_final, new_final)
print("6. Added threshold search to run()")

# === 8. Add CLI args ===
old_cli_end = '    parser.add_argument("--resume", type=str, default=None, help="断点续训 checkpoint 路径")\n    parser.add_argument("--reset_optimizer", action="store_true", help="仅加载模型权重，重置优化器/调度器/超参（用于微调）")'
new_cli_end = '''    parser.add_argument("--norm_mode", type=str, default="percentile", choices=["percentile", "legacy"],
                        help="Normalization mode (V8.5: percentile default)")
    parser.add_argument("--auto_threshold", action="store_true", default=True,
                        help="Auto-search best threshold after training")
    parser.add_argument("--no_auto_threshold", action="store_false", dest="auto_threshold")
    parser.add_argument("--resume", type=str, default=None, help="断点续训 checkpoint 路径")
    parser.add_argument("--reset_optimizer", action="store_true", help="仅加载模型权重，重置优化器/调度器/超参（用于微调）")'''
code = code.replace(old_cli_end, new_cli_end)
print("7. Added CLI args")

# === 9. Add norm_mode and auto_find_threshold to config dict ===
old_config_end = '''        "resume_path": args.resume,
        "reset_optimizer": args.reset_optimizer,
        "max_grad_norm": 1.0,
    }'''
new_config_end = '''        "resume_path": args.resume,
        "reset_optimizer": args.reset_optimizer,
        "max_grad_norm": 1.0,
        "norm_mode": args.norm_mode,
        "auto_find_threshold": args.auto_threshold,
    }'''
code = code.replace(old_config_end, new_config_end)
print("8. Added config entries")

# === 10. Add norm_mode to dataset creation and print ===
code = code.replace(
    'get_transforms(config["img_size"], "train", use_color_aug=True),\n            config["img_size"],\n            in_channels=config["in_channels"],\n            debug=config.get("debug_mode", False),',
    'get_transforms(config["img_size"], "train", use_color_aug=True),\n            config["img_size"],\n            in_channels=config["in_channels"],\n            norm_mode=norm_mode,\n            debug=config.get("debug_mode", False),'
)
code = code.replace(
    'get_transforms(config["img_size"], "val", use_color_aug=False),\n            config["img_size"],\n            in_channels=config["in_channels"],\n            debug=False,',
    'get_transforms(config["img_size"], "val", use_color_aug=False),\n            config["img_size"],\n            in_channels=config["in_channels"],\n            norm_mode=norm_mode,\n            debug=False,'
)

# Add norm_mode extraction before datasets
code = code.replace(
    'train_ds = FarmlandDataset(',
    'norm_mode = config.get("norm_mode", "percentile")  # V8.5\n        train_ds = FarmlandDataset('
)
print("9. Added norm_mode to dataset creation")

# Add norm_mode print
code = code.replace(
    'print(f"输入通道数: {config[\'in_channels\']}")\n',
    'print(f"输入通道数: {config[\'in_channels\']}")\n        print(f"归一化模式: {norm_mode}")\n'
)
print("10. Added norm_mode print")

# === Write ===
with open(dst, "w", encoding="utf-8") as f:
    f.write(code)

print(f"\nV8.5 written: {dst}")
print(f"Lines: {len(code.splitlines())}")
