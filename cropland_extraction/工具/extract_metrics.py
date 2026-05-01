"""提取 V8 和 V9 训练指标对比"""
import torch
import json

ckpt_v8 = torch.load(r"D:\Work space\DeepLearning\farm\cropland_extraction\results\v8v3\best_model.pth", map_location="cpu")

# V8 数据
print("=" * 60)
print("V8 best_model.pth")
print("=" * 60)
cfg = ckpt_v8.get("config", {})
print(f"Encoder: {cfg.get('encoder_name', '?')} | Channels: {cfg.get('in_channels', '?')}")
print(f"Batch: {cfg.get('batch_size', '?')} | LR: {cfg.get('lr', '?')}")
print(f"Loss: {cfg.get('loss_type', '?')} | Image size: {cfg.get('img_size', '?')}")
print()

print(f"Best IoU: {ckpt_v8['best_iou']:.4f}")
print(f"Best Boundary IoU: {ckpt_v8.get('best_bdy_iou', 'N/A')}")
print(f"Epoch: {ckpt_v8.get('epoch', 'N/A')}")
print(f"Total train epochs: {len(ckpt_v8['train_losses'])}")
print(f"Val IoU count: {len(ckpt_v8.get('val_ious', []))}")
print()

val_ious = ckpt_v8.get("val_ious", [])
val_bdy = ckpt_v8.get("val_bdy_ious", [])
print(f"Max IoU: {max(val_ious):.4f}" if val_ious else "No IoU data")
if val_bdy:
    print(f"Max Boundary IoU: {max(val_bdy):.4f}")

# IoU curve
if len(val_ious) > 0:
    step = max(1, len(val_ious) // 20)
    print("\nIoU progression:")
    for i in range(0, len(val_ious), step):
        print(f"  epoch {i+1:3d}: IoU={val_ious[i]:.4f}")
    print(f"  epoch {len(val_ious):3d}: IoU={val_ious[-1]:.4f}")

# Final epoch losses
tl = ckpt_v8["train_losses"][-1]
bl = ckpt_v8["boundary_losses"][-1]
dl = ckpt_v8["distance_losses"][-1]
print(f"\nFinal epoch train losses (total: {tl:.4f}):")
print(f"  Boundary Loss: {bl:.4f}")
print(f"  Distance Loss: {dl:.4f}")
print(f"  Seg Loss (est): {tl - 0.3 * bl - 0.25 * dl:.4f}")

# Check config for min_boundary_weight
print(f"\nConfig params:")
print(f"  boundary_weight: {cfg.get('boundary_weight', '?')}")
print(f"  min_boundary_weight: {cfg.get('min_boundary_weight', '?')}")
print(f"  dropout_rate: {cfg.get('dropout_rate', '?')}")
print(f"  weight_decay: {cfg.get('weight_decay', '?')}")
print(f"  img_size: {cfg.get('img_size', '?')}")

# V9 finetune data (from conversation)
print("\n" + "=" * 60)
print("V9_4090 finetune (from user output)")
print("=" * 60)
v9_epochs = [
    (2, 0.5853, 0.2578),
    (5, 0.7167, 0.3695),
    (48, 0.7059, 0.3573),
    (54, 0.7034, 0.3551),
]
print("epoch | IoU  | Boundary IoU")
for ep, iou, biou in v9_epochs:
    print(f"  {ep:3d} | {iou:.4f} | {biou:.4f}")
print(f"\nFinal V9 finetune: IoU=0.7240, Boundary IoU=0.3769")
print(f"Best threshold: 0.45 (IoU=0.7238)")

print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
print(f"V8 max IoU: 0.78 (141 epochs)")
print(f"V9 finetune max IoU: 0.7240 (54+28=82 epochs)")
print(f"Delta: {(0.7240 - 0.78) * 100:.1f}% IoU")
print(f"\nV8 Boundary Loss (final): {bl:.4f}")
print("V9 Boundary Loss (epoch 28): 0.5941")
print(f"Delta boundary loss: {(0.5941 - bl):.4f}")
