"""V8 完整训练曲线分析"""
import torch, json

ckpt = torch.load(r"D:\Work space\DeepLearning\farm\cropland_extraction\results\v8v3\best_model.pth", map_location="cpu")

val_ious = ckpt["val_ious"]
val_bdy = ckpt["val_bdy_ious"]
train_losses = ckpt["train_losses"]
bdy_losses = ckpt["boundary_losses"]
dist_losses = ckpt["distance_losses"]

print("=== V8 Full Curve ===\n")

# Per-epoch detailed table (every 5 epochs)
print("Epoch | Train Loss | Seg Loss(est) | Bdy Loss | Dist Loss | Val IoU | Bdy IoU | IoU Delta")
print("-" * 95)
for i in range(0, len(val_ious), max(1, len(val_ious)//10)):
    tl = train_losses[i]
    bl = bdy_losses[i]
    dl = dist_losses[i]
    # boundary_weight dynamic: >20 → 0.25, clamped to 0.3
    bw = 0.3  # min_boundary_weight
    seg_l = tl - bw * bl - 0.25 * dl
    iou = val_ious[i]
    biou = val_bdy[i] if i < len(val_bdy) else 0
    delta = iou - val_ious[max(0,i-1)] if i > 0 else 0
    print(f" {i+1:3d}  | {tl:.4f}     | {seg_l:.4f}       | {bl:.4f}   | {dl:.4f}    | {iou:.4f}  | {biou:.4f}  | {delta:+.4f}")

# IoU growth phases
print(f"\n=== Growth Phases ===")
phases = [
    (1, 8, "Rapid initial learning"),
    (9, 22, "Crossing 0.70"),
    (23, 43, "Plateau around 0.71-0.72"),
    (44, 71, "Breakthrough to 0.75"),
    (72, 99, "Steady climb to 0.77"),
    (100, 141, "Fine-tuning to 0.78"),
]
for s, e, desc in phases:
    si, ei = min(s-1, len(val_ious)-1), min(e-1, len(val_ious)-1)
    total_gain = val_ious[ei] - val_ious[si]
    rate = total_gain / (e - s + 1) if e > s else 0
    print(f"  epochs {s:3d}-{e:3d} ({desc}): +{total_gain:.4f} ({rate*100:.3f}%/epoch)")

# Convergence analysis
print(f"\n=== Convergence ===")
print(f"Epochs to 0.70: next(i for i,v in enumerate(val_ious) if v>=0.70)")  # will fail, just compute below
for target in [0.70, 0.72, 0.74, 0.76, 0.775]:
    for i, v in enumerate(val_ious):
        if v >= target:
            print(f"  Reached {target:.2f} at epoch {i+1}")
            break
    else:
        print(f"  Never reached {target:.2f}")

print(f"  Max IoU: {max(val_ious):.4f} at epoch {val_ious.index(max(val_ious))+1}")
print(f"  Final IoU: {val_ious[-1]:.4f}")
print(f"  Still improving at end: {'Yes' if max(val_ious) == val_ious[-1] else 'No'}")

# Loss convergence
print(f"\n=== Loss Convergence ===")
print(f"Initial Bdy Loss: {bdy_losses[0]:.4f}")
print(f"Final Bdy Loss:   {bdy_losses[-1]:.4f}")
bdy_improved = (bdy_losses[0] - bdy_losses[-1]) / bdy_losses[0] * 100
print(f"Bdy improvement:  {bdy_improved:.1f}%")
print(f"Initial Dist Loss: {dist_losses[0]:.4f}")
print(f"Final Dist Loss:   {dist_losses[-1]:.4f}")
