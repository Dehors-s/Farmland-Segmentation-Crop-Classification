"""Extract and compare metrics from V8, V8.5, V9 best_model.pth files"""
import torch
import os

models = {
    "V8":   r"D:\Work space\DeepLearning\farm\cropland_extraction\results\v8\best_model.pth",
    "V8.5": r"D:\Work space\DeepLearning\farm\cropland_extraction\results\v8.5\best_model.pth",
    "V9":   r"D:\Work space\DeepLearning\farm\cropland_extraction\results\v9\best_model (1).pth",
}

all_data = {}

for name, path in models.items():
    if not os.path.exists(path):
        print(f"\n{'='*60}")
        print(f"{name}: FILE NOT FOUND at {path}")
        continue
    
    ckpt = torch.load(path, map_location="cpu")
    all_data[name] = ckpt
    
    print(f"\n{'='*60}")
    print(f" {name} - {os.path.basename(path)}")
    print(f" Size: {os.path.getsize(path)/1024/1024:.1f} MB")
    print(f" Modified: {os.path.getmtime(path)}")
    print(f"{'='*60}")
    
    # Keys
    print(f"\n  Keys: {[k for k in ckpt.keys() if k != 'model_state_dict']}")
    
    # Config
    cfg = ckpt.get("config", {})
    print(f"\n  Config (key params):")
    for k in ["encoder_name","in_channels","img_size","batch_size","lr","epochs",
              "loss_type","weight_decay","dropout_rate","warmup_epochs",
              "early_stopping_patience","norm_mode","scheduler_type",
              "min_boundary_weight","boundary_weight","swa_start","resume_path",
              "reset_optimizer","data_root","output_dir"]:
        if k in cfg:
            print(f"    {k}: {cfg[k]}")
    
    # Metrics
    bi = ckpt.get("best_iou", "N/A")
    bbi = ckpt.get("best_bdy_iou", "N/A")
    ep = ckpt.get("epoch", "N/A")
    tl = len(ckpt.get("train_losses", []))
    vi = len(ckpt.get("val_ious", []))
    
    print(f"\n  Metrics:")
    print(f"    Total epochs trained: {tl}")
    print(f"    Val IoU entries: {vi}")
    print(f"    Best IoU: {bi:.4f}" if isinstance(bi, float) else f"    Best IoU: {bi}")
    print(f"    Best Bdy IoU: {bbi:.4f}" if isinstance(bbi, float) else f"    Best Bdy IoU: {bbi}")
    print(f"    Epoch saved: {ep}" if isinstance(ep, int) else f"    Epoch saved: {ep}")
    
    val_ious = ckpt.get("val_ious", [])
    val_bdy = ckpt.get("val_bdy_ious", [])
    train_losses = ckpt.get("train_losses", [])
    boundary_losses = ckpt.get("boundary_losses", [])
    distance_losses = ckpt.get("distance_losses", [])
    
    if val_ious:
        max_iou = max(val_ious)
        max_ep = val_ious.index(max_iou) + 1
        print(f"    Max IoU: {max_iou:.4f} at epoch {max_ep}")
        print(f"    First IoU: {val_ious[0]:.4f}")
        print(f"    Last IoU: {val_ious[-1]:.4f}")
    
    if val_bdy:
        max_bdy = max(val_bdy)
        max_bdy_ep = val_bdy.index(max_bdy) + 1
        print(f"    Max Bdy IoU: {max_bdy:.4f} at epoch {max_bdy_ep}")
    
    if train_losses:
        print(f"    Train Loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
        print(f"    Boundary Loss: {boundary_losses[0]:.4f} -> {boundary_losses[-1]:.4f}" if boundary_losses else "")
        print(f"    Distance Loss: {distance_losses[0]:.4f} -> {distance_losses[-1]:.4f}" if distance_losses else "")
    
    # Milestones
    if val_ious:
        print(f"\n  Milestones:")
        for target in [0.60, 0.65, 0.70, 0.72, 0.74, 0.75, 0.76, 0.77]:
            found = False
            for i, v in enumerate(val_ious):
                if v >= target:
                    print(f"    IoU >= {target:.2f}: epoch {i+1}")
                    found = True
                    break
            if not found:
                print(f"    IoU >= {target:.2f}: NEVER REACHED")
    
    # IoU progression summary
    if val_ious:
        n = len(val_ious)
        step = max(1, n // 12)
        print(f"\n  IoU Progression (every ~{100//max(1,n//12):d}%):")
        print(f"    {'Epoch':>5} | {'IoU':>7} | {'Bdy IoU':>8} | {'Train Loss':>11}")
        print(f"    {'-'*40}")
        for i in range(0, n, step):
            bdy = val_bdy[i] if i < len(val_bdy) else 0
            tl_v = train_losses[i] if i < len(train_losses) else 0
            mark = " << MAX" if val_ious[i] == max(val_ious) else ""
            print(f"    {i+1:4d}   | {val_ious[i]:.4f}  | {bdy:.4f}    | {tl_v:.4f}{mark}")
        if (n-1) % step != 0:
            i = n-1
            bdy = val_bdy[i] if i < len(val_bdy) else 0
            tl_v = train_losses[i] if i < len(train_losses) else 0
            mark = " << MAX" if val_ious[i] == max(val_ious) else ""
            print(f"    {i+1:4d}   | {val_ious[i]:.4f}  | {bdy:.4f}    | {tl_v:.4f}{mark}")
    
    # Final losses
    if boundary_losses:
        print(f"\n  Boundary Loss: {boundary_losses[0]:.4f} -> min={min(boundary_losses):.4f} -> last={boundary_losses[-1]:.4f}")
        print(f"  Seg Loss (est): {train_losses[-1] - 0.3*val_bdy[-1] - 0.25*distance_losses[-1]:.4f}" if val_bdy and distance_losses else "")


# ===== COMPARISON TABLE =====
print(f"\n\n{'='*70}")
print(f" COMPARISON TABLE - All Models")
print(f"{'='*70}")
print(f"{'Metric':<25} {'V8':<15} {'V8.5':<15} {'V9':<15}")
print(f"{'-'*70}")

metrics_to_compare = ["best_iou", "best_bdy_iou"]
for m in metrics_to_compare:
    row = f"{m:<25} "
    for name in ["V8", "V8.5", "V9"]:
        if name in all_data:
            v = all_data[name].get(m, "N/A")
            row += f"{v:<15.4f} " if isinstance(v, float) else f"{str(v):<15} "
        else:
            row += f"{'N/A':<15} "
    print(row)

# Epoch counts
row = f"{'epochs trained':<25} "
for name in ["V8", "V8.5", "V9"]:
    if name in all_data:
        v = len(all_data[name].get("train_losses", []))
        row += f"{v:<15} "
    else:
        row += f"{'N/A':<15} "
print(row)

# IoU milestones
for target in [0.70, 0.72, 0.74, 0.75, 0.76, 0.77]:
    row = f"{'epoch to IoU≥'+str(target):<25} "
    for name in ["V8", "V8.5", "V9"]:
        if name in all_data:
            val_ious = all_data[name].get("val_ious", [])
            found = False
            for i, v in enumerate(val_ious):
                if v >= target:
                    row += f"{i+1:<15} "
                    found = True
                    break
            if not found:
                row += f"{'never':<15} "
        else:
            row += f"{'N/A':<15} "
    print(row)

# Boundary loss floor
row = f"{'boundary loss min':<25} "
for name in ["V8", "V8.5", "V9"]:
    if name in all_data:
        bl = all_data[name].get("boundary_losses", [])
        v = min(bl) if bl else "N/A"
        row += f"{v:<15.4f} " if isinstance(v, float) else f"{str(v):<15} "
    else:
        row += f"{'N/A':<15} "
print(row)

print(f"{'='*70}")

# Config comparison
print(f"\n{'='*70}")
print(f" CONFIG COMPARISON")
print(f"{'='*70}")
config_keys = ["encoder_name","in_channels","batch_size","lr","epochs","loss_type",
               "weight_decay","dropout_rate","warmup_epochs","scheduler_type",
               "early_stopping_patience","norm_mode","min_boundary_weight"]
for k in config_keys:
    row = f"{k:<25} "
    for name in ["V8", "V8.5", "V9"]:
        if name in all_data:
            v = all_data[name].get("config", {}).get(k, "?")
            if isinstance(v, float):
                row += f"{v:<15.6f} "
            else:
                row += f"{str(v):<15} "
        else:
            row += f"{'N/A':<15} "
    print(row)
print(f"{'='*70}")
