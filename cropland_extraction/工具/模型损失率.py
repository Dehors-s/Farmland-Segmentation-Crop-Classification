import torch
ckpt = torch.load(r"D:\Work space\DeepLearning\farm\cropland_extraction\results\v8v3\best_model.pth", map_location="cpu")
print("Epoch | Total Loss | Seg Loss | Boundary Loss | Distance Loss")
for i, (tl, bl, dl) in enumerate(zip(ckpt["train_losses"], 
                                       ckpt["boundary_losses"],
                                       ckpt["distance_losses"])):
    seg = tl - bl * 0.6 - dl * 0.25  # 反推 seg loss
    print(f"{i+1:3d}  | {tl:.4f}     | {seg:.4f}   | {bl:.4f}       | {dl:.4f}")