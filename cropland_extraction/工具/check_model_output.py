"""快速检查 V2.5 模型输出是否正常。"""
import sys, os, importlib.util

v25_path = r"D:\Work space\DeepLearning\farm\cropland_extraction\模型推理\u-net矢量化V2.5.py"

spec = importlib.util.spec_from_file_location("v25_module", v25_path)
v25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v25)

import numpy as np

model_path = r"D:\Work space\DeepLearning\farm\cropland_extraction\results\v8v3\best_model.pth"
img_path = r"D:\Work space\DeepLearning\farm\data\大图测试\8010_1.tif"

for mode in ["percentile", "legacy"]:
    print(f"\n{'=' * 60}")
    print(f"V2.5 predict_tiled ({mode})")
    print(f"{'=' * 60}")
    engine = v25.InferenceEngine(
        model_path=model_path,
        encoder_name="resnet50", in_channels=4,
        tile_size=512, tile_overlap=128,
        norm_mode=mode,
    )
    bin_mask, bin_boundary, seg_map, bdy_map, dist_map, img_full, crs, transform = \
        engine.predict_tiled(img_path, seg_threshold=0.5, boundary_threshold=0.1)
    print(f"  seg_map: min={seg_map.min():.6f} max={seg_map.max():.6f} mean={seg_map.mean():.6f}")
    print(f"  bin_mask > 0: {(bin_mask > 0).sum()} / {bin_mask.size} px")
    print(f"  bin_boundary > 0: {(bin_boundary > 0).sum()} / {bin_boundary.size} px")
