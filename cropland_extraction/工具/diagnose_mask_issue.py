#!/usr/bin/env python3
"""
诊断掩膜全黑问题的脚本。
以 shrink_distance=0（不收缩）运行，观察掩膜是否正常。
"""
import sys
sys.path.insert(0, r"D:\Work space\DeepLearning\farm\过程文件")

from prepare_unet_patches import Config, process_dataset

# 使用与原配置相同的参数，但 shrink_distance=0（禁用收缩）
config = Config(
    tif_path=r"D:\Work space\DeepLearning\farm\data\遥感图像\HE017008_2024.tif",
    shp_path=r"D:\Work space\DeepLearning\farm\data\edge\HE017008_2024-3\HE017008_2024-3.shp",
    output_root=r"D:\Work space\DeepLearning\farm\dataset_diagnostic",
    class_field="class",
    positive_class_value=1,
    crop_size=512,
    stride=256,
    drop_background_only=True,
    background_keep_ratio=0.2,
    shrink_distance=0.0,  # ← 改为 0，禁用边界收缩
    allow_missing_tif_crs=True,
    allow_missing_shp_crs=True,
    keep_all_bands=True,
    selected_bands=None,
    image_driver="GTiff",
    rgb_preview_bands=[4, 3, 2],
    audit_every_n_saved=1000,
    audit_sample_count=10,
)

print("=" * 70)
print("掩膜诊断运行：shrink_distance=0（不收缩边界）")
print("=" * 70)
process_dataset(config)
print("=" * 70)
print("诊断完成！检查上方的统计信息。")
print("=" * 70)
