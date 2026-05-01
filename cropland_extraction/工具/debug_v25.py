"""Debug V2.5 vs V2 on a single image."""
import sys, os
sys.path.insert(0, r"D:\Work space\DeepLearning\farm\cropland_extraction\模型推理")

image_path = r"D:\Work space\DeepLearning\farm\data\大图测试\8010_1.tif"
model_path = r"D:\Work space\DeepLearning\farm\cropland_extraction\results\v8v3\best_model.pth"

# === Run V2-style inference (using V2's predict_tiled) ===
# Import V2's engine
sys.modules.pop("MultiTaskUNet", None)
sys.modules.pop("CBAMUNet", None)

# Re-load V2's code fresh
v2_file = r"D:\Work space\DeepLearning\farm\cropland_extraction\模型推理\u-net矢量化V2.py"
with open(v2_file, encoding="utf-8") as f:
    v2_code = f.read()
exec(v2_code, {"__name__": "__v2__", "__file__": v2_file})
V2Engine = [v for k,v in dict(locals()).items() if "InferenceEngine" in str(k)][0]
V2Model = [v for k,v in dict(locals()).items() if "MultiTaskUNet" in str(k)][0]

# Actually this is getting too complicated. Let me just test the V2.5 directly.
