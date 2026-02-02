import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.windows
import numpy as np
import joblib
from tqdm import tqdm
from scipy import stats
from shapely.geometry import box
import warnings
import os

warnings.filterwarnings("ignore")

# ================= 1. 配置路径 =================
SHP_PATH = r"D:\Work space\DeepLearning\耕地分割\data\完整作物边界\作物分布.shp"
TIF_PATH = r"D:\Work space\DeepLearning\耕地分割\data\Sentinel2_TimeSeries_Stack_2023-0000000000-0000006400.tif"
MODEL_PATH = r"D:\Work space\DeepLearning\耕地分割\local_rf_model_tif.pkl"
OUTPUT_SHP = r"D:\Work space\DeepLearning\耕地分割\RF-feature\parcels_classified_2023_opt.shp"

BAND_NAMES = [
    'Month4_NDVI', 'Month4_NIR', 'Month4_GCVI', 'Month4_EVI',
    'Month5_NDVI', 'Month5_NIR', 'Month5_GCVI', 'Month5_EVI',
    'Month6_NDVI', 'Month6_NIR', 'Month6_GCVI', 'Month6_EVI',
    'Month7_NDVI', 'Month7_NIR', 'Month7_GCVI', 'Month7_EVI',
    'Month8_NDVI', 'Month8_NIR', 'Month8_GCVI', 'Month8_EVI',
    'Month9_NDVI', 'Month9_NIR', 'Month9_GCVI', 'Month9_EVI',
    'Month10_NDVI', 'Month10_NIR', 'Month10_GCVI', 'Month10_EVI'
]

# ================= 2. 准备工作 =================
try:
    M7_NDVI_IDX = BAND_NAMES.index('Month7_NDVI')
except:
    M7_NDVI_IDX = -1

print(f"🚀 加载 RF 模型: {MODEL_PATH}")
rf_model = joblib.load(MODEL_PATH)

# 🌟🌟 核心优化：强制改为单线程推理 🌟🌟
# 避免对每个微小地块都启动多线程的巨大开销
if hasattr(rf_model, 'n_jobs'):
    rf_model.n_jobs = 1
    print("⚡ 已开启单线程极速模式 (n_jobs=1)")

print(f"⏳ 读取矢量数据: {SHP_PATH}")
gdf = gpd.read_file(SHP_PATH)
original_count = len(gdf)
print(f"   原始地块总数: {original_count}")

# ================= 3. 加载影像 =================
print(f"🌍 正在将影像读入内存...")
with rasterio.open(TIF_PATH) as src:
    if src.crs is not None and gdf.crs != src.crs:
        print(f"⚠️ 坐标转换: {gdf.crs} -> {src.crs}")
        gdf = gdf.to_crs(src.crs)
    
    src_transform = src.transform
    src_h, src_w = src.height, src.width
    src_bounds = src.bounds
    
    FULL_IMAGE = src.read() 
    print(f"✅ 影像加载完成! Shape: {FULL_IMAGE.shape}")

# ================= 4. 空间裁剪 =================
print("✂️ 正在进行空间过滤...")
img_bbox = box(*src_bounds)
gdf_valid = gdf.cx[src_bounds.left:src_bounds.right, src_bounds.bottom:src_bounds.top].copy()

valid_count = len(gdf_valid)
print(f"   ✅ 待处理地块: {valid_count} (剔除 {original_count - valid_count})")

if valid_count == 0:
    print("❌ 无有效地块")
    exit()

# ================= 5. 推理循环 =================
final_results = np.zeros(original_count, dtype=int)
success_count = 0
zero_count = 0

print("🚀 开始极速推理...")

# mininterval=1.0 减少进度条刷新频率，进一步给 CPU 减负
for idx, row in tqdm(gdf_valid.iterrows(), total=valid_count, desc="Inference", mininterval=1.0):
    original_idx = idx

    try:
        # 1. 计算窗口 (纯数学计算，极快)
        geom_bounds = row.geometry.bounds
        # 手动计算窗口，避免 rasterio 对象开销
        col_start = int((geom_bounds[0] - src_transform.c) / src_transform.a)
        row_start = int((geom_bounds[3] - src_transform.f) / src_transform.e)
        col_stop = int((geom_bounds[2] - src_transform.c) / src_transform.a) + 1
        row_stop = int((geom_bounds[1] - src_transform.f) / src_transform.e) + 1
        
        # 边界修正
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_stop = min(src_h, row_stop)
        col_stop = min(src_w, col_stop)
        
        if row_start >= row_stop or col_start >= col_stop:
            zero_count += 1
            continue
            
        # 2. 内存切片
        window_data = FULL_IMAGE[:, row_start:row_stop, col_start:col_stop]
        
        # 3. 快速生成掩膜
        # 使用切片后的 shape 直接构建
        h_win, w_win = row_stop - row_start, col_stop - col_start
        
        # 构造局部 transform
        win_transform = rasterio.Affine(
            src_transform.a, src_transform.b, src_transform.c + col_start * src_transform.a,
            src_transform.d, src_transform.e, src_transform.f + row_start * src_transform.e
        )
        
        mask = rasterio.features.geometry_mask(
            [row.geometry],
            out_shape=(h_win, w_win),
            transform=win_transform,
            invert=True
        )
        
        # 4. 提取像素
        valid_pixels = window_data.transpose(1, 2, 0)[mask]
        
        if len(valid_pixels) == 0:
            zero_count += 1
            continue
            
        # 5. 极简 NaN 检查 (只查第0列)
        if np.isnan(valid_pixels[0, 0]): # 假设如果第一个是NaN，后面可能也是，进一步加速
             if np.isnan(valid_pixels[:, 0]).any():
                valid_mask = ~np.isnan(valid_pixels[:, 0])
                valid_pixels = valid_pixels[valid_mask]

        if len(valid_pixels) == 0:
            zero_count += 1
            continue

        # 6. 预测 (单线程)
        pixel_preds = rf_model.predict(valid_pixels)
        
        # 7. 规则修正
        if M7_NDVI_IDX != -1:
            # 使用 numpy 布尔索引，极快
            mask_fix = (pixel_preds == 2) & (valid_pixels[:, M7_NDVI_IDX] < 0.62)
            if mask_fix.any(): # 只有在需要修的时候才赋值
                pixel_preds[mask_fix] = 3
            
        # 8. 投票
        if len(pixel_preds) > 0:
            # 只有1个像素时直接取值，避免调用 mode
            if len(pixel_preds) == 1:
                res = pixel_preds[0]
            else:
                res = stats.mode(pixel_preds, keepdims=True)[0][0]
                
            final_results[original_idx] = int(res)
            success_count += 1
        else:
            zero_count += 1

    except Exception:
        zero_count += 1

print(f"📊 统计: 成功 {success_count} | 无效 {zero_count}")

# ================= 6. 保存 =================
print("💾 保存结果...")
gdf['Statistics'] = final_results
gdf_out = gdf[gdf['Statistics'] > 0].copy() # 仅保存有结果的

class_map = {1:'棉花', 2:'玉米', 3:'小麦', 4:'番茄', 5:'打瓜', 6:'葵花', 7:'葡萄', 8:'甜瓜', 9:'西瓜', 10:'其他'}
gdf_out['ClassName'] = gdf_out['Statistics'].map(class_map)

gdf_out.to_file(OUTPUT_SHP, encoding='utf-8')
print(f"🎉 完成! {OUTPUT_SHP}")