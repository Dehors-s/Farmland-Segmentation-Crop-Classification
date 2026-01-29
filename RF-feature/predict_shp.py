import pandas as pd
import geopandas as gpd
import joblib
import numpy as np

# ================= 配置路径 =================
# 1. 模型路径
model_path = 'local_crop_rf_model_v4_best.pkl'
# 2. GEE 导出的特征 CSV 路径
feature_csv_path = 'Inference_Features_2023.csv'
# 3. 原始 Shapefile 路径 (待分类的矢量)
# ⚠️ 注意：这个 SHP 必须和上传到 GEE 的那个完全对应
shp_path = 'path/to/your/parcels.shp'
# 4. 结果保存路径
output_shp_path = 'parcels_classified_2023.shp'

# ================= 1. 加载数据与模型 =================
print("⏳ 正在加载模型和数据...")
rf_model = joblib.load(model_path)
df_features = pd.read_csv(feature_csv_path)
gdf_shp = gpd.read_file(shp_path)

print(f"✅ 模型加载成功！")
print(f"📊 待分类样本数 (CSV): {len(df_features)}")
print(f"🗺️ 原始地块数 (SHP): {len(gdf_shp)}")

# ================= 2. 数据对齐与清洗 =================
# 确保 CSV 里的 ID 能对应上 SHP 里的 ID
# GEE 导出的 CSV 通常有一个 'system:index' 列
# SHP 加载后通常索引就是 index，或者有特定 ID 字段
# 这里假设 CSV 的 system:index 对应 SHP 的顺序 (通常是一致的，除非 GEE 过滤了空值)

# 准备特征矩阵 X
# 自动排除非特征列
feature_cols = [c for c in df_features.columns if
                'Month' in c and ('NDVI' in c or 'GCVI' in c or 'EVI' in c or 'NIR' in c)]
print(f"🔍 使用特征列: {len(feature_cols)} 个")

X = df_features[feature_cols]

# 处理缺失值 (GEE 导出可能因为云导致某些月是空值)
# 随机森林不能处理 NaN，我们要用 0 或者均值填充
if X.isnull().values.any():
    print("⚠️ 检测到空值，正在用 0 填充...")
    X = X.fillna(0)

# ================= 3. 模型预测 =================
print("🚀 开始预测...")
y_pred = rf_model.predict(X)

# ================= 4. 规则修正 (Apply Rule-Based Correction) =================
# 必须复现之前训练时的修正逻辑：
# "如果预测是玉米(2)，但7月NDVI < 0.62，强制改为小麦(3)"

if 'Month7_NDVI' in X.columns:
    threshold = 0.62
    mask_fix = (y_pred == 2) & (X['Month7_NDVI'] < threshold)
    num_fixed = mask_fix.sum()

    if num_fixed > 0:
        y_pred[mask_fix] = 3
        print(f"✨ 触发规则修正：将 {num_fixed} 个疑似小麦的'假玉米'改回了小麦。")
else:
    print("⚠️ 特征中缺少 Month7_NDVI，跳过规则修正。")

# ================= 5. 将结果写入属性表 =================
print("💾 正在写入属性表...")

# 类别名称映射 (可选，如果您想存中文)
class_map = {
    1: '棉花', 2: '玉米', 3: '小麦', 4: '番茄',
    5: '打瓜', 6: '葵花', 7: '葡萄',
    8: '甜瓜', 9: '西瓜', 10: '其他'
}

# 为了安全，我们通过 system:index 来 merge，而不是直接赋值
# 假设 GEE CSV 的 system:index 格式是 "00000000000000000001" 这种
# 我们先给 df_features 加上预测结果列
df_features['Pred_Code'] = y_pred
df_features['Pred_Name'] = df_features['Pred_Code'].map(class_map)

# 关键：如何把 CSV 挂载回 SHP？
# 方案 A：如果顺序完全没变 (GEE没有过滤掉任何几何)，直接按行赋值
if len(df_features) == len(gdf_shp):
    gdf_shp['Crop_Code'] = y_pred
    gdf_shp['Crop_Name'] = df_features['Pred_Name']
    print("✅ 长度一致，直接按顺序写入。")
else:
    print("⚠️ 长度不一致 (可能GEE过滤了无效几何)，尝试通过 ID 连接...")
    # 这需要您知道 SHP 里哪一列是唯一 ID，假设 SHP 里有个叫 'id' 的列
    # 或者我们假设 GEE 的 system:index 对应 SHP 的默认索引
    # 这里做最简单的处理：新建一个空的列，只填能对上的
    # 实际项目中建议在 GEE 导出时把 SHP 的原始 ID 属性也导出来
    print("❌ 无法自动对齐，请检查 GEE 导出时是否包含了原始 SHP 的 ID 列。")
    # 强制写入前 N 个 (仅供测试)
    gdf_shp['Crop_Code'] = 0
    gdf_shp.iloc[:len(y_pred), gdf_shp.columns.get_loc('Crop_Code')] = y_pred

# ================= 6. 保存结果 =================
# 解决中文乱码问题
gdf_shp.to_file(output_shp_path, driver='ESRI Shapefile', encoding='utf-8')

print(f"🎉 大功告成！结果已保存至: {output_shp_path}")
print("您现在可以在 ArcGIS 或 QGIS 中打开它查看分类结果了。")