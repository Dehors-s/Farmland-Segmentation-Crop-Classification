import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm # 进度条库，pip install tqdm

# ================= 1. 配置路径 =================
tif_path = r"D:\Work space\DeepLearning\耕地分割\data\Sentinel2_TimeSeries_Stack_2023-0000000000-0000006400.tif"  # 您的遥感影像路径
shp_path = r"D:\Work space\DeepLearning\耕地分割\data\完整作物边界\作物分布.shp"        # 您的矢量样本路径
label_col = "value"                            # 矢量表中代表作物类别的字段名

# ================= 2. 特征提取 (从图上抠数据) =================
print("⏳ 正在读取矢量和影像...")
gdf = gpd.read_file(shp_path)
print(f"   矢量样本数: {len(gdf)}")

# 提取特征列表
data_list = []
labels_list = []

with rasterio.open(tif_path) as src:
    print(f"   影像波段数: {src.count}")
    
    # 检查坐标系是否一致
    if gdf.crs != src.crs:
        print(f"⚠️ 坐标系不一致，正在转换矢量投影至 {src.crs}...")
        gdf = gdf.to_crs(src.crs)

    print("🚀 开始提取地块特征...")
    # 遍历每一个地块
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        geom = row.geometry
        label = row[label_col]
        
        try:
            # 1. 裁剪：只读取当前地块范围内的像素
            out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True, nodata=np.nan)
            
            # 2. 统计：计算地块内所有像素的平均值 (忽略空值)
            # out_image shape: (Bands, Height, Width) -> 对 H, W 求均值
            mean_values = np.nanmean(out_image, axis=(1, 2))
            
            # 3. 过滤：如果提取全是空值（比如地块在图外），跳过
            if np.isnan(mean_values).all():
                continue
                
            # 存入列表
            data_list.append(mean_values)
            labels_list.append(label)
            
        except Exception as e:
            continue

# 转为 DataFrame
# 自动命名列为 B1, B2, ... B28
feature_names = [f"Band_{i+1}" for i in range(len(data_list[0]))]
X = pd.DataFrame(data_list, columns=feature_names)
y = pd.Series(labels_list)

# 填充可能残留的 NaN
X = X.fillna(0)

print(f"✅ 特征提取完成。有效样本数: {len(X)}")

# ================= 3. 训练随机森林 =================
print("🌲 正在训练随机森林...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# ================= 4. 评估与保存 =================
y_pred = rf.predict(X_test)
print(f"\n🏆 验证集精度: {accuracy_score(y_test, y_pred):.4%}")
print(classification_report(y_test, y_pred))

joblib.dump(rf, 'local_rf_model_tif.pkl')
print("💾 模型已保存为 local_rf_model_tif.pkl")