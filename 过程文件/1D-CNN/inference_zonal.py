import geopandas as gpd
import rasterio
import rasterio.mask
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy import stats  # 用于计算众数
from shapely.geometry import box  # 用于创建边界框几何对象

# ================= 1. 配置路径与参数 =================
# 输入文件
SHP_PATH = r"/\data\完整作物边界\作物分布.shp"  # 待预测的矢量文件
TIF_PATH = r"/\data\Sentinel2_TimeSeries_Stack_2023-0000000000-0000000000.tif"  # 大尺寸多波段遥感影像
MODEL_PATH = r"/\local_cnn_model.pth"  # 训练好的模型权重

# 输出文件
OUTPUT_SHP = "D:\\Work space\\DeepLearning\\farm\\1D-CNN\\My_Parcels_Result.shp" # 结果保存路径

# 数据参数 (必须与训练时一致!)
NUM_MONTHS = 7   # 4月-10月
NUM_FEATS = 4    # NDVI, NIR, GCVI, EVI
INPUT_CHANNELS = 4 
NUM_CLASSES = 10 # 与训练时一致，共10个类别
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 2. 定义模型结构 (必须与训练代码一致) =================
# 如果您之前把模型定义在了单独的文件里，也可以 import 进来
class SimpleTempCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleTempCNN, self).__init__()

        # 第一层卷积：提取基础特征
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)  # 防止过拟合
        )

        # 第二层卷积：提取更高级特征
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 全局平均池化 (Global Average Pooling) - 这比 Flatten 更抗噪
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 分类头
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # x shape: (Batch, 64, 7) -> GAP -> (Batch, 64, 1)
        x = self.gap(x).squeeze(-1)

        logits = self.fc(x)
        return logits

# ================= 3. 加载模型 =================
print(f"🚀 正在加载模型: {MODEL_PATH}")
model = SimpleTempCNN(input_dim=NUM_FEATS, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # 切换到推理模式 (关闭 Dropout/BatchNorm 更新)

# ================= 4. 主推理循环 =================
print(f"⏳ 正在读取矢量数据: {SHP_PATH}")
gdf = gpd.read_file(SHP_PATH)
print(f"   共 {len(gdf)} 个地块需要预测")

# 用于存储预测结果
predicted_classes = []

# 打开大图 (使用 rasterio 上下文管理器，不会一次性读取内存)
with rasterio.open(TIF_PATH) as src:
    print(f"🌍 影像信息: {src.width}x{src.height}, 波段数: {src.count}")
    print(f"   影像边界: {src.bounds}")
    print(f"   矢量数据坐标系: {gdf.crs}")
    print(f"   影像坐标系: {src.crs}")
    
    # 检查坐标系
    if src.crs is not None and gdf.crs != src.crs:
        print(f"⚠️ 坐标系不一致，正在转换矢量投影至 {src.crs}...")
        gdf = gdf.to_crs(src.crs)
    elif src.crs is None:
        print("⚠️ 影像文件缺少坐标系信息，跳过坐标转换...")

    # 创建影像边界的几何对象
    src_bounds_geom = box(*src.bounds)

    # 处理所有地块
    print("\n🚀 开始处理所有地块:")
    success_count = 0
    error_count = 0
    zero_count = 0
    
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Inference"):
        geom = row.geometry
        
        try:
            # 检查是否与影像边界重叠
            if not geom.intersects(src_bounds_geom):
                predicted_classes.append(0)
                zero_count += 1
                continue
                
            # --- 步骤 A: 裁剪 (Masking) ---
            # 只读取当前地块范围内的像素
            # out_image shape: (Bands=28, Height, Width)
            out_image, transform = rasterio.mask.mask(src, [geom], crop=True, nodata=np.nan)
            
            # --- 步骤 B: 数据重塑 (Reshape) ---
            # 我们需要把 (28, H, W) 变成 (Pixels, Channels, Time)
            # 1. 展平为 (28, Pixels)
            C_total, H, W = out_image.shape
            pixels = out_image.reshape(C_total, -1).T # 变成 (Pixels, 28)
            
            # 2. 去除空值 (NaN)
            # 只要某行有 NaN，就说明是无效像素 (背景)
            valid_mask = ~np.isnan(pixels).any(axis=1)
            pixels = pixels[valid_mask]
            
            if len(pixels) == 0:
                predicted_classes.append(0) # 无效地块填 0
                zero_count += 1
                continue
                
            # 3. 重塑为 (N, Time, Feat) -> 转置为 (N, Feat, Time) 用于 CNN
            # 假设波段顺序是: [M4_F1, M4_F2... M5_F1...]
            # 这步很关键，必须和训练时的 reshape 逻辑完全反过来
            N_pix = len(pixels)
            # 先变成 (N, 7个月, 4特征)
            input_tensor = pixels.reshape(N_pix, NUM_MONTHS, NUM_FEATS)
            # 再转置成 (N, 4特征, 7个月) 以适应 Conv1d
            input_tensor = input_tensor.transpose(0, 2, 1)
            
            # --- 步骤 C: 归一化 (Normalization) ---
            # ⚠️ 极其重要：深度学习必须归一化
            # 这里使用简单的 Instance Normalization (减均值除方差)
            # 如果您训练时用了固定的 mean/std，这里最好用固定的
            mean = input_tensor.mean(axis=(0, 2), keepdims=True)
            std = input_tensor.std(axis=(0, 2), keepdims=True) + 1e-6
            input_tensor = (input_tensor - mean) / std

            # --- 步骤 D: 模型预测 ---
            # 转为 Tensor 并送入 GPU
            tensor_x = torch.tensor(input_tensor, dtype=torch.float32).to(DEVICE)
            
            # 分批预测 (如果地块太大，像素太多，显存会爆，所以要切分)
            # 对于一般地块，直接预测即可
            with torch.no_grad():
                outputs = model(tensor_x)
                # 获取每个像素的预测类别
                pixel_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # --- 步骤 E: 多数投票 (Majority Voting) ---
            # 统计该地块内哪个类别出现次数最多
            if len(pixel_preds) > 0:
                mode_result = stats.mode(pixel_preds, keepdims=True)[0][0]
                predicted_classes.append(int(mode_result))
                success_count += 1
            else:
                predicted_classes.append(0)
                zero_count += 1

        except Exception as e:
            # 遇到几何错误等，填 0 跳过
            if idx < 10 or error_count < 10:  # 只打印前10个错误和前10个地块的错误
                print(f"Error at ID {idx}: {e}")
            predicted_classes.append(0)
            error_count += 1

    # 打印统计信息
    print(f"\n📊 预测结果统计:")
    print(f"   总地块数: {len(gdf)}")
    print(f"   成功预测数: {success_count}")
    print(f"   错误数: {error_count}")
    print(f"   零预测数: {zero_count}")



# ================= 5. 保存结果 =================
print("💾 正在写入属性表...")
# 新建字段 'Statistics'
gdf['Statistics'] = predicted_classes

# 可选：如果你想把数字转回中文
class_map = {1:'棉花', 2:'玉米', 3:'小麦', 4:'番茄', 5:'打瓜', 6:'葵花', 7:'葡萄', 8:'甜瓜', 9:'西瓜', 10:'其他'}
gdf['ClassName'] = gdf['Statistics'].map(class_map)

gdf.to_file(OUTPUT_SHP, encoding='utf-8')
print(f"🎉 处理完成！结果已保存至: {OUTPUT_SHP}")