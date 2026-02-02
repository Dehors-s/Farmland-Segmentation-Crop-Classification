import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

# ================= 1. 配置 =================
tif_path = r"D:\Work space\DeepLearning\耕地分割\data\Sentinel2_TimeSeries_Stack_2023-0000000000-0000006400.tif"  # 您的遥感影像路径
shp_path = r"D:\Work space\DeepLearning\耕地分割\data\完整作物边界\作物分布.shp"        # 您的矢量样本路径
label_col = "value"   

# ⚠️ 关键配置：您的 TIF 数据结构
# 请根据 GEE 导出设置修改
NUM_MONTHS = 7   # 4月-10月
NUM_FEATS = 4    # NDVI, NIR, GCVI, EVI
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 2. 数据提取与重塑 =================
# ... (前半部分与 RF 代码一致，读取 TIF 和 SHP) ...
print("⏳ 正在读取矢量和影像...")
gdf = gpd.read_file(shp_path)
data_list = []
labels_list = []

with rasterio.open(tif_path) as src:
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)
    
    print("🚀 提取特征中...")
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        try:
            out_image, _ = rasterio.mask.mask(src, [row.geometry], crop=True, nodata=np.nan)
            mean_values = np.nanmean(out_image, axis=(1, 2))
            if np.isnan(mean_values).all(): continue
            
            data_list.append(mean_values)
            labels_list.append(row[label_col])
        except:
            continue

# --- 数据预处理 ---
# 1. 转换为 Numpy 数组
X_raw = np.array(data_list) # Shape: (N, 28)
y_raw = np.array(labels_list)

# 2. 缺失值填充
X_raw = np.nan_to_num(X_raw)

# 3. 重塑为 3D 张量 (Sample, Channel, Time)
# 假设波段顺序是: [M4_F1, M4_F2, M4_F3, M4_F4, M5_F1...]
# 我们需要把它变成 PyTorch Conv1d 需要的 (N, C, T)
N = len(X_raw)
X_3d = X_raw.reshape(N, NUM_MONTHS, NUM_FEATS) # 先变 (N, T, C)
X_3d = X_3d.transpose(0, 2, 1) # 再转置为 (N, C, T) -> (N, 4, 7)

print(f"✅ 数据重塑完成: {X_3d.shape}")

# 4. 标签编码 (0, 1, 2...)
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

# 5. 标准化
scaler = StandardScaler()
# 展平标准化再变回来
X_scaled = scaler.fit_transform(X_3d.reshape(-1, NUM_FEATS)).reshape(N, NUM_FEATS, NUM_MONTHS)

# 6. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, stratify=y_encoded)

# ================= 3. 定义 1D-CNN 模型 =================
class CropCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CropCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_block(x)
        x = x.squeeze(-1)
        return self.fc(x)

# ================= 4. 训练循环 =================
# Dataset 封装
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(MyDataset(X_test, y_test), batch_size=BATCH_SIZE)

model = CropCNN(input_dim=NUM_FEATS, num_classes=len(np.unique(y_encoded))).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("🔥 开始训练 CNN...")
for epoch in range(EPOCHS):
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1} 完成")

# ================= 5. 评估 =================
model.eval()
preds = []
targets = []
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(DEVICE)
        out = model(bx)
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(by.numpy())

print(classification_report(targets, preds, target_names=[str(c) for c in le.classes_]))
torch.save(model.state_dict(), 'local_cnn_model.pth')
print("💾 模型已保存")