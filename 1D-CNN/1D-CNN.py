import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ================= 1. 配置参数 (防过拟合版) =================
# 您的文件路径
CSV_PATH = 'Training_Data_for_Python_v4.csv'
BATCH_SIZE = 32  # 小 Batch 有助于泛化
EPOCHS = 150  # 稍微减少轮数，避免过拟合
LEARNING_RATE = 0.0005  # 降低学习率，让它学得细致点
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= 2. 数据准备 (保持不变) =================
def load_and_process_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except:
        # 自动回退逻辑
        df = pd.read_csv('Training_Data_for_Python_v2.csv')

    label_col = 'value'
    y = df[label_col].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 动态获取特征列
    # 假设列名格式为 MonthX_Feature
    # 我们需要构建 (N, C, T) 的格式用于 CNN，或者 (N, T, C) 用于 LSTM
    # PyTorch Conv1d 要求的输入是 (Batch, Channels, Length) -> (N, 4, 7)

    months = [4, 5, 6, 7, 8, 9, 10]
    # 自动推断有哪些特征 (NDVI, GCVI...)
    all_cols = df.columns
    sample_cols = [c for c in all_cols if 'Month4_' in c]
    feats = [c.replace('Month4_', '') for c in sample_cols]
    print(f"检测到的特征: {feats}")

    n_samples = len(df)
    n_steps = len(months)
    n_feats = len(feats)

    # 构建 (N, C, T) 格式 -> 注意这里和 LSTM 不一样！CNN 喜欢 Channel 在前
    X_3d = np.zeros((n_samples, n_feats, n_steps))

    for i, m in enumerate(months):
        for j, f in enumerate(feats):
            col_name = f'Month{m}_{f}'
            if col_name in df.columns:
                X_3d[:, j, i] = df[col_name].values
            else:
                X_3d[:, j, i] = 0

    # 标准化 (对每个特征通道单独标准化)
    for c in range(n_feats):
        mean_val = X_3d[:, c, :].mean()
        std_val = X_3d[:, c, :].std() + 1e-6
        X_3d[:, c, :] = (X_3d[:, c, :] - mean_val) / std_val

    return X_3d, y_encoded, le


print("⏳ 正在处理数据 (CNN格式)...")
X, y, label_encoder = load_and_process_data(CSV_PATH)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"✅ 数据形态: {X_train.shape} (样本, 特征, 时间步)")


# Dataset
class CropDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


train_loader = DataLoader(CropDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(CropDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# ================= 3. 定义 1D-CNN 模型 (TempCNN 简化版) =================
class SimpleTempCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleTempCNN, self).__init__()

        # 第一层卷积：提取基础特征
        # input: (Batch, C=4, T=7)
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


# ================= 4. 定义 Focal Loss (解决难分类样本) =================
# 这是专门用来解决“小麦 vs 玉米”分不清的神器
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        log_pt = -self.ce(inputs, targets)
        pt = torch.exp(log_pt)
        loss = self.alpha * (1 - pt) ** self.gamma * (-log_pt)
        return loss.mean()


# ================= 5. 初始化与训练 =================
input_dim = X_train.shape[1]  # 特征数 (4)
num_classes = len(np.unique(y))

model = SimpleTempCNN(input_dim, num_classes).to(DEVICE)
# 使用 Focal Loss 替代普通的 CrossEntropy
criterion = FocalLoss(gamma=2.0)
# 加一点 weight_decay (L2正则化) 进一步抑制过拟合
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print("\n🔥 开始训练 1D-CNN 模型...")
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 简单的验证逻辑
    if (epoch + 1) % 5 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {running_loss / len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_cnn_model.pth')

# ================= 6. 最终评估 =================
print(f"\n🏆 训练结束，加载最佳模型 (Acc: {best_acc:.2f}%) 进行评估...")
model.load_state_dict(torch.load('best_cnn_model.pth'))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

y_true_decoded = label_encoder.inverse_transform(all_labels)
y_pred_decoded = label_encoder.inverse_transform(all_preds)

print("\n📑 1D-CNN 分类报告:")
# 映射类别名
class_map = {1: 'Cotton', 2: 'Corn', 3: 'Wheat', 4: 'Tomato', 5: 'SeedMln', 6: 'SunFlw', 7: 'Grape', 8: 'Melon',
             9: 'WtrMln', 10: 'Others'}
target_names = [class_map.get(i, str(i)) for i in sorted(np.unique(y_true_decoded))]
print(classification_report(y_true_decoded, y_pred_decoded, target_names=target_names))