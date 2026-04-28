import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ================= 1. 配置参数 =================
CSV_PATH = 'Training_Data_for_Python_v4.csv'  # 确保使用 V4 增强版
BATCH_SIZE = 64
EPOCHS = 150  # 深度学习需要多轮训练
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🚀 使用设备: {DEVICE} (如果是 cpu，请检查显卡驱动)")


# ================= 2. 数据准备 (核心：将宽表转为 3D 张量) =================
# 目标形状: (样本数 N, 时间步 T=7, 特征数 C=4)

def load_and_process_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # 为了演示，如果找不到文件，我生成一些假数据
        print("⚠️ 没找到文件，生成随机假数据用于演示...")
        df = pd.DataFrame(np.random.rand(1000, 28),
                          columns=[f'Month{m}_{f}' for m in range(4, 11) for f in ['NDVI', 'NIR', 'GCVI', 'EVI']])
        df['value'] = np.random.randint(1, 11, 1000)

    # 1. 提取标签
    label_col = 'value'
    y = df[label_col].values

    # 2. 重新编码标签 (0, 1, 2...) PyTorch 需要从0开始
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3. 提取特征并 Reshape
    # 也就是把一行 28 个数，变成 [7个月, 4个特征] 的矩阵
    months = [4, 5, 6, 7, 8, 9, 10]
    feats = ['NDVI', 'NIR', 'GCVI', 'EVI']  # 必须与 CSV 列名包含的字串一致

    n_samples = len(df)
    n_steps = len(months)
    n_feats = len(feats)

    X_3d = np.zeros((n_samples, n_steps, n_feats))

    for i, m in enumerate(months):
        for j, f in enumerate(feats):
            col_name = f'Month{m}_{f}'
            if col_name in df.columns:
                X_3d[:, i, j] = df[col_name].values
            else:
                print(f"⚠️ 警告: 缺少列 {col_name}，用 0 填充")

    # 4. 标准化 (对于深度学习极其重要！)
    # 我们把 (N, T, C) 展平 -> 标准化 -> 恢复
    scaler = StandardScaler()
    X_reshaped = X_3d.reshape(-1, n_feats)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_final = X_scaled.reshape(n_samples, n_steps, n_feats)

    return X_final, y_encoded, le


print("⏳ 正在处理数据...")
X, y, label_encoder = load_and_process_data(CSV_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"✅ 数据形态: {X_train.shape}")
print(f"   样本数: {X_train.shape[0]}")
print(f"   时间步: {X_train.shape[1]} (4月-10月)")
print(f"   特征数: {X_train.shape[2]} (NDVI, NIR, GCVI, EVI)")


# ================= 3. 定义 Dataset =================
class CropDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(CropDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(CropDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# ================= 4. 定义 LSTM 模型 (源自 BreizhCrops) =================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()

        # LSTM 层
        # batch_first=True: 输入格式为 (batch, seq, feature)
        # bidirectional=True: 双向 LSTM (能同时看到过去和未来，效果更好)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # 全连接层 (分类头)
        # 因为是双向，所以输入维度是 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]

        # lstm_out shape: [batch, seq_len, hidden_dim * 2]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 我们只取最后一个时间步的输出用于分类
        # 或者取 Global Max Pooling (更适合短序列)
        # 这里演示取最后一个时间步
        last_step_out = lstm_out[:, -1, :]

        logits = self.fc(last_step_out)
        return logits


# 初始化模型
input_dim = X_train.shape[2]  # 4
num_classes = len(np.unique(y))
model = LSTMClassifier(input_dim, hidden_dim=128, num_layers=2, num_classes=num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================= 5. 训练循环 =================
print("\n🔥 开始训练 Deep Learning 模型...")
loss_history = []

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

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# 画出 Loss 曲线
#plt.plot(loss_history)
#plt.title('Training Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.show()

# ================= 6. 评估与报告 =================
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

# 将数字标签还原为原始类别 (比如 0 -> 1(棉花), 1 -> 2(玉米)...)
y_true_decoded = label_encoder.inverse_transform(all_labels)
y_pred_decoded = label_encoder.inverse_transform(all_preds)

print(f"\n🏆 LSTM 测试集精度: {accuracy_score(y_true_decoded, y_pred_decoded):.4%}")
print("\n📑 深度学习分类报告:")
# 映射类别名
class_map = {1: 'Cotton', 2: 'Corn', 3: 'Wheat', 4: 'Tomato', 5: 'SeedMln', 6: 'SunFlw', 7: 'Grape', 8: 'Melon',
             9: 'WtrMln', 10: 'Others'}
target_names = [class_map.get(i, str(i)) for i in sorted(np.unique(y_true_decoded))]

print(classification_report(y_true_decoded, y_pred_decoded, target_names=target_names))

# ================= 7. 保存模型 =================
torch.save(model.state_dict(), 'lstm_crop_model.pth')
print("\n💾 模型已保存为 lstm_crop_model.pth")