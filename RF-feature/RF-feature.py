import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ================= 1. 读取数据 =================
# ⚠️ 确保文件名正确
csv_path = 'Training_Data_for_Python_v4.csv'

try:
    df = pd.read_csv(csv_path)
    print(f"✅ 成功加载数据: {csv_path}")
    print(f"   总样本数: {len(df)}")
except FileNotFoundError:
    print(f"❌ 找不到文件: {csv_path}")
    # 尝试回退读取 v3 或 v2，防止报错
    try:
        csv_path = 'Training_Data_for_Python_v2.csv'
        df = pd.read_csv(csv_path)
        print(f"⚠️ 已自动回退读取: {csv_path}")
    except:
        exit()

# ================= 2. 准备数据 =================
label_col = 'value'
feature_cols = [c for c in df.columns if c != label_col and c != 'system:index' and c != '.geo']
X = df[feature_cols]
y = df[label_col]

# 类别字典 (用于显示)
class_names_dict = {
    1: 'Cotton', 2: 'Corn', 3: 'Wheat', 4: 'Tomato',
    5: 'SeedMln', 6: 'SunFlw', 7: 'Grape',
    8: 'Melon', 9: 'WtrMln', 10: 'Others'
}
unique_labels = sorted(y.unique())
target_names = [class_names_dict.get(i, str(i)) for i in unique_labels]

# 划分数据集 (Stratified 保证类别比例一致)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ================= 3. 训练与超参数搜索 (Grid Search) =================
print("\n⚙️ 正在进行超参数网格搜索 (Grid Search)...")
# 为了节省时间，这里只搜索最关键的参数
param_grid = {
    'n_estimators': [100, 200],
    'min_samples_leaf': [1, 2],  # 叶子节点少一点，允许模型学得更细
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print(f"🏆 最佳参数: {grid_search.best_params_}")

# 基础预测
y_pred = best_rf.predict(X_test)
acc_base = accuracy_score(y_test, y_pred)
print(f"📊 基础模型精度: {acc_base:.4%}")

# ================= 4. 深度诊断：画出“小麦 vs 玉米”的时间曲线 =================
# 这一步是为了寻找“规则修正”的灵感
print("\n🔍 正在绘制诊断曲线...")

# 优先使用 GCVI (如果有)，否则用 NDVI
plot_metric = 'GCVI' if any('GCVI' in c for c in feature_cols) else 'NDVI'
# 筛选出所有相关的月份列
metric_cols = sorted([c for c in feature_cols if plot_metric in c],
                     key=lambda x: int(x.split('Month')[1].split('_')[0]))

if len(metric_cols) > 0:
    # 提取月份数字
    months = [int(c.split('Month')[1].split('_')[0]) for c in metric_cols]

    # 计算 Corn (2) 和 Wheat (3) 的平均曲线
    mean_corn = df[df['value'] == 2][metric_cols].mean()
    mean_wheat = df[df['value'] == 3][metric_cols].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(months, mean_corn, label='Corn (Class 2)', marker='o', color='orange', linewidth=2)
    plt.plot(months, mean_wheat, label='Wheat (Class 3)', marker='x', color='green', linewidth=2)
    plt.title(f'Diagnosis: {plot_metric} Temporal Profile (Corn vs Wheat)', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel(plot_metric)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()
    print(f"👉 请观察上图：哪个月份两者的差距最大？(通常是 7月或8月)")

# ================= 5. 大招：规则修正 (Rule-Based Correction) =================
# 逻辑：如果模型预测是玉米，但 7月NDVI 比较低（像小麦），则强制改判为小麦。
# 阈值设定：根据之前的分析，玉米在7月通常 > 0.65，小麦 < 0.60
correction_col = 'Month7_NDVI'  # 核心特征
threshold = 0.62  # 核心阈值 (可根据上图微调)

print(f"\n🛠️ 执行规则修正 (阈值: {correction_col} < {threshold})...")

if correction_col in X_test.columns:
    y_pred_fixed = y_pred.copy()

    # 找出所有：[预测=玉米] 且 [7月NDVI < 0.62] 的样本
    # 这些很可能是被误判的“春小麦”
    mask_fix = (y_pred == 2) & (X_test[correction_col] < threshold)
    num_fixed = mask_fix.sum()

    # 执行改判：Corn(2) -> Wheat(3)
    y_pred_fixed[mask_fix] = 3

    acc_new = accuracy_score(y_test, y_pred_fixed)
    print(f"✨ 修正了 {num_fixed} 个样本")
    print(f"🚀 修正后精度: {acc_new:.4%} (提升: {acc_new - acc_base:.2%})")

    # 输出最终报告
    print("\n📑 最终分类报告 (修正后):")
    print(classification_report(y_test, y_pred_fixed, target_names=target_names))

    # 画最终混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred_fixed, labels=unique_labels),
                annot=True, fmt='d', cmap='Greens',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Final Confusion Matrix (After Correction)')
    plt.tight_layout()
    plt.show()

else:
    print(f"⚠️ 数据中缺少 {correction_col}，无法执行规则修正。")

# ================= 6. 保存最终方案 =================
# 注意：因为加入了规则修正，单纯保存 rf_model 是不够的。
# 实际使用时，需要载入模型 + 执行同样的规则代码。
joblib.dump(best_rf, 'local_crop_rf_model_v4_best.pkl')
print("\n💾 模型已保存。提示：实际预测时，请记得也加上上述的 '规则修正' 代码块！")