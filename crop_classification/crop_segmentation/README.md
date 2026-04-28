# 遥感作物分类系统

## 项目简介

这是一套用于遥感作物分类的模块化Python代码，支持从多波段GeoTIFF影像和ESRI Shapefile矢量数据中提取特征，训练机器学习或深度学习模型，并对新数据进行推理预测。

## 目录结构

```
crop_segmentation/
│
├── core/                 # 核心功能模块
│   ├── __init__.py
│   ├── data_loader.py    # 数据加载和特征提取
 │   ├── data_loader_optimized.py  # 优化的数据加载器（适用于大影像）
│   ├── models_ml.py      # 机器学习模型工厂
│   └── models_dl.py      # 深度学习模型定义
│
├── interfaces/           # 业务接口模块
│   ├── __init__.py
│   ├── train_interface.py   # 训练入口
│   └── infer_interface.py   # 推理入口
│
└── utils/                # 工具模块
    ├── __init__.py
    └── geo_utils.py      # 地理空间工具函数
```

## 核心功能

### 1. 数据处理
- **波段自适应**: 动态获取影像波段数，自动调整模型输入维度
- **坐标系统一**: 自动检查并转换矢量数据到影像坐标系
- **特征提取**: 支持从地块中提取多波段均值特征

### 2. 模型支持
- **机器学习模型**: 
  - 随机森林 (Random Forest)
  - 支持向量机 (SVM)
  - XGBoost
  - LightGBM
- **深度学习模型**: 
  - 1D-CNN (PyTorch)

### 3. 高性能处理
- **纯Numpy运算**: 避免使用Pandas DataFrame，提高计算效率
- **空间索引过滤**: 快速过滤不在影像范围内的地块
- **内存切片**: 一次性读取影像到内存，减少IO操作

### 4. 工程化设计
- **标准化处理**: 训练和推理使用相同的StandardScaler
- **标签编码**: 自动处理非连续类别ID
- **模型打包**: 训练后的模型包含所有必要组件，确保可移植性
- **错误处理**: 完善的异常捕获和处理机制
- **内存管理**: 智能内存检测，根据系统内存自动选择处理模式
- **超参数搜索**: 内置Grid Search功能，自动寻找最佳模型参数
- **模型评估**: 内置验证集评估功能，提供详细的分类报告

## 安装依赖

### 必要依赖
```bash
pip install geopandas rasterio scikit-learn torch numpy psutil
```

### 可选依赖
```bash
pip install xgboost lightgbm  # 用于XGBoost和LightGBM模型
```

## 使用方法

### 1. 模型训练

#### 1.1 基础训练

```python
from crop_segmentation.interfaces.train_interface import train_pipeline

# 训练随机森林模型
model_path = train_pipeline(
    model_type='rf',  # 支持 'rf', 'svm', 'xgboost', 'lgbm', 'cnn'
    shp_path='path/to/parcels.shp',  # 包含标签的矢量文件
    tif_path='path/to/image.tif',    # 多波段遥感影像
    output_dir='path/to/output',      # 模型保存目录
    label_column='class_id',          # 标签列名
    use_optimized_loader=False,       # 是否使用优化的数据加载器
    force_memory=False,               # 是否强制将影像加载到内存
    grid_search=False                 # 是否使用超参数搜索
)

print(f"模型已保存至: {model_path}")
```

#### 1.2 高级训练选项

##### 使用优化的数据加载器（适用于大影像）

```python
model_path = train_pipeline(
    model_type='rf',
    shp_path='path/to/parcels.shp',
    tif_path='path/to/large_image.tif',
    output_dir='models',
    label_column='class_id',
    use_optimized_loader=True,  # 启用优化的数据加载器
    max_workers=4,              # 使用4个工作进程
    chunk_size=1024              # 分块大小（像素）
)
```

##### 使用超参数搜索

```python
model_path = train_pipeline(
    model_type='rf',
    shp_path='path/to/parcels.shp',
    tif_path='path/to/image.tif',
    output_dir='models_grid_search',
    label_column='class_id',
    grid_search=True  # 启用超参数搜索
)
```

##### 深度学习模型训练

```python
model_path = train_pipeline(
    model_type='cnn',
    shp_path='path/to/parcels.shp',
    tif_path='path/to/image.tif',
    output_dir='models',
    label_column='class_id',
    epochs=100,         # 训练轮数
    batch_size=32,      # 批次大小
    learning_rate=0.001 # 学习率
)
```

### 2. 模型推理

#### 2.1 基础推理

```python
from crop_segmentation.interfaces.infer_interface import predict_pipeline

# 使用训练好的模型进行推理
output_shp = predict_pipeline(
    model_path='path/to/model_bundle.joblib',  # 训练生成的模型文件
    tif_path='path/to/new_image.tif',          # 新的遥感影像
    shp_path='path/to/new_parcels.shp',        # 待预测的矢量文件
    output_shp='path/to/output_result.shp',     # 结果保存路径
    force_memory=False                          # 是否强制将影像加载到内存
)

print(f"推理结果已保存至: {output_shp}")
```

#### 2.2 推理参数说明

```python
output_shp = predict_pipeline(
    model_path='path/to/model_bundle.joblib',
    tif_path='path/to/new_image.tif',
    shp_path='path/to/new_parcels.shp',
    output_shp='results/predicted.shp',
    max_workers=4,      # 工作进程数（为了API一致性）
    chunk_size=1024,     # 分块大小（为了API一致性）
    force_memory=False   # 是否强制内存模式
)
```

## 技术特点

1. **波段自适应**: 代码不写死波段数量，通过`src.count`动态获取
2. **坐标系统一**: 自动检查并转换矢量到影像坐标系
3. **高性能处理**: 纯Numpy矩阵运算，空间索引过滤，内存切片
4. **模型通用性**: 统一的标准化和标签编码处理
5. **工程化输出**: 模型打包保存，确保推理时的独立性

## 注意事项

1. **数据格式**: 输入影像必须是GeoTIFF格式，矢量数据必须是ESRI Shapefile格式
2. **标签格式**: 训练数据的标签列应该包含连续或非连续的类别ID
3. **内存管理**: 系统会自动检测内存情况，当内存不足时会使用流式处理模式
4. **超参数搜索**: 仅支持机器学习模型（rf, svm, xgboost, lgbm），CNN模型不支持
5. **依赖管理**: 若缺少XGBoost或LightGBM，系统会自动降级，只支持Random Forest和SVM
6. **坐标系**: 确保影像文件包含正确的坐标系信息
7. **计算资源**: 超参数搜索会增加计算时间和资源消耗，请根据硬件情况使用
8. **评估模式**: 开启评估模式时，会划分验证集并评估模型性能，关闭时使用完整数据集训练

## 示例代码

### 完整训练和推理示例

#### 示例1: 带内存管理的训练和推理

```python
# 训练模型（自动内存检测）
from crop_segmentation.interfaces.train_interface import train_pipeline

model_path = train_pipeline(
    model_type='rf',
    shp_path='data/train_parcels.shp',
    tif_path='data/sentinel2_image.tif',
    output_dir='models',
    label_column='crop_type',
    use_optimized_loader=True,
    max_workers=4,
    chunk_size=1024,
    force_memory=False  # 自动检测内存
)

# 推理预测（自动内存检测）
from crop_segmentation.interfaces.infer_interface import predict_pipeline

output_shp = predict_pipeline(
    model_path=model_path,
    tif_path='data/new_sentinel2_image.tif',
    shp_path='data/test_parcels.shp',
    output_shp='results/predicted_parcels.shp',
    force_memory=False  # 自动检测内存
)

print(f"推理完成，结果已保存至: {output_shp}")
```

#### 示例2: 带超参数搜索的训练

```python
# 训练模型（使用超参数搜索）
from crop_segmentation.interfaces.train_interface import train_pipeline

model_path = train_pipeline(
    model_type='xgboost',
    shp_path='data/train_parcels.shp',
    tif_path='data/sentinel2_image.tif',
    output_dir='models_grid_search',
    label_column='crop_type',
    grid_search=True  # 启用超参数搜索
)

print(f"超参数搜索完成，最佳模型已保存至: {model_path}")
```

## 模型性能

- **训练速度**: 取决于数据量和模型类型，CNN模型训练时间较长
- **推理速度**: 内存模式下，每秒可处理约180个地块；流式处理模式下速度会略有下降
- **内存占用**: 
  - 内存模式: 主要取决于影像大小，6400x6400的28波段影像约需4.3GB内存
  - 流式处理模式: 内存占用显著降低，适合大影像
- **超参数搜索**: 会增加2-5倍的训练时间，具体取决于参数网格大小
- **并行处理**: 使用优化的数据加载器时，可利用多核心CPU加速处理

## 模型参数接口

### 机器学习模型参数

#### 1. 随机森林 (Random Forest) - `model_type='rf'`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_estimators` | 100 | 决策树的数量 |
| `max_depth` | None | 树的最大深度，None表示不限制 |
| `min_samples_split` | 2 | 分裂内部节点所需的最小样本数 |
| `min_samples_leaf` | 1 | 叶子节点所需的最小样本数 |
| `random_state` | 42 | 随机种子，确保结果可复现 |
| `max_features` | 'auto' | 寻找最佳分裂时考虑的特征数量 |
| `bootstrap` | True | 是否使用自助采样 |
| `class_weight` | None | 类别权重，可设置为'balanced' |

#### 2. 支持向量机 (SVM) - `model_type='svm'`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kernel` | 'rbf' | 核函数类型，可选：'rbf', 'linear', 'poly', 'sigmoid' |
| `C` | 1.0 | 正则化参数，控制惩罚强度 |
| `gamma` | 'scale' | 核函数系数，可选：'scale', 'auto' 或浮点数 |
| `random_state` | 42 | 随机种子，确保结果可复现 |
| `probability` | False | 是否启用概率估计 |
| `class_weight` | None | 类别权重，可设置为'balanced' |
| `degree` | 3 | 多项式核函数的阶数 |
| `coef0` | 0.0 | 多项式和sigmoid核函数的截距 |

#### 3. XGBoost - `model_type='xgboost'`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_estimators` | 100 | 决策树的数量 |
| `max_depth` | 6 | 树的最大深度 |
| `learning_rate` | 0.1 | 学习率，控制每棵树的贡献 |
| `random_state` | 42 | 随机种子，确保结果可复现 |
| `use_label_encoder` | False | 是否使用标签编码器 |
| `eval_metric` | 'mlogloss' | 评估指标 |
| `subsample` | 1.0 | 训练每棵树时使用的样本比例 |
| `colsample_bytree` | 1.0 | 训练每棵树时使用的特征比例 |
| `reg_alpha` | 0.0 | L1正则化系数 |
| `reg_lambda` | 1.0 | L2正则化系数 |

#### 4. LightGBM - `model_type='lgbm'`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_estimators` | 100 | 决策树的数量 |
| `max_depth` | 6 | 树的最大深度 |
| `learning_rate` | 0.1 | 学习率，控制每棵树的贡献 |
| `random_state` | 42 | 随机种子，确保结果可复现 |
| `subsample` | 1.0 | 训练每棵树时使用的样本比例 |
| `colsample_bytree` | 1.0 | 训练每棵树时使用的特征比例 |
| `reg_alpha` | 0.0 | L1正则化系数 |
| `reg_lambda` | 0.0 | L2正则化系数 |
| `num_leaves` | 31 | 叶子节点的最大数量 |
| `min_data_in_leaf` | 20 | 叶子节点所需的最小数据量 |

### 深度学习模型参数

#### 5. 1D-CNN - `model_type='cnn'`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 100 | 训练轮数 |
| `batch_size` | 32 | 批次大小 |
| `learning_rate` | 0.001 | 学习率 |
| `optimizer` | Adam | 优化器（固定为Adam） |
| `loss_function` | CrossEntropyLoss | 损失函数（固定为交叉熵损失） |

### 通用训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_type` | - | 模型类型，必须指定 |
| `shp_path` | - | Shapefile路径，必须指定 |
| `tif_path` | - | GeoTIFF路径，必须指定 |
| `output_dir` | - | 输出目录，必须指定 |
| `label_column` | 'class_id' | 标签列名 |
| `use_optimized_loader` | False | 是否使用优化的数据加载器 |
| `max_workers` | 4 | 工作进程数（仅用于优化的数据加载器） |
| `chunk_size` | 1024 | 分块大小（仅用于优化的数据加载器） |
| `force_memory` | False | 是否强制将影像加载到内存 |
| `grid_search` | False | 是否使用超参数搜索 |
| `eval_mode` | True | 是否开启评估模式，开启后会划分验证集并评估模型 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | - | 模型文件路径，必须指定 |
| `tif_path` | - | GeoTIFF路径，必须指定 |
| `shp_path` | - | Shapefile路径，必须指定 |
| `output_shp` | - | 输出Shapefile路径，必须指定 |
| `force_memory` | False | 是否强制将影像加载到内存 |
| `max_workers` | 4 | 工作进程数（为了API一致性） |
| `chunk_size` | 1024 | 分块大小（为了API一致性） |

## 技术栈

- **核心库**: Python 3.8+, NumPy, SciPy
- **地理处理**: GeoPandas, Rasterio
- **机器学习**: scikit-learn, XGBoost, LightGBM
- **深度学习**: PyTorch
- **工具库**: tqdm (进度条), psutil (内存检测)

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请联系项目维护者。
