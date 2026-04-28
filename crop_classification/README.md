# 地物分类 — 遥感作物分类系统 (Crop Classification)

基于多波段遥感影像与矢量地块的作物分类系统。支持随机森林（RF）、SVM、XGBoost、LightGBM 及 1D-CNN 等多种模型。

## 📁 目录结构

```
crop_classification/
│
├── crop_segmentation/          # 核心分类模块（Python package）
│   ├── core/
│   │   ├── data_loader.py              # 地块特征提取（逐地块）
│   │   ├── data_loader_optimized.py    # 分块并行特征提取（大影像）
│   │   ├── models_ml.py               # 机器学习模型工厂（RF/SVM/XGB/LGBM）
│   │   └── models_dl.py               # 1D-CNN 深度学习模型
│   ├── interfaces/
│   │   ├── train_interface.py          # 训练入口
│   │   └── infer_interface.py          # 推理入口
│   ├── utils/
│   │   └── geo_utils.py               # 地理空间工具
│   ├── test_models/                    # 预训练模型
│   ├── check_model_switch_smoke.py     # 冒烟测试
│   ├── check_model_switch_realdata.py  # 真实数据测试
│   ├── test_module_import.py           # 模块导入测试
│   ├── README.md
│   └── README_parcel_extraction.md     # 地块提取算法说明
│
├── legacy/                     # 历史实验脚本
│   ├── 1D-CNN/                 # 早期 1D-CNN 模型
│   │   ├── 1D-CNN.py
│   │   ├── 1D-CNN-v2.py
│   │   ├── inference_zonal.py
│   │   ├── best_cnn_model.pth
│   │   └── local_cnn_model.pth
│   ├── LSTM/                   # 早期 LSTM 模型
│   │   ├── LSTM.py
│   │   └── lstm_crop_model.pth
│   └── check_duplicate_images.py
│
├── requirements.txt            # 项目依赖
└── README.md
```

## 环境准备

```bash
pip install -r crop_classification/requirements.txt
```

可选加速：
```bash
pip install xgboost lightgbm  # 梯度提升模型
```

## 快速开始

### 1. 冒烟测试

```bash
python crop_classification/crop_segmentation/check_model_switch_smoke.py
```

### 2. 模型训练

```python
from crop_classification.crop_segmentation.interfaces.train_interface import train_pipeline

model_path = train_pipeline(
    model_type='rf',                    # 'rf', 'svm', 'xgboost', 'lgbm', 'cnn'
    shp_path='path/to/parcels.shp',     # 标签矢量
    tif_path='path/to/image.tif',       # 多波段遥感影像
    output_dir='./models',
    label_column='class_id',
    use_optimized_loader=True,          # 大影像启用分块
    grid_search=False                    # 超参数搜索
)
```

### 3. 模型推理

```python
from crop_classification.crop_segmentation.interfaces.infer_interface import predict_pipeline

output_shp = predict_pipeline(
    model_path='path/to/model_bundle.joblib',
    tif_path='path/to/image.tif',
    shp_path='path/to/parcels.shp',
    output_shp='./result.shp'
)
```

## 支持的模型

| 模型 | 类型 | 说明 |
|------|------|------|
| Random Forest | 机器学习 | 鲁棒性好，默认推荐 |
| SVM | 机器学习 | 小样本高维特征 |
| XGBoost | 机器学习 | 梯度提升，精度高 |
| LightGBM | 机器学习 | 训练快速，大样本 |
| 1D-CNN | 深度学习 | 时序/光谱特征提取 |

## 核心特性

- **波段自适应**: 动态获取波段数，自动调整模型输入维度
- **坐标系统一**: 自动将矢量投影到影像坐标系
- **智能内存管理**: 自动检测内存，大影像启用分块处理
- **模型打包**: 训练输出包含 scaler + label encoder 的完整 bundle
- **超参数搜索**: 内置 Grid Search

## 依赖

见 `requirements.txt`。核心依赖：
- geopandas, rasterio (地理空间)
- scikit-learn, numpy, scipy
- torch (1D-CNN)
- tqdm, matplotlib, joblib
- 可选: xgboost, lightgbm
