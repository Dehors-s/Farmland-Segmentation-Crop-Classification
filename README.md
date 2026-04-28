# 耕地分割与作物分类项目 (Farmland Segmentation & Crop Classification)

本项目包含两个独立的子项目：

| 子项目 | 目录 | 说明 |
|--------|------|------|
| 🛰️ **耕地提取** | [`cropland_extraction/`](cropland_extraction/) | 基于 CBAM U-Net 的耕地分割与矢量化 |
| 🌱 **地物分类** | [`crop_classification/`](crop_classification/) | 基于机器学习/深度学习的遥感作物分类 |

## 📁 目录结构

```
farm/
│
├── cropland_extraction/                # 项目一：耕地提取（U-Net 分割 + 矢量化）
│   ├── u-net--CBAMV7.py                # V7 多任务训练（分割+边界+距离）
│   ├── u-net--CBAMV8.py                # V8 多光谱训练（4通道）
│   ├── u-net--CBAMV8_4090.py           # V8 RTX 4090 优化版
│   ├── u-net矢量化V2.py                # 推理 + 矢量化（推荐，支持 --save_shp）
│   ├── u-net矢量化.py                   # 推理 + 矢量化（原始版，LabelMe JSON）
│   ├── labelme.py                      # LabelMe 标注工具启动器
│   ├── compare_models.py               # 模型对比工具
│   ├── run_pipeline.bat                # Windows 训练→矢量化全流程批处理
│   ├── requirements.txt                # 基础依赖
│   ├── requirements_4090.txt           # RTX 4090 环境依赖
│   ├── plan.md                         # 训练计划
│   ├── legacy/                         # 历史实验脚本（U-NET）
│   │   ├── u-net--CBAMV7.py
│   │   ├── u-net分割推理--cbam.py / --cbamV7.py
│   │   ├── u-net矢量化.py / _v2.py
│   │   ├── u-net掩膜坐标提取.py
│   │   ├── prepare_unet_patches.py     # 遥感影像切片预处理
│   │   ├── split_dataset_for_unet.py   # 数据集划分
│   │   ├── diagnose_mask_issue.py / damo.py
│   │   ├── best_model_config.yaml
│   │   └── training_curves.png
│   └── README.md
│
├── crop_classification/                # 项目二：地物分类（ML/DL 遥感分类）
│   ├── crop_segmentation/              # 核心分类模块（Python package）
│   │   ├── core/
│   │   │   ├── data_loader.py          # 地块特征提取（逐地块）
│   │   │   ├── data_loader_optimized.py# 分块并行提取（大影像）
│   │   │   ├── models_ml.py            # RF / SVM / XGBoost / LightGBM
│   │   │   └── models_dl.py            # 1D-CNN (PyTorch)
│   │   ├── interfaces/
│   │   │   ├── train_interface.py      # 训练入口 train_pipeline()
│   │   │   └── infer_interface.py      # 推理入口 predict_pipeline()
│   │   ├── utils/
│   │   │   └── geo_utils.py            # 坐标对齐、内存检测等工具
│   │   ├── test_models/                # 预训练模型 (.joblib)
│   │   ├── check_model_switch_smoke.py # 冒烟测试
│   │   ├── check_model_switch_realdata.py
│   │   ├── test_module_import.py       # 模块导入测试
│   │   ├── __init__.py
│   │   ├── README.md
│   │   └── README_parcel_extraction.md # 地块提取算法说明
│   ├── legacy/                         # 历史实验脚本
│   │   ├── 1D-CNN/                     # 早期 1D-CNN 模型
│   │   ├── LSTM/                       # 早期 LSTM 模型
│   │   └── check_duplicate_images.py
│   ├── requirements.txt                # 项目依赖
│   └── README.md
│
├── data/                               # 共享数据
│   ├── 完整作物边界/                     # 作物分布矢量
│   ├── 遥感图像/                         # 遥感影像
│   └── Sentinel2_*.tif / .ovr           # Sentinel-2 时序堆栈
│
├── dataset/                            # U-Net 训练集
│   ├── train/  val/                    # train/val 含 img/ + lbl/
│   ├── images/  masks/                 # 补充数据
│   └── audits/                         # 数据审核
│
├── requirements.txt                    # 全局依赖
├── README.md                           # 本文件（项目总览）
└── AGENTS.md                           # Agent 上下文说明
```

## 快速开始

### 耕地提取（U-Net）

```bash
# 训练
python cropland_extraction/u-net--CBAMV7.py \
    --data_root ./dataset --output_dir ./models

# 推理矢量化
python cropland_extraction/u-net矢量化V2.py \
    --model ./models/best_model.pth --input ./data --output ./results --save_shp
```

详细说明见 [`cropland_extraction/README.md`](cropland_extraction/README.md)。

### 地物分类

```bash
# 冒烟测试
python crop_classification/crop_segmentation/check_model_switch_smoke.py

# 训练（Python API）
python -c "
from crop_classification.crop_segmentation.interfaces.train_interface import train_pipeline
train_pipeline('rf', shp_path='data/train.shp', tif_path='data/image.tif', output_dir='models')
"
```

详细说明见 [`crop_classification/README.md`](crop_classification/README.md)。

## 环境依赖

每个子项目有独立的 `requirements.txt`：

```bash
# 仅耕地提取
pip install -r cropland_extraction/requirements.txt

# 仅地物分类
pip install -r crop_classification/requirements.txt

# 或安装全局依赖（包含两者）
pip install -r requirements.txt
```

推荐使用 Python 3.8+，GPU 训练需安装对应 CUDA 版本的 PyTorch。

## ⚠️ 注意事项

1.  **大文件忽略**: 模型权重 (`.pth`, `.pkl`)、数据集图片、预测结果等已被 `.gitignore` 忽略，不会进入版本控制。
2.  **路径配置**: 各脚本的默认路径可能需要根据实际环境调整，建议通过命令行参数指定。
3.  **数据格式**: U-Net 数据集要求 `train/img/*.png`, `train/lbl/*.png` 格式；作物分类要求 GeoTIFF + Shapefile。
