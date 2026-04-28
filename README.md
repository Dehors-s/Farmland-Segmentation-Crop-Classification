# 耕地分割与作物分类项目 (Farmland Segmentation & Crop Classification)

本项目包含两个独立的子项目：

| 子项目 | 目录 | 说明 |
|--------|------|------|
| 🛰️ **耕地提取** | [`cropland_extraction/`](cropland_extraction/) | 基于 CBAM U-Net 的耕地分割与矢量化 |
| 🌱 **地物分类** | [`crop_classification/`](crop_classification/) | 基于机器学习/深度学习的遥感作物分类 |

## 📁 目录结构

```
farm/
├── cropland_extraction/        # 耕地提取（U-Net 分割 + 矢量化）
│   ├── u-net--CBAMV7.py        # V7 多任务训练脚本
│   ├── u-net--CBAMV8.py        # V8 多光谱训练脚本
│   ├── u-net--CBAMV8_4090.py   # V8 4090 优化版
│   ├── u-net矢量化V2.py        # 推理 + 矢量化（推荐）
│   ├── legacy/                 # 历史实验脚本
│   └── README.md               # 详细使用说明
│
├── crop_classification/        # 地物分类（RF / SVM / XGBoost / LGBM / CNN）
│   ├── crop_segmentation/      # 核心 Python package
│   │   ├── core/               # 数据加载与模型定义
│   │   ├── interfaces/         # 训练/推理接口
│   │   └── utils/              # 地理空间工具
│   ├── legacy/                 # 历史脚本（1D-CNN, LSTM）
│   └── README.md               # 详细使用说明
│
├── data/                       # 共享数据
├── dataset/                    # 训练数据集
├── models/                     # 训练好的模型
├── results/                    # 预测结果
│
├── Wiki/                       # 项目文档
├── 过程文件/                   # 历史实验文件（归档）
├── paper/ paper_md/ papers_md/ # 论文参考文献
│
├── requirements.txt            # 全局依赖（可选）
├── README.md                   # 本文件（总览）
└── AGENTS.md                   # Agent 上下文说明
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
