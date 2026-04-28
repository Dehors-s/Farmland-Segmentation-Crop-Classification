# 耕地提取 — U-Net 分割与矢量化 (Cropland Extraction)

基于 CBAM U-Net 的耕地分割与矢量化管道。支持多任务学习（分割 + 边界检测 + 距离变换），输出分割掩膜与 LabelMe JSON 矢量轮廓。

## 📁 目录结构

```
cropland_extraction/
│
├── u-net--CBAMV7.py            # V7 多任务训练脚本（分割+边界+距离）
├── u-net--CBAMV8.py            # V8 多光谱训练脚本（4通道输入）
├── u-net--CBAMV8_4090.py       # V8 RTX 4090 优化版（大batch）
├── u-net矢量化.py               # 推理 + 矢量化（LabelMe JSON）
├── u-net矢量化V2.py             # V2 推理矢量化脚本（推荐，支持 --save_shp）
├── labelme.py                  # LabelMe 标注工具启动器
├── compare_models.py           # 模型对比工具
├── run_pipeline.bat            # Windows 批处理：训练→矢量化全流程
│
├── legacy/                     # 历史实验脚本（仅供参考）
│   ├── u-net--CBAMV7.py        # 早期 V7 版本
│   ├── u-net分割推理--cbam.py
│   ├── u-net分割推理--cbamV7.py
│   ├── u-net矢量化_v2.py
│   ├── u-net矢量化.py
│   ├── u-net掩膜坐标提取.py
│   ├── best_model_config.yaml
│   ├── damo.py
│   ├── diagnose_mask_issue.py
│   ├── prepare_unet_patches.py  # 遥感影像切片预处理
│   ├── split_dataset_for_unet.py
│   └── training_curves.png
│
├── requirements.txt            # 基础依赖
├── requirements_4090.txt       # RTX 4090 环境依赖
├── plan.md
└── README.md
```

## 环境准备

```bash
pip install -r cropland_extraction/requirements.txt
```

如需特定 CUDA 版本的 PyTorch，请按 [pytorch.org](https://pytorch.org) 指引单独安装。

## 训练脚本：u-net--CBAMV7.py

多任务（分割 + 边界 + 距离回归）U-Net 训练，集成 CBAM 注意力、距离变换标签、边界监督、混合精度、学习率调度与早停。

### 关键接口
- `set_seed(seed=42)`: 设定随机种子，保证可复现。
- 模块类：`ChannelAttention`、`SpatialAttention`、`CBAM`、`CBAMDecoderBlock`、`CBAMUNet`、`MultiTaskUNet`（前向返回 `seg_logit, boundary_logit, distance_logit`）。
- 数据集：`FarmlandDataset(root_dir, split, transform, img_size, debug)`
  - 训练/验证：目录结构期望为 `root/train/img/*.png`、`root/train/lbl/*.png` 以及 `root/val/img`、`root/val/lbl`。
  - 测试：默认遍历 `root/test/<region>/*.png`，无标签时用全零掩膜。
- 损失：`DiceBCELoss`、`FocalLoss`、`CombinedLoss`（分割）；边界使用 Dice+BCE；距离使用 L1。
- 训练器：`Trainer(config)` 支持混合精度 AMP、梯度累积、warmup、ReduceLROnPlateau、早停。

### 入口参数（命令行）
| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--data_root` | 数据集根目录（必填） | - |
| `--output_dir` | 输出目录 | - |
| `--encoder_name` | 骨干网络（resnet18/34/50） | `resnet50` |
| `--batch_size` | 批大小 | 16 |
| `--epochs` | 轮数 | 50 |
| `--lr` | 学习率 | 1e-4 |
| `--num_workers` | DataLoader 线程数 | 0 |

### 运行示例
```bash
python cropland_extraction/u-net--CBAMV7.py \
  --data_root /path/to/dataset \
  --output_dir /path/to/output \
  --encoder_name resnet50 \
  --batch_size 8 \
  --epochs 80 \
  --lr 5e-4 \
  --num_workers 4
```

输出包括：`best_model.pth`、`final_model.pth`、中间 `checkpoint_epoch_*.pth`、训练曲线图 `training_curves.png`。

## 推理矢量化脚本：u-net矢量化V2.py

完成单图或批量推理、掩膜导出、轮廓矢量化（LabelMe JSON / SHP）、可视化叠加。

### 入口参数（命令行）
| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--model` | 模型权重路径 | 代码内默认路径 |
| `--input` | 单张图片或文件夹 | 代码内默认路径 |
| `--output` | 输出目录 | `./predictions_v7_vectorized` |
| `--threshold` | 分割阈值 | 0.5 |
| `--in_channels` | 输入通道数（V8 模型用 4） | 3 |
| `--save_shp` | 保存为 SHP（ArcGIS 兼容） | False |

### 运行示例
```bash
python cropland_extraction/u-net矢量化V2.py \
  --model /path/to/best_model.pth \
  --input /path/to/images \
  --output ./results \
  --in_channels 4 \
  --save_shp
```

## run_pipeline.bat

Windows 批处理流水线：硬件侦测 → 自适应训练参数 → 训练 → 矢量化推理。
编辑脚本内的路径配置后直接运行：
```bash
cropland_extraction\run_pipeline.bat
```

## 主要特性

- **多任务学习**: 分割 + 边界检测 + 距离变换联合监督
- **CBAM 注意力**: 通道-空间注意力机制提升分割精度
- **混合精度训练**: 支持 AMP 加速
- **早停与学习率调度**: ReduceLROnPlateau + EarlyStopping
- **矢量化输出**: LabelMe JSON / SHP（ArcGIS 兼容）
- **参数扫描**: 批量测试不同阈值/形态学参数组合

## 依赖

见 `requirements.txt`。核心依赖：PyTorch, torchvision, opencv-python, albumentations, matplotlib, tqdm, numpy
