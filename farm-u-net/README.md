# farm-u-net 脚本说明

本目录包含三个主要脚本：
- [farm-u-net/u-net--CBAMV7.py](farm-u-net/u-net--CBAMV7.py)：多任务 CBAM U-Net 训练管线，输出分割、边界、距离三路预测。
- [farm-u-net/u-net矢量化.py](farm-u-net/u-net矢量化.py)：加载已训练模型完成推理、掩膜生成、轮廓矢量化（LabelMe JSON）与可视化。
- [farm-u-net/labelme.py](farm-u-net/labelme.py)：简易启动器，调用本机 `labelme` 可视化/标注工具，可指定默认打开目录。

## 环境准备
1. 推荐使用已有的 conda 环境（脚本会检测 GPU）。
2. 安装依赖：
   ```bash
   pip install -r farm-u-net/requirements.txt
   ```
3. 如需特定 CUDA 版本的 `torch/torchvision`，请按官方指引单独安装对应版本。

## 训练脚本：u-net--CBAMV7.py
多任务（分割 + 边界 + 距离回归）U-Net 训练，集成 CBAM 注意力、距离变换标签、边界监督、混合精度、学习率调度与早停。

### 关键接口
- `set_seed(seed=42)`: 设定随机种子，保证可复现。
- 模块类：`ChannelAttention`、`SpatialAttention`、`CBAM`、`CBAMDecoderBlock`、`CBAMUNet`、`MultiTaskUNet`（前向返回 `seg_logit, boundary_logit, distance_logit`）。
- 数据集：`FarmlandDataset(root_dir, split, transform, img_size, debug)`
  - 训练/验证：目录结构期望为 `root/train/img/*.png`、`root/train/lbl/*.png` 以及 `root/val/img`、`root/val/lbl`。
  - 测试：默认遍历 `root/test/<region>/*.png`，无标签时用全零掩膜。
  - `load_mask` 自动二值化并处理标签反转；`generate_distance_map` 计算距离变换。
- 数据增强：`get_transforms(img_size, phase)` 返回 `albumentations.Compose`，通过 `additional_targets` 同步处理 `distance_map`。
- 损失：`DiceBCELoss`、`FocalLoss`、`CombinedLoss`（分割）；边界使用 Dice+BCE；距离使用 L1。
- 训练器：`Trainer(config)`
  - `train_one_epoch(epoch)`：计算分割/边界/距离损失，支持梯度累积与 AMP。
  - `validate()`：IoU 评估（阈值 0.5）。
  - `save_model(filename)`：保存权重与训练曲线数据；`plot_training_curves()` 生成曲线图。
  - `run()`：整体训练循环，包含 warmup、ReduceLROnPlateau 调度、早停、定期 checkpoint。

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

脚本内部的 `config` 还包含：`img_size`(512)、`encoder_weights`(`imagenet`)、`weight_decay`、`boundary_weight`、`min_boundary_weight`、`use_cbam`、`use_amp`、`gradient_accumulation_steps`、`warmup_epochs`、`early_stopping_patience`、`save_interval`、`plot_interval`、`dropout_rate`、`resume_path` 等，可按需修改。

### 运行示例
```bash
python farm-u-net/u-net--CBAMV7.py \
  --data_root /path/to/dataset \
  --output_dir /path/to/output \
  --encoder_name resnet50 \
  --batch_size 8 \
  --epochs 80 \
  --lr 5e-4 \
  --num_workers 4
```
输出包括：`best_model.pth`、`final_model.pth`、中间 `checkpoint_epoch_*.pth`、训练曲线图 `training_curves.png`。

## 推理矢量化脚本：u-net矢量化.py
完成单图或批量推理、掩膜与边界导出、轮廓矢量化（LabelMe JSON）、可视化叠加，以及批量参数扫。

### 关键接口
- 模型结构与训练脚本保持一致：`MultiTaskUNet`（加载训练权重）。
- `Predictor(model_path, device, dropout_rate, encoder_name)`：
  - `predict(image_path, threshold, boundary_threshold)`：返回分割掩膜、边界掩膜、原图、概率图、距离图。
  - `process_and_save(...)`：保存分割/边界/最终掩膜、可选概率与距离图、叠加图，返回用于矢量化的最终掩膜。
- 矢量化工具：
  - `extract_contours_from_mask_array(mask, min_area, epsilon_factor, morph_kernel, morph_iter)`：生成轮廓点集。
  - `save_to_json(polygons, image_path, output_json_path, image_shape)`：保存为 LabelMe JSON。
  - `visualize_contours(image_img, polygons, output_vis_path)`：在原图上描边可视化。
- 批处理：`run_single_config(...)` 按一组阈值/形态学参数批量处理文件；支持参数扫（笛卡尔积组合）。

### 入口参数（命令行）
| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--model` | 模型权重路径 | 代码内 `default_model` 路径 |
| `--input` | 单张图片或文件夹 | 代码内 `default_input` |
| `--output` | 输出目录 | `./predictions_v7_vectorized` |
| `--threshold` | 统一阈值 | 0.5 |
| `--seg_threshold` | 分割阈值（空则用统一阈值） | None |
| `--boundary_threshold` | 边界阈值（空则 0.2*统一阈值） | None |
| `--dropout` | Dropout 率 | 0.2 |
| `--save_prob` | 保存概率/距离图 | False |
| `--boundary_erode_iter` | 边界腐蚀迭代（扣边前） | 0 |
| `--min_area` | 轮廓最小面积过滤 | 50 |
| `--epsilon` | 轮廓简化系数 | 0.001 |
| `--morph_kernel` | 形态学核大小（0 关闭） | 0 |
| `--morph_iter` | 形态学迭代次数 | 1 |
| `--sweep` | 启用参数扫 | False |
| `--sweep_boundary_thresholds` | 边界阈值列表 | "0.10,0.12,0.15" |
| `--sweep_boundary_erode_iters` | 边界腐蚀次数列表 | "0,1" |
| `--sweep_morph_kernels` | 形态学核大小列表 | "0,3" |
| `--sweep_epsilons` | 轮廓简化系数列表 | "0.004,0.006" |
| `--encoder_name` | 骨干网络（需与权重一致） | `resnet50` |

### 运行示例
单组参数：
```bash
python farm-u-net/u-net矢量化.py \
  --model /path/to/best_model.pth \
  --input /path/to/images_folder \
  --output ./predictions_v7_vectorized \
  --threshold 0.45 \
  --boundary_threshold 0.10 \
  --boundary_erode_iter 1 \
  --morph_kernel 3 \
  --epsilon 0.004
```
参数扫批处理：
```bash
python farm-u-net/u-net矢量化.py \
  --model /path/to/best_model.pth \
  --input /path/to/images_folder \
  --output ./predictions_v7_vectorized \
  --sweep \
  --sweep_boundary_thresholds "0.08,0.10,0.12" \
  --sweep_boundary_erode_iters "0,1" \
  --sweep_morph_kernels "0,3" \
  --sweep_epsilons "0.003,0.005"
```
输出目录结构：
- 非 sweep：`masks/`（掩膜）`json/`（LabelMe）`visualization/`（叠加可视化）。
- sweep：每组配置一个子目录 `run_xx_*`，同样包含上述三类子目录。

## LabelMe 启动脚本：labelme.py
为本机已安装的 `labelme` 提供便捷启动入口，可选指定默认图像目录。

### 关键接口
- `launch_labelme(directory=None)`：
  - 若提供 `directory` 且存在，启动时直接打开该目录。
  - 若未安装 `labelme`，会提示安装命令。

### 入口参数（命令行）
- 直接运行：`python farm-u-net/labelme.py [image_dir]`
  - 若给出 `image_dir`，优先使用该目录。
  - 未提供参数时，脚本尝试使用内部配置的 `DEFAULT_IMAGE_DIR`，不存在则启动默认 labelme。

### 运行示例
```bash
python farm-u-net/labelme.py ./dataset/images
```

## 可选批处理：run_pipeline.bat
[farm-u-net/run_pipeline.bat](farm-u-net/run_pipeline.bat) 提供 Windows 批处理流水线：硬件侦测 -> 自适应训练参数 -> 训练 -> 矢量化推理。按脚本内顶部注释修改路径和超参数即可。

## 小贴士
- 训练/推理前请检查 `encoder_name` 与权重匹配；推理脚本默认不加载 ImageNet 预训练，需要与训练时一致。
- 数据集掩膜若存在反转或灰阶边缘，`FarmlandDataset.load_mask` 的二值化逻辑会自动处理。
- 若显存紧张，可下调 `img_size`、`batch_size` 或开启梯度累积；如需更高吞吐，可调高 `num_workers` 并保留 `pin_memory=True`。
