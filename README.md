# 耕地分割与作物分类项目 (Farmland Segmentation & Crop Classification)

本项目旨在利用深度学习与机器学习技术，实现高精度的耕地提取（分割）与作物分类。项目包含图像分割（U-Net）和基于时序特征的作物分类（RF/CNN/LSTM）两大模块。

## 📁 目录结构

```
耕地分割/
├── U-NET/                      # 图像分割模块
│   ├── u-net--CBAMV7.py        # V7模型训练脚本 (MultiTask: 分割+边界+距离)
│   ├── u-net分割推理--cbamV7.py # V7模型推理脚本 (支持批量预测、结果可视化)
│   └── best_model_config.yaml  # 训练配置文件
├── RF-feature/                 # 随机森林分类模块
│   ├── RF-feature.py           # 随机森林训练与评估
│   └── predict_shp.py          # 结合SHP文件的预测脚本
├── 1D-CNN/                     # 一维卷积神经网络分类模块
│   └── 1D-CNN.py               # 1D-CNN 训练脚本
├── LSTM/                       # 长短期记忆网络分类模块
│   └── LSTM.py                 # LSTM 训练脚本
├── data/                       # 数据目录 (需自行准备)
└── requirements.txt            # 项目依赖列表
```

## 🛠️ 环境依赖

推荐使用 Python 3.8+ 环境。

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```
   *注意: 如果使用 GPU 训练，请确保安装了与 CUDA 版本匹配的 `torch` 和 `torchvision`。*

## 🚀 功能与使用说明

### 1. 耕地分割 (U-NET)

该模块使用改进的 U-Net (集成 CBAM 注意力机制) 进行耕地提取。V7 版本采用了多任务学习架构 (分割 + 边界检测 + 距离变换)，能有效提升分割精度并改善边缘细节。

#### 1.1 模型训练
使用 `u-net--CBAMV7.py` 进行训练。
*   需在脚本中配置训练数据路径。
*   支持自动保存最佳模型权重 (`.pth`)。

#### 1.2 模型推理
使用 `u-net分割推理--cbamV7.py` 对新图片进行预测。

**基本用法**:
```bash
# 命令行运行
python U-NET/u-net分割推理--cbamV7.py --model ./U-NET/best_model.pth --input ./data/test_images --output ./predictions_v7
```

**参数说明**:
*   `--model`: 模型权重路径 (.pth)
*   `--input`: 输入图片或文件夹路径
*   `--output`: 结果保存目录
*   `--threshold`: 分割阈值 (默认 0.5)
*   `--save_prob`: 是否保存概率图

**输出结果**:
*   `*_mask.png`: 最终分割掩膜 (已扣除边界)
*   `*_overlay.jpg`: 可视化叠加图 (红色=耕地, 绿色=边界)

---

### 2. 作物分类 (RF / CNN / LSTM)

该模块基于多时相遥感特征 (CSV数据) 对作物类型进行分类 (如棉花、玉米、小麦等)。

#### 2.1 数据准备
需要准备 CSV 格式的训练数据 (例如 `Training_Data_for_Python_v4.csv`)，包含各类植被指数 (NDVI, GCVI 等) 的时序特征。

#### 2.2 随机森林 (Random Forest)
运行 `RF-feature/RF-feature.py`:
*   自动进行特征重要性分析。
*   执行网格搜索 (Grid Search) 寻找最佳超参数。
*   输出分类报告与混淆矩阵。
*   保存模型为 `.pkl` 文件。

#### 2.3 深度学习分类 (1D-CNN / LSTM)
*   **1D-CNN**: 运行 `1D-CNN/1D-CNN.py`，利用卷积提取时序特征。
*   **LSTM**: 运行 `LSTM/LSTM.py`，利用循环神经网络捕捉长时间依赖关系。
*   这两个脚本均包含完整的数据预处理、模型定义、训练循环及评估代码。

## ⚠️ 注意事项

1.  **大文件忽略**: 项目根目录下的 `.gitignore` 已配置忽略所有模型权重 (`.pth`, `.pkl`)、数据集图片及预测结果目录，提交代码时请注意检查。
2.  **路径配置**: 各脚本中的默认路径可能需要根据您的实际环境进行微调 (建议通过命令行参数指定路径，或修改脚本顶部的配置区域)。
