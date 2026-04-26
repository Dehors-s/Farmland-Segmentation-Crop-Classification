# 地块提取算法说明（Parcel Feature Extraction）

## 1. 目标

给定：
- 多波段遥感影像（GeoTIFF）
- 地块矢量（Shapefile）

输出：
- 每个地块的一条特征向量（当前实现为各波段像素均值）
- 训练时附带标签 `y`，推理时用于模型预测

本项目中地块提取的核心在 `core/data_loader.py` 与 `core/data_loader_optimized.py`。

## 2. 总体思路

算法采用“矢量裁膜 + 像素聚合”的思路：

1. 读取地块矢量和影像。
2. 统一坐标系（将 Shapefile 重投影到影像 CRS）。
3. 对每个地块几何执行 `rasterio.mask.mask`，提取地块内像素。
4. 把每个波段内有效像素（非 NaN）做统计聚合。
5. 当前默认聚合统计量：`每个波段的均值`。
6. 拼接成固定长度特征向量：`[band_1_mean, band_2_mean, ..., band_n_mean]`。

这是一种稳定、可解释、对样本规模友好的地块级特征工程方法。

## 3. 详细流程

### 3.1 坐标系对齐

- 输入地块 `gdf` 与影像 `src` CRS 不一致时，调用 `check_and_reproject(gdf, src.crs)`。
- 目的：保证几何和栅格处于同一空间参考下，避免提取偏移。

### 3.2 地块掩膜裁剪

对每个地块几何 `geom`：

- 使用 `rasterio.mask.mask(src, [geom], crop=True, nodata=np.nan)`。
- 返回地块覆盖范围对应的像素块 `out_image`（形状通常为 `bands x h x w`）。

### 3.3 有效像素筛选

- 把 `out_image` 重排成二维矩阵：`pixels.shape = (h*w, bands)`。
- 过滤任一波段为 `NaN` 的像素行。
- 若地块内无有效像素，跳过该地块（训练）或标记为 `unknown`（推理）。

### 3.4 波段统计聚合

- 对每个波段计算均值：`band_means = np.mean(valid_pixels, axis=0)`。
- 得到地块特征向量，长度等于影像波段数。

### 3.5 训练/推理衔接

- 训练：特征进入 `StandardScaler`、`LabelEncoder` 与模型训练流程。
- 推理：同样提取特征并标准化，再送入模型输出 `pred_id` 与 `pred_class`。

## 4. 伪代码

```python
for parcel in parcels:
    geom = parcel.geometry

    # 1) 地块掩膜提取
    out = mask(raster, geom, nodata=NaN)

    # 2) (bands, h, w) -> (h*w, bands)
    pixels = reshape_to_pixel_rows(out)

    # 3) 过滤无效像素
    valid_pixels = pixels[rows_without_nan]
    if len(valid_pixels) == 0:
        continue

    # 4) 按波段聚合
    feature = mean(valid_pixels, axis=0)

    save(feature)
```

## 5. 复杂度与资源开销

设：
- 地块数为 `P`
- 每个地块有效像素数平均为 `K`
- 波段数为 `B`

则单次提取总体近似为：
- 时间复杂度：`O(P * K * B)`
- 主要内存开销：由单地块裁剪块大小和波段数决定

在大影像场景下，瓶颈通常是 I/O 与掩膜操作次数，而非均值计算本身。

## 6. 当前实现的两种模式

### 6.1 常规模式（`data_loader.py`）

特点：
- 代码直接、稳定、易调试
- 逐地块提取，逻辑最清晰

适用：
- 中小规模影像或内存足够时

### 6.2 优化模式（`data_loader_optimized.py`）

特点：
- 先把影像按 `chunk_size` 分块
- 用空间相交筛出每块关联地块
- 多进程并行处理分块

适用：
- 超大影像与多核 CPU 环境

注意：
- 目前 `process_chunk` 内仍对整幅影像做 `mask`（未显式利用 `window` 限定读取范围），因此性能收益会受限。
- 同一地块若跨多个分块，当前实现可能重复提取同一地块特征，需在上层去重或在流程中增加地块唯一键控制。

## 7. 算法优点

- 可解释性强：每个特征直接对应某波段统计量。
- 与模型解耦：同一特征可喂给 RF/SVM/XGBoost/LGBM/CNN。
- 对多波段自适应：通过 `src.count` 动态确定特征维度。
- 对异常鲁棒：提取失败地块可跳过或标记为 `unknown`。

## 8. 局限与改进建议

当前局限：
- 只使用均值，忽略地块内纹理与分布信息。
- 边界混合像素（mixed pixels）可能稀释类别特征。
- 小地块在低分辨率影像下噪声较大。

建议升级方向：
- 增加统计量：`std / min / max / percentile / skewness`。
- 增加光谱指数：`NDVI, EVI, NDWI` 等。
- 增加时序特征：多期影像下的峰值、振幅、物候时间点。
- 引入几何特征：面积、周长、形状指数。
- 对优化模式增加“地块唯一去重 + 真正窗口化读取”。

## 9. 适用场景

- 地块级作物分类
- 地块经营属性识别
- 需要可解释、工程可落地的遥感特征提取前处理

## 10. 代码定位

- `core/data_loader.py`: 常规地块提取
- `core/data_loader_optimized.py`: 分块并行提取
- `utils/geo_utils.py`: 坐标对齐与内存策略
- `interfaces/train_interface.py`: 训练入口
- `interfaces/infer_interface.py`: 推理入口
