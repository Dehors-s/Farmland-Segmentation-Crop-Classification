# 推理接口

## 模块功能
加载训练好的模型，对新的地块进行分类预测，处理内存管理和坐标转换，并将结果保存为新的Shapefile。

## 核心逻辑/算法

### 工作原理
推理接口负责模型的实际应用，将训练好的模型用于新数据的预测：

1. **模型加载**：加载完整的模型打包，包括模型本身、标准化器、标签编码器等
2. **数据准备**：读取矢量数据并检查坐标系
3. **内存管理**：根据影像大小决定是否将影像加载到内存
4. **空间过滤**：使用空间索引过滤出影像范围内的地块
5. **特征提取**：为每个地块提取特征
6. **模型预测**：使用加载的模型进行预测
7. **结果处理**：解码预测结果并处理无效预测
8. **结果保存**：将预测结果保存为新的Shapefile

### 核心流程
1. 加载模型打包，获取模型及其相关组件
2. 读取矢量数据并检查坐标系
3. 打开影像并根据内存情况决定处理模式
4. 空间过滤出影像范围内的地块
5. 为每个地块提取特征并进行预测
6. 解码预测结果，处理无效值
7. 打印标签映射关系，方便用户理解
8. 将预测结果保存为新的Shapefile

## 代码高光时刻

### 1. 模型加载和初始化
```python
# 1. 加载模型打包
print("📦 加载模型打包...")
model_bundle = joblib.load(model_path)
model = model_bundle['model']
model_type = model_bundle['model_type']
scaler = model_bundle['scaler']
le = model_bundle['label_encoder']

# 获取训练时使用的标签列名
label_column = model_bundle.get('label_column', 'value')  # 默认使用'value'列

print(f"   模型类型: {model_type}")
print(f"   特征数: {model_bundle['num_features']}")
print(f"   类别数: {model_bundle['num_classes']}")
print(f"   训练时使用的标签列: '{label_column}'")
```

**精妙之处**：
- 加载完整的模型打包，确保所有必要组件都可用
- 优雅地处理训练时使用的标签列名，提供默认值
- 详细打印模型信息，方便用户确认加载的模型
- 使用joblib高效加载大型模型

### 2. 内存管理和处理模式选择
```python
# 检查内存是否足够加载整个影像
load_in_memory = should_load_in_memory(src, force_memory)

if load_in_memory:
    # 读取完整影像到内存
    print("   读取完整影像到内存...")
    image_data = src.read()  # 形状为 (bands, height, width)
    transform = src.transform
    print(f"   影像加载完成: 形状={image_data.shape}")
else:
    # 内存不足，不加载到内存中
    print("   内存不足，使用流式处理...")
    image_data = None
    transform = src.transform
```

**精妙之处**：
- 使用与训练接口相同的内存检测逻辑，确保一致性
- 根据内存情况自动选择处理模式
- 清晰的状态打印，让用户了解当前的处理模式
- 为流式处理和内存模式准备不同的变量

### 3. 空间过滤和预测循环
```python
# 4. 空间索引过滤
print("🔍 空间索引过滤...")
# 使用cx索引过滤
filtered_gdf = gdf.cx[src_bounds.left:src_bounds.right, src_bounds.bottom:src_bounds.top]
print(f"   过滤后剩余 {len(filtered_gdf)} 个地块")

# 5. 准备预测结果
predictions = []
prediction_probs = []

# 6. 推理循环
print("🚀 开始推理...")
print(f"   处理模式: {'内存模式' if load_in_memory else '流式处理模式'}")
print(f"   工作进程数: {max_workers}, 分块大小: {chunk_size}")

for idx, row in tqdm(filtered_gdf.iterrows(), total=len(filtered_gdf), desc="Inference"):
    geom = row.geometry
    
    try:
        # 提取特征
        features = extract_features_from_image(src, geom)
        
        if features is not None and len(features) > 0:
            # 标准化特征
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # 预测
            if model_type == 'cnn':
                # 深度学习模型
                pred = model.predict(features_scaled)[0]
                pred_prob = model.predict_proba(features_scaled)[0]
            else:
                # 机器学习模型
                pred = model.predict(features_scaled)[0]
                pred_prob = model.predict_proba(features_scaled)[0]
            
            # 保存预测结果
            predictions.append(pred)
            prediction_probs.append(pred_prob)
        else:
            # 无法提取特征的地块
            predictions.append(-1)
            prediction_probs.append(np.zeros(model_bundle['num_classes']))
            
    except Exception as e:
        # 处理失败的地块
        predictions.append(-1)
        prediction_probs.append(np.zeros(model_bundle['num_classes']))
        continue
```

**精妙之处**：
- 使用GeoPandas的cx索引快速过滤空间范围内的地块
- 使用tqdm提供实时的推理进度显示
- 统一处理机器学习和深度学习模型的预测接口
- 异常处理确保单个地块处理失败不影响整体流程
- 为无法提取特征的地块标记为-1，确保结果的完整性

### 4. 结果解码和标签映射
```python
# 7. 解码预测结果
print("🔤 解码预测结果...")
# 只对有效预测进行解码
valid_predictions = []
for pred in predictions:
    if pred == -1:
        valid_predictions.append('unknown')
    else:
        try:
            valid_predictions.append(le.inverse_transform([pred])[0])
        except:
            valid_predictions.append('unknown')

# 打印标签映射关系
print("\n📋 标签映射关系:")
print("-" * 80)
print(f"{'编码值 (pred_id)':<20} {'原始标签 (value列)':<30} {'预测结果 (pred_class)':<30}")
print("-" * 80)

# 打印所有编码值及其对应的原始标签
for i, label in enumerate(le.classes_):
    print(f"{i:<20} {str(label):<30} {str(label):<30}")
print(f"{-1:<20} {'N/A':<30} {'unknown':<30}")
print("-" * 80)
```

**精妙之处**：
- 安全地解码预测结果，处理无效预测
- 详细打印标签映射关系，方便用户理解预测结果
- 清晰的表格格式，提升可读性
- 包含-1的特殊情况处理，确保映射关系的完整性

### 5. 结果保存
```python
# 8. 保存结果
print("💾 保存结果...")
# 确保输出目录存在
output_dir = os.path.dirname(output_shp)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# 添加预测结果到GeoDataFrame（使用短字段名，避免Shapefile长度限制）
filtered_gdf['pred_class'] = valid_predictions
filtered_gdf['pred_id'] = predictions

# 保存到Shapefile
filtered_gdf.to_file(output_shp, encoding='utf-8')
print(f"   结果已保存至: {output_shp}")

# 9. 统计信息
print("📊 预测统计:")
valid_count = sum(1 for p in predictions if p != -1)
invalid_count = sum(1 for p in predictions if p == -1)
print(f"   有效预测: {valid_count}")
print(f"   无效预测: {invalid_count}")
```

**精妙之处**：
- 自动创建输出目录，确保保存路径的有效性
- 使用短字段名（'pred_class'和'pred_id'），避免Shapefile字段名长度限制
- 保存完整的预测信息，包括原始编码和解码后的类别
- 提供详细的预测统计信息，方便用户了解预测质量
- 使用utf-8编码，支持中文等非ASCII字符

## 使用示例

### 基本用法
```python
from crop_segmentation.interfaces.infer_interface import predict_pipeline

# 进行预测
output_shp = predict_pipeline(
    model_path="output/models/rf_model_bundle.joblib",  # 模型文件路径
    tif_path="path/to/image.tif",  # 影像文件路径
    shp_path="path/to/parcels.shp",  # 矢量文件路径
    output_shp="output/predictions/parcels_classified.shp"  # 输出Shapefile路径
)

print(f"预测完成，结果保存至: {output_shp}")
```

### 处理大影像
```python
# 处理大影像，使用流式处理
predict_pipeline(
    model_path="output/models/xgboost_model_bundle.joblib",
    tif_path="path/to/large_image.tif",
    shp_path="path/to/parcels.shp",
    output_shp="output/predictions/parcels_classified.shp",
    force_memory=False  # 强制使用流式处理
)
```

### 批处理多个文件
```python
import os
from crop_segmentation.interfaces.infer_interface import predict_pipeline

# 批处理多个影像文件
model_path = "output/models/rf_model_bundle.joblib"
shp_path = "path/to/parcels.shp"
output_dir = "output/predictions"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 处理多个影像
tif_files = ["image1.tif", "image2.tif", "image3.tif"]
for tif_file in tif_files:
    tif_path = os.path.join("path/to/images", tif_file)
    output_shp = os.path.join(output_dir, f"{os.path.splitext(tif_file)[0]}_classified.shp")
    
    print(f"处理文件: {tif_file}")
    predict_pipeline(
        model_path=model_path,
        tif_path=tif_path,
        shp_path=shp_path,
        output_shp=output_shp
    )
    print(f"完成处理: {tif_file}\n")
```

## 技术要点

- **模型兼容性**：支持加载和使用不同类型的模型（机器学习和深度学习）
- **内存管理**：智能检测内存情况，选择合适的处理模式
- **空间效率**：使用空间索引过滤，只处理影像范围内的地块
- **错误处理**：健壮的异常处理，确保处理过程不中断
- **结果完整性**：为所有地块提供预测结果，包括无效预测的标记
- **用户友好**：详细的状态信息和结果统计，提升用户体验
- **格式兼容性**：使用标准的Shapefile格式，确保与其他GIS软件的兼容性

## 代码优化建议

1. **并行预测**：为大规模预测添加并行处理支持，提高处理速度
2. **批量预测**：实现批量特征提取和预测，减少模型调用开销
3. **缓存机制**：添加特征提取缓存，避免重复处理相同的地块
4. **多线程处理**：使用多线程处理I/O密集型操作，如文件读写
5. **进度保存**：添加中间结果保存功能，支持中断后恢复
6. **预测概率阈值**：支持设置预测概率阈值，过滤低置信度的预测
7. **多模型集成**：支持加载多个模型进行集成预测，提高准确性
8. **结果可视化**：添加简单的结果可视化功能，快速查看预测效果
9. **日志记录**：添加详细的日志记录，方便调试和问题排查
10. **命令行接口**：提供命令行接口，支持脚本化调用