# 训练接口

## 模块功能
提供统一的训练流水线，支持多种机器学习和深度学习模型，集成数据加载、特征处理、模型训练、超参数搜索和评估功能。

## 核心逻辑/算法

### 工作原理
训练接口采用模块化设计，将整个训练过程分解为多个步骤：

1. **数据加载**：根据影像大小自动选择合适的数据加载方式（内存模式或分块模式）
2. **数据预处理**：标签编码和特征标准化
3. **模型训练**：支持多种模型类型，包括机器学习和深度学习模型
4. **超参数搜索**：使用Grid Search优化模型参数
5. **模型评估**：在验证集上评估模型性能
6. **模型打包**：保存模型及其相关组件（标准化器、标签编码器等）

### 核心流程
1. 根据模型类型和参数准备训练环境
2. 加载数据（自动选择合适的加载器）
3. 根据eval_mode参数决定是否划分验证集
4. 对数据进行预处理（标签编码和特征标准化）
5. 训练模型（支持超参数搜索）
6. 在验证集上评估模型（如果开启评估模式）
7. 打包并保存模型

## 代码高光时刻

### 1. 智能数据加载选择
```python
if use_optimized_loader:
    # 使用优化的数据加载器（适用于大尺寸影像）
    chunk_size = kwargs.pop('chunk_size', 1024)
    max_workers = kwargs.pop('max_workers', 4)
    print(f"   使用优化数据加载器: 分块大小={chunk_size}, 工作进程数={max_workers}")
    X, y, feature_names = load_data_optimized(
        shp_path, tif_path, label_column, 
        chunk_size=chunk_size, 
        max_workers=max_workers
    )
else:
    # 检查内存是否足够加载整个影像
    import rasterio
    with rasterio.open(tif_path) as src:
        load_in_memory = should_load_in_memory(src, force_memory)
    
    if load_in_memory:
        # 使用原始数据加载器
        print("   使用原始数据加载器 (内存模式)")
        X, y, feature_names = load_data(shp_path, tif_path, label_column)
        print(f"   加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
    else:
        # 内存不足，使用分块处理模式
        print("   内存不足，使用分块处理模式")
        chunk_size = kwargs.pop('chunk_size', 1024)
        max_workers = kwargs.pop('max_workers', 4)
        print(f"   分块大小: {chunk_size}, 工作进程数: {max_workers}")
        # 使用优化数据加载器进行分块处理
        X, y, feature_names = load_data_optimized(
            shp_path, tif_path, label_column, 
            chunk_size=chunk_size, 
            max_workers=max_workers
        )
```

**精妙之处**：
- 提供多种数据加载选项，满足不同场景需求
- 自动检测内存情况，选择合适的加载模式
- 当内存不足时，自动切换到分块处理模式
- 支持用户手动指定是否使用优化加载器
- 从kwargs中提取分块相关参数，保持API一致性

### 2. 评估模式实现
```python
# 根据eval_mode参数决定是否划分验证集
if eval_mode:
    print("🔄 开启评估模式，划分训练集和验证集...")
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
    
    # 2. 标签编码（仅使用训练集）
    print("🔤 标签编码...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    print(f"   类别数: {len(le.classes_)}")
    
    # 3. 特征标准化（仅使用训练集）
    print("📊 特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 训练模型...
    
    # 5. 模型评估
    print("📊 模型评估...")
    
    # 评估模型
    y_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val_encoded, y_pred)
    print(f"   验证集准确度: {acc:.4f}")
    
    # 打印分类报告
    print("\n📑 分类报告:")
    # 获取类别名称
    class_names = le.classes_.astype(str)
    print(classification_report(y_val_encoded, y_pred, target_names=class_names))
```

**精妙之处**：
- 支持评估模式和非评估模式的灵活切换
- 使用stratify参数确保训练集和验证集的类别分布一致
- 只在训练集上拟合标签编码器和标准化器，避免数据泄露
- 详细的评估指标，包括准确度和分类报告
- 清晰的状态打印，提升用户体验

### 3. 超参数搜索实现
```python
if grid_search:
    # 使用Grid Search进行超参数搜索
    print("   使用Grid Search进行超参数搜索...")
    
    # 获取模型的参数网格
    if model_type not in param_grids:
        raise ValueError(f"模型类型 {model_type} 不支持Grid Search")
    
    param_grid = param_grids[model_type]
    print(f"   搜索参数网格: {param_grid}")
    
    # 创建基础模型
    base_model = MLModelFactory.create_model(model_type)
    
    # 创建GridSearchCV
    grid_search_cv = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # 执行搜索
    grid_search_cv.fit(X_train_scaled, y_train_encoded)
    
    # 获取最佳模型
    model = grid_search_cv.best_estimator_
    print(f"   最佳参数: {grid_search_cv.best_params_}")
    print(f"   最佳交叉验证分数: {grid_search_cv.best_score_:.4f}")
```

**精妙之处**：
- 为每种模型类型定义了合理的参数网格
- 使用MLModelFactory创建基础模型，保持代码一致性
- 配置GridSearchCV使用3折交叉验证和准确率评分
- 使用n_jobs=-1充分利用多核CPU加速搜索
- 自动获取并使用最佳模型和参数
- 清晰的搜索结果打印

### 4. 模型打包保存
```python
# 5. 保存模型打包
print("💾 保存模型打包...")
model_bundle = {
    'model': model,
    'model_type': model_type,
    'scaler': scaler,
    'label_encoder': le,
    'feature_names': feature_names,
    'num_features': len(feature_names),
    'num_classes': len(le.classes_),
    'label_column': label_column  # 保存训练时使用的标签列名
}

# 生成保存路径
model_filename = f'{model_type}_model_bundle.joblib'
model_path = os.path.join(output_dir, model_filename)

# 保存
joblib.dump(model_bundle, model_path)
print(f"   模型已保存至: {model_path}")
```

**精妙之处**：
- 完整的模型打包，包含模型及其所有依赖组件
- 保存模型类型、特征名称、类别数量等元数据
- 保存训练时使用的标签列名，确保推理时的一致性
- 使用joblib高效保存大型模型
- 清晰的保存路径生成和状态打印

## 使用示例

### 基本用法
```python
from crop_segmentation.interfaces.train_interface import train_pipeline

# 训练随机森林模型
train_pipeline(
    model_type="rf",  # 模型类型
    shp_path="path/to/parcels.shp",  # 矢量文件路径
    tif_path="path/to/image.tif",  # 影像文件路径
    output_dir="output/models",  # 输出目录
    label_column="class_id",  # 标签列名
    eval_mode=True  # 开启评估模式
)
```

### 使用深度学习模型
```python
# 训练CNN模型
train_pipeline(
    model_type="cnn",  # 深度学习模型
    shp_path="path/to/parcels.shp",
    tif_path="path/to/image.tif",
    output_dir="output/models",
    label_column="class_id",
    eval_mode=True,
    # CNN特定参数
    epochs=100,  # 训练轮数
    batch_size=32,  # 批量大小
    learning_rate=0.001  # 学习率
)
```

### 使用超参数搜索
```python
# 使用Grid Search训练XGBoost模型
train_pipeline(
    model_type="xgboost",
    shp_path="path/to/parcels.shp",
    tif_path="path/to/image.tif",
    output_dir="output/models",
    label_column="class_id",
    eval_mode=True,
    grid_search=True  # 开启超参数搜索
)
```

### 处理大影像
```python
# 处理大尺寸影像
train_pipeline(
    model_type="lgbm",
    shp_path="path/to/parcels.shp",
    tif_path="path/to/large_image.tif",
    output_dir="output/models",
    label_column="class_id",
    eval_mode=True,
    use_optimized_loader=True,  # 使用优化数据加载器
    chunk_size=2048,  # 分块大小
    max_workers=8  # 并行进程数
)
```

## 支持的模型类型

| 模型类型 | 描述 | 适用场景 |
|---------|------|----------|
| `rf` | 随机森林 | 通用分类，速度快，效果好 |
| `svm` | 支持向量机 | 小样本，高维特征 |
| `xgboost` | XGBoost | 高精度，适合复杂数据 |
| `lgbm` | LightGBM | 速度快，内存占用低 |
| `cnn` | 卷积神经网络 | 深度学习，自动特征提取 |

## 技术要点

- **统一接口**：为所有模型类型提供一致的训练接口
- **智能内存管理**：自动检测并适应内存情况
- **灵活的评估模式**：支持带验证集的评估和全数据集训练
- **高效的超参数搜索**：使用并行计算加速Grid Search
- **完整的模型打包**：保存所有必要的组件，确保推理时的一致性
- **详细的状态反馈**：清晰的打印信息，方便用户监控训练过程

## 代码优化建议

1. **参数验证**：添加更严格的参数验证，确保输入参数的有效性
2. **模型选择策略**：根据数据规模和特征维度自动推荐合适的模型类型
3. **学习率调度**：为深度学习模型添加学习率调度器
4. **早停机制**：添加早停功能，避免过拟合
5. **模型集成**：支持模型集成（ensemble），提高预测准确性
6. **分布式训练**：为深度学习模型添加分布式训练支持
7. **更丰富的评估指标**：添加F1-score、AUC等更多评估指标
8. **参数网格优化**：为不同模型类型提供更精细的参数网格