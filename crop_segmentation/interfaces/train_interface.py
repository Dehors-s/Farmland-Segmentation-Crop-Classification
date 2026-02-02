import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from ..core.data_loader import load_data
from ..core.data_loader_optimized import load_data_optimized
from ..core.models_dl import CNNTrainer
from ..utils.geo_utils import should_load_in_memory

# 尝试导入MLModelFactory
try:
    from ..core.models_ml import MLModelFactory
    ML_MODEL_AVAILABLE = True
except ImportError as e:
    ML_MODEL_AVAILABLE = False
    print(f"⚠️ 机器学习模型导入失败: {e}")
    print("   仅支持深度学习模型 'cnn'")


def train_pipeline(model_type, shp_path, tif_path, output_dir, label_column='class_id', use_optimized_loader=False, force_memory=False, grid_search=False, eval_mode=True, **kwargs):
    """
    训练流水线
    
    Args:
        model_type (str): 模型类型，支持 'rf', 'svm', 'xgboost', 'lgbm', 'cnn'
        shp_path (str): Shapefile路径
        tif_path (str): GeoTIFF路径
        output_dir (str): 输出目录
        label_column (str): 标签列名
        use_optimized_loader (bool): 是否使用优化的数据加载器（适用于大尺寸影像）
        force_memory (bool): 是否强制将影像加载到内存
        grid_search (bool): 是否使用超参数搜索
        eval_mode (bool): 是否开启评估模式，开启后会划分验证集并评估模型
        **kwargs: 模型参数
    
    Returns:
        str: 保存的模型文件路径
    """
    # 定义各模型的默认参数网格
    param_grids = {
        'rf': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'xgboost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 1.0]
        },
        'lgbm': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 1.0]
        }
    }
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    print("📦 加载数据...")
    print(f"   标签列: '{label_column}'")
    
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
        
        # 4. 创建并训练模型
        print(f"🚀 训练 {model_type} 模型...")
        
        if model_type == 'cnn':
            # 深度学习模型
            input_dim = X_train.shape[1]
            num_classes = len(le.classes_)
            model = CNNTrainer(input_dim, num_classes, **kwargs)
            model.fit(X_train_scaled, y_train_encoded)
            print("   训练完成!")
        else:
            # 机器学习模型
            if not ML_MODEL_AVAILABLE:
                raise ValueError("机器学习模型不可用，请安装所需依赖或使用 'cnn' 模型")
            
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
            else:
                # 直接使用提供的参数训练模型
                model = MLModelFactory.create_model(model_type, **kwargs)
                model.fit(X_train_scaled, y_train_encoded)
            
            print("   训练完成!")
        
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
        
        # 使用完整数据集重新训练（可选，这里使用训练集训练的模型）
        # 注意：为了保持一致性，我们使用训练集训练的模型，而不是重新训练
    else:
        print("🔄 关闭评估模式，使用完整数据集训练...")
        # 2. 标签编码（使用完整数据集）
        print("🔤 标签编码...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(f"   类别数: {len(le.classes_)}")
        
        # 3. 特征标准化（使用完整数据集）
        print("📊 特征标准化...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 4. 创建并训练模型
        print(f"🚀 训练 {model_type} 模型...")
        
        if model_type == 'cnn':
            # 深度学习模型
            input_dim = X.shape[1]
            num_classes = len(le.classes_)
            model = CNNTrainer(input_dim, num_classes, **kwargs)
            model.fit(X_scaled, y_encoded)
            print("   训练完成!")
        else:
            # 机器学习模型
            if not ML_MODEL_AVAILABLE:
                raise ValueError("机器学习模型不可用，请安装所需依赖或使用 'cnn' 模型")
            
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
                grid_search_cv.fit(X_scaled, y_encoded)
                
                # 获取最佳模型
                model = grid_search_cv.best_estimator_
                print(f"   最佳参数: {grid_search_cv.best_params_}")
                print(f"   最佳交叉验证分数: {grid_search_cv.best_score_:.4f}")
            else:
                # 直接使用提供的参数训练模型
                model = MLModelFactory.create_model(model_type, **kwargs)
                model.fit(X_scaled, y_encoded)
            
            print("   训练完成!")
        
        # 不进行模型评估
        print("⚠️ 评估模式已关闭，跳过模型评估")
    
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
    
    return model_path
