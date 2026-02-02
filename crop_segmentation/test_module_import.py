#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模块导入和基本功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))


print("🚀 开始测试模块导入...")

# 测试基础工具模块
try:
    from crop_segmentation.utils.geo_utils import check_and_reproject, get_window_from_geom
    print("✅ utils.geo_utils 导入成功")
except Exception as e:
    print(f"❌ utils.geo_utils 导入失败: {e}")

# 测试数据加载模块
try:
    from crop_segmentation.core.data_loader import load_data, extract_features_from_image
    print("✅ core.data_loader 导入成功")
except Exception as e:
    print(f"❌ core.data_loader 导入失败: {e}")

# 测试机器学习模型模块
try:
    from crop_segmentation.core.models_ml import MLModelFactory
    print("✅ core.models_ml 导入成功")
except Exception as e:
    print(f"❌ core.models_ml 导入失败: {e}")

# 测试深度学习模型模块
try:
    from crop_segmentation.core.models_dl import CropCNN, CNNTrainer
    print("✅ core.models_dl 导入成功")
except Exception as e:
    print(f"❌ core.models_dl 导入失败: {e}")

# 测试训练接口模块
try:
    from crop_segmentation.interfaces.train_interface import train_pipeline
    print("✅ interfaces.train_interface 导入成功")
except Exception as e:
    print(f"❌ interfaces.train_interface 导入失败: {e}")

# 测试推理接口模块
try:
    from crop_segmentation.interfaces.infer_interface import predict_pipeline
    print("✅ interfaces.infer_interface 导入成功")
except Exception as e:
    print(f"❌ interfaces.infer_interface 导入失败: {e}")

print("\n📊 模块导入测试完成！")

# 测试MLModelFactory的基本功能
try:
    print("\n🧪 测试 MLModelFactory...")
    # 创建随机森林模型
    rf_model = MLModelFactory.create_model('rf')
    print("✅ MLModelFactory.create_model('rf') 成功")
    
    # 创建SVM模型
    svm_model = MLModelFactory.create_model('svm')
    print("✅ MLModelFactory.create_model('svm') 成功")
    
    # 尝试创建XGBoost模型（可选）
    try:
        xgb_model = MLModelFactory.create_model('xgboost')
        print("✅ MLModelFactory.create_model('xgboost') 成功")
    except ImportError as e:
        print(f"⚠️ MLModelFactory.create_model('xgboost') 跳过: {e}")
    
    # 尝试创建LightGBM模型（可选）
    try:
        lgbm_model = MLModelFactory.create_model('lgbm')
        print("✅ MLModelFactory.create_model('lgbm') 成功")
    except ImportError as e:
        print(f"⚠️ MLModelFactory.create_model('lgbm') 跳过: {e}")
    
    print("✅ MLModelFactory 测试完成")
    
except Exception as e:
    print(f"❌ MLModelFactory 测试失败: {e}")

# 测试CropCNN的基本功能
try:
    print("\n🧪 测试 CropCNN...")
    import torch
    # 创建模型实例
    cnn_model = CropCNN(input_dim=10, num_classes=5)
    print("✅ CropCNN 初始化成功")
    
    # 测试前向传播
    test_input = torch.randn(32, 10)  # 32个样本，10个特征
    output = cnn_model(test_input)
    print(f"✅ CropCNN 前向传播成功，输出形状: {output.shape}")
    
except Exception as e:
    print(f"❌ CropCNN 测试失败: {e}")

print("\n🎉 所有测试完成！")
