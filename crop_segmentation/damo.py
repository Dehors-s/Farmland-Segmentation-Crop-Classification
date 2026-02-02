import os
import sys

correct_path = os.path.abspath('..') 
sys.path.insert(0, correct_path)

try:
    from crop_segmentation.interfaces.train_interface import train_pipeline
    print("✅ 成功导入train_pipeline")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    # 尝试直接导入模块
    try:
        # 检查目录结构
        print(f"检查目录: {os.listdir(os.path.abspath('..'))}")
        if 'crop_segmentation' in os.listdir(os.path.abspath('..')):
            print(f"crop_segmentation目录内容: {os.listdir(os.path.join(os.path.abspath('..'), 'crop_segmentation'))}")
    except Exception as e2:
        print(f"检查目录失败: {e2}")
    sys.exit(1)
    
#符合Windows多进程规范，避免无限递归 ：确保 train_pipeline 函数只在主进程中执行一次    
if __name__ == '__main__':
    # print("开始训练模型...")
    # model_path = train_pipeline(
    #     model_type='svm',  # 选择随机森林模型rf (Random Forest) - 随机森林，svm (Support Vector Machine) - 支持向量机，xgboost (XGBoost) - 梯度提升树，lgbm (LightGBM) - 轻量级梯度提升机，深度学习模型-cnn (Convolutional Neural Network) - 卷积神经网络
    #     shp_path=r'D:\Work space\DeepLearning\耕地分割\data\完整作物边界\作物分布.shp',  # 训练数据
    #     tif_path=r'D:\Work space\DeepLearning\耕地分割\data\Sentinel2_TimeSeries_Stack_2023-0000000000-0000000000.tif',  # 遥感影像
    #     output_dir='models',  # 模型保存目录
    #     label_column='value',  # 标签列名
    #     # n_estimators=100,  # 随机森林模型参数
    #     use_optimized_loader=True,  # 启用优化的数据加载器
    #     max_workers=4,  # 使用8个工作进程
    #     chunk_size=1024,  # 分块大小
    #     force_memory=False,  # 自动检测内存情况
    #     grid_search=False  # 启用Grid Search进行超参数搜索
    # )
    # print(f"模型训练完成，保存至: {model_path}")

# 步骤2: 模型推理
    from crop_segmentation.interfaces.infer_interface import predict_pipeline

    print("开始模型推理...")
    output_shp = predict_pipeline(
        model_path=r'D:\Work space\DeepLearning\耕地分割\models\rf_model_bundle.joblib',  # 训练好的模型，可以填本地模型路径
        tif_path=r'D:\Work space\DeepLearning\耕地分割\data\Sentinel2_TimeSeries_Stack_2023-0000000000-0000006400.tif',  # 新影像
        shp_path=r'D:\Work space\DeepLearning\耕地分割\data\完整作物边界\作物分布.shp',  # 待预测地块
        output_shp='results/predicted_crops_new.shp',  # 新的结果保存路径
        # max_workers=4,  # 使用4个工作进程
        # chunk_size=1024,  # 分块大小
        force_memory=False  # 自动检测内存情况，如需强制加载到内存设为True
    )
    print(f"推理完成，结果保存至: {output_shp}")

# # 步骤3: 查看结果
#     import geopandas as gpd

#     print("查看预测结果...")
#     gdf = gpd.read_file(output_shp)
#     print(f"预测地块数: {len(gdf)}")
#     print(f"实际列名: {list(gdf.columns)}")
#     print(f"预测类别分布:")
#     print(gdf['pred_class'].value_counts())
    
#     # 打印前5条记录，展示三个字段的对应关系
#     print("\n🔍 前5条记录示例:")
#     print("-" * 100)
#     print(f"{'ID':<5} {'原始value列':<15} {'pred_id':<10} {'pred_class':<20}")
#     print("-" * 100)
    
#     for i in range(min(5, len(gdf))):
#         row = gdf.iloc[i]
#         value_col = row.get('value', 'N/A')
#         pred_id = row.get('pred_id', 'N/A')
#         pred_class = row.get('pred_class', 'N/A')
#         print(f"{i:<5} {str(value_col):<15} {str(pred_id):<10} {str(pred_class):<20}")
#     print("-" * 100)
