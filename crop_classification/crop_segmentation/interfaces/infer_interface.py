import os
import joblib
import numpy as np
import geopandas as gpd
import rasterio
from tqdm import tqdm
from ..utils.geo_utils import check_and_reproject, should_load_in_memory
from ..core.data_loader import extract_features_from_image


def predict_pipeline(model_path, tif_path, shp_path, output_shp, force_memory=False, **kwargs):
    """
    推理流水线
    
    Args:
        model_path (str): 模型文件路径
        tif_path (str): GeoTIFF路径
        shp_path (str): Shapefile路径
        output_shp (str): 输出Shapefile路径
        force_memory (bool): 是否强制将影像加载到内存中
        **kwargs: 其他参数（为了API一致性）
    
    Returns:
        str: 输出Shapefile路径
    """
    # 获取max_workers和chunk_size参数（为了API一致性）
    max_workers = kwargs.pop('max_workers', 4)
    chunk_size = kwargs.pop('chunk_size', 1024)
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
    
    # 2. 读取矢量数据
    print("⏳ 读取矢量数据...")
    gdf = gpd.read_file(shp_path)
    print(f"   共 {len(gdf)} 个地块")
    print(f"   矢量数据坐标系: {gdf.crs}")
    
    # 3. 打开影像并读取到内存
    print("🌍 读取影像...")
    with rasterio.open(tif_path) as src:
        print(f"   影像坐标系: {src.crs}")
        print(f"   影像边界: {src.bounds}")
        
        # 检查坐标系
        if src.crs is None:
            print("⚠️ 影像文件缺少坐标系信息，跳过坐标转换...")
        elif gdf.crs is None:
            print("⚠️ 矢量文件缺少坐标系信息，跳过坐标转换...")
        else:
            # 检查并转换坐标系
            if gdf.crs != src.crs:
                print(f"🔄 坐标系不一致，正在转换矢量投影至 {src.crs}...")
                gdf = check_and_reproject(gdf, src.crs)
                print("   坐标转换完成!")
            else:
                print("✅ 坐标系一致，无需转换")
        
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
        
        # 获取影像边界
        src_bounds = src.bounds
        
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
    
    # 打印说明
    print("\n💡 说明:")
    print("- 原始标签 (value列): 训练数据中'value'列的原始值")
    print("- 编码值 (pred_id): LabelEncoder编码后的整数值，模型内部使用")
    print("- 预测结果 (pred_class): 解码后的类别值，与原始标签一致")
    print("- unknown: 无法预测或特征提取失败的情况")
    
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
    
    return output_shp
