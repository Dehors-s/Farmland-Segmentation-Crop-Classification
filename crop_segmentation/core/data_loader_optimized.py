import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.geometry import box
from ..utils.geo_utils import check_and_reproject


def load_data_optimized(shp_path, tif_path, label_column, chunk_size=1024, max_workers=4):
    """
    优化的从Shapefile和GeoTIFF加载数据并提取特征的函数
    
    针对大尺寸遥感影像的优化：
    1. 分块处理：将影像分成多个块，每个块独立处理
    2. 并行计算：使用多进程并行处理多个块
    3. 空间索引：快速找到与每个块相交的地块
    4. 内存效率：只加载和处理需要的部分
    
    Args:
        shp_path (str): Shapefile路径
        tif_path (str): GeoTIFF路径
        label_column (str): 标签列名
        chunk_size (int): 分块大小（像素）
        max_workers (int): 最大并行工作进程数
    
    Returns:
        tuple: (X, y, feature_names)
            X: 特征矩阵 (numpy array)
            y: 标签向量 (numpy array)
            feature_names: 特征名称列表
    """
    # 读取矢量数据
    print(f"📦 读取矢量数据: {shp_path}")
    gdf = gpd.read_file(shp_path)
    print(f"   共 {len(gdf)} 个地块")
    
    # 打开影像文件
    with rasterio.open(tif_path) as src:
        # 检查并转换坐标系
        gdf = check_and_reproject(gdf, src.crs)
        
        # 动态获取波段数
        num_bands = src.count
        
        # 生成特征名称
        feature_names = [f'band_{i+1}' for i in range(num_bands)]
        
        # 准备存储特征和标签
        all_features = []
        all_labels = []
        
        # 计算分块数量
        n_chunks_x = (src.width + chunk_size - 1) // chunk_size
        n_chunks_y = (src.height + chunk_size - 1) // chunk_size
        total_chunks = n_chunks_x * n_chunks_y
        print(f"🌍 影像分块: {n_chunks_x}x{n_chunks_y} = {total_chunks} 块")
        
        # 生成所有分块
        chunks = []
        for i in range(n_chunks_x):
            for j in range(n_chunks_y):
                # 计算分块边界
                col_start = i * chunk_size
                col_stop = min((i + 1) * chunk_size, src.width)
                row_start = j * chunk_size
                row_stop = min((j + 1) * chunk_size, src.height)
                
                # 创建分块边界框
                chunk_bounds = src.window_bounds(rasterio.windows.Window(col_start, row_start, 
                                                                       col_stop - col_start, 
                                                                       row_stop - row_start))
                chunk_geom = box(*chunk_bounds)
                
                # 找到与分块相交的地块
                intersecting_gdf = gdf[gdf.intersects(chunk_geom)]
                
                if len(intersecting_gdf) > 0:
                    chunks.append((i, j, col_start, row_start, col_stop, row_stop, intersecting_gdf))
        
        print(f"🔍 有效分块数: {len(chunks)}")
        
        # 并行处理分块
        print(f"🚀 并行处理分块 (工作进程数: {max_workers})...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有分块任务
            future_to_chunk = {}
            for chunk_info in chunks:
                i, j, col_start, row_start, col_stop, row_stop, intersecting_gdf = chunk_info
                future = executor.submit(
                    process_chunk,
                    tif_path,
                    col_start, row_start, col_stop, row_stop,
                    intersecting_gdf,
                    label_column,
                    num_bands
                )
                future_to_chunk[future] = (i, j)
            
            # 收集结果
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_features, chunk_labels = future.result()
                    all_features.extend(chunk_features)
                    all_labels.extend(chunk_labels)
                    print(f"✅ 完成分块 {chunk_idx[0]},{chunk_idx[1]}")
                except Exception as e:
                    print(f"❌ 处理分块 {chunk_idx[0]},{chunk_idx[1]} 失败: {e}")
        
        # 转换为numpy数组
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"📊 数据加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    return X, y, feature_names


def process_chunk(tif_path, col_start, row_start, col_stop, row_stop, gdf, label_column, num_bands):
    """
    处理单个影像分块
    
    Args:
        tif_path (str): GeoTIFF路径
        col_start, row_start, col_stop, row_stop (int): 分块边界
        gdf (GeoDataFrame): 与分块相交的地块
        label_column (str): 标签列名
        num_bands (int): 波段数
    
    Returns:
        tuple: (chunk_features, chunk_labels)
    """
    chunk_features = []
    chunk_labels = []
    
    # 打开影像文件
    with rasterio.open(tif_path) as src:
        # 读取分块数据
        window = rasterio.windows.Window(col_start, row_start, 
                                       col_stop - col_start, 
                                       row_stop - row_start)
        
        # 处理每个地块
        for idx, row in gdf.iterrows():
            geom = row.geometry
            label = row[label_column]
            
            try:
                # 使用rasterio.mask提取地块内的像素
                out_image, out_transform = rasterio.mask.mask(
                    src, [geom], crop=True, nodata=np.nan
                )
                
                # 展平数据并去除NaN值
                pixels = out_image.reshape(num_bands, -1).T
                valid_pixels = pixels[~np.isnan(pixels).any(axis=1)]
                
                if len(valid_pixels) > 0:
                    # 计算每个波段的均值作为特征
                    band_means = np.mean(valid_pixels, axis=0)
                    chunk_features.append(band_means)
                    chunk_labels.append(label)
                    
            except Exception as e:
                # 跳过处理失败的地块
                continue
    
    return chunk_features, chunk_labels


def extract_features_from_image_optimized(src, geom, chunk_size=1024):
    """
    优化的从影像中提取单个地块特征的函数
    
    Args:
        src (DatasetReader): 影像文件读取器
        geom (Geometry): 地块几何对象
        chunk_size (int): 分块大小（像素）
    
    Returns:
        numpy array: 提取的特征
    """
    try:
        # 使用rasterio.mask提取地块内的像素
        out_image, out_transform = rasterio.mask.mask(
            src, [geom], crop=True, nodata=np.nan
        )
        
        # 展平数据并去除NaN值
        num_bands = src.count
        pixels = out_image.reshape(num_bands, -1).T
        valid_pixels = pixels[~np.isnan(pixels).any(axis=1)]
        
        if len(valid_pixels) > 0:
            # 计算每个波段的均值作为特征
            band_means = np.mean(valid_pixels, axis=0)
            return band_means
        else:
            return None
            
    except Exception as e:
        return None
