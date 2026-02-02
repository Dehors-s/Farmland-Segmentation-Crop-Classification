import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from ..utils.geo_utils import check_and_reproject


def load_data(shp_path, tif_path, label_column):
    """
    从Shapefile和GeoTIFF加载数据并提取特征
    
    Args:
        shp_path (str): Shapefile路径
        tif_path (str): GeoTIFF路径
        label_column (str): 标签列名
    
    Returns:
        tuple: (X, y, feature_names)
            X: 特征矩阵 (numpy array)
            y: 标签向量 (numpy array)
            feature_names: 特征名称列表
    """
    # 读取矢量数据
    gdf = gpd.read_file(shp_path)
    
    # 打开影像文件
    with rasterio.open(tif_path) as src:
        # 检查并转换坐标系
        gdf = check_and_reproject(gdf, src.crs)
        
        # 动态获取波段数
        num_bands = src.count
        
        # 生成特征名称
        feature_names = [f'band_{i+1}' for i in range(num_bands)]
        
        # 准备存储特征和标签
        X_list = []
        y_list = []
        
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
                    X_list.append(band_means)
                    y_list.append(label)
                    
            except Exception as e:
                # 跳过处理失败的地块
                continue
        
        # 转换为numpy数组
        X = np.array(X_list)
        y = np.array(y_list)
    
    return X, y, feature_names


def extract_features_from_image(src, geom):
    """
    从影像中提取单个地块的特征
    
    Args:
        src (DatasetReader): 影像文件读取器
        geom (Geometry): 地块几何对象
    
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
