import geopandas as gpd
import numpy as np
import psutil
import os
from rasterio.transform import rowcol


def check_and_reproject(gdf, target_crs):
    """
    检查并重新投影矢量数据到目标坐标系
    
    Args:
        gdf (GeoDataFrame): 输入的矢量数据
        target_crs (CRS): 目标坐标系
    
    Returns:
        GeoDataFrame: 重新投影后的矢量数据
    """
    if gdf.crs is None:
        # 如果矢量数据没有坐标系，直接返回
        return gdf
    
    if gdf.crs != target_crs:
        # 如果坐标系不一致，进行重投影
        gdf = gdf.to_crs(target_crs)
    
    return gdf


def get_window_from_geom(geom, transform):
    """
    从几何体和仿射变换参数获取numpy切片所需的行列索引
    
    Args:
        geom (Geometry): 输入的几何对象
        transform (Affine): 仿射变换参数
    
    Returns:
        tuple: (row_start, row_stop, col_start, col_stop) 切片索引
    """
    # 获取几何体的边界框
    bounds = geom.bounds
    left, bottom, right, top = bounds
    
    # 将地理坐标转换为影像行列号
    row_start, col_start = rowcol(transform, left, top)
    row_stop, col_stop = rowcol(transform, right, bottom)
    
    # 确保索引为整数且顺序正确
    row_start, row_stop = min(row_start, row_stop), max(row_start, row_stop)
    col_start, col_stop = min(col_start, col_stop), max(col_start, col_stop)
    
    # 确保索引非负
    row_start = max(0, row_start)
    col_start = max(0, col_start)
    
    # 加1以包含结束索引（Python切片是左闭右开）
    row_stop += 1
    col_stop += 1
    
    return row_start, row_stop, col_start, col_stop


def calculate_image_memory(width, height, bands, dtype='float32'):
    """
    计算影像加载到内存所需的字节数
    
    Args:
        width (int): 影像宽度
        height (int): 影像高度
        bands (int): 波段数
        dtype (str): 数据类型
    
    Returns:
        int: 所需内存（字节）
    """
    # 计算每种数据类型的字节大小
    dtype_sizes = {
        'uint8': 1,
        'int8': 1,
        'uint16': 2,
        'int16': 2,
        'uint32': 4,
        'int32': 4,
        'float32': 4,
        'float64': 8
    }
    
    # 获取数据类型对应的字节大小
    byte_size = dtype_sizes.get(dtype, 4)  # 默认使用float32
    
    # 计算总字节数
    total_bytes = width * height * bands * byte_size
    
    return total_bytes


def check_memory_availability(image_memory, safety_factor=1.3):
    """
    检查系统是否有足够的内存来加载影像
    
    Args:
        image_memory (int): 影像所需内存（字节）
        safety_factor (float): 安全系数，默认为1.3（预留30%的冗余空间）
    
    Returns:
        bool: 是否有足够的内存
    """
    # 获取系统可用内存
    available_memory = psutil.virtual_memory().available
    
    # 计算需要的内存（带安全系数）
    required_memory = image_memory * safety_factor
    
    # 检查是否有足够的内存
    has_enough_memory = available_memory >= required_memory
    
    # 转换为人类可读的格式
    def bytes_to_gb(bytes_value):
        return bytes_value / (1024 ** 3)
    
    print(f"💾 内存检查:")
    print(f"   影像所需内存: {bytes_to_gb(image_memory):.2f} GB")
    print(f"   带安全系数 ({safety_factor}x) 所需内存: {bytes_to_gb(required_memory):.2f} GB")
    print(f"   系统可用内存: {bytes_to_gb(available_memory):.2f} GB")
    print(f"   内存是否足够: {'✅ 是' if has_enough_memory else '❌ 否'}")
    
    return has_enough_memory


def should_load_in_memory(src, force_memory=False, safety_factor=1.3):
    """
    判断是否应该将影像加载到内存中
    
    Args:
        src (DatasetReader): rasterio影像读取器
        force_memory (bool): 是否强制加载到内存
        safety_factor (float): 安全系数，默认为1.3（预留30%的冗余空间）
    
    Returns:
        bool: 是否应该加载到内存
    """
    if force_memory:
        print("🔧 强制模式: 影像将被加载到内存中")
        return True
    
    # 计算影像所需内存
    image_memory = calculate_image_memory(
        width=src.width,
        height=src.height,
        bands=src.count,
        dtype=src.dtypes[0]
    )
    
    # 检查内存是否足够
    return check_memory_availability(image_memory, safety_factor)
