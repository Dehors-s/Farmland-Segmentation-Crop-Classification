# 地理工具

## 模块功能
提供地理空间数据处理的核心工具函数，包括坐标转换、内存管理、影像内存计算和地理坐标到影像坐标的转换。

## 核心逻辑/算法

### 工作原理
地理工具模块封装了遥感和GIS处理中常用的底层功能：

1. **坐标转换**：检查并重新投影矢量数据到目标坐标系
2. **内存管理**：计算影像内存需求并检查系统内存可用性
3. **空间索引**：将地理坐标转换为影像行列坐标
4. **内存决策**：根据影像大小和系统内存情况决定是否加载到内存

### 核心流程
1. **坐标系统处理**：确保矢量数据和影像数据使用相同的坐标系
2. **内存计算**：计算影像加载到内存所需的空间
3. **内存检查**：检查系统是否有足够的内存加载影像
4. **处理模式决策**：基于内存检查结果决定处理模式

## 代码高光时刻

### 1. 坐标转换函数
```python
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
```

**精妙之处**：
- 安全检查：处理矢量数据没有坐标系的情况
- 高效处理：只在坐标系不一致时才进行重投影
- 清晰的函数文档和参数说明
- 返回处理后的GeoDataFrame，支持方法链式调用

### 2. 地理坐标到影像坐标的转换
```python
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
```

**精妙之处**：
- 完整的坐标转换：从几何体边界框到影像行列索引
- 健壮性：处理坐标顺序问题和负索引情况
- 符合Python切片规范：加1以适应左闭右开的切片规则
- 清晰的返回值格式，直接用于numpy切片

### 3. 影像内存计算
```python
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
```

**精妙之处**：
- 全面的 dtype 支持：覆盖常见的影像数据类型
- 安全的默认值：默认使用float32作为数据类型
- 简单直接的计算逻辑：宽度 × 高度 × 波段数 × 每个像素字节数
- 清晰的参数说明和返回值定义

### 4. 内存可用性检查
```python
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
```

**精妙之处**：
- 安全系数：通过safety_factor参数预留冗余内存
- 实时系统信息：使用psutil获取真实的系统可用内存
- 人类可读的输出：将字节转换为GB，提供清晰的内存使用情况
- 详细的状态打印：包括影像内存需求、安全系数、系统可用内存等
- 返回布尔值，方便上层函数进行决策

### 5. 内存加载决策函数
```python
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
```

**精妙之处**：
- 强制模式支持：允许用户强制将影像加载到内存
- 智能决策：根据影像大小和系统内存情况自动判断
- 完整的参数传递：将safety_factor传递给check_memory_availability
- 使用src.dtypes[0]获取真实的影像数据类型，确保内存计算准确
- 清晰的状态打印，提升用户体验

## 使用示例

### 基本用法
```python
from crop_segmentation.utils.geo_utils import (
    check_and_reproject, get_window_from_geom,
    calculate_image_memory, check_memory_availability,
    should_load_in_memory
)
import geopandas as gpd
import rasterio

# 1. 坐标转换示例
print("\n1. 坐标转换示例:")
gdf = gpd.read_file("path/to/parcels.shp")
with rasterio.open("path/to/image.tif") as src:
    target_crs = src.crs
    reprojected_gdf = check_and_reproject(gdf, target_crs)
    print(f"原始CRS: {gdf.crs}")
    print(f"目标CRS: {target_crs}")
    print(f"重投影后CRS: {reprojected_gdf.crs}")

# 2. 内存计算示例
print("\n2. 内存计算示例:")
width, height, bands = 5000, 5000, 10
dtype = 'float32'
memory_needed = calculate_image_memory(width, height, bands, dtype)
print(f"影像大小: {width}x{height}x{bands}")
print(f"数据类型: {dtype}")
print(f"所需内存: {memory_needed / (1024**3):.2f} GB")

# 3. 内存可用性检查
print("\n3. 内存可用性检查:")
has_enough_memory = check_memory_availability(memory_needed)
print(f"内存是否足够: {has_enough_memory}")

# 4. 内存加载决策
print("\n4. 内存加载决策:")
with rasterio.open("path/to/image.tif") as src:
    load_decision = should_load_in_memory(src)
    print(f"是否加载到内存: {load_decision}")
```

### 在数据加载器中使用
```python
from crop_segmentation.utils.geo_utils import should_load_in_memory
import rasterio

# 在数据加载器中使用
with rasterio.open("path/to/image.tif") as src:
    # 检查是否应该加载到内存
    if should_load_in_memory(src):
        # 内存模式
        print("使用内存模式加载数据")
        # 加载整个影像到内存
        image_data = src.read()
    else:
        # 分块模式
        print("使用分块模式加载数据")
        # 使用分块处理
```

## 技术要点

- **坐标系统处理**：确保矢量数据和影像数据使用相同的坐标系
- **内存管理**：智能检测系统内存情况，避免内存溢出
- **空间索引**：高效的地理坐标到影像坐标的转换
- **安全计算**：考虑数据类型和安全系数，确保内存计算的准确性
- **用户友好**：详细的状态信息和人类可读的内存使用报告
- **模块化设计**：每个函数专注于单一职责，便于测试和维护

## 代码优化建议

1. **缓存机制**：为频繁使用的坐标转换结果添加缓存，减少重复计算
2. **并行计算**：对于大规模的坐标转换，考虑使用并行计算
3. **内存预测**：添加内存使用趋势预测，提前预警内存不足情况
4. **自适应安全系数**：根据系统内存总量自动调整安全系数
5. **更多CRS支持**：添加对更多坐标系格式的支持，如EPSG代码字符串
6. **错误处理增强**：添加更详细的错误处理和异常信息
7. **性能监控**：添加函数执行时间监控，便于性能优化
8. **文档增强**：添加更多使用示例和边缘情况处理说明

## 与其他模块的集成

| 模块 | 地理工具的使用方式 | 集成点 |
|------|-------------------|--------|
| 数据加载器 | 内存管理、坐标转换 | should_load_in_memory, check_and_reproject |
| 训练接口 | 内存管理 | should_load_in_memory |
| 推理接口 | 内存管理、坐标转换 | should_load_in_memory, check_and_reproject |
| 核心处理 | 空间索引 | get_window_from_geom |

## 性能考量

| 操作 | 时间复杂度 | 空间复杂度 | 优化建议 |
|------|------------|------------|----------|
| 坐标转换 | O(n) | O(n) | 使用缓存，考虑并行计算 |
| 内存计算 | O(1) | O(1) | 无需优化 |
| 内存检查 | O(1) | O(1) | 无需优化 |
| 坐标到行列转换 | O(1) | O(1) | 无需优化 |

## 总结

地理工具模块是整个遥感作物分类系统的基础组件，提供了关键的地理空间处理功能和内存管理能力。它通过智能的内存检测和坐标转换，确保系统能够高效处理不同大小和坐标系的遥感数据，为上层模块提供稳定可靠的基础服务。

该模块的设计体现了以下原则：
- **稳健性**：处理各种边缘情况和异常情况
- **高效性**：优化计算逻辑，减少不必要的操作
- **可维护性**：清晰的代码结构和详细的文档
- **用户友好**：提供详细的状态信息和错误提示

通过这些设计原则，地理工具模块为整个系统的稳定运行和高效处理提供了有力保障。