# 机器学习模型工厂

## 模块功能
提供统一的机器学习模型创建接口，支持多种分类算法，包括随机森林、支持向量机、XGBoost和LightGBM。

## 核心逻辑/算法

### 工作原理
机器学习模型工厂采用工厂设计模式，根据用户指定的模型类型创建相应的分类模型实例：

1. **模型类型检测**：根据输入的模型类型字符串选择相应的模型创建方法
2. **依赖检查**：在创建XGBoost和LightGBM模型时，检查相应的依赖是否可用
3. **参数配置**：使用默认参数并允许用户通过kwargs覆盖默认值
4. **模型实例化**：创建并返回配置好的模型实例

### 核心流程
1. 接收模型类型和参数
2. 根据模型类型调用相应的创建方法
3. 检查依赖（对于XGBoost和LightGBM）
4. 配置模型参数（默认参数 + 用户参数）
5. 创建并返回模型实例

## 代码高光时刻

### 1. 工厂类设计
```python
class MLModelFactory:
    """
    机器学习模型工厂类，用于创建不同类型的分类模型
    """
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        根据模型类型创建并返回配置好的模型实例
        
        Args:
            model_type (str): 模型类型，支持 'rf', 'svm', 'xgboost', 'lgbm'
            **kwargs: 模型参数
        
        Returns:
            配置好的模型实例
        """
        if model_type == 'rf':
            return MLModelFactory._create_random_forest(**kwargs)
        elif model_type == 'svm':
            return MLModelFactory._create_svm(**kwargs)
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Please install it with 'pip install xgboost'")
            return MLModelFactory._create_xgboost(**kwargs)
        elif model_type == 'lgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'")
            return MLModelFactory._create_lightgbm(**kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
```

**精妙之处**：
- 工厂设计模式：提供统一的模型创建接口，隐藏具体实现细节
- 静态方法：使用@staticmethod装饰器，无需创建工厂实例即可使用
- 依赖检查：在创建需要额外依赖的模型时进行检查，提供清晰的错误信息
- 异常处理：对不支持的模型类型抛出明确的ValueError
- 清晰的函数文档：详细说明参数和返回值

### 2. 随机森林模型创建
```python
@staticmethod
def _create_random_forest(**kwargs):
    """
    创建随机森林分类器
    
    Args:
        **kwargs: 模型参数
    
    Returns:
        RandomForestClassifier 实例
    """
    # 默认参数
    default_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
    
    # 更新参数
    default_params.update(kwargs)
    
    return RandomForestClassifier(**default_params)
```

**精妙之处**：
- 合理的默认参数：设置了常用的随机森林参数默认值
- 参数更新机制：允许用户通过kwargs覆盖默认参数
- 清晰的函数文档：说明函数用途、参数和返回值

### 3. SVM模型创建
```python
@staticmethod
def _create_svm(**kwargs):
    """
    创建支持向量机分类器
    
    Args:
        **kwargs: 模型参数
    
    Returns:
        SVC 实例
    """
    # 默认参数
    default_params = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'random_state': 42
    }
    
    # 更新参数
    default_params.update(kwargs)
    
    return SVC(**default_params)
```

**精妙之处**：
- 合理的默认参数：使用'rbf'核作为默认值，适合大多数分类任务
- 参数更新机制：允许用户灵活调整SVM参数
- 清晰的函数文档：说明函数用途和参数

### 4. 可选依赖处理
```python
# 尝试导入可选依赖
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
```

**精妙之处**：
- 优雅的依赖处理：使用try-except块尝试导入可选依赖
- 全局标志：使用全局变量标记依赖是否可用
- 无侵入性：即使缺少可选依赖，模块仍能正常导入和使用其他模型

## 使用示例

### 基本用法
```python
from crop_segmentation.core.models_ml import MLModelFactory

# 创建随机森林模型
rf_model = MLModelFactory.create_model('rf')
print("随机森林模型:", rf_model)

# 创建带自定义参数的随机森林模型
rf_model_custom = MLModelFactory.create_model(
    'rf',
    n_estimators=200,
    max_depth=10,
    min_samples_split=5
)
print("自定义随机森林模型:", rf_model_custom)

# 创建SVM模型
svm_model = MLModelFactory.create_model('svm')
print("SVM模型:", svm_model)

# 创建XGBoost模型（如果已安装）
try:
    xgb_model = MLModelFactory.create_model('xgboost')
    print("XGBoost模型:", xgb_model)
except ImportError as e:
    print(f"无法创建XGBoost模型: {e}")

# 创建LightGBM模型（如果已安装）
try:
    lgbm_model = MLModelFactory.create_model('lgbm')
    print("LightGBM模型:", lgbm_model)
except ImportError as e:
    print(f"无法创建LightGBM模型: {e}")
```

### 在训练流水线中使用
```python
from crop_segmentation.core.models_ml import MLModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 准备数据
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 使用模型工厂创建模型
model = MLModelFactory.create_model('rf', n_estimators=100)

# 训练模型
model.fit(X_train, y_train)

# 评估模型
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"模型准确率: {accuracy:.4f}")
```

## 支持的模型类型

| 模型类型 | 描述 | 优势 | 适用场景 |
|---------|------|------|----------|
| `rf` | 随机森林 | 速度快，效果好，不易过拟合 | 通用分类任务 |
| `svm` | 支持向量机 | 小样本表现好，高维特征 | 小数据集，复杂决策边界 |
| `xgboost` | XGBoost | 高精度，处理复杂数据 | 需要最高精度的场景 |
| `lgbm` | LightGBM | 速度快，内存占用低 | 大规模数据集 |

## 技术要点

- **工厂设计模式**：提供统一的模型创建接口，简化模型选择和配置
- **可选依赖处理**：优雅处理可选依赖，提高模块的健壮性
- **默认参数配置**：为每种模型提供合理的默认参数，同时支持用户自定义
- **清晰的错误信息**：在依赖缺失或模型类型不支持时提供明确的错误信息
- **模块化设计**：每种模型的创建逻辑独立封装，便于维护和扩展

## 代码优化建议

1. **参数验证**：添加参数验证，确保用户提供的参数对所选模型有效
2. **更多模型支持**：添加对其他常用分类算法的支持，如KNN、决策树等
3. **模型配置预设**：提供预定义的模型配置预设，如'fast'、'accurate'等
4. **参数文档**：为每种模型的参数添加更详细的文档和建议值
5. **模型评估集成**：添加简单的模型评估方法，方便用户快速评估模型性能
6. **并行参数**：为支持并行计算的模型添加默认的n_jobs参数

## 总结

机器学习模型工厂模块为系统提供了统一、灵活的模型创建接口，支持多种常用的分类算法。它通过工厂设计模式和可选依赖处理，既简化了模型选择和配置，又提高了模块的健壮性和可扩展性。

该模块的设计体现了以下原则：
- **简洁性**：提供简单直观的接口，隐藏复杂的实现细节
- **灵活性**：支持用户自定义模型参数，适应不同的应用场景
- **健壮性**：优雅处理可选依赖和错误情况
- **可扩展性**：易于添加新的模型类型和功能