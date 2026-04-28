from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
            'random_state': 42,
            'probability': True
        }
        
        # 更新参数
        default_params.update(kwargs)
        
        return SVC(**default_params)
    
    @staticmethod
    def _create_xgboost(**kwargs):
        """
        创建XGBoost分类器
        
        Args:
            **kwargs: 模型参数
        
        Returns:
            XGBClassifier 实例
        """
        # 默认参数
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        }
        
        # 更新参数
        default_params.update(kwargs)
        
        return xgb.XGBClassifier(**default_params)
    
    @staticmethod
    def _create_lightgbm(**kwargs):
        """
        创建LightGBM分类器
        
        Args:
            **kwargs: 模型参数
        
        Returns:
            LGBMClassifier 实例
        """
        # 默认参数
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        # 更新参数
        default_params.update(kwargs)
        
        return lgb.LGBMClassifier(**default_params)
