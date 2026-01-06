from xgboost import XGBClassifier
from csma_loss import get_csma_objective

class XGBoostCSMA(XGBClassifier):
    """
    XGBoost-CSMA 模型封装类。
    """
    def __init__(self, a=10, c=0.5, loss_type='cross_entropy', **kwargs):
        """
        Args:
            a (float): CSMA 参数 mu (slope). 默认值 10 (经验值).
            c (float): CSMA 参数 v (margin). 默认值 0.5 (经验值).
            loss_type (str): 基础损失函数类型 'cross_entropy' 或 'weighted_cross_entropy'.
            **kwargs: 其他 XGBoost 参数，可覆盖默认值。
        """
        self.a = a
        self.c = c
        self.loss_type = loss_type
        
        default_params = {
            'max_depth': 7,       
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 3,
            'gamma': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'n_jobs': 8,
            'random_state': 2024
        }
        
        default_params.update(kwargs)
        
        # 初始化父类
        super().__init__(**default_params)

    def fit(self, X, y, **kwargs):
        """
        重写 fit 方法，在训练开始前动态绑定 CSMA objective。
        """
        # 生成带有当前 a, c 参数的自定义目标函数
        self.objective = get_csma_objective(self.a, self.c, self.loss_type)
        
        return super().fit(X, y, **kwargs)