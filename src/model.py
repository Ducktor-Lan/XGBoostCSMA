from xgboost import XGBClassifier
from csma_loss import get_csma_objective

class XGBoostCSMA(XGBClassifier):
    """
    XGBoost-CSMA Model Wrapper Class.
    """
    def __init__(self, a=10, c=0.5, loss_type='weighted_cross_entropy', **kwargs):
        """
        Args:
            a (float): CSMA parameter mu (slope). Default is 10 (empirical value).
            c (float): CSMA parameter v (margin). Default is 0.5 (empirical value).
            loss_type (str): Base loss function type 'cross_entropy' or 'weighted_cross_entropy'.
            **kwargs: Other XGBoost parameters, overriding defaults.
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
        
        super().__init__(**default_params)

    def fit(self, X, y, **kwargs):
        self.objective = get_csma_objective(self.a, self.c, self.loss_type)
        return super().fit(X, y, **kwargs)