import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, recall_score, precision_score, 
                             f1_score, accuracy_score, brier_score_loss, 
                             average_precision_score, cohen_kappa_score, 
                             matthews_corrcoef, log_loss)
from scipy.stats import ks_2samp
try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

def load_and_preprocess_data(file_path):
    try:
        # 尝试读取 GBK 编码
        data = pd.read_csv(file_path, encoding='gbk')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='utf-8')
        
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    
    # 1. 显式移除已知的 ID 类列
    drop_cols = ['code', 'year', 'name', '证券代码', '证券简称', '简称', 'Stkcd']
    features = features.drop([c for c in drop_cols if c in features.columns], axis=1)
    
    features = features.select_dtypes(include=[np.number])
    
    # 检查是否误删了所有列
    if features.shape[1] == 0:
        raise ValueError("错误：预处理后没有剩余的数值特征列。请检查 CSV 文件格式是否正确读取。")

    feature_names = list(features.columns)
    
    # 缺失值填充
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    features_imputed = imputer.fit_transform(features)
    features = pd.DataFrame(features_imputed,columns=feature_names)
    
    return features, labels

def g_mean_score(label, pred):
    tpr = recall_score(label, pred)
    neg_indices = np.where(label == 0)[0]
    tn = sum([1 for i in range(len(pred)) if (pred[i] == 0 and np.array(label)[i] == 0)])
    tnr = tn / len(neg_indices) if len(neg_indices) > 0 else 0
    return np.sqrt(tpr * tnr)

def calc_metrics(y_true, y_prob):
    y_pred = np.rint(y_prob)
    ks = ks_2samp(y_prob[y_true == 1], y_prob[y_true != 1]).statistic
    
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'Gmean': g_mean_score(y_true, y_pred),
        'TPR': recall_score(y_true, y_pred),
        'KS': ks,
        'F1': f1_score(y_true, y_pred),
        'LogLoss': log_loss(y_true, y_prob)
    }