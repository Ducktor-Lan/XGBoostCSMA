
import numpy as np
from math import sqrt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, average_precision_score, log_loss,
    cohen_kappa_score, matthews_corrcoef
)
from scipy.stats import ks_2samp

def sigmoid(x): 
    return 1./(1. +  np.exp(-x))

def calc_tpr(label, pred):
    """True Positive Rate (Sensitivity/Recall)"""
    return recall_score(label, pred)

def calc_tnr(label, pred):
    """True Negative Rate (Specificity)"""
    label = np.array(label)
    pred = np.array(pred)
    tn = np.sum((pred == 0) & (label == 0))
    neg = np.sum(label == 0)
    if neg == 0:
        return 0.0
    return tn / neg

def calc_g_mean(label, pred):
    """Geometric Mean of TPR and TNR"""
    tpr = calc_tpr(label, pred)
    tnr = calc_tnr(label, pred)
    return sqrt(tpr * tnr)

def calc_type1_error(proba, y):
    """Type I Error: False Positive Rate"""
    labels = np.array(y)
    count_neg = len(labels[labels==0])
    if count_neg == 0:
        return 0.0
    
    # Threshold at 0.5
    preds = (proba >= 0.5).astype(int)
    
    # FP: Predicted 1 (Distress) but actually 0 (Healthy)
    fp = np.sum((preds == 1) & (labels == 0))
    return float(fp / count_neg)

def calc_type2_error(proba, y):
    """Type II Error: False Negative Rate"""
    labels = np.array(y)
    count_pos = len(labels[labels==1])
    if count_pos == 0:
        return 0.0
        
    # Threshold at 0.5
    preds = (proba >= 0.5).astype(int)
    
    # FN: Predicted 0 (Healthy) but actually 1 (Distress)
    fn = np.sum((preds == 0) & (labels == 1))
    return float(fn / count_pos)

def calc_ks(y_pred, y_true):
    """Kolmogorov-Smirnov Statistic"""
    y_true = np.array(y_true)
    return ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic

def evaluate_model(y_true, y_pred, y_proba):
    """
    Calculate all metrics and return as a dictionary.
    """
    return {
        'auc': roc_auc_score(y_true, y_proba),
        'acc': accuracy_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'bs': brier_score_loss(y_true, y_proba),
        'e1': calc_type1_error(y_proba, y_true),
        'e2': calc_type2_error(y_proba, y_true),
        'ap': average_precision_score(y_true, y_proba),
        'tpr': calc_tpr(y_true, y_pred),
        'tnr': calc_tnr(y_true, y_pred),
        'gmean': calc_g_mean(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'ks': calc_ks(y_proba, y_true),
        'logloss': log_loss(y_true, y_proba)
    }
