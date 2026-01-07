import numpy as np
from scipy.misc import derivative

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def get_csma_objective(a, c, loss_type='cross_entropy', weights=None):
    """
    Generate CSMA (Confidence-Scaled Margin Adaptation) objective function.
    
    Args:
        a (float): CSMA slope parameter.
        c (float): CSMA margin parameter.
        loss_type (str): 'cross_entropy' or 'weighted_cross_entropy'.
        weights (dict): Weights for 'weighted_cross_entropy', e.g. {'pos': 0.8, 'neg': 0.2}.
    """
    if weights is None:
        weights = {'pos': 0.8, 'neg': 0.2}

    def csma_loss_func(y_pred, y_true):
        # 1. Calculate base Loss (li)
        p = sigmoid(y_pred)
        
        if loss_type == 'weighted_cross_entropy':
            # -w_pos * t * log(p) - w_neg * (1-t) * log(1-p)
            li = -(weights['pos'] * y_true * np.log(p)) - (weights['neg'] * (1 - y_true) * np.log(1 - p))
        else:
            # Standard cross entropy
            li = -(y_true * np.log(p)) - ((1 - y_true) * np.log(1 - p))
        
        # 2. Calculate dynamic weight w(li)
        # Weight = 1 / (1 + exp(mu * (v - li)))
        weight = 1.0 / (1.0 + np.exp(a * (c - li)))
        
        return li * weight

    def objective(y_true, y_pred):
        partial_loss = lambda x: csma_loss_func(x, y_true)
        grad = derivative(partial_loss, y_pred, n=1, dx=1e-3)
        hess = derivative(partial_loss, y_pred, n=2, dx=1e-3)
        return grad, hess

    return objective