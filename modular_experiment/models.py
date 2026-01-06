
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier

# Imbalanced-learn
from imblearn.under_sampling import (
    ClusterCentroids, NearMiss, RandomUnderSampler,
    EditedNearestNeighbours, AllKNN, TomekLinks,
    OneSidedSelection, CondensedNearestNeighbour,
    NeighbourhoodCleaningRule
)
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imbalanced_ensemble.ensemble import (
    SMOTEBoostClassifier, SMOTEBaggingClassifier,
    RUSBoostClassifier, UnderBaggingClassifier,
    BalanceCascadeClassifier
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import custom ensembles if available
try:
    from ensemble.self_paced_ensemble import SelfPacedEnsemble
    from ensemble.equalizationensemble import EASE
    from ensemble.ECUBoost_RF import ECUBoostRF
    from ensemble.hub_ensemble import HashBasedUndersamplingEnsemble
except ImportError:
    print("Warning: Custom ensemble modules not found. Some methods may fail.")


from .config import SEED

def get_classifier(method_name, n_jobs=-1):
    """
    Factory function to get a classifier instance with default params.
    """
    if method_name == 'lr':
        return LogisticRegression()
        
    elif method_name == 'lda':
        return LDA()
        
    elif method_name == 'dt':
        return DecisionTreeClassifier(
            min_samples_split=2,
            random_state=SEED
        )
        
    elif method_name == 'rf':
        return RandomForestClassifier(
            n_estimators=100,
            n_jobs=n_jobs,
            random_state=SEED
        )
        
    elif method_name == 'knn':
        return KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=n_jobs
        )
        
    elif method_name == 'xgb':
        return XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            min_child_weight=1,
            gamma=0,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            objective="binary:logistic",
            n_jobs=n_jobs,
            random_state=SEED,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
    elif method_name == 'lgb':
        return LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=31,
            max_depth=-1,
            learning_rate=0.1,
            n_estimators=100,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            scale_pos_weight=1.0,
            objective='binary',
            n_jobs=n_jobs,
            random_state=SEED
        )
        
    elif method_name == 'ada':
        return AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            algorithm='SAMME.R',
            random_state=SEED
        )
        
    elif method_name == 'gbdt':
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=SEED
        )
        
    else:
        raise ValueError(f"Unknown method: {method_name}")

def get_sampler(imb_method, ratio, n_jobs=-1, X_shape=None):
    """
    Factory for Imbalanced-learn samplers.
    X_shape is needed for ADASYN/SMOTE neighbor validation.
    """
    
    # --- Undersampling ---
    if imb_method == 'RUS':
        return RandomUnderSampler(sampling_strategy=ratio, random_state=SEED)
    elif imb_method == 'NM':
        return NearMiss(sampling_strategy=ratio, n_jobs=n_jobs)
    elif imb_method == 'CC':
        return ClusterCentroids(sampling_strategy=ratio, random_state=SEED)
    elif imb_method == 'Tomek':
        return TomekLinks(sampling_strategy='auto', n_jobs=n_jobs)
    elif imb_method == 'ENN':
        return EditedNearestNeighbours(sampling_strategy='auto', n_jobs=n_jobs)
    elif imb_method == 'NCR':
        return NeighbourhoodCleaningRule(sampling_strategy='auto', n_jobs=n_jobs)
    elif imb_method == 'ALLKNN':
        return AllKNN(sampling_strategy='auto', n_jobs=n_jobs)
    elif imb_method == 'OSS':
        return OneSidedSelection(sampling_strategy='auto', random_state=SEED, n_jobs=n_jobs)
    elif imb_method == 'CNN':
        return CondensedNearestNeighbour(sampling_strategy='auto', random_state=SEED, n_jobs=n_jobs)
        
    # --- Oversampling ---
    elif imb_method == 'ROS':
        return RandomOverSampler(sampling_strategy=ratio, random_state=SEED)
        
    elif imb_method in ['SMOTE', 'BorderlineSMOTE', 'ADASYN']:
        # Dynamic neighbors check
        n_neighbors = 5
        if X_shape:
            # ensure neighbors < n_samples
            n_neighbors = min(4, X_shape[0] - 1)
            if n_neighbors < 2: n_neighbors = 2
            
        if imb_method == 'SMOTE':
            return SMOTE(sampling_strategy=ratio, k_neighbors=n_neighbors, random_state=SEED, n_jobs=n_jobs)
        elif imb_method == 'BorderlineSMOTE':
            return BorderlineSMOTE(sampling_strategy=ratio, k_neighbors=n_neighbors, random_state=SEED, n_jobs=n_jobs)
        elif imb_method == 'ADASYN':
             # ADASYN needs more careful handling
            if X_shape and X_shape[0] <= 5:
                print(f"Warning: Not enough samples for ADASYN. Skipping.")
                return None
            return ADASYN(sampling_strategy=ratio, n_neighbors=n_neighbors, random_state=SEED, n_jobs=n_jobs)
            
    # --- Hybrid ---
    elif imb_method == 'SMOTEENN':
        return SMOTEENN(sampling_strategy=ratio, random_state=SEED, n_jobs=n_jobs)
    elif imb_method == 'SMOTETomek':
        return SMOTETomek(sampling_strategy=ratio, random_state=SEED, n_jobs=n_jobs)
        
    return None

def get_ensemble_clf(imb_method, ratio, n_jobs=-1):
    """Factory for Ensemble Classifiers"""
    
    if imb_method == 'SMOTEBoost':
        # Note: SMOTEBoostClassifier implementation might differ in params
        return SMOTEBoostClassifier(n_estimators=50, random_state=SEED)
        
    elif imb_method == 'RUSBoost':
        return RUSBoostClassifier(n_estimators=50, algorithm='SAMME.R', random_state=SEED)
        
    elif imb_method == 'SMOTEBagging':
        return SMOTEBaggingClassifier(n_estimators=100, random_state=SEED, n_jobs=n_jobs)
        
    elif imb_method == 'UnderBagging':
        return UnderBaggingClassifier(n_estimators=100, random_state=SEED, n_jobs=n_jobs)
        
    elif imb_method == 'BalanceCascade':
        return BalanceCascadeClassifier(n_estimators=100, random_state=SEED, n_jobs=n_jobs)
        
    elif imb_method == 'ease':
        try:
            return EASE(base_estimator=RandomForestClassifier(n_estimators=100), n_estimators=200)
        except NameError:
            print("EASE not available.")
            return None
            
    elif imb_method == 'SelfPacedEnsemble':
        try:
            return SelfPacedEnsemble(base_estimator=DecisionTreeClassifier(), n_estimators=100, k_bins=10)
        except NameError:
             print("SelfPacedEnsemble not available.")
             return None
             
    # ... add others as needed
    
    return None
