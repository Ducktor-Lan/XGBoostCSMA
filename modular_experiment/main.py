
import pandas as pd
import numpy as np
import os
import warnings
from time import time
import traceback

from .config import (
    DATASETS, METHODS, IMB_METHODS, SAMPLING_STRATEGIES,
    RESULT_DIR, USE_ENSEMBLE
)
from .data import get_data_splits
from .models import get_classifier, get_sampler, get_ensemble_clf
from .metrics import evaluate_model

# Suppress warnings
warnings.filterwarnings('ignore')

def run_experiment():
    print("Starting Modular Experiment...")
    
    for dataset in DATASETS:
        print(f"\nProcessing Dataset: {dataset}")
        dataset_results = []
        
        try:
            # Load and Preprocess Data (Fixing Leakage)
            X_train, y_train, X_valid, y_valid, X_test, y_test = get_data_splits(dataset)
            print(f"  Train shape: {X_train.shape}, Valid shape: {X_valid.shape}, Test shape: {X_test.shape}")
        except Exception as e:
            print(f"  Error loading dataset {dataset}: {e}")
            continue
            
        # Iterate over Imbalance Methods
        # If IMB_METHODS is empty (e.g. baseline), we still want to run once.
        # But based on config, if no imb methods are selected, we might just want to run baseline.
        # For this refactor, we assume the user checks the flags in config.
        
        # We need to handle the loop carefully. Use a list of (imb_method, method) tuples?
        # Or just nested loops as original.
        
        scan_methods = IMB_METHODS if IMB_METHODS else ['None']
        
        for imb_method in scan_methods:
            for method in METHODS:
                
                # Check if this combination makes sense
                # e.g. some ensembles are specific classifiers themselves
                
                # Best score tracking (per method/imb_method pair across ratios)
                best_valid_score = -1
                best_model = None
                best_ratio = None
                
                print(f"  Running {method} with {imb_method}...")
                
                for ratio in SAMPLING_STRATEGIES:
                    # print(f"    Ratio: {ratio}")
                    
                    try:
                        clf = get_classifier(method)
                        sampler = None
                        
                        # Logic to determine training path
                        is_ensemble_method = imb_method in [
                            'SMOTEBoost', 'SMOTEBagging', 'RUSBoost', 'UnderBagging', 
                            'BalanceCascade', 'BCRF', 'HUE', 'SelfPacedEnsemble', 'ease'
                        ]
                        
                        if is_ensemble_method:
                            # 1. Ensemble Method (Integrated Sampling)
                            # These methods replace the base classifier
                            clf = get_ensemble_clf(imb_method, ratio)
                            if clf is None:
                                continue # Skip if not available
                            
                            clf.fit(X_train, y_train)
                            
                        elif imb_method != 'None':
                            # 2. Resampling + Standard Classifier
                            sampler = get_sampler(imb_method, ratio, X_shape=X_train.shape)
                            if sampler is None:
                                continue
                                
                            X_res, y_res = sampler.fit_resample(X_train, y_train)
                            clf.fit(X_res, y_res)
                            
                        else:
                            # 3. No Imbalance Handling (Baseline)
                            clf.fit(X_train, y_train)
                        
                        # Validation
                        # Ensure we handle probability prediction correctly
                        if hasattr(clf, "predict_proba"):
                            y_valid_proba = clf.predict_proba(X_valid)[:, 1]
                        else:
                            # Fallback for models without probability (e.g. SVM sometimes)
                            y_valid_proba = clf.predict(X_valid)
                            
                        y_valid_pred = clf.predict(X_valid)
                        
                        # Selection Metric: G-Mean on Validation
                        # Re-implement G-mean calc locally or import
                        from .metrics import calc_g_mean
                        valid_score = calc_g_mean(y_valid, y_valid_pred)
                        
                        if valid_score > best_valid_score:
                            best_valid_score = valid_score
                            best_model = clf
                            best_ratio = ratio
                            
                    except Exception as e:
                        # Catch errors in individual folds/ratios to not crash the whole exp
                        # traceback.print_exc()
                        print(f"    Error with ratio {ratio}: {str(e)}")
                        continue
                
                # End of Ratio Loop
                if best_model is not None:
                    print(f"    Best Ratio: {best_ratio}, Valid G-Mean: {best_valid_score:.4f}")
                    
                    # Test Evaluation
                    if hasattr(best_model, "predict_proba"):
                        y_test_proba = best_model.predict_proba(X_test)[:, 1]
                    else:
                        y_test_proba = best_model.predict(X_test)
                        
                    y_test_pred = best_model.predict(X_test)
                    
                    metrics = evaluate_model(y_test, y_test_pred, y_test_proba)
                    
                    # Add metadata
                    metrics.update({
                        'dataset': dataset,
                        'method': method,
                        'imb_method': imb_method,
                        'best_ratio': best_ratio,
                        'valid_gmean': best_valid_score
                    })
                    
                    dataset_results.append(metrics)
                    
        # Save Dataset Results
        if dataset_results:
            df_res = pd.DataFrame(dataset_results)
            save_path = os.path.join(RESULT_DIR, f'{dataset}_refactored_results.csv')
            df_res.to_csv(save_path, index=False)
            print(f"  Saved results for {dataset} to {save_path}")
        else:
            print(f"  No results generated for {dataset}.")
            
    print("\nExperiment Completed.")

if __name__ == "__main__":
    run_experiment()
