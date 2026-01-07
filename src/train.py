import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
import pickle

from model import XGBoostCSMA
import utils

import warnings
warnings.filterwarnings("ignore", message=".*Parameters.*might not be used.*")

# --- Configuration Paths ---
root_path = './' 
dataset_path = os.path.join(root_path, 'dataset/')
save_path = os.path.join(root_path, 'performance/')
results_save_path = os.path.join(root_path, 'results/')
shap_save_path = os.path.join(results_save_path, 'SHAP/')

for p in [save_path, results_save_path, shap_save_path]:
    if not os.path.exists(p):
        os.makedirs(p)

def run_experiment():
    dataset_name = 'T1'
    
    # Hyperparameter search space
    a_values = np.arange(5, 16, 1)    # mu
    c_values = np.arange(0.1, 2.1, 0.1) # v

    print(f"Loading dataset: {dataset_name}...")
    csv_path = os.path.join(dataset_path, f"{dataset_name}.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    features, labels = utils.load_and_preprocess_data(csv_path)
    
    # Data splitting
    with open(dataset_path + '{}.pickle'.format(dataset_name),'rb') as f:
        shuffle_index = pickle.load(f)
    train_ratio = 0.7
    valid_ratio = 0.15
    train_index = shuffle_index[:int(features.shape[0] * train_ratio)]
    valid_index = shuffle_index[int(features.shape[0] * train_ratio):int(features.shape[0] * (train_ratio + valid_ratio))]
    test_index = shuffle_index[int(features.shape[0] * (train_ratio + valid_ratio)):]
    X_train,y_train = features.iloc[train_index],labels.iloc[train_index]
    X_valid,y_valid = features.iloc[valid_index],labels.iloc[valid_index]
    X_test,y_test = features.iloc[test_index],labels.iloc[test_index]
    
    print(f"Data shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # --- 1. Hyperparameter Search ---
    best_score = 0
    best_params = {'a': 10, 'c': 0.5}
    gmean_results = {}

    print("Starting Hyperparameter Search...")
    for a in a_values:
        gmean_results[a] = {}
        for c in c_values:
            model = XGBoostCSMA(a=a, c=c, eval_metric=['auc'], early_stopping_rounds=20)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                verbose=False
            )
            
            y_valid_pred = np.rint(model.predict_proba(X_valid)[:, 1])
            score = utils.g_mean_score(y_valid, y_valid_pred)
            gmean_results[a][c] = score

            print(f"G-mean: {score:.4f} (a={a}, c={c:.1f})")
            
            if score > best_score:
                best_score = score
                best_params = {'a': a, 'c': c}
                print(f"New best G-mean: {best_score:.4f} (a={a}, c={c:.1f})")

    # Save search results
    pd.DataFrame(gmean_results).T.to_csv(os.path.join(results_save_path, f'grid_search_{dataset_name}.csv'))
    
    # --- 2. Train Final Model ---
    print(f"\nTraining Final Model with Best Params: {best_params}")
    final_model = XGBoostCSMA(
        a=best_params['a'], 
        c=best_params['c'],
        eval_metric='auc',
        early_stopping_rounds=20
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=False
    )
    
    # --- 3. Test Set Evaluation ---
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    metrics = utils.calc_metrics(y_test, y_test_proba)
    
    print("\nTest Metrics:")
    print(metrics)
    pd.DataFrame([metrics]).to_csv(os.path.join(save_path, f'XGBoost-CSMA_metrics.csv'), index=False)

    # --- 4. SHAP Explanation (Feedback Loop Component) ---
    print("\nGenerating SHAP explanations...")
    X_test_viz = X_test.copy()
    X_test_viz.columns = [f'$X_{{{i+1}}}$' for i in range(X_test.shape[1])]
    
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_viz)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_viz, plot_type='violin', show=False)
    plt.savefig(os.path.join(shap_save_path, 'summary_plot.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Experiment finished.")

if __name__ == "__main__":
    run_experiment()