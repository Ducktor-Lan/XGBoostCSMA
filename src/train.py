import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

from model import XGBoostCSMA
import utils

import warnings
warnings.filterwarnings("ignore", message=".*Parameters.*might not be used.*")

# --- 配置路径 ---
root_path = './' 
dataset_path = os.path.join(root_path, 'dataset/')
save_path = os.path.join(root_path, 'performance/')
results_save_path = os.path.join(root_path, 'results/')
shap_save_path = os.path.join(results_save_path, 'SHAP/')

for p in [save_path, results_save_path, shap_save_path]:
    if not os.path.exists(p):
        os.makedirs(p)

def run_experiment():
    dataset_name = 'T2'
    
    # 超参数搜索空间
    a_values = np.arange(5, 16, 1)    # mu
    c_values = np.arange(0.1, 2.1, 0.1) # v

    print(f"Loading dataset: {dataset_name}...")
    file_name = dataset_name.split('_')[-1]
    csv_path = os.path.join(dataset_path, f"{dataset_name}.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    features, labels = utils.load_and_preprocess_data(csv_path)
    
    # 数据切分 (70% Train, 15% Valid, 15% Test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features, labels, test_size=0.15, random_state=2024, stratify=labels
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, random_state=2024, stratify=y_train_full
    )
    
    print(f"Data shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # --- 1. 超参数搜索 ---
    best_score = 0
    best_params = {'a': 10, 'c': 0.5}
    gmean_results = {}

    print("Starting Hyperparameter Search...")
    for a in a_values:
        gmean_results[a] = {}
        for c in c_values:
            model = XGBoostCSMA(a=a, c=c, eval_metric='auc', early_stopping_rounds=20)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                verbose=False
            )
            
            y_valid_pred = np.rint(model.predict_proba(X_valid)[:, 1])
            score = utils.g_mean_score(y_valid, y_valid_pred)
            gmean_results[a][c] = score
            
            if score > best_score:
                best_score = score
                best_params = {'a': a, 'c': c}
                print(f"New best G-mean: {best_score:.4f} (a={a}, c={c:.1f})")

    # 保存搜索结果
    pd.DataFrame(gmean_results).T.to_csv(os.path.join(results_save_path, f'grid_search_{dataset_name}.csv'))
    
    # --- 2. 最终模型训练 ---
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
    
    # --- 3. 测试集评估 ---
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    metrics = utils.calc_metrics(y_test, y_test_proba)
    
    print("\nTest Metrics:")
    print(metrics)
    pd.DataFrame([metrics]).to_csv(os.path.join(save_path, f'XGBoost-CSMA_metrics.csv'), index=False)

    # --- 4. SHAP 解释 (Feedback Loop Component) ---
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