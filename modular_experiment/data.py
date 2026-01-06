
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .config import DATASET_DIR, SHUFFLE_DIR, SEED, TRAIN_RATIO, VALID_RATIO

def load_dataset(dataset_name):
    """
    Load dataset from CSV.
    Assumes file naming convention dataset_name.csv in data directory.
    """
    # Construct path
    csv_path = os.path.join(DATASET_DIR, f'{dataset_name}.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    data = pd.read_csv(csv_path, encoding='gbk')
    
    # Separate features and labels
    # Assumes last column is label
    # drop first column (ID?) if it looks like an index
    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'], axis=1)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Drop identifier columns
    cols_to_drop = ['code', 'year']
    X = X.drop([c for c in cols_to_drop if c in X.columns], axis=1)
    
    return X, y

def get_data_splits(dataset_name, use_scaler=True):
    """
    Load data, split into Train/Valid/Test, and apply preprocessing 
    (Imputation + Scaling) WITHOUT data leakage.
    """
    X, y = load_dataset(dataset_name)
    columns = X.columns
    
    # --- splitting ---
    # Try to load fixed shuffle indices if available
    # Suffix is same as dataset name now, e.g. T1
    shuffle_file = os.path.join(SHUFFLE_DIR, 'Zfull', f'{dataset_name}.pickle')
    
    if os.path.exists(shuffle_file):
        print(f"Loading shuffle index from {shuffle_file}")
        with open(shuffle_file, 'rb') as f:
            shuffle_index = pickle.load(f)
            
        n_samples = X.shape[0]
        n_train = int(n_samples * TRAIN_RATIO)
        n_valid = int(n_samples * VALID_RATIO)
        
        train_idx = shuffle_index[:n_train]
        valid_idx = shuffle_index[n_train : n_train + n_valid]
        test_idx = shuffle_index[n_train + n_valid:]
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    else:
        print(f"Shuffle index not found using random split (SEED={SEED}).")
        # Fallback to random split if pickle doesn't exist
        # Split Train vs (Valid + Test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=TRAIN_RATIO, random_state=SEED, stratify=y
        )
        # Split Valid vs Test
        remaining_ratio = 1.0 - TRAIN_RATIO
        valid_portion = VALID_RATIO / remaining_ratio
        
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, train_size=valid_portion, random_state=SEED, stratify=y_temp
        )

    # --- Preprocessing (Fixing Leakage) ---
    
    # 1. Imputation
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    # Fit ONLY on training data
    X_train = imputer.fit_transform(X_train)
    # Transform valid and test
    X_valid = imputer.transform(X_valid)
    X_test = imputer.transform(X_test)
    
    # 2. Scaling
    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        
    # Convert back to DataFrame/Series for convenience if needed, 
    # but numpy arrays are usually better for sklearn/xgboost 
    # (except for column name preservation). 
    # To be safe with some imblearn methods that might like DataFrames, 
    # we can recreate them.
    X_train = pd.DataFrame(X_train, columns=columns)
    X_valid = pd.DataFrame(X_valid, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)
    
    # Reset indices for y
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

