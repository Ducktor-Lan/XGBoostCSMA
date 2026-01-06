
import os

# --- Random Seed ---
SEED = 2024

# --- Paths ---
# Base directory is the project root (one level up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
SHUFFLE_DIR = os.path.join(BASE_DIR, 'shuffle_index')

# Make sure result directory exists
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Datasets ---
# List of datasets to process
DATASETS = [
    'T1',
    'T2',
    'T3',
    'T4'
]

# --- Methods ---
# Classifiers to use
METHODS = ['xgb'] 
# Available: 'lr', 'lda', 'dt', 'rf', 'knn', 'xgb', 'lgb', 'ada', 'gbdt'

# --- Imbalance Handling Methods ---
USE_UNDERSAMP = False
USE_OVERSAMP = False
USE_HYBRID = False
USE_ENSEMBLE = True
USE_COST_SENSITIVE = False

IMB_METHODS = []

if USE_UNDERSAMP:
    IMB_METHODS.extend(['RUS', 'ENN', 'NCR', 'Tomek', 'ALLKNN', 'NM', 'CC'])

if USE_OVERSAMP:
    IMB_METHODS.extend(['ROS', 'SMOTE', 'ADASYN', 'BorderlineSMOTE'])

if USE_HYBRID:
    IMB_METHODS.extend(['SMOTETomek'])

if USE_ENSEMBLE:
    IMB_METHODS.extend([
        'SMOTEBoost', 'SMOTEBagging', 'RUSBoost', 'UnderBagging', 
        'BalanceCascade', 'BCRF', 'HUE', 'SelfPacedEnsemble', 'ease'
    ])

if USE_COST_SENSITIVE:
    IMB_METHODS.extend(['MetaCost'])

# --- Sampling Strategies ---
if USE_COST_SENSITIVE:
    # Cost ratios for cost-sensitive learning
    SAMPLING_STRATEGIES = [
        9.000000000000002, 10.111111111111114, 11.500000000000007,
        13.285714285714295, 15.666666666666682, 19.00000000000003,
        24.000000000000046, 32.33333333333343, 49.000000000000234,
        99.00000000000102, 110.11111111111101, 123.99999999999989,
        141.85714285714272
    ]
else:
    # Sampling ratios for resampling methods
    SAMPLING_STRATEGIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Split Ratios
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
# Test ratio is the remainder
