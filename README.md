# Confidence-scaled Margin Adaptation Boosting

Code implementation for "Confidence-scaled margin adaptation boosting for interpretable financial distress prediction".

## Reproducibility Package Information

- **Date of assembly**: 2025-01-07  
- **Author**: Xingyu Lan  
- **Contact**: xingyu-lan@outlook.com


## Data Availability

The data that support the findings of this study are available from the China Stock Market & Accounting Research (CSMAR) database. Restrictions apply to the availability of these data, which were used under license for the current study and are not publicly available. The data are proprietary and subject to commercial copyright protection. However, data are available from the authors upon reasonable request and with permission of CSMAR, or can be obtained directly by third parties via the CSMAR website ([https://data.csmar.com/](https://data.csmar.com/)) subject to subscription fees or institutional access.

For detailed instructions on obtaining the data, please refer to [dataset/README.md](dataset/README.md).

## Project Overview

This project implements the CSMA-Boosting algorithm and provides a framework for evaluating financial distress prediction models. It includes both a modular experiment framework for large-scale evaluation and standalone scripts for specific model training.

## Environment Setup

The comparative experiments in this study involve a diverse set of imbalanced learning algorithms that rely on strict version compatibility between python, scikit-learn, imbalanced-learn, and the specialized imbalanced-ensemble library. Inconsistencies in library versions may lead to reproducibility failures. Therefore, to ensure a seamless replication process, we explicitly manage dependencies using Conda. A ready-to-use configuration file (`environment.yml`) is provided in the root directory of the repository, allowing users to clone the exact experimental environment with a single command:

```bash
conda env create -f environment.yml
conda activate ease
```

## Project Structure

A comprehensive guide to the files and directories in this repository.

*   **`dataset/`**
    *   This directory serves as a placeholder for the datasets used in the experiments.
    *   The raw data files are proprietary and cannot be redistributed due to licensing restrictions.
    *   Authorized users can obtain the data directly from the CSMAR database and place the processed CSV files in this directory following the instructions in `dataset/README.md`.

*   **`modular_experiment/`**
    *   *Framework for running large-scale, configurable benchmarks.*
    *   `config.py`: Central configuration file. Defines the list of datasets, classifiers (`METHODS`), imbalance handling methods (`IMB_METHODS`), and parameters like `SEED`.
    *   `data.py`: Handles data loading and preprocessing. Contains functions to split data into train, validation, and test sets.
    *   `main.py`: The main entry point script. Iterates through all configurations defined in `config.py`, trains models, and saves results.
    *   `models.py`: Factory module. Contains `get_classifier`, `get_sampler`, and `get_ensemble_clf` functions to instantiate models (e.g., XGBoost, Random Forest) and sampling techniques (e.g., SMOTE, RUS).
    *   `metrics.py`: Defines evaluation metrics used to assess model performance, such as G-mean, AUC, F1-score, etc.

*   **`src/`**
    *   *Core source code and standalone scripts for single-model training.*
    *   `csma_loss.py`: Implements the **Confidence-Scaled Margin Adaptation (CSMA)** objective function. This contains the custom loss logic used by the boosting model.
    *   `model.py`: Defines the `XGBoostCSMA` class. This is a wrapper around XGBoost that injects the custom CSMA objective function during training.
    *   `train.py`: A standalone script to train a `XGBoostCSMA` model on a single dataset (default T1). It performs hyperparameter search, trains the final model, evaluates it, and plots SHAP explanations.
    *   `utils.py`: Utility functions for data loading (`load_and_preprocess_data`), metric calculation (`calc_metrics`, `g_mean_score`), and other helpers.
    *   `example.ipynb`: A simple Jupyter Notebook demonstrating the workflow of loading data, training the CSMA model, and viewing results step-by-step.

*   **`production/`**
    *   *Directory for generated outputs and analysis figures.*
    *   `Figure6/`: Subdirectory for storing generated figures (e.g., SHAP summary plots).
    *   `gmean_matrix_Mfull/`: Subdirectory for storing performance matrices (e.g., G-mean scores across different ratios).

## Usage

### 1. Modular Experiment (Recommended)

To run the comprehensive experiment suite (multiple datasets, methods, and sampling strategies):

1.  Modify `code/modular_experiment/config.py` to select the datasets and methods you want to run.
2.  Run the main script from the `code` directory:

    ```bash
    python -m modular_experiment.main
    ```

### 2. Loss Functions

The framework supports two underlying loss functions for the CSMA (Confidence-Scaled Margin Adaptation) wrapper:

1.  **Standard Cross-Entropy (Default)**
    *   `loss_type='cross_entropy'`
    *   Standard logistic loss.

2.  **Weighted Cross-Entropy**
    *   `loss_type='weighted_cross_entropy'`
    *   Weighted version: `crossent = -pos_weight * (t * log(p)) - neg_weight * ((1 - t) * log(1 - p))`
    *   This allows adjusting the importance of positive and negative classes.

CSMA acts as a probability mapping wrapper that can be combined with these base losses. It maps the original probability (0.5 threshold) into a dynamic interval scaled by the confidence margin.

### 3. Standalone Training

To run a specific training session for the CSMA-Boosting model (default on T1 dataset):

```bash
cd src
python train.py
```

This script will:
*   Perform hyperparameter grid search.
*   Train the final model.
*   Evaluate on the test set.
*   Generate SHAP explanation plots.
