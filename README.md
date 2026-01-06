# Confidence-scaled Margin Adaptation Boosting

Code implementation for "Confidence-scaled margin adaptation boosting for interpretable financial distress prediction".

## Project Overview

This project implements the CSMA-Boosting algorithm and provides a framework for evaluating financial distress prediction models. It includes both a modular experiment framework for large-scale evaluation and standalone scripts for specific model training.

## Environment Setup

The project uses a Conda environment to manage dependencies.

1.  **Create the environment**:
    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the environment**:
    ```bash
    conda activate ease
    ```

## Project Structure

*   **`dataset/`**: Contains the dataset files (e.g., T1.csv, T2.csv, etc.).
*   **`modular_experiment/`**: A modular framework for running experiments across multiple datasets and methods.
    *   `config.py`: Configuration for datasets, methods, and parameters.
    *   `main.py`: Main entry point for running experiments.
*   **`src/`**: Core source code and standalone training scripts.
    *   `model.py`: Implementation of `XGBoostCSMA`.
    *   `train.py`: Standalone script for training CSMA-Boosting on a specific dataset.
*   **`results/`**: Directory where experiment results are saved.

## Usage

### 1. Modular Experiment (Recommended)

To run the comprehensive experiment suite (multiple datasets, methods, and sampling strategies):

1.  Modify `code/modular_experiment/config.py` to select the datasets and methods you want to run.
2.  Run the main script from the `code` directory:

    ```bash
    python -m modular_experiment.main
    ```

### 3. Loss Functions

The framework supports two underlying loss functions for the CSMA (Confidence-Scaled Margin Adaptation) wrapper:

1.  **Standard Cross-Entropy (Default)**
    *   `loss_type='cross_entropy'`
    *   Standard logistic loss.

2.  **Weighted Cross-Entropy**
    *   `loss_type='weighted_cross_entropy'`
    *   Weighted version: `crossent = -0.8 * (t * log(p)) - 0.2 * ((1 - t) * log(1 - p))`
    *   This allows adjusting the importance of positive and negative classes.

CSMA acts as a probability mapping wrapper that can be combined with these base losses. It maps the original probability (0.5 threshold) into a dynamic interval scaled by the confidence margin.

### 4. Standalone Training

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

## Results

*   **Metric Results**: Saved in `results/` (for modular experiments) or `performance/` (for standalone script).
*   **SHAP Plots**: Saved in `results/SHAP/`.
