# Project 1 - Synapse - Binary Classification

This project implements a binary classification model using regularized logistic regression with gradient descent to predict labels for a given dataset.

## Project Overview

The goal of this project is to build a binary classifier that can accurately predict labels (-1 or 1) to predict the presence of myocardial infarction or angina in test data. The solution uses a regularized logistic regression model with several optimization techniques including feature selection, class balancing, and gradient descent optimization.

## Project Structure

```
project-1-synapse/
├── dataset/                          # Dataset directory
│   ├── x_train.csv                   # Training features
│   ├── x_test.csv                    # Test features
│   ├── y_train.csv                   # Training labels
│   └── sample_submission.csv         # Sample submission format
├── predictions/                      # Generated prediction files
│   ├── ...csv                        # Various prediction outputs
├── scripts/                          # Python modules
│   ├── helper.py                     # Data loading and utility functions
│   ├── preprocessing.py              # 2026: cleaning, feature selection, class balancing
│   ├── data_experiments.py           # 2026: compares preprocessing / balancing strategies
│   ├── models.py                     # Model classes (RegularizedLogisticRegression)
│   ├── evaluate_utils.py             # Evaluation metrics and visualization
│   ├── data_utils.py                 # Data exploration utilities
│   └── model_utils.py                # Model visualization utilities
├── exploring.ipynb                   # Contains hyperparameter tuning with 5-fold CV
├── run.ipynb                         # Main submission notebook
├── implementations.py                # The 6 required ML methods (graded; must stay at repo root)
├── project1_description.pdf          # Project description document (Fall 2026)
└── README.md                         # This file
```

## Key Features & Techniques

### 1. **Regularized Logistic Regression**
   - Lambda regularization to prevent overfitting
   - Gradient descent optimization
   - Handles class imbalance using negative sample reduction

### 2. **Feature Selection**
   - Selects top N features based on absolute correlation with labels
   - Default: 56 most correlated features

### 3. **Class Balancing**
   - Negative sample reduction technique
   - Reduces the influence of negative samples to balance the dataset
   - reduction factor: 0.14

### 4. **Data Preprocessing**
   - NaN values replaced with 0
   - Feature normalization (min-max scaling to [0, 1])
   - Label format conversion: labels are mapped from {-1, 1} to {0, 1}
     immediately before calling the optimization routines
     (`logistic_regression`, `reg_logistic_regression`) as required by the
     project specification

### 5. **Model Hyperparameters**
   
   These hyperparameters were determined through extensive **5-fold cross-validation** in `exploring.ipynb`:
   - **Learning rate (gamma)**: 1.4
   - **L2 regularization (lambda_)**: 1.5e-4
   - **Max iterations**: 500
   - **Max features**: 56
   - **Negative reduction**: 0.14
   
   The hyperparameter search process tested various combinations of gamma values (ranging from 0.1 to 4+), max_features (ranging from 5 to 100+), and negative_reduction values to find optimal performance based on cross-validation validation loss and F1 scores.

## Data Pipeline (2026 update)

`scripts/preprocessing.py` replaces the ad-hoc preprocessing above with a
pipeline that is **fitted on the training split only** and then applied to the
test split (last year min-max scaling was fitted separately on the test set).

### Cleaning and encoding (`Preprocessor`)
1. Drop 31 survey-administration columns (interview date, phone bookkeeping,
   `SEQNO`, `_PSU`, sampling weights, ...).
2. Map BRFSS special codes: `7/9`, `77/99`, `777/999`, `7777/9999`
   ("don't know" / "refused") become NaN; `88/888/8888` ("none") become 0.
   Without this, "refused" = 9 sits at the top of ordinal scales and flips
   the sign of correlations (e.g. `BPHIGH4`).
3. Drop columns with more than 50 % missing values and constant columns
   (321 -> 151 columns kept).
4. Column types: binary (2 values, mapped to 0/1), categorical (3-8 values,
   one-hot encoded, first category dropped) and numeric (kept as a number).
5. Impute with the training median (numeric) or mode (binary / categorical)
   and add a `<name>_missing` indicator for columns with >= 5 % missing
   values. Several indicators carry real signal (e.g. `_HCVU651_missing`).
6. Z-score standardise with training mean / std and prepend a bias column
   (327 output columns).

### Feature selection
- `rank_features(x, y)`: columns sorted by absolute point-biserial correlation
  with the label (computed on cleaned, encoded data).
- `drop_collinear(x, y, threshold=0.95)`: removes one column of each pair with
  |corr| > 0.95, keeping the one more correlated with the label
  (327 -> 231 columns).
- `select_features(x, y, k, protect=[0])`: collinearity filter followed by
  top-`k` correlation ranking; the bias column is always kept.

### Class balancing (positives are 8.8 % of the data)
- `undersample(x, y, neg_per_pos)` / `oversample(x, y, neg_per_pos)`: done
  **once** before training (last year resampled at every gradient step).
- `class_weights(y)` + `weighted_reg_logistic_regression(...)`: cost-sensitive
  training, no resampling.
- `best_f1_threshold(probs, y_val)`: tunes the decision threshold on a
  validation split.

### Results (`python -m scripts.data_experiments`)
Same model for every row (regularised logistic regression, GD, 400 steps,
gamma 0.5, lambda 1e-4). Split 70 / 10 / 20 train / val / test, seed 42.
Threshold tuned on the validation split; scores on the test split.

| Pipeline | Features | Balancing | Acc @ tuned thr | F1 @ tuned thr |
|---|---|---|---|---|
| last year (NaN->0, min-max, per-step neg. reduction) | 56 | neg. reduction 0.14 | 0.845 | 0.386 |
| new | 60 | none | 0.867 | 0.422 |
| new | 60 | undersample 1:3 | 0.864 | 0.422 |
| new | 100 | undersample 1:3 | 0.864 | 0.423 |
| new | all (230) | none | 0.875 | **0.427** |
| new | all (230) | class weights | 0.878 | 0.426 |
| new | all (230) | undersample 1:3 | 0.880 | 0.425 |

Take-aways:
- Cleaning + proper encoding is worth about +0.04 F1 and +0.03 accuracy over
  last year's pipeline.
- Once the threshold is tuned, all balancing strategies land within 0.005 F1
  of each other; for logistic regression, resampling mostly shifts the
  intercept. `undersample 1:3` is a good default because it also gives the
  best F1 at the untuned 0.5 threshold (0.425) and trains on 4x fewer rows.
- F1 saturates around 60-100 selected features; using all 230 non-collinear
  columns adds only ~0.005.

Recommended usage:
```python
from scripts.preprocessing import Preprocessor, load_feature_names, select_features, undersample
names = load_feature_names("dataset")
pre = Preprocessor(names)
tx_train = pre.fit_transform(x_train)          # fit on train only
tx_test = pre.transform(x_test)
cols = select_features(tx_train, y01_train, k=100, protect=[0])
tx_bal, y_bal = undersample(tx_train[:, cols], y01_train, neg_per_pos=3, seed=42)
```

##  Getting Started

### Prerequisites

- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab


### Running the Project

**Note**: The hyperparameters used in `run.ipynb` have already been optimized through 5-fold cross-validation in `exploring.ipynb`. You do not need to run `exploring.ipynb` as it is very large and time-consuming . The optimized parameters are already set in `run.ipynb`.

**Train and Generate Predictions**:
   
   The `run.ipynb` notebook contains the complete pipeline:
   - Data loading
   - Preprocessing
   - Model training
   - Evaluation
   - Prediction generation

2. **Output**:
   - Predictions will be saved in `predictions/synapse-logreg-filtered-balanced-final.csv`
   - Evaluation metrics (Accuracy, Precision, Recall, F1) are displayed during training

## Model Architecture

### RegularizedLogisticRegression

The main model class implements:

1. **Feature Selection**: Sorts features by absolute correlation with labels and selects top N
2. **Weight Initialization**: Random initialization
3. **Training Loop**:
   - Applies negative sample reduction for class balancing
   - Converts labels from {-1, 1} to {0, 1} right before entering the
     logistic optimization functions (`reg_logistic_regression` expects
     {0,1})
   - Performs gradient descent with L2 regularization
   - Tracks loss per iteration
   - Supports early stopping based on loss convergence
4. **Prediction**: 
   - Uses sigmoid function to get probabilities
   - Maps probabilities >= 0.5 to class 1, otherwise -1

##  Code Modules

### `scripts/helper.py`
- `load_csv_data()`: Loads training and test data from CSV files
- `standardize_zero_to_one()`: Normalizes features to [0, 1] range
- `create_csv_submission()`: Creates submission CSV file
- `random_split_data()`: Splits data into train/test sets
- `random_k_fold()`: Splits data into k folds for cross-validation
- `negative_reduction()`: Reduces negative samples for class balancing
- `sort_abs_correlation()`: Sorts features by correlation with labels

### `scripts/preprocessing.py`
- `Preprocessor`: train-fitted cleaning / imputation / one-hot / standardisation
- `select_features()`, `rank_features()`, `drop_collinear()`: feature selection
- `undersample()`, `oversample()`, `class_weights()`, `best_f1_threshold()`: balancing

### `scripts/data_experiments.py`
- Reproduces the results table in *Data Pipeline (2026 update)*

### `scripts/models.py`
- `Model`: Abstract base class for all models
- `RegularizedLogisticRegression`: Main classification model
- `MSEGradientDescent`: Alternative model for comparison
- `AlwaysNegative`, `BernoulliNaiveBayes`: Baseline models

### `implementations.py`
- Core machine learning algorithms:
  - `reg_logistic_regression()`: Regularized logistic regression with gradient descent
  - `logistic_loss()`: Logistic loss computation
  - `sigmoid()`: Sigmoid activation function
  - Other utility functions (MSE, ridge regression, etc.)

### `scripts/evaluate_utils.py`
- `evaluate_model()`: Comprehensive model evaluation with visualization
- `confusion_matrix()`: Computes confusion matrix components
- `accuracy()`, `precision()`, `recall()`, `f1_score()`: Classification metrics

### `scripts/model_utils.py`
- `cross_validation()` and `cross_validation_f1()`: Perform k-fold cross-validation
- `tune()` and `tune_f1()`: Hyperparameter tuning with cross-validation (used in `exploring.ipynb` with k=5)
- `plot_losses()`, `plot_confusion_matrix()`: Visualization utilities
- `plot_tuning_results()`: Visualize hyperparameter tuning results

## Results

The model is evaluated using standard classification metrics:
- **Accuracy**: ~79%
- **Precision**: ~25%
- **Recall**: ~70%
- **F1 Score**: ~37%


##  Customization

You can modify hyperparameters in `run.ipynb`:

```python
model = RegularizedLogisticRegression(
    gamma=1.4,                    # Learning rate
    max_iters=500,                # Maximum iterations
    lambda_=1.5e-4,              # L2 regularization parameter
    max_features=56,              # Number of top features to use
    negative_reduction=0.14       # Negative sample reduction factor
)
```

## Notes

- The model accepts labels in {-1, 1} format and converts them to {0, 1}
  immediately before calling `logistic_regression`/`reg_logistic_regression`,
  per the project description
- Random seed is set to 42 for reproducibility
- Training data is split 80/20 for validation in `run.ipynb`
- **Hyperparameter Selection**: The hyperparameters used in `run.ipynb` (gamma=1.3, lambda_=1.5e-4, max_features=56, negative_reduction=0.14) were determined through extensive **5-fold cross-validation** in `exploring.ipynb`. This notebook contains the full hyperparameter search process.



