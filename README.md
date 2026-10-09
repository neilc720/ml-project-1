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
│   ├── implementations.py            # Core ML algorithms (GD, logistic regression)
│   ├── models.py                     # Model classes (RegularizedLogisticRegression)
│   ├── evaluate_utils.py             # Evaluation metrics and visualization
│   ├── data_utils.py                 # Data exploration utilities
│   └── model_utils.py                # Model visualization utilities
├── exploring.ipynb                   # Contains hyperparameter tuning with 5-fold CV
├── run.ipynb                         # Main submission notebook
├── project1_description.pdf          # Project description document
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

### `scripts/models.py`
- `Model`: Abstract base class for all models
- `RegularizedLogisticRegression`: Main classification model
- `MSEGradientDescent`: Alternative model for comparison
- `AlwaysNegative`, `BernoulliNaiveBayes`: Baseline models

### `scripts/implementations.py`
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



