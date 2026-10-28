# Heart-disease classification from BRFSS 2015

*EPFL CS-433 Machine Learning, Project 1 — Jiawei Lin, Reza Abbasi, and Neil
Chadli*

This project classifies the course target `_MICHD` from the 2015 Behavioral
Risk Factor Surveillance System (BRFSS). A positive label means that a
respondent reported coronary heart disease or myocardial infarction. The model
is written using only NumPy and the Python standard library.

The main modeling choice is to preserve missing answers. BRFSS uses conditional
questions, so an unanswered item can describe the path a respondent took
through the survey. The final system combines four missing-aware
gradient-boosted decision tree models with a regularized logistic regression.

> This is a classifier of self-reported disease status. It must not be
> interpreted as a model of future clinical risk.

## Quick start

### Requirements

- Python 3.9 or newer
- NumPy
- About 1.1 GB of free space for the parsed-data cache

Run all commands from the repository root. If `dataset/` is not already
present, extract the supplied course data first:

```sh
unzip -n dataset.zip
python run.py
```

`python run.py` loads `models/final_model.npz` and writes
`predictions/final_submission.csv`. It also checks whether the generated file
is byte-identical to the submitted prediction file. The first run builds a
cache under `outputs/cache/`; later runs reuse it. Runtime is roughly 30
seconds with an existing cache on the machine used for this project. The first
run also parses the CSV files and can take longer.

Use `python run.py --help` to select different data, model, output, or cache
paths.

## Method

### Data preparation

The raw data contain 328,135 labeled rows, 109,379 test rows, and 321 survey
variables. The preparation performed in `features.py` is:

1. Remove 30 administrative and survey-design columns, leaving 291 model
   variables.
2. Recode BRFSS values meaning "do not know," "refused," or "not asked" as
   missing values.
3. Recode values meaning "none" as zero rather than missing.
4. Keep highly missing variables because their missingness can reflect survey
   routing.

Every learned preprocessing statistic is fitted on training data only. During
cross-validation, this means fitting it separately inside each training fold.

### Feature generation

The two model families use different representations:

- **Tree features:** each variable is converted to at most 64 observed-value
  bins plus a separate missing-value bin. Every split learns whether missing
  values should go left or right.
- **Logistic features:** low-cardinality variables are one-hot encoded; rare
  levels are grouped as "other." Continuous variables are clipped,
  standardized, and expanded with ten quantile-bin indicators. Variables with
  missing training values also receive a missingness indicator.

### Models and validation

| Component | Principal settings |
|---|---|
| Validation | Five stratified folds; seed 2026; fold-fitted preprocessing |
| Wide logistic regression | Newton optimization; L2 penalty `1e-4` |
| Four GBDTs | Seeds 1-4; depth 8; learning rate 0.03; 64 observed bins plus missing; minimum leaf size 50; row/column sampling 0.85/0.40; L2 penalty 10 |
| Final ensemble | Average the four GBDTs, then blend 85% tree and 15% logistic probabilities |
| Decision rule | Predict `+1` at probability `0.2131444789469242` or above; otherwise predict `-1` |

The stored GBDTs contain 566, 550, 598, and 583 trees. These counts and the
decision threshold were derived from five-fold out-of-fold predictions before
the components were refitted on all labeled rows.

## Validation results

The following positive-class F1 scores use the same five stratified folds and
are rounded to three decimals.

| Model | F1 |
|---|---:|
| Ridge regression | 0.420 |
| Ridge with missingness indicators | 0.429 |
| Regularized logistic regression with missingness indicators | 0.437 |
| Wide logistic regression | 0.441 |
| Single GBDT (mean F1 across four seeds) | 0.447 |
| **Final ensemble** | **0.449** |

Selecting each fold's threshold using the other folds gives F1 0.4477, close
to the pooled out-of-fold value of 0.4485. The comparison shows that preserving
missingness and modeling nonlinear interactions provide the main gains;
averaging and blending add a smaller improvement.

## Reproducing training and evaluation

### Full retraining

```sh
python run.py --train
```

This command reads the training recipe from `models/final_model.npz`, performs
five-fold cross-validation, chooses the threshold and tree counts, refits every
component on all labeled rows, and writes:

```text
outputs/folds.npz
outputs/cv.npz
outputs/final_model.npz
outputs/final_submission.csv
```

Retraining takes about 30-45 minutes on a 12-core machine and is memory
intensive. On a smaller machine, reduce parallelism:

```sh
python run.py --train --workers 2 --threads 1
```

### Evaluation scripts

After retraining, reproduce the final-model metrics with:

```sh
python -m scripts.evaluate_cv
```

Reproduce the ridge and logistic course baselines with:

```sh
python -m scripts.course_baselines
```

The baseline script reconstructs the folds from the saved recipe and does not
require a full retraining run. Both scripts accept `--json PATH` for
machine-readable results. The older `scripts.data_experiments` module contains
optional exploratory experiments and is not part of the final reproduction
path.

## Required course implementations

`implementations.py` contains the six functions required by the assignment:

| Function | Method |
|---|---|
| `mean_squared_error_gd` | Linear regression by full-batch gradient descent |
| `mean_squared_error_sgd` | Linear regression by stochastic gradient descent |
| `least_squares` | Least squares through the normal equations |
| `ridge_regression` | L2-regularized least squares |
| `logistic_regression` | Logistic regression by gradient descent |
| `reg_logistic_regression` | L2-regularized logistic regression |

Each function returns `(weights, loss)`. For the regularized methods, the
returned loss does not include the penalty term, as required by the project
specification.

## Repository structure

| Path | Purpose |
|---|---|
| `run.py` | Reproduce submitted predictions or retrain the complete model |
| `features.py` | CSV loading, BRFSS recoding, caching, and feature generation |
| `model.py` | NumPy GBDT, wide logistic regression, ensemble, folds, and model serialization |
| `implementations.py` | Six required course algorithms |
| `helpers.py` | Course CSV submission helper |
| `models/final_model.npz` | Saved submitted model and training recipe |
| `predictions/final_submission.csv` | Submitted predictions |
| `scripts/evaluate_cv.py` | Final-model cross-validation metrics |
| `scripts/course_baselines.py` | Ridge and logistic baseline evaluation |

Generated caches and retraining artifacts are written under `outputs/`.

## Limitations

- Some high-signal variables are follow-up questions asked after a reported
  cardiovascular event. The system therefore learns survey routing as well as
  health and lifestyle patterns.
- Survey-routing patterns may not transfer to a differently designed
  questionnaire.
- GBDT early stopping uses the fold later scored, and the pooled F1 threshold
  is selected from the same out-of-fold predictions. The reported score is
  therefore slightly optimistic and is not an independent test estimate.
- Probability calibration was not evaluated.

No external datasets or external machine-learning libraries are used.
