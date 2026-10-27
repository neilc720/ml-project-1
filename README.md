# Heart-disease prediction from BRFSS 2015 (CS-433 Project 1)

Jiawei Lin, Reza Abbasi and Neil Chadli · EPFL CS-433 Machine Learning, fall 2026

We predict `_MICHD`, self-reported coronary heart disease or myocardial infarction, from the 321
survey features of the course's extract of the 2015 Behavioral Risk Factor Surveillance System
(BRFSS). Predictions are scored by F1 on the positive class, and only Python's standard library and
NumPy may be used.

Our model averages four gradient-boosted decision trees (GBDTs) and blends them with a wide logistic
regression, all written in NumPy. Its key ingredient is treating missing answers as information:
BRFSS asks many questions only after a qualifying answer, so whether a question was answered says
something about the respondent. On five-fold cross-validation over all 328,135 labelled rows the
model reaches **F1 0.4485** (0.4477 when each fold's threshold is chosen on the other folds),
ROC-AUC 0.8708 and average precision 0.4368. `python run.py` regenerates the submitted predictions,
`predictions/final_submission.csv`, byte for byte.

## Results

All models below are scored on the same five folds over all labelled rows. F1 chooses the decision
threshold on the pooled out-of-fold predictions; cross-fold F1 chooses each fold's threshold on the
other four folds only.

| Model | F1 | Cross-fold F1 | ROC-AUC |
|---|---|---|---|
| Ridge regression¹ | 0.4200 | 0.4188 | 0.8536 |
| + missingness indicators | 0.4285 | 0.4278 | 0.8574 |
| Regularised logistic regression, gradient descent¹ | 0.4295 | 0.4290 | 0.8604 |
| + missingness indicators | 0.4374 | 0.4371 | 0.8639 |
| Wide logistic regression (Newton's method) | 0.4406 | 0.4401 | 0.8656 |
| One GBDT (mean over seeds 1–4) | 0.4467 | 0.4456 | 0.8702 |
| Four-GBDT bag | 0.4481 | 0.4470 | 0.8707 |
| **Final blend (submitted)** | **0.4485** | **0.4477** | **0.8708** |
| Final blend at threshold 0.5 | 0.2494 | – | 0.8708 |

¹ Course methods from `implementations.py` on mean-imputed, standardised features with centred
missingness indicators, λ = 10⁻⁴. Gradient descent runs 10,000 steps and ends within 0.0005 F1 of
Newton's method on the same objective.

- **The threshold matters most.** Only 8.83% of the rows are positive, and at the default threshold
  0.5 the blend's F1 drops from 0.4485 to 0.2494. The F1-optimal threshold, 0.2131, labels 13.1% of
  the out-of-fold rows and 13.3% of the test rows positive.
- **Missing answers carry signal.** Missingness indicators add 0.0085 F1 to ridge regression and
  0.0079 to logistic regression. In development experiments on 85% of the rows, an earlier GBDT
  configuration lost 0.0096 F1 when missing values were median-imputed and 0.0156 when columns with
  more than 50% missing values were dropped.
- **Nonlinear models help; bagging and blending barely do.** The wide features add 0.0032 over the
  best course baseline and one GBDT another 0.0060. Bagging four seeds (+0.0014) and adding the
  logistic regression (+0.0004) are close to the 0.0009 seed-to-seed spread of a single GBDT.

## Quick start

```sh
unzip -n dataset.zip   # creates dataset/ with x_train.csv, y_train.csv and x_test.csv
python run.py          # models/final_model.npz -> predictions/final_submission.csv
```

`run.py` takes about 30 seconds. It caches the parsed CSVs in `outputs/cache/` and ends with

```text
Saved 109379 predictions to .../predictions/final_submission.csv (positive rate 0.133)
The CSV is byte-identical to the submitted predictions.
```

The second line compares the SHA-256 of the written file with that of the submitted predictions.
`python run.py --help` lists the options for other data folders, model files and output paths.

## Requirements

Python 3.9 or newer and NumPy, nothing else. The submitted CSV is reproduced byte for byte with
Python 3.9 and NumPy 1.23 (the course's grading environment) and with Python 3.13 and NumPy 2.3 and
2.5.

## Repository structure

| Path | Contents |
|---|---|
| `run.py` | Entry point: predictions from the saved model, or `--train` to cross-validate and refit it |
| `features.py` | CSV loading and caching, BRFSS code recoding, tree binning, wide logistic-regression features |
| `model.py` | Histogram GBDT, Newton logistic regression, F1 threshold, stratified folds, blended model, saving and loading |
| `implementations.py` | The six required course methods |
| `helpers.py` | Course helpers; `run.py` writes its CSV with `create_csv_submission` |
| `models/final_model.npz` | The submitted model: weights and training recipe as arrays and JSON, loaded without pickle |
| `predictions/final_submission.csv` | The submitted predictions: `Id,Prediction`, 109,379 rows, labels −1 and 1 |
| `scripts/evaluate_cv.py` | Final-model rows of the results table, from the out-of-fold predictions of `--train` |
| `scripts/course_baselines.py` | Course-method rows of the results table, on the final model's folds |
| `scripts/preprocessing.py`, `scripts/data_experiments.py`, `scripts/helper.py` | Earlier logistic-regression experiments with feature selection and class balancing; run `python -m scripts.data_experiments --quick` |
| `dataset.zip` | The course data |
| `project1_description.pdf` | The project handout |
| `outputs/` | Generated and git-ignored: parsed-CSV cache and results of `--train` and the scripts |

## Pipeline

### 1. Loading and recoding (`features.py`)

- `load_data` parses the three CSVs once into `.npy` files in `outputs/cache/`, keyed by each CSV's
  SHA-256, checks that the feature and label IDs line up, and maps the labels −1/+1 to 0/1. Of the
  328,135 training rows, 28,975 (8.83%) are positive.
- Thirty interview, telephone and survey-design columns are dropped (`IDENTIFIER_COLUMNS`). `_STATE`
  is kept, which leaves 291 features.
- A per-column table built from the 2015 BRFSS codebook (`SENTINELS`) maps "don't know", "refused"
  and missing codes to NaN and "none" codes, such as 88 for "no days", to 0. Columns without such
  codes are left unchanged.
- Mostly-missing columns are kept, because many questions are asked only after a qualifying answer.
  After recoding, `HAREHAB1` (outpatient rehabilitation after a heart attack) is answered by only
  758 training respondents, all of them positive.

### 2. Tree features (`TreeBinner`)

Each column is binned to `uint8`. Bin 0 holds missing values and at most 64 bins hold observed
values: a column with at most 64 distinct values gets one bin per value, and any other column gets
quantile cut points computed from up to 100,000 sampled values (seed 2026).

### 3. Wide features for logistic regression (`WideFeatureMap`)

- Columns with at most 30 distinct values, and `_STATE`, `EXRACT11` and `EXRACT21`, are categorical:
  one indicator per level that covers at least 0.1% of the rows, plus an "other" indicator for rarer
  and unseen levels.
- Other columns are clipped to their 0.5th and 99.5th percentiles and standardised, with missing
  values set to 0, and get up to ten quantile-bin indicators.
- Every column with missing training values also gets a missingness indicator.
- On all labelled rows this gives 1,938 features, including the bias.

Every statistic (cut points, levels, clipping bounds, means) is fitted on the training rows only;
inside cross-validation, on the four training folds.

### 4. Models (`model.py`)

**Gradient-boosted trees (`GBDTClassifier`).** Newton boosting on the logistic loss: starting from
the training log-odds, every tree fits the gradient `p − y` and Hessian `p(1 − p)` of the current
predictions. Trees grow level by level. For each level, one `np.bincount` per 64-feature chunk
builds the gradient, Hessian and count histograms of all nodes, and the larger child of every split
gets its histogram by subtracting the smaller child's from the parent's. Every feature, threshold
and direction for missing values is scored with the gain
`½ [G_L² / (H_L + λ) + G_R² / (H_R + λ) − G² / (H + λ)]`, so each split learns which side
unanswered questions go to. A leaf outputs `−G / (H + λ)`, clipped to ±4 and multiplied by the
learning rate. Ties are broken deterministically.

| Setting | Value | Setting | Value |
|---|---|---|---|
| Models | 4, seeds 1–4 | Trees | 566, 550, 598, 583 |
| Maximum depth | 8 | Learning rate | 0.03 |
| Rows per leaf | ≥ 50 | Hessian sum per leaf | ≥ 1 |
| L2 penalty on leaf values (λ) | 10 | Bins | 64 + missing |
| Rows per tree | 85%, without replacement | Columns per tree | 40% |

**Wide logistic regression (`NewtonLogistic`).** Mean binary cross-entropy plus 10⁻⁴ ‖w‖², with the
bias unpenalised, minimised by Newton's method with a backtracking line search. The Hessian is
accumulated over 16,384-row chunks, and the final fit converges in six iterations.

**Blend and threshold (`FinalModel`).**

```text
p = 0.85 × mean(p_GBDT1, …, p_GBDT4) + 0.15 × p_logistic
label = +1 if p ≥ 0.2131444789469242, else −1
```

The threshold maximises the F1 of the blended out-of-fold predictions, searched exactly over every
distinct score.

### 5. Cross-validation and refit (`python run.py --train`)

1. Split all labelled rows into five stratified folds (seed 2026).
2. Fit each of the five components on four folds, with every feature statistic computed on those
   rows, and predict the fifth. The 25 fits run in up to seven worker processes with one BLAS
   thread each. The GBDTs stop early on the held-out fold's log-loss, with a patience of 100 trees
   and at most `max(200, 2 × development tree count)` trees.
3. Choose the threshold on the blended out-of-fold predictions, and set each GBDT's tree count to
   `round(1.1 × mean best iteration over the folds)`.
4. Refit every component on all labelled rows, save `outputs/final_model.npz`, predict
   `x_test.csv` into `outputs/final_submission.csv`, and compare it with the submitted file.

Everything else was fixed beforehand by five-fold cross-validation on 85% of the labelled rows: the
hyperparameters, seeds, blend weights and tree-count rule. It is stored as the training recipe
inside `models/final_model.npz`, with development tree counts of 507, 543, 534 and 511.

## Required methods (`implementations.py`)

| Function | Method |
|---|---|
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` | Linear regression by full-batch gradient descent |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` | Linear regression by stochastic gradient descent, one sample per step |
| `least_squares(y, tx)` | Least squares via the normal equations |
| `ridge_regression(y, tx, lambda_)` | Ridge regression via the normal equations |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)` | Logistic regression by gradient descent, `y` in {0, 1} |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | The same with the penalty `lambda_ * ‖w‖²` |

Every function takes 1-D vectors, adds no intercept, and returns `(w, loss)`: the last weight vector
and the loss at it, without the penalty term.

- The squared loss is `‖y − tx w‖² / (2N)`, and ridge regression solves
  `(txᵀ tx + 2Nλ I) w = txᵀ y`.
- SGD samples with replacement; seed it with `np.random.seed`.
- The normal equations are solved with `np.linalg.solve`, falling back to a pseudoinverse if the
  matrix is singular; `np.linalg.lstsq` is never used.
- Inputs are validated: shapes, finite values, and 0/1 labels for the logistic methods.

`scripts/course_baselines.py` uses `ridge_regression` and `reg_logistic_regression` for the
baselines in the results table.

## Reproducing every number

| Command | Writes | Time |
|---|---|---|
| `python run.py` | `predictions/final_submission.csv` | about 30 s |
| `python run.py --train` | `outputs/final_model.npz`, `outputs/final_submission.csv`, `outputs/folds.npz`, `outputs/cv.npz` (out-of-fold probabilities and best iterations) | 30–45 min on 12 cores |
| `python -m scripts.evaluate_cv` | Logs the final-model rows of the results table and the average precision; needs `--train` first | seconds |
| `python -m scripts.course_baselines` | Logs the course-method rows of the results table | 20–40 min |

Training is deterministic: sampling is seeded, ties are broken deterministically, and every worker
uses one BLAS thread. On 27 October 2026, `python run.py --train` (Python 3.13.9, NumPy 2.3.5,
macOS arm64) rebuilt the stored model identically in every entry and the submitted CSV byte for
byte. The logistic model's 16,384-row chunks fix the order of its floating-point sums, so other
NumPy or BLAS builds can differ in the last bits of those sums.

The scripts behind the development experiments (grid search, holdout and development-fold
comparisons) are not included; every step that produces the submitted model is.

## Limitations

- **Post-diagnosis questions.** `HAREHAB1` and other cardiovascular follow-up questions are asked
  only after a reported heart attack, so part of the signal is the survey's routing. The model
  classifies reported disease status; it does not predict future events.
- **Transfer.** The model learns BRFSS's skip logic and would not carry over to data collected with a
  different questionnaire.
- **Selection effects.** The GBDTs stop early on the fold they score, and the pooled threshold is
  chosen on the scored predictions, so out-of-fold F1 is slightly optimistic. Cross-fold F1 removes
  only the threshold part. The final model has no independent holdout score; an earlier version
  scored 0.4480 on a 15% holdout, in line with its cross-validation score.
- **Calibration** of the blended probabilities was not evaluated.
