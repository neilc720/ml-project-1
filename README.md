# CS-433 Project 1: heart-disease prediction from BRFSS 2015 (NumPy only)

This repository is our team's submission for EPFL CS-433 Project 1 (fall 2026). The task is to
predict `_MICHD`, self-reported coronary heart disease or myocardial infarction, from the 321 survey
columns of the course's BRFSS 2015 extract. Predictions are scored by F1 on the positive class, and
only NumPy and the Python standard library may be used.

Our model averages four gradient-boosted tree models written in NumPy and blends them with a wide
logistic regression. Its key ingredient is treating missing answers as information: BRFSS asks many
questions only after a qualifying answer, and every tree split learns which way missing values go.
On 5-fold cross-validation over all 328,135 labelled rows it reaches F1 **0.4485** (0.4477 when each
fold's threshold is chosen on the other folds). `python run.py` regenerates the submitted predictions
byte for byte, and `python run.py --train` retrains the model from the raw CSVs.

## Results

All scores are F1 on the positive class. Holdout and nested-threshold scores use thresholds chosen
without the rows being scored; pooled out-of-fold scores select and evaluate the threshold on the
same pooled predictions. The protected holdout is the same 15% of the labelled rows (49,220 rows)
for every model with a holdout score in the table, and it was read only after that model's choices
were frozen.

| Model | Holdout F1 [95% CI] | 5-fold F1 on all labelled rows, pooled / nested |
|---|---|---|
| Last year's approach, rebuilt on the same split | 0.3819 | – |
| First 2026 pipeline on the new preprocessing (logistic regression + depth-2 boosting) | 0.4302 [0.4195, 0.4414] | – |
| Final method, first version (five GBDTs + wide logistic regression) | 0.4480 [0.4363, 0.4596] | 0.4477 / 0.4464 |
| **Final model, submitted** (four GBDTs + wide logistic regression) | not scored | **0.4485** / **0.4477** |

The submitted model came out of a grid search judged by cross-validation only, so the holdout stayed
closed. Its gain over the first version is within noise: the paired bootstrap interval of the
all-rows F1 difference is [−0.0006, +0.0021], and the two versions agree on 98.9% of the test rows.
Holdout intervals are paired row bootstraps. "Nested" means that each fold's threshold was chosen on
the other four folds.

## Running the code

```sh
unzip -n dataset.zip   # creates dataset/ with x_train.csv, y_train.csv and x_test.csv
python run.py          # models/champion_v2.npz -> predictions/champion_v2.csv
python run.py --train  # optional: cross-validate and refit from the raw CSVs into outputs/
python -m scripts.evaluate_cv       # after --train: final-model metrics and ablations
python -m scripts.course_baselines  # course-method baselines on the same folds (20-40 min)
```

`run.py` needs Python 3.9 or newer and NumPy, nothing else. It reproduces the submitted CSV byte for
byte with Python 3.13.9 and NumPy 2.5.4, and also with Python 3.9.24 and NumPy 1.23.5, and it reports
whether its output matches the submission. The first run parses the CSVs into `outputs/cache/` in
about a minute; later runs take about 30 seconds.

`--train` contains the complete training procedure described below. With Python 3.13.9 and NumPy
2.5.4 it takes 30 to 45 minutes on a 12-core laptop and reproduces the submission exactly. It writes
the refit model (`champion_v2.npz`), its predictions, the fold assignment (`folds.npz`) and every
component's out-of-fold probabilities (`cv.npz`) to `outputs/`. `python run.py --help` lists the
options for other data folders, model files, output paths and worker counts.

## Repository structure

| Path | Contents |
|---|---|
| `run.py` | Entry point: predictions from the saved model, or `--train` to cross-validate and refit it |
| `features.py` | CSV loading and caching, BRFSS sentinel recoding, tree binning and wide one-hot features |
| `model.py` | Histogram GBDT, Newton logistic regression, F1 threshold, stratified folds, blended model, saving and loading |
| `models/champion_v2.npz` | Weights and training recipe of the submitted model: JSON strings and arrays, loaded without pickle |
| `predictions/champion_v2.csv` | The submitted predictions: `Id,Prediction`, 109,379 rows, labels −1 and 1 |
| `implementations.py` | The six required course methods |
| `helpers.py` | Course helpers; `run.py` writes its CSV with `create_csv_submission` |
| `scripts/preprocessing.py` | Jiawei Lin's earlier preprocessing, correlation-based feature selection and balancing utilities |
| `scripts/data_experiments.py` | Earlier logistic-regression experiments using that preprocessing and feature selection |
| `scripts/helper.py` | Helpers required by the earlier experiment script |
| `scripts/evaluate_cv.py` | Final-model metrics and ablations (pooled and cross-fold F1, ROC-AUC, average precision) from the `--train` out-of-fold predictions |
| `scripts/course_baselines.py` | Baselines built from the required methods (ridge regression, regularised logistic regression by GD) on the final model's folds |
| `dataset.zip`, `project1_description.pdf` | Course data and project handout |
| `outputs/` | Generated and git-ignored: parsed-CSV cache and `--train` results |

`scripts/preprocessing.py`, `scripts/data_experiments.py` and `scripts/helper.py` are preserved
from commit `74a2d5f`. Run the earlier experiments with
`python -m scripts.data_experiments --quick`. The current `run.py` pipeline does not call their
correlation ranking or collinearity filtering; it reuses the administrative exclusion list while
keeping `_STATE`.

## Method

### 1. Missing values carry signal

BRFSS asks many questions only after a qualifying answer. `HAREHAB1` ("following your heart attack,
did you go to outpatient rehabilitation?") is asked only of respondents who report a heart attack.
In the course data it is 99.8% missing, and all 773 respondents who answered it are positive. The
diabetes follow-up module (22% positive when asked, 7–8% when not), `BPMEDS` (16% vs 3.8%), the
cognitive-decline module (23–29%), `STREHAB1` (36%) and the heart-attack-symptom questions (21%)
behave the same way. A pipeline that drops mostly-missing columns or imputes them erases this
signal. Trees that learn at every split which side missing values go to keep it. This is
"missingness incorporated in attributes" (Twala et al., 2008), the approach Josse et al. (2024)
recommend when missingness is informative.

### 2. Data preparation (`features.py`)

1. Drop 30 interview, telephone and survey-design columns, the preprocessing step's `ADMIN_COLUMNS`
   except `_STATE`, which is kept.
2. Recode each column's codebook answers: "don't know", "refused" and "not asked" codes become
   missing, and "none" codes become 0. The codes come from an explicit per-column table built from
   the CDC 2015 BRFSS codebook (`SENTINELS`). For `MAXVO2_`, `FC60_`, `PAFREQ1_`, `PAFREQ2_` and
   `STRFREQ_`, this CSV stores the missing code with the codebook's implied decimals applied (999 and
   99), so the table lists both spellings.
3. Keep all 291 remaining columns, including the mostly-missing ones.
4. For the trees, bin every column into at most 64 quantile bins (`uint8`), with bin 0 reserved for
   missing values. Columns with at most 64 distinct values get one bin per value.

All statistics, such as cut points, levels and clipping bounds, are fitted on the training rows only.

### 3. Gradient-boosted trees (`model.py`)

Newton boosting on the logistic loss: every tree fits the gradient `p − y` and Hessian `p(1 − p)` of
the current predictions, starting from the log-odds of the base rate. Each tree samples 85% of the
rows without replacement and 40% of the columns, then grows level by level to depth 8.

- **Histograms.** For each level, one flattened `np.bincount` per 64-feature chunk accumulates the
  gradient, Hessian and row-count histograms of all nodes at once. Only the smaller child of every
  split is counted; the larger child is the parent minus the smaller one (Ke et al., 2017).
- **Splits.** Every feature, threshold bin and direction for missing values is scored with the
  regularised Newton gain `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)]` (Chen & Guestrin, 2016).
  Children need at least 50 rows and a Hessian sum of at least 1. Ties go to the lowest feature,
  the lowest threshold and "missing goes right", so a fit is fully deterministic.
- **Leaves.** The value is `−G/(H+λ)`, clipped to ±4, times the learning rate.

| Setting | Value | Setting | Value |
|---|---|---|---|
| Models | 4, seeds 1–4 | Trees | 566, 550, 598, 583 |
| Depth | 8 | Learning rate | 0.03 |
| Rows per leaf | ≥ 50 | L2 on leaf values (λ) | 10 |
| Columns per tree | 40% | Rows per tree | 85% |
| Bins | 64 + missing | Early-stopping patience (CV only) | 100 trees |

### 4. Wide logistic regression (`model.py`, features in `features.py`)

Columns with at most 30 distinct values, plus `_STATE`, `EXRACT11` and `EXRACT21`, become one
indicator per level that covers at least 0.1% of the rows, plus an "other" level. Other columns are
clipped to their 0.5% and 99.5% quantiles and standardised, and they get up to ten quantile-bin
indicators, a piecewise encoding in the spirit of Gorishniy et al. (2022). Every column with missing
training values also gets a "missing" indicator. The 291 columns give 1,938 features. An L2 penalty
of 1e-4 (bias unpenalised) is minimised by Newton's method with a backtracking line search, with the
Hessian accumulated over 16,384-row chunks; it converges in six iterations. Alone the model scores
0.4395 out-of-fold F1, and in the blend it adds about +0.001.

### 5. Blend and threshold

```text
p = 0.85 × mean(p_GBDT1, …, p_GBDT4) + 0.15 × p_logistic        label = +1 if p ≥ 0.2131, else −1
```

The threshold maximises F1 of the blended out-of-fold predictions, searched exactly over every
distinct score. For a rare class the F1-optimal threshold lies far below 0.5 (Lipton et al., 2014):
13.1% of the out-of-fold rows and 13.3% of the test rows are predicted positive.

## Training and cross-validation (`python run.py --train`)

1. Split all labelled rows into 5 stratified folds (seed 2026).
2. Fit each of the five components on four folds, with every preprocessing statistic computed on
   those rows, and predict the fifth. The GBDTs stop early on the held-out fold's log-loss, with at
   most `max(200, 2 × frozen tree count)` trees. The 25 fits run in a pool of up to seven worker
   processes with one BLAS thread each.
3. Choose the threshold on the blended out-of-fold predictions, and set each GBDT's tree count to
   1.1 × its mean best iteration over the folds.
4. Refit every component on all labelled rows, save `outputs/champion_v2.npz`, predict `x_test.csv`
   and check the CSV against the submission.

The recipe stored in `models/champion_v2.npz` fixes everything decided before this step: component
settings, seeds, blend weights and the tree-count rule.

## Evidence

Model selection used 85% of the labelled rows (278,915) for 5-fold cross-validation and kept 15%
(49,220) as a protected holdout. Unless stated otherwise, the numbers below are pooled out-of-fold F1
on identical development folds.

| Variant (best GBDT configuration of the first round) | F1 |
|---|---|
| Keep all columns, learned missing directions | **0.4466** |
| Impute medians before the trees | 0.4370 |
| Remove the cardiovascular module (`HAREHAB1` and six related questions) | 0.4356 |
| Drop columns with more than 50% missing values, as the earlier pipeline did | 0.4310 |
| Earlier depth-2 booster | 0.4280 |
| Wide logistic regression alone | 0.4395 |
| Earlier pipeline's recipe (dense logistic regression + depth-2 boosting), rebuilt | 0.4295 |

Other changes that did not help: class weighting and undersampling, keeping the administrative
columns (fold 0: 0.4452 with them, 0.4478 without), and a rule forcing `HAREHAB1` respondents
positive (+0.0002, because the model already flags 647 of the 654 in the development set).

**Protected holdout, read once.** The first version of the final method (five GBDTs and the logistic
regression, blended 0.65/0.35) scored 0.4480 [0.4363, 0.4596], against 0.4302 [0.4195, 0.4414] for
the earlier pipeline. The paired bootstrap difference was +0.0178 [+0.0116, +0.0242].

**Grid search, cross-validation only.** Four coordinate blocks covered depth × leaf size, column ×
row sampling, L2 × minimum child weight, and bins × learning rate, plus a penalty grid for the
logistic model. Every point was fitted with one seed. The incumbent and at most five points within
0.0025 of the block's best got a second seed (successive halving; Li et al., 2018), because the
seed-to-seed spread is about 0.0013 (Bouthillier et al., 2021). A block winner had to beat the
incumbent's two-seed mean by 0.001 F1, with a log-loss at most 0.0002 worse, and then win again on
two fresh seeds. Only the sampling change survived: 40% of the columns and 85% of the rows per tree
instead of 50% and 70% (fresh seeds: 0.4459 vs 0.4448). Depth 10–12, other leaf sizes, stronger L2,
larger child weights, more bins and a lower learning rate all stayed within +0.0007, and the logistic
model's penalty stayed at 1e-4. The bag of four seeds of the new setting and its blend weight
0.85/0.15 were then chosen on out-of-fold predictions.

**Final 5-fold run over all labelled rows.** The submitted model reaches F1 0.4485 (nested 0.4477),
ROC-AUC 0.8708 and average precision 0.4368. The first version reaches 0.4477 (nested 0.4464) on the
same folds.

**Baselines and ablations on the same folds.** `python -m scripts.evaluate_cv` (after `--train`)
and `python -m scripts.course_baselines` score every model below on the final run's five folds over
all labelled rows. "Cross-fold" F1 chooses each fold's threshold on the other four folds.

| Model | F1 | Cross-fold F1 | ROC-AUC |
|---|---|---|---|
| Ridge regression (course method), mean-imputed standardised features | 0.4200 | 0.4188 | 0.8536 |
| + centred missingness indicators | 0.4285 | 0.4278 | 0.8574 |
| Regularised logistic regression by GD (course method) | 0.4295 | 0.4290 | 0.8604 |
| + centred missingness indicators | 0.4374 | 0.4371 | 0.8639 |
| Wide logistic regression (Newton) | 0.4406 | 0.4401 | 0.8656 |
| One GBDT (mean over seeds 1–4) | 0.4467 | 0.4456 | 0.8702 |
| Four-GBDT bag | 0.4481 | 0.4470 | 0.8707 |
| **Final blend** | **0.4485** | **0.4477** | **0.8708** |
| Final blend at threshold 0.5 | 0.2494 | – | 0.8708 |

The course baselines use λ = 1e-4. Gradient descent runs 10,000 steps of size 1/L from the training
log-odds and lands within 0.0005 F1 of Newton's method on the same objective (0.4294 and 0.4379).
The indicators are centred because raw 0/1 indicators of questions skipped together raise the
largest eigenvalue of the design's Gram matrix from about 14 to 119, which shrinks the step tenfold:
with raw indicators, 10,000 steps reached only 0.4327. The tuned threshold is the largest single
factor, and seed bagging (+0.0014) and the logistic component (+0.0004) are close to the 0.0009
seed-to-seed spread of a single GBDT.

## Reproducibility

- `python run.py` reproduces the submitted CSV byte for byte; its SHA-256 starts with
  `d49680ff34febe4a`, and `run.py` checks it.
- On 19 October 2026, `python run.py --train` (Python 3.13.9, NumPy 2.5.4, macOS arm64) reproduced
  the original training run bit for bit: the same folds, out-of-fold predictions and best
  iterations, a model file identical to `models/champion_v2.npz` in every entry, and a byte-identical
  CSV.
- Training is deterministic: seeded sampling, deterministic tie-breaking and one BLAS thread per
  worker process. The GBDT's feature-chunk size never changes a result. The logistic model's
  16,384-row chunks set the order of its floating-point sums, so they are part of its stored
  configuration, and other NumPy or BLAS builds can differ in the last bits of those sums.
- The code here is a condensed version of our development code. The experiment drivers behind the
  evidence above (ablations, grid search and the holdout read) are not included; every step that
  produces the submitted model is.

## Limitations

- **Post-diagnosis features.** `HAREHAB1` and the other cardiovascular-module questions are provided
  features but are asked only after a reported heart attack, so part of the signal is the survey's
  routing (Kaufman et al., 2012). Without that module, the first version scored 0.4376 on the
  holdout.
- **Transfer.** The model learns BRFSS's skip logic, so it would not carry over to data collected
  with a different questionnaire (Groenwold, 2020).
- **Selection effects.** Early stopping uses the held-out fold, so out-of-fold F1 is slightly
  optimistic (Cawley & Talbot, 2010). The first version's holdout score (0.4480) matches its
  development out-of-fold score (0.4477), which suggests the effect is small. The submitted model
  has no holdout score of its own.
