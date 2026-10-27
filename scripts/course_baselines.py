"""Baselines built from the required methods in implementations.py, on the final model's folds.

Run from the repository root (no ``--train`` run needed; the folds are re-derived from the
recipe stored in models/champion_v2.npz, exactly as ``run.py --train`` derives them):

    python -m scripts.course_baselines

Features: the same 291 columns and sentinel recoding as the final model, standardised with the
mean and standard deviation of the observed training-fold values; missing values become 0 (the
training mean). A bias column is prepended. The "+ missing indicators" variant appends one
missingness indicator per feature with missing training values, centred by its training mean:
raw 0/1 indicators of questions skipped together raise the largest eigenvalue of tx'tx/N from
about 14 to 119, which shrinks the gradient-descent step tenfold. Models:

* ridge regression (normal equations) on 0/1 labels, scoring rows by ``tx @ w``;
* regularised logistic regression by gradient descent (10,000 steps by default), started at the
  training log-odds, with step size 1/L for the Lipschitz constant L of the gradient;
* as a convergence check, the same logistic objective solved by Newton's method (``model.py``,
  bias unpenalised), which should match the gradient-descent scores.

Every learned statistic is fitted on the four training folds only. Thresholds are chosen like
the final model's (see scripts/evaluate_cv.py). NumPy and the standard library only.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from features import IDENTIFIER_COLUMNS, SENTINELS, load_data
from implementations import reg_logistic_regression, ridge_regression
from model import NewtonLogistic, stratified_fold_ids
from scripts.evaluate_cv import summarize

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("course_baselines")


def recode(x: np.ndarray, names: list[str]) -> np.ndarray:
    """Drop the identifier columns and apply the sentinel table (missing -> NaN, none -> 0)."""
    keep = [j for j, name in enumerate(names) if name not in IDENTIFIER_COLUMNS]
    out = np.array(x[:, keep], dtype=float)
    for k, j in enumerate(keep):
        missing, zero = SENTINELS.get(names[j], ((), ()))
        out[np.isin(out[:, k], zero), k] = 0.0
        out[np.isin(out[:, k], missing), k] = np.nan
    return out


def fit_design(x_train: np.ndarray, indicators: bool) -> dict[str, np.ndarray]:
    """Standardisation statistics and indicator columns, from training rows only."""
    std = np.nanstd(x_train, axis=0)
    missing = np.isnan(x_train)
    indicate = (
        missing.any(axis=0) if indicators else np.zeros(x_train.shape[1], dtype=bool)
    )
    return {
        "mean": np.nanmean(x_train, axis=0),
        "std": np.where(std > 0, std, 1.0),
        "indicate": indicate,
        "missing_rate": missing[:, indicate].mean(axis=0),
    }


def design(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Bias, standardised values (missing -> 0) and the centred missing indicators."""
    missing = np.isnan(x)
    z = np.where(missing, 0.0, (x - stats["mean"]) / stats["std"])
    indicators = missing[:, stats["indicate"]] - stats["missing_rate"]
    return np.column_stack([np.ones(len(x)), z, indicators])


def fit_logistic(
    y: np.ndarray, tx: np.ndarray, lambda_: float, max_iters: int
) -> np.ndarray:
    """Course GD with step 1/L, L = largest eigenvalue of tx'tx/N over 4, plus 2 lambda."""
    lipschitz = np.linalg.eigvalsh(tx.T @ tx / len(y))[-1] / 4 + 2 * lambda_
    initial_w = np.zeros(tx.shape[1])
    initial_w[0] = np.log(y.mean() / (1 - y.mean()))
    w, loss = reg_logistic_regression(
        y, tx, lambda_, initial_w, max_iters, 1.0 / lipschitz
    )
    logger.debug("logistic GD: training loss %.5f", loss)
    return w


def out_of_fold(
    x: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    indicators: bool,
    lambda_: float,
    max_iters: int,
) -> dict[str, np.ndarray]:
    """Out-of-fold scores of ridge, GD logistic and Newton logistic for one variant."""
    scores = {name: np.zeros(len(y)) for name in ("ridge", "logistic", "newton")}
    for k in np.unique(fold_id):
        train, held_out = fold_id != k, fold_id == k
        stats = fit_design(x[train], indicators)
        tx_train, tx_val = design(x[train], stats), design(x[held_out], stats)
        w_ridge, _ = ridge_regression(y[train], tx_train, lambda_)
        scores["ridge"][held_out] = tx_val @ w_ridge
        w_logistic = fit_logistic(y[train], tx_train, lambda_, max_iters)
        scores["logistic"][held_out] = tx_val @ w_logistic
        newton = NewtonLogistic(lambda_=lambda_, max_iter=50).fit(tx_train, y[train])
        scores["newton"][held_out] = tx_val @ newton.coef_
        logger.info(
            "indicators=%s: fold %d done (%d columns)", indicators, k, tx_train.shape[1]
        )
    return scores


def main(argv: list[str] | None = None) -> None:
    """Log pooled / cross-fold F1, ROC-AUC and AP of the course baselines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "dataset"))
    parser.add_argument("--outputs", default=str(ROOT / "outputs"))
    parser.add_argument("--model", default=str(ROOT / "models" / "champion_v2.npz"))
    parser.add_argument("--lambda_", type=float, default=1e-4)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--json", help="optional path for the metrics as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_data(args.data, Path(args.outputs) / "cache")
    with np.load(args.model, allow_pickle=False) as stored:
        recipe = json.loads(str(stored["recipe"]))
    y = data["y"]
    fold_id = stratified_fold_ids(y, n_splits=recipe["n_splits"], seed=recipe["seed"])
    x = recode(data["x"], list(data["names"]))
    rows: list[dict[str, Any]] = []
    for indicators in (False, True):
        scores = out_of_fold(x, y, fold_id, indicators, args.lambda_, args.max_iters)
        suffix = " + missing indicators" if indicators else ""
        for method, label in (
            ("ridge", "Ridge regression"),
            ("logistic", "Reg. logistic GD"),
            ("newton", "Logistic, Newton check"),
        ):
            rows.append(summarize(label + suffix, y, scores[method], fold_id))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
