"""Metrics and ablations of the final model from its 5-fold out-of-fold predictions.

Run from the repository root after ``python run.py --train``:

    python -m scripts.evaluate_cv    # reads outputs/cv.npz and outputs/folds.npz

``cv.npz`` holds one out-of-fold probability column per component (GBDT seeds 1-4, then the
wide logistic regression), each fitted on four folds and scored on the fifth. Every row below
uses these same predictions. "Pooled" F1 chooses the threshold on all out-of-fold scores and
evaluates it on the same scores; "cross-fold" F1 chooses each fold's threshold on the other
four folds' out-of-fold scores. Neither is fully nested: the GBDTs early-stop on the fold they
score, and the other folds' scores come from models trained partly on the scored fold.
NumPy and the standard library only.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from features import load_data
from model import best_threshold

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("evaluate_cv")


def _f1(y: np.ndarray, predicted: np.ndarray) -> float:
    """Positive-class F1 of boolean predictions."""
    tp = float(np.sum(predicted & (y == 1)))
    return 2 * tp / max(float(np.sum(predicted) + np.sum(y == 1)), 1.0)


def cross_fold_f1(y: np.ndarray, scores: np.ndarray, fold_id: np.ndarray) -> float:
    """F1 of the pooled decisions when each fold's threshold is chosen on the other folds."""
    predicted = np.zeros(len(y), dtype=bool)
    for k in np.unique(fold_id):
        held_out = fold_id == k
        threshold, _ = best_threshold(y[~held_out], scores[~held_out])
        predicted[held_out] = scores[held_out] >= threshold
    return _f1(y, predicted)


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney statistic; tied scores share their mean rank)."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    mean_rank = np.cumsum(counts) - (counts - 1) / 2.0
    n_pos = float(np.sum(y == 1))
    n_neg = len(y) - n_pos
    rank_sum = float(np.sum(mean_rank[inverse][y == 1]))
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Sum over distinct thresholds of the recall gained times the precision reached."""
    order = np.argsort(-scores, kind="stable")
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.cumsum(ys)[ends]
    recall = tp / ys.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * tp / (ends + 1)))


def summarize(
    name: str, y: np.ndarray, scores: np.ndarray, fold_id: np.ndarray
) -> dict[str, Any]:
    """Pooled and cross-fold F1, ROC-AUC and average precision of one score vector."""
    threshold, pooled = best_threshold(y, scores)
    row = {
        "model": name,
        "f1_pooled": pooled,
        "threshold": threshold,
        "f1_cross_fold": cross_fold_f1(y, scores, fold_id),
        "roc_auc": roc_auc(y, scores),
        "average_precision": average_precision(y, scores),
    }
    logger.info(
        "%-38s F1 pooled %.4f (threshold %.4f) | cross-fold %.4f | AUC %.4f | AP %.4f",
        name,
        row["f1_pooled"],
        threshold,
        row["f1_cross_fold"],
        row["roc_auc"],
        row["average_precision"],
    )
    return row


def blend(oof: np.ndarray, weights: list[float]) -> np.ndarray:
    """Blended probability exactly as ``run.train`` forms it: GBDT mean, then the weights."""
    gbdts = [oof[:, k] for k in range(oof.shape[1] - 1)]
    return np.column_stack([np.mean(gbdts, axis=0), oof[:, -1]]) @ np.asarray(weights)


def main(argv: list[str] | None = None) -> None:
    """Log the final model's metrics and ablations; optionally save them as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "dataset"))
    parser.add_argument("--outputs", default=str(ROOT / "outputs"))
    parser.add_argument("--model", default=str(ROOT / "models" / "final_model.npz"))
    parser.add_argument("--json", help="optional path for the metrics as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(args.outputs)
    y = load_data(args.data, out_dir / "cache")["y"]
    with np.load(out_dir / "cv.npz") as cv, np.load(out_dir / "folds.npz") as folds:
        oof, iterations, fold_id = cv["oof"], cv["iterations"], folds["fold_id"]
    with np.load(args.model, allow_pickle=False) as stored:
        recipe = json.loads(str(stored["recipe"]))
    blended = blend(oof, recipe["weights"])
    gbdt_bag = np.mean([oof[:, k] for k in range(oof.shape[1] - 1)], axis=0)

    rows = [summarize("Final blend (submitted)", y, blended, fold_id)]
    threshold = rows[0]["threshold"]
    retrained = out_dir / f"{recipe['name']}.npz"
    for path in (Path(args.model), retrained):
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as stored:
            saved = json.loads(str(stored["spec"]))["threshold"]
        logger.info(
            "Pooled threshold %r vs %s: %r (%s)",
            threshold,
            path,
            saved,
            "identical" if threshold == saved else "different",
        )
    rows[0]["f1_at_0.5"] = _f1(y, blended >= 0.5)
    logger.info("Final blend at threshold 0.5: F1 %.4f", rows[0]["f1_at_0.5"])
    rows.append(summarize("Four-GBDT bag only", y, gbdt_bag, fold_id))
    for k in range(oof.shape[1] - 1):
        rows.append(summarize(f"Single GBDT, seed {k + 1}", y, oof[:, k], fold_id))
    rows.append(summarize("Wide logistic regression only", y, oof[:, -1], fold_id))
    logger.info("Best iterations per component and fold:\n%s", iterations)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
