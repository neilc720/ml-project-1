"""
Compare preprocessing / feature-selection / balancing strategies.

Every configuration trains the *same* model (regularised logistic regression
by gradient descent, from implementations.py) so that differences in the
scores come from the data pipeline only.

Run from the repository root:

    python -m scripts.data_experiments            # loads dataset/ (slow, ~1 min)
    python -m scripts.data_experiments --quick    # fewer configurations

Splits: 70% train / 10% validation / 20% test (seed 42). The validation split
is only used to tune the decision threshold; all reported numbers are on the
test split.
"""

import argparse
import sys
import time

import numpy as np

from implementations import reg_logistic_regression, sigmoid
from scripts.helper import (
    load_csv_data,
    negative_reduction,
    sort_abs_correlation,
    standardize_zero_to_one,
)
from scripts.preprocessing import (
    Preprocessor,
    best_f1_threshold,
    class_weights,
    f1_score,
    load_feature_names,
    oversample,
    select_features,
    undersample,
    weighted_reg_logistic_regression,
)

SEED = 42
GAMMA = 0.5
MAX_ITERS = 400
LAMBDA = 1e-4


def split_indices(n, seed=SEED):
    """70 / 10 / 20 train / val / test row indices."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr, n_val = int(0.7 * n), int(0.1 * n)
    return idx[:n_tr], idx[n_tr : n_tr + n_val], idx[n_tr + n_val :]


def fit_logreg(tx, y01, sample_weights=None):
    """Train logistic regression; returns weights."""
    w0 = np.zeros(tx.shape[1])
    if sample_weights is None:
        w, _ = reg_logistic_regression(y01, tx, LAMBDA, w0, MAX_ITERS, GAMMA)
    else:
        w, _ = weighted_reg_logistic_regression(
            y01, tx, sample_weights, LAMBDA, w0, MAX_ITERS, GAMMA
        )
    return w


def evaluate(w, tx_val, y_val, tx_te, y_te):
    """Accuracy / F1 at threshold 0.5 and at the validation-tuned threshold."""
    p_val, p_te = sigmoid(tx_val @ w), sigmoid(tx_te @ w)
    pos_te = y_te == 1
    thr, _ = best_f1_threshold(p_val, y_val)
    return {
        "acc@0.5": np.mean((p_te >= 0.5) == pos_te),
        "f1@0.5": f1_score(pos_te, p_te >= 0.5),
        "thr": thr,
        "acc@thr": np.mean((p_te >= thr) == pos_te),
        "f1@thr": f1_score(pos_te, p_te >= thr),
    }


def run_baseline(x, y, tr, val, te):
    """Last year's pipeline: NaN->0, min-max, top-56 |corr|, per-step
    negative subsampling (reduction 0.14)."""
    xn = standardize_zero_to_one(np.nan_to_num(x))
    xn = np.nan_to_num(xn)  # constant columns give 0/0
    y01 = ((y + 1) / 2).astype(float)
    cols = sort_abs_correlation(xn[tr], y[tr])[:56]
    rng_state = np.random.get_state()
    np.random.seed(SEED)
    w = np.random.rand(56)
    for _ in range(MAX_ITERS):
        ry, rx = negative_reduction(y01[tr], xn[tr][:, cols], 0.14)
        w, _ = reg_logistic_regression(ry, rx, 1.5e-4, w, 1, 1.4)
    np.random.set_state(rng_state)
    return evaluate(w, xn[val][:, cols], y[val], xn[te][:, cols], y[te])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--cache", default=None, help="dir with x.npy, y.npy")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    names = load_feature_names(args.data)
    if args.cache:
        x, y = np.load(f"{args.cache}/x.npy"), np.load(f"{args.cache}/y.npy")
    else:
        x, _, y, _, _ = load_csv_data(args.data)
    tr, val, te = split_indices(len(y))
    y01 = ((y + 1) / 2).astype(float)
    print(
        f"rows: train {len(tr)}, val {len(val)}, test {len(te)}; "
        f"positives {np.mean(y == 1):.3%}"
    )

    results = []

    def log(name, res, t0):
        results.append((name, res))
        print(
            f"{name:<48} acc@0.5={res['acc@0.5']:.4f} f1@0.5={res['f1@0.5']:.4f} | "
            f"thr={res['thr']:.2f} acc={res['acc@thr']:.4f} f1={res['f1@thr']:.4f} "
            f"({time.time() - t0:.0f}s)"
        )
        sys.stdout.flush()

    t0 = time.time()
    log(
        "baseline (last year: nan->0, minmax, top56, negred)",
        run_baseline(x, y, tr, val, te),
        t0,
    )

    # New preprocessing, fitted on the training rows only.
    t0 = time.time()
    pre = Preprocessor(names)
    tx_tr = pre.fit_transform(x[tr])
    tx_val, tx_te = pre.transform(x[val]), pre.transform(x[te])
    print(f"preprocessed: {tx_tr.shape[1]} columns ({time.time() - t0:.0f}s)")

    ks = [30, 60, 100, None] if args.quick else [20, 40, 60, 100, 150, None]
    balancers = [
        ("none", lambda a, b: (a, b, None)),
        ("undersample 1:1", lambda a, b: (*undersample(a, b, 1.0, SEED), None)),
        ("undersample 1:3", lambda a, b: (*undersample(a, b, 3.0, SEED), None)),
        ("oversample 1:1", lambda a, b: (*oversample(a, b, 1.0, SEED), None)),
        ("class weights", lambda a, b: (a, b, class_weights(b))),
    ]
    if args.quick:
        balancers = [balancers[0], balancers[1], balancers[4]]

    for k in ks:
        t0 = time.time()
        cols = select_features(tx_tr, y01[tr], k=k, protect=[0])
        label_k = f"k={k if k else 'all'}({len(cols) - 1})"
        for bname, balance in balancers:
            t0 = time.time()
            a, b, sw = balance(tx_tr[:, cols], y01[tr])
            w = fit_logreg(a, b, sw)
            log(
                f"new | {label_k:<12} | {bname}",
                evaluate(w, tx_val[:, cols], y[val], tx_te[:, cols], y[te]),
                t0,
            )

    best = max(results, key=lambda r: r[1]["f1@thr"])
    print(f"\nbest by test F1 (tuned threshold): {best[0]}  f1={best[1]['f1@thr']:.4f}")


if __name__ == "__main__":
    main()
