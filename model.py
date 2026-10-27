"""Model side of the final solution: histogram GBDT, Newton logistic regression and their blend.

* ``GBDTClassifier``: Newton boosting on the logistic loss with level-wise histogram trees (one
  vectorised ``np.bincount`` per level, sibling histograms by subtraction), a learned direction for
  missing values at every split, row/column subsampling and early stopping.
* ``NewtonLogistic``: L2 logistic regression solved by Newton's method over row chunks.
* ``FinalModel``: mean of the GBDTs blended with the logistic model, a fixed decision threshold, and
  a single-file ``.npz`` format (no pickle) via ``save_model`` / ``load_model``.

NumPy and the standard library only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from features import (
    MISSING_BIN,
    TreeBinner,
    WideFeatureMap,
    tree_feature_config,
    wide_feature_config,
)

logger = logging.getLogger(__name__)
_MAX_KEYS = 2**31 - 1
_MAX_HISTOGRAM = 2**27  # entries per histogram array (1 GB of float64)


def _ordered(y: Any, scores: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels and scores sorted by decreasing score, and the last index of each run of equal scores."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("Nonempty aligned 1-D labels and scores required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores")
    order = np.argsort(-scores, kind="stable")
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    return ys, ss, ends


def best_threshold(y: Any, scores: Any) -> tuple[float, float]:
    """(threshold, F1) maximising F1 over all distinct scores, predicting positive at ``score >= threshold``."""
    ys, ss, ends = _ordered(y, scores)
    tp = np.cumsum(ys)[ends]
    f1 = 2 * tp / np.maximum(ends + 1 + ys.sum(), 1)
    index = int(np.argmax(f1))  # deterministic: higher threshold wins ties
    return float(ss[ends[index]]), float(f1[index])


def _binary(y: Any) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isin(y, (0, 1)).all():
        raise ValueError("nonempty one-dimensional 0/1 labels required")
    return y


def stratified_fold_ids(y: Any, n_splits: int = 5, seed: int = 2026) -> np.ndarray:
    """Per-row fold id in ``0..n_splits-1`` with each class spread evenly."""
    y = _binary(y)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    rng = np.random.default_rng(seed)
    fold_id = np.empty(len(y), dtype=np.int8)
    for label in np.unique(y):
        rows = rng.permutation(np.flatnonzero(y == label))
        fold_id[rows] = np.arange(len(rows)) % n_splits
    return fold_id


@dataclass(frozen=True)
class GBDTParams:
    """Hyper-parameters of the boosted ensemble and of each tree."""

    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 6
    min_samples_leaf: int = 100
    min_child_weight: float = 1.0
    l2: float = 1.0
    colsample: float = 0.5
    subsample: float = 0.8
    max_leaf_value: float = 4.0
    early_stopping_rounds: int = 100
    seed: int = 2026
    feature_chunk: int = 64
    min_gain: float = 1e-10

    def __post_init__(self) -> None:
        if self.n_estimators < 0 or self.max_depth < 1 or self.max_depth > 12:
            raise ValueError("n_estimators must be >= 0 and 1 <= max_depth <= 12")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must lie in (0, 1]")
        if not (0 < self.colsample <= 1 and 0 < self.subsample <= 1):
            raise ValueError("colsample and subsample must lie in (0, 1]")
        if self.min_samples_leaf < 1 or self.min_child_weight < 0 or self.l2 < 0:
            raise ValueError(
                "min_samples_leaf >= 1, min_child_weight >= 0 and l2 >= 0 required"
            )
        if self.max_leaf_value <= 0 or self.feature_chunk < 1:
            raise ValueError("max_leaf_value and feature_chunk must be positive")


@dataclass
class TreeArrays:
    """One fitted tree in heap layout; ``value`` already includes the learning rate."""

    feature: np.ndarray
    threshold: np.ndarray
    nan_left: np.ndarray
    value: np.ndarray

    @property
    def depth(self) -> int:
        """Maximum depth implied by the number of heap nodes."""
        return int(np.log2(len(self.feature) + 1)) - 1


TREE_DTYPES = {
    "feature": np.int32,
    "threshold": np.uint8,
    "nan_left": bool,
    "value": np.float64,
}


@dataclass
class SplitChoice:
    """Best split per histogram slot; ``feature`` is local to the tree's columns."""

    feature: np.ndarray
    threshold: np.ndarray
    nan_left: np.ndarray


def build_histograms(
    binned: np.ndarray,
    slot: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    n_slots: int,
    n_bins: int,
    feature_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient, Hessian and count histograms of shape ``(n_slots, n_features, n_bins)``.

    Rows with ``slot < 0`` are skipped. One flattened ``np.bincount`` per feature
    chunk accumulates every slot and bin at once, which is what makes level-wise
    growth cheap in NumPy.
    """
    n_rows, n_features = binned.shape
    chunk = min(int(feature_chunk), n_features)
    if (
        n_slots * n_features * n_bins > _MAX_HISTOGRAM
        or n_slots * chunk * n_bins > _MAX_KEYS
    ):
        raise ValueError("histogram too large; reduce depth, bins or features")
    rows = np.flatnonzero(slot >= 0)
    shape = (n_slots, n_features, n_bins)
    G, H, C = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    if not len(rows):
        return G, H, C
    slot_rows = slot[rows].astype(np.int64)
    g_rows, h_rows = grad[rows].astype(np.float64), hess[rows].astype(np.float64)
    for start in range(0, n_features, chunk):
        stop = min(start + chunk, n_features)
        width = stop - start
        keys = (slot_rows[:, None] * width + np.arange(width)[None, :]) * n_bins
        keys = keys + binned[rows[:, None], np.arange(start, stop)[None, :]]
        flat = keys.ravel()
        size = n_slots * width * n_bins
        G[:, start:stop, :] = np.bincount(
            flat, weights=np.repeat(g_rows, width), minlength=size
        ).reshape(n_slots, width, n_bins)
        H[:, start:stop, :] = np.bincount(
            flat, weights=np.repeat(h_rows, width), minlength=size
        ).reshape(n_slots, width, n_bins)
        C[:, start:stop, :] = np.bincount(flat, minlength=size).reshape(
            n_slots, width, n_bins
        )
    return G, H, C


def split_chunk(
    n_slots: int, n_bins: int, feature_chunk: int, budget: int = 2**21
) -> int:
    """Features per split-search chunk so one (slot, feature, bin, direction) array stays near ``budget`` cells.

    Deep trees with many bins otherwise allocate gigabytes of temporaries. The chunk size never
    changes the result: candidates are scanned in the same (feature, threshold, direction) order and
    ties keep the earlier chunk.
    """
    return int(
        max(1, min(int(feature_chunk), budget // max(1, n_slots * (n_bins - 1))))
    )


def find_best_splits(
    G: np.ndarray, H: np.ndarray, C: np.ndarray, params: GBDTParams
) -> SplitChoice:
    """Best (feature, threshold, missing direction) per slot under the Newton objective.

    Candidates are evaluated in feature chunks; ties resolve toward the lowest
    feature, the lowest threshold and 'missing goes right', so results are
    reproducible. A slot with no admissible candidate gets ``feature == -1``.
    """
    n_slots, n_features, n_bins = G.shape
    lam = params.l2
    g_tot, h_tot, c_tot = (
        G[:, 0, :].sum(axis=1),
        H[:, 0, :].sum(axis=1),
        C[:, 0, :].sum(axis=1),
    )
    parent = 0.5 * g_tot**2 / (h_tot + lam)
    best_gain = np.full(n_slots, -np.inf)
    best_feature = np.full(n_slots, -1, dtype=np.int64)
    best_bin = np.zeros(n_slots, dtype=np.int64)
    best_nan_left = np.zeros(n_slots, dtype=bool)
    chunk = min(split_chunk(n_slots, n_bins, params.feature_chunk), n_features)
    for start in range(0, n_features, chunk):
        stop = min(start + chunk, n_features)
        cum_g = np.cumsum(G[:, start:stop, 1:], axis=2)
        cum_h = np.cumsum(H[:, start:stop, 1:], axis=2)
        cum_c = np.cumsum(C[:, start:stop, 1:], axis=2)
        # Last axis: direction 0 sends missing right, direction 1 sends it left.
        gl = np.stack([cum_g, cum_g + G[:, start:stop, :1]], axis=-1)
        hl = np.stack([cum_h, cum_h + H[:, start:stop, :1]], axis=-1)
        cl = np.stack([cum_c, cum_c + C[:, start:stop, :1]], axis=-1)
        gr = g_tot[:, None, None, None] - gl
        hr = h_tot[:, None, None, None] - hl
        cr = c_tot[:, None, None, None] - cl
        valid = (
            (cl >= params.min_samples_leaf)
            & (cr >= params.min_samples_leaf)
            & (hl >= params.min_child_weight)
            & (hr >= params.min_child_weight)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = (
                0.5 * (gl**2 / (hl + lam) + gr**2 / (hr + lam))
                - parent[:, None, None, None]
            )
        gain = np.where(valid & np.isfinite(gain), gain, -np.inf)
        flat = gain.reshape(n_slots, -1)
        index = np.argmax(flat, axis=1)
        chunk_gain = flat[np.arange(n_slots), index]
        feature, rest = np.divmod(index, (n_bins - 1) * 2)
        threshold, direction = np.divmod(rest, 2)
        better = chunk_gain > best_gain
        best_gain = np.where(better, chunk_gain, best_gain)
        best_feature = np.where(better, feature + start, best_feature)
        best_bin = np.where(better, threshold + 1, best_bin)
        best_nan_left = np.where(better, direction == 1, best_nan_left)
    usable = best_gain > params.min_gain
    return SplitChoice(
        feature=np.where(usable, best_feature, -1),
        threshold=best_bin,
        nan_left=best_nan_left & usable,
    )


def grow_tree(
    binned: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    params: GBDTParams,
    n_bins: int,
    feature_ids: np.ndarray,
) -> TreeArrays:
    """Grow one level-wise tree on pre-binned rows (the tree's row/column sample).

    ``feature_ids`` maps the local columns of ``binned`` to global feature indices stored in the
    returned tree. The larger child of every split gets its histogram by subtraction from the parent.
    """
    n_rows, n_features = binned.shape
    depth_max = params.max_depth
    n_nodes = 2 ** (depth_max + 1) - 1
    feature = np.full(n_nodes, -1, dtype=np.int32)
    threshold = np.zeros(n_nodes, dtype=np.uint8)
    nan_left = np.zeros(n_nodes, dtype=bool)
    node_of_row = np.zeros(n_rows, dtype=np.int32)
    slot = np.zeros(n_rows, dtype=np.int32)
    level_nodes = np.zeros(1, dtype=np.int64)
    G, H, C = build_histograms(
        binned, slot, grad, hess, 1, n_bins, params.feature_chunk
    )
    for depth in range(depth_max):
        split = find_best_splits(G, H, C, params)
        has_split = split.feature >= 0
        if not has_split.any():
            break
        nodes = level_nodes[has_split]
        feature[nodes] = feature_ids[split.feature[has_split]]
        threshold[nodes] = split.threshold[has_split]
        nan_left[nodes] = split.nan_left[has_split]
        rows = np.flatnonzero(slot >= 0)
        rows = rows[has_split[slot[rows]]]
        s = slot[rows]
        bins = binned[rows, split.feature[s]]
        go_left = np.where(
            bins == MISSING_BIN, split.nan_left[s], bins <= split.threshold[s]
        )
        child = 2 * node_of_row[rows] + 1 + (~go_left).astype(np.int32)
        node_of_row[rows] = child
        if depth + 1 == depth_max:
            break
        children = np.stack([2 * nodes + 1, 2 * nodes + 2], axis=1).ravel()
        lookup = np.full(n_nodes, -1, dtype=np.int32)
        lookup[children] = np.arange(len(children), dtype=np.int32)
        new_slot = np.full(n_rows, -1, dtype=np.int32)
        new_slot[rows] = lookup[child]
        n_children = len(children)
        counts = np.bincount(new_slot[rows], minlength=n_children)
        left_bigger = counts[0::2] >= counts[1::2]
        small = np.where(
            left_bigger, np.arange(1, n_children, 2), np.arange(0, n_children, 2)
        )
        large = np.where(
            left_bigger, np.arange(0, n_children, 2), np.arange(1, n_children, 2)
        )
        is_large = np.zeros(n_children, dtype=bool)
        is_large[large] = True
        hist_slot = new_slot.copy()
        hist_slot[rows[is_large[new_slot[rows]]]] = -1
        G2, H2, C2 = build_histograms(
            binned, hist_slot, grad, hess, n_children, n_bins, params.feature_chunk
        )
        parent_index = np.flatnonzero(has_split)
        G2[large] = G[parent_index] - G2[small]
        H2[large] = np.maximum(H[parent_index] - H2[small], 0.0)
        C2[large] = np.maximum(C[parent_index] - C2[small], 0.0)
        G, H, C, slot, level_nodes = G2, H2, C2, new_slot, children
    leaf_g = np.bincount(
        node_of_row, weights=grad.astype(np.float64), minlength=n_nodes
    )
    leaf_h = np.bincount(
        node_of_row, weights=hess.astype(np.float64), minlength=n_nodes
    )
    denominator = leaf_h + params.l2
    value = np.zeros(n_nodes, dtype=np.float64)
    np.divide(-leaf_g, denominator, out=value, where=denominator > 0)
    value = (
        np.clip(value, -params.max_leaf_value, params.max_leaf_value)
        * params.learning_rate
    )
    value[feature >= 0] = 0.0
    return TreeArrays(
        feature=feature, threshold=threshold, nan_left=nan_left, value=value
    )


def apply_tree(tree: TreeArrays, binned: np.ndarray) -> np.ndarray:
    """Leaf node id of every row of ``binned`` (global feature layout)."""
    n_rows = binned.shape[0]
    node = np.zeros(n_rows, dtype=np.int32)
    rows = np.arange(n_rows)
    for _ in range(tree.depth):
        feature = tree.feature[node]
        is_leaf = feature < 0
        bins = binned[rows, np.maximum(feature, 0)]
        go_left = np.where(
            bins == MISSING_BIN, tree.nan_left[node], bins <= tree.threshold[node]
        )
        child = 2 * node + 1 + (~go_left).astype(np.int32)
        node = np.where(is_leaf, node, child)
    return node


def predict_tree(tree: TreeArrays, binned: np.ndarray) -> np.ndarray:
    """Leaf values (learning rate included) for every row of ``binned``."""
    return tree.value[apply_tree(tree, binned)]


def sigmoid(scores: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    values = np.exp(-np.abs(scores))
    return np.where(scores >= 0, 1.0 / (1.0 + values), values / (1.0 + values))


def _log_loss(y: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, scores) - y * scores))


def _as_binned(x: Any) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.dtype != np.uint8:
        raise ValueError("binned features must be a two-dimensional uint8 matrix")
    return x


def _as_labels(y: Any, n_rows: int) -> np.ndarray:
    y = _binary(y)
    if len(y) != n_rows:
        raise ValueError("labels and binned rows differ in length")
    return y


class GBDTClassifier:
    """Boosted histogram trees with early stopping; see the module docstring."""

    def __init__(self, params: GBDTParams | None = None) -> None:
        self.params = params or GBDTParams()
        self.trees_: list[TreeArrays] = []
        self.intercept_ = 0.0
        self.n_bins_ = 0
        self.n_features_ = 0
        self.best_iteration_ = 0
        self.n_estimators_ = 0

    # -------------------------------------------------------------- fitting
    def fit(
        self,
        binned: Any,
        y: Any,
        binned_val: Any = None,
        y_val: Any = None,
        n_bins: int | None = None,
    ) -> "GBDTClassifier":
        """Fit on binned rows; validation rows (optional) only drive early stopping on log-loss."""
        p = self.params
        binned = _as_binned(binned)
        y = _as_labels(y, binned.shape[0])
        has_val = binned_val is not None
        if has_val:
            binned_val = _as_binned(binned_val)
            y_val = _as_labels(y_val, binned_val.shape[0])
        max_bin = int(binned.max()) if binned.size else 0
        if has_val and binned_val.size:
            max_bin = max(max_bin, int(binned_val.max()))
        self.n_bins_ = int(n_bins) if n_bins is not None else max_bin + 1
        if self.n_bins_ <= max_bin:
            raise ValueError("n_bins must exceed the largest bin index")
        n_rows, self.n_features_ = binned.shape
        prior = min(max(float(np.sum(y) / n_rows), 1e-6), 1 - 1e-6)
        self.intercept_ = float(np.log(prior / (1 - prior)))
        score = np.full(n_rows, self.intercept_)
        val_score = np.full(binned_val.shape[0] if has_val else 0, self.intercept_)
        best_loss, best_iteration = (
            _log_loss(y_val, val_score) if has_val else np.inf
        ), 0
        rng = np.random.default_rng(p.seed)
        n_cols = max(1, int(round(p.colsample * self.n_features_)))
        n_sub = max(1, int(round(p.subsample * n_rows)))
        self.trees_ = []
        for iteration in range(1, p.n_estimators + 1):
            prob = sigmoid(score)
            grad = (prob - y).astype(np.float32)
            hess = (prob * (1 - prob)).astype(np.float32)
            rows = (
                np.sort(rng.choice(n_rows, n_sub, replace=False))
                if n_sub < n_rows
                else np.arange(n_rows)
            )
            cols = (
                np.sort(rng.choice(self.n_features_, n_cols, replace=False))
                if n_cols < self.n_features_
                else np.arange(self.n_features_)
            )
            sample = (
                binned
                if (n_sub == n_rows and n_cols == self.n_features_)
                else binned[np.ix_(rows, cols)]
            )
            tree = grow_tree(sample, grad[rows], hess[rows], p, self.n_bins_, cols)
            self.trees_.append(tree)
            score = score + predict_tree(tree, binned)
            if has_val:
                val_score = val_score + predict_tree(tree, binned_val)
                loss = _log_loss(y_val, val_score)
                if loss < best_loss:
                    best_loss, best_iteration = loss, iteration
                if iteration - best_iteration >= p.early_stopping_rounds:
                    break
        if has_val:
            self.trees_ = self.trees_[:best_iteration]
        self.best_iteration_ = best_iteration if has_val else len(self.trees_)
        self.n_estimators_ = len(self.trees_)
        return self

    # ----------------------------------------------------------- prediction
    def decision_function(self, binned: Any) -> np.ndarray:
        """Additive scores (log-odds)."""
        binned = _as_binned(binned)
        if binned.shape[1] != self.n_features_:
            raise ValueError("feature count differs from the fitted model")
        score = np.full(binned.shape[0], self.intercept_)
        for tree in self.trees_:
            score += predict_tree(tree, binned)
        return score

    def predict_proba(self, binned: Any) -> np.ndarray:
        """Positive-class probabilities."""
        return sigmoid(self.decision_function(binned))

    # -------------------------------------------------------- serialisation
    def _stacked(self) -> dict[str, np.ndarray]:
        n_nodes = 2 ** (self.params.max_depth + 1) - 1
        if not self.trees_:
            return {
                name: np.zeros((0, n_nodes), dtype=dtype)
                for name, dtype in TREE_DTYPES.items()
            }
        return {
            name: np.stack([getattr(t, name) for t in self.trees_])
            for name in TREE_DTYPES
        }

    def _meta(self) -> dict[str, Any]:
        return {
            "params": asdict(self.params),
            "intercept": self.intercept_,
            "n_bins": self.n_bins_,
            "n_features": self.n_features_,
            "best_iteration": self.best_iteration_,
            "n_estimators": self.n_estimators_,
        }

    @classmethod
    def _restore(
        cls, meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
    ) -> "GBDTClassifier":
        model = cls(GBDTParams(**meta["params"]))
        model.intercept_ = float(meta["intercept"])
        model.n_bins_, model.n_features_ = int(meta["n_bins"]), int(meta["n_features"])
        model.best_iteration_, model.n_estimators_ = int(meta["best_iteration"]), int(
            meta["n_estimators"]
        )
        model.trees_ = [
            TreeArrays(
                **{
                    k: np.asarray(arrays[k][i], dtype=dtype)
                    for k, dtype in TREE_DTYPES.items()
                }
            )
            for i in range(len(arrays["feature"]))
        ]
        return model


ChunkSource = Callable[[], Iterable[np.ndarray]]


def _chunker(x: Any) -> ChunkSource:
    if callable(x):
        return x
    matrix = np.asarray(x)
    if matrix.ndim != 2:
        raise ValueError(
            "x must be a two-dimensional matrix or a callable yielding row chunks"
        )
    return lambda: iter([matrix])


@dataclass
class _State:
    loss: float
    gradient: np.ndarray
    hessian: np.ndarray


class NewtonLogistic:
    """Chunked IRLS/Newton solver with a backtracking line search; column 0 is an unpenalised bias."""

    def __init__(
        self,
        lambda_: float = 1e-4,
        max_iter: int = 15,
        tol: float = 1e-6,
        max_halvings: int = 20,
    ) -> None:
        if lambda_ < 0 or max_iter < 1 or tol <= 0:
            raise ValueError("lambda_ >= 0, max_iter >= 1 and tol > 0 required")
        self.lambda_, self.max_iter, self.tol = (
            float(lambda_),
            int(max_iter),
            float(tol),
        )
        self.max_halvings = int(max_halvings)
        self.coef_: np.ndarray = np.zeros(0)
        self.n_iter_ = 0

    def _evaluate(self, chunks: ChunkSource, y: np.ndarray, coef: np.ndarray) -> _State:
        d = len(coef)
        gradient, hessian = np.zeros(d), np.zeros((d, d))
        loss, start = 0.0, 0
        for block in chunks():
            block = np.asarray(block, dtype=np.float64)
            stop = start + block.shape[0]
            yb = y[start:stop]
            scores = block @ coef
            prob = sigmoid(scores)
            loss += float(np.sum(np.logaddexp(0.0, scores) - yb * scores))
            gradient += block.T @ (prob - yb)
            hessian += (block * (prob * (1 - prob))[:, None]).T @ block
            start = stop
        if start != len(y):
            raise ValueError("chunks yielded a different number of rows than y")
        penalty = np.ones(d)
        penalty[0] = 0.0
        loss = loss / len(y) + self.lambda_ * float(np.sum(penalty * coef**2))
        gradient = gradient / len(y) + 2 * self.lambda_ * penalty * coef
        hessian = hessian / len(y) + 2 * self.lambda_ * np.diag(penalty)
        return _State(loss, gradient, hessian)

    def fit(self, x: Any, y: Any) -> "NewtonLogistic":
        """Fit on a matrix or on a callable returning an iterable of row chunks."""
        chunks = _chunker(x)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or not np.isin(y, (0, 1)).all():
            raise ValueError("y must be a one-dimensional 0/1 vector")
        first = next(iter(chunks()))
        coef = np.zeros(np.asarray(first).shape[1])
        prior = float(np.sum(y) / len(y))
        coef[0] = np.log(prior / (1 - prior))
        state = self._evaluate(chunks, y, coef)
        self.n_iter_ = 0
        for iteration in range(1, self.max_iter + 1):
            if float(np.abs(state.gradient).max()) < self.tol:
                break
            try:
                direction = np.linalg.solve(state.hessian, state.gradient)
            except np.linalg.LinAlgError:
                # Singular Hessian: minimum-norm step (numpy.linalg.lstsq is not used).
                pseudo_inverse = np.linalg.pinv(state.hessian, hermitian=True)
                direction = pseudo_inverse @ state.gradient
            slope = float(state.gradient @ direction)
            step = 1.0
            for _ in range(self.max_halvings):
                candidate = coef - step * direction
                trial = self._evaluate(chunks, y, candidate)
                if trial.loss <= state.loss - 1e-4 * step * slope:
                    break
                step /= 2
            else:
                logger.warning(
                    "line search failed at iteration %d; stopping", iteration
                )
                break
            coef, state = candidate, trial
            self.n_iter_ = iteration
        self.coef_ = coef
        return self

    def decision_function(self, x: Any) -> np.ndarray:
        """Linear scores for a matrix or chunk callable."""
        return np.concatenate(
            [np.asarray(b, dtype=np.float64) @ self.coef_ for b in _chunker(x)()]
        )

    def predict_proba(self, x: Any) -> np.ndarray:
        """Positive-class probabilities."""
        return sigmoid(self.decision_function(x))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state."""
        return {
            "lambda_": self.lambda_,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "coef": self.coef_.tolist(),
            "n_iter": self.n_iter_,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "NewtonLogistic":
        """Rebuild a fitted solver."""
        model = cls(state["lambda_"], state["max_iter"], state["tol"])
        model.coef_ = np.asarray(state["coef"], dtype=np.float64)
        model.n_iter_ = int(state["n_iter"])
        return model


def fit_gbdt(
    config: Mapping[str, Any], names: list[str], x: np.ndarray, y: np.ndarray
) -> tuple[TreeBinner, GBDTClassifier]:
    """Tree binner + GBDT with a fixed number of trees, fitted on all rows of ``x`` (the final refit)."""
    binner = TreeBinner(names, tree_feature_config(config.get("features", {}))).fit(x)
    model = GBDTClassifier(GBDTParams(**config["model"]))
    model.fit(binner.transform(x), y, n_bins=int(binner.n_bins_.max()))
    return binner, model


def fit_wide_lr(
    config: Mapping[str, Any], names: list[str], x: np.ndarray, y: np.ndarray
) -> tuple[WideFeatureMap, NewtonLogistic]:
    """Wide feature map + chunked Newton logistic regression fitted on ``x``."""
    feature_map = WideFeatureMap(
        names, wide_feature_config(config.get("features", {}))
    ).fit(x)
    logistic = NewtonLogistic(**config.get("model", {})).fit(
        lambda: feature_map.transform_chunks(x), y
    )
    return feature_map, logistic


@dataclass(frozen=True)
class FinalModelSpec:
    """What to train and how to combine it; see the module docstring."""

    name: str
    gbdt: tuple[dict[str, Any], ...]
    wide_lr: dict[str, Any] | None
    weights: tuple[float, ...]
    threshold: float
    feature_names: tuple[str, ...]
    data_hashes: dict[str, str]
    notes: str = ""

    def __post_init__(self) -> None:
        n_components = (1 if self.gbdt else 0) + (1 if self.wide_lr else 0)
        if n_components == 0 or len(self.weights) != n_components:
            raise ValueError(
                "weights must have one entry per component (GBDT bag, wide LR)"
            )
        if abs(sum(self.weights) - 1.0) > 1e-9 or min(self.weights) < 0:
            raise ValueError("weights must be non-negative and sum to one")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must lie in (0, 1)")

    def to_json(self) -> dict[str, Any]:
        """Plain dict with lists instead of tuples."""
        state = asdict(self)
        state["gbdt"], state["weights"] = list(state["gbdt"]), list(state["weights"])
        state["feature_names"] = list(state["feature_names"])
        return state

    @classmethod
    def from_json(cls, state: Mapping[str, Any]) -> "FinalModelSpec":
        """Inverse of :meth:`to_json`."""
        return cls(
            name=state["name"],
            gbdt=tuple(dict(c) for c in state["gbdt"]),
            wide_lr=dict(state["wide_lr"]) if state.get("wide_lr") else None,
            weights=tuple(float(w) for w in state["weights"]),
            threshold=float(state["threshold"]),
            feature_names=tuple(state["feature_names"]),
            data_hashes=dict(state.get("data_hashes", {})),
            notes=str(state.get("notes", "")),
        )


class FinalModel:
    """Fitted components of a :class:`FinalModelSpec`."""

    def __init__(
        self,
        spec: FinalModelSpec,
        binners: list[TreeBinner],
        gbdts: list[GBDTClassifier],
        feature_map: WideFeatureMap | None = None,
        logistic: NewtonLogistic | None = None,
    ) -> None:
        self.spec, self.binners, self.gbdts = spec, binners, gbdts
        self.feature_map, self.logistic = feature_map, logistic

    def component_probabilities(self, x: Any) -> np.ndarray:
        """Columns: GBDT seed-bag average (if any), wide LR (if any)."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.spec.feature_names):
            raise ValueError("x must have one column per spec feature name")
        columns = []
        if self.gbdts:
            columns.append(
                np.mean(
                    [
                        g.predict_proba(b.transform(x))
                        for b, g in zip(self.binners, self.gbdts)
                    ],
                    axis=0,
                )
            )
        if self.logistic is not None and self.feature_map is not None:
            columns.append(
                self.logistic.predict_proba(
                    lambda: self.feature_map.transform_chunks(x)
                )
            )
        return np.column_stack(columns)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Blended positive-class probability."""
        return self.component_probabilities(x) @ np.asarray(self.spec.weights)

    def predict(self, x: Any) -> np.ndarray:
        """Competition labels in {-1, 1} at the spec threshold."""
        return np.where(self.predict_proba(x) >= self.spec.threshold, 1, -1)


def save_model(
    model: FinalModel, path: str | Path, recipe: Mapping[str, Any] | None = None
) -> None:
    """Write every component (and the training recipe) to one compressed ``.npz`` file."""
    arrays: dict[str, Any] = {"spec": json.dumps(model.spec.to_json())}
    if recipe is not None:
        arrays["recipe"] = json.dumps(dict(recipe))
    for i, (binner, gbdt) in enumerate(zip(model.binners, model.gbdts)):
        arrays[f"binner_{i}"] = json.dumps(binner.to_dict())
        arrays[f"gbdt_{i}_meta"] = json.dumps(gbdt._meta())
        arrays.update(
            {f"gbdt_{i}_{key}": value for key, value in gbdt._stacked().items()}
        )
    if model.feature_map is not None and model.logistic is not None:
        arrays["wide_features"] = json.dumps(model.feature_map.to_dict())
        arrays["wide_lr"] = json.dumps(model.logistic.to_dict())
    np.savez_compressed(Path(path), **arrays)


def load_model(path: str | Path) -> tuple[FinalModel, dict[str, Any] | None]:
    """Inverse of :func:`save_model`; returns the model and its training recipe (if stored)."""
    with np.load(Path(path), allow_pickle=False) as data:
        spec = FinalModelSpec.from_json(json.loads(str(data["spec"])))
        recipe = json.loads(str(data["recipe"])) if "recipe" in data.files else None
        binners, gbdts = [], []
        for i in range(len(spec.gbdt)):
            binners.append(TreeBinner.from_dict(json.loads(str(data[f"binner_{i}"]))))
            arrays = {key: data[f"gbdt_{i}_{key}"] for key in TREE_DTYPES}
            gbdts.append(
                GBDTClassifier._restore(json.loads(str(data[f"gbdt_{i}_meta"])), arrays)
            )
        feature_map = logistic = None
        if spec.wide_lr:
            feature_map = WideFeatureMap.from_dict(
                json.loads(str(data["wide_features"]))
            )
            logistic = NewtonLogistic.from_dict(json.loads(str(data["wide_lr"])))
    return FinalModel(spec, binners, gbdts, feature_map, logistic), recipe
