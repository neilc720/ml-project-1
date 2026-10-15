"""
Data preprocessing for the BRFSS 2015 heart-disease dataset (project 1).

The module is organised in three parts:

1. Cleaning and encoding (``Preprocessor``): survey-code handling, missing
   values, one-hot encoding, standardisation. Everything is *fitted on the
   training set only* and then applied identically to the test set, so no
   statistics leak from test to train.
2. Feature selection: ranking by correlation with the label, removal of
   (near-)constant and collinear features.
3. Class balancing: under-/over-sampling, class weights and decision
   threshold tuning for the 9% / 91% label imbalance.

Only numpy is used, as required by the project rules.
"""

import os

import numpy as np

# --------------------------------------------------------------------------- #
# Dataset knowledge
# --------------------------------------------------------------------------- #

# Survey administration columns: interview date, phone-type bookkeeping,
# sequence numbers, ... They carry no medical information and some (SEQNO,
# _PSU) are sample identifiers that would only let a model memorise rows.
ADMIN_COLUMNS = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENUM",
    "PVTRESD1",
    "COLGHOUS",
    "STATERES",
    "CELLFON3",
    "LADULT",
    "CTELNUM1",
    "CELLFON2",
    "CADULT",
    "PVTRESD2",
    "CCLGHOUS",
    "CSTATE",
    "LANDLINE",
    "QSTVER",
    "QSTLANG",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_DUALUSE",
    "_DUALCOR",
    "_LLCPWT",
]


def load_feature_names(data_path):
    """
    Read the column names of x_train.csv (without the leading ``Id`` column).

    Args:
        data_path (str): folder that contains x_train.csv

    Returns:
        list[str]: feature names, aligned with the columns returned by
        ``helper.load_csv_data``.
    """
    with open(os.path.join(data_path, "x_train.csv")) as f:
        header = f.readline().strip().split(",")
    return header[1:]


def clean_survey_codes(x):
    """
    Map BRFSS "don't know / refused" codes to NaN and "none" codes to 0.

    BRFSS uses 7 / 9 (1-digit questions), 77 / 99 (2-digit), 777 / 999 and
    7777 / 9999 for "don't know" and "refused", and 88 / 888 for "none"
    (e.g. zero days). Which convention applies is inferred from the largest
    value observed in the column, so a column whose values go up to 9 is
    treated as a 1-digit question, a column whose values go up to 99 as a
    2-digit question, and so on.

    Args:
        x (np.ndarray): shape (N, D), raw features (NaN allowed)

    Returns:
        np.ndarray: cleaned copy of ``x``
    """
    x = x.astype(float, copy=True)
    col_max = np.nanmax(np.where(np.isnan(x), -np.inf, x), axis=0)
    for j in range(x.shape[1]):
        m = col_max[j]
        col = x[:, j]
        if m <= 9:
            missing, none = (7, 9), ()
        elif m <= 99:
            missing, none = (77, 99), (88,)
        elif m <= 999:
            missing, none = (777, 999), (888,)
        elif m <= 9999:
            missing, none = (7777, 9999), (8888,)
        else:
            continue
        for code in none:
            col[col == code] = 0.0
        for code in missing:
            col[col == code] = np.nan
    return x


# --------------------------------------------------------------------------- #
# Cleaning and encoding
# --------------------------------------------------------------------------- #


class Preprocessor:
    """
    Train-fitted cleaning / imputation / encoding / standardisation pipeline.

    Usage::

        pre = Preprocessor(feature_names)
        tx_train = pre.fit_transform(x_train)
        tx_test = pre.transform(x_test)       # uses train statistics only

    Steps performed by ``fit``:

    1. drop the survey-administration columns (``ADMIN_COLUMNS``),
    2. replace BRFSS special codes (see ``clean_survey_codes``),
    3. drop columns with more than ``nan_threshold`` missing values and
       constant columns,
    4. assign each remaining column a type: ``binary`` (2 distinct values),
       ``categorical`` (3 .. ``onehot_max_card`` distinct values, one-hot
       encoded) or ``numeric`` (everything else, kept as a number),
    5. impute missing values with the training median (numeric) or mode
       (binary / categorical) and, for columns with at least
       ``missing_indicator_min`` missing values, add a 0/1 "was missing"
       indicator column,
    6. z-score standardise every output column with training mean / std,
    7. optionally prepend a bias column of ones.

    Args:
        feature_names (list[str] | None): names of the columns of ``x``; when
            None the admin columns are not dropped by name.
        nan_threshold (float): drop columns with a larger fraction of NaN.
        onehot_max_card (int): one-hot encode columns with 3..this many
            distinct values. Set to 0 to disable one-hot encoding (all
            columns are then kept as single numeric columns).
        missing_indicator_min (float): add a missing indicator for columns
            whose NaN fraction is at least this value.
        add_bias (bool): prepend a constant 1 column to the output.
        drop_columns (list[str]): columns to drop by name.
    """

    def __init__(
        self,
        feature_names=None,
        nan_threshold=0.5,
        onehot_max_card=8,
        missing_indicator_min=0.05,
        add_bias=True,
        drop_columns=ADMIN_COLUMNS,
    ):
        self.feature_names = feature_names
        self.nan_threshold = nan_threshold
        self.onehot_max_card = onehot_max_card
        self.missing_indicator_min = missing_indicator_min
        self.add_bias = add_bias
        self.drop_columns = drop_columns

    # ---- fitting ---------------------------------------------------------- #

    def fit(self, x):
        """Learn all statistics from the training features ``x`` (N, D)."""
        n, d = x.shape
        names = self.feature_names or [f"f{j}" for j in range(d)]
        keep = np.ones(d, dtype=bool)
        if self.feature_names is not None:
            drop = set(self.drop_columns)
            keep &= np.array([nm not in drop for nm in names])

        x = clean_survey_codes(x)
        nan_rate = np.isnan(x).mean(axis=0)
        keep &= nan_rate <= self.nan_threshold

        self.columns_ = []  # list of dicts describing each kept input column
        for j in np.where(keep)[0]:
            col = x[:, j]
            values = np.unique(col[~np.isnan(col)])
            if len(values) <= 1:
                continue  # constant column: no information
            info = {"index": j, "name": names[j], "nan_rate": nan_rate[j]}
            if len(values) == 2:
                info["kind"] = "binary"
                info["fill"] = _mode(col)
                info["values"] = values  # mapped to 0 / 1
            elif 3 <= len(values) <= self.onehot_max_card:
                info["kind"] = "categorical"
                info["fill"] = _mode(col)
                info["values"] = values
            else:
                info["kind"] = "numeric"
                info["fill"] = np.nanmedian(col)
            info["indicator"] = nan_rate[j] >= self.missing_indicator_min
            self.columns_.append(info)

        # Expand once (without standardisation) to compute mean / std.
        raw, self.output_names_ = self._expand(x)
        self.mean_ = raw.mean(axis=0)
        self.std_ = raw.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x):
        """Apply the fitted pipeline to ``x`` (N, D) and return (N, D')."""
        x = clean_survey_codes(x)
        raw, _ = self._expand(x)
        out = (raw - self.mean_) / self.std_
        if self.add_bias:
            out = np.c_[np.ones(out.shape[0]), out]
        return out

    def fit_transform(self, x):
        """``fit`` then ``transform`` on the same data."""
        return self.fit(x).transform(x)

    @property
    def feature_names_out(self):
        """Names of the output columns (including ``bias`` if added)."""
        return (["bias"] if self.add_bias else []) + self.output_names_

    # ---- internals -------------------------------------------------------- #

    def _expand(self, x):
        """Impute + encode the kept columns of (cleaned) ``x``. No scaling."""
        blocks, names = [], []
        for info in self.columns_:
            col = x[:, info["index"]]
            missing = np.isnan(col)
            filled = np.where(missing, info["fill"], col)
            kind = info["kind"]
            if kind == "binary":
                blocks.append((filled == info["values"][1]).astype(float)[:, None])
                names.append(info["name"])
            elif kind == "categorical":
                # One-hot; the first (lowest) category is dropped because the
                # bias column already spans it. Unseen categories map to all 0.
                for v in info["values"][1:]:
                    blocks.append((filled == v).astype(float)[:, None])
                    names.append(f"{info['name']}={v:g}")
            else:
                blocks.append(filled[:, None])
                names.append(info["name"])
            if info["indicator"]:
                blocks.append(missing.astype(float)[:, None])
                names.append(f"{info['name']}_missing")
        return np.hstack(blocks), names


def _mode(col):
    """Most frequent non-NaN value of a 1-D array."""
    values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
    return values[np.argmax(counts)]


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #


def correlation_with_label(x, y):
    """
    Pearson correlation of every column of ``x`` with ``y`` (point-biserial
    correlation when ``y`` is binary). Constant columns get 0.

    Args:
        x (np.ndarray): (N, D) features, no NaN
        y (np.ndarray): (N,) labels

    Returns:
        np.ndarray: (D,) correlations
    """
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (xc * yc[:, None]).sum(axis=0) / denom
    return np.nan_to_num(corr)


def rank_features(x, y):
    """
    Feature indices sorted by decreasing absolute correlation with the label.

    Args:
        x (np.ndarray): (N, D) features
        y (np.ndarray): (N,) labels

    Returns:
        np.ndarray: (D,) column indices, most informative first
    """
    return np.argsort(-np.abs(correlation_with_label(x, y)))


def drop_collinear(x, y, threshold=0.95, protect=()):
    """
    Greedily remove one column of every pair with |corr| > ``threshold``,
    keeping the column that is more correlated with the label.

    Args:
        x (np.ndarray): (N, D) features
        y (np.ndarray): (N,) labels
        threshold (float): absolute correlation above which two columns are
            considered redundant
        protect (iterable[int]): column indices never to drop (e.g. the bias)

    Returns:
        np.ndarray: indices of the columns to keep (increasing order)
    """
    d = x.shape[1]
    label_corr = np.abs(correlation_with_label(x, y))
    std = x.std(axis=0)
    xs = (x - x.mean(axis=0)) / np.where(std == 0, 1, std)
    corr = np.abs(xs.T @ xs / x.shape[0])
    keep = np.ones(d, dtype=bool)
    protect = set(protect)
    for i in np.argsort(-label_corr):  # strongest features claim first
        if not keep[i]:
            continue
        redundant = np.where((corr[i] > threshold) & keep)[0]
        for j in redundant:
            if j != i and j not in protect:
                keep[j] = False
    return np.where(keep)[0]


def select_features(x, y, k=None, collinear_threshold=0.95, protect=()):
    """
    Full selection: drop collinear columns, then keep the ``k`` columns most
    correlated with the label.

    Args:
        x (np.ndarray): (N, D) features
        y (np.ndarray): (N,) labels
        k (int | None): number of features to keep; None keeps all
            non-collinear columns
        collinear_threshold (float): see ``drop_collinear``; None disables
        protect (iterable[int]): column indices always kept and not counted
            in ``k`` (e.g. ``[0]`` for a bias column)

    Returns:
        np.ndarray: selected column indices (increasing order)
    """
    protect = list(protect)
    candidates = np.arange(x.shape[1])
    if collinear_threshold is not None:
        candidates = drop_collinear(x, y, collinear_threshold, protect)
    candidates = np.array([c for c in candidates if c not in protect], dtype=int)
    if k is not None and k < len(candidates):
        ranked = rank_features(x[:, candidates], y)
        candidates = candidates[ranked[:k]]
    return np.sort(np.concatenate([np.array(protect, dtype=int), candidates]))


# --------------------------------------------------------------------------- #
# Class balancing
# --------------------------------------------------------------------------- #


def undersample(x, y, neg_per_pos=1.0, seed=None):
    """
    Randomly drop negatives so that there are ``neg_per_pos`` negatives per
    positive (positives are all kept). Done once, before training.

    Args:
        x (np.ndarray): (N, D) features
        y (np.ndarray): (N,) labels; the positive class is the max label
        neg_per_pos (float): target ratio #negatives / #positives
        seed (int | None): RNG seed

    Returns:
        tuple[np.ndarray, np.ndarray]: shuffled (x, y) subset
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y == y.max())[0]
    neg = np.where(y != y.max())[0]
    n_neg = min(len(neg), int(round(neg_per_pos * len(pos))))
    idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(idx)
    return x[idx], y[idx]


def oversample(x, y, neg_per_pos=1.0, seed=None):
    """
    Duplicate positives (sampling with replacement) until there are
    ``neg_per_pos`` negatives per positive. All negatives are kept.

    Args:
        x (np.ndarray): (N, D) features
        y (np.ndarray): (N,) labels; the positive class is the max label
        neg_per_pos (float): target ratio #negatives / #positives
        seed (int | None): RNG seed

    Returns:
        tuple[np.ndarray, np.ndarray]: shuffled (x, y)
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y == y.max())[0]
    neg = np.where(y != y.max())[0]
    n_pos = max(len(pos), int(round(len(neg) / neg_per_pos)))
    idx = np.concatenate([neg, rng.choice(pos, n_pos, replace=True)])
    rng.shuffle(idx)
    return x[idx], y[idx]


def class_weights(y):
    """
    Per-sample weights that give both classes the same total weight
    (weights average to 1 over the dataset).

    Args:
        y (np.ndarray): (N,) labels

    Returns:
        np.ndarray: (N,) sample weights
    """
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    w_class = len(y) / (len(classes) * counts)
    return w_class[inverse]


def weighted_reg_logistic_regression(
    y, tx, sample_weights, lambda_, initial_w, max_iters, gamma
):
    """
    Regularised logistic regression (GD) with a per-sample weighted loss:
    ``mean_i s_i * logloss_i + lambda_ * ||w||^2``. With ``class_weights(y)``
    this is cost-sensitive learning, an alternative to resampling.

    Args:
        y (np.ndarray): (N,) labels in {0, 1}
        tx (np.ndarray): (N, D) features
        sample_weights (np.ndarray): (N,) non-negative weights
        lambda_ (float): L2 regularisation strength
        initial_w (np.ndarray): (D,) initial weights
        max_iters (int): number of GD steps
        gamma (float): step size

    Returns:
        tuple[np.ndarray, float]: final weights and (unregularised, weighted)
        loss
    """
    w = initial_w.astype(float, copy=True)
    s = sample_weights / sample_weights.sum()  # normalise to sum 1
    for _ in range(max_iters):
        p = _sigmoid(tx @ w)
        grad = tx.T @ (s * (p - y)) + 2 * lambda_ * w
        w = w - gamma * grad
    p = _sigmoid(tx @ w)
    eps = 1e-15
    loss = -np.sum(s * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    return w, loss


def best_f1_threshold(probs, y_true, grid=None):
    """
    Choose the probability threshold that maximises F1 on validation data.
    Shifting the threshold below 0.5 is the cheapest way to compensate for
    class imbalance when the model is trained on the natural distribution.

    Args:
        probs (np.ndarray): (N,) predicted P(y = positive)
        y_true (np.ndarray): (N,) labels, positive class = max label
        grid (np.ndarray | None): thresholds to try (default 0.05 .. 0.95)

    Returns:
        tuple[float, float]: (best threshold, F1 at that threshold)
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    pos = y_true == y_true.max()
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(pos, probs >= t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def f1_score(y_true_pos, y_pred_pos):
    """F1 score from two boolean arrays (True = positive class)."""
    tp = np.sum(y_true_pos & y_pred_pos)
    fp = np.sum(~y_true_pos & y_pred_pos)
    fn = np.sum(y_true_pos & ~y_pred_pos)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))
