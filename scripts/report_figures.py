"""Generate publication-ready figures used by the project report.

Run from the repository root after the cross-validation artifacts exist::

    python -m scripts.report_figures

The script reads the same saved out-of-fold predictions as
``scripts.evaluate_cv``; it does not train or tune a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import load_data
from model import best_threshold
from scripts.evaluate_cv import average_precision, blend

ROOT = Path(__file__).resolve().parent.parent


def precision_recall_curve(
    y: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return precision and recall at every distinct score threshold."""
    order = np.argsort(-scores, kind="stable")
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.cumsum(ys)[ends]
    precision = tp / (ends + 1)
    recall = tp / np.sum(ys)
    return np.r_[1.0, precision], np.r_[0.0, recall]


def operating_point(
    y: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    """Return precision, recall, and F1 at one decision threshold."""
    predicted = scores >= threshold
    positive = y == 1
    tp = float(np.sum(predicted & positive))
    precision = tp / float(np.sum(predicted))
    recall = tp / float(np.sum(positive))
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def threshold_curves(
    y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return precision, recall, and F1 over an ordered threshold grid."""
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_positive = (y[order] == 1).astype(np.int64)
    positive_prefix = np.r_[0, np.cumsum(sorted_positive)]
    first_predicted = np.searchsorted(sorted_scores, thresholds, side="left")
    predicted = len(y) - first_predicted
    total_positive = positive_prefix[-1]
    true_positive = total_positive - positive_prefix[first_predicted]
    precision = true_positive / np.maximum(predicted, 1)
    recall = true_positive / total_positive
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


def main() -> None:
    """Create the report's two-panel out-of-fold evaluation figure."""
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    y = load_data(ROOT / "dataset", ROOT / "outputs" / "cache")["y"]
    with np.load(ROOT / "outputs" / "cv.npz") as cv:
        oof = cv["oof"]
    with np.load(ROOT / "models" / "final_model.npz", allow_pickle=False) as stored:
        recipe = json.loads(str(stored["recipe"]))

    scores = blend(oof, recipe["weights"])
    threshold, _ = best_threshold(y, scores)
    precision, recall = precision_recall_curve(y, scores)
    point_precision, point_recall, point_f1 = operating_point(y, scores, threshold)
    default_precision, default_recall, default_f1 = operating_point(y, scores, 0.5)
    prevalence = float(np.mean(y == 1))
    ap = average_precision(y, scores)
    thresholds = np.linspace(0.0, 0.6, 301)
    threshold_precision, threshold_recall, threshold_f1 = threshold_curves(
        y, scores, thresholds
    )

    # A few thousand points are visually exact at column width and keep the
    # vector PDF small enough for the report repository.
    keep = np.unique(np.linspace(0, len(recall) - 1, 3000, dtype=int))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_pr, ax_threshold) = plt.subplots(
        1, 2, figsize=(6.9, 2.15), constrained_layout=True
    )
    ax_pr.plot(
        recall[keep],
        precision[keep],
        color="#0072B2",
        linewidth=1.35,
        label=f"Final blend (AP = {ap:.3f})",
    )
    ax_pr.axhline(
        prevalence,
        color="#666666",
        linestyle="--",
        linewidth=0.9,
        label=f"Class prevalence = {prevalence:.3f}",
    )
    ax_pr.scatter(
        [point_recall],
        [point_precision],
        s=26,
        marker="D",
        color="#D55E00",
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
        label=f"Selected $t$ = {threshold:.3f} ($F_1$ = {point_f1:.3f})",
    )
    ax_pr.set_title("(a) Precision-recall", loc="left", fontsize=8, fontweight="bold")
    ax_pr.set(
        xlabel="Recall", ylabel="Precision", xlim=(0.0, 1.0), ylim=(0.0, 1.0)
    )
    ax_pr.set_xticks(np.linspace(0.0, 1.0, 6))
    ax_pr.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_pr.legend(loc="upper right", frameon=False, handlelength=2.2)

    ax_threshold.plot(
        thresholds,
        threshold_precision,
        color="#0072B2",
        linewidth=1.25,
        label="Precision",
    )
    ax_threshold.plot(
        thresholds,
        threshold_recall,
        color="#D55E00",
        linestyle="--",
        linewidth=1.25,
        label="Recall",
    )
    ax_threshold.plot(
        thresholds,
        threshold_f1,
        color="#009E73",
        linestyle="-.",
        linewidth=1.25,
        label="$F_1$",
    )
    ax_threshold.axvline(threshold, color="#222222", linestyle=":", linewidth=0.9)
    ax_threshold.axvline(0.5, color="#777777", linestyle=":", linewidth=0.9)
    ax_threshold.scatter(
        [threshold],
        [point_f1],
        s=23,
        marker="D",
        color="#009E73",
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    ax_threshold.scatter(
        [0.5],
        [default_f1],
        s=22,
        marker="o",
        color="#777777",
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    ax_threshold.annotate(
        f"selected: {point_f1:.3f}",
        xy=(threshold, point_f1),
        xytext=(0.245, 0.515),
        fontsize=7,
        arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.55},
    )
    ax_threshold.annotate(
        f"at 0.5: {default_f1:.3f}",
        xy=(0.5, default_f1),
        xytext=(0.405, 0.145),
        fontsize=7,
        arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.55},
    )
    ax_threshold.set_title(
        "(b) Decision-threshold sensitivity", loc="left", fontsize=8, fontweight="bold"
    )
    ax_threshold.set(
        xlabel="Decision threshold",
        ylabel="Score",
        xlim=(0.0, 0.6),
        ylim=(0.0, 1.0),
    )
    ax_threshold.set_xticks(np.linspace(0.0, 0.6, 7))
    ax_threshold.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_threshold.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
        ncol=3,
        handlelength=1.8,
    )

    for ax in (ax_pr, ax_threshold):
        ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    pdf_path = output_dir / "oof_evaluation_panels.pdf"
    png_path = output_dir / "oof_evaluation_panels.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {pdf_path.relative_to(ROOT)} and {png_path.relative_to(ROOT)}")
    print(
        f"threshold={threshold:.15f} precision={point_precision:.6f} "
        f"recall={point_recall:.6f} f1={point_f1:.6f} "
        f"average_precision={ap:.6f} prevalence={prevalence:.6f}; "
        f"at_0.5 precision={default_precision:.6f} recall={default_recall:.6f} "
        f"f1={default_f1:.6f}"
    )


if __name__ == "__main__":
    main()
