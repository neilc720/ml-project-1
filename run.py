"""Reproduce the submitted predictions from the saved model, or retrain the model from the raw CSVs.

    python run.py            # models/champion_v2.npz -> predictions/champion_v2.csv (the submission)
    python run.py --train    # retrain from dataset/ (30 to 45 minutes on 12 cores) into outputs/

Unzip dataset.zip first; it creates dataset/. The model (see README.md) averages four NumPy
gradient-boosted tree models, blends them 0.85/0.15 with a wide logistic regression and labels a row +1
when the blended probability reaches the stored threshold. Both modes report whether the CSV they write
is byte-identical to the submitted one. Nothing is uploaded anywhere.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import json
import logging
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from features import TreeBinner, file_hash, load_data, tree_feature_config
from helpers import create_csv_submission
from model import (
    FinalModel,
    FinalModelSpec,
    GBDTClassifier,
    GBDTParams,
    best_threshold,
    fit_gbdt,
    fit_wide_lr,
    load_model,
    save_model,
    stratified_fold_ids,
)

ROOT = Path(__file__).resolve().parent
SUBMITTED_SHA256 = "d49680ff34febe4ae44c9070346f797eb7ae3c0d02cf37f87bc47c69e7874ee7"
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
)
logger = logging.getLogger("run")


@dataclass(frozen=True)
class Task:
    """One component fitted in a worker process on the rows outside ``fold`` (``-1`` keeps all rows)."""

    family: str
    config_json: str
    fold: int
    data_dir: str
    cache_dir: str
    folds_path: str


def _rows(
    task: Task,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Training and held-out rows of the task's fold, the feature names and the component configuration."""
    data = load_data(task.data_dir, task.cache_dir)
    with np.load(task.folds_path) as stored:
        fold_id = stored["fold_id"]
    train, held_out = np.flatnonzero(fold_id != task.fold), np.flatnonzero(
        fold_id == task.fold
    )
    x, y = data["x"], data["y"]
    x_train, x_val = np.asarray(x[train], dtype=float), np.asarray(
        x[held_out], dtype=float
    )
    return (
        x_train,
        y[train],
        x_val,
        y[held_out],
        list(data["names"]),
        json.loads(task.config_json),
    )


def cv_fold(task: Task) -> tuple[np.ndarray, int]:
    """Fit one component on four folds; return its probabilities on the fifth and its iteration count.

    The count is the early-stopping best iteration for a GBDT (it sets the final tree count) and the
    number of Newton steps for the logistic regression (reported only).
    """
    x_train, y_train, x_val, y_val, names, config = _rows(task)
    if task.family == "wide_lr":
        feature_map, logistic = fit_wide_lr(config, names, x_train, y_train)
        proba = logistic.predict_proba(partial(feature_map.transform_chunks, x_val))
        return proba.astype(np.float32), logistic.n_iter_
    binner = TreeBinner(names, tree_feature_config(config.get("features", {}))).fit(
        x_train
    )
    binned_train, binned_val = binner.transform(x_train), binner.transform(x_val)
    del x_train, x_val  # release about 0.8 GB of float64 rows before boosting
    gbdt = GBDTClassifier(GBDTParams(**config["model"]))
    gbdt.fit(binned_train, y_train, binned_val, y_val, n_bins=int(binner.n_bins_.max()))
    return gbdt.predict_proba(binned_val).astype(np.float32), gbdt.best_iteration_


def refit(task: Task) -> tuple[Any, Any]:
    """Fit one component on all labelled rows: (binner, GBDT) or (feature map, logistic regression)."""
    x_train, y_train, _, _, names, config = _rows(task)
    fit = fit_gbdt if task.family == "gbdt" else fit_wide_lr
    return fit(config, names, x_train, y_train)


def run_pool(
    function: Callable[[Task], Any],
    tasks: Sequence[Task],
    workers: int,
    threads: int,
    label: str,
) -> list:
    """Run tasks in ``spawn`` worker processes with ``threads`` BLAS threads each; results keep task order."""
    for name in THREAD_VARS:
        os.environ[name] = str(threads)
    results, started = [], time.time()
    with multiprocessing.get_context("spawn").Pool(
        max(1, min(workers, len(tasks)))
    ) as pool:
        for done, result in enumerate(pool.imap(function, tasks), 1):
            results.append(result)
            logger.info(
                "%s: %d/%d components fitted (%.0fs)",
                label,
                done,
                len(tasks),
                time.time() - started,
            )
    return results


def cv_config(
    component: Mapping[str, Any], family: str, cap: Mapping[str, int]
) -> dict[str, Any]:
    """Configuration for cross-validation: GBDTs may grow up to max(minimum, factor x frozen) trees."""
    model = dict(component["model"])
    if family == "gbdt":
        model["n_estimators"] = max(
            int(cap["minimum"]), int(cap["factor"]) * int(model["n_estimators"])
        )
    return {"features": component.get("features", {}), "model": model}


def train(
    recipe: Mapping[str, Any],
    data_dir: Path,
    cache_dir: Path,
    out_dir: Path,
    workers: int,
    threads: int,
) -> FinalModel:
    """5-fold CV over all labelled rows fixes the threshold and the tree counts; then every component is refit.

    Writes ``folds.npz``, ``cv.npz`` (out-of-fold probabilities and iteration counts, one column or row
    per component: the GBDTs, then the logistic regression) and the refit model to ``out_dir``.
    """
    data = load_data(data_dir, cache_dir, include_test=True)
    y, n_splits = data["y"], int(recipe["n_splits"])
    fold_id = stratified_fold_ids(y, n_splits=n_splits, seed=int(recipe["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "folds.npz", fold_id=fold_id)
    paths = (str(data_dir), str(cache_dir), str(out_dir / "folds.npz"))
    components = [("gbdt", c) for c in recipe["gbdt"]] + [
        ("wide_lr", recipe["wide_lr"])
    ]
    tasks = [
        Task(family, json.dumps(cv_config(c, family, recipe["cv_tree_cap"])), k, *paths)
        for family, c in components
        for k in range(n_splits)
    ]
    results = run_pool(cv_fold, tasks, workers, threads, "cross-validation")
    oofs, iterations = [], []
    for i in range(len(components)):
        oof = np.full(len(y), np.nan, dtype=np.float32)
        for k in range(n_splits):
            oof[fold_id == k] = results[i * n_splits + k][0]
        oofs.append(oof)
        iterations.append([results[i * n_splits + k][1] for k in range(n_splits)])
    np.savez(
        out_dir / "cv.npz", oof=np.column_stack(oofs), iterations=np.asarray(iterations)
    )
    blended = np.column_stack([np.mean(oofs[:-1], axis=0), oofs[-1]]) @ np.asarray(
        recipe["weights"]
    )
    threshold, f1 = best_threshold(y, blended)
    multiplier = float(recipe["tree_multiplier"])
    gbdt_configs = tuple(
        {
            **c,
            "model": {
                **c["model"],
                "n_estimators": max(1, int(round(multiplier * np.mean(best)))),
            },
        }
        for c, best in zip(recipe["gbdt"], iterations[:-1])
    )
    spec = FinalModelSpec(
        name=recipe["name"],
        gbdt=gbdt_configs,
        wide_lr=recipe["wide_lr"],
        weights=tuple(recipe["weights"]),
        threshold=threshold,
        feature_names=tuple(data["names"]),
        data_hashes=dict(data["hashes"]),
        notes=f"refit on all {len(y)} labeled rows; threshold from pooled {n_splits}-fold OOF",
    )
    trees = [c["model"]["n_estimators"] for c in gbdt_configs]
    logger.info(
        "pooled out-of-fold F1 %.4f at threshold %r; final tree counts %s",
        f1,
        threshold,
        trees,
    )
    families = [("gbdt", c) for c in spec.gbdt] + [("wide_lr", spec.wide_lr)]
    fitted = run_pool(
        refit,
        [Task(f, json.dumps(c), -1, *paths) for f, c in families],
        workers,
        threads,
        "refit",
    )
    binners, gbdts = (list(part) for part in zip(*fitted[:-1]))
    path = out_dir / f"{spec.name}.npz"
    save_model(FinalModel(spec, binners, gbdts, *fitted[-1]), path, recipe)
    return load_model(path)[0]


def predict(model: FinalModel, data_dir: Path, cache_dir: Path, output: Path) -> None:
    """Write the ±1 predictions for x_test.csv and report whether they match the submitted file."""
    data = load_data(data_dir, cache_dir, include_test=True)
    if list(model.spec.feature_names) != list(data["names"]):
        raise ValueError("feature names or order differ from the saved model")
    if model.spec.data_hashes != data["hashes"]:
        logger.warning("The CSV files differ from the ones the model was trained on.")
    labels = model.predict(np.asarray(data["x_test"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(data["test_ids"], labels, str(output))
    logger.info(
        "Saved %d predictions to %s (positive rate %.3f)",
        len(labels),
        output,
        np.mean(labels == 1),
    )
    if file_hash(output) == SUBMITTED_SHA256:
        logger.info("The CSV is byte-identical to the submitted predictions.")
    else:
        logger.warning(
            "The CSV differs from the submitted predictions (SHA-256 %s...).",
            SUBMITTED_SHA256[:16],
        )


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "dataset"),
        help="folder with the three course CSV files",
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "champion_v2.npz"),
        help="saved model and recipe",
    )
    parser.add_argument(
        "--predictions",
        help="CSV to write (default: predictions/champion_v2.csv, or OUTPUTS/champion_v2.csv with --train)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="retrain from the raw CSVs instead of loading the model",
    )
    parser.add_argument(
        "--outputs",
        default=str(ROOT / "outputs"),
        help="parsed-CSV cache and --train results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(7, (os.cpu_count() or 4) - 2)),
        help="--train: worker processes",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="--train: BLAS threads per worker process",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    data_dir, out_dir = Path(args.data), Path(args.outputs)
    model, recipe = load_model(args.model)
    if args.train:
        if recipe is None:
            raise SystemExit(f"{args.model} holds no training recipe")
        model = train(
            recipe, data_dir, out_dir / "cache", out_dir, args.workers, args.threads
        )
    default = (out_dir if args.train else ROOT / "predictions") / "champion_v2.csv"
    predict(
        model,
        data_dir,
        out_dir / "cache",
        Path(args.predictions) if args.predictions else default,
    )


if __name__ == "__main__":
    main()
