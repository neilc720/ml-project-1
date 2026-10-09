from matplotlib import pyplot as plt
import numpy as np
from scripts.evaluate_utils import confusion_matrix, f1_score
from scripts.helper import random_k_fold
from collections import defaultdict

def plot_losses(list_of_losses, title, labels=None, log_scale=True):
    """
    Plot multiple loss curves on the same figure.

    Args:
        list_of_losses (list[list[float]]): A list containing multiple loss lists.
        title (str): Title of the plot.
        labels (list[str], optional): Labels for each loss list (same length as list_of_losses).
        log_scale (bool, optional): Whether to use log scale on the y-axis.
    """
    plt.figure(figsize=(8, 5))

    for i, losses in enumerate(list_of_losses):
        label = labels[i] if labels and i < len(labels) else f"Loss {i+1}"
        plt.plot(losses, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(tp, fp, fn, tn, title=None):
    plt.matshow([[tp, fp], [fn, tn]], cmap=plt.cm.Blues)
    if title:
        plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["Positive", "Negative"])
    plt.yticks([0, 1], ["Positive", "Negative"])
    plt.text(0, 0, tp, ha="center", va="center")
    plt.text(0, 1, fp, ha="center", va="center")
    plt.text(1, 0, fn, ha="center", va="center")
    plt.text(1, 1, tn, ha="center", va="center")
    plt.show()


def cross_validation(folds, model, kwargs, display=False, stop_loss=0):
    validation_loss = 0
    training_loss = 0
    training_losses = []
    k = len(folds)
    for e in range(k):
        x_train_cv = np.vstack([folds[i][0] for i in range(k) if i != e])
        y_train_cv = np.hstack([folds[i][1] for i in range(k) if i != e])
        x_val_cv = folds[e][0]
        y_val_cv = folds[e][1]

        if display:
            print(f"Validating on fold {e+1}/{k} (args: {kwargs})")
        # Train the model
        m = model(**kwargs)
        train_loss = m.fit(y_train_cv, x_train_cv, display=False, stop_loss=stop_loss)

        # Compute loss on validation set
        loss = m.loss(y_val_cv, x_val_cv)
        training_losses.append(train_loss)
        training_loss += train_loss[-1]
        validation_loss += loss
        if display:
            print(f"Validation loss for fold {e+1}: {loss}")
    return validation_loss / k, training_loss / k, training_losses


def cross_validation_f1(folds, model, kwargs, display=False, stop_loss=0):
    validation_f1 = 0
    tps = 0
    tns = 0
    fps = 0
    fns = 0
    k = len(folds)
    for e in range(k):
        x_train_cv = np.vstack([folds[i][0] for i in range(k) if i != e])
        y_train_cv = np.hstack([folds[i][1] for i in range(k) if i != e])
        x_val_cv = folds[e][0]
        y_val_cv = folds[e][1]

        if display:
            print(f"Validating on fold {e+1}/{k} (args: {kwargs})")
        # Train the model
        m = model(**kwargs)
        _ = m.fit(y_train_cv, x_train_cv, display=False, stop_loss=stop_loss)

        # Compute F1
        y_val_pred = m.predict(x_val_cv)
        tp, tn, fp, fn = confusion_matrix(y_val_cv, y_val_pred)
        tps += tp
        tns += tn
        fps += fp
        fns += fn
        validation_f1 += f1_score(tp, fp, fn)
        if display:
            print(f"Validation F1 score for fold {e+1}: {validation_f1}")
    return validation_f1 / k, tps / k, tns / k, fps / k, fns / k

def tune(x, y, k, model, args):
    folds = random_k_fold(x, y, k)
    validation_losses = []
    training_losses = []
    count = 0
    for a in args:
        count += 1
        print(f"Tuning with args ({count}/{len(args)}): {a}")
        validation_loss, training_loss, all_training_losses = cross_validation(
            folds, model, a, display=False, stop_loss=1e-2
        )
        validation_losses.append(validation_loss)
        training_losses.append(training_loss)
        print(f"Average validation cross-validation loss: {validation_loss}")
        plot_losses(all_training_losses, f"Training loss during cross-validation {a}")

    return validation_losses, training_losses


def tune_f1(x, y, k, model, args):
    folds = random_k_fold(x, y, k)
    f1_validation_scores = []
    count = 0
    for a in args:
        count += 1
        print(f"Tuning with args ({count}/{len(args)}): {a}")
        validation_f1, tps, tns, fps, fns = cross_validation_f1(
            folds, model, a, display=False, stop_loss=1e-2
        )
        f1_validation_scores.append(validation_f1)
        print(f"Average validation cross-validation f1: {validation_f1}")
        plot_confusion_matrix(tps, fps, fns, tns, title=f"Validation result for {a}")
    return f1_validation_scores

def plot_tuning_results(
    tuning_args,
    iterative_arg,
    validation_losses,
    training_losses,
    title,
    validation_deviation=None,
    training_deviation=None,
    x_log=True,
    y_log=True,
    metric="Loss",
):
    ite = [a[iterative_arg] for a in tuning_args]
    if training_losses is not None:
        if training_deviation is None:
            plt.plot(ite, training_losses, label=f"Training {metric}")
        else:
            plt.errorbar(
                ite, training_losses, yerr=training_deviation, label=f"Training {metric}"
            )
    if validation_deviation is None:
        plt.plot(ite, validation_losses, label=f"Validation {metric}")
    else:
        plt.errorbar(
            ite, validation_losses, yerr=validation_deviation, label=f"Validation {metric}"
        )

    plt.xlabel(iterative_arg)
    plt.ylabel(metric)
    plt.title(title)
    if x_log:
        plt.xscale("log")
    if y_log:
        plt.yscale("log")
    plt.legend()
    plt.show()

def show_printing_result_loss(
    args_tuned,
    validation_losses,
    training_losses,
    arg,
    small_marge=0,
    big_marge=0,
    x_log=True,
    y_log=True
):
    # 1. Group results by same arg value
    grouped_val = defaultdict(list)
    grouped_train = defaultdict(list)
    for a, v, t in zip(args_tuned, validation_losses, training_losses):
        grouped_val[a[arg]].append(v)
        grouped_train[a[arg]].append(t)

    # 2. Compute mean (and optionally std)
    arg_values = sorted(grouped_val.keys())
    val_means = [np.mean(grouped_val[g]) for g in arg_values]
    train_means = [np.mean(grouped_train[g]) for g in arg_values]
    val_stds = [np.std(grouped_val[g]) for g in arg_values]
    train_stds = [np.std(grouped_train[g]) for g in arg_values]

    # 3. Slice margins if desired
    start = small_marge
    end = len(arg_values) - big_marge if big_marge > 0 else len(arg_values)

    arg_values = arg_values[start:end]
    val_means = val_means[start:end]
    train_means = train_means[start:end]
    val_stds = val_stds[start:end]
    train_stds = train_stds[start:end]

    # 4. Plot (assuming your plot_tuning_results takes x, val, train)
    plot_tuning_results(
        [{arg: g} for g in arg_values],  # or {arg: g} if generic
        arg,
        val_means,
        train_means,
        f"Loss (mean ± std)",
        validation_deviation=val_stds,
        training_deviation=train_stds,
        x_log=x_log,
        y_log=y_log,
        metric="Loss",
    )

    # Optionally print summary
    print("\nAveraged results:")
    for g, v, t, vs, ts in zip(
        arg_values, val_means, train_means, val_stds, train_stds
    ):
        print(f"{arg}={g:.4g} | val={v:.4f}±{vs:.4f} | train={t:.4f}±{ts:.4f}")

def show_printing_result_two_losses(
    args_tuned,
    validation_losses,
    training_losses,
    arg1,
    arg2,
    x_log=True,
    y_log=True,
):
    """
    Plot loss vs two hyperparameters with two x-axes:
      - Bottom axis: arg1
      - Top axis: arg2
    """
    # 1. Group results by same arg value
    grouped_val_arg1 = defaultdict(list)
    grouped_train_arg1 = defaultdict(list)
    grouped_val_arg2 = defaultdict(list)
    grouped_train_arg2 = defaultdict(list)
    for a, v, t in zip(args_tuned, validation_losses, training_losses):
        grouped_val_arg1[a[arg1]].append(v)
        grouped_train_arg1[a[arg1]].append(t)
        grouped_val_arg2[a[arg2]].append(v)
        grouped_train_arg2[a[arg2]].append(t)

    # 2. Compute mean (and optionally std)
    arg_values_arg1 = sorted(grouped_val_arg1.keys())
    val_means_arg1 = [np.mean(grouped_val_arg1[g]) for g in arg_values_arg1]
    train_means_arg1 = [np.mean(grouped_train_arg1[g]) for g in arg_values_arg1]
    val_stds_arg1 = [np.std(grouped_val_arg1[g]) for g in arg_values_arg1]
    train_stds_arg1 = [np.std(grouped_train_arg1[g]) for g in arg_values_arg1]

    arg_values_arg2 = sorted(grouped_val_arg2.keys())
    val_means_arg2 = [np.mean(grouped_val_arg2[g]) for g in arg_values_arg2]
    train_means_arg2 = [np.mean(grouped_train_arg2[g]) for g in arg_values_arg2]
    val_stds_arg2 = [np.std(grouped_val_arg2[g]) for g in arg_values_arg2]
    train_stds_arg2 = [np.std(grouped_train_arg2[g]) for g in arg_values_arg2]

    # Plotting
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(arg1)
    ax1.set_ylabel('Loss')
    ax1.errorbar(
        arg_values_arg1, val_means_arg1, yerr=val_stds_arg1, label=f"{arg1} validation loss", color="blue"
    )
    ax1.errorbar(
        arg_values_arg1, train_means_arg1, yerr=train_stds_arg1, label=f"{arg1} training loss", color="green"
    )
    ax1.tick_params(axis='y')
    if x_log:
        ax1.set_xscale('log')
    if y_log:
        ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax2 = ax1.twiny()
    color = 'tab:red'
    ax2.set_xlabel(arg2)
    ax2.errorbar(
        arg_values_arg2, val_means_arg2, yerr=val_stds_arg2, label=f"{arg2} validation loss", color="red"
    )
    ax2.errorbar(
        arg_values_arg2, train_means_arg2, yerr=train_stds_arg2, label=f"{arg2} training loss", color="orange"
    )
    ax2.tick_params(axis='y')
    if x_log:
        ax2.set_xscale('log')
    if y_log:
        ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    plt.title("Loss (mean + std)")
    plt.show()

    # Optionally print summary
    print("\nAveraged results:")
    for g, v, t, vs, ts in zip(
            arg_values_arg1, val_means_arg1, train_means_arg1, val_stds_arg1, train_stds_arg1
    ):
        print(f"{arg1}={g:.4g} | val={v:.4f}±{vs:.4f} | train={t:.4f}±{ts:.4f}")
    print("\nAveraged results:")
    for g, v, t, vs, ts in zip(
            arg_values_arg2, val_means_arg2, train_means_arg2, val_stds_arg2, train_stds_arg2
    ):
        print(f"{arg2}={g:.4g} | val={v:.4f}±{vs:.4f} | train={t:.4f}±{ts:.4f}")


def show_printing_result_f1(
    args_tuned,
    f1_validation_scores,
    arg,
    small_marge=0,
    big_marge=0,
    x_log=True,
    y_log=False
):
    # 1. Group results by same arg value
    grouped_f1 = defaultdict(list)
    for a, f1 in zip(args_tuned, f1_validation_scores):
        grouped_f1[a[arg]].append(f1)

    # 2. Compute mean (and optionally std)
    arg_values = sorted(grouped_f1.keys())
    f1_means = [np.mean(grouped_f1[g]) for g in arg_values]
    f1_stds = [np.std(grouped_f1[g]) for g in arg_values]

    # 3. Slice margins if desired
    start = small_marge
    end = len(arg_values) - big_marge if big_marge > 0 else len(arg_values)

    arg_values = arg_values[start:end]
    f1_means = f1_means[start:end]
    f1_stds = f1_stds[start:end]

    # 4. Plot (assuming your plot_tuning_results takes x, val, train)
    plot_tuning_results(
        [{arg: g} for g in arg_values],  # or {arg: g} if generic
        arg,
        f1_means,
        None,
        f"F1 Score (mean ± std)",
        validation_deviation=f1_stds,
        training_deviation=None,
        x_log=x_log,
        y_log=y_log,
        metric="F1 Score",
    )

    # Optionally print summary
    print("\nAveraged results:")
    for g, f1, fs in zip(
        arg_values, f1_means, f1_stds
    ):
        print(f"{arg}={g:.4g} | val F1={f1:.4f}±{fs:.4f}")