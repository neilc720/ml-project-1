import numpy as np


def confusion_matrix(y, y_pred):
    """
    Computes the confusion matrix
    :param y: numpy array of shape=(N, ) true labels
    :param y_pred: numpy array of shape=(N, ) predicted labels
    :return: confusion matrix as a numpy array of shape=(2,2)
    """
    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == -1) & (y_pred == -1))
    fp = np.sum((y == -1) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == -1))
    return tp, tn, fp, fn


def accuracy(tp, tn, fp, fn) -> float:
    """
    Computes the accuracy
    :param tp: true positives
    :param tn: true negatives
    :param fp: false positives
    :param fn: false negatives
    :return: accuracy as a float
    """
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp, fp) -> float:
    """
    Computes the precision
    :param tp: true positives
    :param fp: false positives
    :return: precision as a float
    """
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(tp, fn) -> float:
    """
    Computes the recall
    :param tp: true positives
    :param fn: false negatives
    :return: recall as a float
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(tp, fp, fn) -> float:
    """
    Computes the F1 score
    :param tp: true positives
    :param fp: false positives
    :param fn: false negatives
    :return: F1 score as a float
    """
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0


def evaluate_model(model, x, y):
    """
    Evaluates a model and prints metrics with a confusion matrix visualization.
    :param model: Model instance with a predict method
    :param x: Input features
    :param y: True labels
    """
    import matplotlib.pyplot as plt

    y_pred = model.predict(x)
    tp, tn, fp, fn = confusion_matrix(y, y_pred)
    acc = accuracy(tp, tn, fp, fn)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(tp, fp, fn)
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", rec)
    print("F1 Score: ", f1)
    plt.matshow([[tp, fp], [fn, tn]], cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["Positive", "Negative"])
    plt.yticks([0, 1], ["Positive", "Negative"])
    plt.text(0, 0, tp, ha="center", va="center")
    plt.text(0, 1, fp, ha="center", va="center")
    plt.text(1, 0, fn, ha="center", va="center")
    plt.text(1, 1, tn, ha="center", va="center")
    plt.show()
