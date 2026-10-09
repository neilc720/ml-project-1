import numpy as np


def mean_squared_error_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Compute the Mean Squared Error loss.
    :param y: Label
    :param tx: features
    :param w: weights
    :return: loss
    """
    e = y - tx @ w
    loss = np.sum(e**2) / (2 * len(y))
    return loss


def mean_squared_error_gradient(
    y: np.ndarray, tx: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """
    Compute the gradient of the Mean Squared Error loss.
    :param y: Label
    :param tx: features
    :param w: weights
    :return: gradient
    """
    e = y - tx @ w
    grad = -1 / len(y) * tx.T @ e
    return grad


def mean_squared_error_gd(
    y, tx, initial_w, max_iters, gamma
) -> tuple[np.ndarray[float], float]:
    """
    Linear regression using gradient descent.
    :param y: Label
    :param tx: features
    :param initial_w: weights
    :param max_iters: number of iterations maximum to perform
    :param gamma: learning rate
    :return: weights, loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        # Update weights
        w = w - gamma * mean_squared_error_gradient(y, tx, w)
    # Compute loss
    loss = mean_squared_error_loss(y, tx, w)

    return w, loss


def mean_squared_error_sgd(
    y, tx, initial_w, max_iters, gamma
) -> tuple[np.ndarray[float], float]:
    """
    Linear regression using stochastic gradient descent.
    :param y: Label
    :param tx: features
    :param initial_w: weights
    :param max_iters: number of iterations maximum to perform
    :param gamma: learning rate
    :return: weights, loss
    """
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for i in range(0, len(y), batch_size):
            # Compute gradient
            grad = mean_squared_error_gradient(y[i:i+batch_size], tx[i:i+batch_size], w)
            # Update weights
            w = w - gamma * grad
    # Compute loss
    loss = mean_squared_error_loss(y, tx, w)
    return w, loss


def normal_equations_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Compute the loss for normal equations.
    :param y: Label
    :param tx: features
    :param w: weights
    :return: loss
    """
    e = y - tx @ w
    loss = 0.5 * np.mean(e**2)
    return loss


def least_squares(y, tx) -> tuple[np.ndarray[float], float]:
    """
    Least squares regression using normal equations.
    :param y: Label
    :param tx: features
    :return: weights, loss
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = normal_equations_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_) -> tuple[np.ndarray[float], float]:
    """
    Ridge regression using normal equations.
    :param y: Label
    :param tx: features
    :param lambda_: regularization parameter
    :return: weights, loss
    """
    N, D = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.eye(D), tx.T @ y)
    loss = normal_equations_loss(y, tx, w)
    return w, loss


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Compute the logistic loss.
    :param y: Label
    :param tx: features
    :param w: weights
    :return: loss
    """
    pred = tx @ w
    loss = -np.mean(y * np.log(sigmoid(pred)) + (1 - y) * np.log(1 - sigmoid(pred)))
    return loss


def logistic_regression(
    y, tx, initial_w, max_iters, gamma
) -> tuple[np.ndarray[float], float]:
    """
    Logistic regression using gradient descent y in {0,1}.
    :param y: Label
    :param tx: features
    :param initial_w: weights
    :param max_iters: number of iterations maximum to perform
    :param gamma: learning rate
    :return: weights, loss
    """

    w = initial_w
    for n_iter in range(max_iters):
        pred = tx @ w
        # Compute gradient
        grad = -1 / len(y) * tx.T @ (y - sigmoid(pred))
        # Update weights
        w = w - gamma * grad
    # Compute loss
    loss = logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma
) -> tuple[np.ndarray[float], float]:
    """
    Regularized logistic regression using gradient descent y in {0,1}.
    :param y: Label
    :param tx: features
    :param initial_w: weights
    :param max_iters: number of iterations maximum to perform
    :param gamma: learning rate
    :param lambda_: regularization parameter
    :return: weights, loss
    """

    w = initial_w
    for n_iter in range(max_iters):
        pred = tx @ w
        # Compute gradient
        grad = -1 / len(y) * tx.T @ (y - sigmoid(pred)) + 2 * lambda_ * w
        # Update weights
        w = w - gamma * grad
    # Compute loss
    loss = logistic_loss(y, tx, w)
    return w, loss
