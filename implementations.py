"""The six required course methods, using only NumPy and the standard library.

Vectors have shape (n,), and every method returns its final weights and the
unpenalized mean loss at those weights. No intercept is added automatically.
"""

import operator

import numpy as np


def _data(y, tx, *, binary=False):
    """Validate the public methods' input contract once, before optimization."""
    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    if y.ndim != 1 or tx.ndim != 2 or tx.shape[0] != y.size or y.size == 0:
        raise ValueError("y must have shape (N,) and tx shape (N, D), with N > 0")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(tx)):
        raise ValueError("y and tx must contain finite values")
    if binary and not np.all((y == 0) | (y == 1)):
        raise ValueError("logistic regression requires labels in {0, 1}")
    return y, tx


def _weights(w, n_features):
    w = np.array(w, dtype=float, copy=True)
    if w.ndim != 1 or w.size != n_features or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite 1D vector with D entries")
    return w


def _iterations(max_iters, gamma):
    try:
        max_iters = operator.index(max_iters)
    except TypeError as error:
        raise ValueError("max_iters must be a nonnegative integer") from error
    if max_iters < 0:
        raise ValueError("max_iters must be a nonnegative integer")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    return max_iters


def _regularization(lambda_):
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and nonnegative")


def mean_squared_error_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """Return sum((y - tx @ w)**2) / (2 N), including the required factor 0.5."""
    residual = np.asarray(y) - np.asarray(tx) @ np.asarray(w)
    return 0.5 * np.mean(residual**2)


def mean_squared_error_gradient(
    y: np.ndarray, tx: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Gradient of the half mean squared error."""
    y = np.asarray(y)
    tx = np.asarray(tx)
    return tx.T @ (tx @ np.asarray(w) - y) / len(y)


def mean_squared_error_gd(
    y, tx, initial_w, max_iters, gamma
) -> tuple[np.ndarray, float]:
    """Run exactly max_iters full-batch gradient updates for linear regression."""
    y, tx = _data(y, tx)
    w = _weights(initial_w, tx.shape[1])
    max_iters = _iterations(max_iters, gamma)
    for _ in range(max_iters):
        w -= gamma * mean_squared_error_gradient(y, tx, w)
    return w, mean_squared_error_loss(y, tx, w)


def mean_squared_error_sgd(
    y, tx, initial_w, max_iters, gamma
) -> tuple[np.ndarray, float]:
    """Run max_iters random single-example updates (sampling with replacement).

    max_iters counts updates, not epochs. Set np.random.seed before calling this
    method to reproduce sampling without changing the required public signature.
    The returned loss is evaluated on the entire input dataset.
    """
    y, tx = _data(y, tx)
    w = _weights(initial_w, tx.shape[1])
    max_iters = _iterations(max_iters, gamma)
    for _ in range(max_iters):
        index = np.random.randint(y.size)
        gradient = tx[index] * (tx[index] @ w - y[index])
        w -= gamma * gradient
    return w, mean_squared_error_loss(y, tx, w)


def normal_equations_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """Backward-compatible name for the half mean squared error."""
    return mean_squared_error_loss(y, tx, w)


def _solve_normal_equations(gram, rhs):
    """Solve the normal equations, using a pseudoinverse if they are singular."""
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        # Duplicate or all-zero features need not prevent a valid least-squares
        # solution. The pseudoinverse selects the minimum-norm solution of these
        # normal equations; numpy.linalg.lstsq is deliberately not used.
        return np.linalg.pinv(gram, hermitian=True) @ rhs


def least_squares(y, tx) -> tuple[np.ndarray, float]:
    """Least squares via tx.T @ tx @ w = tx.T @ y."""
    y, tx = _data(y, tx)
    w = _solve_normal_equations(tx.T @ tx, tx.T @ y)
    return w, normal_equations_loss(y, tx, w)


def ridge_regression(y, tx, lambda_) -> tuple[np.ndarray, float]:
    """Minimize half MSE + lambda_ * ||w||²; return the unpenalized half MSE.

    All supplied coefficients, including an intercept column if present, are
    penalized consistently with the required course objective.
    """
    y, tx = _data(y, tx)
    _regularization(lambda_)
    n_samples, n_features = tx.shape
    gram = tx.T @ tx + 2 * n_samples * lambda_ * np.eye(n_features)
    w = _solve_normal_equations(gram, tx.T @ y)
    return w, normal_equations_loss(y, tx, w)


def sigmoid(t):
    """Evaluate the logistic sigmoid without exponentiating positive values."""
    t = np.asarray(t, dtype=float)
    # Underflow simply rounds probabilities near 0/1 to their limiting values.
    with np.errstate(under="ignore"):
        exp_negative_abs = np.exp(-np.abs(t))
    return np.where(
        t >= 0,
        1 / (1 + exp_negative_abs),
        exp_negative_abs / (1 + exp_negative_abs),
    )


def logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """Mean binary cross-entropy, evaluated stably from logits for y in {0, 1}."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.all((y == 0) | (y == 1)):
        raise ValueError("logistic regression requires a 1D label vector in {0, 1}")
    logits = np.asarray(tx) @ np.asarray(w)
    # The signed-margin form also avoids subtracting nearly equal large values
    # for a confident, correct positive prediction.
    with np.errstate(under="ignore"):
        return np.mean(np.logaddexp(0.0, (1 - 2 * y) * logits))


def logistic_regression(y, tx, initial_w, max_iters, gamma) -> tuple[np.ndarray, float]:
    """Run full-batch gradient descent on mean binary cross-entropy."""
    return reg_logistic_regression(y, tx, 0.0, initial_w, max_iters, gamma)


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma
) -> tuple[np.ndarray, float]:
    """Optimize mean binary cross-entropy + lambda_ * ||w||² using GD.

    Labels must be 0/1. The penalty gradient is 2 * lambda_ * w, and the
    returned loss excludes this penalty, as required by the project.
    """
    y, tx = _data(y, tx, binary=True)
    w = _weights(initial_w, tx.shape[1])
    max_iters = _iterations(max_iters, gamma)
    _regularization(lambda_)
    for _ in range(max_iters):
        gradient = tx.T @ (sigmoid(tx @ w) - y) / y.size + 2 * lambda_ * w
        w -= gamma * gradient
    return w, logistic_loss(y, tx, w)
