from scripts.helper import negative_reduction, sort_abs_correlation
from implementations import *


class Model:
    w: np.ndarray

    def loss(self, y, tx) -> float:
        """
        Return the loss of the model.
        :param y: labels
        :param tx: features
        :return: loss
        """
        pass

    def fit(
        self, y, tx, display=False, stop_loss=0, from_previous_loss=5
    ) -> list[float]:
        """
        Fit the model to the data.
        :param y: labels
        :param tx: features
        :param display: print loss during training
        :param stop_loss: early stopping loss threshold
        :param from_previous_loss: related to the stop loss
        :return: list of losses during training
        """
        pass

    def predict(self, x) -> np.ndarray[int]:
        """
        Predict the labels for the given data.
        :param x: features
        :return: labels
        """
        pass


class AlwaysNegative(Model):
    """
    A model that always predicts -1.
    """

    def __init__(self):
        pass

    def loss(self, y, tx) -> float:
        return 0

    def fit(
        self, y, tx, display=False, stop_loss=0, from_previous_loss=5
    ) -> list[float]:
        return [0]

    def predict(self, x) -> np.ndarray[int]:
        return -1 * np.ones(x.shape[0], dtype=int)


class BernoulliNaiveBayes(Model):
    p: float

    def __init__(self, p):
        self.p = p

    def loss(self, y, tx) -> float:
        return 0

    def fit(
        self, y, tx, display=False, stop_loss=0, from_previous_loss=5
    ) -> list[float]:
        return [0]

    def predict(self, x) -> np.ndarray[int]:
        n_samples = x.shape[0]
        random_values = np.random.rand(n_samples)
        predictions = np.where(random_values < self.p, 1, -1)
        return predictions


class MSEGradientDescent(Model):
    """
    Mean Squared Error regression using Gradient Descent.
    """

    gamma: float
    max_iters: int

    def __init__(self, gamma: float, max_iters: int):
        self.gamma = gamma
        self.max_iters = max_iters

    def loss(self, y, tx) -> float:
        return mean_squared_error_loss(y, tx, self.w)

    def fit(
        self, y, tx, display=False, stop_loss=0, from_previous_loss=5
    ) -> list[float]:
        """
        Linear regression using gradient descent.
        :param display:
        :param y:
        :param tx:
        :return:
        """
        self.w = np.random.rand(tx.shape[1])
        losses = []
        for n_iter in range(self.max_iters):
            self.w, loss = mean_squared_error_gd(y, tx, self.w, 1, self.gamma)
            losses.append(loss)
            if display:
                print(f"Gradient Descent({n_iter + 1}/{self.max_iters}): loss={loss}")
            if (
                n_iter > from_previous_loss
                and losses[-from_previous_loss + 1] - loss < stop_loss * self.gamma * 10
            ):
                break
        return losses

    def predict(self, x) -> np.ndarray[float]:
        return np.sign(x @ self.w)


class RegularizedLogisticRegression(Model):
    """
    Logistic regression with lambda regularization.
    """

    gamma: float
    max_iters: int
    lambda_: float
    max_features: int
    negative_reduction: float
    w: np.ndarray
    sorted_indices: np.ndarray

    def __init__(
        self,
        gamma: float,
        max_iters: int,
        lambda_: float,
        max_features: int,
        negative_reduction=1.0,
    ):
        self.gamma = gamma
        self.max_iters = max_iters
        self.lambda_ = lambda_
        self.max_features = max_features
        self.negative_reduction = negative_reduction

    def loss(self, y, tx) -> float:
        """
        Loss without the regularization term
        """
        # Per project description, convert labels to {0,1} BEFORE entering
        # logistic loss/optimization routines.
        y_01 = (y + 1) / 2
        filtered_tx = tx[:, self.sorted_indices]
        return logistic_loss(y_01, filtered_tx, self.w)

    def fit(
        self, y, tx, display=False, stop_loss=0, from_previous_loss=10
    ) -> list[float]:
        """
        Logistic regression using gradient descent.
        :param display:
        :param y: Label in {-1, 1} format
        :param tx: features
        :return: final weights, final loss
        """
        # Per project description, convert labels to {0,1} BEFORE entering
        # logistic loss/optimization routines (logistic_regression and
        # reg_logistic_regression expect y in {0,1}).
        y_01 = (y + 1) / 2

        self.sorted_indices = sort_abs_correlation(tx, y)[: self.max_features]
        filtered_tx = tx[:, self.sorted_indices]
        self.w = np.random.rand(filtered_tx.shape[1])
        losses = []
        for n_iter in range(self.max_iters):
            # negative_reduction expects labels in {0, 1} format
            reduced_y, reduced_filtered_tx = negative_reduction(
                y_01, filtered_tx, self.negative_reduction
            )
            # Ensure y is {0,1} before entering reg_logistic_regression
            self.w, _ = reg_logistic_regression(
                reduced_y, reduced_filtered_tx, self.lambda_, self.w, 1, self.gamma
            )
            data_loss = logistic_loss(y_01, filtered_tx, self.w)
            losses.append(data_loss)
            if display:
                print(
                    f"Logistic regression ({n_iter + 1}/{self.max_iters}): loss={data_loss}"
                )
            if (
                n_iter > from_previous_loss
                and 0
                < losses[-from_previous_loss + 1] - data_loss
                < stop_loss * self.gamma
            ):
                break
        return losses

    def predict(self, x) -> np.ndarray[int]:
        """
        Predict from 0/1 probabilities then map to -1/1
        :param x: features
        :return: labels
        """
        filtered_x = x[:, self.sorted_indices]
        probabilities = sigmoid(filtered_x @ self.w)
        return np.where(probabilities >= 0.5, 1, -1)
