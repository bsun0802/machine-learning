from typing import List
import numpy as np


class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """Normal equation: inv(X_transpose * X) * X_transpose * y """
        X = np.insert(np.array(features), 0, 1, axis=1)  # fit intercept
        y = np.array(values).reshape((-1, 1))
        self.weight = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, features: List[List[float]]) -> List[float]:
        """X @ weight, since weight is column vector"""
        pred = np.insert(np.array(features), 0, 1, axis=1) @ self.weight
        return pred.ravel().tolist()

    def get_weights(self) -> List[float]:
        return self.weight.ravel().tolist()


class LinearRegressionWithL2Loss(LinearRegression):
    '''Use L2 loss for weight regularization'''

    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        X = np.insert(np.array(features), 0, 1, axis=1)  # fit intercept
        y = np.array(values).reshape((-1, 1))
        self.weight = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ X.T @ y


if __name__ == '__main__':
    print(np.__version__)
