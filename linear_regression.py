from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """Normal equation: inv(X_transpose * X) * X_transpose * y """
        X = numpy.insert(numpy.array(features), 0, 1, axis=1)
        y = numpy.array(values).reshape((-1, 1))
        self.weight = numpy.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, features: List[List[float]]) -> List[float]:
        """X @ weight, since weight is column vector"""
        pred = numpy.insert(numpy.array(features), 0, 1, axis=1) @ self.weight
        return pred.ravel().tolist()

    def get_weights(self) -> List[float]:
        return self.weight.ravel().tolist()


class LinearRegressionWithL2Loss(LinearRegression):
    '''Use L2 loss for weight regularization'''

    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        X = numpy.insert(numpy.array(features), 0, 1, axis=1)
        y = numpy.array(values).reshape((-1, 1))
        self.weight = numpy.linalg.inv(X.T @ X + self.alpha * numpy.eye(X.shape[1])) @ X.T @ y


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
