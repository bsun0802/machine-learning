import numpy as np
from typing import List


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
    def to_pow(old_feature, n):
        return [x ** n for x in old_feature]

    def polynomialize(old_feature, k):
        new_feature = old_feature[:]
        for n in range(2, k + 1):
            new_feature.extend(to_pow(old_feature, n))
        return new_feature
    return [polynomialize(f, k) for f in features]


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return np.sqrt(
        np.sum([(ele - point2[i]) ** 2 for i, ele in enumerate(point1)])
    )


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.dot(point1, point2)


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    return np.exp(-.5 * np.dot(point1, point2))


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    precision, recall = precision_recall(real_labels, predicted_labels)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


def precision_recall(known, pred):
    tn = fp = fn = tp = 0
    for actual, predic in zip(known, pred):
        if actual == predic == 0:
            tn += 1
        elif actual == predic == 1:
            tp += 1
        elif actual == 0 and predic == 1:
            fp += 1
        elif actual == 1 and predic == 0:
            fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector to unit magnitude. For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        return [np.array(f) / np.sqrt(np.dot(f, f)) for f in features]


class MinMaxScaler:
    """
    Assume the parameters are valid when __call__
                is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self._newly_created = True

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        def scale():
            X = np.array(features)
            return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

        def transform():
            X = np.array(features)
            return (X - self.min) / (self.max - self.min)

        if self._newly_created:
            self.min = np.amin(features, axis=None)
            self.max = np.amax(features, axis=None)
            self._newly_created = False
            return scale().tolist()
        else:
            return transform().tolist()
