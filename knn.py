from __future__ import division, print_function
from typing import List

import numpy as np
from collections import Counter


class KNN:
    """k-Nearest-Neighbor model for binary classification.

       KNN is non-parametric model, it carries all training data with it. So traning kNN means read in training teatures and corresponding labels."""

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.training_features = features
        self.labels = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        def majority(li):
            c = Counter(li)
            m = c.most_common()
            max_freq = Counter(li).most_common(1)[0][1]
            majorities = [ele[0] for ele in m if ele[1] == max_freq]
            if len(majorities) > 1:
                "ties in majority vote, output an arbitrary label in tied most-frequent label"
            return majorities[0]

        pred = []
        for point1 in features:
            pd = [self.distance_function(point1, point2) for point2 in self.training_features]
            knearest = [self.labels[i] for i in np.argpartition(pd, self.k)[:self.k]]
            pred.append(majority(knearest))
        return pred


if __name__ == '__main__':
    print(np.__version__)
