from __future__ import division, print_function
from typing import List

import numpy as np
from utils import euclidean_distance


class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args :
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''

        self.nb_features = 2
        self.w = [0 for i in range(0, nb_features + 1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  :
            features : List of features. First element of each feature vector is 1
            to account for bias
            labels : label of each feature [-1,1]

            Returns :
                True/ False : return True if the algorithm converges else False.
        '''
        for i in range(self.max_iteration):
            prev_w = self.w[:]
            for x, y in zip(features, labels):
                if y * np.dot(self.w, x) <= self.margin:
                    self.w = (np.add(self.w, y * np.divide(x, np.linalg.norm(x)))).tolist()
            if euclidean_distance(prev_w, self.w) < self.margin:
                print("converged in ", i, "iterations")
                return True
        return False

    def reset(self):
        self.w = [0 for i in range(0, self.nb_features + 1)]

    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  :
            features : List of features. First element of each feature vector is 1
            to account for bias

            Returns :
                labels : List of integers of [-1,1]
        '''
        return ((np.dot(features, self.w) > self.margin) * 2 - 1).tolist()  # (True, false) -> (2, 0), minus 1 to (1, -1).

    def get_weights(self) -> List[float]:
        return self.w


if __name__ == '__main__':
    print(np.__version__)
