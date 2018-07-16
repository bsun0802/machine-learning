from numbers import Number

import numpy as np
import scipy as sp

from utils import one_hot, softmax, sigmoid, accuracy_score, minibatch
from logiRegr_data import toy_data_binary, data_loader_mnist
from logiRegr_data import toy_data_multiclass_3_classes_non_separable, \
    toy_data_multiclass_5_classes


class LogisticRegression:
    def binary_train(self, X, y, w0=None, b0=None, eta=0.5, max_iterations=1000):
        """ Inputs:
            - X: training features, a N-by-D numpy array, where N is the
            number of training points and D is the dimensionality of features
            - y: binary training labels, a N dimensional numpy array where
            N is the number of training points, indicating the labels of
            training data
            - eta: learning rate

            Returns:
            - w: D-dimensional vector, a numpy array which is the weight
            vector of logistic regression
            - b: scalar, which is the bias of logistic regression

            Find the optimal parameters w and b for inputs X and y.
            Use the Mini-batch gradient descent.
        """
        N, D = X.shape
        assert len(np.unique(y)) == 2
        y = np.array(y)

        w = np.zeros(D)
        if w0 is not None:
            w = w0
        assert len(w) == D, f"w0 length does not match number of features: {D}"

        b = 0
        if b0 is not None:
            b = b0

        w = np.append(b, w)
        X = np.insert(X, 0, 1, axis=1)
        tol = 1e-7

        for i in range(max_iterations):
            idx = minibatch(X)
            w_next = (w + eta / N
                      * np.dot(X[idx, :].T, (y[idx] - sigmoid(np.dot(X[idx, :], w)))))
            if sp.spatial.distance.euclidean(w, w_next) < tol:
                print(f"Converged in {i} iterations.")
                break
            w = w_next

        b = w[0]
        w = w[1:]
        assert w.shape == (D,)
        self.w = w
        self.b = b
        return w, b

    def binary_predict(self, X):
        """
        Inputs:
        - X: testing features, a N-by-D numpy array, where N is the
        number of training points and D is the dimensionality of features

        Returns:
        - preds: N dimensional vector of binary predictions: {0, 1}
        """
        w = np.append(self.b, self.w)
        X = np.insert(X, 0, 1, axis=1)
        return (np.dot(X, w) > 0).astype(int)

    def multinomial_train(self, X, y, C,
                          w0=None,
                          b0=None,
                          eta=0.5,
                          max_iterations=1000):
        """ Inputs:
            - X: training features, a N-by-D numpy array, where N is the
            number of training points and D is the dimensionality of features
            - y: multiclass training labels, a N dimensional numpy array where
            N is the number of training points, indicating the labels of
            training data
            - C: number of classes in the data
            - eta: learning rate
            - max_iterations: maximum number for iterations to perform

            Returns:
            - w: C-by-D weight matrix of multinomial logistic regression, where
            C is the number of classes and D is the dimensionality of features.
            - b: bias vector of length C, where C is the number of classes
        """

        N, D = X.shape

        w = np.zeros((C, D))
        if w0 is not None:
            w = w0
        assert w.shape == (C, D), f"check your w0, its dimension should be: {(C, D)}"

        b = np.zeros(C)
        if b0 is not None:
            b = b0

        W = np.hstack((b.reshape(-1, 1), w)).T  # shape (D+1, C), you could ignore this, I use this
        # shape(N, D + 1),  to visulize vectorization when implement it
        X = np.insert(X, 0, 1, axis=1)
        Y = one_hot(y, nb_class=C)
        P = softmax(X @ W) - Y

        tol = 1e-5
        for it in range(max_iterations):
            idx = minibatch(X)
            W_prev = W
            W = W - eta / N * X[idx, :].T @ P[idx, :]
            P[idx, :] = softmax(X[idx, :] @ W) - Y[idx, :]
            if np.max(np.abs(W_prev - W)) < tol:
                print(f"Converged in {it} iters.")
                break

        w = W.T[:, 1:]
        b = W.T[:, 0]
        assert w.shape == (C, D)
        assert b.shape == (C,)
        self.w = w
        self.b = b

    def multi_class_predict(self, X):
        """ Inputs:
            - X: testing features, a N-by-D numpy array, where N is the
            number of training points and D is the dimensionality of features
            - w: weights of the trained multinomial classifier
            - b: bias terms of the trained multinomial classifier

            Returns:
            - preds: N dimensional vector of multiclass predictions.
            Outputted predictions should be from {0, C - 1}, where
            C is the number of classes

            Make predictions for multinomial classifier.
        """
        N, D = X.shape
        W = np.hstack((self.b.reshape(-1, 1), self.w))
        X = np.insert(X, 0, 1, axis=1)
        preds = np.argmax(softmax(X @ W.T), axis=1)
        assert preds.shape == (N,)
        return preds

    def OVR_train(self, X, y, C, w0=None, b0=None, eta=0.5, max_iterations=1000):
        """ Inputs:
            - X: training features, a N-by-D numpy array, where N is the
            number of training points and D is the dimensionality of features
            - y: multiclass training labels, a N dimensional numpy array,
            indicating the labels of each training point
            - C: number of classes in the data
            - w0: initial value of weight matrix
            - b0: initial value of bias term
            - eta: learning rate
            - max_iterations: maximum number of iterations for gradient descent

            Returns:
            - w: a C-by-D weight matrix of OVR logistic regression
            - b: bias vector of length C
        """
        N, D = X.shape

        w = np.zeros((C, D))
        if w0 is not None:
            w = w0

        b = np.zeros(C)
        if b0 is not None:
            b = b0
        assert w.shape == (C, D), 'wrong shape of weights matrix'
        assert b.shape == (C,), 'wrong shape of bias terms vector'
        for k in range(C):
            y_ovr = (y == k).astype(float)
            w[k], b[k] = self.binary_train(X, y_ovr)
        self.w = w
        self.b = b

    def OVR_predict(self, X):
        """ Inputs:
            - X: testing features, a N-by-D numpy array, where N is the
            number of training points and D is the dimensionality of features
            - w: weights of the trained OVR model
            - b: bias terms of the trained OVR model

            Returns:
            - preds: vector of class label predictions.
            Outputted predictions should be from {0, C - 1}, where
            C is the number of classes.

            Make predictions using OVR strategy and predictions from binary
            classifier.
        """
        N, D = X.shape
        preds = self.multi_class_predict(X)
        assert preds.shape == (N,)
        return preds


class LogisticRegressionWithL2(LogisticRegression):
    """ Logistic Regression with L-2 regularization.
        By adding alpha * |W|^2 , the cost function become convex.
        Gradient Descent(use average of gradient over all samples) will converge.

        lambda is a keyword in python,
        so the constant lambda in the formula is replaced by alpha
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def binary_train(self, X, y, w0=None, b0=None, eta=.5, max_iterations=1000):
        """Return weight vector w, and bias term b"""
        y = np.array(y)
        assert len(np.unique(y)) == 2
        N, D = X.shape

        w = np.zeros(D)
        if w0 is not None:
            w = w0
        assert len(w) == D, f"w0 length does not match number of features: {D}"

        b = 0
        if b0 is not None:
            b = b0
        assert isinstance(b, Number)

        tol = 1e-6
        for t in range(max_iterations):
            b_next = b - eta * np.mean(sigmoid(X @ w + b) - y)
            w_next = w - eta / N * (X.T @ (sigmoid(X @ w + b) - y) + self.alpha * w)
            # use updated b
            # w_next = w - eta / N * (X.T @ (sigmoid(X @ w + b_next) - y) + self.alpha * w)
            if max(abs(b - b_next), abs((w - w_next)).max()) <= tol:
                print(f"Converged in {t} iterations.")
                break
            b, w = b_next, w_next
        self.w = w
        self.b = b
        return w, b

    def multinomial_train(self, X, y, C,
                          w0=None,
                          b0=None,
                          eta=0.5,
                          max_iterations=1000):
        """ Inputs:
            - X: training features, a N-by-D numpy array, where N is the
            number of training points and D is the dimensionality of features
            - y: multiclass training labels, a N dimensional numpy array where
            N is the number of training points, indicating the labels of
            training data
            - C: number of classes in the data
            - eta: learning rate
            - max_iterations: maximum number for iterations to perform

            Returns:
            - w: C-by-D weight matrix of multinomial logistic regression, where
            C is the number of classes and D is the dimensionality of features.
            - b: bias vector of length C, where C is the number of classes
        """

        N, D = X.shape

        w = np.zeros((C, D))
        if w0 is not None:
            w = w0
        assert w.shape == (C, D), f"check your w0, its dimension should be: {(C, D)}"

        b = np.zeros(C)
        if b0 is not None:
            b = b0
        assert len(b) == C, f"bias term should have length: {C}, input: {len(b)}"

        W = w.T  # shape (D, C)
        Y = one_hot(y, nb_class=C)

        tol = 1e-5
        for it in range(max_iterations):
            P = softmax(X @ W + b) - Y
            b_next = b - eta * np.mean(P, axis=0)
            W_next = W - eta / N * (X.T @ P + self.alpha * W)
            if max(abs(W - W_next).max(), max(b - b_next)) <= tol:
                print(f"Converged in {it} iterations.")
                break
            W = W_next
            b = b_next
        w = W.T
        assert w.shape == (C, D)
        assert b.shape == (C,)
        self.w = w
        self.b = b


def run_binary():
    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()
    logiRegr = LogisticRegressionWithL2(alpha=.1)

    logiRegr.binary_train(X_train, y_train)

    train_preds = logiRegr.binary_predict(X_train)
    preds = logiRegr.binary_predict(X_test)
    print('train acc: %f, test acc: %f' %
          (accuracy_score(y_train, train_preds),
           accuracy_score(y_test, preds)))

    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()
    logiRegr = LogisticRegressionWithL2(alpha=.1)

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train]
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test]

    logiRegr.binary_train(X_train, binarized_y_train)

    train_preds = logiRegr.binary_predict(X_train)
    preds = logiRegr.binary_predict(X_test)
    print('train acc: %f, test acc: %f' %
          (accuracy_score(binarized_y_train, train_preds),
           accuracy_score(binarized_y_test, preds)))


def run_multiclass():
    datasets = [(toy_data_multiclass_3_classes_non_separable(),
                 'Synthetic data', 3),
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5),
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data
        logiRegr = LogisticRegressionWithL2(alpha=.1)

        print('One-versus-rest:')
        logiRegr.OVR_train(X_train, y_train, C=num_classes)
        train_preds = logiRegr.OVR_predict(X_train)
        preds = logiRegr.OVR_predict(X_test)
        print('train acc: %f, test acc: %f' %
              (accuracy_score(y_train, train_preds),
               accuracy_score(y_test, preds)))

        print('Multinomial:')
        logiRegr.multinomial_train(X_train, y_train, C=num_classes)
        train_preds = logiRegr.multi_class_predict(X_train)
        preds = logiRegr.multi_class_predict(X_test)
        print('train acc: %f, test acc: %f' %
              (accuracy_score(y_train, train_preds),
               accuracy_score(y_test, preds)))


if __name__ == '__main__':

    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()
    if not args.type or args.type == 'multiclass':
        run_multiclass()
