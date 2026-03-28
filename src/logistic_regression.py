#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from src.metrics import accuracy, f1_macro


def softmax(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


class LogisticRegression:
    """
    Regressão logística multi-classe (softmax + cross-entropy
    """

    def __init__(self, X=None, y=None, n_classes=5, standardize=False, l2_lambda=1e-4):
        self.n_classes = n_classes
        self.l2_lambda = l2_lambda
        self.standardized = standardize
        self.history = {}

        if X is not None and y is not None:
            if standardize:
                self.mu = np.mean(X, axis=0)
                self.sigma = np.std(X, axis=0)
                self.sigma[self.sigma == 0] = 1
                X_st = (X - self.mu) / self.sigma
                self.X = np.hstack((np.ones([X.shape[0], 1]), X_st))
            else:
                self.X = np.hstack((np.ones([X.shape[0], 1]), X))

            self.y = y

            if len(y.shape) == 1 or y.shape[1] == 1:
                self.Y_onehot = np.zeros((self.y.shape[0], self.n_classes))
                for i, val in enumerate(self.y.flatten()):
                    self.Y_onehot[i, int(val)] = 1
            else:
                self.Y_onehot = y

            self.theta = np.zeros((self.X.shape[1], self.n_classes))

    def probability(self, instance):
        n_features_with_bias = self.theta.shape[0]
        x = np.empty([n_features_with_bias])

        x[0] = 1
        x[1:] = np.array(instance[:n_features_with_bias - 1])

        if self.standardized:
            x[1:] = (x[1:] - self.mu) / self.sigma

        return softmax(np.dot(x, self.theta))

    def predict(self, instance):
        p = self.probability(instance)
        return np.argmax(p)

    def cost_function(self, theta=None):
        if theta is None:
            theta = self.theta

        m = self.X.shape[0]
        P = softmax(np.dot(self.X, theta))
        P = np.clip(P, 1e-15, 1 - 1e-15)

        cost = -np.sum(self.Y_onehot * np.log(P)) / m
        l2_cost = (self.l2_lambda / (2 * m)) * np.sum(theta[1:, :] ** 2)

        return cost + l2_cost

    def gradient_descent(self, alpha=0.1, iters=200, batch_size=None, verbose=True):
        m = self.X.shape[0]
        if batch_size is None:
            batch_size = m

        for epoch in range(1, iters + 1):
            indices = np.random.permutation(m)
            for start in range(0, m - batch_size + 1, batch_size):
                idx = indices[start:start + batch_size]
                X_b, Y_b = self.X[idx], self.Y_onehot[idx]

                P = softmax(np.dot(X_b, self.theta))
                dZ = (P - Y_b) / batch_size
                dW = np.dot(X_b.T, dZ)

                reg = (self.l2_lambda / batch_size) * np.vstack([np.zeros((1, self.n_classes)), self.theta[1:, :]])
                self.theta -= alpha * (dW + reg)

            if verbose and epoch % 20 == 0:
                print(f"[Epoch {epoch}/{iters}] cost: {self.cost_function():.4f}")

    def build_model(self, alpha=0.1, iters=200, batch_size=64):
        self.gradient_descent(alpha, iters, batch_size)

    def optim_model(self):
        self.gradient_descent()

    def print_coefs(self):
        print(self.theta)

    def score(self, X_test, Y_onehot_test):
        if self.standardized:
            X_test = (X_test - self.mu) / self.sigma
        X_test_bias = np.hstack((np.ones([X_test.shape[0], 1]), X_test))
        P = softmax(np.dot(X_test_bias, self.theta))
        return accuracy(Y_onehot_test, P), f1_macro(Y_onehot_test, P)

    def save(self, path):
        if self.standardized:
            np.savez(path, theta=self.theta, mu=self.mu, sigma=self.sigma)
        else:
            np.savez(path, theta=self.theta)
        print(f"modelo guardado em {path}")

    def load(self, path):
        data = np.load(path)
        self.theta = data['theta']
        if 'mu' in data:
            self.mu = data['mu']
            self.sigma = data['sigma']
            self.standardized = True
        print(f"modelo carregado de {path}")