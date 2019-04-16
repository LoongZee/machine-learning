#coding=utf-8
from __future__ import division
import random
import math
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class Multiboosting:
    def __init__(self, X, y, n_estimators=10):
        self.X = X
        self.y = y

        self.n_estimators = n_estimators
        self.n_subcommittees = int(math.sqrt( n_estimators ))
        self.I = [int(math.ceil((i+1)*self.n_estimators/self.n_subcommittees)) for i in range(self.n_subcommittees)]
        self.I[self.n_subcommittees-1]= n_estimators
        self.N = self.y.shape[0]
        self.weights = [1 / self.N for _ in range(self.N)]
        self.alphas = []
        self.estimators = None
        self.count = 0

    def init_estimator(self):
        indices = [i for i in np.random.choice(self.X.shape[0], self.N, p=self.weights)]
        X_tree = np.array([self.X[i, :] for i in indices])
        y_tree = np.array([self.y[i] for i in indices])
        print ("%s / %s" % (self.count, self.n_estimators))

        t1 = time.time()
        clf = DecisionTreeClassifier(max_depth = 15,random_state=0)
        clf.fit(X_tree, y_tree)
        t2 = time.time()

        print ("tree generation time: %s" % (t2 - t1))

        predictions = clf.predict(self.X)

        return clf, predictions

    def train(self):
        self.count = 0
        self.estimators = []
        t1 = time.time()
        k = 0;
        for t_num in range(self.n_estimators):
            self.count += 1
            if k< self.n_subcommittees and  self.I[k] == (t_num+1):
                w = [(-math.log(random.randint(1, 999) / 1000)) for _ in range(self.N)]   #泊松分布
                w_sum = sum(w)
                self.weights = [(i / w_sum) for i in w]   #归一化
                k = k+1
            estimator, y_pred = self.init_estimator()
            errors = np.array([ y_i != y_p for y_i, y_p in zip(self.y, y_pred)])
            agreements = [-1 if e else 1 for e in errors]
            epsilon = sum(errors * self.weights)
            while epsilon >0.5:
                w = [(-math.log(random.randint(1, 999) / 1000)) for _ in range(self.N)]   #泊松分布
                w_sum = sum(w)
                self.weights = [(i / w_sum) for i in w]   #归一化
                k = k+1
                estimator, y_pred = self.init_estimator()
                errors = np.array([y_i != y_p for y_i, y_p in zip(self.y, y_pred)])
                agreements = [-1 if e else 1 for e in errors]
                epsilon = sum(errors * self.weights)
            if epsilon == 0:
                Beta = math.pow(10,-10)
                alpha = 0.5 * np.log(1/Beta)
                self.alphas.append(alpha)
                self.estimators.append(estimator)
                w = [(-math.log(random.randint(1, 999) / 1000)) for _ in range(self.N)]   #泊松分布
                w_sum = sum(w)
                self.weights = [(i / w_sum) for i in w]   #归一化
                k = k+1
            else:                       #计算alpha+更新权值
                print ("epsilon: %s" % epsilon)
                alpha = 0.5 * np.log((1 - epsilon) / epsilon)
                z = 2 * np.sqrt(epsilon * ( 1 - epsilon))
                self.weights = np.array([(weight / z) * np.exp(-1 * alpha * agreement)
                                         for weight, agreement in zip(self.weights, agreements)])
                print ("weights sum: %s" % sum(self.weights))
                self.alphas.append(alpha)
                self.estimators.append(estimator)
        t2 = time.time()
        print ("train took %s s" % (t2 - t1))

    def predict(self, X):
        predicts = np.array([estimator.predict(X) for estimator in self.estimators])
        weighted_prdicts = [[(p_i, alpha) for p_i in p] for alpha, p in zip(self.alphas, predicts)]

        H = []
        for i in range(X.shape[0]):
            bucket = []
            for j in range(len(self.alphas)):
                bucket.append(weighted_prdicts[j][i])
            H.append(bucket)

        return [self.weighted_majority_vote(h) for h in H]

    def weighted_majority_vote(self, h):
        weighted_vote = {}
        for label, weight in h:
            if label in weighted_vote:
                weighted_vote[label] = weighted_vote[label] + weight
            else:
                weighted_vote[label] = weight

        max_weight = 0
        max_vote = 0
        for vote, weight in weighted_vote.items():
            if max_weight < weight:
                max_weight = weight
                max_vote = vote

        return max_vote
