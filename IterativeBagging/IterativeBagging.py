#coding=utf-8

import time
import math
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class IterativeBagging:
    def __init__(self, X_train, y_train, n_estimators=10):
        self.X = X_train
        self.y = y_train
        self.n_estimators = n_estimators
        self.N = self.y.shape[0]
        self.index = range(self.N)
        self.estimators = []
        self.count = 0

    def init_Bagging(self,X_train,y_train):
        #X_noise = 0.2 * np.random.random((X_train.shape[0], X_train.shape[1])) - 0.1
        #X_train = X_train + X_noise
        av_error = [0 for _ in range(self.N)]
        av_n = [0 for _ in range(self.N)]
        base_estimators = []
        for t_num in range(self.n_estimators):
            indices = [i for i in np.random.choice(X_train.shape[0], self.N)]
            X_tree = np.array([X_train[i, :] for i in indices])
            y_tree = np.array([y_train[i] for i in indices])

            oob_indices = list(set(self.index).difference(set(indices)))
            X_oob = np.array([X_train[i, :] for i in oob_indices])
            y_oob = np.array([y_train[i] for i in oob_indices])
            print ("n_estimators: %s" % (t_num+1))
            print ("train_num: %s,oob_num: %s" % (y_tree.shape[0],y_oob.shape[0]))

            t1 = time.time()
            clf = DecisionTreeRegressor(max_depth = 15,random_state=0)
            clf.fit(X_tree, y_tree)
            base_estimators.append(clf)

            t2 = time.time()
            print ("tree generation time: %s" % (t2 - t1))

            predictions = clf.predict(X_oob)
            for i, indices in zip(range(len(oob_indices)),oob_indices):
                av_error[indices] +=  (y_oob[i] - predictions[i])
                av_n[indices] += 1

        for i in range(self.N):
            if av_n[i] !=0:
                av_error[i] = av_error[i] / av_n[i]

        self.count += 1   #计数迭代的次数
        self.estimators.append(base_estimators) #记录每次迭代的基学习器
        return av_error

    def train(self):
        #误差迭代
        #第一次迭代
        Y_noise = 0.2 * np.random.random((self.X.shape[0])) - 0.1
        y = self.y + Y_noise

        av_error = self.init_Bagging(self.X, y)
        sqrt_error = self.compute_sqrt_error(av_error)
        print (sqrt_error)

        #第二次及以后的迭代
        while True:
            av_error = self.init_Bagging(self.X, av_error)
            sqrt_res = self.compute_sqrt_error(av_error)
            print (sqrt_res)
            if (sqrt_error/sqrt_res)<1.1:
                break
            else:
                sqrt_error = sqrt_res

    def compute_sqrt_error(self, av_error):
        sum_error = 0
        for i in range(len(av_error)):
            sum_error = sum_error + av_error[i] * av_error[i]
        sqrt_error = math.sqrt(sum_error)
        return sqrt_error


    def predict(self,X):
        N_samples = X.shape[0]
        sum_predicts = [0 for _ in range(N_samples)]
        for i in range(self.count):
            av_predicts = [0 for _ in range(N_samples)]
            for j in range(self.n_estimators):
                predict = self.estimators[i][j].predict(X)
                av_predicts = [a+b for a, b in zip(av_predicts, predict)]
            for k in range(N_samples):
                av_predicts[k] /= self.n_estimators
                sum_predicts[k] += av_predicts[k]
        return sum_predicts

    def getClassifier(self,sum_predicts):
        res_predicts = []
        for i in range(len(sum_predicts)):
            bias = []
            for j in range(len(self._classes)):
                bias.append(abs(self._classes[j]-sum_predicts[i]))
            indice =  bias.index(min(bias))
            res_predicts.append(self._classes[indice])
        return res_predicts
