from sklearn import datasets
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification

# %matplotlib inline


def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]


# 正则化数据集X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


# 标准化数据集X
def standardize(X):
    X_std = np.array(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # 分母不能等于0的情况
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    for i in range(X.shape[0]):
        for col in range(X.shape[1]):
            if std[col]:
                X_std[i][col] = (X_std[i][col] - mean[col]) / std[col]

    return X_std


# 划分数据集为训练集和测试集
def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    n_train_samples = int(X.shape[0] * (1 - test_size))
    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    return x_train, x_test, y_train, y_test


def accuracy(y, y_pred):
    y = y.reshape(y.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    return np.sum(y == y_pred) / len(y)


class KNN():
    '''
    K邻近分类算法：
    :parameter
    k:int 最邻近个数
    '''

    def __init__(self, k=5):
        self.k = k

    # 计算一个样本与训练集中所有样本的欧氏距离的平方
    def euclidean_distance(self, one_sample, X_train):
        one_sample = one_sample.reshape(1, -1)
        X_train = X_train.reshape(X_train.shape[0], -1)
        distances = np.power(np.tile(one_sample, (X_train.shape[0], 1)) - X_train, 2).sum(axis=1)
        return distances

    # 获取k个近邻的类别标签
    def get_k_neighbor_labels(self, distances, y_train, k):
        k_neighbor_labels = []
        for distance in np.sort(distances)[:k]:
            label = y_train[distances == distance]
            print('lable:{0},shape:{1}'.format(label,label.shape))
            if label.shape[0]>1:
                for i in range(label.shape[0]):
                    k_neighbor_labels.append(label[i][0])
            else:
                k_neighbor_labels.append(label[0][0])
        print(len(np.array(k_neighbor_labels).reshape(-1)))
        return np.array(k_neighbor_labels).reshape(-1)[:k]

    # 进行标签统计，得票最多的标签就是该测试样本的预测标签
    def vote(self, one_sample, X_train, y_train, k):
        distances = self.euclidean_distance(one_sample, X_train)
        # print('distances\'s shape: {0}'.format(distances.shape))
        y_train = y_train.reshape(y_train.shape[0], 1)
        k_neighbor_labels = self.get_k_neighbor_labels(distances, y_train, k)
        print('shape of k_neighbor_labels: {0}'.format(k_neighbor_labels.shape))
        print(k_neighbor_labels)
        find_label, find_count = 0, 0
        for label, count in Counter(k_neighbor_labels).items():
            if count > find_count:
                find_count = count
                find_label = label
        return find_label

    # 对测试集进行预测
    def predict(self, X_test, X_train, y_train):
        y_pred = []
        for sampel in X_test:
            label = self.vote(sampel, X_train, y_train, self.k)
            y_pred.append(label)
        print('y_pred: {0}'.format(y_pred))
        return np.array(y_pred)


if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_data = normalize(iris_data,axis=1,p=2)
    iris_data = standardize(iris_data)
    iris_dataset = (np.array(iris_data),np.array(iris.target))
    print(iris_dataset)
    data = make_classification(n_samples=200, n_features=4, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2)
    #print(data)
    X,y = iris_dataset[0],iris_dataset[1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,shuffle=True)
    clf = KNN(k=5)
    y_pred = clf.predict(X_test,X_train,y_train)

    acc = accuracy(y_test,y_pred)
    print('Accuracy: {0}%'.format(acc*100))
