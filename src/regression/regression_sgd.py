#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io

@version : 1.0
@file : regression_sgd.py
@software : PyCharm
@time : 2018/8/14 18:56

迭代方式实现线性回归算法,这里没有使用numpy,全部使用的是基本的python操作
所以遇到矩阵相乘,采用的遍历来进行计算
"""
from random import seed
from random import randrange
from math import sqrt
from itertools import islice


# 加载csv文件
def load_csv(filename):
    dataset = list()
    input_file = open(filename)
    # islice 跳过第一行,None表示直到迭代器耗尽
    for row in islice(input_file, 1, None):
        row_list = list(row.split(";"))
        dataset.append(row_list)
    input_file.close()
    return dataset


def str_column_to_float(dataset, column):
    """
    将column列数据string转为float
    :param dataset:
    :param column:
    :return:
    """
    for row in dataset:
        row[column] = float(row[column].strip())


def dataset_minmax(dataset):
    """
    找出每一列的最小最大值
    :param dataset:
    :return: 每一列对应的最小和最大值
    """
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


def normalize_dataset(dataset, minmax):
    """
    归一化每一列
    :param dataset:
    :param minmax:
    :return:
    """
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def cross_validation_split(dataset, n_folds):
    """
    将数据集分成n_folds份
    :param dataset:
    :param n_folds:
    :return: 一个list,包含分隔后的每一份
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def rmse_metric(actual, predicted):
    """
    计算rmse损失
    :param actual:
    :param predicted:
    :return:
    """
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)  # ** 表示平方
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """
    评估算法
    :param dataset:
    :param algorithm:
    :param n_folds:
    :param args:
    :return:
    """
    folds = cross_validation_split(dataset, n_folds)
    scores = list()  # 每一个样本的损失值
    for fold in folds:
        train_set = list(folds)  # 这是根据folds创建新的folds,相当于克隆
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)
    return scores


def predict(row, coefficients):
    """
    预测函数
    :param row: 待预测样本
    :param coefficients: 线性回归系数
    :return: 预测结果
    """
    yhat = coefficients[0]  # 系数的第一项是截距
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat


def coefficients_sgd(train, l_rate, n_epoch):
    """
    采用随机梯度下降算法计算线性回归的系数
    :param train: 训练样本(包含真实结果)
    :param l_rate: 学习率
    :param n_epoch: epoch数,一次epoch表示跑完一次整个训练集
    :return:
    """
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]

    return coef


def linear_regression_sgd(train, test, l_rate, n_epoch):
    """
    采用sgd算法实现的线性回归算法
    :param train:
    :param test:
    :param l_rate:
    :param n_epoch:
    :return:
    """
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return predictions


def run():
    seed(1)
    # winequality-white.csv数据集有4898条数据
    # filename = 'winequality-white.csv'
    # winequality-red.csv数据集有1599条数据
    filename = 'winequality-red.csv'
    filepath = './data/wine-quality/' + filename
    dataset = load_csv(filepath)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)  # 因为读入的数据是string的,所以先进行转换
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    n_folds = 5  # 5倍交叉验证
    l_rate = 0.01
    n_epoch = 50
    scores = evaluate_algorithm(dataset, linear_regression_sgd, n_folds, l_rate, n_epoch)
    print("得分: %s" % scores)
    print("平均 RMSE: %.3f" % (sum(scores) / float(len(scores))))


if __name__ == '__main__':
    run()
