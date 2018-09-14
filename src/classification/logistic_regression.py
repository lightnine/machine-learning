#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io

@version : 1.0
@file : logistic_regression.py
@software : PyCharm
@time : 2018/9/14 20:20

"""
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.special import expit


def load_data_set():
    """
    载入特定的testSet.txt 数据
    提取特征和目标列
    :return: 特征列和目标列
    """
    data_arr = []
    label_arr = []
    fr = open('./data/testSet.txt')  # 只有两列特征的数据集
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_arr.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_arr.append(float(line_arr[-1]))
    return data_arr, label_arr


def sigmoid(x):
    """
    返回sigmoid函数结果
    当x是负的，并且绝对值很大时，np.exp(-x)就会很大，有可能会溢出
    :param x:
    :return:
    """
    # 优化sigmoid，消除警告信息，因为x为绝对值较大的复数时，可以直接返回0
    return 0.0 if x < -709 else 1 / (1 + np.exp(-x))
    # return 1.0 / (1 + np.exp(-x))


def grad_ascent(data_arr, class_label):
    """
    梯度下降算法计算逻辑回归
    :param data_arr:
    :param class_label:
    :return: 权重
    """
    data_mat = np.mat(data_arr)  # 将python中的数组转为numpy中的矩阵
    label_mat = np.mat(class_label).transpose()
    m, n = np.shape(data_mat)
    alpha = 0.001
    max_iter = 500
    weights = np.ones((n, 1))
    for k in range(max_iter):
        predict_label = sigmoid(data_mat * weights)
        error = (predict_label - label_mat)
        # 这里是关键，由梯度下降算法，带入代价函数即可求出此式
        weights = weights - alpha * data_mat.transpose() * error  # 原先weights是narray，经过此步，自动转为matrix
    return weights


def stochastic_grad_ascent0(data_arr, class_labels):
    """
    单样本的随机梯度下降算法，每次只取一个样本进行梯度下降，求取权重
    注意这里没有进行迭代，所以算法的结果会很不好
    stochastic_grad_ascent1进行了迭代
    :param data_arr:
    :param class_labels:
    :return:
    """
    m, n = np.shape(data_arr)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        predict_label = sigmoid(sum(data_arr[i] * weights))
        error = predict_label - class_labels[i]
        weights = weights - alpha * error * data_arr[i]
    return weights


def stochastic_grad_ascent1(data_arr, class_labels, base_alpha=0.01, max_iter=150):
    """
    随机梯度下降算法，迭代计算，将训练样本迭代max_iter次
    :param data_arr: 特征
    :param class_labels: 标签
    :param base_alpha: 基础的步长
    :param max_iter: 最大迭代次数
    :return:
    """
    m, n = np.shape(data_arr)
    weights = np.ones(n)
    for j in range(max_iter):
        data_index = range(m)
        length = len(list(data_index))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + base_alpha
            rand_index = int(random.uniform(0, length))
            predict_label = sigmoid(sum(data_arr[rand_index] * weights))
            error = predict_label - class_labels[rand_index]
            weights = weights - alpha * error * data_arr[rand_index]
            length = length - 1
    return weights


def plot_line(weight):
    # weight是numpy中的matrix，可以通过type查看其类型，
    # 这里getA函数将numpy中的matrix转为numpy中的array,
    # 这里如果不转换，则下面计算y，画图的时候因为x是narray类型，而y是matrix类型，会报错
    weights = weight.getA()
    data_arr, label_arr = load_data_set()
    data_numpy_arr = np.array(data_arr)  # 将python中的数组转为numpy中的数组
    n = np.shape(data_numpy_arr)[0]  # 求取训练样本个数
    # xcord1和xcord2分别记录正例点和负例点
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_arr[i]) == 1:  # 1表示正例
            xcord1.append(data_numpy_arr[i, 1])  # 取第二列特征，因为第一列为截距
            ycord1.append(data_numpy_arr[i, 2])
        else:
            xcord2.append(data_numpy_arr[i, 1])
            ycord2.append(data_numpy_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    # x,y分别对应前两个特征，weights[0]表示截距，画出此直线
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def run_weight_plot():
    """
    采用梯度下降算法逻辑回归
    :return:
    """
    data_arr, label_arr = load_data_set()
    weights = grad_ascent(data_arr, label_arr)
    print(weights)
    print(type(weights))
    plot_line(weights)


def run_stoc0():
    """
    采用随机梯度下降算法进行逻辑回归
    :return:
    """
    data_arr, label_arr = load_data_set()
    weights = stochastic_grad_ascent0(np.array(data_arr), label_arr)
    print(weights)
    print(type(weights))
    # print(weights.transpose())
    # print(np.mat(weights))
    plot_line(np.mat(weights).T)  # 将numpy的array类型转为matrix,装置是因为在画图的时候取得是第几行


def run_stoc1():
    data_arr, label_arr = load_data_set()
    weights = stochastic_grad_ascent1(np.array(data_arr), label_arr, max_iter=1000)
    plot_line(np.mat(weights).T)


# 针对horsColic数据进行处理
# 数据地址 http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/
# 源数据存在缺失值，这里采用的数据是处理过之后的
def classify_vector(x, weights):
    predict_prob = sigmoid(sum(x * weights))
    if predict_prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    fr_train = open('./data/horseColicTraining.txt')
    fr_test = open('./data/horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        cur_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(cur_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(cur_line[-1]))
    train_weights = stochastic_grad_ascent1(np.array(training_set), training_labels,
                                            base_alpha=0.001, max_iter=400)
    error_count = 0
    num_test = 0
    for line in fr_test.readlines():
        num_test += 1
        cur_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(cur_line[i]))
        if int(classify_vector(np.array(line_arr), train_weights)) != int(cur_line[-1]):
            error_count += 1
    error_rate = float(error_count) / num_test
    print("测试集的错误率为： %f" % error_rate)
    return error_rate


def multi_test():
    num_test = 10
    error_sum = 0.0
    for i in range(num_test):
        error_sum += colic_test()
    print("在 %d次迭代之后，平均错误率是：%f" % (num_test, error_sum / float(num_test)))


def run_horse_data():
    # colic_test()
    multi_test()


if __name__ == '__main__':
    # run_weight_plot()
    # run_stoc0()
    # run_stoc1()
    run_horse_data()
