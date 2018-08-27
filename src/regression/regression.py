#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io

@version : 1.0
@file : regression.py
@software : PyCharm
@time : 2018/8/11 15:59

"""
import numpy as np
import matplotlib.pyplot as plt


def load_data_set(filename):
    """
    读取文件内容,\t分隔,返回数据和标签,都是数组
    :param filename:
    :return:
    """
    num_feat = len(open(filename).readline().split('\t')) - 1
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def standard_regression(x_arr, y_arr):
    """
    这里直接采用代数解直接求取权重大小.这里有个条件是xTx的逆必须存在,
    如果不存在,则直接返回
    :param x_arr:
    :param y_arr:
    :return:
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    xTx = x_mat.T * x_mat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵是奇异的,逆不存在")
        return
    ws = xTx.I * (x_mat.T * y_mat)
    return ws


def lwlr(test_point, x_arr, y_arr, k=0.1):
    """
    局部加权线性回归算法,采用算术表达式直接求出
    局部加权线性回归就是用待测试点附近的点来估测待测试点的预测值,其中k决定了附近的点的范围
    :param test_point: 待测试的点
    :param x_arr:
    :param y_arr:
    :param k:
    :return: 返回测试点的预测值
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    m = np.shape(x_mat)[0]  # 多少行数据
    weights = np.mat(np.eye((m)))  # W的初始化
    # 此for循环计算W
    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = np.exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))
    xTx = x_mat.T * (weights * x_mat)
    if np.linalg.det(xTx) == 0.0:
        print("当前矩阵是奇异的,所以不能求逆")
        return
    # 这里ws表示局部线性回归中的权重
    ws = xTx.I * (x_mat.T * (weights * y_mat))  # 完整的代数解
    return test_point * ws


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    """
    测试局部线性回归
    :param test_arr:所有的待测试数据,一行表示一个待测试数据
    :param x_arr:
    :param y_arr:
    :param k:
    :return:返回每一行待测试数据的预测值
    """
    m = np.shape(test_arr)[0]
    y_hat = np.zeros(m)
    # 调用lwlr来计算待测试数据的预测值
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def run_lwlr():
    x_arr, y_arr = load_data_set("./data/ex0.txt")
    # 比较不同k值下的区别
    y_hat = lwlr_test(x_arr, x_arr, y_arr, 0.003)
    y_hat_1 = lwlr_test(x_arr, x_arr, y_arr, 0.1)
    y_hat_2 = lwlr_test(x_arr, x_arr, y_arr, 0.5)
    y_hat_3 = lwlr_test(x_arr, x_arr, y_arr, 1)

    x_mat = np.mat(x_arr)
    str_index = x_mat[:, 1].argsort(0)  # 按列返回数组值从小到大的索引值
    # x_mat[str_index]的shape是(200, 1 , 2)
    x_sort = x_mat[str_index][:, 0, :]
    fig = plt.figure()

    # k为0.003时的图像
    ax = fig.add_subplot(221)
    ax.set_title("k=0.003")
    ax.plot(x_sort[:, 1], y_hat[str_index])
    # 画出原始数据的散点图,
    # flatten将矩阵的第二列压成一行,A[0]取出二维数组的第一行元素
    ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).T.flatten().A[0],
               s=2, c='red')
    # k为0.1时的图像
    ax = fig.add_subplot(222)
    ax.set_title("k=0.1")
    ax.plot(x_sort[:, 1], y_hat_1[str_index])
    ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).T.flatten().A[0],
               s=2, c='red')
    # k为0.5时的图像
    ax = fig.add_subplot(223)
    ax.set_title("k=0.5")
    ax.plot(x_sort[:, 1], y_hat_2[str_index])
    ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).T.flatten().A[0],
               s=2, c='red')
    # k为1时的图像
    ax = fig.add_subplot(224)
    ax.set_title("k=1")
    ax.plot(x_sort[:, 1], y_hat_3[str_index])
    ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).T.flatten().A[0],
               s=2, c='red')

    plt.show()


def run():
    x_arr, y_arr = load_data_set("./data/ex0.txt")
    ws = standard_regression(x_arr, y_arr)
    print("权重值:\n", ws)
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # matrix.A  中的 A ：array
    # 表示将矩阵 matrix转换为二维数组
    # matrix.A[0] :取二维数组中第一行元素
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])

    x_copy = x_mat.copy()
    # 排序的目的是为了在plot中可以使用
    x_copy.sort(0)
    y_hat = x_copy * ws  # 预测结果
    ax.plot(x_copy[:, 1], y_hat)
    plt.show()

    # 因为y_hat是经过排序后计算得到,所以这里使用x_mat重新计算预测结果
    y_pre = x_mat * ws
    print("y_hat的维数:\n", np.shape(y_pre))
    print("y_hat.T的维数:\n", np.shape(y_pre.T))
    print("y_mat的维数:\n", np.shape(y_mat))

    # 预测值与真实值的相关系数,这里的输出是个矩阵
    # results[i][j]表示第i个随机变量与第j个随机变量的相关系数.
    corr = np.corrcoef(y_pre.T, y_mat)
    print("相关系数:\n", corr)
    # 与corr的结果相同
    print(np.corrcoef(y_mat, y_pre.T))


if __name__ == '__main__':
    # 线性回归算法
    # run()
    # 局部线性回归
    run_lwlr()
