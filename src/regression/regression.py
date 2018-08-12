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


def stand_regression(x_arr, y_arr):
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


def run():
    x_arr, y_arr = load_data_set("./data/ex0.txt")
    ws = stand_regression(x_arr, y_arr)
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

    print("y_hat的维数:\n", np.shape(y_hat))
    print("y_hat.T的维数:\n", np.shape(y_hat.T))
    print("y_mat的维数:\n", np.shape(y_mat))

    # 预测值与真实值的相关系数,这里的输出是个矩阵
    # results[i][j]表示第i个随机变量与第j个随机变量的相关系数.
    corr = np.corrcoef(y_hat.T, y_mat)
    print("相关系数:\n", corr)
    # 与corr的结果相同
    print(np.corrcoef(y_mat, y_hat.T))


if __name__ == '__main__':
    run()
