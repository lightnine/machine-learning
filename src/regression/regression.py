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
    :param x_arr: 训练数据
    :param y_arr: 训练数据回归值
    :param k:
    :return:返回每一行待测试数据的预测值
    """
    m = np.shape(test_arr)[0]
    y_hat = np.zeros(m)
    # 调用lwlr来计算待测试数据的预测值
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def rss_error(y_arr, y_hat_arr):
    """
    计算误差(方差)
    :param y_arr: 真实值
    :param y_hat_arr: 预测值
    :return:
    """
    return ((y_arr - y_hat_arr) ** 2).sum()


def ridge_regression(x_mat, y_mat, lam=0.2):
    """
    岭回归算法,直接采用公式进行计算
    :param x_mat:
    :param y_mat:
    :param lam: lambda值
    :return: 岭回归算法的权重
    """
    xTx = x_mat.T * x_mat
    denom = xTx + lam * np.eye(np.shape(x_mat)[1])
    if np.linalg.det(denom) == 0.0:
        print("矩阵不可逆，无法进行岭回归算法的计算")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    """
    对于不同的lambda进行岭回归的计算
    :param x_arr:
    :param y_arr:
    :return:
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    # 归一化
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_means = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_means) / x_var
    num_test_pts = 30  # 控制有多少个lambda
    w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))
    # 保存log(lambda)的值
    log_lam_mat = np.zeros((1, 30))
    # 求出每个lambda对应的岭回归权重值
    for i in range(num_test_pts):
        ws = ridge_regression(x_mat, y_mat, np.exp(i - 10))
        w_mat[i, :] = ws.T
        log_lam_mat[:, i] = i - 10
    return w_mat, log_lam_mat


def stage_wise(x_arr, y_arr, eps=0.01, num_iter=100):
    """
    forward stagewise regression算法（前向梯度算法）
    是一种近似的 lasso算法
    :param x_arr:
    :param y_arr:
    :param eps:每次特征权重的变化步长
    :param num_iter: 迭代次数
    :return:
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mean = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_mean) / x_var
    m, n = np.shape(x_mat)
    ws = np.zeros((n, 1))
    ws_best = ws.copy()
    return_mat = np.zeros((num_iter, n))  # 保存每次迭代最好的权重值
    for i in range(num_iter):
        # print(ws.T)
        lowest_error = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test
                rss_err = rss_error(y_mat.A, y_test.A)  # 将矩阵转为数组
                if rss_err < lowest_error:
                    lowest_error = rss_err
                    ws_best = ws_test
        ws = ws_best.copy()
        return_mat[i, :] = ws.T
    return return_mat


def run_stage():
    x_arr, y_arr = load_data_set('./data/abalone.txt')
    all_w_001 = stage_wise(x_arr, y_arr, 0.001, 5000)
    print(all_w_001)
    all_w_01 = stage_wise(x_arr, y_arr, 0.01, 200)
    print(all_w_01)


def run_ridge():
    ab_x, ab_y = load_data_set('./data/abalone.txt')
    ridge_weights, log_lam_mat = ridge_test(ab_x, ab_y)
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    # 这里为了正确显示x坐标，所以进行了如下处理，也可以直接画出ridge_weights
    for i in range(np.shape(ridge_weights)[1]):
        label_name = "w" + str(i + 1)
        plt.plot(log_lam_mat.T, ridge_weights[:, i], label=label_name)
        # ax.legend("w"+str(i))
    # ax.plot(ridge_weights)
    plt.xlim((-10, 20))
    plt.ylim((-1.0, 2.5))
    plt.xlabel(r'log($\lambda$)')
    plt.ylabel('w')
    # 显示图例
    plt.legend(loc='upper right')
    plt.grid(True)
    # ax = fig.add_subplot(122)
    # ax.plot(ridge_weights)
    fig.show()


def run_abalone():
    """
    采用abalone数据集，使用局部线性回归算法预测鲍鱼的年龄
    这里的数据已经经过处理，条数：4177条
    数据集网址：https://archive.ics.uci.edu/ml/datasets/abalone
    :return:
    """
    ab_x, ab_y = load_data_set('./data/abalone.txt')
    y_hat_01 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 0.1)
    y_hat_1 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 1)
    y_hat_10 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 10)
    error01 = rss_error(ab_y[0:99], y_hat_01.T)
    error1 = rss_error(ab_y[0:99], y_hat_1.T)
    error10 = rss_error(ab_y[0:99], y_hat_10.T)
    print("k取0.1时的，采用局部加权算法的误差为：", error01)
    print("k取1时的，采用局部加权算法的误差为：", error1)
    print("k取10时的，采用局部加权算法的误差为：", error10)

    # 对于新数据的预测能力，这里是为了查看k值对于过拟合的影响
    y_hat_new_01 = lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 0.1)  # 新数据为ab_x[100:199]
    y_hat_new_1 = lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 1)
    y_hat_new_10 = lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 10)
    error_new_01 = rss_error(ab_y[100:199], y_hat_new_01.T)
    error_new_1 = rss_error(ab_y[100:199], y_hat_new_1.T)
    error_new_10 = rss_error(ab_y[100:199], y_hat_new_10.T)
    print("k取0.1时，局部加权算法对于新数据的误差为：", error_new_01)
    print("k取1时，局部加权算法对于新数据的误差为：", error_new_1)
    print("k取10时，局部加权算法对于新数据的误差为：", error_new_10)

    # 标准线性回归
    ws = standard_regression(ab_x[0:99], ab_y[0:99])
    y_hat = np.mat(ab_x[100:199]) * ws
    error_stand = rss_error(ab_y[100:199], y_hat.T.A)  # A表示将矩阵转为数组
    print("标准线性回归算法对于新数据的误差：", error_stand)


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
    # run_lwlr()
    # 预测鲍鱼的年龄
    # run_abalone()
    # ridge算法
    # run_ridge()
    # forward stagewise regression算法
    run_stage()
