#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@author : lightnine
@site : https://ligntnine.github.io

@version : 1.0
@file : gda.py
@software : PyCharm
@time : 2018/9/16 20:43

GDA(gaussian discriminant analysis)算法（高斯判别分析算法），
介绍了如何使用高斯判别模型，同时比较了与逻辑回归算法的区别
分别采用了一个真实数据集和一个完全从正态分布产生的数据集进行比较
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class GDA:
    def __init__(self, train_data, train_label):
        """
        GDA算法构造函数
        :param train_data: 训练数据
        :param train_label: 训练数据标签
        """
        self.train_data = train_data
        self.train_label = train_label
        self.positive_num = 0
        self.negative_num = 0
        positive_data = []
        negative_data = []
        for (data, label) in zip(self.train_data, self.train_label):
            if label == 1:  # 正样本
                self.positive_num += 1
                positive_data.append(list(data))
            else:  # 负样本
                self.negative_num += 1
                negative_data.append(list(data))
        # 以下过程是计算GDA算法中的四个参数
        row, col = np.shape(train_data)
        # positive表示正样本的概率值
        self.positive = self.positive_num * 1.0 / row
        self.negative = 1 - self.positive
        self.mu_positive = np.sum(positive_data, 0) * 1.0 / self.positive_num
        self.mu_negative = np.sum(negative_data, 0) * 1.0 / self.negative_num
        positive_deta = positive_data - self.mu_positive
        negative_deta = negative_data - self.mu_negative
        # 协方差
        self.sigma = []
        for deta in positive_deta:
            deta = deta.reshape(1, col)
            ans = deta.T.dot(deta)  # 维度是col * col
            self.sigma.append(ans)
        for deta in negative_deta:
            deta = deta.reshape(1, col)
            ans = deta.T.dot(deta)
            self.sigma.append(ans)
        self.sigma = np.array(self.sigma)
        self.sigma = np.sum(self.sigma, 0)
        self.sigma = self.sigma / row
        self.mu_positive = self.mu_positive.reshape(1, col)
        self.mu_negative = self.mu_negative.reshape(1, col)

    def gaussian(self, x, mean, cov):
        """
        高斯分布概率密度函数
        :param x:输入数据
        :param mean:均值向量
        :param cov:协方差矩阵
        :return:x的概率
        """
        # 特征的数量
        dim = np.shape(cov)[0]
        # 这里是为了处理cov不可逆以及为0的情况
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        xdiff = (x - mean).reshape((1, dim))
        # 概率密度
        prob = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(covdet), 0.5)) * \
               np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob

    def predict(self, test_data):
        predict_label = []
        for data in test_data:
            # 分别计算测试数据属于正样本和负样本的概率，比较大小，决定测试数据的预测分类
            positive_pro = self.gaussian(data, self.mu_positive, self.sigma)
            negative_pro = self.gaussian(data, self.mu_negative, self.sigma)
            if positive_pro >= negative_pro:
                predict_label.append(1)
            else:
                predict_label.append(0)
        return predict_label


def run_real_data():
    """
    测试GDA算法和逻辑回归算法
    这里采用的数据集是breast_cancer数据
    从实验结果来看，GDA算法的结果要稍微高于逻辑回归算法
    :return:
    """
    # 导入乳腺癌数据
    breast_cancer = load_breast_cancer()
    data = np.array(breast_cancer.data)
    label = np.array(breast_cancer.target)
    data = MinMaxScaler().fit_transform(data)
    # 解决画图中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=1 / 4)
    # 数据可视化
    plt.cla()
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label)  # 这里用数据的前两列来标注数据，c表示颜色
    plt.title('乳腺癌测试数据集显示')
    plt.show()
    # GDA结果
    gda = GDA(train_data, train_label)
    test_predict = gda.predict(test_data)
    gda_accuracy = accuracy_score(test_label, test_predict)
    print("GDA的准确率为：", gda_accuracy)

    # 数据可视化
    plt.scatter(test_data[:, 0], test_data[:, 1], c=np.array(test_predict))
    plt.title('GDA分类结果展示')
    plt.show()

    # Logistic回归结果
    lr = LogisticRegression()
    lr.fit(train_data, train_label)
    test_predict = lr.predict(test_data)
    logistic_accuracy = accuracy_score(test_label, test_predict)
    print("Logistic回归的正确率是：", logistic_accuracy)

    # 数据可视化
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_predict)
    plt.title("Logistic回归分类结果显示")
    plt.show()


def run_real_gassian_data():
    """
    这个例子展示了使用完全从高斯分布生成的数据分别进行GDA和逻辑回归的测试
    两个算法对于完全服从高斯分布的结果都表现的非常好，测试数据的准确率均是100%
    :return:
    """
    # 产生服从高斯分布的两类数据，均值不同，但是协方差相同
    mean0 = [2, 3]
    cov = np.mat([[1, 0], [0, 2]])
    # 数据1
    negative_data = np.random.multivariate_normal(mean0, cov, 500).T
    negative_label = np.zeros(np.shape(negative_data)[1])
    # 数据2
    mean1 = [7, 8]
    positive_data = np.random.multivariate_normal(mean1, cov, 500).T
    positive_label = np.ones(np.shape(positive_data)[1])

    # 这里将一维数组进行拼接
    data = np.array([np.concatenate((negative_data[0], positive_data[0])), np.concatenate((negative_data[1],
                                                                                           positive_data[1]))])
    label = np.array([np.concatenate((negative_label, positive_label))])
    data = data.T
    label = label.flatten()
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=1 / 4)

    # 解决画图中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    #  数据可视化
    plt.figure()
    plt.clf()
    # 画出所有的数据
    # plt.plot(negative_data[0], negative_data[1], 'ko')
    # plt.plot(positive_data[0], positive_data[1], 'gs')
    plt.plot(mean0[0], mean0[1], 'rx', markersize=20)
    plt.plot(mean1[0], mean1[1], 'y*', markersize=20)
    # 画出测试数据
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label)
    plt.title("从高斯分布生成的测试数据")
    plt.xlabel('特征 (0)')
    plt.ylabel('特征 (1)')
    plt.show()

    # GDA结果
    gda = GDA(train_data, train_label)
    test_predict = gda.predict(test_data)
    gda_accuracy = accuracy_score(test_label, test_predict)
    print("GDA的准确率为：", gda_accuracy)

    # 数据可视化
    plt.scatter(test_data[:, 0], test_data[:, 1], c=np.array(test_predict))
    plt.title('GDA分类结果展示')
    plt.show()

    # Logistic回归结果
    lr = LogisticRegression()
    lr.fit(train_data, train_label)
    test_predict = lr.predict(test_data)
    logistic_accuracy = accuracy_score(test_label, test_predict)
    print("Logistic回归的正确率是：", logistic_accuracy)

    # 数据可视化
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_predict)
    plt.title("Logistic回归分类结果显示")
    plt.show()


if __name__ == '__main__':
    run_real_gassian_data()
    # run_main()
