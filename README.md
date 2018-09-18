# machine-learning
机器学习算法,python3实现

## 回归算法
### 线性回归算法
一般线性回归算法[代码位置](https://github.com/lightnine/machine-learning/blob/master/src/regression/regression.py)
包括**常规线性回归，局部加权线性回归，岭回归以及前向梯度回归**

## 分类算法
### logistics regression
逻辑回归算法[代码位置](https://github.com/lightnine/machine-learning/blob/master/src
/classification/logistic_regression.py)
里面分别采用梯度下降和随机梯度下降算法实现，跑了两个数据集.
一个是简单的只有两个特征的。另外一个数据集是用的UCI的horse colic数据集.
通过这个代码可以了解logistic算法的一些知识

### GDA(Gaussian discriminant analysis)
gda算法[代码位置](https://github.com/lightnine/machine-learning/blob/master/src
/classification/gda.py)
里面分别运行了一个真实数据集和一个从高斯分布产生的数据集，展示了如何求取gda算法。
求取gda算法的过程比较简单，只需要根据训练数据计算概率，两个类别的均值，协方差矩阵。然后针对测试数据分别计算属于两个类别的概率，比较大小，
选择概率大的作为测试数据的类别即可