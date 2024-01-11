# K-Nearest Neighbor K近邻算法

> 该文章作为机器学习的第三篇文章，主要介绍的是K紧邻算法，这是机器学习中最简单的一种分类算法，也是机器学习中最基础的一种算法。
>
> 难度系数：⭐
>
> 更多相关工作请参考：[Github](https://github.com/swx3027925806/MachineLearning)

## 算法介绍

K近邻算法（K-Nearest Neighbor，KNN）是一种基本分类与回归方法。该方法的思想是：如果一个样本在特征空间中距离一个集合中的样本最近的k个样本中的大多数属于某一个类别，则该样本也属于这个类别。该方法在定类决策上具有高度的**局部性**，属于**懒惰学习**（Lazy Learning）与**距离近原则**。

### 算法原理解析
首先，我们需要明确KNN算法的基本步骤：
1. **计算距离：** 对于给定的新样本，我们需要计算它与训练集中每个样本的距离。距离的计算可以使用欧几里得距离、曼哈顿距离等不同的度量方式。
2. **选择最近邻：** 根据计算出的距离，选择距离最近的k个样本作为最近邻。
3. **进行分类：** 基于这k个最近邻的类别标签进行投票，得票最多的类别就是新样本的预测类别。

接下来，我会详细解释每个步骤的实现细节：

#### 计算距离

1. **欧几里得距离：** 这是最常见的距离计算方式，适用于连续特征。如果两个样本分别为 $(x_1)$ 和 $(x_2)$，每个特征维度分别为 $(d_1)$ 和 $(d_2)$，则它们之间的欧几里得距离为：
$$
D(x_1, x_2) = \sqrt{\sum_{i=1}^{d}(x_{1i} - x_{2i})^2}
$$
2. **曼哈顿距离：** 也称为城市街区距离，适用于离散特征或有序特征。其计算方式为：
$$
D(x_1, x_2) = \sum_{i=1}^{d}|x_{1i} - x_{2i}|
$$

#### 选择最近邻

在得到每个样本与新样本的距离后，我们需要选择距离最小的k个样本作为最近邻。通常可以使用排序算法对距离进行排序，然后选择最小的k个。
进行分类：

对于每个最近邻，根据其类别标签进行投票。通常使用简单投票法，即每个最近邻的投票权重与其距离成反比，距离越近的最近邻投票权重越大。

#### 进行分类

最后，将所有投票的结果进行汇总，得票最多的类别即为新样本的预测类别。

这就是KNN算法的基本实现原理。需要注意的是，KNN算法的效果很大程度上取决于k的选择和距离度量的方式。在实际应用中，可以通过交叉验证等方法来选择合适的k值。同时，对于连续特征，可能需要进行离散化或使用其他度量方式来计算距离。

## 数据集介绍

本次实验依旧采用鸢尾花数据集作为实验数据，如果对这部分有不确定的同学可以访问[机器学习原理到Python代码实现之NaiveBayes【朴素贝叶斯】](https://blog.csdn.net/qq_44961028/article/details/135450691)这篇文章，看一下其中数据分析部分。

鸢尾花(Iris)数据集是一个常用的分类实验数据集，由Fisher在1936年收集整理。该数据集包含150个样本，每个样本有四个属性：花萼长度、花萼宽度、花瓣长度和花瓣宽度，这四个属性用于预测鸢尾花属于Setosa、Versicolour或Virginica三个种类中的哪一类。

鸢尾花数据集的特点是具有多重变量，即每个样本都有多个特征，这使得它成为进行多元分类任务的一个理想选择。通过分析这些特征，可以了解不同鸢尾花品种之间的差异，并建立分类器来自动识别未知样本的种类。

鸢尾花数据集的来源是实际的鸢尾花测量数据，因此它具有实际背景和应用价值。这个数据集经常被用于机器学习和数据挖掘算法的实验和验证，因为它提供了多变量分析的一种有效方式。

在本次朴素贝叶斯分类中，我们计划采用这个数据集作为我们的实验对象。

## 代码实现

这里我们依旧提供自己实现的KNN算法和sklearn库中的KNN算法实现调用。

### KNN算法构建

考虑到一般我们所验证的数据都是在小数据集上，所以我们将数据转换成numpy格式，方便后续的计算。
KNN其实是没有任何参数的，所以我们只需要将数据集传入即可。
主要的问题是在验证阶段，KNN将验证的数据集与训练的数据集进行比较，这会花费大量的时间。
以下是代码实现：





```python
# 准备好我们需要使用的第三方包
import numpy as np
import pandas as pd
```


```python
# 构建K近邻算法
# 为了提升代码的效率，我们这里将代码转移到numpy格式
class KNN:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.mean = None            # 通过记录训练集的mean和std来进行归一化，提升性能
        self.std = None

    def fit(self, X_train, y_train):
        # 将X_train数据进行归一化处理
        self.X_train = self.get_normalized_data(X_train)
        # 将X_train增加一个维度，目的是在测试时可以和测试集快速匹配
        self.X_train = np.expand_dims(self.X_train, axis=1)
        # 将y_train赋值给self.y_train
        self.y_train = y_train
    
    # 对X_train数据进行归一化处理
    def get_normalized_data(self, X_train):
        # 计算X_train的均值
        self.mean = np.mean(X_train, axis=0)
        # 计算X_train的标准差
        self.std = np.std(X_train, axis=0)
        # 返回归一化处理后的X_train
        return (X_train - self.mean) / self.std

    # 对X数据进行归一化处理
    def feature_normalization(self, X):
        # 返回归一化处理后的X
        return (X - self.mean) / self.std

    # 计算欧式距离
    def euclidean_distance(self, x1, x2):
        # 返回x1和x2之间的欧式距离
        return np.sqrt(np.sum((x1 - x2)**2))

    # 预测函数
    def predict(self, X_test, k=3):
        # 将X_train复制k份，拼接到X_test上
        x_train = np.tile(self.X_train, (1, X_test.shape[0], 1))
        # 将X_test进行归一化处理
        X_test = self.feature_normalization(X_test)
        # 将X_test增加一个维度
        X_test = np.expand_dims(X_test, axis=0)
        # 将X_test复制k份，拼接到x_train上
        X_test = np.tile(X_test, (x_train.shape[0], 1, 1))
        # 计算x_train和X_test之间的欧式距离
        distance = np.sqrt(np.sum((x_train - X_test)**2, axis=2))
        # 将distance按列排序，取出前k个距离
        distance = np.argsort(distance, axis=0)[:k, :]
        # 取出距离对应的y_train
        k_nearest_y = self.y_train[distance.T]
        # 计算k个距离对应的标签的最大值
        pred = np.max(k_nearest_y, axis=1)
        # 返回预测结果
        return pred

    # 计算准确率
    def score(self, X_test, y_test, k=3):
        # 调用predict函数，预测结果
        y_predict = self.predict(X_test, k)
        # 返回预测结果和真实结果的匹配率
        return np.sum(y_predict == y_test) / len(y_test)
```

### 思考题

我们已经重构出了KNN算法，但现在的代码是最有的吗？这里给大家一些提示，作为后期优化的思路；

1. 在测试阶段，我们是选择直接选择最近的K个中出现最多的元素，那可不可以优化？
2. K近邻在测试时需要消耗大量的计算资源，有没有什么方法可以减少计算量？【降维和聚类】关于这个点其实在现在的大模型中也被广泛应用。

### 数据加载


```python
# 加载数据集

train_dataset = pd.read_csv('dataset\\iris_training.csv')
test_dataset = pd.read_csv('dataset\\iris_test.csv')

X_train = train_dataset.drop('virginica', axis=1).to_numpy()
y_train = train_dataset['virginica'].to_numpy()

X_test = test_dataset.drop('virginica', axis=1).to_numpy()
y_test = test_dataset['virginica'].to_numpy()
```

### 算法调用


```python
# 实验
knn = KNN()
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test, 3)
accuracy
```




    0.9333333333333333



KNN的算法相较于其他算法无疑是简单的，但是其计算量是巨大的。可以看到其主要计算都集中在self.predict(X_test, k)这一步。
接下来我们通过SKlearn的方式实现以下：




```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
accuracy
```




    0.9666666666666667



## 实验对比

和朴素贝叶斯一致，通过我们自己的数据集来验证一下K近邻算法的性能，这里西安简单对于距离做一个加权。


```python
# 构建K近邻算法
# 为了提升代码的效率，我们这里将代码转移到numpy格式
class KNN:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.mean = None            # 通过记录训练集的mean和std来进行归一化，提升性能
        self.std = None
        self.clas = None

    def fit(self, X_train, y_train):
        # 将X_train数据进行归一化处理
        self.X_train = self.get_normalized_data(X_train)
        # 将X_train增加一个维度，目的是在测试时可以和测试集快速匹配
        self.X_train = np.expand_dims(self.X_train, axis=1)
        # 将y_train赋值给self.y_train
        self.y_train = y_train
        self.clas = len(np.unique(y_train))
    
    # 对X_train数据进行归一化处理
    def get_normalized_data(self, X_train):
        # 计算X_train的均值
        self.mean = np.mean(X_train, axis=0)
        # 计算X_train的标准差
        self.std = np.std(X_train, axis=0)
        # 返回归一化处理后的X_train
        return (X_train - self.mean) / self.std

    # 对X数据进行归一化处理
    def feature_normalization(self, X):
        # 返回归一化处理后的X
        return (X - self.mean) / self.std

    # 计算欧式距离
    def euclidean_distance(self, x1, x2):
        # 返回x1和x2之间的欧式距离
        return np.sqrt(np.sum((x1 - x2)**2))

    # 预测函数
    def predict(self, X_test, k=3):
        # 将X_train复制k份，拼接到X_test上
        x_train = np.tile(self.X_train, (1, X_test.shape[0], 1))
        # 将X_test进行归一化处理
        X_test = self.feature_normalization(X_test)
        # 将X_test增加一个维度
        X_test = np.expand_dims(X_test, axis=0)
        # 将X_test复制k份，拼接到x_train上
        X_test = np.tile(X_test, (x_train.shape[0], 1, 1))
        # 计算x_train和X_test之间的欧式距离
        distance = np.sqrt(np.sum((x_train - X_test)**2, axis=2))
        # 将distance按列排序，取出前k个距离
        weight = np.arange(k, 0, -1) / np.sum(np.arange(k, 0, -1))
        distance_index = np.argsort(distance, axis=0)[:k, :]
        distance = distance[:k, :].T * weight
        k_nearest_y = self.y_train[distance_index.T]

        pred = np.zeros((k_nearest_y.shape[0], self.clas))
        for i in range(self.clas):
            pred[:, i] = np.sum(distance * (k_nearest_y == i), axis=1)
        # 返回预测结果
        return np.argmax(pred, axis=1)

    # 计算准确率
    def score(self, X_test, y_test, k=3):
        # 调用predict函数，预测结果
        y_predict = self.predict(X_test, k)
        # 返回预测结果和真实结果的匹配率
        return np.sum(y_predict == y_test) / len(y_test)
```


```python
train_dataset = pd.read_csv('dataset\\Mobile_phone_price_range_estimate_train.csv').to_numpy()
test_dataset = pd.read_csv('dataset\\Mobile_phone_price_range_estimate_test.csv').to_numpy()

X_train = train_dataset[:, :-1]
y_train = train_dataset[:, -1]
X_test = test_dataset[:, :-1]
y_test = test_dataset[:, -1]

knn = KNN()
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test, 1)
print("KNN K=1 acc:%7.5f" % accuracy)
accuracy = knn.score(X_test, y_test, 3)
print("KNN K=3 acc:%7.5f" % accuracy)
accuracy = knn.score(X_test, y_test, 8)
print("KNN K=8 acc:%7.5f" % accuracy)
accuracy = knn.score(X_test, y_test, 12)
print("KNN K=12 acc:%7.5f" % accuracy)
accuracy = knn.score(X_test, y_test, 32)
print("KNN K=32 acc:%7.5f" % accuracy)
accuracy = knn.score(X_test, y_test, 128)
print("KNN K=128 acc:%7.5f" % accuracy)
```

    KNN K=1 acc:0.49000
    KNN K=3 acc:0.48250
    KNN K=8 acc:0.54500
    KNN K=12 acc:0.56000
    KNN K=32 acc:0.58000
    KNN K=128 acc:0.65250
    

## 总结

在本章中，我们确实实现了K近邻（KNN）算法。KNN是一种基本的机器学习算法，它根据输入数据的K个最近邻的类别来进行分类或回归预测。

KNN算法具有简单、易于实现、无需训练模型的特点，因此它常被用作其他机器学习算法的基线模型。


