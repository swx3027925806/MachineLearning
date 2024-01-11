# K-Means 聚类算法

> 该文章作为机器学习的第四篇文章，主要介绍的是K-Means聚类算法，这是我们介绍的第一个无监督算法，在这里我们将对什么是无监督，为什么要有无监督等也会有一些介绍，算法不难，大家且看且思考。
>
> 难度系数：⭐
>
> 更多相关工作请参考：[Github](https://github.com/swx3027925806/MachineLearning)

## 算法介绍

K-Means算法是一种**无监督**的聚类分析算法，通过迭代过程将数据划分为K个聚类。该算法以**距离作为数据对象间相似度的衡量标准**，将数据对象分配给距离其最近的聚类中心，并重新计算每个聚类的聚类中心，直到满足终止条件。K-Means算法广泛应用于图像处理、文本挖掘、社交网络分析等领域，具有处理大量数据集、连续型数据和多种形式数据的能力。然而，该算法也存在一些缺点，如**对初始值敏感**、可能**陷入局部最优解**等。因此，在实际应用中，需要根据具体情况选择合适的簇群数量和初始中心点，以提高算法的准确性和效率。

### 无监督算法介绍
无监督学习与监督学习不同，无监督学习使用的输入数据是没有标注过的，这意味着数据只给出了输入变量（自变量X）而没有给出相应的输出变量（因变量）。在无监督学习中，算法本身将发掘数据中有趣的结构。具体来说，无监督学习的主要任务包括聚类和降维等。聚类是一种无监督学习方法，其目标是将数据样本分成不同的组，使得同一组内的样本彼此相似，而不同组之间的样本差异较大。常见的聚类算法包括K-Means、层次聚类、DBSCAN等。

### 算法原理解析

K-Means算法是一种经典的聚类分析算法，**通过迭代过程将数据划分为K个聚类**。该算法以距离作为数据对象间相似度的衡量标准，将数据对象分配给距离其最近的聚类中心，并重新计算每个聚类的聚类中心，直到满足终止条件。

K-Means算法的原理可以概括为以下几个步骤：

1. 随机选择K个对象作为初始聚类中心。
2. 计算每个对象与各个聚类中心的距离，然后将每个对象分配给距离其最近的聚类中心。
3. 重新计算每个聚类的聚类中心，这是通过将该聚类中所有对象的坐标取平均值得出的。
4. 重复步骤2和3，直到满足某个终止条件，如没有（或最小数目）对象被重新分配给不同的聚类，没有（或最小数目）聚类中心再发生变化，误差平方和局部最小等。

在K-Means算法中，距离的度量方式可以采用欧氏距离、曼哈顿距离等，而聚类的相似度则是通过计算各聚类中对象的均值所获得的一个“中心对象”（引力中心）来进行计算的。算法的目标是将数据划分为K个聚类，使得同一聚类中的对象相似度较高，而不同聚类中的对象相似度较小。

#### 计算距离

1. **欧几里得距离：** 这是最常见的距离计算方式，适用于连续特征。如果两个样本分别为 $(x_1)$ 和 $(x_2)$，每个特征维度分别为 $(d_1)$ 和 $(d_2)$，则它们之间的欧几里得距离为：
$$
D(x_1, x_2) = \sqrt{\sum_{i=1}^{d}(x_{1i} - x_{2i})^2}
$$
2. **曼哈顿距离：** 也称为城市街区距离，适用于离散特征或有序特征。其计算方式为：
$$
D(x_1, x_2) = \sum_{i=1}^{d}|x_{1i} - x_{2i}|
$$

## 数据集介绍

本次实验依旧采用鸢尾花数据集作为实验数据，如果对这部分有不确定的同学可以访问[机器学习原理到Python代码实现之NaiveBayes【朴素贝叶斯】](https://blog.csdn.net/qq_44961028/article/details/135450691)这篇文章，看一下其中数据分析部分。

鸢尾花(Iris)数据集是一个常用的分类实验数据集，由Fisher在1936年收集整理。该数据集包含150个样本，每个样本有四个属性：花萼长度、花萼宽度、花瓣长度和花瓣宽度，这四个属性用于预测鸢尾花属于Setosa、Versicolour或Virginica三个种类中的哪一类。

鸢尾花数据集的特点是具有多重变量，即每个样本都有多个特征，这使得它成为进行多元分类任务的一个理想选择。通过分析这些特征，可以了解不同鸢尾花品种之间的差异，并建立分类器来自动识别未知样本的种类。

鸢尾花数据集的来源是实际的鸢尾花测量数据，因此它具有实际背景和应用价值。这个数据集经常被用于机器学习和数据挖掘算法的实验和验证，因为它提供了多变量分析的一种有效方式。

在本次我们将采用K-Means算法对鸢尾花数据集进行聚类分析，并尝试找到最佳的聚类数K。因为是我监督算法，故我们在训练阶段将会隐去标签。

## 代码实现

### 造轮子

需要注意的是，我们需要将聚类完成的结果和具有标签的结果去做匹配，这是一个需要注意的部分。

在本次的算法实现中，我们将约定聚类中心和标签数量等同。在预测阶段，通过聚类结果和预测结果匹配，得到最终的正确率。具体实现为`recursion`和`match`。


```python
import numpy as np
import pandas as pd
```


```python
class MyKMeans:
    def __init__(self, k=5) -> None:
        self.k = k
        self.centroids = None

    def fit(self, data):
        # 随机选择k个样本作为初始聚类中心
        self.centroids = data[np.random.choice(range(len(data)), self.k, replace=False)]
        
        while True:
            # 计算每个样本到聚类中心的距离
            distances = np.sqrt(((data - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            
            # 分配每个样本到最近的聚类中心
            labels = np.argmin(distances, axis=0)

            # 更新聚类中心
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])

            # 如果聚类中心不再变化，则停止迭代
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def label_mapping(self, labels):
        # 将标签映射回原始类别
        mapping = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}

    def predict(self, data):
        # 计算每个样本到聚类中心的距离
        distances = np.sqrt(((data - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        
        # 分配每个样本到最近的聚类中心
        pred = np.argmin(distances, axis=0)

        return pred

    def scores(self, pred, label):
        pred_index = [np.where(pred == i) for i in range(self.k)]
        label_index = [np.where(label == i) for i in range(self.k)]
        return self.recursion(pred_index, label_index) / len(label)
        
    def match(self, pred_index, label_index):
        match_lens = 0
        for i in range(self.k):
            match_lens += len(list(set(pred_index[i][0]) & set(label_index[i][0])))
        return match_lens
    
    def recursion(self, pred_index, label_index, deep=0):
        # 全排列 pred_index 中的元素
        max_lens = self.match(pred_index, label_index)
        for i in range(deep+1, len(pred_index)):
            pred_index[deep], pred_index[i] = pred_index[i], pred_index[deep]
            max_lens = max(self.recursion(pred_index, label_index, deep + 1), max_lens)
            pred_index[deep], pred_index[i] = pred_index[i], pred_index[deep]
        return max_lens

```

### 数据加载及验证


```python
train_dataset = pd.read_csv('dataset\\iris_training.csv')
test_dataset = pd.read_csv('dataset\\iris_test.csv')

X_train = train_dataset.drop('virginica', axis=1).to_numpy()
y_train = train_dataset['virginica'].to_numpy()

X_test = test_dataset.drop('virginica', axis=1).to_numpy()
y_test = test_dataset['virginica'].to_numpy()
```


```python
kmeans1 = MyKMeans(k=3)
kmeans1.fit(X_train)
pred = kmeans1.predict(X_test)
accuracy = kmeans1.scores(pred, y_test)
accuracy
```




    0.9



### SKLearn方法调用

需要注意的是SKLrean本身并没有可以计算预测结果和真实结果准确率的方法，故我们这里直接调用自己造的轮子即可。


```python
from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters=3)
kmeans2.fit(X_train)
pred = kmeans2.predict(X_test)
accuracy = kmeans1.scores(pred, y_test)
accuracy
```

    d:\annaconda\envs\MachineLearning\Lib\site-packages\sklearn\cluster\_kmeans.py:1416: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=10)
    




    0.9



## 实验对比

同样的，我们现在继续对手机价格数据集来分析一下实验，需要注意的是，由于K-means的性能和之前随机初始化的点存在一定的关联，所以会出现性能不稳定的现象：


```python
train_dataset = pd.read_csv('dataset\\Mobile_phone_price_range_estimate_train.csv').to_numpy()
test_dataset = pd.read_csv('dataset\\Mobile_phone_price_range_estimate_test.csv').to_numpy()

X_train = train_dataset[:, :-1]
y_train = train_dataset[:, -1]
X_test = test_dataset[:, :-1]
y_test = test_dataset[:, -1]

kmeans = MyKMeans(4)
kmeans.fit(X_train)
pred = kmeans.predict(X_test)
accuracy = kmeans.scores(pred, y_test)
accuracy
```




    0.6525



## 总结

K-Means算法是一种无监督学习的聚类算法，其基本原理是通过迭代过程将数据划分为K个聚类，使得同一聚类中的对象相似度较高，而不同聚类中的对象相似度较小。该算法首先随机选择K个对象作为初始聚类中心，然后根据对象与各个聚类中心的距离，将每个对象分配给距离其最近的聚类中心。接着，算法重新计算每个聚类的聚类中心，这是通过将该聚类中所有对象的坐标取平均值得出的。算法重复执行这一过程，直到满足终止条件。

K-Means算法具有简单、高效、可扩展性强等优点，因此在许多领域得到了广泛应用。然而，该算法也存在一些缺点，如对初始值敏感、可能陷入局部最优解等。为了解决这些问题，一些改进的K-Means算法被提出，如K-Means++、K-Means等。在后期如果有时间会做详细介绍。
