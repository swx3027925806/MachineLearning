# MachineLearning 食用指南

该项目是对机器学习中常见的算法进行学习以及复现，可以作为日常的学习笔记。

## 使用说明

首先，你需要先安装好相关依赖，作者使用的是Python3.11作为编译器。其他相关的配置在`requirements.txt`中。
请通过一下指令安装：

```
pip install -r requirements.txt
```

项目中，每个文件夹下都是一个机器学习算法的实现，通常作者会采用从头实现和掉包实现两种模式。
`*.ipynb`是`jupyter`文件，可以直接clone下来，在具备环境的情况下可以直接运行。markdown是博客原文，相关博客在CSDN上有所展示。

### 项目目录

这份表格阐明了每个算法的难易程度和相关行，作为给大家的学习路线指导：

| 算法名称                  | 难度[1-5]   | 前置算法             | 备注            |
|--------------------------|-------------| --------------------|-----------------|
| LinearRegression         | 2           | 无                  | 线性回归算法     |
| NaiveBayes               | 3           | 无                  | 朴素贝叶斯算法   |
| KNN                      | 1           | 无                  | K近邻算法        |
| KMeans                   | 1           | 无                  | K均值聚类算法    |
| PolynomialRegression     | 3           | LinearRegression    | 多项式回归算法   |

## Jupyter2Markdown
```
jupyter nbconvert --to markdown notebook.ipynb
```

## 作者相关
欢迎大家交流学习，作者看邮箱不是很及时，见谅【doge】

- 姓名：佘文轩
- 邮箱：3027925806@qq.com
- CSDN：[https://blog.csdn.net/qq_44961028](https://blog.csdn.net/qq_44961028)