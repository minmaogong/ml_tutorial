"""
    通过逻辑回归处理多分类任务案例：手写数字识别
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 划分训练集和测试集
from sklearn.preprocessing import MinMaxScaler # 数据归一化
from sklearn.linear_model import LogisticRegression # 逻辑回归模型

# 1. 加载数据集
digit = pd.read_csv("../data/train.csv")
print(digit.shape)
# 用灰度图显示这张手写数字图片。 digit.iloc[10, 1:] -> DataFrame digit.iloc[10, 1:].values -> ndarray 。 cmap: 颜色映射表，gray: 灰度图
plt.imshow(digit.iloc[10, 1:].values.reshape(28, 28), cmap='gray') # digit.iloc[10:, 1:] 10: 第11行 1: 从第2列多最后一列 取第 11 个样本的 784 个像素值 reshape(28, 28) 把一维的 784 个像素值变成 28 x 28 的二维图片矩阵
plt.show()

# 2. 划分训练集和测试集
X = digit.drop("label", axis=1) # 特征数据
y = digit["label"] # 标签数据
# 划分训练集和测试集，测试集占比 30%， random_state=100：随机种子
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# 3. 特征工程-归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) # 训练集归一化
x_test = scaler.transform(x_test) # 测试集归一化

# 4. 模型训练
model = LogisticRegression() # 这是一个0-9的多分类任务，LogisticRegression会自动使用softmax回归函数来处理多分类问题
model.fit(x_train, y_train)

# 5. 模型评估
model.score(x_test, y_test) # 计算模型在测试集上的准确率

# 6. 模型预测
plt.imshow(digit.iloc[123, 1:].values.reshape(28, 28), cmap='gray') # 显示第 124 个样本的手写数字图片
plt.show()
# sklearn 的模型预测时，要求输入是二维数组：(样本数量, 特征数量)
print(model.predict(digit.iloc[123, 1:].values.reshape(1, -1))) # 预测第 123 个样本的手写数字图片的标签。 reshape(1, -1) 将一维的 784 个像素值变成模型可以接受的二维的 1 x 784 的矩阵

