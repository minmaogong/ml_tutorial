"""
    拟合、欠拟合、过拟合
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # 线性回归模型
from sklearn.metrics import mean_squared_error # 均方误差损失函数

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成数据
n = 300
X = np.linspace(-3, 3, n).reshape(-1, 1) # linspace(-3, 3, n).reshape(-1, 1) 生成-3到3之间的n个数，并将其转换为n行1列的二维数组
y = np.sin(X) + np.random.uniform(-0.5, 0.5, n).reshape(-1, 1) # 生成目标变量y，sin(X)是一个非线性函数，np.random.uniform(-0.5, 0.5, n).reshape(-1, 1)生成一个范围在-0.5到0.5之间的随机噪声，并将其转换为n行1列的二维数组

# 画出散点图
fig, ax = plt.subplots(1, 3, figsize=(15, 4)) # 创建一个包含1行3列子图的图形，fig是整个图形对象，ax是一个包含3个子图对象的数组
ax[0].scatter(X, y, color='y') # 在第一个子图上绘制散点图，X是自变量，y是因变量，color='y'设置点的颜色为黄色
ax[1].scatter(X, y, color='y')
ax[2].scatter(X, y, color='y')

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 将数据集划分为训练集和测试集，test_size=0.2表示测试集占总数据的20%，random_state=42设置随机种子以确保结果可重复

# 3. 定义模型
model = LinearRegression() # 创建一个线性回归模型对象，LinearRegression()是scikit-learn库中的一个类，用于执行线性回归分析

# 4. 模型训练和测试
# 训练
model.fit(X_train, y_train) # 使用训练数据拟合模型，fit()方法接受训练数据X_train和y_train作为输入，调整模型参数以最小化预测值与实际值之间的误差

# 打印训练好的模型参数
print("斜率：", model.coef_) # 打印模型的系数，model.coef_属性包含了线性回归模型的系数，即每个特征对目标变量的影响程度
print("截距：", model.intercept_) # 打印模型的截距，model.intercept_属性包含了线性回归模型的截距，即当所有特征值为0时，目标变量的预测值

# 测试（预测）
y_pred = model.predict(X_test) # 使用测试数据进行预测，predict()方法接受测试数据X_test作为输入，返回模型对测试数据的预测结果y_pred

# 计算误差
test_loss = mean_squared_error(y_test, y_pred) # 计算均方误差，mean_squared_error()函数接受测试数据的实际值y_test和预测值y_pred作为输入，返回均方误差mse，均方误差是评估回归模型性能的常用指标，值越小表示模型的预测能力越好
train_loss = mean_squared_error(y_train, model.predict(X_train)) # 计算训练集上的均方误差，model.predict(X_train)使用训练数据进行预测，返回训练数据的预测结果，然后与实际值y_train进行比较计算均方误差


# 画出拟合曲线
ax[0].plot(X, model.predict(X), color='r') # 在第一个子图上绘制拟合曲线，X是自变量，model.predict(X)是模型对X的预测结果，color='r'设置线条颜色为红色
ax[0].text(-3, 1, f"测试误差: {test_loss:.4f}")
ax[0].text(-3, 1.3, f"训练误差: {train_loss:.4f}")
plt.show()
