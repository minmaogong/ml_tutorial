"""
    正则化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge # 线性回归，Lasso回归，岭回归模型
from sklearn.metrics import mean_squared_error # 均方误差损失函数

from sklearn.preprocessing import PolynomialFeatures # 多项式特征生成器，PolynomialFeatures()是scikit-learn库中的一个类，用于生成多项式特征，degree参数指定多项式的最高次数，interaction_only参数指定是否仅生成交互特征，include_bias参数指定是否包含偏置项（常数项）

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成数据
n = 300
X = np.linspace(-3, 3, n).reshape(-1, 1) # linspace(-3, 3, n).reshape(-1, 1) 生成-3到3之间的n个数，并将其转换为n行1列的二维数组
y = np.sin(X) + np.random.uniform(-0.5, 0.5, n).reshape(-1, 1) # 生成目标变量y，sin(X)是一个非线性函数，np.random.uniform(-0.5, 0.5, n).reshape(-1, 1)生成一个范围在-0.5到0.5之间的随机噪声，并将其转换为n行1列的二维数组

# 画出散点图
fig, ax = plt.subplots(2, 3, figsize=(15, 8)) # 创建一个包含1行3列子图的图形，fig是整个图形对象，ax是一个包含3个子图对象的数组
ax[0, 0].scatter(X, y, color='y') # 在第一个子图上绘制散点图，X是自变量，y是因变量，color='y'设置点的颜色为黄色
ax[0, 1].scatter(X, y, color='y')
ax[0, 2].scatter(X, y, color='y')

# 2. 划分训练集和测试集
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 将数据集划分为训练集和测试集，test_size=0.2表示测试集占总数据的20%，random_state=42设置随机种子以确保结果可重复

# 3. 特征工程：转换为20次多项式特征
poly = PolynomialFeatures(degree=20) # 创建一个多项式特征生成器对象，degree=20表示生成5次多项式特征，interaction_only=False表示生成所有的多项式特征（包括交互特征），include_bias=False表示不包含偏置项（常数项）
x_train = poly.fit_transform(x_train) # 使用训练数据拟合多项式特征生成器，并将训练数据转换为多项式特征，fit_transform()方法首先拟合多项式特征生成器以学习训练数据的特征，然后将训练数据转换为多项式特征，返回一个新的特征矩阵x_train2
x_test = poly.transform(X_test) # 将测试数据转换为多项式特征，transform()方法使用已经拟合好的多项式特征生成器将测试数据转换为多项式特征，返回一个新的特征矩阵x_test2

# 4. 定义不同模型，分别进行训练和测试
# 4.1 不加正则化
model = LinearRegression() # 创建一个线性回归模型对象，LinearRegression()是scikit-learn库中的一个类，用于执行线性回归分析

# 训练
model.fit(x_train, y_train) # 使用训练数据拟合模型，fit()方法接受训练数据X_train和y_train作为输入，调整模型参数以最小化预测值与实际值之间的误差
print(model.coef_)

# 测试（预测）
y_pred = model.predict(x_test) # 使用测试数据进行预测，predict()方法接受测试数据X_test作为输入，返回模型对测试数据的预测结果y_pred

# 计算误差
test_loss = mean_squared_error(y_test, y_pred) # 计算均方误差，mean_squared_error()函数接受测试数据的实际值y_test和预测值y_pred作为输入，返回均方误差mse，均方误差是评估回归模型性能的常用指标，值越小表示模型的预测能力越好
train_loss = mean_squared_error(y_train, model.predict(x_train)) # 计算训练集上的均方误差，model.predict(X_train)使用训练数据进行预测，返回训练数据的预测结果，然后与实际值y_train进行比较计算均方误差


# 画出拟合曲线
ax[0,0].plot(X, model.predict(poly.transform(X)), color='r') # 在第一个子图上绘制拟合曲线，X是自变量，model.predict(X)是模型对X的预测结果，color='r'设置线条颜色为红色
ax[0,0].text(-3, 1, f"测试误差: {test_loss:.4f}")
ax[0,0].text(-3, 1.3, f"训练误差: {train_loss:.4f}")

# 用模型参数绘制直方图
ax[1,0].bar(np.arange(21), model.coef_.reshape(-1)) # 在第二行第一个子图上绘制模型参数的直方图，np.arange(21)生成一个从0到20的整数数组，model.coef_.reshape(-1)将模型的系数展平为一维数组，bar()函数接受x轴位置和对应的高度作为输入，绘制出每个特征对应的系数值

# 4.2 L1正则化-Lasso回归
model = Lasso(alpha=0.01) # 创建一个Lasso回归模型对象，alpha参数是正则化强度的控制参数，值越大表示正则化效果越强，默认为1.0
# 训练
model.fit(x_train, y_train) # 使用训练数据拟合模型，fit()方法接受训练数据X_train和y_train作为输入，调整模型参数以最小化预测值与实际值之间的误差
# print(model.coef_)

# 测试（预测）
y_pred = model.predict(x_test) # 使用测试数据进行预测，predict()方法接受测试数据X_test作为输入，返回模型对测试数据的预测结果y_pred

# 计算误差
test_loss = mean_squared_error(y_test, y_pred) # 计算均方误差，mean_squared_error()函数接受测试数据的实际值y_test和预测值y_pred作为输入，返回均方误差mse，均方误差是评估回归模型性能的常用指标，值越小表示模型的预测能力越好
train_loss = mean_squared_error(y_train, model.predict(x_train)) # 计算训练集上的均方误差，model.predict(X_train)使用训练数据进行预测，返回训练数据的预测结果，然后与实际值y_train进行比较计算均方误差


# 画出拟合曲线
ax[0,1].plot(X, model.predict(poly.transform(X)), color='r') # 在第一个子图上绘制拟合曲线，X是自变量，model.predict(X)是模型对X的预测结果，color='r'设置线条颜色为红色
ax[0,1].text(-3, 1, f"测试误差: {test_loss:.4f}")
ax[0,1].text(-3, 1.3, f"训练误差: {train_loss:.4f}")

# 用模型参数绘制直方图
ax[1,1].bar(np.arange(21), model.coef_.reshape(-1)) # 在第二行第一个子图上绘制模型参数的直方图，np.arange(21)生成一个从0到20的整数数组，model.coef_.reshape(-1)将模型的系数展平为一维数组，bar()函数接受x轴位置和对应的高度作为输入，绘制出每个特征对应的系数值

# 4.3 L2正则化-岭回归
model = Ridge(alpha=0.5) # 创建一个岭回归模型对象，alpha是正则化强度参数，值越大表示正则化效果越强，默认值为1.0

# 训练
model.fit(x_train, y_train) # 使用训练数据拟合模型，fit()方法接受训练数据X_train和y_train作为输入，调整模型参数以最小化预测值与实际值之间的误差
# print(model.coef_)

# 测试（预测）
y_pred = model.predict(x_test) # 使用测试数据进行预测，predict()方法接受测试数据X_test作为输入，返回模型对测试数据的预测结果y_pred

# 计算误差
test_loss = mean_squared_error(y_test, y_pred) # 计算均方误差，mean_squared_error()函数接受测试数据的实际值y_test和预测值y_pred作为输入，返回均方误差mse，均方误差是评估回归模型性能的常用指标，值越小表示模型的预测能力越好
train_loss = mean_squared_error(y_train, model.predict(x_train)) # 计算训练集上的均方误差，model.predict(X_train)使用训练数据进行预测，返回训练数据的预测结果，然后与实际值y_train进行比较计算均方误差


# 画出拟合曲线
ax[0,2].plot(X, model.predict(poly.transform(X)), color='r') # 在第一个子图上绘制拟合曲线，X是自变量，model.predict(X)是模型对X的预测结果，color='r'设置线条颜色为红色
ax[0,2].text(-3, 1, f"测试误差: {test_loss:.4f}")
ax[0,2].text(-3, 1.3, f"训练误差: {train_loss:.4f}")

# 用模型参数绘制直方图
ax[1,2].bar(np.arange(21), model.coef_.reshape(-1)) # 在第二行第一个子图上绘制模型参数的直方图，np.arange(21)生成一个从0到20的整数数组，model.coef_.reshape(-1)将模型的系数展平为一维数组，bar()函数接受x轴位置和对应的高度作为输入，绘制出每个特征对应的系数值

plt.show()