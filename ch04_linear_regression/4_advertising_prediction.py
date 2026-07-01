# 案例：广告投放效果预测

import pandas as pd
from sklearn.preprocessing import StandardScaler # 标准化
from sklearn.model_selection import train_test_split # 划分训练集和测试集
from sklearn.linear_model import LinearRegression, SGDRegressor # 线性回归模型-正规方程 线性回归模型-随机梯度下降
from sklearn.metrics import mean_squared_error # 使用均方误差MSE，做模型评价指标

# 加载数据集
advertising = pd.read_csv('../data/advertising.csv')

# 数据清洗
advertising.drop(advertising.columns[0], axis=1, inplace=True) # 删除第一列
advertising.dropna(inplace=True) # 删除缺失值
advertising.info()
print(advertising.head(5))

# 划分训练集和测试集
X = advertising.drop('Sales', axis=1) # 特征 = 排除Sales标签列后的所有列都是特征
y = advertising['Sales'] # 标签 = Sales列
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) # 划分训练集和测试集，测试集占30%，随机种子为0

# 特征工程 - 标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 计算训练集的均值和标准差，并进行标准化转换
x_test = scaler.transform(x_test) # 使用训练集的均值和标准差对测试集进行标准化转换，只能做转换，不能做fit_transform

# 使用正规方程法拟合线性回归模型
normal_equation = LinearRegression()
normal_equation.fit(x_train, y_train)
print("正规方程法解得模型系数：", normal_equation.coef_)
print("正规方程法解得模型偏置：", normal_equation.intercept_)

# 使用随机梯度下降拟合线性回归模型
gradient_descent = SGDRegressor()
gradient_descent.fit(x_train, y_train)
print("随机梯度下降法解得模型系数：", gradient_descent.coef_)
print("随机梯度下降法解得模型偏置：", gradient_descent.intercept_)

# 使用均方误差MSE评估模型，MSE 越小，模型越好
print("正规方程法均方误差MSE：", mean_squared_error(y_test, normal_equation.predict(x_test))) # 真实结果与预测结果的均方误差
print("随机梯度下降法均方误差MAE：", mean_squared_error(y_test, gradient_descent.predict(x_test)))

# 使用决定系数R²评估模型，R² 越接近1，模型越好
print("正规方程法绝对系数R²：", normal_equation.score(x_test, y_test)) # 真实结果与预测结果的R²
print("随机梯度下降法绝对系数R²：", gradient_descent.score(x_test, y_test)) # 真实结果与预测结果的R²
