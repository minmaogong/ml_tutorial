import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV # 网络搜索+交叉验证


# 1. 准备数据
dataset = pd.read_csv('../data/heart_disease.csv')
# 数据清洗
dataset.dropna(inplace=True)
dataset.info()
print(dataset.head(5))

# 2. 数据集划分
# 划分特征和标签
X = dataset.drop('是否患有心脏病', axis=1)
y = dataset['是否患有心脏病']

print(X.shape)
print(y.shape)

# 3. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 4. 特征工程
# 数值型特征，需要标准化或者归一化转换
numerical_features = ["年龄", "静息血压", "胆固醇", "最大心率", "运动后的ST下降", "主血管数量"]
# 类别型特征，需要进行独热编码
categorical_features = ["胸痛类型", "静息心电图结果", "峰值ST段的斜率", "地中海贫血"]
# 二元特征
binary_features = ["性别", "空腹血糖", "运动性心绞痛"]
# 创建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        # 对数值型特征进行标准化转换
        ("num", StandardScaler(), numerical_features),
        # 对类别型特征进行独热编码转换，使用drop="first"避免多重共线性
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        # 二元特征不用进行处理
        ("binary", "passthrough", binary_features)
    ]
)
# 执行特征转换
x_train = preprocessor.fit_transform(x_train) # 计算训练集的统计信息并进行转换
x_test = preprocessor.transform(x_test) # 使用训练集计算的信息对测试集进行转换，只能做转换，不能做fit_transform
print(x_train.shape, x_test.shape)

# 5. 定义模型（包括网格搜索+交叉验证）
knn = KNeighborsClassifier()

# 定义参数网格
grid = {"n_neighbors": list(range(1, 11)), "p": [1, 2]} # p=1: 欧式距离，p=2: 曼哈顿距离
models = GridSearchCV(estimator=knn, param_grid=grid, cv=10) # cv: K折交叉验证，默认是5折，这里设置为10折

# 6. 训练所有模型
models.fit(x_train, y_train)

# 7. 查看结果
print(pd.DataFrame(models.cv_results_).to_string()) # 所有交叉验证结果
print(models.best_estimator_) # 最佳模型
print(models.best_params_) # 最佳参数
print(models.best_score_) # 最佳得分

# 8. 使用最佳模型进行评估
knn = models.best_estimator_
print(knn.score(x_test, y_test)) # 测试集
