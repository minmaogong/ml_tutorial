from sklearn.metrics import classification_report
from sklearn.datasets import make_classification # 自动生成分类数据集
from sklearn.model_selection import train_test_split # 生成一个二分类数据集
from sklearn.linear_model import LogisticRegression # 逻辑回归模型（分类模型）

# 1. 生成数据集
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42) # 生成一个二分类数据集，包含1000个样本，每个样本有10个特征，随机种子为42
print(X.shape, y.shape) # (1000, 10) (1000,)

# 2. 划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 定义模型
model = LogisticRegression()

# 4. 训练模型
model.fit(x_train, y_train)

# 5. 预测（测试）
y_pred = model.predict(x_test)

# 6. 生成评估报告
report = classification_report(y_true=y_test, y_pred=y_pred)
print(report)


# ROC-AUC
from sklearn.metrics import roc_auc_score
y_proba = model.predict_proba(x_test)[:, 1] # 预测概率，返回一个二维数组，每行包含两个元素，分别表示样本属于每个类别的概率；我们取第二列（[:, 1]）来获取样本属于正类（类别1）的概率。
print(y_proba)
auc = roc_auc_score(y_true=y_test, y_score=y_proba)

