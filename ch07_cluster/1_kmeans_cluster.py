"""
    聚类：一种无监督学习算法
        K-means聚类：K均值聚类算法
        层次聚类：
            聚合聚类
            分裂聚类
        密度聚类
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans # K均值聚类
from sklearn.datasets import make_blobs # 生成聚类数据集
from sklearn.metrics import silhouette_score # 轮廓系数
from sklearn.metrics import calinski_harabasz_score # Calinski-Harabasz指数

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 使用make_blobs 生成 3 个簇，每个簇 100 个点 cluster_std: 表示簇的标准差，簇越大，点越分散 n_features: 表示每个点的特征数。X: 生成的样本特征，y_true: 生成的样本标签
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=2, n_features=2)

fig, ax = plt.subplots(2, figsize=(8, 8))
# scatter: 绘制散点图，X[:, 0]表示第一列数据，X[:, 1]表示第二列数据，s表示点的大小，c表示点的颜色，label表示图例
ax[0].scatter(X[:, 0], X[:, 1], s=50, c="gray", label="原始数据")
ax[0].set_title("原始数据")
# legend: 显示图例
ax[0].legend()

# 使用 K-means 聚类算法进行聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X) # 预测每个点的簇标签
centers = kmeans.cluster_centers_ # 获取簇中心点坐标
print("簇中心点坐标：", centers)

ax[1].scatter(X[:, 0], X[:, 1], s=50, c=y_kmeans)
ax[1].scatter(centers[:, 0], centers[:, 1], s=200, c="red", marker="o", label="簇中心")
ax[1].set_title("K-means 聚类结果(K=3)")
ax[1].legend()

plt.show()

# 模型评估
wcss = kmeans.inertia_ # 计算簇内误差平方和（Within-Cluster Sum of Squares，WCSS），用于评估聚类效果
print("评价指标（簇内误差平方和）WCSS：", wcss) # 簇内误差平方和越小，说明簇内数据点越紧密，聚类效果越好

si = silhouette_score(X, y_kmeans) # 计算轮廓系数（Silhouette Coefficient），用于评估聚类效果，取值范围为[-1, 1]，值越大表示聚类效果越好
print("评价指标（轮廓系数）Silhouette Coefficient：", si)

ch = calinski_harabasz_score(X, y_kmeans) # 计算Calinski-Harabasz指数（Calinski-Harabasz Index），用于评估聚类效果，值越大表示聚类效果越好
print("评价指标（Calinski-Harabasz指数）Calinski-Harabasz Index：", ch)
