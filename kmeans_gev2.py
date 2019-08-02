import random
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.stats import genextreme
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time



# 正规化数据集 X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


def BMM_data(data, n_blocks):
    dist_block = np.split(data, n_blocks)
    dist_block_max = np.zeros(n_blocks)
    for idx, block in enumerate(dist_block):
        dist_block_max[idx] = np.max(block)

    return dist_block_max



class Kmeans():
    """Kmeans聚类算法.

    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数.
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon,
        则说明算法已经收敛
    """

    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i
        # distances = np.zeros(self.k)
        # for i in range(self.k):
        #     distances[i] = euclidean_distance(sample, centroids[i])
        # closest_i = np.argmin(distances)
        # return closest_i

    def min_gev_prob(self, sample, centroids, gev_param):
        distances = euclidean_distance(sample, centroids)
        prob = np.zeros(self.k)
        for i in range(self.k):
            prob[i] = genextreme.cdf(distances[i], gev_param[i][0], gev_param[i][1], gev_param[i][2])
        closest_i = np.argmin(prob)
        return closest_i

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            # centroid_i = self.min_gev_prob(sample, centroids, gev_param)
            clusters[centroid_i].append(sample_i)
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def fit_gev(self, centroids, X, n_blocks):
        genextreme_param = np.zeros((self.k, 3))
        n_samples = np.shape(X)[0]
        for cent in range(self.k):
            dist_c = np.zeros(n_samples)
            dist_c = euclidean_distance(centroids[cent, :], X)
            dist_block_max = BMM_data(dist_c, n_blocks)

            # fit genextreme
            c, loc, scale = genextreme.fit(dist_block_max)
            genextreme_param[cent][0] = c
            genextreme_param[cent][1] = loc
            genextreme_param[cent][2] = scale

        return genextreme_param

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X, n_blocks):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)

        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for _ in range(self.max_iterations):

            # gev_param = self.fit_gev(centroids, X, n_blocks)
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids

            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X)

            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break

        return self.get_cluster_labels(clusters, X)


def main():
    plt.figure(figsize=(12, 12))

    n_samples = 10000
    random_state = 2
    n_features = 2
    n_centers = 10
    n_blocks = 500

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=random_state)

    # Incorrect number of clusters
    start_t = time()
    y_pred = KMeans(n_clusters=n_centers, init='random', random_state=random_state).fit_predict(X)
    end_t = time()


    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("K-Means t:{:.2f}s".format(end_t - start_t))
    print("K-Means t:{:.2f}s".format(end_t - start_t))

    # Load the dataset
    # X, y = datasets.make_blobs(n_samples=10000,
    #                            n_features=3,
    #                            centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
    #                            cluster_std=[0.2, 0.1, 0.2, 0.2],
    #                            random_state=9)

    # 用Kmeans算法进行聚类
    start_t = time()
    clf = Kmeans(k=n_centers)
    y_pred = clf.predict(X, n_blocks)
    end_t = time()

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("K-Means+GEV t:{:.2f}s".format(end_t - start_t))
    print("K-Means+GEV t:{:.2f}s".format(end_t - start_t))
    plt.show()

    # 可视化聚类效果
    # fig = plt.figure(figsize=(12, 8))
    # ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    # plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2])
    # plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2])
    # plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], X[y == 2][:, 2])
    # plt.scatter(X[y == 3][:, 0], X[y == 3][:, 1], X[y == 3][:, 2])
    # plt.show()


if __name__ == "__main__":
    main()